"""Conservative automatic control-to-approach association, without new models.

Geometry proposes compatibility, not legal applicability or model accuracy. All
compatible observations participate in conflict detection before quality filters:
an unreadable competing signal/sign must not disappear from the ownership graph.
"""
from copy import deepcopy
from dataclasses import replace
from math import isfinite

import numpy as np

from .approaches import ApproachResolver, aligned_approaches
from .geometry import point_in_polygon


POLICY = "mutual_unique_control_v1"


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and isfinite(value)


def _box(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(_number(v) for v in value) and value[2] > value[0] and value[3] > value[1])


def _scene_errors(approaches):
    errors, identities = [], set()
    for index, approach in enumerate(approaches):
        try:
            identity = approach["id"]
            polygon = np.asarray(approach["polygon"], dtype=float)
            line = np.asarray(approach["stop_line"], dtype=float)
            direction = np.asarray(approach["direction"], dtype=float)
            valid_id = type(identity) in (int, str) and (not isinstance(identity, str) or bool(identity.strip()))
            valid = (valid_id and identity not in identities and polygon.ndim == 2
                     and polygon.shape[0] >= 3 and polygon.shape[1] == 2
                     and line.shape == (2, 2) and direction.shape == (2,)
                     and all(np.isfinite(v).all() for v in (polygon, line, direction))
                     and np.linalg.norm(line[1] - line[0]) > 0
                     and np.isclose(np.linalg.norm(direction), 1.)
                     and _number(approach["width"]) and approach["width"] > 0)
            if valid:
                identities.add(identity)
            else:
                errors.append(index)
        except (KeyError, TypeError, ValueError):
            errors.append(index)
    return errors


class ConflictAwareApproachResolver(ApproachResolver):
    """Drop-in resolver with auditable mutual uniqueness and temporal support.

Only association metadata is added to input observations; measured boxes,
states, scores and timestamps are retained. Construct a new instance whenever
the scene geometry/revision changes, as Analyzer already does. This policy keeps
the existing geometric thresholds and temporal lamp checks; it adds no learned
semantics, calibration, legal validation or confidence estimate.
"""

    def __init__(self, approaches):
        self.scene_errors = _scene_errors(approaches)
        super().__init__(deepcopy(approaches) if not self.scene_errors else [])
        self.graphs = {}
        self.last_signal_timestamp = None

    def _graph(self, kind, observations):
        graph = {"policy": POLICY, "kind": kind, "scene_errors": self.scene_errors[:],
                 "observations": [], "approaches": []}
        invalid = any(not isinstance(o, dict) or not _box(o.get("box")) for o in observations)
        reverse = {a["id"]: [] for a in self.approaches}
        for index, observation in enumerate(observations):
            matches = (aligned_approaches(observation["box"], self.approaches)
                       if isinstance(observation, dict) and _box(observation.get("box")) else [])
            record = {"observation_index": index, "source_pointer": f"/{kind}_observations/{index}",
                      "candidate_approaches": matches, "status": "withheld", "reason": "no_compatible_approach"}
            graph["observations"].append(record)
            for identity in matches:
                reverse[identity].append(index)
        graph["approaches"] = [{"approach_id": key, "observation_indices": value} for key, value in reverse.items()]
        for record in graph["observations"]:
            matches = record["candidate_approaches"]
            record["competing_observation_indices"] = sorted({i for key in matches for i in reverse[key]
                                                               if i != record["observation_index"]})
            if self.scene_errors:
                record["reason"] = "invalid_scene_geometry"
            elif invalid:
                # An unlocatable observation cannot safely be excluded as a competitor.
                record["reason"] = "invalid_observation_geometry"
            elif len(matches) > 1:
                record["reason"] = "ambiguous_approach"
            elif matches and record["competing_observation_indices"]:
                record["reason"] = "competing_observations"
            elif matches:
                record.update(status="candidate", reason="mutual_unique_geometry")
        self.graphs[kind] = graph
        return graph

    @staticmethod
    def _publish(observations, graph):
        for observation, record in zip(observations, graph["observations"]):
            if not isinstance(observation, dict):
                continue
            observation["associated_approach"] = (record["candidate_approaches"][0]
                                                   if record["status"] == "associated" else None)
            observation["candidate_approaches"] = record["candidate_approaches"][:]
            observation["association"] = deepcopy(record)
            observation["association"]["policy"] = POLICY
            observation.pop("association_basis", None)
            if record["status"] == "associated":
                observation["association_basis"] = (
                    "Mutual unique compatible observation and corridor; experimental geometry"
                    + (" with current temporal lamp support" if graph["kind"] == "signal" else ""))

    def assign_signs(self, observations):
        self.resolved_signs = {}
        graph = self._graph("sign", observations)
        for observation, record in zip(observations, graph["observations"]):
            if record["status"] != "candidate":
                continue
            turns = observation.get("prohibited_turns", [])
            supported = (observation.get("restriction") in ("no_entry", "no_vehicles")
                         or isinstance(turns, (list, tuple, set)) and bool(turns)
                         and all(isinstance(turn, str) and turn in {"left", "right", "straight", "u_turn"}
                                 for turn in turns))
            if observation.get("status") != "recognised_candidate" or not supported:
                record.update(status="withheld", reason="unresolved_sign_meaning")
            else:
                record.update(status="associated", reason="mutual_unique_recognised_candidate")
        self._publish(observations, graph)
        for observation, record in zip(observations, graph["observations"]):
            if record["status"] == "associated":
                self.resolved_signs[record["candidate_approaches"][0]] = [deepcopy(observation)]

    def update_signals(self, observations, timestamp, stable):
        graph = self._graph("signal", observations)
        valid_time = _number(timestamp) and timestamp >= 0
        time_order = valid_time and (self.last_signal_timestamp is None or timestamp > self.last_signal_timestamp)
        if valid_time:
            self.last_signal_timestamp = timestamp
        reason = ("invalid_timestamp" if not valid_time else "nonincreasing_timestamp" if not time_order
                  else "camera_not_stable" if not stable else None)
        if reason:
            self.pending.clear()
            self.resolved_signals.clear()
        selected, indices = [], []
        for index, (observation, record) in enumerate(zip(observations, graph["observations"])):
            if reason:
                record.update(status="withheld", reason=reason)
            if record["status"] != "candidate":
                continue
            observed_at = observation.get("timestamp", timestamp)
            if not _number(observed_at) or abs(observed_at - timestamp) > 1e-6:
                record.update(status="withheld", reason="observation_timestamp_mismatch")
            elif observation.get("state") not in ("red", "amber", "green"):
                record.update(status="withheld", reason="unresolved_signal_state")
            elif not _number(observation.get("model_score")) or not .7 <= observation["model_score"] <= 1:
                record.update(status="withheld", reason="insufficient_signal_score")
            else:
                selected.append(deepcopy(observation))
                indices.append(index)
        if reason is None:
            # Passing only mutual-unique candidates also removes the pending history
            # of any approach that became ambiguous or lost its visible lamp.
            super().update_signals(selected, timestamp, stable)
            for index, selected_observation in zip(indices, selected):
                record = graph["observations"][index]
                associated = selected_observation["associated_approach"] is not None
                record.update(status="associated" if associated else "pending",
                              reason="mutual_unique_temporal_support" if associated else "temporal_support_required")
        graph["timestamp"] = timestamp if valid_time else None
        self._publish(observations, graph)
        for index in indices:
            record = graph["observations"][index]
            if record["status"] == "associated":
                identity = record["candidate_approaches"][0]
                resolved = deepcopy(observations[index])
                resolved["timestamp"] = timestamp
                self.resolved_signals[identity] = resolved

    def association_snapshot(self):
        return {"policy": POLICY, "scope": "Experimental geometric compatibility; legal applicability unverified",
                "scene_errors": self.scene_errors[:], "graphs": deepcopy(self.graphs),
                'geometry_policy': 'Explicit configuration required; review-only and legacy appearance excluded.',
                'withheld_geometry': deepcopy(self.withheld_geometry)}

    def scene_for(self, vehicle, scene, timestamp):
        # This resolver owns automatic control fields. Never inherit a stale
        # association from a prior vehicle, ambiguous corridor or cached scene.
        evidence = deepcopy(scene.source_evidence)
        for key in ("approach", "applicable_sign", "signal_association", "crossing_association", "direction_marking"):
            evidence.pop(key, None)
        clean = replace(scene, stop_line=None, crossing_polygon=[], signal_state="unknown",
                        signal_associated=False, signal_confidence=0., signal_observed_at=None,
                        permitted_direction=None, direction_polygon=[],
                        restricted_polygon=[], restricted_classes=set(), prohibited_turns=set(),
                        source_evidence=evidence)
        matches = [a for a in self.approaches if point_in_polygon(vehicle.ground_point, a["polygon"])]
        reason = ("invalid_scene_geometry" if self.scene_errors else "camera_not_stable" if not scene.stationary_camera
                  else "insufficient_scene_confidence" if not _number(scene.confidence) or scene.confidence < .8
                  else "invalid_timestamp" if not _number(timestamp) or timestamp < 0
                  else "ambiguous_vehicle_approach" if len(matches) > 1
                  else "no_vehicle_approach" if not matches else "unique_vehicle_approach")
        evidence["control_association"] = {**self.association_snapshot(), "vehicle_reason": reason,
                                           "candidate_approaches": [a["id"] for a in matches]}
        if reason != "unique_vehicle_approach":
            return clean
        return super().scene_for(vehicle, clean, timestamp)
