from dataclasses import replace

import cv2
import numpy as np

from .geometry import area, intersection, point_in_polygon
from ivp.marking_policy import configured_geometry_policy, geometry_is_configured


def infer_approaches(arrows, stop_lines, road_mask):
    approaches = []
    for arrow in arrows:
        if not geometry_is_configured(arrow):
            continue
        direction = np.asarray(arrow["direction"], dtype=float)
        direction /= max(1e-9, np.linalg.norm(direction))
        centre = np.asarray(arrow["arrow_polygon"], dtype=float).mean(axis=0)
        arrow_length = np.ptp(np.asarray(arrow["arrow_polygon"]) @ direction)
        matches = []
        for marking in stop_lines:
            if not geometry_is_configured(marking):
                continue
            first, second = np.asarray(marking["line"], dtype=float)
            segment = second - first
            length = np.linalg.norm(segment)
            if length < 20 or abs(np.dot(segment / length, direction)) > .3:
                continue
            system = np.column_stack([direction, -segment])
            if abs(np.linalg.det(system)) < 1e-6:
                continue
            distance, fraction = np.linalg.solve(system, first - centre)
            if not .1 <= fraction <= .9 or not 0 < distance < 8 * max(20, arrow_length):
                continue
            matches.append((float(distance), first, second, float(length), marking))
        if not matches:
            continue
        matches.sort(key=lambda value: value[0])
        distance, first, second, length, marking = matches[0]
        normal = np.array([-(second - first)[1], (second - first)[0]])
        if np.dot(normal, direction) < 0:
            first, second = second, first
        rear = direction * (distance + max(arrow_length, length))
        forward = direction * length * .6
        polygon = np.asarray([first - rear, second - rear, second + forward, first + forward])
        mask = np.zeros(road_mask.shape, np.uint8)
        cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
        support = np.count_nonzero((road_mask > 0) & (mask > 0)) / max(1, mask.sum())
        if support < .6:
            continue
        midpoint = (first + second) / 2
        if any(np.linalg.norm(midpoint - np.asarray(a["stop_line"]).mean(axis=0)) < length * .15 for a in approaches):
            continue
        approaches.append({**configured_geometry_policy(), "id": len(approaches), "polygon": polygon.tolist(), "stop_line": [first.tolist(), second.tolist()],
                           "direction": direction.tolist(), "width": length, "road_support": support,
                           "arrow": arrow, "marking": marking,
                           "basis": "Road-supported corridor from an independently detected arrow and perpendicular stop-line candidate"})
    return approaches


def aligned_approaches(box, approaches):
    centre = np.asarray([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
    matches = []
    for approach in approaches:
        midpoint = np.asarray(approach["stop_line"]).mean(axis=0)
        direction = np.asarray(approach["direction"])
        difference = centre - midpoint
        along = np.dot(difference, direction)
        lateral = abs(direction[0] * difference[1] - direction[1] * difference[0])
        if 0 < along < 4 * approach["width"] and lateral < approach["width"] * .45:
            matches.append(approach["id"])
    return matches


class ApproachResolver:
    def __init__(self, approaches):
        self.approaches = [a for a in approaches if geometry_is_configured(a)
                           and ('marking' not in a or geometry_is_configured(a['marking']))
                           and ('arrow' not in a or geometry_is_configured(a['arrow']))]
        self.withheld_geometry = [{'id': a.get('id'), 'reason': 'explicit_geometry_configuration_required'}
                                  for a in approaches if not any(a is kept for kept in self.approaches)]
        self.pending = {}
        self.resolved_signals = {}
        self.resolved_signs = {}
        self.resolved_crossings = {}

    def assign_crossings(self, crossings):
        self.resolved_crossings = {}
        candidates = {}
        for crossing in crossings:
            polygon = np.asarray(crossing.get("polygon", []), dtype=np.float32)
            crossing["associated_approach"] = None
            crossing["candidate_approaches"] = []
            if not geometry_is_configured(crossing):
                crossing['association_basis'] = 'Withheld: review-only appearance cannot configure a crossing zone.'
                continue
            if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] != 2 or not np.isfinite(polygon).all():
                continue
            hull = cv2.convexHull(polygon)
            size = cv2.contourArea(hull)
            if size <= 1:
                continue
            centre = polygon.mean(axis=0)
            for approach in self.approaches:
                corridor = cv2.convexHull(np.asarray(approach["polygon"], np.float32))
                overlap, _ = cv2.intersectConvexConvex(hull, corridor)
                midpoint = np.asarray(approach["stop_line"]).mean(axis=0)
                along = np.dot(centre - midpoint, approach["direction"])
                if overlap / size >= .5 and 0 < along < .6 * approach["width"]:
                    crossing["candidate_approaches"].append(approach["id"])
            if len(crossing["candidate_approaches"]) == 1:
                candidates.setdefault(crossing["candidate_approaches"][0], []).append(crossing)
        for identity, choices in candidates.items():
            if len(choices) == 1:
                choices[0]["associated_approach"] = identity
                choices[0]["association_basis"] = "Unique crossing candidate overlapping the corridor beyond its oriented stop line; experimental geometry"
                self.resolved_crossings[identity] = choices[0]

    def assign_signs(self, observations):
        self.resolved_signs = {}
        for observation in observations:
            observation["associated_approach"] = None
            matches = aligned_approaches(observation["box"], self.approaches)
            observation["candidate_approaches"] = matches
            if observation.get("status") == "recognised_candidate" and len(matches) == 1:
                self.resolved_signs.setdefault(matches[0], []).append(observation)
        for identity, signs in list(self.resolved_signs.items()):
            if len(signs) != 1:
                del self.resolved_signs[identity]
                continue
            signs[0]["associated_approach"] = identity
            signs[0]["association_basis"] = "Unique sign candidate aligned with an arrow/stop-line corridor; experimental, exception plates not interpreted"

    def update_signals(self, observations, timestamp, stable):
        self.resolved_signals = {}
        if not stable:
            self.pending.clear()
            return
        grouped = {}
        for observation in observations:
            observation["timestamp"] = timestamp
            observation["associated_approach"] = None
            matches = aligned_approaches(observation["box"], self.approaches)
            observation["candidate_approaches"] = matches
            if len(matches) == 1 and observation["state"] in {"red", "amber", "green"} and observation["model_score"] >= .7:
                grouped.setdefault(matches[0], []).append(observation)
        for approach in self.approaches:
            identity = approach["id"]
            choices = grouped.get(identity, [])
            if len(choices) != 1:
                self.pending.pop(identity, None)
                continue
            observation = choices[0]
            previous = self.pending.get(identity)
            matching = (previous is not None and previous[3] == observation["state"] and 0 < timestamp - previous[1] <= .5
                        and intersection(previous[4], observation["box"]) / max(1, min(area(previous[4]), area(observation["box"]))) > .5)
            first, count = (previous[0], previous[2] + 1) if matching else (timestamp, 1)
            self.pending[identity] = (first, timestamp, count, observation["state"], observation["box"])
            if count >= 3 and timestamp - first >= .3:
                observation["associated_approach"] = identity
                observation["association_basis"] = "Unique aligned visible signal with consistent temporal lamp state; experimental geometric association"
                self.resolved_signals[identity] = observation

    def scene_for(self, vehicle, scene, timestamp):
        matches = [a for a in self.approaches if point_in_polygon(vehicle.ground_point, a["polygon"])]
        if len(matches) != 1:
            return scene
        approach = matches[0]
        crossing = self.resolved_crossings.get(approach["id"])
        observation = self.resolved_signals.get(approach["id"])
        if observation and not 0 <= timestamp - observation["timestamp"] <= .5:
            observation = None
        changes = {"approach_polygon": approach["polygon"], "stop_line": tuple(approach["stop_line"]),
                   "crossing_polygon": crossing["polygon"] if crossing else [],
                   "permitted_direction": tuple(approach["direction"]), "direction_polygon": approach["polygon"],
                   "source_evidence": {**scene.source_evidence, "approach": approach, "signal_association": observation, "crossing_association": crossing}}
        if observation:
            changes.update(signal_state=observation["state"], signal_associated=True, signal_confidence=.8, signal_observed_at=observation["timestamp"])
        signs = self.resolved_signs.get(approach["id"], [])
        if signs:
            sign = signs[0]
            changes["source_evidence"]["applicable_sign"] = sign
            changes["prohibited_turns"] = set(sign.get("prohibited_turns", []))
            if sign.get("restriction") in {"no_entry", "no_vehicles"}:
                first, second = np.asarray(approach["stop_line"])
                forward = np.asarray(approach["direction"]) * approach["width"] * .6
                changes["restricted_polygon"] = np.asarray([first, second, second + forward, first + forward]).tolist()
                changes["restricted_classes"] = {"car", "motorcycle", "bus", "truck", "bicycle"}
        return replace(scene, **changes)
