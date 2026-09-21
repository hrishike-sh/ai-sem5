from dataclasses import dataclass, field
from math import hypot

from .contracts import Decision, Detection, Rule, Verdict
from .geometry import crossed_segment, point_in_polygon, signed_distance


@dataclass
class SceneEvidence:
    scene_id: int = 0
    stationary_camera: bool = False
    confidence: float = 0.0
    approach_polygon: list = field(default_factory=list)
    stop_line: tuple | None = None
    crossing_polygon: list = field(default_factory=list)
    footpath_polygons: list = field(default_factory=list)
    restricted_polygon: list = field(default_factory=list)
    restricted_classes: set = field(default_factory=set)
    permitted_direction: tuple | None = None
    direction_polygon: list = field(default_factory=list)
    prohibited_turns: set = field(default_factory=set)
    signal_state: str = "unknown"
    signal_associated: bool = False
    signal_confidence: float = 0.0
    signal_observed_at: float | None = None
    source_evidence: dict = field(default_factory=dict)

    def red(self, timestamp):
        return (self.signal_associated and self.signal_state == "red" and self.signal_confidence >= .8
                and self.signal_observed_at is not None and 0 <= timestamp - self.signal_observed_at <= .5)


@dataclass
class RiderEvidence:
    associated_people: list[Detection] = field(default_factory=list)
    bare_heads: list[Detection] = field(default_factory=list)
    helmet_heads: list[Detection] = field(default_factory=list)
    association_confident: bool = False
    visibility_complete: bool = False
    association_basis: str = "Single-frame geometric association"


class RuleEngine:
    def __init__(self):
        self.history = {}
        self.stop_entries = {}
        self.track_context = {}
        self.scene_id = None

    def reset(self, scene_id):
        self.history.clear()
        self.stop_entries.clear()
        self.track_context.clear()
        self.scene_id = scene_id

    def evaluate(self, vehicle, timestamp, scene, riders=None, video=False, observed_turn=None, turn_context=None):
        if scene.scene_id != self.scene_id:
            self.reset(scene.scene_id)
        if not timestamp >= 0:
            raise ValueError("Timestamp must be nonnegative")
        decisions = []

        def decide(rule, verdict, reason, **evidence):
            decisions.append(Decision(rule, verdict, reason, timestamp, vehicle.track_id, vehicle.box, evidence))

        bike = vehicle.label == "motorcycle"
        for rule in (Rule.HELMET, Rule.RIDERS):
            if not bike:
                decide(rule, Verdict.UNKNOWN, "This rider rule applies to motorcycles")
            elif riders is None or not riders.association_confident:
                decide(rule, Verdict.UNKNOWN, "Riders cannot be associated with this motorcycle confidently")
            elif rule == Rule.HELMET and riders.bare_heads:
                decide(rule, Verdict.VIOLATION, "Visible bare head associated with a motorcycle rider", heads=[d.box for d in riders.bare_heads], association=riders.association_basis)
            elif rule == Rule.RIDERS and len(riders.associated_people) >= 3:
                decide(rule, Verdict.VIOLATION, "At least three distinct people associated with one motorcycle", rider_count=len(riders.associated_people), association=riders.association_basis)
            elif not riders.visibility_complete:
                decide(rule, Verdict.UNKNOWN, "Occlusion or missing head/rider detections prevent a complete assessment")
            else:
                decide(rule, Verdict.CLEAR, "Visible riders do not meet the violation condition")

        point = vehicle.ground_point
        track = self.history.setdefault(vehicle.track_id, []) if video and vehicle.track_id is not None else []
        if track and timestamp <= track[-1][0]:
            raise ValueError("Tracked timestamps must increase; reset tracks at a new clip or scene")
        context = (repr(scene.stop_line), repr(scene.approach_polygon), scene.stationary_camera)
        # Match the continuity limit used by trajectory and turn-entry tracking.
        if track and (timestamp - track[-1][0] > .75 or self.track_context.get(vehicle.track_id) != context
                      or hypot(point[0] - track[-1][1][0], point[1] - track[-1][1][1])
                      > 5 * max(8., vehicle.box[2] - vehicle.box[0])):
            track.clear()
            self.stop_entries.pop(vehicle.track_id, None)
        if video and vehicle.track_id is not None:
            self.track_context[vehicle.track_id] = context
        previous = track[-1] if track else None
        scene_ready = scene.stationary_camera and scene.confidence >= .8
        in_approach = point_in_polygon(point, scene.approach_polygon)
        red = scene.red(timestamp)
        if not red or not scene_ready or not in_approach:
            self.stop_entries.pop(vehicle.track_id, None)
        elif scene.stop_line is not None and track:
            anchor = next((p for p in reversed(track) if signed_distance(p[1], scene.stop_line) < -3), None)
            if anchor and all(p[2] for p in track if p[0] >= anchor[0]) and crossed_segment(anchor[1], point, scene.stop_line, margin=3):
                self.stop_entries.setdefault(vehicle.track_id, {"timestamp": timestamp, "from": anchor[1], "to": point})
            if signed_distance(point, scene.stop_line) <= 3:
                self.stop_entries.pop(vehicle.track_id, None)
        signal_known = (scene.signal_associated and scene.signal_confidence >= .8
                        and scene.signal_state in {"red", "amber", "green"}
                        and scene.signal_observed_at is not None and 0 <= timestamp - scene.signal_observed_at <= .5)
        for rule in (Rule.RED_LIGHT, Rule.STOP_LINE, Rule.CROSSING):
            if not scene_ready or not scene.approach_polygon or not signal_known:
                decide(rule, Verdict.UNKNOWN, "A stable approach and its current controlling signal are required")
            elif rule == Rule.CROSSING and not scene.crossing_polygon:
                decide(rule, Verdict.UNKNOWN, "Crossing boundary is not established")
            elif rule != Rule.CROSSING and scene.stop_line is None:
                decide(rule, Verdict.UNKNOWN, "Approach stop line is not established")
            elif rule == Rule.RED_LIGHT and (not video or previous is None):
                decide(rule, Verdict.UNKNOWN, "Crossing requires an observed tracked transition in video")
            elif rule == Rule.STOP_LINE and (not video or previous is None):
                decide(rule, Verdict.UNKNOWN, "Encroachment requires an observed entry and sustained stopped position in video")
            elif rule == Rule.STOP_LINE and red and in_approach and signed_distance(point, scene.stop_line) > 3 and vehicle.track_id not in self.stop_entries:
                decide(rule, Verdict.UNKNOWN, "Vehicle is beyond the line, but entry during red was not observed")
            elif rule == Rule.CROSSING and (not video or len(track) < 2 or timestamp - track[0][0] < 1):
                decide(rule, Verdict.UNKNOWN, "Obstruction requires sustained occupancy in video")
            else:
                violation = False
                if red and in_approach:
                    if rule == Rule.RED_LIGHT:
                        anchor = next((p for p in reversed(track) if signed_distance(p[1], scene.stop_line) < -3), None)
                        violation = bool(anchor and all(p[2] for p in track if p[0] >= anchor[0])
                                         and crossed_segment(anchor[1], point, scene.stop_line, margin=3))
                    elif rule == Rule.STOP_LINE:
                        recent = [p for p in track if timestamp - p[0] <= 1.25]
                        tolerance = max(3, .1 * (vehicle.box[2] - vehicle.box[0]))
                        violation = (vehicle.track_id in self.stop_entries and bool(recent)
                                     and timestamp - recent[0][0] >= 1
                                     and all(p[2] and signed_distance(p[1], scene.stop_line) > 3
                                             and hypot(point[0] - p[1][0], point[1] - p[1][1]) <= tolerance for p in recent))
                    else:
                        recent = [p for p in track if timestamp - p[0] <= 1.5]
                        violation = (point_in_polygon(point, scene.crossing_polygon) and bool(recent)
                                     and all(p[2] and point_in_polygon(p[1], scene.crossing_polygon) for p in recent)
                                     and hypot(point[0] - recent[0][1][0], point[1] - recent[0][1][1]) < 8)
                decide(rule, Verdict.VIOLATION if violation else Verdict.CLEAR, "Observed approach geometry and associated signal evaluated", signal=scene.signal_state,
                       signal_association=scene.source_evidence.get("signal_association"), stop_line=scene.stop_line,
                       approach_id=scene.source_evidence.get("approach", {}).get("id"),
                       crossing_association=scene.source_evidence.get("crossing_association") if rule == Rule.CROSSING else None,
                       observed_red_entry=self.stop_entries.get(vehicle.track_id) if rule == Rule.STOP_LINE else None)

        for rule in (Rule.WRONG_WAY, Rule.FOOTPATH, Rule.RESTRICTED, Rule.TURN):
            if not scene_ready:
                decide(rule, Verdict.UNKNOWN, "Reliable scene geometry and a stationary camera are required")
                continue
            if rule == Rule.WRONG_WAY:
                if not video or previous is None or scene.permitted_direction is None or not scene.approach_polygon:
                    decide(rule, Verdict.UNKNOWN, "Video motion and independently established legal direction are required")
                    continue
                recent = [p for p in track if timestamp - p[0] <= 2]
                direction_zone = scene.direction_polygon or scene.approach_polygon
                if not point_in_polygon(point, direction_zone) or not all(point_in_polygon(p[1], direction_zone) for p in recent):
                    decide(rule, Verdict.UNKNOWN, "Direction evidence does not cover the complete observed motion segment")
                    continue
                dx, dy = point[0] - recent[0][1][0], point[1] - recent[0][1][1]
                motion = hypot(dx, dy)
                direction = hypot(*scene.permitted_direction)
                if motion < 15 or direction == 0 or timestamp - recent[0][0] < .5:
                    decide(rule, Verdict.UNKNOWN, "Insufficient movement for a direction decision")
                    continue
                cosine = (dx * scene.permitted_direction[0] + dy * scene.permitted_direction[1]) / (motion * direction)
                violation = in_approach and cosine < -.7
            elif rule == Rule.FOOTPATH:
                if vehicle.label not in {"motorcycle", "bicycle"} or not scene.footpath_polygons:
                    decide(rule, Verdict.UNKNOWN, "A two-wheeler and a resolved footpath boundary are required")
                    continue
                recent = [p for p in track if timestamp - p[0] <= 1.5]
                minimum_motion = max(5, .15 * (vehicle.box[2] - vehicle.box[0]))
                if (not video or not recent or timestamp - recent[0][0] < .5
                        or hypot(point[0] - recent[0][1][0], point[1] - recent[0][1][1]) < minimum_motion):
                    decide(rule, Verdict.UNKNOWN, "Motion is required to distinguish riding from parking")
                    continue
                violation = any(point_in_polygon(point, p) and all(point_in_polygon(sample[1], p) for sample in recent) for p in scene.footpath_polygons)
            elif rule == Rule.RESTRICTED:
                if not scene.restricted_polygon or not scene.restricted_classes:
                    decide(rule, Verdict.UNKNOWN, "A readable applicable restriction and its entry area are required")
                    continue
                if not video or previous is None:
                    decide(rule, Verdict.UNKNOWN, "Entry requires a tracked boundary transition")
                    continue
                violation = (vehicle.label in scene.restricted_classes and point_in_polygon(point, scene.restricted_polygon)
                             and not point_in_polygon(previous[1], scene.restricted_polygon))
            else:
                if (not video or not turn_context or observed_turn is None
                        or turn_context.get("track_id") != vehicle.track_id
                        or turn_context.get("scene_id") != scene.scene_id
                        or not 0 <= timestamp - turn_context.get("entry_timestamp", float("inf")) <= 8):
                    decide(rule, Verdict.UNKNOWN, "An applicable prohibition and resolved video trajectory are required")
                    continue
                violation = observed_turn in turn_context["prohibited_turns"]
            evidence = {}
            if rule == Rule.WRONG_WAY:
                evidence = {"direction_cosine": cosine, "permitted_direction": scene.permitted_direction, "direction_region": direction_zone,
                            "marking": scene.source_evidence.get("direction_marking"), "observed_motion": [dx, dy]}
            elif rule == Rule.FOOTPATH:
                evidence = {"ground_point": point, "footpath_polygons": scene.footpath_polygons}
            elif rule == Rule.RESTRICTED:
                evidence = {"restricted_classes": sorted(scene.restricted_classes), "entry_region": scene.restricted_polygon,
                            "sign": scene.source_evidence.get("applicable_sign")}
            elif rule == Rule.TURN:
                evidence = {"observed_turn": observed_turn, "prohibited_turns": turn_context["prohibited_turns"],
                            "sign": turn_context["sign"], "observed_entry": turn_context}
            decide(rule, Verdict.VIOLATION if violation else Verdict.CLEAR, "Observed movement evaluated against independently established scene evidence", **evidence)
        if video and vehicle.track_id is not None:
            track.append((timestamp, point, red))
            self.history[vehicle.track_id] = [p for p in track if timestamp - p[0] <= 3]
            if len(self.history) > 2048:
                self.history = {key: value for key, value in self.history.items() if value and timestamp - value[-1][0] <= 3}
                self.stop_entries = {key: value for key, value in self.stop_entries.items() if key in self.history}
                self.track_context = {key: value for key, value in self.track_context.items() if key in self.history}
        return decisions
