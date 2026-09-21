from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Rule(str, Enum):
    HELMET = "helmet_absence"
    RIDERS = "triple_riding"
    RED_LIGHT = "red_light_crossing"
    STOP_LINE = "stop_line_encroachment"
    CROSSING = "crossing_obstruction_during_red"
    WRONG_WAY = "wrong_way"
    FOOTPATH = "footpath_riding"
    RESTRICTED = "restricted_entry"
    TURN = "prohibited_turn"


class Verdict(str, Enum):
    VIOLATION = "violation"
    CLEAR = "no_violation_observed"
    UNKNOWN = "cannot_assess"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[float, float, float, float]
    track_id: int | None = None
    model: str = ""

    def __post_init__(self):
        x1, y1, x2, y2 = self.box
        if not (0 <= self.confidence <= 1 and x2 > x1 and y2 > y1):
            raise ValueError("Invalid detection confidence or box")

    @property
    def ground_point(self):
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


@dataclass
class Decision:
    rule: Rule
    verdict: Verdict
    reason: str
    timestamp: float
    track_id: int | None = None
    box: tuple[float, float, float, float] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


@dataclass
class FrameAnalysis:
    timestamp: float
    detections: list[Detection]
    decisions: list[Decision]
    scene_id: int = 0

    def to_dict(self):
        return asdict(self)
