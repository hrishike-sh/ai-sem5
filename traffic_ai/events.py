from dataclasses import dataclass, field
from uuid import uuid4

from .contracts import Rule, Verdict


@dataclass
class Event:
    event_id: str
    scene_id: int
    track_id: int | None
    rule: Rule
    started_at: float
    last_seen_at: float
    observations: int
    box: tuple
    evidence: dict = field(default_factory=dict)


class EventCollector:
    def __init__(self, persistence=.4, gap=1.0):
        self.persistence = persistence
        self.gap = gap
        self.pending = {}
        self.active = {}
        self.reported_turn_entries = {}
        self.scene_id = None

    def update(self, decisions, timestamp, scene_id=0, image=False):
        if self.scene_id != scene_id:
            self.pending.clear()
            self.active.clear()
            self.reported_turn_entries.clear()
            self.scene_id = scene_id
        new_events = []
        displayed = []
        for decision in decisions:
            if decision.verdict != Verdict.VIOLATION or decision.box is None:
                continue
            identity = decision.track_id if decision.track_id is not None else tuple(decision.box)
            if not image and decision.track_id is None:
                continue
            entry = decision.evidence.get("observed_entry") if decision.rule == Rule.TURN else None
            entry_time = entry.get("entry_timestamp") if entry else None
            key = (identity, decision.rule, entry_time)
            existing = self.active.get(key)
            if existing and timestamp - existing.last_seen_at > self.gap:
                del self.active[key]
                existing = None
            if existing:
                existing.last_seen_at = timestamp
                existing.observations += 1
                existing.box = decision.box
                displayed.append(decision)
                continue
            if entry_time is not None and key in self.reported_turn_entries:
                displayed.append(decision)
                continue
            first, last, count = self.pending.get(key, (timestamp, timestamp, 0))
            if timestamp - last > self.gap:
                first, count = timestamp, 0
            count += 1
            self.pending[key] = (first, timestamp, count)
            transition = decision.rule in {Rule.RED_LIGHT, Rule.RESTRICTED, Rule.TURN}
            if image or transition or (count >= 2 and timestamp - first >= self.persistence):
                event = Event(uuid4().hex, scene_id, decision.track_id, decision.rule, first, timestamp, count, decision.box, decision.evidence)
                self.active[key] = event
                if entry_time is not None:
                    self.reported_turn_entries[key] = entry_time
                self.pending.pop(key, None)
                new_events.append(event)
                displayed.append(decision)
        self.pending = {k: v for k, v in self.pending.items() if timestamp - v[1] <= self.gap}
        self.active = {k: v for k, v in self.active.items() if timestamp - v.last_seen_at <= self.gap}
        self.reported_turn_entries = {k: v for k, v in self.reported_turn_entries.items() if timestamp - v <= 8}
        return new_events, displayed
