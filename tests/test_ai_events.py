from traffic_ai.contracts import Decision, Rule, Verdict
from traffic_ai.events import EventCollector


def violation(timestamp, rule=Rule.HELMET, track=1):
    return Decision(rule, Verdict.VIOLATION, "test", timestamp, track, (0, 0, 10, 20))


def test_flicker_persistence_and_episode_deduplication():
    events = EventCollector()
    assert not events.update([violation(0)], 0)[0]
    assert not events.update([violation(.2)], .2)[0]
    created, displayed = events.update([violation(.5)], .5)
    assert len(created) == len(displayed) == 1
    assert not events.update([violation(.8)], .8)[0]
    assert not events.update([violation(2)], 2)[0]
    assert len(events.update([violation(2.5)], 2.5)[0]) == 1


def test_transition_not_lost_and_cut_resets_identity():
    events = EventCollector()
    assert len(events.update([violation(0, Rule.RED_LIGHT)], 0)[0]) == 1
    assert not events.update([violation(.2, Rule.RED_LIGHT)], .2)[0]
    assert len(events.update([violation(.3, Rule.RED_LIGHT)], .3, scene_id=1)[0]) == 1


def test_untracked_video_cannot_accumulate_events():
    events = EventCollector()
    assert not events.update([violation(0, track=None)], 0)[0]
    assert len(events.update([violation(0, track=None)], 0, image=True)[0]) == 1


def turn_violation(timestamp, entry_timestamp):
    decision = violation(timestamp, Rule.TURN)
    decision.evidence = {'observed_entry': {'entry_timestamp': entry_timestamp, 'approach_id': 2}}
    return decision


def test_same_turn_entry_is_not_reported_again_after_detection_gap():
    events = EventCollector()
    assert len(events.update([turn_violation(1, .5)], 1)[0]) == 1
    events.update([], 2.1)
    created, displayed = events.update([turn_violation(2.2, .5)], 2.2)
    assert not created
    assert len(displayed) == 1


def test_distinct_observed_entry_can_report_within_event_gap():
    events = EventCollector()
    assert len(events.update([turn_violation(1, .5)], 1)[0]) == 1
    assert len(events.update([turn_violation(1.5, 1.2)], 1.5)[0]) == 1


def test_turn_entry_memory_expires_and_scene_reset_clears_it():
    events = EventCollector()
    events.update([turn_violation(1, .5)], 1)
    events.update([], 9)
    assert not events.reported_turn_entries
    events.update([turn_violation(9.1, 9)], 9.1)
    assert len(events.update([turn_violation(9.2, 9)], 9.2, scene_id=1)[0]) == 1
