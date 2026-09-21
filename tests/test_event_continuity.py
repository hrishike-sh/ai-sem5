"""Synthetic confirmation contracts; verdict inputs are not traffic labels."""
from copy import deepcopy

import pytest

from traffic_ai.contracts import Decision, Rule, Verdict
from traffic_ai.event_continuity import ContradictionAwareEventCollector as Collector
from tests import test_ai_events as legacy


def assessment(timestamp, verdict=Verdict.VIOLATION, rule=Rule.WRONG_WAY, track=1):
    return Decision(rule, verdict, "Synthetic event-aggregation input", timestamp, track,
                    (0, 0, 20, 20), {"synthetic": True})


def test_explicit_clear_breaks_pending_confirmation_before_a_new_finding():
    collector = Collector()
    assert not collector.update([assessment(0.)], 0.)[0]
    assert not collector.update([assessment(.2, Verdict.CLEAR)], .2)[0]
    created, displayed = collector.update([assessment(.5)], .5)
    assert not created and not displayed, "Pre-clear support must not confirm the new pending episode"


def test_same_frame_violation_and_clear_cannot_emit_or_display_a_finding():
    collector = Collector()
    collector.update([assessment(0.)], 0.)
    created, displayed = collector.update([assessment(.5), assessment(.5, Verdict.CLEAR)], .5)
    assert not created and not displayed, "Contradictory frame assessments must withhold confirmation"


def test_actual_rule_engine_clear_transition_breaks_unconfirmed_wrong_way_episode():
    from traffic_ai.contracts import Detection
    from traffic_ai.rules import RuleEngine, SceneEvidence

    polygon = [[0, 0], [200, 0], [200, 200], [0, 200]]
    scene = SceneEvidence(stationary_camera=True, confidence=.95, approach_polygon=polygon,
                          direction_polygon=polygon, permitted_direction=(1., 0.),
                          source_evidence={"synthetic": True})
    rules, collector, verdicts = RuleEngine(), Collector(), []
    for timestamp, x in ((0., 100), (.6, 70), (.8, 130), (1.1, 60)):
        vehicle = Detection("car", .95, (x - 10, 80, x + 10, 100), 1)
        decisions = rules.evaluate(vehicle, timestamp, scene, video=True)
        verdicts.append(next(d.verdict for d in decisions if d.rule == Rule.WRONG_WAY))
        created, displayed = collector.update(decisions, timestamp)
    assert verdicts == [Verdict.UNKNOWN, Verdict.VIOLATION, Verdict.CLEAR, Verdict.VIOLATION]
    assert not created and not displayed, "A real RuleEngine CLEAR must break pending event support"


@pytest.mark.parametrize("case", [
    legacy.test_flicker_persistence_and_episode_deduplication,
    legacy.test_transition_not_lost_and_cut_resets_identity,
    legacy.test_untracked_video_cannot_accumulate_events,
    legacy.test_same_turn_entry_is_not_reported_again_after_detection_gap,
    legacy.test_distinct_observed_entry_can_report_within_event_gap,
    legacy.test_turn_entry_memory_expires_and_scene_reset_clears_it,
])
def test_existing_event_contracts_with_new_collector(monkeypatch, case):
    monkeypatch.setattr(legacy, "EventCollector", Collector)
    case()


def test_clear_trace_new_support_and_raw_evidence_are_preserved():
    collector = Collector()
    positive, clear = assessment(0.), assessment(.2, Verdict.CLEAR)
    source_before = deepcopy([positive.to_dict(), clear.to_dict()])
    collector.update([positive], 0.)
    collector.update([clear], .2)
    trace = collector.confirmation_snapshot()
    assert trace["cleared_pending"][0]["first_observed_at"] == 0.
    assert trace["cleared_pending"][0]["observations"] == 1
    assert trace["cleared_pending"][0]["source_decision_indices"] == [0]
    collector.update([assessment(.5)], .5)
    pending = collector.confirmation_snapshot()["pending"][0]
    assert pending["first_observed_at"] == .5 and pending["observations"] == 1
    assert pending["reason"] == "persistence_not_yet_satisfied"
    created, displayed = collector.update([assessment(.95)], .95)
    assert len(created) == len(displayed) == 1 and created[0].started_at == .5
    assert created[0].observations == 2 and created[0].evidence["synthetic"] is True
    assert source_before == [positive.to_dict(), clear.to_dict()]
    snapshot = collector.confirmation_snapshot()
    snapshot["displayed_decision_indices"].append(999)
    assert collector.confirmation_snapshot()["displayed_decision_indices"] == [0]


def test_unknown_gap_tolerance_and_confirmed_episode_dedup_are_unchanged():
    collector = Collector()
    collector.update([assessment(0.)], 0.)
    assert collector.update([assessment(.2, Verdict.UNKNOWN)], .2) == ([], [])
    assert collector.confirmation_snapshot()["pending"][0]["reason"] == "current_assessment_unknown_gap_tolerance"
    created, _ = collector.update([assessment(.5)], .5)
    identity = created[0].event_id
    assert collector.update([assessment(.6, Verdict.CLEAR)], .6) == ([], [])
    created, displayed = collector.update([assessment(.7)], .7)
    assert not created and len(displayed) == 1
    assert next(iter(collector.active.values())).event_id == identity


def test_same_frame_conflicts_are_order_independent_for_images_and_transitions():
    for rule, image, track in ((Rule.HELMET, True, None), (Rule.RED_LIGHT, False, 1)):
        for reverse in (False, True):
            collector = Collector()
            values = [assessment(0., rule=rule, track=track), assessment(0., Verdict.CLEAR, rule, track)]
            if reverse:
                values.reverse()
            before = deepcopy([d.to_dict() for d in values])
            assert collector.update(values, 0., image=image) == ([], [])
            conflict = collector.confirmation_snapshot()["contradictions"][0]
            assert conflict["reason"] == "conflicting_frame_assessments"
            assert len(conflict["clear_decision_indices"]) == len(conflict["violation_decision_indices"]) == 1
            assert before == [d.to_dict() for d in values]


def test_clear_resets_only_its_identity_rule_and_scene_cut_is_independent():
    collector = Collector()
    collector.update([assessment(0.), assessment(0., rule=Rule.HELMET, track=2)], 0.)
    collector.update([assessment(.2, Verdict.CLEAR), assessment(.2, rule=Rule.HELMET, track=2)], .2)
    created, _ = collector.update([assessment(.5), assessment(.5, rule=Rule.HELMET, track=2)], .5)
    assert [(event.track_id, event.rule) for event in created] == [(2, Rule.HELMET)]
    collector.update([assessment(.6, Verdict.CLEAR)], .6, scene_id=1)
    snapshot = collector.confirmation_snapshot()
    assert snapshot["scene_reset"] and not snapshot["cleared_pending"]
    assert not collector.pending and not collector.active
