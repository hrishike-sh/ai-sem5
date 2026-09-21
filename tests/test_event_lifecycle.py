"""Event lifecycle software contracts with explicitly synthetic observations."""
from copy import deepcopy

import pytest

from traffic_ai.contracts import Decision, Detection, Rule, Verdict
from traffic_ai.event_lifecycle import LifecycleAwareEventCollector as Collector
from traffic_ai.rules import RuleEngine, SceneEvidence


def synthetic_scene(timestamp):
    return SceneEvidence(stationary_camera=True, confidence=.95,
                         approach_polygon=[(0, 0), (400, 0), (400, 200), (0, 200)],
                         stop_line=((0, 50), (400, 50)), signal_state="red",
                         signal_associated=True, signal_confidence=.95,
                         signal_observed_at=timestamp, source_evidence={"synthetic": True})


@pytest.mark.parametrize("points", [
    [(0., 50, 40), (.1, 50, 60), (.2, 250, 40), (.3, 250, 60)],
    [(0., 50, 40), (.1, 50, 60), (.9, 50, 40), (1., 50, 60)],
], ids=["observed-position-jump", "observed-tracking-gap"])
def test_rule_engine_fresh_crossing_cannot_extend_pre_discontinuity_event(points):
    rules, collector, all_events, verdicts = RuleEngine(), Collector(), [], []
    first_saved = None
    for timestamp, x, y in points:
        vehicle = Detection("car", .95, (x - 10, y - 20, x + 10, y), 1)
        decisions = rules.evaluate(vehicle, timestamp, synthetic_scene(timestamp), video=True)
        before = deepcopy([decision.to_dict() for decision in decisions])
        verdicts.append(next(d.verdict for d in decisions if d.rule == Rule.RED_LIGHT))
        created, _ = collector.update(decisions, timestamp, detections=[vehicle],
                                      camera_stable=True, scene_revision=1)
        assert before == [decision.to_dict() for decision in decisions]
        all_events.extend(event for event in created if event.rule == Rule.RED_LIGHT)
        if len(all_events) == 1 and first_saved is None:
            first_saved = deepcopy(all_events[0])
    assert verdicts == [Verdict.UNKNOWN, Verdict.VIOLATION, Verdict.UNKNOWN, Verdict.VIOLATION]
    assert len(rules.history[1]) == 2  # Producer history contains only the new segment.
    assert len(all_events) == 2, "Fresh post-reset crossing must not inherit old event deduplication"
    assert all_events[0] == first_saved, "Historical event evidence must not be extended across a reset"
    assert all_events[0].event_id != all_events[1].event_id


def assessment(timestamp, verdict=Verdict.VIOLATION, track=1, rule=Rule.HELMET):
    return Decision(rule, verdict, "Synthetic lifecycle fixture", timestamp, track,
                    (40, 20, 60, 40), {"synthetic": True})


def detection(track=1, x=50):
    return Detection("motorcycle", .95, (x - 10, 20, x + 10, 40), track)


def test_observed_unknown_and_short_absence_preserve_gap_tolerance():
    collector = Collector()
    collector.update([assessment(0.)], 0., detections=[detection()])
    assert collector.update([assessment(.2, Verdict.UNKNOWN)], .2, detections=[detection()]) == ([], [])
    assert collector.update([], .4, detections=[]) == ([], [])
    created, _ = collector.update([assessment(.5)], .5, detections=[detection()])
    assert len(created) == 1 and created[0].started_at == 0. and created[0].observations == 2
    assert created[0].evidence == {"synthetic": True}


def test_recorded_long_absence_ends_pending_support_without_a_clear_verdict():
    collector = Collector()
    collector.update([assessment(0.)], 0., detections=[detection()])
    assert collector.update([], .8, detections=[]) == ([], [])
    assert not collector.pending
    trace = collector.lifecycle_snapshot()["boundaries"][0]
    assert trace["reason"] == "tracking_observation_gap" and trace["discarded_states"]["pending"] == 1
    assert not collector.update([assessment(.9)], .9, detections=[detection()])[0]
    created, _ = collector.update([assessment(1.4)], 1.4, detections=[detection()])
    assert len(created) == 1 and created[0].started_at == .9


def test_numeric_identity_change_and_local_jump_do_not_reset_other_tracks():
    collector = Collector()
    collector.update([assessment(0.), assessment(0., track=2)], 0.,
                     detections=[detection(), detection(2)])
    # Same numeric ID jumps; track 2 remains observed and independently supported.
    assert not collector.update([assessment(.2), assessment(.2, track=2)], .2,
                                detections=[detection(x=250), detection(2)])[0]
    created, _ = collector.update([assessment(.5, track=3), assessment(.5, track=2)], .5,
                                  detections=[detection(3), detection(2)])
    assert [(event.track_id, event.started_at) for event in created] == [(2, 0.)]
    assert not collector.update([assessment(.6, track=3)], .6, detections=[detection(3)])[0]
    assert not collector.active.get((3, Rule.HELMET, None))


def test_scene_identity_reset_preserves_historical_event_and_unknowns():
    collector = Collector()
    created, _ = collector.update([assessment(0., rule=Rule.RED_LIGHT)], 0.,
                                  detections=[detection()], camera_stable=True, scene_revision=1)
    historical, saved = created[0], deepcopy(created[0])
    values = {"scene_id": 1, "scene_revision": 2,
              "camera_stable": False, "detections": [detection()]}
    unknown = assessment(.1, Verdict.UNKNOWN, rule=Rule.RED_LIGHT)
    before = deepcopy(unknown.to_dict())
    assert collector.update([unknown], .1, **values) == ([], [])
    assert not collector.active and historical == saved and unknown.to_dict() == before
    assert "scene_identity_changed" in [row["reason"] for row in collector.lifecycle_snapshot()["boundaries"]]
    values["camera_stable"] = True
    created, _ = collector.update([assessment(.2, rule=Rule.RED_LIGHT)], .2, **values)
    assert len(created) == 1 and created[0].event_id != historical.event_id
    assert historical == saved


def test_camera_and_revision_metadata_alone_do_not_reclassify_independent_evidence():
    collector = Collector()
    created, _ = collector.update([assessment(0., rule=Rule.RED_LIGHT)], 0.,
                                  detections=[detection()], camera_stable=True, scene_revision=1)
    original = created[0]
    unknown = assessment(.1, Verdict.UNKNOWN, rule=Rule.RED_LIGHT)
    assert collector.update([unknown], .1, detections=[detection()],
                            camera_stable=False, scene_revision=2) == ([], [])
    assert not collector.lifecycle_snapshot()["boundaries"]
    assert original.last_seen_at == 0. and original.observations == 1
    created, _ = collector.update([assessment(.2, rule=Rule.RED_LIGHT)], .2,
                                  detections=[detection()], camera_stable=True, scene_revision=2)
    assert not created and unknown.verdict == Verdict.UNKNOWN


def test_actual_camera_warmup_and_unknowns_expire_prior_episode_without_extra_reset_policy():
    from dataclasses import replace

    from traffic_ai.event_continuity import ContradictionAwareEventCollector
    from traffic_ai.scene_lifecycle import RefreshAwareCameraStability
    from tests.test_scene_lifecycle import controlled_sequence

    sequence = controlled_sequence()
    sequence.insert(3, (1.25, sequence[0][1]))
    sequence.append((2.7, sequence[-1][1]))
    for factory in (ContradictionAwareEventCollector, Collector):
        collector, camera, rules, events, source_states = factory(), RefreshAwareCameraStability(), RuleEngine(), [], []
        revision = 1
        camera.acknowledge_scene(0, revision)
        for timestamp, pixels in sequence:
            stable, _ = camera.observe(pixels, timestamp)
            if stable and camera.scene_refresh_required:
                revision += 1
                assert camera.acknowledge_scene(0, revision)
            scene = replace(synthetic_scene(timestamp), stationary_camera=stable)
            y = 60 if timestamp in (1.25, 2.7) else 40
            vehicle = Detection("car", .95, (40, y - 20, 60, y), 1)
            decisions = rules.evaluate(vehicle, timestamp, scene, video=True)
            verdict = next(d.verdict for d in decisions if d.rule == Rule.RED_LIGHT)
            source_states.append((timestamp, verdict))
            if not stable:
                assert verdict == Verdict.UNKNOWN
            metadata = {"detections": [vehicle], "camera_stable": stable, "scene_revision": revision} if factory is Collector else {}
            created, _ = collector.update(decisions, timestamp, **metadata)
            events.extend(event for event in created if event.rule == Rule.RED_LIGHT)
        assert [(event.started_at, event.last_seen_at) for event in events] == [(1.25, 1.25), (2.7, 2.7)]
        assert [timestamp for timestamp, verdict in source_states if verdict == Verdict.VIOLATION] == [1.25, 2.7]
        assert camera.context_ready and revision == 2


def test_unrecorded_lifecycle_metadata_preserves_existing_policy_and_is_explicit():
    collector = Collector()
    collector.update([assessment(0.)], 0.)
    created, _ = collector.update([assessment(.9)], .9)
    assert len(created) == 1 and created[0].started_at == 0.
    snapshot = collector.confirmation_snapshot()
    assert snapshot["lifecycle"]["detection_observations"] == "not_recorded"
    assert snapshot["lifecycle"]["camera_stable"] is None
    snapshot["lifecycle"]["boundaries"].append({"invented": True})
    assert not collector.lifecycle_snapshot()["boundaries"]


def test_image_assessment_does_not_require_video_lifecycle_metadata():
    collector = Collector()
    created, displayed = collector.update([assessment(0., track=None)], 0., image=True,
                                          camera_stable=False, scene_revision=1)
    assert len(created) == len(displayed) == 1
    assert created[0].evidence["synthetic"] is True


def test_actual_analyzer_rehearses_explicit_lifecycle_wiring_with_mocked_inference(monkeypatch):
    from dataclasses import replace
    from types import SimpleNamespace

    import numpy as np

    from tests.test_scene_lifecycle import build_analyzer

    instance = build_analyzer(monkeypatch, False)
    points = [(0., 50, 40), (.1, 50, 60), (.2, 250, 40), (.3, 250, 60)]

    class Camera:
        def reset(self):
            pass

        def observe(self, *args):
            return True, 0.  # Explicit synthetic camera observation, not measured footage.

    class Resolver:
        def update_signals(self, *args):
            pass

        def scene_for(self, vehicle, scene, timestamp):
            return replace(scene, signal_observed_at=timestamp)

    class BridgedCollector(Collector):
        def update(self, decisions, timestamp, scene_id=0, image=False, **observed):
            if observed:
                # Primary now supplies this contract directly; verify and forward
                # its values rather than replacing the production argument path.
                assert observed == {"detections": instance.perception.observed,
                                    "camera_stable": instance.scene.stationary_camera,
                                    "scene_revision": instance.scene_revision}
                return super().update(decisions, timestamp, scene_id, image, **observed)
            # Historical isolated rehearsal against the unwired Analyzer.
            return super().update(decisions, timestamp, scene_id, image,
                                  detections=instance.perception.observed,
                                  camera_stable=instance.scene.stationary_camera,
                                  scene_revision=instance.scene_revision)

    def detect(*args, **kwargs):
        _, x, y = points[instance.frames]
        instance.perception.observed = [Detection("car", .95, (x - 10, y - 20, x + 10, y), 1)]
        return instance.perception.observed

    original_record = instance._record_scene

    def record(*args):
        original_record(*args)
        instance.scene = replace(synthetic_scene(0.), scene_id=instance.scene_id)
        instance.approach_resolver = Resolver()

    instance.camera, instance.events = Camera(), BridgedCollector()
    instance.perception.detect, instance._record_scene = detect, record
    pixels = np.zeros((200, 400, 3), np.uint8)
    all_events, states = [], []
    for timestamp, _, _ in points:
        rendered, analysis, created = instance.analyze(
            SimpleNamespace(color_rgb=pixels, timestamp=timestamp), pixels, video=True)
        assert len(analysis.decisions) == len(Rule)
        verdict = next(d.verdict for d in analysis.decisions if d.rule == Rule.RED_LIGHT)
        states.append(verdict)
        assert bool(rendered.any()) == (verdict == Verdict.VIOLATION)
        all_events.extend(created)
    assert states == [Verdict.UNKNOWN, Verdict.VIOLATION, Verdict.UNKNOWN, Verdict.VIOLATION]
    assert len(all_events) == 2 and instance.event_counts[Rule.RED_LIGHT.value] == 2
