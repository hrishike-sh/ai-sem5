from traffic_ai.contracts import Detection
from traffic_ai.riders import RiderAssociator, associate_riders
import json
from pathlib import Path

from traffic_ai.contracts import Rule, Verdict
from traffic_ai.rules import RuleEngine, SceneEvidence


def detections(bike_x=0, person_x=0):
    bike = Detection("motorcycle", .95, (20 + bike_x, 60, 100 + bike_x, 120), 1)
    person = Detection("person", .95, (40 + person_x, 20, 70 + person_x, 110), 2)
    head = Detection("no_helmet", .95, (46 + person_x, 22, 62 + person_x, 40))
    return bike, person, head


def test_shared_motion_confirms_rider_and_head():
    associator = RiderAssociator()
    for index in range(8):
        bike, person, head = detections(index * 2, index * 2)
        result = associator.observe([bike], [person], [head], index * .1, True)[0]
        if index < 5:
            assert not result.association_confident
    assert result.association_confident
    assert len(result.bare_heads) == 1
    assert not result.visibility_complete


def test_pedestrian_beside_moving_bike_is_not_confirmed():
    associator = RiderAssociator()
    for index in range(8):
        bike, person, head = detections(index * 2, 0)
        result = associator.observe([bike], [person], [head], index * .1, True)[0]
        assert not result.association_confident


def test_stationary_pair_and_camera_motion_do_not_prove_shared_motion():
    associator = RiderAssociator()
    for index in range(8):
        bike, person, head = detections()
        assert not associator.observe([bike], [person], [head], index * .1, True)[0].association_confident
    for index in range(8, 16):
        bike, person, head = detections(index * 2, index * 2)
        assert not associator.observe([bike], [person], [head], index * .1, False)[0].association_confident
    assert not associator.pairs


def test_tracking_gap_loses_rider_confirmation():
    associator = RiderAssociator()
    for index in range(8):
        bike, person, head = detections(index * 2, index * 2)
        associator.observe([bike], [person], [head], index * .1, True)
    assert not associator.observe([bike], [person], [head], 3, True)[0].association_confident


def test_shared_head_is_not_counted_as_two_bare_heads():
    bike, person, head = detections()
    other = Detection("person", .9, (42, 20, 72, 110), 3)
    result = associate_riders([bike], [person, other], [head])[0]
    assert not result.bare_heads
    assert not result.association_confident


def test_recorded_indian_images_reject_background_riders_without_losing_helmet_evidence():
    cases = json.loads((Path(__file__).parent / "fixtures/rider_association_cases.json").read_text())
    expected_counts = [[1, 1], [2, 1, 1, 1], [0, 1, 1, 1, 0]]
    for case, counts, helmet_count in zip(cases, expected_counts, [1, 1, 2]):
        detections = [Detection(**d) for d in case["detections"]]
        bikes = [d for d in detections if d.label == "motorcycle"]
        people = [d for d in detections if d.label == "person"]
        heads = [d for d in detections if d.model == "iam-tsr/yolov8n-helmet-detection"]
        associations = associate_riders(bikes, people, heads)
        assert [len(associations[i].associated_people) for i in range(len(bikes))] == counts
        engine = RuleEngine()
        violations = [d.rule for i, bike in enumerate(bikes)
                      for d in engine.evaluate(bike, 0, SceneEvidence(), associations[i]) if d.verdict == Verdict.VIOLATION]
        assert Rule.RIDERS not in violations
        assert violations.count(Rule.HELMET) == helmet_count


def test_three_compatible_riders_remain_detectable():
    bike = Detection("motorcycle", .95, (10, 60, 110, 120))
    people = [Detection("person", .95, (x, 20, x + 20, 110)) for x in [20, 45, 70]]
    evidence = associate_riders([bike], people, [])[0]
    decisions = RuleEngine().evaluate(bike, 0, SceneEvidence(), evidence)
    assert next(d.verdict for d in decisions if d.rule == Rule.RIDERS) == Verdict.VIOLATION
