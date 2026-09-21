"""Evaluation bookkeeping tests, independent of any live model."""
import hashlib
import json

from PIL import Image
import pytest

from scripts.evaluate_local_vision import load_cases, score


def test_scoring_separates_completed_miss_false_positive_and_fallback():
    case = {"allowed_features":["car","road"],"required_features":["car"],"expected_uncertainties":["occlusion"]}
    record = {"state":"complete","selection":{"visible_features":["road"],"uncertainty_codes":[]},
              "runtime":{"owned_server_stopped":True}}
    miss = score(record,case)
    assert miss["completed"] and miss["supported_claims_only"] and not miss["feature_target_met"]
    assert miss["missing_required_features"] == ["car"] and miss["missing_expected_uncertainties"] == ["occlusion"]
    false_positive = score({**record,"selection":{"visible_features":["person"]}},case)
    assert false_positive["unsupported_features"] == ["person"] and not false_positive["supported_claims_only"]
    fallback = score({"state":"unavailable","runtime":{"owned_server_stopped":True}},case)
    assert not fallback["completed"] and not fallback["feature_target_met"]
    assert fallback["fallback_has_no_observation"] and fallback["missing_required_features"] == []
    assert not score({"state":"cancelled","observation":{}},case)["fallback_has_no_observation"]


def test_input_manifest_pins_hashes_and_keeps_labels_separate(tmp_path):
    image = tmp_path/"image.png"
    Image.new("RGB",(8,8),"black").save(image)
    manifest = tmp_path/"cases.json"
    case = {"id":"blank","image":"image.png","image_sha256":hashlib.sha256(image.read_bytes()).hexdigest(),
            "allowed_features":[],"required_features":[]}
    document = {"synthetic":True,"cases":[case]}
    manifest.write_text(json.dumps(document))
    loaded = load_cases(manifest)
    assert loaded["cases"][0]["image"] == str(image.resolve())
    image.write_bytes(b"changed")
    with pytest.raises(ValueError,match="changed before inference"):
        load_cases(manifest)


@pytest.mark.parametrize("change", [{"id":"../outside"},{"allowed_features":["legal_violation"]},
                                    {"required_features":["car"]}])
def test_unsupported_labels_and_escaping_output_names_rejected(tmp_path,change):
    path = tmp_path/"cases.json"
    path.write_text(json.dumps({"synthetic":True,"cases":[{
        "id":"case","image":"unused.png","image_sha256":"none","allowed_features":[],**change}]}))
    with pytest.raises(ValueError):
        load_cases(path)
