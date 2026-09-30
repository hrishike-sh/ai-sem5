"""Synthetic protocol/evidence fixtures only; no live subscription requests."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
import time

import pytest
from PIL import Image

from traffic_ai.violation_auth import AuthError, PLAN_SCOPES, SubscriptionGrant
from traffic_ai.violation_llm import ExplanationError, SubscriptionExplainer, card_fingerprint
from traffic_ai.violation_images import image_fingerprint, prepare_evidence_image


@pytest.fixture
def card():
    return {"id": "synthetic-card", "rule": "speeding", "status": "insufficient_evidence",
            "timestamp": 2.5, "track_id": 7, "reason": "No calibrated ground speed is available.",
            "facts": [{"id": "motion", "text": "The track moved 12 pixels.", "value": 12},
                      {"id": "timing", "text": "The observed interval was 0.5 seconds.", "value": 0.5}],
            "missing_facts": [{"id": "calibration", "text": "Road-plane calibration is missing."}],
            "frame_ids": [4, 5], "source_revision": "synthetic-revision"}


def selection(card):
    return {"card_id": card["id"], "card_fingerprint": card_fingerprint(card),
            "fact_ids": ["motion"], "missing_fact_ids": ["calibration"], "frame_ids": [4]}


def event(value):
    return b"data: " + json.dumps(value).encode() + b"\n\n"


def completed(raw):
    text = raw if isinstance(raw, str) else json.dumps(raw)
    return event({"type": "response.completed", "response": {"status": "completed", "model": "fixture-model",
        "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}]}})


def done_message(card):
    return {"type": "response.output_item.done", "output_index": 0,
            "item": {"id": "msg_synthetic", "type": "message", "status": "completed", "phase": "final_answer",
                     "role": "assistant", "content": [{"type": "output_text", "text": json.dumps(selection(card))}]}}


def final_event(output=None):
    return {"type": "response.completed", "response": {"status": "completed", "model": "fixture-model",
                                                        "output": output if output is not None else []}}


@pytest.mark.parametrize("populated_final", [False, True])
def test_completed_item_replay_with_empty_or_matching_final(card, populated_final):
    done = done_message(card)
    stream = b"".join(event(item) for item in [
        {"type": "response.created", "response": {"status": "in_progress"}},
        {"type": "response.output_item.added", "output_index": 0,
         "item": {"id": "msg_synthetic", "type": "message", "status": "in_progress"}},
        {"type": "response.output_text.delta", "output_index": 0, "delta": "untrusted partial text"},
        done, final_event([done["item"]] if populated_final else None),
    ])
    service, _, _ = explainer(card, stream)
    result = service.explain(card, True)
    assert result["fact_ids"] == ["motion"] and result["frame_ids"] == [4]
    assert "untrusted partial" not in result["explanation"]


@pytest.mark.parametrize("case", [
    "negative-index", "bool-index", "large-index", "string-index", "missing-index", "gap",
    "duplicate-index", "duplicate-id", "conflict", "index-mismatch", "missing-item", "missing-id",
    "unfinished", "malformed-content", "tool", "no-completed", "delta-only", "late-item",
    "error-after-item", "error-after-completed", "duplicate-completed", "malformed-type", "malformed-status",
])
def test_completed_item_replay_rejects_ambiguous_or_incomplete_stream(card, case):
    done = done_message(card)
    ending = final_event()
    events = [done, ending]
    if case in {"negative-index", "bool-index", "large-index", "string-index", "gap"}:
        done["output_index"] = {"negative-index": -1, "bool-index": True, "large-index": 32,
                                "string-index": "0", "gap": 1}[case]
    elif case == "missing-index":
        del done["output_index"]
    elif case == "duplicate-index":
        events.insert(1, deepcopy(done))
    elif case == "duplicate-id":
        other = deepcopy(done)
        other["output_index"] = 1
        events.insert(1, other)
    elif case == "conflict":
        ending["response"]["output"] = [deepcopy(done["item"])]
        ending["response"]["output"][0]["content"][0]["text"] = "{}"
    elif case == "index-mismatch":
        done["output_index"] = 1
        ending["response"]["output"] = [deepcopy(done["item"])]
    elif case == "missing-item":
        del done["item"]
    elif case == "missing-id":
        del done["item"]["id"]
    elif case == "unfinished":
        done["item"]["status"] = "in_progress"
    elif case == "malformed-content":
        done["item"]["content"] = [{}]
    elif case == "tool":
        done["item"]["type"] = "function_call"
    elif case == "no-completed":
        events.pop()
    elif case == "delta-only":
        events[0] = {"type": "response.output_text.delta", "delta": json.dumps(selection(card))}
    elif case == "late-item":
        events.reverse()
    elif case in {"error-after-item", "error-after-completed"}:
        events.insert(1 if case == "error-after-item" else 2, {"type": "error", "code": "failed"})
    elif case == "duplicate-completed":
        events.append(deepcopy(ending))
    elif case == "malformed-type":
        done["item"]["type"] = []
    elif case == "malformed-status":
        done["item"]["status"] = []
    service, _, _ = explainer(card, b"".join(event(item) for item in events))
    with pytest.raises(ExplanationError):
        service.explain(card, True)


def test_done_item_selection_still_rejects_invented_evidence(card):
    done = done_message(card)
    invented = selection(card)
    invented["fact_ids"] = ["speed_120_kmh"]
    done["item"]["content"][0]["text"] = json.dumps(invented)
    service, _, _ = explainer(card, event(done) + event(final_event()))
    with pytest.raises(ExplanationError):
        service.explain(card, True)


class FakeAuth:
    def __init__(self, grant=None):
        self.calls = 0
        self.value = grant or SubscriptionGrant("synthetic-oauth", "oaiapp_fixture", "fixture-user",
                                                 frozenset(PLAN_SCOPES), time.time() + 600)

    def status(self):
        return {"available": True, "provider": "synthetic_fixture", "message": "Synthetic test fixture"}

    def grant(self):
        self.calls += 1
        return self.value


class FakeHTTP:
    def __init__(self, stream):
        self.stream = stream
        self.calls = []
        self.models = {"models": [{"slug": "fixture-model", "visibility": "list", "display_name": "Synthetic"}]}

    def json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.models

    @contextmanager
    def open(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if isinstance(self.stream, BaseException):
            raise self.stream
        with io.BytesIO(self.stream) as response:
            yield response


def explainer(card, raw=None):
    http = FakeHTTP(completed(selection(card)) if raw is None else raw)
    auth = FakeAuth()
    return SubscriptionExplainer(auth, http=http), auth, http


@pytest.mark.parametrize("consent", [False, None, "true", 1, []])
def test_no_network_or_grant_without_literal_consent(card, consent):
    service, auth, http = explainer(card)
    with pytest.raises(ExplanationError, match="Approve"):
        service.explain(card, consent)
    assert auth.calls == 0 and http.calls == []


def test_grounded_fixture_response_and_tool_free_request(card):
    original = deepcopy(card)
    service, auth, http = explainer(card)
    result = service.explain(card, True)
    assert card == original and auth.calls == 1
    assert result["explanation"] == "\n\n".join([card["reason"], card["facts"][0]["text"] + ": 12",
                                               "Missing evidence: " + card["missing_facts"][0]["text"]])
    assert result["status"] == "insufficient_evidence"
    assert result["frame_ids"] == [4] and result["card_fingerprint"] == card_fingerprint(card)
    url, params = http.calls[1]
    assert url == "https://api.openai.com/v1/responses"
    payload = params["data"]
    assert payload["tools"] == [] and payload["tool_choice"] == "none"
    assert payload["store"] is False and payload["stream"] is True
    assert not {"max_output_tokens", "temperature", "metadata", "previous_response_id"} & set(payload)
    assert payload["text"]["format"]["schema"]["additionalProperties"] is False
    assert "synthetic-oauth" not in json.dumps(result)
    sent_card = json.loads(payload["input"][0]["content"])
    assert sent_card["facts"][0]["text"] == card["facts"][0]["text"] + ": 12"
    assert "source_revision" not in sent_card and "review" not in sent_card


@pytest.mark.parametrize("change", [
    {"fact_ids": ["speed_120_kmh"]}, {"frame_ids": [999]}, {"frame_ids": [True]},
    {"frame_ids": ["4"]}, {"missing_fact_ids": ["invented"]}, {"missing_fact_ids": []},
    {"fact_ids": []}, {"fact_ids": ["motion", "motion"]}, {"card_id": "other"},
    {"card_fingerprint": "old"}, {"status": "violation"}, {"explanation": "Speed was 120 km/h."},
])
def test_rejects_unsupported_ids_verdicts_and_prose(card, change):
    value = selection(card)
    value.update(change)
    service, _, _ = explainer(card, completed(value))
    with pytest.raises(ExplanationError):
        service.explain(card, True)


@pytest.mark.parametrize("raw", ["{broken", "[]", '{"fact_ids":[],"fact_ids":[]}', "NaN", "x" * 20000],
                         ids=["broken", "array", "duplicate-key", "nan", "oversized"])
def test_rejects_malformed_json_duplicate_keys_and_oversized_text(card, raw):
    service, _, _ = explainer(card, completed(raw))
    with pytest.raises(ExplanationError):
        service.explain(card, True)


@pytest.mark.parametrize("stream", [
    b"", b"data: [DONE]\n\n", b"data: {\"type\":\"response.completed\"}",
    event({"type": "response.output_text.delta", "delta": "plausible"}),
    event({"type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}}),
    event({"type": "response.incomplete"}),
    event({"type": "response.output_item.added", "item": {"type": "function_call"}}),
    b":" + b"x" * 262145,
], ids=["empty", "done-only", "partial", "delta-only", "failed", "incomplete", "tool", "oversized"])
def test_only_completed_bounded_stream_can_succeed(card, stream):
    service, _, _ = explainer(card, stream)
    with pytest.raises(ExplanationError):
        service.explain(card, True)


def test_timeout_has_safe_error_no_retry(card):
    service, _, http = explainer(card, TimeoutError("secret server information"))
    with pytest.raises(ExplanationError) as error:
        service.explain(card, True)
    assert error.value.code == "timeout" and "secret" not in str(error.value)
    assert len(http.calls) == 2


@pytest.mark.parametrize("token,scopes", [("sk-synthetic-platform-key", PLAN_SCOPES), ("synthetic-oauth", {"openid"})])
def test_api_key_and_identity_only_grants_never_make_network_call(card, token, scopes):
    grant = SubscriptionGrant(token, "oaiapp_fixture", "fixture", frozenset(scopes), time.time() + 600)
    http = FakeHTTP(b"")
    with pytest.raises(ExplanationError, match="authorized"):
        SubscriptionExplainer(FakeAuth(grant), http=http).explain(card, True)
    assert not http.calls


def test_missing_login_reports_unavailable_without_discovering_other_credentials(card):
    class NoLogin(FakeAuth):
        def status(self):
            return {"available": False, "provider": "chatgpt_subscription", "message": "Sign in required"}
        def grant(self):
            raise AuthError("Sign in required", "authorization_required")
    http = FakeHTTP(b"")
    service = SubscriptionExplainer(NoLogin(), http=http)
    assert service.status()["available"] is False
    with pytest.raises(ExplanationError, match="Sign in required"):
        service.explain(card, True)
    assert not http.calls


def test_only_account_catalog_models_are_allowed(card):
    service, _, http = explainer(card)
    service.model = "unentitled-model"
    with pytest.raises(ExplanationError, match="unavailable"):
        service.explain(card, True)
    assert len(http.calls) == 1


def test_card_mutation_during_call_rejects_result(card):
    service, auth, _ = explainer(card)
    original_grant = auth.grant
    def mutate():
        card["source_revision"] = "changed"
        return original_grant()
    auth.grant = mutate
    with pytest.raises(ExplanationError) as error:
        service.explain(card, True)
    assert error.value.code == "stale_card"


def test_invalid_card_never_leaves_machine(card):
    service, auth, http = explainer(card)
    card["facts"][0]["value"] = float("nan")
    with pytest.raises(ExplanationError):
        service.explain(card, True)
    assert auth.calls == 0 and not http.calls


def test_status_never_requests_inference(card):
    service, auth, http = explainer(card)
    assert service.status()["provider"] == "synthetic_fixture"
    assert not http.calls and not auth.calls


def test_outbound_catalog_omits_private_metadata_but_preserves_canonical_values(card):
    card["facts"][0]["value"] = {"pixels": 12, "source_path": "C:\\private\\clip.mp4",
                                  "reviewer_note": "private note", "nested": ["/home/person/clip.mp4"]}
    card["review"] = {"note": "private top-level note"}
    service, _, http = explainer(card)
    result = service.explain(card, True)
    prompt = http.calls[1][1]["data"]["input"][0]["content"]
    assert "private" not in prompt and "/home/person" not in prompt
    assert 'pixels' in prompt
    assert result["facts"][0]["value"] == card["facts"][0]["value"]


def test_usage_limit_after_streaming_never_delivers_partial_explanation(card):
    stream = event({"type": "response.output_text.delta", "delta": "partial"}) + event({
        "type": "response.failed", "response": {"error": {"code": "subscription_sharing_usage_limit_exceeded"}}})
    service, _, _ = explainer(card, stream)
    with pytest.raises(ExplanationError) as error:
        service.explain(card, True)
    assert error.value.code == "usage_limit"


@pytest.fixture
def images():
    return [prepare_evidence_image(Image.new("RGB", (32, 24), "orange"), frame_id=4, timestamp=2.0)]


def visual_selection(card, images):
    value = selection(card)
    value["image_fingerprint"] = image_fingerprint(images)
    value["visual_observations"] = [
        {"frame_id": image["id"], "category": category, "observation": observation}
        for image in images for category, observation in [
            ("head_visibility", "unclear"), ("objects", "person_and_two_wheeler_visible"),
            ("direction_context", "direction_context_unverified")]]
    return value


def test_images_use_supported_content_and_leave_findings_immutable(card, images):
    original_card, original_images = deepcopy(card), deepcopy(images)
    service, auth, http = explainer(card, completed(visual_selection(card, images)))
    result = service.explain(card, True, images=images)
    assert auth.calls == 1 and len(http.calls) == 2
    assert http.calls[0][1]["max_bytes"] == 4 * 1024 * 1024
    payload = http.calls[1][1]["data"]
    assert payload["tools"] == [] and payload["tool_choice"] == "none"
    assert payload["store"] is False and payload["stream"] is True
    content = payload["input"][0]["content"]
    assert [item["type"] for item in content] == ["input_text", "input_text", "input_image"]
    assert content[2] == {"type": "input_image", "image_url": images[0]["data_url"], "detail": "auto"}
    assert json.loads(content[0]["text"])["image_fingerprint"] == image_fingerprint(images)
    assert "untrusted" in payload["instructions"] and "EVERY missing_fact ID" in payload["instructions"]
    schema = payload["text"]["format"]["schema"]
    assert set(schema["required"]) == set(visual_selection(card, images))
    variants = schema["properties"]["visual_observations"]["items"]["anyOf"]
    assert all(variant["additionalProperties"] is False for variant in variants)
    assert all(variant["properties"]["frame_id"] == {"type": "integer", "enum": [4]} for variant in variants)
    assert result["vision_mode"] == "selected_frames"
    assert result["image_fingerprint"] == image_fingerprint(images)
    assert len(result["visual_observations"]) == 3
    assert all(item["text"].startswith("Model observation: ") for item in result["visual_observations"])
    assert "not recorded findings" in result["visual_notice"]
    assert "data_url" not in json.dumps(result)
    assert result["missing_facts"] == card["missing_facts"]
    for key in ("rule", "status", "reason"):
        assert result[key] == card[key]
    assert result["explanation"] == explainer(card)[0].explain(card, True)["explanation"]
    assert card == original_card and images == original_images


def test_facts_only_explicitly_reports_no_images_or_observations(card):
    service, _, http = explainer(card)
    result = service.explain(card, True)
    assert result["vision_mode"] == "facts_only"
    assert result["images_sent"] == [] and result["visual_observations"] == []
    assert result["image_fingerprint"] is None and "no images" in result["visual_notice"]
    assert isinstance(http.calls[1][1]["data"]["input"][0]["content"], str)


@pytest.mark.parametrize("case", [
    "invented-frame", "unsent-frame", "bool-frame", "typed-frame", "extra-prose", "speed", "legal-verdict",
    "identity", "category", "wrong-category-value", "duplicate", "missing-category", "empty", "wrong-images",
    "erase-missing-facts", "rewrite-finding", "array-category", "array-observation",
])
def test_visual_schema_rejects_unbound_measurements_prose_and_missing_evidence(card, images, case):
    value = visual_selection(card, images)
    first = value["visual_observations"][0]
    if case in {"invented-frame", "unsent-frame", "bool-frame", "typed-frame"}:
        first["frame_id"] = {"invented-frame": 900, "unsent-frame": 5, "bool-frame": True, "typed-frame": "4"}[case]
    elif case == "extra-prose":
        first["text"] = "The rider is guilty."
    elif case in {"speed", "legal-verdict", "identity", "wrong-category-value"}:
        first["observation"] = {"speed": "120 km/h", "legal-verdict": "helmet_violation",
                                "identity": "named_person", "wrong-category-value": "person_visible"}[case]
    elif case == "category":
        first["category"] = "legal_conclusion"
    elif case == "duplicate":
        value["visual_observations"][1] = deepcopy(first)
    elif case == "missing-category":
        value["visual_observations"].pop()
    elif case == "empty":
        value["visual_observations"] = []
    elif case == "wrong-images":
        value["image_fingerprint"] = "0" * 64
    elif case == "erase-missing-facts":
        value["missing_fact_ids"] = []
    elif case == "rewrite-finding":
        value["status"] = "violation"
    elif case == "array-category":
        first["category"] = []
    elif case == "array-observation":
        first["observation"] = []
    service, _, _ = explainer(card, completed(value))
    with pytest.raises(ExplanationError):
        service.explain(card, True, images)


def test_visual_result_is_refused_when_no_images_were_sent(card, images):
    service, _, _ = explainer(card, completed(visual_selection(card, images)))
    with pytest.raises(ExplanationError):
        service.explain(card, True)


def test_image_request_never_silently_falls_back_to_facts_only(card, images):
    service, _, http = explainer(card)
    with pytest.raises(ExplanationError):
        service.explain(card, True, images)
    assert len(http.calls) == 2


@pytest.mark.parametrize("case", ["url", "hash", "unknown-frame", "too-many", "empty"])
def test_invalid_image_input_fails_before_grant_or_network(card, images, case):
    if case == "url":
        images[0]["data_url"] = "https://attacker.invalid/image.jpg"
    elif case == "hash":
        images[0]["sha256"] = "0" * 64
    elif case == "unknown-frame":
        images[0]["id"] = 9
    elif case == "too-many":
        images *= 4
    else:
        images.clear()
    service, auth, http = explainer(card)
    with pytest.raises(ExplanationError) as error:
        service.explain(card, True, images)
    assert error.value.code == "invalid_images"
    assert auth.calls == 0 and http.calls == []


def test_image_consent_precedes_decoding(card, images, monkeypatch):
    def forbidden(*args):
        pytest.fail("Image decoding was attempted before consent")
    monkeypatch.setattr("traffic_ai.violation_llm.validate_evidence_images", forbidden)
    service, auth, http = explainer(card)
    with pytest.raises(ExplanationError) as error:
        service.explain(card, False, images)
    assert error.value.code == "consent_required" and not auth.calls and not http.calls


def test_prepared_image_revision_change_during_request_rejects_result(card, images):
    service, auth, _ = explainer(card, completed(visual_selection(card, images)))
    original_grant = auth.grant
    def mutate():
        images[0]["timestamp"] = 88
        return original_grant()
    auth.grant = mutate
    with pytest.raises(ExplanationError) as error:
        service.explain(card, True, images)
    assert error.value.code == "stale_card"


def test_three_selected_frames_keep_type_and_category_bindings(card, images):
    card["frame_ids"].append("context-frame")
    images.extend([prepare_evidence_image(Image.new("RGB", (24, 32)), frame_id=frame, timestamp=3)
                   for frame in (5, "context-frame")])
    service, _, http = explainer(card, completed(visual_selection(card, images)))
    result = service.explain(card, True, images)
    assert len(result["visual_observations"]) == 9
    assert [item["id"] for item in result["images_sent"]] == [4, 5, "context-frame"]
    content = http.calls[1][1]["data"]["input"][0]["content"]
    assert sum(item["type"] == "input_image" for item in content) == 3
