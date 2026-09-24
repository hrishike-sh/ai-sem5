"""Synthetic protocol/evidence fixtures only; no live subscription requests."""
from contextlib import contextmanager
from copy import deepcopy
import io
import json
import time

import pytest

from traffic_ai.violation_auth import AuthError, PLAN_SCOPES, SubscriptionGrant
from traffic_ai.violation_llm import ExplanationError, SubscriptionExplainer, card_fingerprint


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
