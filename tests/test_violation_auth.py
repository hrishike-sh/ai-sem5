"""Synthetic OAuth protocol fixtures. Never authorize an account or use real tokens."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import base64
import io
import json
import os
import threading
import time
from urllib.parse import parse_qs, urlsplit

import pytest

from traffic_ai.violation_auth import (
    AUTHORIZE, DISCOVERY, ISSUER, JWKS, PLAN_SCOPES, RESOURCE, SCOPES, TOKEN,
    AuthError, ProtectedStore, SubscriptionAuth, SubscriptionHTTP,
)


class MemoryStore:
    """In-memory fixture substitute: no real credential configuration."""
    def __init__(self):
        self.value = {}
        self.saves = 0
        self.lock = threading.RLock()

    def load(self):
        return deepcopy(self.value)

    def save(self, value):
        self.saves += 1
        self.value = deepcopy(value)

    @contextmanager
    def transaction(self):
        with self.lock:
            yield


def token_response(**changes):
    result = {"access_token": "synthetic-access", "refresh_token": "synthetic-refresh",
              "id_token": "synthetic-id", "token_type": "Bearer", "expires_in": 3600, "scope": SCOPES}
    result.update(changes)
    return result


class FakeHTTP:
    def __init__(self):
        self.calls = []
        self.tokens = token_response()

    def json(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if url == DISCOVERY:
            return {"issuer": ISSUER, "authorization_endpoint": AUTHORIZE, "token_endpoint": TOKEN,
                    "jwks_uri": JWKS, "revocation_endpoint": ISSUER + "/api/accounts/oauth/revoke"}
        assert url == TOKEN
        if isinstance(self.tokens, BaseException):
            raise self.tokens
        return deepcopy(self.tokens)

    @contextmanager
    def open(self, url, **kwargs):
        self.calls.append((url, kwargs))
        yield io.BytesIO(b"")


@pytest.fixture
def service():
    store, http = MemoryStore(), FakeHTTP()
    verifier = lambda token, client, nonce: {"sub": "synthetic-user", "nonce": nonce}
    return SubscriptionAuth(store=store, http=http, identity_verifier=verifier)


def start(service):
    result = service.start(True)
    return parse_qs(urlsplit(result["authorization_url"]).query)


def connect(service):
    params = start(service)
    return service.complete({"state": params["state"][0], "code": "synthetic-code", "client_id": "oaiapp_fixture"})


@pytest.mark.parametrize("approval", [False, None, 1, "true"])
def test_start_requires_literal_approval_and_has_no_side_effects(service, approval):
    with pytest.raises(AuthError) as error:
        service.start(approval)
    assert error.value.code == "consent_required"
    assert service.store.saves == 0 and not service.http.calls


def test_approved_url_has_pkce_scope_state_nonce_and_persistent_host(service):
    first = start(service)
    assert first["client_id"] == ["dynamic_agent_client"]
    assert first["agent_name_hint"] == ["ViolationLens"]
    assert first["redirect_uri"] == ["http://127.0.0.1:8521/auth/callback"]
    assert first["resource"] == [RESOURCE] and set(first["scope"][0].split()) == set(SCOPES.split())
    challenge = base64.urlsafe_b64encode(hashlib.sha256(service._pending["verifier"].encode()).digest()).rstrip(b"=").decode()
    assert first["code_challenge"] == [challenge] and first["code_challenge_method"] == ["S256"]
    assert not service.http.calls  # start constructs a URL; no registration is executed
    second = start(service)
    assert first["ext_agent_host_id"] == second["ext_agent_host_id"]
    assert first["state"] != second["state"] and first["nonce"] != second["nonce"]
    assert "code_verifier" not in first and "id_token_hint" not in first


def test_status_before_connection_is_local_and_unavailable(service):
    assert service.status()["available"] is False
    assert service.store.saves == 0 and not service.http.calls


def test_valid_callback_commits_only_verified_own_grant(service):
    status = connect(service)
    assert status["available"] and status["connected"]
    assert "synthetic-access" not in json.dumps(status)
    token_call = next(kwargs for url, kwargs in service.http.calls if url == TOKEN)
    assert token_call["data"]["client_id"] == "oaiapp_fixture"
    assert token_call["data"]["redirect_uri"] == service.callback_uri
    assert token_call["data"]["resource"] == RESOURCE
    assert "client_secret" not in token_call["data"]
    grant = service.grant()
    assert grant.valid() and PLAN_SCOPES <= grant.scopes
    assert "synthetic-access" not in repr(grant)


def test_wrong_state_and_browser_session_make_no_exchange(service):
    params = parse_qs(urlsplit(service.start(True, session_id="browser-a")["authorization_url"]).query)
    for query, session in [({"state": "wrong"}, "browser-a"), ({"state": params["state"][0]}, "browser-b")]:
        with pytest.raises(AuthError) as error:
            service.complete(query, session_id=session)
        assert error.value.code == "state_invalid"
    assert not service.http.calls and not service.store.value.get("credential")


def test_denial_consumes_attempt_without_token_exchange(service):
    state = start(service)["state"][0]
    with pytest.raises(AuthError) as error:
        service.complete({"state": state, "error": "access_denied"})
    assert error.value.code == "authorization_denied" and not service.http.calls
    with pytest.raises(AuthError):
        service.complete({"state": state, "code": "fixture", "client_id": "oaiapp_fixture"})


def test_successful_callback_cannot_be_replayed(service):
    state = start(service)["state"][0]
    query = {"state": state, "code": "fixture", "client_id": "oaiapp_fixture"}
    service.complete(query)
    before = len(service.http.calls)
    with pytest.raises(AuthError, match="expired"):
        service.complete(query)
    assert len(service.http.calls) == before


@pytest.mark.parametrize("client_id", ["dynamic_agent_client", "", "other_client"])
def test_missing_or_unissued_client_cannot_exchange(service, client_id):
    state = start(service)["state"][0]
    with pytest.raises(AuthError):
        service.complete({"state": state, "code": "fixture", "client_id": client_id})
    assert not service.http.calls


@pytest.mark.parametrize("changes", [{"scope": "openid profile"}, {"access_token": "sk-fixture"},
                                    {"expires_in": float("nan")}, {"expires_in": True},
                                    {"token_type": "Basic"}, {"refresh_token": ""}])
def test_bad_grants_do_not_become_connected(service, changes):
    service.http.tokens = token_response(**changes)
    with pytest.raises(AuthError):
        connect(service)
    assert service.status()["available"] is False
    assert "credential" not in service.store.value


def test_bad_identity_does_not_write_credentials(service):
    def reject(*args):
        raise AuthError("Identity rejected", "identity_invalid")
    service._verify_override = reject
    with pytest.raises(AuthError):
        connect(service)
    assert "credential" not in service.store.value


def test_failed_initial_exchange_retains_issued_client_for_next_approved_attempt(service):
    service.http.tokens = AuthError("expired code", "invalid_grant")
    with pytest.raises(AuthError):
        connect(service)
    assert "credential" not in service.store.value
    params = start(service)
    assert params["client_id"] == ["oaiapp_fixture"] and "agent_name_hint" not in params


def test_reauthorization_retains_client_host_and_rejects_account_switch(service):
    connect(service)
    old = deepcopy(service.store.value)
    params = start(service)
    assert params["client_id"] == ["oaiapp_fixture"] and "agent_name_hint" not in params
    assert params["ext_agent_host_id"] == [old["host_id"]]
    service._verify_override = lambda *args: {"sub": "wrong-user"}
    with pytest.raises(AuthError) as error:
        service.complete({"state": params["state"][0], "code": "new"})
    assert error.value.code == "identity_mismatch" and service.store.value == old


def test_reauthorization_rejects_different_client_before_exchange(service):
    connect(service)
    state = start(service)["state"][0]
    before = len(service.http.calls)
    with pytest.raises(AuthError):
        service.complete({"state": state, "code": "new", "client_id": "oaiapp_other"})
    assert len(service.http.calls) == before


def test_refresh_replaces_rotation_and_serializes_concurrent_calls(service):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    service.http.tokens = token_response(access_token="rotated-access", refresh_token="rotated-refresh")
    service.http.calls.clear()
    values = []
    threads = [threading.Thread(target=lambda: values.append(service.grant())) for _ in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(values) == 3 and all(g.access_token == "rotated-access" for g in values)
    assert len(service.http.calls) == 1
    assert service.http.calls[0][1]["data"]["grant_type"] == "refresh_token"
    assert "scope" not in service.http.calls[0][1]["data"]
    assert service.store.value["credential"]["refresh_token"] == "rotated-refresh"


def test_uncertain_refresh_clears_grant_and_never_retries_rotating_token(service):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    service.http.tokens = AuthError("network interrupted", "timeout")
    with pytest.raises(AuthError, match="fresh sign-in"):
        service.grant()
    assert "credential" not in service.store.value
    count = len(service.http.calls)
    with pytest.raises(AuthError):
        service.grant()
    assert len(service.http.calls) == count
    assert service.store.value["registration"]["client_id"] == "oaiapp_fixture"


@pytest.mark.parametrize("failure", [OSError("synthetic private storage detail"),
                                    AuthError("synthetic private DPAPI detail", "secure_storage_unavailable")],
                         ids=["disk", "dpapi"])
def test_refresh_cannot_start_when_pending_marker_cannot_be_persisted(service, monkeypatch, failure):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    previous = deepcopy(service.store.value)
    service.http.calls.clear()
    def fail_save(value):
        assert value["refresh_pending"] is True and "credential" not in value
        raise failure
    monkeypatch.setattr(service.store, "save", fail_save)
    with pytest.raises(AuthError) as error:
        service.grant()
    assert error.value.code == "secure_storage_unavailable"
    assert "synthetic private" not in str(error.value)
    assert not service.http.calls and service.store.value == previous


def _reopened_fixture(service):
    """A fresh service sees only persisted memory bytes, never previous locals."""
    store = MemoryStore()
    store.value = deepcopy(service.store.value)
    return SubscriptionAuth(store=store, http=FakeHTTP(), identity_verifier=service._verify_override)


def test_post_rotation_persistence_failure_never_replays_old_token(service, monkeypatch):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    registration = deepcopy(service.store.value["registration"])
    host_id = service.store.value["host_id"]
    service.http.calls.clear()
    original_save, original_json = service.store.save, service.http.json
    def fail_final_save(value):
        if "credential" in value:
            raise OSError("synthetic private disk path")
        original_save(value)
    def refresh_after_pending(url, **kwargs):
        assert service.store.value["refresh_pending"] is True
        assert "credential" not in service.store.value
        return original_json(url, **kwargs)
    monkeypatch.setattr(service.store, "save", fail_final_save)
    monkeypatch.setattr(service.http, "json", refresh_after_pending)
    with pytest.raises(AuthError, match="fresh sign-in") as error:
        service.grant()
    assert error.value.code == "secure_storage_unavailable" and "private" not in str(error.value)
    assert len(service.http.calls) == 1
    assert service.store.value == {"host_id": host_id, "registration": registration, "refresh_pending": True}
    status = service.status()
    assert not status["available"] and not status["connected"] and status["approval_required"]
    assert "synthetic-access" not in json.dumps(status) and "synthetic-refresh" not in json.dumps(status)
    with pytest.raises(AuthError, match="fresh sign-in"):
        service.grant()
    reopened = _reopened_fixture(service)
    with pytest.raises(AuthError, match="fresh sign-in"):
        reopened.grant()
    assert len(service.http.calls) == 1 and not reopened.http.calls


@pytest.mark.parametrize("failure", [AuthError("synthetic response secret", "invalid_grant"),
                                    AuthError("synthetic transport secret", "timeout")], ids=["invalid-grant", "timeout"])
def test_failed_refresh_retains_durable_pending_state_without_cleanup_write(service, monkeypatch, failure):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    service.http.calls.clear()
    service.http.tokens = failure
    save = service.store.save
    writes = []
    def only_pending_write(value):
        writes.append(deepcopy(value))
        assert len(writes) == 1  # Recovery must not depend on a second disk write.
        save(value)
    monkeypatch.setattr(service.store, "save", only_pending_write)
    with pytest.raises(AuthError) as error:
        service.grant()
    assert error.value.code == failure.code and "synthetic" not in str(error.value)
    assert service.store.value["refresh_pending"] is True and "credential" not in service.store.value
    reopened = _reopened_fixture(service)
    with pytest.raises(AuthError):
        reopened.grant()
    assert not reopened.http.calls and len(service.http.calls) == 1


def test_process_interruption_during_refresh_leaves_no_replayable_credential(service, monkeypatch):
    class SyntheticInterruption(BaseException):
        pass
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    def interrupted_request(*args, **kwargs):
        assert service.store.value["refresh_pending"] is True
        raise SyntheticInterruption()
    monkeypatch.setattr(service.http, "json", interrupted_request)
    with pytest.raises(SyntheticInterruption):
        service.grant()
    reopened = _reopened_fixture(service)
    with pytest.raises(AuthError, match="fresh sign-in"):
        reopened.grant()
    assert not reopened.http.calls and "credential" not in reopened.store.value


def test_uncertain_final_write_that_committed_uses_new_grant_without_rotation(service, monkeypatch):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    service.http.calls.clear()
    service.http.tokens = token_response(access_token="synthetic-rotated", refresh_token="synthetic-rotated-refresh")
    save = service.store.save
    def commit_then_fail(value):
        save(value)
        if "credential" in value:
            raise OSError("synthetic failure after atomic replacement")
    monkeypatch.setattr(service.store, "save", commit_then_fail)
    with pytest.raises(AuthError):
        service.grant()
    reopened = _reopened_fixture(service)
    assert reopened.grant().access_token == "synthetic-rotated"
    assert "refresh_pending" not in reopened.store.value
    assert not reopened.http.calls and len(service.http.calls) == 1


def test_other_service_waits_for_in_progress_refresh_then_uses_new_grant(service, monkeypatch):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    service.http.calls.clear()
    service.http.tokens = token_response(access_token="synthetic-rotated", refresh_token="synthetic-rotated-refresh")
    other = SubscriptionAuth(store=service.store, http=service.http, identity_verifier=service._verify_override)
    entered, release, second_waiting = threading.Event(), threading.Event(), threading.Event()
    original_json, original_transaction = service.http.json, service.store.transaction
    def blocked_refresh(url, **kwargs):
        entered.set()
        assert release.wait(3)
        return original_json(url, **kwargs)
    @contextmanager
    def observed_transaction():
        if threading.current_thread().name == "second-refresh":
            second_waiting.set()
        with original_transaction():
            yield
    monkeypatch.setattr(service.http, "json", blocked_refresh)
    monkeypatch.setattr(service.store, "transaction", observed_transaction)
    values, errors = [], []
    def call(auth):
        try:
            values.append(auth.grant())
        except BaseException as error:
            errors.append(type(error).__name__)
    first = threading.Thread(target=call, args=(service,))
    second = threading.Thread(target=call, args=(other,), name="second-refresh")
    try:
        first.start()
        assert entered.wait(3)
        second.start()
        assert second_waiting.wait(3)
    finally:
        release.set()
        first.join(3)
        if second.ident is not None:
            second.join(3)
    assert not first.is_alive() and not second.is_alive() and not errors
    assert len(values) == 2 and all(value.access_token == "synthetic-rotated" for value in values)
    assert len(service.http.calls) == 1 and "refresh_pending" not in service.store.value


def test_fresh_authorized_sign_in_recovers_pending_record_with_same_registration(service):
    connect(service)
    registration = deepcopy(service.store.value["registration"])
    host = service.store.value["host_id"]
    service.store.value.pop("credential")
    service.store.value["refresh_pending"] = True
    assert not service.status()["available"]
    assert connect(service)["connected"] is True
    assert service.store.value["registration"] == registration and service.store.value["host_id"] == host
    assert "refresh_pending" not in service.store.value


def test_expired_grant_without_refresh_permission_never_mutates_or_calls(service):
    connect(service)
    service.store.value["credential"]["expires_at"] = 0
    previous, saves = deepcopy(service.store.value), service.store.saves
    service.http.calls.clear()
    with pytest.raises(AuthError, match="renewal"):
        service.grant(refresh=False)
    assert service.store.value == previous and service.store.saves == saves and not service.http.calls


def test_disconnect_clears_pending_marker_without_reusing_lost_grant(service):
    connect(service)
    service.store.value.pop("credential")
    service.store.value["refresh_pending"] = True
    service.http.calls.clear()
    result = service.disconnect(True)
    assert not result["available"] and result["remote_revoked"] is False
    assert "refresh_pending" not in service.store.value and not service.http.calls


def test_disconnect_revokes_then_retains_mapping_and_host(service):
    connect(service)
    host = service.store.value["host_id"]
    status = service.disconnect(True)
    assert status["remote_revoked"] is True and status["available"] is False
    assert "credential" not in service.store.value
    assert service.store.value["host_id"] == host and service.store.value["registration"]


def test_secure_store_is_inert_until_save(tmp_path):
    store = ProtectedStore(tmp_path / "not-created")
    assert store.load() == {} and not store.directory.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows DPAPI")
def test_synthetic_record_is_dpapi_encrypted_and_round_trips(tmp_path):
    store = ProtectedStore(tmp_path / "fixture")
    # These are invented test bytes, not authentication credentials.
    value = {"fixture": "synthetic-sensitive-text"}
    with store.transaction():
        store.save(value)
    assert b"synthetic-sensitive-text" not in store.path.read_bytes()
    assert store.load() == value


def test_http_refuses_unofficial_endpoint_and_platform_key_before_open():
    http = SubscriptionHTTP()
    for url, kwargs in [("https://evil.example/", {}), (RESOURCE + "/models", {"bearer": "sk-fixture"})]:
        with pytest.raises(AuthError):
            with http.open(url, **kwargs):
                pytest.fail("Network request must never be reached")


def test_catalog_response_size_is_explicit_and_auth_default_stays_small(monkeypatch):
    http = SubscriptionHTTP()
    raw = b'{"models":[],"synthetic_padding":"' + b"x" * 300000 + b'"}'
    calls = []
    @contextmanager
    def fake_open(url, **kwargs):
        calls.append((url, kwargs))
        yield io.BytesIO(raw)
    monkeypatch.setattr(http, "open", fake_open)
    with pytest.raises(AuthError) as error:
        http.json(RESOURCE + "/models")
    assert error.value.code == "response_too_large"
    assert http.json(RESOURCE + "/models", max_bytes=4 * 1024 * 1024)["models"] == []
    with pytest.raises(AuthError) as error:
        http.json(TOKEN)
    assert error.value.code == "response_too_large"
    assert all("max_bytes" not in kwargs for _, kwargs in calls)


def test_json_exact_size_boundary_and_one_extra_byte(monkeypatch):
    http = SubscriptionHTTP()
    raw = b'{"fixture":"synthetic"}'
    @contextmanager
    def fake_open(*args, **kwargs):
        yield io.BytesIO(raw)
    monkeypatch.setattr(http, "open", fake_open)
    assert http.json(DISCOVERY, max_bytes=len(raw)) == {"fixture": "synthetic"}
    with pytest.raises(AuthError) as error:
        http.json(DISCOVERY, max_bytes=len(raw) - 1)
    assert error.value.code == "response_too_large"


@pytest.mark.parametrize("limit", [None, True, 0, -1, 1.5, "400000", 4 * 1024 * 1024 + 1])
def test_invalid_response_size_limit_never_opens_network(monkeypatch, limit):
    http = SubscriptionHTTP()
    monkeypatch.setattr(http, "open", lambda *a, **k: pytest.fail("No network request is allowed"))
    with pytest.raises(AuthError) as error:
        http.json(RESOURCE + "/models", max_bytes=limit)
    assert error.value.code == "invalid_response_limit"


@pytest.mark.parametrize("uri", ["http://localhost:8521/auth/callback", "https://127.0.0.1:8521/auth/callback",
                               "http://127.0.0.1:8521/callback", "http://example.com:8521/auth/callback"])
def test_callback_is_fixed_loopback_origin(uri):
    with pytest.raises(AuthError):
        SubscriptionAuth(callback_uri=uri)


def test_real_jwt_validation_with_synthetic_signed_token(service):
    jwt = pytest.importorskip("jwt")
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    jwk = json.loads(jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key()))
    jwk["kid"] = "synthetic-key"
    service._verify_override = None
    original_json = service.http.json
    service.http.json = lambda url, **kw: {"keys": [jwk]} if url == JWKS else original_json(url, **kw)
    claims = {"sub": "synthetic-user", "iss": ISSUER, "aud": "oaiapp_fixture",
              "exp": time.time() + 300, "iat": time.time(), "nonce": "expected"}
    signed = jwt.encode(claims, key, algorithm="RS256", headers={"kid": "synthetic-key"})
    assert service._identity(signed, "oaiapp_fixture", "expected")["sub"] == "synthetic-user"
    for client, nonce in [("oaiapp_wrong", "expected"), ("oaiapp_fixture", "wrong")]:
        with pytest.raises(AuthError):
            service._identity(signed, client, nonce)
    forged = jwt.encode(claims, "synthetic-shared-key-at-least-32-bytes", algorithm="HS256", headers={"kid": "synthetic-key"})
    with pytest.raises(AuthError):
        service._identity(forged, "oaiapp_fixture", "expected")
    for overrides in ({"iss": "https://other.example"}, {"exp": time.time() - 60}):
        invalid = jwt.encode({**claims, **overrides}, key, algorithm="RS256", headers={"kid": "synthetic-key"})
        with pytest.raises(AuthError):
            service._identity(invalid, "oaiapp_fixture", "expected")
