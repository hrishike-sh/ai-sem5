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
