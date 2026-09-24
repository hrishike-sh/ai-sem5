"""App-owned Sign in with ChatGPT grants; never reads Codex/browser credentials.

Protocol: https://developers.openai.com/siwc/token-sharing-open-source/sign-in
Creating this service is inert. Only an explicitly approved start creates local
state; only its one-time callback can exchange a code. The loopback HTTP server
must protect start/disconnect against CSRF and redact callback query strings.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
import ctypes
from dataclasses import dataclass, field
import hashlib
import hmac
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid


ISSUER = "https://auth.openai.com"
AUTHORIZE = ISSUER + "/api/accounts/authorize"
TOKEN = ISSUER + "/api/accounts/oauth/token"
DISCOVERY = ISSUER + "/.well-known/openid-configuration"
JWKS = ISSUER + "/.well-known/jwks.json"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
PLAN_SCOPES = {"resource.invoke", "chatgpt.tokens.use.direct"}
PROVIDER = "chatgpt_subscription"
USAGE_URL = "https://chatgpt.com/settings/usage"


class AuthError(ValueError):
    def __init__(self, message, code="auth_unavailable"):
        super().__init__(message)
        self.code = code


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("Nonfinite JSON")

    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    except (ValueError, TypeError, UnicodeError, RecursionError):
        raise AuthError("The service returned invalid JSON.", "invalid_response") from None


def read_chunks(response, limit, deadline):
    """Bound bytes and wall time; each network read also has a <=5s timeout."""
    total = 0
    while True:
        if time.monotonic() >= deadline:
            raise AuthError("The subscription request timed out.", "timeout")
        chunk = response.read1(min(8192, limit + 1 - total))
        if not chunk:
            return
        total += len(chunk)
        if total > limit:
            raise AuthError("The service response exceeded the size limit.", "response_too_large")
        yield chunk


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AuthError("A redirected authentication request was refused.", "redirect_refused")


class SubscriptionHTTP:
    """Only official endpoints; no proxy, redirects, API-key fallback or retries."""
    def __init__(self):
        self._opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())

    @contextmanager
    def open(self, url, *, data=None, bearer=None, form=False, timeout=30):
        parsed = urllib.parse.urlsplit(url)
        if (parsed.scheme != "https" or parsed.netloc not in {"auth.openai.com", "api.openai.com"}
                or parsed.query or parsed.fragment or parsed.username):
            raise AuthError("Unsupported subscription endpoint.", "endpoint_refused")
        headers = {"Accept": "application/json"}
        if bearer is not None:
            if not isinstance(bearer, str) or not bearer or bearer.startswith("sk-") or any(c.isspace() for c in bearer):
                raise AuthError("A ChatGPT plan grant is required; API keys are not accepted.", "subscription_required")
            headers["Authorization"] = "Bearer " + bearer
        body = None
        if data is not None:
            if form:
                body = urllib.parse.urlencode(data).encode("ascii")
                headers["Content-Type"] = "application/x-www-form-urlencoded"
            else:
                body = json.dumps(data, allow_nan=False).encode("utf-8")
                headers["Content-Type"] = "application/json"
        try:
            with self._opener.open(urllib.request.Request(url, data=body, headers=headers),
                                   timeout=min(timeout, 5)) as response:
                yield response
        except urllib.error.HTTPError as error:
            code = "http_error"
            if error.code in {401, 403}:
                code = "authorization_required"
            elif error.code == 429:
                code = "usage_limit"
            elif error.code == 400:
                # Token errors are classified without exposing any provider text.
                try:
                    detail = strict_json(b"".join(read_chunks(error, 16384, time.monotonic() + 5)))
                    if detail.get("error") == "invalid_grant":
                        code = "invalid_grant"
                except (AuthError, OSError, AttributeError):
                    pass
            error.close()
            raise AuthError("ChatGPT could not complete the request. Check connection and plan access.", code) from None
        except (TimeoutError, urllib.error.URLError, OSError):
            raise AuthError("The subscription service could not be reached or timed out.", "timeout") from None

    def json(self, url, **kwargs):
        deadline = time.monotonic() + kwargs.get("timeout", 30)
        with self.open(url, **kwargs) as response:
            raw = b"".join(read_chunks(response, 262144, deadline))
        return strict_json(raw)


class _Blob(ctypes.Structure):
    _fields_ = [("size", ctypes.c_ulong), ("data", ctypes.POINTER(ctypes.c_ubyte))]


def _dpapi(value, decrypt=False):
    if os.name != "nt":
        raise AuthError("Protected subscription storage currently requires Windows.", "secure_storage_unavailable")
    buffer = (ctypes.c_ubyte * len(value)).from_buffer_copy(value)
    source, target = _Blob(len(value), buffer), _Blob()
    crypt = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    operation = crypt.CryptUnprotectData if decrypt else crypt.CryptProtectData
    operation.argtypes = [ctypes.POINTER(_Blob), ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong, ctypes.POINTER(_Blob)]
    operation.restype = ctypes.c_int
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    kernel.LocalFree.restype = ctypes.c_void_p
    # CRYPTPROTECT_UI_FORBIDDEN, current Windows user scope (not machine scope).
    if not operation(ctypes.byref(source), None, None, None, None, 1, ctypes.byref(target)):
        raise AuthError("Windows could not protect or unlock the subscription record.", "secure_storage_unavailable")
    try:
        return ctypes.string_at(target.data, target.size)
    finally:
        kernel.LocalFree(target.data)


class ProtectedStore:
    """One Windows-user encrypted record, atomic writes and cross-process lock."""
    def __init__(self, directory=None):
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        self.directory = Path(directory) if directory else base / "ViolationLens" / "auth"
        self.path = self.directory / "subscription.dpapi"
        self._lock = threading.RLock()

    def load(self):
        if not self.path.exists():
            return {}
        try:
            with self.path.open("rb") as stream:
                encrypted = stream.read(131073)
            if len(encrypted) > 131072 or not encrypted.startswith(b"VLS1"):
                raise ValueError
            result = strict_json(_dpapi(encrypted[4:], decrypt=True))
            if not isinstance(result, dict):
                raise ValueError
            return result
        except (OSError, ValueError):
            raise AuthError("The protected subscription record cannot be read.", "secure_storage_unavailable") from None

    def save(self, value):
        encrypted = b"VLS1" + _dpapi(json.dumps(value, allow_nan=False).encode())
        self.directory.mkdir(parents=True, exist_ok=True)
        name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.directory, prefix="subscription-", delete=False) as stream:
                name = stream.name
                stream.write(encrypted)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(name, self.path)
        finally:
            if name and os.path.exists(name):
                os.unlink(name)

    @contextmanager
    def transaction(self):
        if os.name != "nt":
            raise AuthError("Protected subscription storage currently requires Windows.", "secure_storage_unavailable")
        import msvcrt
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            with (self.directory / "subscription.lock").open("a+b") as lock:
                if lock.seek(0, 2) == 0:
                    lock.write(b"0")
                    lock.flush()
                deadline = time.monotonic() + 5
                while True:
                    try:
                        lock.seek(0)
                        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                        break
                    except OSError:
                        if time.monotonic() >= deadline:
                            raise AuthError("Another subscription operation is busy.", "auth_busy") from None
                        time.sleep(0.05)
                try:
                    yield
                finally:
                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


@dataclass(frozen=True)
class SubscriptionGrant:
    access_token: str = field(repr=False)
    client_id: str
    subject: str = field(repr=False)
    scopes: frozenset[str]
    expires_at: float

    def valid(self):
        return (isinstance(self.access_token, str) and bool(self.access_token)
                and not self.access_token.startswith("sk-")
                and not any(c.isspace() for c in self.access_token)
                and self.client_id.startswith("oaiapp_") and bool(self.subject)
                and PLAN_SCOPES <= self.scopes and math.isfinite(self.expires_at)
                and self.expires_at > time.time())


class SubscriptionAuth:
    """Single selected account. Reauthorization must retain its verified identity."""
    def __init__(self, storage_dir=None, callback_uri="http://127.0.0.1:8521/auth/callback", *,
                 store=None, http=None, identity_verifier=None):
        parsed = urllib.parse.urlsplit(callback_uri)
        if (parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or not parsed.port
                or parsed.path != "/auth/callback" or parsed.query or parsed.fragment or parsed.username):
            raise AuthError("OAuth requires the app's 127.0.0.1 loopback callback.", "invalid_callback")
        self.callback_uri = callback_uri
        self.store = store if store is not None else ProtectedStore(storage_dir)
        self.http = http if http is not None else SubscriptionHTTP()
        self._verify_override = identity_verifier  # deterministic fixture seam; never exposed over HTTP
        self._pending = None
        self._pending_lock = threading.RLock()

    def _dependencies_ready(self):
        return self._verify_override is not None or (importlib.util.find_spec("jwt") is not None
                                                     and importlib.util.find_spec("cryptography") is not None)

    def status(self):
        result = {"available": False, "provider": PROVIDER, "connected": False,
                  "message": "Continue with ChatGPT to authorize this app's plan usage.",
                  "usage_url": USAGE_URL, "approval_required": True}
        if not self._dependencies_ready():
            result.update(message="The sign-in verification dependency is unavailable.", code="dependency_unavailable")
            return result
        try:
            record = self.store.load()
            credential = record.get("credential")
            if credential:
                scopes = set(credential.get("scopes", []))
                if not PLAN_SCOPES <= scopes:
                    result.update(message="ChatGPT plan usage was not granted.", code="plan_scope_missing")
                elif credential.get("access_token") and credential.get("subject"):
                    result.update(available=True, connected=True, approval_required=False,
                                  message="Using ChatGPT plan. Account eligibility is checked when requesting a model.")
        except (AuthError, TypeError, KeyError):
            result.update(message="The protected subscription record is unavailable.", code="secure_storage_unavailable")
        return result

    def start(self, approve=False, *, session_id=""):
        if approve is not True:
            raise AuthError("Approve Continue with ChatGPT before creating a sign-in request.", "consent_required")
        if not self._dependencies_ready():
            raise AuthError("The sign-in verification dependency is unavailable.", "dependency_unavailable")
        with self._pending_lock, self.store.transaction():
            record = self.store.load()
            if not record.get("host_id"):
                record["host_id"] = "urn:uuid:" + str(uuid.uuid4())
                self.store.save(record)
            registration = record.get("registration", {})
            client_id = registration.get("client_id", record.get("pending_client_id", "dynamic_agent_client"))
            verifier = secrets.token_urlsafe(64)
            state, nonce = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
            self._pending = {"state": state, "nonce": nonce, "verifier": verifier,
                             "expires_at": time.time() + 600, "client_id": client_id,
                             "subject": registration.get("subject"), "session_id": session_id}
            params = {"client_id": client_id, "ext_agent_host_id": record["host_id"],
                      "response_type": "code", "redirect_uri": self.callback_uri, "scope": SCOPES,
                      "resource": RESOURCE, "state": state, "nonce": nonce,
                      "code_challenge_method": "S256", "code_challenge": base64.urlsafe_b64encode(
                          hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()}
            if client_id == "dynamic_agent_client":
                params["agent_name_hint"] = "ViolationLens"
            # No tokens in frontend URLs; omission of the optional ID-token hint is supported.
            return {"authorization_url": AUTHORIZE + "?" + urllib.parse.urlencode(params),
                    "expires_in": 600, "provider": PROVIDER, "usage_url": USAGE_URL}

    def _discovery(self):
        discovery = self.http.json(DISCOVERY)
        expected = {"issuer": ISSUER, "authorization_endpoint": AUTHORIZE,
                    "token_endpoint": TOKEN, "jwks_uri": JWKS}
        if not isinstance(discovery, dict) or any(discovery.get(k) != v for k, v in expected.items()):
            raise AuthError("OpenAI identity discovery could not be verified.", "discovery_invalid")
        return discovery

    def _identity(self, token, client_id, nonce):
        if self._verify_override:
            return self._verify_override(token, client_id, nonce)
        import jwt
        try:
            header = jwt.get_unverified_header(token)
            # Only public-key signatures; never choose a symmetric algorithm from a token.
            if header.get("alg") not in {"RS256", "ES256"} or not isinstance(header.get("kid"), str):
                raise ValueError
            keys = self.http.json(JWKS)
            matches = [key for key in keys.get("keys", []) if key.get("kid") == header["kid"]]
            if len(matches) != 1:
                raise ValueError
            key = jwt.PyJWK.from_dict(matches[0], algorithm=header["alg"])
            claims = jwt.decode(token, key.key, algorithms=[header["alg"]], audience=client_id,
                                issuer=ISSUER, leeway=5,
                                options={"require": ["sub", "exp", "iat", "iss", "aud"]})
            if not isinstance(claims.get("sub"), str) or not claims["sub"]:
                raise ValueError
            if nonce is not None and not hmac.compare_digest(str(claims.get("nonce", "")), nonce):
                raise ValueError
            if claims.get("azp", client_id) != client_id:
                raise ValueError
            return claims
        except (jwt.PyJWTError, ValueError, TypeError, KeyError, AttributeError):
            raise AuthError("The ChatGPT identity token could not be verified.", "identity_invalid") from None

    def _credential(self, tokens, client_id, claims, old=None):
        old = old or {}
        if (not isinstance(tokens, dict) or not isinstance(tokens.get("token_type"), str)
                or tokens["token_type"].lower() != "bearer"):
            raise AuthError("The sign-in returned no usable plan grant.", "invalid_grant")
        scopes = tokens.get("scope")
        if not isinstance(scopes, str):
            raise AuthError("The sign-in did not confirm its granted permissions.", "plan_scope_missing")
        if not PLAN_SCOPES <= set(scopes.split()):
            raise AuthError("ChatGPT plan usage was not granted.", "plan_scope_missing")
        expiry = tokens.get("expires_in")
        if type(expiry) not in (int, float) or not math.isfinite(expiry) or not 0 < expiry <= 86400:
            raise AuthError("The sign-in returned invalid expiry information.", "invalid_grant")
        access = tokens.get("access_token")
        refresh = tokens.get("refresh_token", old.get("refresh_token"))
        if (not isinstance(access, str) or not access or access.startswith("sk-")
                or any(c.isspace() for c in access) or not isinstance(refresh, str) or not refresh):
            raise AuthError("A renewable ChatGPT plan grant is required.", "subscription_required")
        return {"access_token": access, "refresh_token": refresh,
                "id_token": tokens.get("id_token", old.get("id_token")), "client_id": client_id,
                "subject": claims["sub"], "issuer": ISSUER, "scopes": scopes.split(),
                "expires_at": time.time() + expiry}

    def complete(self, query, *, session_id=""):
        if not isinstance(query, dict) or any(not isinstance(v, str) or len(v) > 32768 for v in query.values()):
            raise AuthError("Invalid sign-in callback.", "callback_invalid")
        with self._pending_lock:
            pending = self._pending
            if (not pending or time.time() > pending["expires_at"]
                    or not hmac.compare_digest(query.get("state", ""), pending["state"])
                    or session_id != pending["session_id"]):
                raise AuthError("The sign-in callback is expired or does not match this attempt.", "state_invalid")
            self._pending = None  # consume once, including denied/failed attempts
        if query.get("error"):
            raise AuthError("ChatGPT sign-in was not authorized.", "authorization_denied")
        client_id = query.get("client_id", pending["client_id"])
        if (not client_id.startswith("oaiapp_") or (pending["client_id"] != "dynamic_agent_client"
                                                   and client_id != pending["client_id"])):
            raise AuthError("The sign-in returned a different or missing client registration.", "client_mismatch")
        if not query.get("code"):
            raise AuthError("The sign-in returned no authorization code.", "callback_invalid")
        with self.store.transaction():
            record = self.store.load()
            current_registration = record.get("registration", {})
            if current_registration and current_registration.get("client_id") != pending["client_id"]:
                raise AuthError("The selected registration changed during sign-in.", "client_mismatch")
            if not current_registration:
                # A code exchange can fail after registration succeeded. Retain the
                # issued ID so the next approved attempt doesn't create another app.
                record["pending_client_id"] = client_id
                self.store.save(record)
            self._discovery()
            tokens = self.http.json(TOKEN, form=True, data={"grant_type": "authorization_code",
                "client_id": client_id, "code": query["code"], "code_verifier": pending["verifier"],
                "redirect_uri": self.callback_uri, "resource": RESOURCE})
            if not isinstance(tokens, dict) or not isinstance(tokens.get("id_token"), str):
                raise AuthError("The sign-in returned no identity token.", "identity_invalid")
            claims = self._identity(tokens["id_token"], client_id, pending["nonce"])
            if pending["subject"] and claims.get("sub") != pending["subject"]:
                raise AuthError("The sign-in used a different ChatGPT account.", "identity_mismatch")
            credential = self._credential(tokens, client_id, claims)
            record.pop("pending_client_id", None)
            record.update(registration={"client_id": client_id, "subject": claims["sub"]}, credential=credential)
            self.store.save(record)
        return self.status()

    def grant(self, refresh=True):
        # Called only after per-explanation consent. Serialize the whole refresh and
        # atomic replacement across threads/processes; never replay a rotating token.
        if not self.store.load().get("credential"):
            raise AuthError("Continue with ChatGPT before requesting an explanation.", "authorization_required")
        with self.store.transaction():
            record = self.store.load()
            credential = record.get("credential", {})
            if credential.get("expires_at", 0) <= time.time() + 30:
                if not refresh:
                    raise AuthError("The ChatGPT grant needs renewal.", "authorization_required")
                try:
                    tokens = self.http.json(TOKEN, form=True, data={"grant_type": "refresh_token",
                        "client_id": credential["client_id"], "refresh_token": credential["refresh_token"],
                        "resource": RESOURCE})
                    if not isinstance(tokens, dict):
                        raise AuthError("The refreshed grant was invalid.", "invalid_grant")
                    if "scope" not in tokens:
                        tokens["scope"] = " ".join(credential["scopes"])
                    claims = {"sub": credential["subject"]}
                    if tokens.get("id_token"):
                        claims = self._identity(tokens["id_token"], credential["client_id"], None)
                        if claims["sub"] != credential["subject"]:
                            raise AuthError("The refreshed grant changed account identity.", "identity_mismatch")
                    credential = self._credential(tokens, credential["client_id"], claims, credential)
                    record["credential"] = credential
                    self.store.save(record)
                except AuthError as error:
                    # An uncertain refresh may have rotated remotely. Clear credentials
                    # and require fresh sign-in instead of retrying the old refresh token.
                    record.pop("credential", None)
                    self.store.save(record)
                    raise AuthError("The ChatGPT session needs a fresh sign-in.", error.code) from None
            grant = SubscriptionGrant(credential["access_token"], credential["client_id"],
                                      credential["subject"], frozenset(credential["scopes"]), credential["expires_at"])
            if not grant.valid():
                raise AuthError("A valid ChatGPT plan grant is required.", "subscription_required")
            return grant

    def disconnect(self, approve=False):
        if approve is not True:
            raise AuthError("Confirm disconnect before clearing this app's sign-in.", "consent_required")
        remote_revoked = False
        with self._pending_lock, self.store.transaction():
            self._pending = None
            record = self.store.load()
            credential = record.get("credential")
            if credential:
                try:
                    endpoint = self._discovery().get("revocation_endpoint")
                    if not isinstance(endpoint, str) or urllib.parse.urlsplit(endpoint).netloc != "auth.openai.com":
                        raise AuthError("Revocation endpoint unavailable.")
                    with self.http.open(endpoint, form=True, data={"token": credential["refresh_token"],
                            "token_type_hint": "refresh_token", "client_id": credential["client_id"]}):
                        remote_revoked = True
                except AuthError:
                    pass
                record.pop("credential", None)
                self.store.save(record)
        result = self.status()
        result["remote_revoked"] = remote_revoked
        result["message"] = ("Disconnected from ChatGPT." if remote_revoked else
                             "Disconnected locally. Remote revocation was not confirmed; manage access in ChatGPT settings.")
        return result
