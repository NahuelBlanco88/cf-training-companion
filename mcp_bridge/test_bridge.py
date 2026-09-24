"""Offline tests; deliberately never import or start the production app.py."""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from mcp.server.auth.provider import AccessToken
from mcp.server.fastmcp.exceptions import ToolError

import mcp_bridge.server as bridge
from mcp_bridge.auth import JwtVerifier
from mcp_bridge.backend import Backend
from mcp_bridge.config import Config


@pytest.fixture
def config():
    return Config(
        api_url="http://127.0.0.1:9000", api_token="local-test-secret",
        public_url="http://127.0.0.1:8080/mcp", oauth_issuer="http://127.0.0.1:9001/",
        oauth_audience="http://127.0.0.1:8080/mcp",
        oauth_jwks_url="http://127.0.0.1:9001/jwks", allowed_subject="test-user",
    )


@pytest.fixture
def authorized(monkeypatch):
    token = AccessToken(token="fake", client_id="test", scopes=["training:read", "training:write"])
    monkeypatch.setattr(bridge, "get_access_token", lambda: token)


def call(server, name, arguments):
    return asyncio.run(server._tool_manager.call_tool(name, arguments))


def test_server_refuses_missing_config(monkeypatch):
    for name in (
        "CF_API_URL", "CF_API_TOKEN", "MCP_PUBLIC_URL", "OAUTH_ISSUER",
        "OAUTH_AUDIENCE", "OAUTH_JWKS_URL", "MCP_ALLOWED_SUBJECT",
    ):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ValueError, match="CF_API_TOKEN"):
        Config.from_environment()


def test_oauth_rejects_wrong_subject_audience_and_expiry(config):
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = JwtVerifier(config)
    verifier.jwks = SimpleNamespace(get_signing_key_from_jwt=lambda _: SimpleNamespace(key=key.public_key()))

    def token(**overrides):
        claims = {
            "iss": config.oauth_issuer, "aud": config.oauth_audience,
            "sub": config.allowed_subject, "iat": datetime.now(timezone.utc),
            "exp": datetime.now(timezone.utc) + timedelta(minutes=5),
            "scope": "training:read training:write",
        }
        claims.update(overrides)
        return jwt.encode(claims, key, algorithm="RS256")

    assert asyncio.run(verifier.verify_token(token())).subject == "test-user"
    assert asyncio.run(verifier.verify_token(token(sub="someone-else"))) is None
    assert asyncio.run(verifier.verify_token(token(aud="wrong-resource"))) is None
    assert asyncio.run(verifier.verify_token(token(exp=datetime.now(timezone.utc) - timedelta(minutes=1)))) is None


def test_no_anonymous_mcp_tools(config):
    async def check():
        app = bridge.create_server(config).streamable_http_app()
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),
                                     base_url="http://127.0.0.1:8080") as client:
            return await client.post(
                "/mcp", headers={"Accept": "application/json, text/event-stream"},
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
    response = asyncio.run(check())
    assert response.status_code == 401
    assert "resource_metadata" in response.headers.get("www-authenticate", "")


def test_authenticated_mcp_handshake_lists_tools(config):
    class LocalVerifier:
        async def verify_token(self, token):
            if token != "local-test-token":
                return None
            return AccessToken(token=token, client_id="test", subject="test-user",
                               resource=config.oauth_audience,
                               scopes=["training:read", "training:write"])

    async def check():
        app = bridge.create_server(config, verifier=LocalVerifier()).streamable_http_app()
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8080",
                headers={"Authorization": "Bearer local-test-token",
                         "Accept": "application/json, text/event-stream"},
            ) as client:
                init = await client.post("/mcp", json={
                    "jsonrpc": "2.0", "id": 1, "method": "initialize",
                    "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                               "clientInfo": {"name": "offline-test", "version": "1"}},
                })
                listed = await client.post("/mcp", json={
                    "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {},
                })
                return init, listed

    init, listed = asyncio.run(check())
    assert init.status_code == 200
    assert listed.status_code == 200
    tools = listed.json()["result"]["tools"]
    assert {tool["name"] for tool in tools} >= {"find_workouts", "log_metcon", "log_workout_sets"}
    assert all(tool["securitySchemes"][0]["type"] == "oauth2" for tool in tools)


def test_read_last_uses_actual_latest_route(config, authorized):
    calls = []

    def handler(request):
        calls.append((request.url.path, request.url.params.get("exercise")))
        assert request.headers["authorization"] == "Bearer local-test-secret"
        return httpx.Response(200, json={"id": 55, "date": "2026-09-24", "exercise": "Squat"})

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    result = call(server, "get_last_workout", {"exercise": "Squat"})
    assert result["id"] == 55
    assert calls == [("/workouts/last", "Squat")]


def test_bulk_verifies_new_ids_even_when_existing_row_makes_count_match_false(config, authorized):
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["authorization"] == "Bearer local-test-secret"
        if request.method == "POST":
            assert request.url.path == "/workouts/bulk"
            assert "force" not in request.url.params
            return httpx.Response(200, json={"saved": 2, "ids": [91, 92]})
        return httpx.Response(200, json={"match": False, "logged": [
            {"id": 12, "date": "2026-09-24", "exercise": "Squat", "set_number": 1},
            {"id": 91, "date": "2026-09-24", "exercise": "Squat", "set_number": 2, "reps": 5},
            {"id": 92, "date": "2026-09-24", "exercise": "Squat", "set_number": 3, "reps": 5},
        ]})

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    result = call(server, "log_workout_sets", {"workouts": [
        {"date": "2026-09-24", "exercise": "Squat", "set_number": 2, "reps": 5},
        {"date": "2026-09-24", "exercise": "Squat", "set_number": 3, "reps": 5},
    ]})
    assert result["verified"] is True
    assert result["ids"] == [91, 92]
    assert len(requests) == 2


def test_metcon_duplicate_never_posts(config, authorized):
    calls = []

    def handler(request):
        calls.append(request.method)
        return httpx.Response(200, json=[
            {"id": 44, "date": "2026-09-24", "name": "Fran", "workout_type": "for_time",
             "score_time_seconds": 210}
        ])

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    with pytest.raises(ToolError, match="matching metcon already exists"):
        call(server, "log_metcon", {"metcon": {
            "date": "2026-09-24", "name": "Fran", "workout_type": "for_time", "score_time_seconds": 210,
        }})
    assert calls == ["GET"]


def test_metcon_readback_reports_unverified_when_new_id_is_absent(config, authorized):
    calls = []

    def handler(request):
        calls.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json=[])
        return httpx.Response(200, json={"id": 101})

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    result = call(server, "log_metcon", {"metcon": {
        "date": "2026-09-24", "name": "Intervals", "workout_type": "interval",
        "score_display": "5 rounds completed",
    }})
    assert result["saved"] is True
    assert result["verified"] is False
    assert result["id"] == 101
    assert calls == ["GET", "POST", "GET"]


def test_bulk_refuses_repeated_sets_within_one_call(config, authorized):
    def handler(request):
        pytest.fail("No HTTP call should occur for duplicate set numbers")

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    with pytest.raises(ToolError, match="Repeated date/exercise/set number"):
        call(server, "log_workout_sets", {"workouts": [
            {"date": "2026-09-24", "exercise": "Squat", "set_number": 2},
            {"date": "2026-09-24", "exercise": "squat", "set_number": 2},
        ]})


def test_lost_write_response_does_not_retry(config, authorized):
    attempts = []

    def handler(request):
        attempts.append(request.method)
        raise httpx.ReadTimeout("lost response", request=request)

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    with pytest.raises(ToolError, match="before any retry"):
        call(server, "log_workout_sets", {"workouts": [
            {"date": "2026-09-24", "exercise": "Squat", "set_number": 1},
        ]})
    assert attempts == ["POST"]


def test_backend_error_after_write_is_uncertain_and_never_retried(config, authorized):
    attempts = []

    def handler(request):
        attempts.append(request.method)
        return httpx.Response(503, json={"detail": "internal details should stay private"})

    server = bridge.create_server(config, backend=Backend(config, httpx.MockTransport(handler)))
    with pytest.raises(ToolError, match="inspect records before retrying") as error:
        call(server, "log_workout_sets", {"workouts": [
            {"date": "2026-09-24", "exercise": "Squat", "set_number": 1},
        ]})
    assert "internal details" not in str(error.value)
    assert attempts == ["POST"]


def test_no_edit_delete_or_force_tools(config):
    server = bridge.create_server(config)
    listed = asyncio.run(server.list_tools())
    names = {tool.name for tool in listed}
    assert not any(word in name for name in names for word in ("delete", "undo", "edit", "force"))
    assert all(tool.model_dump(by_alias=True)["securitySchemes"] for tool in listed)
