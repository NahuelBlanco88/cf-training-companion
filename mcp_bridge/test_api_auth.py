"""Offline auth boundary checks; never import or start app.py."""

import pytest

from api_auth import AuthConfig, authorize, compatibility_credential_status


GPT = "g" * 48
MCP = "m" * 48
READER = "r" * 48
ENFORCED = AuthConfig("enforce", GPT, MCP, READER)


@pytest.mark.parametrize("method,path", [
    ("GET", "/workouts"), ("GET", "/metcons"), ("GET", "/stats"),
    ("GET", "/analytics/prs"), ("GET", "/export/csv"),
    ("GET", "/debug/dbinfo"), ("GET", "/openapi.json"),
    ("POST", "/workouts/verify"), ("POST", "/workouts"),
    ("POST", "/workouts/bulk"), ("POST", "/metcons"),
    ("POST", "/metcons/bulk"), ("PUT", "/workouts/123"),
    ("DELETE", "/workouts/undo_last"), ("DELETE", "/metcons/12"),
    ("PATCH", "/workouts/123"), ("PATCH", "/metcons/12"),
])
def test_data_routes_reject_missing_or_wrong_credentials(method, path):
    assert authorize(method, path, None, ENFORCED) == 401
    assert authorize(method, path, "Bearer wrong", ENFORCED) == 401
    assert authorize(method, path, GPT, ENFORCED) == 401


def test_roles_limit_writes_and_destructive_routes():
    assert authorize("GET", "/workouts", f"Bearer {READER}", ENFORCED) is None
    assert authorize("POST", "/workouts/verify", f"Bearer {READER}", ENFORCED) is None
    assert authorize("POST", "/workouts/bulk", f"Bearer {READER}", ENFORCED) == 403
    assert authorize("POST", "/workouts/bulk", f"Bearer {MCP}", ENFORCED) is None
    assert authorize("POST", "/metcons", f"Bearer {MCP}", ENFORCED) == 403
    assert authorize("POST", "/metcons", f"Bearer {MCP}", ENFORCED,
                     duplicate_guard=True) is None
    assert authorize("POST", "/workouts/bulk", f"Bearer {MCP}", ENFORCED,
                     frozenset({"force"})) == 403
    assert authorize("POST", "/workouts", f"Bearer {MCP}", ENFORCED) == 403
    assert authorize("POST", "/metcons/bulk", f"Bearer {MCP}", ENFORCED) == 403
    assert authorize("DELETE", "/workouts/undo_last", f"Bearer {MCP}", ENFORCED) == 403
    assert authorize("PUT", "/workouts/123", f"Bearer {MCP}", ENFORCED) == 403
    assert authorize("DELETE", "/metcons/123", f"Bearer {MCP}", ENFORCED) == 403
    assert authorize("PATCH", "/workouts/123", f"Bearer {MCP}", ENFORCED) is None
    assert authorize("PATCH", "/metcons/123", f"Bearer {MCP}", ENFORCED) is None
    assert authorize("PATCH", "/workouts/123", f"Bearer {READER}", ENFORCED) == 403
    assert authorize("PUT", "/workouts/123", f"Bearer {GPT}", ENFORCED) is None
    assert authorize("DELETE", "/workouts/undo_last", f"Bearer {GPT}", ENFORCED) is None
    assert authorize("POST", "/unknown", f"Bearer {GPT}", ENFORCED) == 403


def test_public_health_and_preflight_and_staged_compatibility():
    for method, path in (("GET", "/health"), ("GET", "/"), ("OPTIONS", "/workouts")):
        assert authorize(method, path, None, ENFORCED) is None
    assert authorize("POST", "/workouts", None, AuthConfig("compat")) is None
    assert authorize("PATCH", "/workouts/123", None, AuthConfig("compat")) == 403
    assert authorize("PATCH", "/metcons/123", f"Bearer {GPT}", AuthConfig("compat")) == 403


def test_compatibility_observes_gpt_key_without_enforcing_it(monkeypatch):
    monkeypatch.setenv("CF_API_AUTH_MODE", "compat")
    monkeypatch.setenv("CF_API_GPT_TOKEN", GPT)
    config = AuthConfig.from_environment()
    assert compatibility_credential_status(f"Bearer {GPT}", config) == "gpt"
    assert compatibility_credential_status(None, config) == "missing"
    assert compatibility_credential_status("Bearer wrong", config) == "unrecognized"
    assert compatibility_credential_status(f"Bearer {GPT}", AuthConfig("compat")) == "unrecognized"
    assert authorize("POST", "/workouts/bulk", None, config) is None
    assert authorize("POST", "/workouts/bulk", "Bearer wrong", config) is None


def test_enforced_mode_requires_distinct_secrets(monkeypatch):
    monkeypatch.setenv("CF_API_AUTH_MODE", "enforce")
    monkeypatch.setenv("CF_API_GPT_TOKEN", GPT)
    monkeypatch.setenv("CF_API_MCP_TOKEN", GPT)
    with pytest.raises(ValueError, match="distinct"):
        AuthConfig.from_environment()
    monkeypatch.setenv("CF_API_MCP_TOKEN", MCP)
    settings = AuthConfig.from_environment()
    assert settings.mode == "enforce"
    assert settings.gpt_token == GPT


def test_enforced_mode_fails_closed_when_missing_token(monkeypatch):
    monkeypatch.setenv("CF_API_AUTH_MODE", "enforce")
    monkeypatch.setenv("CF_API_GPT_TOKEN", GPT)
    monkeypatch.delenv("CF_API_MCP_TOKEN", raising=False)
    with pytest.raises(ValueError, match="at least 32"):
        AuthConfig.from_environment()
