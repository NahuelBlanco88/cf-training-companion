"""Small, explicit credential gate for the existing CF-Log API.

This module never imports the database application. Its default compatibility
mode is for a staged GPT Action cutover only; it must not be considered secure.
"""

from dataclasses import dataclass
import hmac
import os


@dataclass(frozen=True)
class AuthConfig:
    mode: str
    gpt_token: str = ""
    mcp_token: str = ""
    read_token: str = ""

    @classmethod
    def from_environment(cls) -> "AuthConfig":
        mode = os.getenv("CF_API_AUTH_MODE", "compat")
        if mode not in ("compat", "enforce"):
            raise ValueError("CF_API_AUTH_MODE must be compat or enforce")
        if mode == "compat":
            # Existing GPT Action currently sends no credential. Explicitly
            # require a later, reviewed switch to enforce after GPT cutover.
            return cls(mode=mode)
        tokens = (
            os.getenv("CF_API_GPT_TOKEN", ""),
            os.getenv("CF_API_MCP_TOKEN", ""),
            os.getenv("CF_API_READ_TOKEN", ""),
        )
        if any(len(value) < 32 or value.strip() != value for value in tokens[:2]):
            raise ValueError("Enforced API auth needs distinct GPT and MCP tokens of at least 32 characters")
        if tokens[2] and (len(tokens[2]) < 32 or tokens[2].strip() != tokens[2]):
            raise ValueError("Read token must have at least 32 characters")
        active = [value for value in tokens if value]
        if len(set(active)) != len(active):
            raise ValueError("API tokens must be distinct")
        return cls(mode=mode, gpt_token=tokens[0], mcp_token=tokens[1], read_token=tokens[2])


def required_capability(method: str, path: str) -> str | None:
    """Public health and CORS, read-only GET, and explicit permitted writes."""
    if method == "OPTIONS" or (method == "GET" and path in ("/", "/health")):
        return None
    if method == "GET" or (method == "HEAD" and path in ("/docs", "/openapi.json")):
        return "read"
    if method == "POST" and path == "/workouts/verify":
        return "read"
    if method == "POST" and path in (
        "/workouts", "/workouts/bulk", "/metcons", "/metcons/bulk",
    ):
        return "create"
    if method == "PATCH" and (
        (path.startswith("/workouts/") and path[len("/workouts/"):].isdigit())
        or (path.startswith("/metcons/") and path[len("/metcons/"):].isdigit())
    ):
        return "correct"
    if method in ("PUT", "DELETE") and (
        path == "/workouts/undo_last"
        or (path.startswith("/workouts/") and path[len("/workouts/"):].isdigit())
        or (path.startswith("/metcons/") and path[len("/metcons/"):].isdigit())
    ):
        return "edit"
    return "deny"


def authorize(method: str, path: str, authorization: str | None,
              config: AuthConfig, query_keys: frozenset[str] = frozenset(),
              duplicate_guard: bool = False) -> int | None:
    """Return an HTTP error status or None for allowed requests."""
    if config.mode == "compat":
        # New conditional edits are only for authenticated MCP clients.
        # Staging must not expose an additional public edit route.
        if method == "PATCH":
            return 403
        return None
    capability = required_capability(method, path)
    if capability is None:
        return None
    if not authorization or not authorization.startswith("Bearer "):
        return 401
    token = authorization[len("Bearer "):]
    if not token or token.strip() != token:
        return 401
    if hmac.compare_digest(token, config.gpt_token):
        role = "gpt"
    elif hmac.compare_digest(token, config.mcp_token):
        role = "mcp"
    elif config.read_token and hmac.compare_digest(token, config.read_token):
        role = "reader"
    else:
        return 401
    if capability == "deny":
        return 403
    if role == "mcp" and capability == "create":
        # A leaked bridge token must not be able to turn off duplicate checks
        # or use legacy bulk metcon insertion without a duplicate guard.
        if (path not in ("/workouts/bulk", "/metcons") or "force" in query_keys
                or (path == "/metcons" and not duplicate_guard)):
            return 403
    if role == "gpt" or capability == "read" or (role == "mcp" and capability in ("create", "correct")):
        return None
    return 403
