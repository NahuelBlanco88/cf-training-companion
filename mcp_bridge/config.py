"""Configuration fails closed when authentication is missing."""

from dataclasses import dataclass
from os import environ
from urllib.parse import urlparse


@dataclass(frozen=True)
class Config:
    api_url: str
    api_token: str
    public_url: str
    oauth_issuer: str
    oauth_audience: str
    oauth_jwks_url: str
    allowed_subject: str

    def __post_init__(self) -> None:
        for field in ("api_token", "allowed_subject"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must be configured")
        for field in ("api_url", "public_url", "oauth_issuer", "oauth_audience", "oauth_jwks_url"):
            url = urlparse(getattr(self, field))
            if not url.hostname or url.username or url.password:
                raise ValueError(f"{field} must be an absolute URL without credentials")
            if url.scheme != "https" and not (url.scheme == "http" and url.hostname in ("localhost", "127.0.0.1")):
                raise ValueError(f"{field} must use HTTPS (or localhost for local tests)")
        if self.public_url.rstrip("/") != self.oauth_audience.rstrip("/"):
            raise ValueError("oauth_audience must equal the MCP public URL")
        if urlparse(self.public_url).path != "/mcp":
            raise ValueError("MCP_PUBLIC_URL must end in /mcp")

    @classmethod
    def from_environment(cls) -> "Config":
        keys = {
            "api_url": "CF_API_URL", "api_token": "CF_API_TOKEN",
            "public_url": "MCP_PUBLIC_URL", "oauth_issuer": "OAUTH_ISSUER",
            "oauth_audience": "OAUTH_AUDIENCE", "oauth_jwks_url": "OAUTH_JWKS_URL",
            "allowed_subject": "MCP_ALLOWED_SUBJECT",
        }
        missing = [value for value in keys.values() if not environ.get(value)]
        if missing:
            raise ValueError("Missing configuration: " + ", ".join(missing))
        return cls(**{key: environ[value] for key, value in keys.items()})
