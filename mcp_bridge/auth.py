"""Verify OAuth access tokens from a separately managed identity provider."""

import asyncio

import jwt
from mcp.server.auth.provider import AccessToken

from .config import Config


class JwtVerifier:
    def __init__(self, config: Config):
        self.config = config
        self.jwks = jwt.PyJWKClient(config.oauth_jwks_url)

    async def verify_token(self, token: str) -> AccessToken | None:
        try:
            signing_key = await asyncio.to_thread(self.jwks.get_signing_key_from_jwt, token)
            claims = jwt.decode(
                token, signing_key.key, algorithms=["RS256"],
                audience=self.config.oauth_audience,
                issuer=self.config.oauth_issuer,
                options={"require": ["exp", "iat", "sub", "aud", "iss"]},
            )
        except (jwt.PyJWTError, ValueError, OSError):
            return None
        if claims["sub"] != self.config.allowed_subject:
            return None
        raw_scopes = claims.get("scope", "")
        if not isinstance(raw_scopes, str):
            return None
        return AccessToken(
            token=token, client_id=str(claims.get("azp", claims["sub"])),
            scopes=raw_scopes.split(), expires_at=claims["exp"],
            resource=self.config.oauth_audience, subject=claims["sub"],
        )
