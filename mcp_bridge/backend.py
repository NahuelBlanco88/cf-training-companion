"""Fixed-route client for the existing CF-Log REST API; never executes SQL."""

from collections.abc import Mapping
from typing import Any

import httpx

from .config import Config


class BackendError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        super().__init__(f"Backend returned HTTP {status}: {detail[:300]}")


class OutcomeUnknown(Exception):
    """A write may have committed; the caller must read before retrying."""


class Backend:
    def __init__(self, config: Config, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self.transport = transport

    async def request(
        self, method: str, path: str, *, params: Mapping[str, Any] | None = None,
        body: dict[str, Any] | None = None,
    ) -> Any:
        # Callers pass only constant paths. No generic URL/SQL tool is exposed.
        assert path.startswith("/") and ".." not in path
        try:
            async with httpx.AsyncClient(
                base_url=self.config.api_url.rstrip("/"),
                headers={"Authorization": f"Bearer {self.config.api_token}"},
                timeout=httpx.Timeout(8.0), follow_redirects=False,
                trust_env=False, transport=self.transport,
            ) as client:
                response = await client.request(method, path, params=params, json=body)
        except httpx.RequestError as exc:
            if method != "GET":
                raise OutcomeUnknown("The write response was lost; inspect existing records before any retry") from exc
            raise BackendError(0, "The backend could not be reached") from exc
        if response.status_code < 200 or response.status_code >= 300:
            if method != "GET" and (response.status_code < 400 or response.status_code >= 500):
                raise OutcomeUnknown("Write result is uncertain; inspect records before retrying")
            detail = "Request rejected" if response.status_code < 500 else "Backend unavailable"
            if response.status_code == 409:
                try:
                    detail = str(response.json().get("detail", "Possible duplicate"))
                except (ValueError, AttributeError):
                    detail = "Possible duplicate"
            raise BackendError(response.status_code, detail)
        try:
            return response.json()
        except ValueError as exc:
            if method != "GET":
                raise OutcomeUnknown("The backend did not return a valid write receipt; inspect records before retrying") from exc
            raise BackendError(0, "The backend returned invalid JSON") from exc

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, body: dict[str, Any]) -> Any:
        return await self.request("POST", path, body=body)
