"""Focused, authenticated MCP tools for the existing training-log API.

This module never imports app.py: its startup can alter a database schema.
"""

from collections import defaultdict
from datetime import date
import logging
from os import environ
from typing import Any, Literal
from urllib.parse import urlparse

from mcp.server.auth.middleware.auth_context import get_access_token
from mcp.server.auth.settings import AuthSettings
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from mcp.types import ToolAnnotations
from pydantic import AnyHttpUrl, Field

from .auth import JwtVerifier
from .backend import Backend, BackendError
from .config import Config
from .models import MetconChanges, MetconInput, WorkoutChanges, WorkoutInput


READ = ToolAnnotations(readOnlyHint=True, destructiveHint=False, openWorldHint=False)
WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False, openWorldHint=False)
CORRECT = ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False)
TOOL_SCOPES = {
    "log_workout_sets": "training:write", "log_metcon": "training:write",
    "correct_workout_set": "training:edit", "correct_metcon": "training:edit",
}
METRICS = {
    "prs": ("/analytics/prs", "exercise"),
    "estimated_1rm": ("/analytics/estimated_1rm", "exercise"),
    "volume": ("/analytics/volume", None),
    "timeline": ("/analytics/timeline", "exercise"),
    "metcon_prs": ("/analytics/metcon_prs", None),
    "metcon_timeline": ("/analytics/metcon_timeline", "name"),
    "movement_frequency": ("/analytics/movement_frequency", None),
    "intensity_distribution": ("/analytics/intensity_distribution", "exercise"),
    "week_over_week": ("/analytics/week_over_week", "exercise"),
    "training_density": ("/analytics/training_density", None),
    "trend": ("/analytics/trend", "exercise"),
}


def _access(scope: str) -> None:
    token = get_access_token()
    if token is None or scope not in token.scopes:
        raise PermissionError(f"An authorized account with {scope} is required")


def _filters(**values: Any) -> dict[str, Any]:
    return {key: value.isoformat() if isinstance(value, date) else value
            for key, value in values.items() if value is not None}


def _date_range(start: date | None, end: date | None) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError("start must be on or before end")


def _confirmed_fields(saved: dict[str, Any], provided: dict[str, Any], *, title_field: str) -> bool:
    for key, value in provided.items():
        actual = saved.get(key)
        if key == title_field and isinstance(actual, str) and isinstance(value, str):
            if actual.casefold() != value.casefold():
                return False
        elif actual != value:
            return False
    return True


class AuthenticatedFastMCP(FastMCP):
    async def list_tools(self):
        # SDK 1.x accepts this standard MCP field as an extra on Tool.
        tools = await super().list_tools()
        for tool in tools:
            schemes = [{"type": "oauth2", "scopes": [TOOL_SCOPES.get(tool.name, "training:read")]}]
            tool.securitySchemes = schemes
            tool.meta = {**(tool.meta or {}), "securitySchemes": schemes}
        return tools


def create_server(config: Config, *, backend: Backend | None = None, verifier: Any = None) -> FastMCP:
    """Create an OAuth-protected server; no live API calls occur at construction."""
    backend = backend or Backend(config)
    public = urlparse(config.public_url)
    hostname = public.hostname
    assert hostname is not None
    server = AuthenticatedFastMCP(
        name="CF Training Companion",
        instructions=(
            "Use reads to resolve dates and record IDs before logging or correcting. "
            "One performed working set is one row. Never infer missing training fields. "
            "If a write is uncertain, inspect records and never retry blindly."
        ),
        token_verifier=verifier or JwtVerifier(config),
        auth=AuthSettings(
            issuer_url=AnyHttpUrl(config.oauth_issuer),
            resource_server_url=AnyHttpUrl(config.public_url),
            required_scopes=[],  # Each tool checks its own declared scope.
            validate_token_resource=True,
        ),
        host="0.0.0.0", port=int(environ.get("PORT", "8080")),
        stateless_http=True, json_response=True,
        transport_security=TransportSecuritySettings(
            allowed_hosts=[hostname, public.netloc, f"{hostname}:443", "localhost", "127.0.0.1"],
        ),
    )

    @server.tool(title="Find workout sets", annotations=READ)
    async def find_workouts(
        exercise: str | None = None, start: date | None = None, end: date | None = None,
        cycle: int | None = None, week: int | None = None, day: int | None = None,
        limit: int = Field(default=200, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        """Read logged strength sets; results are oldest first and limited to 500 rows."""
        _access("training:read")
        _date_range(start, end)
        return await backend.get("/workouts", _filters(exercise=exercise, start=start, end=end,
                                                       cycle=cycle, week=week, day=day, limit=limit))

    @server.tool(title="Find metcons", annotations=READ)
    async def find_metcons(
        name: str | None = None, start: date | None = None, end: date | None = None,
        cycle: int | None = None, week: int | None = None, day: int | None = None,
        limit: int = Field(default=200, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        """Read logged conditioning results; results are oldest first and limited to 500 rows."""
        _access("training:read")
        _date_range(start, end)
        return await backend.get("/metcons", _filters(name=name, start=start, end=end,
                                                      cycle=cycle, week=week, day=day, limit=limit))

    @server.tool(title="Get a training day", annotations=READ)
    async def get_training_day(cycle: int, week: int, day: int) -> dict[str, Any]:
        """Get both workout sets and metcons for one cycle, week and day."""
        _access("training:read")
        if min(cycle, week, day) < 0:
            raise ValueError("Cycle, week and day must be nonnegative")
        return await backend.get("/day_summary", {"cycle": cycle, "week": week, "day": day})

    @server.tool(title="Get last workout for exercise", annotations=READ)
    async def get_last_workout(exercise: str) -> dict[str, Any]:
        """Retrieve the true latest set by date and ID. Supports partial exercise name."""
        _access("training:read")
        return await backend.get("/workouts/last", {"exercise": exercise})

    @server.tool(title="Get last metcon by name", annotations=READ)
    async def get_last_metcon(name: str) -> dict[str, Any]:
        """Retrieve the latest matching metcon by date and ID."""
        _access("training:read")
        return await backend.get("/metcons/last", {"name": name})

    @server.tool(title="Inspect a workout set by ID", annotations=READ)
    async def get_workout_by_id(workout_id: int = Field(ge=1)) -> dict[str, Any]:
        """Read one set and its version. Provide that version for an intentional correction."""
        _access("training:read")
        return await backend.get(f"/workouts/{workout_id}")

    @server.tool(title="Inspect a metcon by ID", annotations=READ)
    async def get_metcon_by_id(metcon_id: int = Field(ge=1)) -> dict[str, Any]:
        """Read one metcon and its version before an intentional correction."""
        _access("training:read")
        return await backend.get(f"/metcons/{metcon_id}")

    @server.tool(title="Get training statistics", annotations=READ)
    async def get_training_stats() -> dict[str, Any]:
        """Read database-derived counts and simple summary statistics."""
        _access("training:read")
        return await backend.get("/stats")

    @server.tool(title="Analyze training", annotations=READ)
    async def analyze_training(
        metric: Literal[
            "prs", "estimated_1rm", "volume", "timeline", "metcon_prs",
            "metcon_timeline", "movement_frequency", "intensity_distribution",
            "week_over_week", "training_density", "trend",
        ],
        exercise: str | None = None, name: str | None = None,
        cycle: int | None = None, week: int | None = None,
        start: date | None = None, end: date | None = None,
    ) -> dict[str, Any]:
        """Use an existing analytics endpoint; never derive PRs by comparing unlike units."""
        _access("training:read")
        _date_range(start, end)
        path, required = METRICS[metric]
        if required and not (exercise if required == "exercise" else name):
            raise ValueError(f"{required} is required for {metric}")
        params = _filters(exercise=exercise, name=name, cycle=cycle, week=week,
                          start=start, end=end)
        allowed = {
            "prs": {"exercise"}, "estimated_1rm": {"exercise"},
            "volume": {"cycle", "week", "start", "end"},
            "timeline": {"exercise", "start", "end"},
            "metcon_prs": {"name"},
            "metcon_timeline": {"name", "start", "end"},
            "movement_frequency": {"cycle", "week", "start", "end"},
            "intensity_distribution": {"exercise", "cycle", "week", "start", "end"},
            "week_over_week": {"exercise", "cycle"},
            "training_density": {"cycle", "week", "start", "end"},
            "trend": {"exercise", "start", "end"},
        }[metric]
        if set(params) - allowed:
            raise ValueError(f"Unsupported filters for {metric}: {sorted(set(params) - allowed)}")
        return await backend.get(path, params)

    @server.tool(title="Log working sets", annotations=WRITE)
    async def log_workout_sets(workouts: list[WorkoutInput]) -> dict[str, Any]:
        """Insert performed working sets once, then check returned IDs via a separate read. Never auto-retry a write."""
        _access("training:write")
        if not 1 <= len(workouts) <= 50:
            raise ValueError("Provide 1 to 50 performed working sets")
        seen = set()
        for workout in workouts:
            key = (workout.date, workout.exercise.casefold(), workout.set_number,
                   workout.cycle, workout.week, workout.day)
            if key in seen:
                raise ValueError("Repeated date/exercise/set number in one request")
            seen.add(key)
        rows = [workout.model_dump(mode="json", exclude_none=True) for workout in workouts]
        receipt = await backend.post("/workouts/bulk", {"workouts": rows})
        ids = receipt.get("ids") if isinstance(receipt, dict) else None
        valid_receipt = (
            isinstance(receipt, dict)
            and type(receipt.get("saved")) is int
            and receipt["saved"] == len(rows)
            and isinstance(ids, list)
            and len(ids) == len(rows)
            and all(type(row_id) is int and row_id > 0 for row_id in ids)
            and len(set(ids)) == len(ids)
        )
        if not valid_receipt:
            return {"saved": True, "verified": False, "ids": ids,
                    "warning": "Unexpected write receipt. Inspect stored records before any retry."}
        groups: dict[tuple[Any, ...], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
        for row_id, row in zip(ids, rows):
            key = (row["date"], row["exercise"], row.get("cycle"), row.get("week"), row.get("day"))
            groups[key].append((row_id, row))
        verified_ids: set[int] = set()
        try:
            for (when, exercise, cycle, week, day), expected in groups.items():
                result = await backend.get("/workouts/verify", _filters(
                    date=when, exercise=exercise, cycle=cycle, week=week, day=day))
                if not isinstance(result, dict) or not isinstance(result.get("logged"), list):
                    raise ValueError("Invalid verification response")
                logged = {item["id"]: item for item in result["logged"]
                          if isinstance(item, dict) and type(item.get("id")) is int}
                for row_id, row in expected:
                    if row_id in logged and _confirmed_fields(logged[row_id], row, title_field="exercise"):
                        verified_ids.add(row_id)
        except (BackendError, ValueError, TypeError, KeyError, AttributeError):
            pass  # The POST receipt was successful; do not hide its IDs or retry.
        complete = len(verified_ids) == len(ids)
        return {"saved": True, "ids": ids, "verified": complete,
                "verified_ids": sorted(verified_ids),
                "warning": None if complete else "Some IDs could not be read back. Do not retry automatically."}

    @server.tool(title="Log a metcon", annotations=WRITE)
    async def log_metcon(metcon: MetconInput) -> dict[str, Any]:
        """Log one performed metcon; check for prior matching results and read back the saved ID."""
        _access("training:write")
        row = metcon.model_dump(mode="json", exclude_none=True)
        query = {"name": row["name"], "start": row["date"], "end": row["date"], "limit": 500}
        previous = await backend.get("/metcons", query)
        if not isinstance(previous, list) or not all(isinstance(item, dict) for item in previous):
            raise ValueError("Invalid duplicate-check response; write refused")
        if len(previous) >= 500:
            raise ValueError("Too many matching results for a safe duplicate check; write refused")
        if any(_confirmed_fields(item, row, title_field="name") for item in previous):
            raise ValueError("A matching metcon already exists. Inspect existing records before writing")
        receipt = await backend.post("/metcons", row, headers={"X-CF-Prevent-Duplicate": "true"})
        row_id = receipt.get("id") if isinstance(receipt, dict) else None
        if type(row_id) is not int or row_id <= 0:
            return {"saved": True, "verified": False,
                    "warning": "Unexpected write receipt. Inspect records before any retry."}
        try:
            matches = await backend.get("/metcons", query)
            if not isinstance(matches, list):
                raise ValueError("Invalid read-back response")
            confirmed = next((item for item in matches if item.get("id") == row_id), None)
            verified = confirmed is not None and _confirmed_fields(confirmed, row, title_field="name")
        except (BackendError, ValueError, TypeError, AttributeError):
            verified = False
        return {"saved": True, "id": row_id, "verified": verified,
                "warning": None if verified else "Saved ID could not be read back. Do not retry automatically."}

    @server.tool(title="Correct one workout set", annotations=CORRECT)
    async def correct_workout_set(
        workout_id: int = Field(ge=1), expected_version: str = Field(pattern=r"^[0-9a-f]{64}$"),
        changes: WorkoutChanges = Field(),
    ) -> dict[str, Any]:
        """Correct only supplied fields after the user identifies the set. Read by ID first; stale versions fail."""
        _access("training:edit")
        patch = changes.model_dump(mode="json", exclude_unset=True)
        result = await backend.patch(f"/workouts/{workout_id}",
                                     {"expected_version": expected_version, "changes": patch})
        if not isinstance(result, dict) or not isinstance(result.get("record"), dict):
            return {"saved": True, "verified": False,
                    "warning": "Unexpected correction receipt. Inspect the ID before retrying."}
        try:
            latest = await backend.get(f"/workouts/{workout_id}")
            verified = (isinstance(latest, dict) and latest.get("version") == result.get("version")
                        and latest.get("record", {}).get("id") == workout_id
                        and _confirmed_fields(latest["record"], patch, title_field="exercise"))
        except (BackendError, ValueError, TypeError, AttributeError):
            verified = False
        return {"saved": True, "verified": verified, "record": result["record"],
                "version": result.get("version"),
                "warning": None if verified else "Correction could not be read back. Do not retry automatically."}

    @server.tool(title="Correct one metcon", annotations=CORRECT)
    async def correct_metcon(
        metcon_id: int = Field(ge=1), expected_version: str = Field(pattern=r"^[0-9a-f]{64}$"),
        changes: MetconChanges = Field(),
    ) -> dict[str, Any]:
        """Correct only supplied fields after the user identifies the metcon. Stale versions fail."""
        _access("training:edit")
        patch = changes.model_dump(mode="json", exclude_unset=True)
        result = await backend.patch(f"/metcons/{metcon_id}",
                                     {"expected_version": expected_version, "changes": patch})
        if not isinstance(result, dict) or not isinstance(result.get("record"), dict):
            return {"saved": True, "verified": False,
                    "warning": "Unexpected correction receipt. Inspect the ID before retrying."}
        try:
            latest = await backend.get(f"/metcons/{metcon_id}")
            verified = (isinstance(latest, dict) and latest.get("version") == result.get("version")
                        and latest.get("record", {}).get("id") == metcon_id
                        and _confirmed_fields(latest["record"], patch, title_field="name"))
        except (BackendError, ValueError, TypeError, AttributeError):
            verified = False
        return {"saved": True, "verified": verified, "record": result["record"],
                "version": result.get("version"),
                "warning": None if verified else "Correction could not be read back. Do not retry automatically."}

    return server


def main() -> None:
    # httpx INFO logs full URLs, including workout search terms and dates.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    create_server(Config.from_environment()).run(transport="streamable-http")


if __name__ == "__main__":
    main()
