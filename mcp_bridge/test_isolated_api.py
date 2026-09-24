"""Exercise production route wiring only against a fresh temporary SQLite file.

Never import the repository's destructive test fixture or enter app lifespan.
"""

import asyncio
import importlib.util
import sys
from pathlib import Path

import httpx
from sqlalchemy import event


GPT = "g" * 48
MCP = "m" * 48


def test_authenticated_partial_corrections_and_metcon_duplicates(tmp_path, monkeypatch):
    database = tmp_path / "isolated.db"
    database.touch()  # app.py selects only a filesystem path that already exists.
    monkeypatch.delenv("CLOUD_SQL_CONNECTION_NAME", raising=False)
    monkeypatch.setenv("CFLOG_DB", str(database))
    monkeypatch.setenv("CF_API_AUTH_MODE", "enforce")
    monkeypatch.setenv("CF_API_SCHEMA_INIT_MODE", "verify")
    monkeypatch.setenv("CF_API_GPT_TOKEN", GPT)
    monkeypatch.setenv("CF_API_MCP_TOKEN", MCP)
    monkeypatch.delenv("CF_API_READ_TOKEN", raising=False)

    # Unique import name keeps this isolated instance out of the rest of pytest.
    spec = importlib.util.spec_from_file_location(
        "isolated_cflog_app", Path(__file__).resolve().parent.parent / "app.py",
    )
    app_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, app_module)
    spec.loader.exec_module(app_module)

    async def exercise():
        # ASGITransport does not enter FastAPI lifespan. Schema creation and
        # every data write below apply only to the new tmp_path SQLite file.
        async with app_module.engine.begin() as connection:
            await connection.run_sync(app_module.Base.metadata.create_all)
        sql = []
        event.listen(app_module.engine.sync_engine, "before_cursor_execute",
                     lambda _conn, _cursor, statement, _params, _context, _many: sql.append(statement))
        async with app_module.lifespan(app_module.app):
            assert not any(word in statement.upper() for statement in sql
                           for word in ("CREATE ", "ALTER ", "DROP ", "INSERT ", "DELETE ", "UPDATE "))
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app_module.app, raise_app_exceptions=False),
            base_url="http://test.local",
        ) as client:
            gpt = {"Authorization": f"Bearer {GPT}"}
            mcp = {"Authorization": f"Bearer {MCP}"}
            assert (await client.get("/workouts")).status_code == 401
            assert (await client.post("/workouts/bulk?force=true", headers=mcp,
                                      json={"workouts": []})).status_code == 403
            workout = {"date": "2026-09-24", "exercise": "Squat", "set_number": 1,
                       "reps": 5, "value": 75, "unit": "kg", "notes": "keep this"}
            duplicate_batch = await client.post("/workouts/bulk", headers=mcp,
                json={"workouts": [{**workout, "date": "2026-09-25"},
                                   {**workout, "date": "2026-09-25"}]})
            assert duplicate_batch.status_code == 409
            assert (await client.get("/workouts?start=2026-09-25&end=2026-09-25",
                                     headers=mcp)).json() == []
            created = await client.post("/workouts/bulk", headers=mcp,
                                        json={"workouts": [workout]})
            assert created.status_code == 200, created.text
            workout_id = created.json()["ids"][0]
            original = await client.get(f"/workouts/{workout_id}", headers=mcp)
            assert original.status_code == 200, original.text
            version = original.json()["version"]
            corrected = await client.patch(f"/workouts/{workout_id}", headers=mcp,
                json={"expected_version": version, "changes": {"value": 77.5}})
            assert corrected.status_code == 200, corrected.text
            assert corrected.json()["record"]["value"] == 77.5
            assert corrected.json()["record"]["notes"] == "keep this"
            assert corrected.json()["record"]["reps"] == 5
            assert corrected.json()["record"]["reps_raw"] == "5"
            stale = await client.patch(f"/workouts/{workout_id}", headers=mcp,
                json={"expected_version": version, "changes": {"reps": 3}})
            assert stale.status_code == 412
            assert (await client.put(f"/workouts/{workout_id}", headers=mcp,
                                     json=workout)).status_code == 403
            assert (await client.delete(f"/workouts/{workout_id}", headers=mcp)).status_code == 403
            assert (await client.get(f"/workouts/{workout_id}", headers=mcp)).json()["record"]["reps"] == 5

            # The live table stores reps as VARCHAR; legacy rows can contain
            # expressions, which must remain visible and unchanged on a patch.
            async with app_module.async_session() as session:
                legacy = app_module.Workout(date="2026-09-23", exercise="Legacy Press",
                                            reps="1+1", value=50.0, unit="kg")
                session.add(legacy)
                await session.commit()
                legacy_id = legacy.id
            legacy_read = await client.get(f"/workouts/{legacy_id}", headers=mcp)
            assert legacy_read.status_code == 200, legacy_read.text
            assert legacy_read.json()["record"]["reps"] == 2
            assert legacy_read.json()["record"]["reps_raw"] == "1+1"
            legacy_change = await client.patch(f"/workouts/{legacy_id}", headers=mcp,
                json={"expected_version": legacy_read.json()["version"],
                      "changes": {"notes": "checked"}})
            assert legacy_change.status_code == 200, legacy_change.text
            assert legacy_change.json()["record"]["reps_raw"] == "1+1"
            estimated = await client.get("/analytics/estimated_1rm?exercise=Legacy%20Press", headers=mcp)
            assert estimated.status_code == 200, estimated.text
            assert estimated.json()["based_on_reps"] == 2
            intensity = await client.get("/analytics/intensity_distribution?exercise=Legacy%20Press",
                                         headers=mcp)
            assert intensity.status_code == 200, intensity.text
            assert intensity.json()["total_sets"] == 1

            metcon = {"date": "2026-09-24", "name": "Fran", "workout_type": "for_time",
                      "score_time_seconds": 210}
            assert (await client.post("/metcons", headers=mcp, json=metcon)).status_code == 403
            guard = {**mcp, "X-CF-Prevent-Duplicate": "true"}
            first = await client.post("/metcons", headers=guard, json=metcon)
            assert first.status_code == 200, first.text
            assert (await client.post("/metcons", headers=guard, json=metcon)).status_code == 409
            metcon_id = first.json()["id"]
            previous = await client.get(f"/metcons/{metcon_id}", headers=mcp)
            assert previous.status_code == 200
            changed = await client.patch(f"/metcons/{metcon_id}", headers=mcp,
                json={"expected_version": previous.json()["version"],
                      "changes": {"score_time_seconds": 205}})
            assert changed.status_code == 200, changed.text
            assert changed.json()["record"]["name"] == "Fran"
            assert changed.json()["record"]["score_time_seconds"] == 205
            assert (await client.patch(f"/metcons/{metcon_id}", headers=mcp,
                json={"expected_version": previous.json()["version"],
                      "changes": {"notes": "late"}})).status_code == 412

            # Legacy GPT routes still work with the GPT credential.
            assert (await client.get("/workouts", headers=gpt)).status_code == 200
        await app_module.engine.dispose()

    asyncio.run(exercise())
