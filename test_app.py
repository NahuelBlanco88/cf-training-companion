"""
Test suite for CF-Log API v11.
Uses an in-memory SQLite database via aiosqlite.
"""
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Override env BEFORE importing app so it uses in-memory SQLite
import os
os.environ.pop("CLOUD_SQL_CONNECTION_NAME", None)
os.environ.pop("CFLOG_DB", None)
os.environ["RATE_LIMIT_REQUESTS"] = "10000"  # effectively disable for tests

from app import app, engine, _init_db, Base, _rate_limit_store  # noqa: E402

transport = ASGITransport(app=app)


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    """Create a fresh database for each test."""
    _rate_limit_store.clear()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ─── Sample data ─────────────────────────────────────────────────────────────

VALID_WORKOUT = {
    "date": "2026-01-15",
    "exercise": "Back Squat",
    "set_number": 1,
    "reps": 5,
    "value": 100.0,
    "unit": "kg",
    "rpe": 8.0,
    "cycle": 1,
    "week": 1,
    "day": 1,
    "notes": "felt good",
}


# ─── Health & Root ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "v11" in r.json()["message"]


@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    assert data["db_connected"] is True
    assert "SQLite" in data["db_type"]


# ─── Input Validation ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_invalid_date_format(client):
    w = {**VALID_WORKOUT, "date": "15-01-2026"}
    r = await client.post("/workouts", json=w)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_invalid_date_calendar(client):
    w = {**VALID_WORKOUT, "date": "2026-02-30"}
    r = await client.post("/workouts", json=w)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_rpe_out_of_range(client):
    w = {**VALID_WORKOUT, "rpe": 11.0}
    r = await client.post("/workouts", json=w)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_exercise_name_normalized(client):
    w = {**VALID_WORKOUT, "exercise": "back squat"}
    r = await client.post("/workouts", json=w, params={"force": True})
    assert r.status_code == 200
    # Verify it's stored as Title Case
    r2 = await client.get("/workouts", params={"exercise": "Back Squat"})
    assert r2.status_code == 200
    data = r2.json()
    assert len(data) == 1
    assert data[0]["exercise"] == "Back Squat"


@pytest.mark.asyncio
async def test_empty_exercise_rejected(client):
    w = {**VALID_WORKOUT, "exercise": "   "}
    r = await client.post("/workouts", json=w)
    assert r.status_code == 422


# ─── CRUD ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_get_workout(client):
    r = await client.post("/workouts", json=VALID_WORKOUT)
    assert r.status_code == 200
    assert r.json()["message"] == "Workout saved"

    r2 = await client.get("/workouts")
    data = r2.json()
    assert len(data) == 1
    assert data[0]["exercise"] == "Back Squat"
    assert data[0]["value"] == 100.0


@pytest.mark.asyncio
async def test_bulk_add(client):
    workouts = [
        {**VALID_WORKOUT, "set_number": i, "value": 100.0 + i * 5}
        for i in range(1, 4)
    ]
    r = await client.post("/workouts/bulk", json={"workouts": workouts})
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] == 3
    assert len(data["ids"]) == 3


@pytest.mark.asyncio
async def test_bulk_empty_rejected(client):
    r = await client.post("/workouts/bulk", json={"workouts": []})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_workout(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/workouts")
    wid = r.json()[0]["id"]

    updated = {**VALID_WORKOUT, "value": 120.0}
    r2 = await client.put(f"/workouts/{wid}", json=updated)
    assert r2.status_code == 200
    assert r2.json()["value"] == 120.0


@pytest.mark.asyncio
async def test_delete_workout(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/workouts")
    wid = r.json()[0]["id"]

    r2 = await client.delete(f"/workouts/{wid}")
    assert r2.status_code == 200

    r3 = await client.get("/workouts")
    assert len(r3.json()) == 0


@pytest.mark.asyncio
async def test_delete_nonexistent(client):
    r = await client.delete("/workouts/99999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_last_workout(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/workouts/last", params={"exercise": "squat"})
    assert r.status_code == 200
    assert r.json()["exercise"] == "Back Squat"


@pytest.mark.asyncio
async def test_last_workout_not_found(client):
    r = await client.get("/workouts/last", params={"exercise": "unicorn"})
    assert r.status_code == 404


# ─── Duplicate Detection ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_duplicate_detection(client):
    r1 = await client.post("/workouts", json=VALID_WORKOUT)
    assert r1.status_code == 200

    r2 = await client.post("/workouts", json=VALID_WORKOUT)
    assert r2.status_code == 409
    assert "duplicate" in r2.json()["detail"].lower()


@pytest.mark.asyncio
async def test_duplicate_force_override(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.post("/workouts", json=VALID_WORKOUT, params={"force": True})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_bulk_duplicate_detection(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.post(
        "/workouts/bulk", json={"workouts": [VALID_WORKOUT]}
    )
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_no_false_duplicate_without_set_number(client):
    """When set_number is omitted, multiple sets should be allowed (no dup block)."""
    w = {**VALID_WORKOUT}
    del w["set_number"]
    r1 = await client.post("/workouts", json=w)
    assert r1.status_code == 200
    r2 = await client.post("/workouts", json=w)
    assert r2.status_code == 200  # should NOT be 409


# ─── Search ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_search_workouts(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/workouts/search", params={"q": "squat"})
    assert r.status_code == 200
    assert len(r.json()) == 1


@pytest.mark.asyncio
async def test_search_exercise(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/search_exercise", params={"exercise": "squat"})
    assert r.status_code == 200
    assert len(r.json()["logged"]) == 1


@pytest.mark.asyncio
async def test_search_with_special_chars(client):
    """SQL injection characters in search should be escaped, not error."""
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/workouts/search", params={"q": "squat%'; DROP TABLE"})
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_workouts_by_cwd(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get(
        "/workouts/by_cwd", params={"cycle": 1, "week": 1, "day": 1}
    )
    assert r.status_code == 200
    assert len(r.json()) == 1


# ─── Tags ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_tags_normalized(client):
    w = {**VALID_WORKOUT, "tags": " PR Attempt , Deload , "}
    r = await client.post("/workouts", json=w)
    assert r.status_code == 200

    r2 = await client.get("/workouts")
    assert r2.json()[0]["tags"] == "pr attempt,deload"


@pytest.mark.asyncio
async def test_search_by_tag(client):
    w = {**VALID_WORKOUT, "tags": "pr attempt"}
    await client.post("/workouts", json=w)
    r = await client.get("/workouts/search", params={"q": "squat", "tag": "pr"})
    assert r.status_code == 200
    assert len(r.json()) == 1


@pytest.mark.asyncio
async def test_search_by_tag_no_match(client):
    w = {**VALID_WORKOUT, "tags": "deload"}
    await client.post("/workouts", json=w)
    r = await client.get("/workouts/search", params={"q": "squat", "tag": "pr"})
    assert r.status_code == 200
    assert len(r.json()) == 0


# ─── Analytics ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_sessions"] == 1
    assert "Back Squat" in data["bests"]


@pytest.mark.asyncio
async def test_prs(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/analytics/prs")
    assert r.status_code == 200
    assert "Back Squat" in r.json()["prs"]


@pytest.mark.asyncio
async def test_prs_filtered(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/analytics/prs", params={"exercise": "Back Squat"})
    assert r.status_code == 200
    assert "Back Squat" in r.json()["prs"]


@pytest.mark.asyncio
async def test_weekly_summary(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get(
        "/analytics/weekly_summary", params={"cycle": 1, "week": 1}
    )
    assert r.status_code == 200
    data = r.json()
    assert data["total_sets"] == 1
    assert "1" in str(data["days_logged"])


@pytest.mark.asyncio
async def test_progress_compare(client):
    w1 = {**VALID_WORKOUT, "week": 1}
    w2 = {**VALID_WORKOUT, "week": 2, "set_number": 2}
    await client.post("/workouts", json=w1)
    await client.post("/workouts", json=w2)
    r = await client.get(
        "/analytics/progress_compare",
        params={"exercise": "Back Squat", "cycle": 1, "week1": 1, "week2": 2},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["week1_sets"] == 1
    assert data["week2_sets"] == 1


@pytest.mark.asyncio
async def test_repmax(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/analytics/repmax", params={"exercise": "Back Squat"})
    assert r.status_code == 200
    assert r.json()["best_value"] == 100.0


@pytest.mark.asyncio
async def test_repmax_invalid_mode(client):
    r = await client.get(
        "/analytics/repmax", params={"exercise": "x", "mode": "invalid"}
    )
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_consistency(client):
    # Need days 1-4 for 100% consistency
    for d in range(1, 5):
        w = {**VALID_WORKOUT, "day": d, "set_number": d}
        await client.post("/workouts", json=w, params={"force": True})
    r = await client.get("/analytics/consistency", params={"cycle": 1})
    assert r.status_code == 200
    assert 1 in r.json()["weeks_completed_100"]


# ─── 1RM Calculator ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_estimated_1rm_epley(client):
    # 100kg x 5 reps → Epley: 100*(1 + 5/30) = 116.67
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get(
        "/analytics/estimated_1rm",
        params={"exercise": "Back Squat", "formula": "epley"},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["estimated_1rm_kg"] == pytest.approx(116.7, abs=0.1)
    assert data["based_on_value"] == 100.0
    assert data["based_on_reps"] == 5


@pytest.mark.asyncio
async def test_estimated_1rm_brzycki(client):
    # 100kg x 5 reps → Brzycki: 100*(36/(37-5)) = 112.5
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get(
        "/analytics/estimated_1rm",
        params={"exercise": "Back Squat", "formula": "brzycki"},
    )
    assert r.status_code == 200
    assert r.json()["estimated_1rm_kg"] == pytest.approx(112.5, abs=0.1)


@pytest.mark.asyncio
async def test_estimated_1rm_no_data(client):
    r = await client.get(
        "/analytics/estimated_1rm", params={"exercise": "Unicorn Lift"}
    )
    assert r.status_code == 200
    assert r.json()["estimated_1rm_kg"] is None


@pytest.mark.asyncio
async def test_estimated_1rm_invalid_formula(client):
    r = await client.get(
        "/analytics/estimated_1rm",
        params={"exercise": "Back Squat", "formula": "invalid"},
    )
    assert r.status_code == 400


# ─── Volume Analytics ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_volume(client):
    workouts = [
        {**VALID_WORKOUT, "set_number": i, "reps": 5, "value": 100.0}
        for i in range(1, 4)
    ]
    await client.post("/workouts/bulk", json={"workouts": workouts})
    r = await client.get("/analytics/volume", params={"cycle": 1, "week": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["grand_total_sets"] == 3
    assert data["grand_total_volume"] == 1500.0  # 3 sets * 5 reps * 100kg


@pytest.mark.asyncio
async def test_volume_empty(client):
    r = await client.get("/analytics/volume", params={"cycle": 99})
    assert r.status_code == 200
    assert r.json()["grand_total_sets"] == 0


# ─── Exercise Timeline ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_timeline(client):
    w1 = {**VALID_WORKOUT, "date": "2026-01-10", "value": 90.0}
    w2 = {**VALID_WORKOUT, "date": "2026-01-15", "value": 100.0, "set_number": 2}
    await client.post("/workouts", json=w1)
    await client.post("/workouts", json=w2)
    r = await client.get(
        "/analytics/timeline", params={"exercise": "Back Squat"}
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["timeline"]) == 2
    assert data["timeline"][0]["best_value"] == 90.0
    assert data["timeline"][1]["best_value"] == 100.0


@pytest.mark.asyncio
async def test_timeline_empty(client):
    r = await client.get(
        "/analytics/timeline", params={"exercise": "Unicorn Lift"}
    )
    assert r.status_code == 200
    assert len(r.json()["timeline"]) == 0


# ─── Verify ──────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_verify_match(client):
    workouts = [
        {**VALID_WORKOUT, "set_number": i} for i in range(1, 4)
    ]
    await client.post("/workouts/bulk", json={"workouts": workouts})
    r = await client.get(
        "/workouts/verify",
        params={"date": "2026-01-15", "exercise": "Back Squat", "expected_sets": 3},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["match"] is True
    assert data["actual_sets"] == 3


@pytest.mark.asyncio
async def test_verify_mismatch(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get(
        "/workouts/verify",
        params={"date": "2026-01-15", "exercise": "Back Squat", "expected_sets": 5},
    )
    assert r.status_code == 200
    assert r.json()["match"] is False


# ─── Undo ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_undo_last(client):
    workouts = [
        {**VALID_WORKOUT, "set_number": i} for i in range(1, 4)
    ]
    await client.post("/workouts/bulk", json={"workouts": workouts})

    r = await client.request("DELETE", "/workouts/undo_last", params={"count": 3})
    assert r.status_code == 200
    assert r.json()["deleted"] == 3

    r2 = await client.get("/workouts")
    assert len(r2.json()) == 0


@pytest.mark.asyncio
async def test_undo_last_empty(client):
    r = await client.request("DELETE", "/workouts/undo_last", params={"count": 1})
    assert r.status_code == 404


# ─── Debug ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_dbinfo(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/debug/dbinfo")
    assert r.status_code == 200
    assert r.json()["workout_rows"] == 1
    assert "SQLite" in r.json()["db_type"]


@pytest.mark.asyncio
async def test_debug_exercises(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/debug/exercises")
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["exercise"] == "Back Squat"


# ─── Export CSV ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_csv(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    r = await client.get("/export/csv")
    assert r.status_code == 200
    data = r.json()
    assert data["rows"] == 1
    assert "Back Squat" in data["csv"]
    assert "tags" in data["csv"]  # new column present


# ─── Rate Limiting Headers ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rate_limit_headers(client):
    r = await client.get("/")
    assert "X-RateLimit-Limit" in r.headers
    assert "X-RateLimit-Remaining" in r.headers
