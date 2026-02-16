"""
Test suite for CF-Log API v13.
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
    "cycle": 1,
    "week": 1,
    "day": 1,
    "notes": "felt good",
}

VALID_METCON = {
    "date": "2026-01-20",
    "name": "Fran",
    "workout_type": "for_time",
    "description": "21-15-9 Thrusters (43kg) & Pull-ups",
    "score_time_seconds": 245,
    "score_display": "4:05",
    "rx": "rx",
    "notes": "PR!",
    "tags": "benchmark,pr",
}


# ─── Health & Root ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_root(client):
    r = await client.get("/")
    assert r.status_code == 200
    assert "v13" in r.json()["message"]


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
    data = r.json()
    assert data["id"] is not None
    assert data["exercise"] == "Back Squat"
    assert data["value"] == 100.0

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
    assert data["total_metcons"] == 0
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
    data = r.json()
    assert data["total"] > 0
    assert any(w["cycle"] == 1 and w["week"] == 1 for w in data["completed_weeks"])


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
    data = r.json()
    assert data["workout_rows"] == 1
    assert data["metcon_rows"] == 0
    assert "SQLite" in data["db_type"]


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


# =============================================================================
# METCON TESTS
# =============================================================================

# ─── Metcon Validation ───────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metcon_invalid_date(client):
    m = {**VALID_METCON, "date": "20-01-2026"}
    r = await client.post("/metcons", json=m)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_metcon_empty_name_rejected(client):
    m = {**VALID_METCON, "name": "   "}
    r = await client.post("/metcons", json=m)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_metcon_invalid_workout_type(client):
    m = {**VALID_METCON, "workout_type": "crossfit"}
    r = await client.post("/metcons", json=m)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_metcon_invalid_rx(client):
    m = {**VALID_METCON, "rx": "modified"}
    r = await client.post("/metcons", json=m)
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_metcon_name_normalized(client):
    m = {**VALID_METCON, "name": "fran"}
    r = await client.post("/metcons", json=m)
    assert r.status_code == 200
    r2 = await client.get("/metcons", params={"name": "Fran"})
    assert r2.status_code == 200
    assert len(r2.json()) == 1
    assert r2.json()[0]["name"] == "Fran"


@pytest.mark.asyncio
async def test_metcon_tags_normalized(client):
    m = {**VALID_METCON, "tags": " Benchmark , PR , "}
    r = await client.post("/metcons", json=m)
    assert r.status_code == 200
    r2 = await client.get("/metcons")
    assert r2.json()[0]["tags"] == "benchmark,pr"


# ─── Metcon CRUD ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_add_and_get_metcon(client):
    r = await client.post("/metcons", json=VALID_METCON)
    assert r.status_code == 200
    data = r.json()
    assert data["id"] is not None
    assert data["name"] == "Fran"
    assert data["workout_type"] == "for_time"
    assert data["score_time_seconds"] == 245

    r2 = await client.get("/metcons")
    data = r2.json()
    assert len(data) == 1
    assert data[0]["name"] == "Fran"
    assert data[0]["workout_type"] == "for_time"
    assert data[0]["score_time_seconds"] == 245
    assert data[0]["rx"] == "rx"


@pytest.mark.asyncio
async def test_metcon_bulk(client):
    metcons = [
        {**VALID_METCON, "name": f"WOD {i}", "date": f"2026-01-{20+i:02d}"}
        for i in range(1, 4)
    ]
    r = await client.post("/metcons/bulk", json={"metcons": metcons})
    assert r.status_code == 200
    data = r.json()
    assert data["saved"] == 3
    assert len(data["ids"]) == 3


@pytest.mark.asyncio
async def test_metcon_bulk_empty_rejected(client):
    r = await client.post("/metcons/bulk", json={"metcons": []})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_edit_metcon(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons")
    mid = r.json()[0]["id"]

    updated = {**VALID_METCON, "score_time_seconds": 220, "score_display": "3:40"}
    r2 = await client.put(f"/metcons/{mid}", json=updated)
    assert r2.status_code == 200
    assert r2.json()["score_time_seconds"] == 220


@pytest.mark.asyncio
async def test_delete_metcon(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons")
    mid = r.json()[0]["id"]

    r2 = await client.delete(f"/metcons/{mid}")
    assert r2.status_code == 200

    r3 = await client.get("/metcons")
    assert len(r3.json()) == 0


@pytest.mark.asyncio
async def test_delete_metcon_not_found(client):
    r = await client.delete("/metcons/99999")
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_last_metcon(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons/last", params={"name": "fran"})
    assert r.status_code == 200
    assert r.json()["name"] == "Fran"


@pytest.mark.asyncio
async def test_last_metcon_not_found(client):
    r = await client.get("/metcons/last", params={"name": "unicorn"})
    assert r.status_code == 404


# ─── Metcon Search & Query ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metcon_search(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons/search", params={"q": "fran"})
    assert r.status_code == 200
    assert len(r.json()) == 1


@pytest.mark.asyncio
async def test_metcon_search_by_type(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons/search", params={"q": "fran", "workout_type": "for_time"})
    assert r.status_code == 200
    assert len(r.json()) == 1

    r2 = await client.get("/metcons/search", params={"q": "fran", "workout_type": "amrap"})
    assert r2.status_code == 200
    assert len(r2.json()) == 0


@pytest.mark.asyncio
async def test_metcon_search_by_rx(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons/search", params={"q": "fran", "rx": "rx"})
    assert r.status_code == 200
    assert len(r.json()) == 1

    r2 = await client.get("/metcons/search", params={"q": "fran", "rx": "scaled"})
    assert r2.status_code == 200
    assert len(r2.json()) == 0


@pytest.mark.asyncio
async def test_metcon_query_filters(client):
    m1 = {**VALID_METCON, "cycle": 1, "week": 1}
    m2 = {**VALID_METCON, "name": "Murph", "cycle": 1, "week": 2,
           "workout_type": "for_time", "score_time_seconds": 2400, "score_display": "40:00"}
    await client.post("/metcons", json=m1)
    await client.post("/metcons", json=m2)

    r = await client.get("/metcons", params={"cycle": 1, "week": 1})
    assert r.status_code == 200
    assert len(r.json()) == 1
    assert r.json()[0]["name"] == "Fran"


@pytest.mark.asyncio
async def test_metcon_search_by_tag(client):
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/metcons/search", params={"q": "fran", "tag": "benchmark"})
    assert r.status_code == 200
    assert len(r.json()) == 1

    r2 = await client.get("/metcons/search", params={"q": "fran", "tag": "deload"})
    assert r2.status_code == 200
    assert len(r2.json()) == 0


# ─── Metcon Analytics: PRs ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metcon_prs_for_time(client):
    """For timed WODs, PR = fastest time."""
    m1 = {**VALID_METCON, "score_time_seconds": 300, "score_display": "5:00"}
    m2 = {**VALID_METCON, "date": "2026-02-01", "score_time_seconds": 245, "score_display": "4:05"}
    await client.post("/metcons", json=m1)
    await client.post("/metcons", json=m2)

    r = await client.get("/analytics/metcon_prs", params={"name": "fran"})
    assert r.status_code == 200
    prs = r.json()["benchmarks"]
    assert len(prs) == 1
    assert prs[0]["name"] == "Fran"
    assert prs[0]["best_time_seconds"] == 245
    assert prs[0]["best_time_display"] == "4:05"


@pytest.mark.asyncio
async def test_metcon_prs_amrap(client):
    """For AMRAP, PR = most rounds+reps."""
    m1 = {
        "date": "2026-01-20", "name": "Cindy", "workout_type": "amrap",
        "description": "20min AMRAP: 5 Pull-ups, 10 Push-ups, 15 Squats",
        "score_rounds": 18, "score_reps": 3, "score_display": "18+3", "rx": "rx",
    }
    m2 = {
        "date": "2026-02-20", "name": "Cindy", "workout_type": "amrap",
        "score_rounds": 20, "score_reps": 0, "score_display": "20+0", "rx": "rx",
    }
    await client.post("/metcons", json=m1)
    await client.post("/metcons", json=m2)

    r = await client.get("/analytics/metcon_prs", params={"name": "cindy"})
    assert r.status_code == 200
    prs = r.json()["benchmarks"]
    assert len(prs) == 1
    assert prs[0]["best_rounds"] == 20
    assert prs[0]["best_reps"] == 0


@pytest.mark.asyncio
async def test_metcon_prs_all(client):
    """Get all PRs when no name filter."""
    await client.post("/metcons", json=VALID_METCON)
    amrap = {
        "date": "2026-01-25", "name": "Cindy", "workout_type": "amrap",
        "score_rounds": 18, "score_reps": 3, "rx": "rx",
    }
    await client.post("/metcons", json=amrap)

    r = await client.get("/analytics/metcon_prs")
    assert r.status_code == 200
    prs = r.json()["benchmarks"]
    assert len(prs) == 2
    names = {p["name"] for p in prs}
    assert "Fran" in names
    assert "Cindy" in names


@pytest.mark.asyncio
async def test_metcon_prs_rx_filter(client):
    m_rx = {**VALID_METCON, "rx": "rx", "score_time_seconds": 300}
    m_scaled = {**VALID_METCON, "date": "2026-02-01", "rx": "scaled", "score_time_seconds": 200}
    await client.post("/metcons", json=m_rx)
    await client.post("/metcons", json=m_scaled)

    r = await client.get("/analytics/metcon_prs", params={"name": "fran", "rx": "rx"})
    prs = r.json()["benchmarks"]
    assert len(prs) == 1
    assert prs[0]["rx"] == "rx"
    assert prs[0]["best_time_seconds"] == 300


# ─── Metcon Analytics: Timeline ──────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metcon_timeline(client):
    m1 = {**VALID_METCON, "date": "2026-01-10", "score_time_seconds": 300, "score_display": "5:00"}
    m2 = {**VALID_METCON, "date": "2026-02-10", "score_time_seconds": 260, "score_display": "4:20"}
    m3 = {**VALID_METCON, "date": "2026-03-10", "score_time_seconds": 245, "score_display": "4:05"}
    await client.post("/metcons", json=m1)
    await client.post("/metcons", json=m2)
    await client.post("/metcons", json=m3)

    r = await client.get("/analytics/metcon_timeline", params={"name": "fran"})
    assert r.status_code == 200
    data = r.json()
    assert data["name"] == "fran"
    assert data["workout_type"] == "for_time"
    assert len(data["timeline"]) == 3
    # Verify chronological order
    assert data["timeline"][0]["score_time_seconds"] == 300
    assert data["timeline"][1]["score_time_seconds"] == 260
    assert data["timeline"][2]["score_time_seconds"] == 245


@pytest.mark.asyncio
async def test_metcon_timeline_date_range(client):
    m1 = {**VALID_METCON, "date": "2026-01-10", "score_time_seconds": 300}
    m2 = {**VALID_METCON, "date": "2026-02-10", "score_time_seconds": 260}
    m3 = {**VALID_METCON, "date": "2026-03-10", "score_time_seconds": 245}
    await client.post("/metcons", json=m1)
    await client.post("/metcons", json=m2)
    await client.post("/metcons", json=m3)

    r = await client.get("/analytics/metcon_timeline", params={
        "name": "fran", "start": "2026-02-01", "end": "2026-02-28"
    })
    assert r.status_code == 200
    assert len(r.json()["timeline"]) == 1


@pytest.mark.asyncio
async def test_metcon_timeline_empty(client):
    r = await client.get("/analytics/metcon_timeline", params={"name": "unicorn"})
    assert r.status_code == 200
    assert len(r.json()["timeline"]) == 0


# ─── Stats with Metcons ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_stats_with_metcons(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total_sessions"] == 1
    assert data["total_metcons"] == 1


@pytest.mark.asyncio
async def test_dbinfo_with_metcons(client):
    await client.post("/workouts", json=VALID_WORKOUT)
    await client.post("/metcons", json=VALID_METCON)
    r = await client.get("/debug/dbinfo")
    assert r.status_code == 200
    data = r.json()
    assert data["workout_rows"] == 1
    assert data["metcon_rows"] == 1


# ─── Metcon workout types ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_metcon_all_valid_types(client):
    """All six valid workout types should be accepted."""
    valid_types = ["for_time", "amrap", "emom", "chipper", "interval", "other"]
    for wt in valid_types:
        m = {**VALID_METCON, "name": f"Test {wt}", "workout_type": wt}
        r = await client.post("/metcons", json=m)
        assert r.status_code == 200, f"Failed for workout_type={wt}"


@pytest.mark.asyncio
async def test_metcon_all_valid_rx(client):
    """All three valid rx values should be accepted."""
    for rx_val in ["rx", "scaled", "rx_plus"]:
        m = {**VALID_METCON, "name": f"Test {rx_val}", "rx": rx_val}
        r = await client.post("/metcons", json=m)
        assert r.status_code == 200, f"Failed for rx={rx_val}"


# ─── Day Summary ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_day_summary_empty(client):
    r = await client.get("/day_summary", params={"cycle": 1, "week": 1, "day": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["workouts"] == []
    assert data["metcons"] == []
    assert data["total_workout_sets"] == 0
    assert data["total_metcons"] == 0


@pytest.mark.asyncio
async def test_day_summary_with_data(client):
    # Add workouts
    for i in range(1, 4):
        w = {**VALID_WORKOUT, "set_number": i, "value": 100.0 + i * 5}
        await client.post("/workouts", json=w)
    # Add metcon
    m = {**VALID_METCON, "cycle": 1, "week": 1, "day": 1}
    await client.post("/metcons", json=m)
    # Query day summary
    r = await client.get("/day_summary", params={"cycle": 1, "week": 1, "day": 1})
    assert r.status_code == 200
    data = r.json()
    assert data["total_workout_sets"] == 3
    assert data["total_metcons"] == 1
    assert len(data["workouts"]) == 3
    assert len(data["metcons"]) == 1


# ─── Metcon CSV Export ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_export_metcons_csv(client):
    m = {**VALID_METCON, "cycle": 1, "week": 1, "day": 1}
    await client.post("/metcons", json=m)
    r = await client.get("/export/metcons_csv")
    assert r.status_code == 200
    data = r.json()
    assert data["filename"] == "metcons.csv"
    assert data["rows"] == 1
    assert "Fran" in data["csv"]


@pytest.mark.asyncio
async def test_export_metcons_csv_empty(client):
    r = await client.get("/export/metcons_csv")
    assert r.status_code == 200
    data = r.json()
    assert data["rows"] == 0
