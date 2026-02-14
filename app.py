# app.py
# =============================================================================
# CF-Log API — Workouts & Analytics (FastAPI + SQLAlchemy 2.x async, Pydantic v2)
# Results-only: no plan storage. One row per working set for maximum precision.
# v11.0.0 — async, rate-limited, validated, with tagging & advanced analytics
# =============================================================================

from __future__ import annotations

import csv
import io
import math
import os
import re
import logging
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path as OSPath
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as FPath
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import (
    Float,
    Integer,
    String,
    asc,
    desc,
    func,
    select,
    text,
)
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("cf-log-api")

# -----------------------------------------------------------------------------
# DB connection
# Priority:
#   1) Cloud SQL (PostgreSQL) if CLOUD_SQL_CONNECTION_NAME is set
#   2) env CFLOG_DB (absolute path to cf_log.db)
#   3) ./data/cf_log.db
#   4) ./cf_log.db  (fallback)
# -----------------------------------------------------------------------------
_cloud_sql = os.getenv("CLOUD_SQL_CONNECTION_NAME")  # e.g. project:region:instance
_db_user = os.getenv("DB_USER", "postgres")
_db_pass = os.getenv("DB_PASSWORD", "")
_db_name = os.getenv("DB_NAME", "cf_log")

if _cloud_sql:
    _socket_path = f"/cloudsql/{_cloud_sql}"
    DB_PATH = f"postgresql+asyncpg://{_db_user}:{_db_pass}@/{_db_name}?host={_socket_path}"
    engine = create_async_engine(DB_PATH, echo=False, pool_pre_ping=True)
    log.info(f"Using Cloud SQL (async): {_cloud_sql}")
else:
    env_db = os.getenv("CFLOG_DB")
    candidates = [
        env_db,
        str((OSPath(__file__).parent / "data" / "cf_log.db").resolve()),
        str((OSPath(__file__).parent / "cf_log.db").resolve()),
    ]
    DB_PATH = next((p for p in candidates if p and OSPath(p).exists()), candidates[-1])
    engine = create_async_engine(f"sqlite+aiosqlite:///{DB_PATH}", echo=False)
    log.info(f"Using SQLite (async): {DB_PATH}")

async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# -----------------------------------------------------------------------------
# SQLAlchemy models
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Workout(Base):
    __tablename__ = "workout"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String, nullable=False)           # YYYY-MM-DD
    exercise: Mapped[str] = mapped_column(String, nullable=False)       # canonical name
    set_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    rpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cycle: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    iso_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # comma-separated tags

    # Legacy columns kept for backward compat with old data
    sets: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    plan_day_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


# -----------------------------------------------------------------------------
# Startup: create tables & run migrations
# -----------------------------------------------------------------------------
async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Migrate: add columns if they don't exist
    async with engine.begin() as conn:
        for col_name, col_sql in [
            ("set_number", "ALTER TABLE workout ADD COLUMN set_number INTEGER"),
            ("tags", "ALTER TABLE workout ADD COLUMN tags VARCHAR"),
        ]:
            try:
                await conn.execute(text(f"SELECT {col_name} FROM workout LIMIT 1"))
                log.info(f"Column {col_name} already exists")
            except Exception:
                try:
                    await conn.execute(text(col_sql))
                    log.info(f"Added {col_name} column to workout table")
                except Exception as e:
                    log.warning(f"Migration for {col_name}: {e}")


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class HealthOut(BaseModel):
    ok: bool = True
    db_connected: bool = True
    db_type: str
    timestamp: str


class GenericResponse(BaseModel):
    message: str


class DBInfoOut(BaseModel):
    db_type: str
    workout_rows: int


class ExerciseCountOut(BaseModel):
    exercise: str
    count: int


class WorkoutIn(BaseModel):
    """One working set. GPT must send one of these per set."""
    date: str
    exercise: str
    set_number: Optional[int] = None
    reps: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    rpe: Optional[float] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    iso_week: Optional[int] = None
    day: Optional[int] = None
    notes: Optional[str] = ""
    tags: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        if not _DATE_RE.match(v):
            raise ValueError("date must be YYYY-MM-DD format")
        try:
            datetime.strptime(v, "%Y-%m-%d")
        except ValueError:
            raise ValueError("date is not a valid calendar date")
        return v

    @field_validator("exercise")
    @classmethod
    def normalize_exercise(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("exercise name cannot be empty")
        return v.title()

    @field_validator("rpe")
    @classmethod
    def validate_rpe(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and (v < 1 or v > 10):
            raise ValueError("rpe must be between 1 and 10")
        return v

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        cleaned = ",".join(t.strip().lower() for t in v.split(",") if t.strip())
        return cleaned or None


class WorkoutOut(BaseModel):
    id: int
    date: str
    exercise: str
    set_number: Optional[int] = None
    reps: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    rpe: Optional[float] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    iso_week: Optional[int] = None
    day: Optional[int] = None
    notes: Optional[str] = ""
    tags: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class BulkWorkoutIn(BaseModel):
    workouts: List[WorkoutIn]


class BulkWorkoutOut(BaseModel):
    saved: int
    ids: List[int]


class StatsOut(BaseModel):
    total_sessions: int
    bests: Dict[str, float] = Field(default_factory=dict)
    last: Optional[WorkoutOut] = None


class PRsOut(BaseModel):
    prs: Dict[str, float] = Field(default_factory=dict)


class WeeklySummaryOut(BaseModel):
    cycle: int
    week: Optional[int] = None
    days_logged: Dict[int, int] = Field(default_factory=dict)
    total_sets: int


class ProgressCompareOut(BaseModel):
    exercise: str
    cycle: int
    week1: int
    week2: int
    week1_sets: int
    week2_sets: int
    delta: int


class RepmaxOut(BaseModel):
    exercise: str
    best_value: Optional[float] = None
    unit: Optional[str] = None
    workout_id: Optional[int] = None


class ConsistencyOut(BaseModel):
    weeks_completed_100: List[int] = Field(default_factory=list)


class SearchExerciseOut(BaseModel):
    exercise: str
    logged: List[WorkoutOut] = Field(default_factory=list)


class CsvExportOut(BaseModel):
    filename: str
    rows: int
    csv: str


class OneRMOut(BaseModel):
    exercise: str
    estimated_1rm_kg: Optional[float] = None
    based_on_value: Optional[float] = None
    based_on_reps: Optional[int] = None
    formula: str = "epley"


class VolumeOut(BaseModel):
    exercise: str
    total_volume: float
    total_sets: int
    total_reps: int
    unit: Optional[str] = None


class WeeklyVolumeOut(BaseModel):
    cycle: Optional[int] = None
    week: Optional[int] = None
    exercises: List[VolumeOut] = Field(default_factory=list)
    grand_total_volume: float = 0.0
    grand_total_sets: int = 0


class TimelinePointOut(BaseModel):
    date: str
    best_value: float
    unit: Optional[str] = None
    total_sets: int
    total_reps: int


class ExerciseTimelineOut(BaseModel):
    exercise: str
    timeline: List[TimelinePointOut] = Field(default_factory=list)


class VerifyOut(BaseModel):
    date: str
    exercise: Optional[str] = None
    expected_sets: Optional[int] = None
    actual_sets: int
    match: bool
    logged: List[WorkoutOut] = Field(default_factory=list)


class UndoOut(BaseModel):
    deleted: int
    ids: List[int]


class DuplicateWarning(BaseModel):
    message: str
    existing_ids: List[int]


# -----------------------------------------------------------------------------
# App (with lifespan)
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    await _init_db()
    yield
    await engine.dispose()


app = FastAPI(
    title="CF-Log API",
    description="CrossFit Training Log & Analytics API. Results only — one row per working set.",
    version="11.0.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Rate limiting middleware (simple in-memory, per-IP)
# -----------------------------------------------------------------------------
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "60"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))  # seconds


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limit_store[client_ip] = [
        t for t in _rate_limit_store[client_ip] if t > window_start
    ]

    if len(_rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        return Response(
            content='{"detail":"Rate limit exceeded. Try again later."}',
            status_code=429,
            media_type="application/json",
        )

    _rate_limit_store[client_ip].append(now)
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(
        RATE_LIMIT_REQUESTS - len(_rate_limit_store[client_ip])
    )
    return response


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _escape_like(s: str) -> str:
    """Escape SQL LIKE special characters to prevent injection."""
    return s.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _safe_like(column, term: str):
    """Build a safe case-insensitive LIKE expression with proper escaping."""
    escaped = _escape_like(term.strip().lower())
    pattern = f"%{escaped}%"
    return func.lower(column).like(pattern, escape="\\")


def _row_to_out(w: Workout) -> WorkoutOut:
    return WorkoutOut(
        id=w.id,
        date=w.date,
        exercise=w.exercise,
        set_number=w.set_number,
        reps=int(w.reps) if w.reps is not None else None,
        value=w.value,
        unit=w.unit,
        rpe=w.rpe,
        cycle=w.cycle,
        week=w.week,
        iso_week=w.iso_week,
        day=w.day,
        notes=w.notes,
        tags=w.tags,
    )


def _db_type() -> str:
    """Return a safe description of the DB type (no credentials)."""
    if _cloud_sql:
        return f"Cloud SQL PostgreSQL ({_cloud_sql})"
    return "SQLite"


# -----------------------------------------------------------------------------
# Duplicate detection helper
# -----------------------------------------------------------------------------
async def _check_duplicates(
    session: AsyncSession, w: WorkoutIn
) -> List[int]:
    """Return IDs of existing rows that match date+exercise+set_number+cycle+week+day.

    Requires set_number to be present for the check to run. Without it, the query
    would be too broad (date+exercise only) and block multi-set logging for clients
    that don't send set_number.
    """
    if w.set_number is None:
        return []  # Can't reliably detect duplicates without set_number
    stmt = select(Workout.id).where(
        Workout.date == w.date,
        func.lower(Workout.exercise) == w.exercise.lower(),
        Workout.set_number == w.set_number,
    )
    if w.cycle is not None:
        stmt = stmt.where(Workout.cycle == w.cycle)
    if w.week is not None:
        stmt = stmt.where(Workout.week == w.week)
    if w.day is not None:
        stmt = stmt.where(Workout.day == w.day)
    result = await session.execute(stmt)
    return [row[0] for row in result.all()]


# -----------------------------------------------------------------------------
# Health / Root
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    db_connected = False
    try:
        async with async_session() as s:
            await s.execute(text("SELECT 1"))
            db_connected = True
    except Exception as e:
        log.error(f"Health check DB query failed: {e}")
    return HealthOut(
        ok=db_connected,
        db_connected=db_connected,
        db_type=_db_type(),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/", response_model=GenericResponse)
async def root() -> GenericResponse:
    return GenericResponse(message="CF-Log API v11 is running")


# -----------------------------------------------------------------------------
# Debug
# -----------------------------------------------------------------------------
@app.get("/debug/dbinfo", response_model=DBInfoOut)
async def dbinfo() -> DBInfoOut:
    async with async_session() as s:
        result = await s.execute(text("SELECT COUNT(*) FROM workout"))
        workout_rows = result.scalar_one()
    return DBInfoOut(db_type=_db_type(), workout_rows=int(workout_rows))


@app.get("/debug/exercises", response_model=List[ExerciseCountOut])
async def debug_exercises(limit: int = Query(50, ge=1, le=500)) -> List[ExerciseCountOut]:
    async with async_session() as s:
        result = await s.execute(
            select(Workout.exercise, func.count())
            .group_by(Workout.exercise)
            .order_by(desc(func.count()))
            .limit(limit)
        )
        rows = result.all()
    return [ExerciseCountOut(exercise=e or "", count=int(c)) for e, c in rows]


# -----------------------------------------------------------------------------
# Workouts — single set (with duplicate detection)
# -----------------------------------------------------------------------------
@app.post("/workouts", response_model=GenericResponse)
async def add_workout(
    w: WorkoutIn,
    force: bool = Query(False, description="Skip duplicate check"),
) -> GenericResponse:
    async with async_session() as s:
        if not force:
            dups = await _check_duplicates(s, w)
            if dups:
                raise HTTPException(
                    409,
                    detail=f"Possible duplicate. Existing IDs: {dups}. Use force=true to save anyway.",
                )
        obj = Workout(**w.model_dump())
        s.add(obj)
        await s.commit()
    return GenericResponse(message="Workout saved")


# -----------------------------------------------------------------------------
# Workouts — bulk (with duplicate detection)
# -----------------------------------------------------------------------------
@app.post("/workouts/bulk", response_model=BulkWorkoutOut)
async def add_workouts_bulk(
    body: BulkWorkoutIn,
    force: bool = Query(False, description="Skip duplicate check"),
) -> BulkWorkoutOut:
    if not body.workouts:
        raise HTTPException(400, "Empty workout list")
    ids: List[int] = []
    async with async_session() as s:
        if not force:
            all_dups: List[int] = []
            for w in body.workouts:
                dups = await _check_duplicates(s, w)
                all_dups.extend(dups)
            if all_dups:
                raise HTTPException(
                    409,
                    detail=f"Possible duplicates found. Existing IDs: {all_dups}. Use force=true to save anyway.",
                )
        for w in body.workouts:
            obj = Workout(**w.model_dump())
            s.add(obj)
            await s.flush()
            ids.append(obj.id)
        await s.commit()
    return BulkWorkoutOut(saved=len(ids), ids=ids)


# -----------------------------------------------------------------------------
# Search workouts (safe LIKE)
# IMPORTANT: define /workouts/search BEFORE any dynamic /workouts/{...}
# -----------------------------------------------------------------------------
@app.get("/workouts/search", response_model=List[WorkoutOut])
async def search_workouts(
    q: str = Query(..., min_length=1, description="exercise name or part of it"),
    cycle: Optional[int] = Query(None, ge=0),
    week: Optional[int] = Query(None, ge=0),
    day: Optional[int] = Query(None, ge=0),
    iso_week: Optional[int] = Query(None, ge=0),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
    tag: Optional[str] = Query(None, description="filter by tag"),
) -> List[WorkoutOut]:
    tokens = [t for t in q.strip().replace(",", " ").split() if t]
    if not tokens:
        return []
    async with async_session() as s:
        cond = _safe_like(Workout.exercise, q)
        for t in tokens:
            cond = cond | _safe_like(Workout.exercise, t)
        stmt = select(Workout).where(cond)
        if cycle is not None:
            stmt = stmt.where(Workout.cycle == cycle)
        if week is not None:
            stmt = stmt.where(Workout.week == week)
        if day is not None:
            stmt = stmt.where(Workout.day == day)
        if iso_week is not None:
            stmt = stmt.where(Workout.iso_week == iso_week)
        if start:
            stmt = stmt.where(Workout.date >= start)
        if end:
            stmt = stmt.where(Workout.date <= end)
        if tag:
            stmt = stmt.where(_safe_like(Workout.tags, tag))
        stmt = stmt.order_by(asc(Workout.date), asc(Workout.id)).limit(500)
        result = await s.execute(stmt)
        rows = result.scalars().all()
    return [_row_to_out(w) for w in rows]


@app.get("/workouts", response_model=List[WorkoutOut])
async def query_workouts(
    exercise: Optional[str] = None,
    cycle: Optional[int] = None,
    week: Optional[int] = None,
    iso_week: Optional[int] = None,
    day: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    tag: Optional[str] = None,
) -> List[WorkoutOut]:
    stmt = select(Workout)
    if exercise:
        stmt = stmt.where(_safe_like(Workout.exercise, exercise))
    if cycle is not None:
        stmt = stmt.where(Workout.cycle == cycle)
    if week is not None:
        stmt = stmt.where(Workout.week == week)
    if iso_week is not None:
        stmt = stmt.where(Workout.iso_week == iso_week)
    if day is not None:
        stmt = stmt.where(Workout.day == day)
    if start:
        stmt = stmt.where(Workout.date >= start)
    if end:
        stmt = stmt.where(Workout.date <= end)
    if tag:
        stmt = stmt.where(_safe_like(Workout.tags, tag))
    stmt = stmt.order_by(asc(Workout.date), asc(Workout.id))
    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()
    return [_row_to_out(w) for w in rows]


@app.get("/workouts/by_cwd", response_model=List[WorkoutOut])
async def workouts_by_cycle_week_day(
    cycle: int = Query(..., ge=0),
    week: int = Query(..., ge=0),
    day: int = Query(..., ge=0),
) -> List[WorkoutOut]:
    async with async_session() as s:
        result = await s.execute(
            select(Workout)
            .where(Workout.cycle == cycle, Workout.week == week, Workout.day == day)
            .order_by(asc(Workout.date), asc(Workout.id))
        )
        rows = result.scalars().all()
    return [_row_to_out(w) for w in rows]


@app.get("/workouts/last", response_model=WorkoutOut)
async def last_workout(exercise: str = Query(..., min_length=1)) -> WorkoutOut:
    async with async_session() as s:
        result = await s.execute(
            select(Workout)
            .where(_safe_like(Workout.exercise, exercise))
            .order_by(desc(Workout.date), desc(Workout.id))
            .limit(1)
        )
        row = result.scalar()
        if not row:
            raise HTTPException(404, "No workout found for that exercise")
        return _row_to_out(row)


# IMPORTANT: /workouts/undo_last must be defined BEFORE /workouts/{workout_id}
@app.delete("/workouts/undo_last", response_model=UndoOut)
async def undo_last(
    count: int = Query(..., ge=1, le=50, description="Number of most recent rows to delete"),
) -> UndoOut:
    async with async_session() as s:
        result = await s.execute(
            select(Workout).order_by(desc(Workout.id)).limit(count)
        )
        rows = result.scalars().all()
        if not rows:
            raise HTTPException(404, "No workouts to undo")
        ids = [w.id for w in rows]
        for w in rows:
            await s.delete(w)
        await s.commit()
    return UndoOut(deleted=len(ids), ids=ids)


@app.put("/workouts/{workout_id}", response_model=WorkoutOut)
async def edit_workout(
    workout_id: int = FPath(..., ge=1), body: WorkoutIn = Body(...)
) -> WorkoutOut:
    async with async_session() as s:
        w = await s.get(Workout, workout_id)
        if not w:
            raise HTTPException(404, "Workout not found")
        for k, v in body.model_dump().items():
            setattr(w, k, v)
        await s.commit()
        await s.refresh(w)
        return _row_to_out(w)


@app.delete("/workouts/{workout_id}", response_model=GenericResponse)
async def delete_workout(workout_id: int = FPath(..., ge=1)) -> GenericResponse:
    async with async_session() as s:
        w = await s.get(Workout, workout_id)
        if not w:
            raise HTTPException(404, "Workout not found")
        await s.delete(w)
        await s.commit()
    return GenericResponse(message="Workout deleted")


# -----------------------------------------------------------------------------
# Verification feedback loop
# -----------------------------------------------------------------------------
@app.get("/workouts/verify", response_model=VerifyOut)
async def verify_workouts(
    date: str = Query(..., description="YYYY-MM-DD"),
    exercise: Optional[str] = Query(None),
    expected_sets: Optional[int] = Query(None, ge=1),
    cycle: Optional[int] = Query(None, ge=0),
    week: Optional[int] = Query(None, ge=0),
    day: Optional[int] = Query(None, ge=0),
) -> VerifyOut:
    """Verify logged workouts. GPT calls this after every save to confirm data."""
    stmt = select(Workout).where(Workout.date == date)
    if exercise:
        stmt = stmt.where(_safe_like(Workout.exercise, exercise))
    if cycle is not None:
        stmt = stmt.where(Workout.cycle == cycle)
    if week is not None:
        stmt = stmt.where(Workout.week == week)
    if day is not None:
        stmt = stmt.where(Workout.day == day)
    stmt = stmt.order_by(asc(Workout.set_number), asc(Workout.id))
    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()
    actual = len(rows)
    match = expected_sets == actual if expected_sets is not None else actual > 0
    return VerifyOut(
        date=date,
        exercise=exercise,
        expected_sets=expected_sets,
        actual_sets=actual,
        match=match,
        logged=[_row_to_out(w) for w in rows],
    )


# -----------------------------------------------------------------------------
# Stats & Analytics
# -----------------------------------------------------------------------------
@app.get("/stats", response_model=StatsOut)
async def stats() -> StatsOut:
    async with async_session() as s:
        total_result = await s.execute(text("SELECT COUNT(*) FROM workout"))
        total = total_result.scalar_one()
        last_result = await s.execute(
            select(Workout).order_by(desc(Workout.date), desc(Workout.id)).limit(1)
        )
        last_row = last_result.scalar()
        bests_result = await s.execute(
            select(Workout.exercise, func.max(Workout.value))
            .where(Workout.value.is_not(None))
            .group_by(Workout.exercise)
        )
        bests_rows = bests_result.all()
    bests = {e: float(v) for e, v in bests_rows if e}
    last = _row_to_out(last_row) if last_row else None
    return StatsOut(total_sessions=int(total), bests=bests, last=last)


@app.get("/analytics/prs", response_model=PRsOut)
async def analytics_prs(exercise: Optional[str] = None) -> PRsOut:
    async with async_session() as s:
        stmt = select(Workout.exercise, func.max(Workout.value)).where(
            Workout.value.is_not(None)
        )
        if exercise:
            stmt = stmt.where(func.lower(Workout.exercise) == exercise.lower())
        stmt = stmt.group_by(Workout.exercise)
        result = await s.execute(stmt)
        rows = result.all()
    prs = {e: float(v) for e, v in rows if e}
    return PRsOut(prs=prs)


@app.get("/analytics/weekly_summary", response_model=WeeklySummaryOut)
async def analytics_weekly_summary(
    cycle: int, week: Optional[int] = None
) -> WeeklySummaryOut:
    async with async_session() as s:
        stmt = select(Workout.day, func.count()).where(Workout.cycle == cycle)
        if week is not None:
            stmt = stmt.where(Workout.week == week)
        stmt = stmt.group_by(Workout.day)
        result = await s.execute(stmt)
        rows = result.all()

        total_stmt = select(func.count()).select_from(Workout).where(Workout.cycle == cycle)
        if week is not None:
            total_stmt = total_stmt.where(Workout.week == week)
        total_result = await s.execute(total_stmt)
        total = total_result.scalar_one()
    days = {int(d): int(c) for d, c in rows if d is not None}
    return WeeklySummaryOut(
        cycle=cycle, week=week, days_logged=days, total_sets=int(total)
    )


@app.get("/analytics/progress_compare", response_model=ProgressCompareOut)
async def analytics_progress_compare(
    exercise: str, cycle: int, week1: int, week2: int
) -> ProgressCompareOut:
    async with async_session() as s:
        r1 = await s.execute(
            select(func.count()).select_from(Workout).where(
                Workout.cycle == cycle,
                Workout.week == week1,
                func.lower(Workout.exercise) == exercise.lower(),
            )
        )
        w1 = r1.scalar_one()
        r2 = await s.execute(
            select(func.count()).select_from(Workout).where(
                Workout.cycle == cycle,
                Workout.week == week2,
                func.lower(Workout.exercise) == exercise.lower(),
            )
        )
        w2 = r2.scalar_one()
    return ProgressCompareOut(
        exercise=exercise,
        cycle=cycle,
        week1=week1,
        week2=week2,
        week1_sets=int(w1),
        week2_sets=int(w2),
        delta=int(w2) - int(w1),
    )


@app.get("/analytics/repmax", response_model=RepmaxOut)
async def analytics_repmax(exercise: str, mode: str = "max") -> RepmaxOut:
    allowed = {"max", "min", "auto"}
    if mode not in allowed:
        raise HTTPException(400, f"mode must be one of {sorted(allowed)}")

    async with async_session() as s:
        result = await s.execute(
            select(Workout).where(
                func.lower(Workout.exercise) == exercise.lower(),
                Workout.value.is_not(None),
            )
        )
        rows = result.scalars().all()

    if not rows:
        return RepmaxOut(exercise=exercise)

    effective = mode
    if mode == "auto":
        timey = {"sec", "s", "seconds", "min", "minutes"}
        effective = (
            "min"
            if any((w.unit or "").strip().lower() in timey for w in rows)
            else "max"
        )

    key = lambda w: float(w.value)  # noqa: E731
    best = max(rows, key=key) if effective == "max" else min(rows, key=key)

    return RepmaxOut(
        exercise=exercise,
        best_value=float(best.value),
        unit=best.unit,
        workout_id=best.id,
    )


@app.get("/analytics/consistency", response_model=ConsistencyOut)
async def analytics_consistency(cycle: Optional[int] = None) -> ConsistencyOut:
    REQUIRED_DAYS = {1, 2, 3, 4}
    async with async_session() as s:
        stmt = select(Workout.cycle, Workout.week).group_by(
            Workout.cycle, Workout.week
        )
        if cycle is not None:
            stmt = stmt.where(Workout.cycle == cycle)
        result = await s.execute(stmt)
        weeks = result.all()

        full: List[int] = []
        for cyc, wk in weeks:
            if cyc is None or wk is None:
                continue
            days_result = await s.execute(
                select(Workout.day).where(
                    Workout.cycle == cyc, Workout.week == wk
                )
            )
            days = {d for (d,) in days_result.all() if d is not None}
            if REQUIRED_DAYS.issubset(days):
                full.append(int(wk))
    return ConsistencyOut(weeks_completed_100=sorted(full))


# -----------------------------------------------------------------------------
# 1RM Calculator (Epley & Brzycki formulas)
# -----------------------------------------------------------------------------
@app.get("/analytics/estimated_1rm", response_model=OneRMOut)
async def estimated_1rm(
    exercise: str,
    formula: str = Query("epley", description="epley or brzycki"),
) -> OneRMOut:
    if formula not in ("epley", "brzycki"):
        raise HTTPException(400, "formula must be 'epley' or 'brzycki'")

    async with async_session() as s:
        result = await s.execute(
            select(Workout).where(
                func.lower(Workout.exercise) == exercise.lower(),
                Workout.value.is_not(None),
                Workout.reps.is_not(None),
                Workout.reps > 0,
            )
        )
        rows = result.scalars().all()

    if not rows:
        return OneRMOut(exercise=exercise, formula=formula)

    best_1rm = 0.0
    best_row = None
    for w in rows:
        weight = float(w.value)
        reps = int(w.reps)
        if reps == 1:
            e1rm = weight
        elif formula == "epley":
            e1rm = weight * (1 + reps / 30)
        else:  # brzycki
            e1rm = weight * (36 / (37 - reps)) if reps < 37 else weight
        if e1rm > best_1rm:
            best_1rm = e1rm
            best_row = w

    if best_row is None:
        return OneRMOut(exercise=exercise, formula=formula)

    return OneRMOut(
        exercise=exercise,
        estimated_1rm_kg=round(best_1rm, 1),
        based_on_value=float(best_row.value),
        based_on_reps=int(best_row.reps),
        formula=formula,
    )


# -----------------------------------------------------------------------------
# Volume Analytics
# -----------------------------------------------------------------------------
@app.get("/analytics/volume", response_model=WeeklyVolumeOut)
async def analytics_volume(
    cycle: Optional[int] = None,
    week: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> WeeklyVolumeOut:
    stmt = select(Workout).where(
        Workout.value.is_not(None), Workout.reps.is_not(None)
    )
    if cycle is not None:
        stmt = stmt.where(Workout.cycle == cycle)
    if week is not None:
        stmt = stmt.where(Workout.week == week)
    if start:
        stmt = stmt.where(Workout.date >= start)
    if end:
        stmt = stmt.where(Workout.date <= end)

    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()

    exercises: Dict[str, dict] = {}
    for w in rows:
        name = w.exercise
        if name not in exercises:
            exercises[name] = {
                "total_volume": 0.0,
                "total_sets": 0,
                "total_reps": 0,
                "unit": w.unit,
            }
        reps = int(w.reps) if w.reps else 0
        val = float(w.value) if w.value else 0.0
        exercises[name]["total_volume"] += val * reps
        exercises[name]["total_sets"] += 1
        exercises[name]["total_reps"] += reps

    exercise_list = [
        VolumeOut(exercise=name, **data) for name, data in sorted(exercises.items())
    ]
    grand_vol = sum(e.total_volume for e in exercise_list)
    grand_sets = sum(e.total_sets for e in exercise_list)

    return WeeklyVolumeOut(
        cycle=cycle,
        week=week,
        exercises=exercise_list,
        grand_total_volume=round(grand_vol, 1),
        grand_total_sets=grand_sets,
    )


# -----------------------------------------------------------------------------
# Exercise History Timeline
# -----------------------------------------------------------------------------
@app.get("/analytics/timeline", response_model=ExerciseTimelineOut)
async def exercise_timeline(
    exercise: str = Query(..., min_length=1),
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> ExerciseTimelineOut:
    stmt = select(Workout).where(
        func.lower(Workout.exercise) == exercise.lower(),
        Workout.value.is_not(None),
    )
    if start:
        stmt = stmt.where(Workout.date >= start)
    if end:
        stmt = stmt.where(Workout.date <= end)
    stmt = stmt.order_by(asc(Workout.date), asc(Workout.id))

    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()

    by_date: Dict[str, dict] = {}
    for w in rows:
        d = w.date
        if d not in by_date:
            by_date[d] = {"best": 0.0, "unit": w.unit, "sets": 0, "reps": 0}
        val = float(w.value)
        if val > by_date[d]["best"]:
            by_date[d]["best"] = val
            by_date[d]["unit"] = w.unit
        by_date[d]["sets"] += 1
        by_date[d]["reps"] += int(w.reps) if w.reps else 0

    timeline = [
        TimelinePointOut(
            date=d,
            best_value=data["best"],
            unit=data["unit"],
            total_sets=data["sets"],
            total_reps=data["reps"],
        )
        for d, data in sorted(by_date.items())
    ]

    return ExerciseTimelineOut(exercise=exercise, timeline=timeline)


# -----------------------------------------------------------------------------
# Search exercise (results only — no plans)
# -----------------------------------------------------------------------------
@app.get("/search_exercise", response_model=SearchExerciseOut)
async def search_exercise(
    exercise: str,
    cycle: Optional[int] = None,
    week: Optional[int] = None,
    iso_week: Optional[int] = None,
    day: Optional[int] = None,
    tag: Optional[str] = None,
) -> SearchExerciseOut:
    async with async_session() as s:
        stmt = select(Workout).where(_safe_like(Workout.exercise, exercise))
        if cycle is not None:
            stmt = stmt.where(Workout.cycle == cycle)
        if week is not None:
            stmt = stmt.where(Workout.week == week)
        if iso_week is not None:
            stmt = stmt.where(Workout.iso_week == iso_week)
        if day is not None:
            stmt = stmt.where(Workout.day == day)
        if tag:
            stmt = stmt.where(_safe_like(Workout.tags, tag))
        result = await s.execute(
            stmt.order_by(asc(Workout.date), asc(Workout.id))
        )
        rows = result.scalars().all()
        logged = [_row_to_out(w) for w in rows]
    return SearchExerciseOut(exercise=exercise, logged=logged)


# -----------------------------------------------------------------------------
# Export CSV
# -----------------------------------------------------------------------------
@app.get("/export/csv", response_model=CsvExportOut)
async def export_csv() -> CsvExportOut:
    async with async_session() as s:
        result = await s.execute(
            select(Workout).order_by(asc(Workout.date), asc(Workout.id))
        )
        rows = result.scalars().all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "date", "exercise", "set_number", "reps", "value", "unit",
        "rpe", "cycle", "week", "iso_week", "day", "notes", "tags",
    ])
    for w in rows:
        writer.writerow([
            w.id, w.date, w.exercise, w.set_number,
            w.reps, w.value, w.unit, w.rpe,
            w.cycle, w.week, w.iso_week, w.day,
            (w.notes or ""), (w.tags or ""),
        ])

    return CsvExportOut(filename="workouts.csv", rows=len(rows), csv=buf.getvalue())
