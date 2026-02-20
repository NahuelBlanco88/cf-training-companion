# app.py
# =============================================================================
# CF-Log API — Workouts & Analytics (FastAPI + SQLAlchemy 2.x async, Pydantic v2)
# Results-only: no plan storage. One row per working set for maximum precision.
# v13.0.0 — returns IDs on create, query limits, consistency fix, GPT-optimized
# =============================================================================

from __future__ import annotations

import csv
import io
import os
import re
import logging
import time
import traceback
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path as OSPath
from typing import AsyncGenerator, Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi import Path as FPath
from fastapi.responses import JSONResponse
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
# -----------------------------------------------------------------------------
_cloud_sql = os.getenv("CLOUD_SQL_CONNECTION_NAME")
_db_user = os.getenv("DB_USER", "postgres")
_db_pass = os.getenv("DB_PASSWORD", "")
_db_name = os.getenv("DB_NAME", "cf_log")

if _cloud_sql:
    _socket_path = f"/cloudsql/{_cloud_sql}"
    DB_PATH = f"postgresql+asyncpg://{_db_user}:{_db_pass}@/{_db_name}?host={_socket_path}"
    engine = create_async_engine(
        DB_PATH, echo=False, pool_pre_ping=True,
        pool_size=20, max_overflow=30, pool_timeout=30,
    )
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
    date: Mapped[str] = mapped_column(String, nullable=False)
    exercise: Mapped[str] = mapped_column(String, nullable=False)
    set_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    reps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    cycle: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    iso_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    # Legacy columns kept for backward compat
    rpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sets: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    plan_day_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)


class Metcon(Base):
    """A conditioning workout / benchmark / WOD result."""
    __tablename__ = "metcon"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String, nullable=False)           # YYYY-MM-DD
    name: Mapped[str] = mapped_column(String, nullable=False)           # "Fran", "Murph", or custom
    workout_type: Mapped[str] = mapped_column(String, nullable=False)   # for_time, amrap, emom, chipper, interval, other
    description: Mapped[Optional[str]] = mapped_column(String, nullable=True)  # the prescription
    score_time_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # for timed WODs
    score_rounds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)        # for AMRAP
    score_reps: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)          # extra reps in AMRAP
    score_display: Mapped[Optional[str]] = mapped_column(String, nullable=True)        # "3:45", "12+8"
    rx: Mapped[Optional[str]] = mapped_column(String, nullable=True)    # rx, scaled, rx_plus
    time_cap_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cycle: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    iso_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    tags: Mapped[Optional[str]] = mapped_column(String, nullable=True)


# -----------------------------------------------------------------------------
# Startup: create tables & run migrations
# -----------------------------------------------------------------------------
async def _init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    # Migrations for workout table
    # Each column uses its own transaction so a failed SELECT doesn't abort
    # the ALTER TABLE in PostgreSQL (PG aborts entire txn on any error).
    for col_name, col_sql in [
        ("set_number", "ALTER TABLE workout ADD COLUMN set_number INTEGER"),
        ("tags", "ALTER TABLE workout ADD COLUMN tags VARCHAR"),
        ("iso_week", "ALTER TABLE workout ADD COLUMN iso_week INTEGER"),
    ]:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"SELECT {col_name} FROM workout LIMIT 1"))
        except Exception:
            try:
                async with engine.begin() as conn:
                    await conn.execute(text(col_sql))
                    log.info(f"Added {col_name} column to workout table")
            except Exception as e:
                log.warning(f"Migration for {col_name} (workout): {e}")
    # Migrations for metcon table
    for col_name, col_sql in [
        ("iso_week", "ALTER TABLE metcon ADD COLUMN iso_week INTEGER"),
    ]:
        try:
            async with engine.begin() as conn:
                await conn.execute(text(f"SELECT {col_name} FROM metcon LIMIT 1"))
        except Exception:
            try:
                async with engine.begin() as conn:
                    await conn.execute(text(col_sql))
                    log.info(f"Added {col_name} column to metcon table")
            except Exception as e:
                log.warning(f"Migration for {col_name} (metcon): {e}")


# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_VALID_WORKOUT_TYPES = {"for_time", "amrap", "emom", "chipper", "interval", "other"}
_VALID_RX = {"rx", "scaled", "rx_plus"}


def _validate_date_str(v: str) -> str:
    if not _DATE_RE.match(v):
        raise ValueError("date must be YYYY-MM-DD format")
    try:
        datetime.strptime(v, "%Y-%m-%d")
    except ValueError:
        raise ValueError("date is not a valid calendar date")
    return v


def _normalize_tags_str(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    cleaned = ",".join(t.strip().lower() for t in v.split(",") if t.strip())
    return cleaned or None


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
    metcon_rows: int


class ExerciseCountOut(BaseModel):
    exercise: str
    count: int


# ── Workout schemas (RPE removed) ───────────────────────────────────────────

class WorkoutIn(BaseModel):
    """One working set. GPT must send one of these per set."""
    date: str
    exercise: str
    set_number: Optional[int] = None
    reps: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    notes: Optional[str] = ""
    tags: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        return _validate_date_str(v)

    @field_validator("exercise")
    @classmethod
    def normalize_exercise(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("exercise name cannot be empty")
        return v.title()

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_tags_str(v)


class WorkoutOut(BaseModel):
    id: int
    date: str
    exercise: str
    set_number: Optional[int] = None
    reps: Optional[int] = None
    value: Optional[float] = None
    unit: Optional[str] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    notes: Optional[str] = ""
    tags: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class BulkWorkoutIn(BaseModel):
    workouts: List[WorkoutIn]


class BulkWorkoutOut(BaseModel):
    saved: int
    ids: List[int]


# ── Metcon schemas ───────────────────────────────────────────────────────────

class MetconIn(BaseModel):
    """A conditioning/benchmark workout result."""
    date: str
    name: str                                      # "Fran", "Murph", "Tuesday Conditioning"
    workout_type: str                               # for_time, amrap, emom, chipper, interval, other
    description: Optional[str] = None               # "21-15-9 Thrusters & Pull-ups"
    score_time_seconds: Optional[int] = None        # total time in seconds (for timed WODs)
    score_rounds: Optional[int] = None              # completed rounds (AMRAP)
    score_reps: Optional[int] = None                # extra reps beyond last full round (AMRAP)
    score_display: Optional[str] = None             # human-readable: "3:45", "12+8", "185 reps"
    rx: Optional[str] = None                        # rx, scaled, rx_plus
    time_cap_seconds: Optional[int] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    notes: Optional[str] = ""
    tags: Optional[str] = None

    @field_validator("date")
    @classmethod
    def validate_date(cls, v: str) -> str:
        return _validate_date_str(v)

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("metcon name cannot be empty")
        return v.title()

    @field_validator("workout_type")
    @classmethod
    def validate_workout_type(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in _VALID_WORKOUT_TYPES:
            raise ValueError(f"workout_type must be one of {sorted(_VALID_WORKOUT_TYPES)}")
        return v

    @field_validator("rx")
    @classmethod
    def validate_rx(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in _VALID_RX:
            raise ValueError(f"rx must be one of {sorted(_VALID_RX)}")
        return v

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, v: Optional[str]) -> Optional[str]:
        return _normalize_tags_str(v)


class MetconOut(BaseModel):
    id: int
    date: str
    name: str
    workout_type: str
    description: Optional[str] = None
    score_time_seconds: Optional[int] = None
    score_rounds: Optional[int] = None
    score_reps: Optional[int] = None
    score_display: Optional[str] = None
    rx: Optional[str] = None
    time_cap_seconds: Optional[int] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None
    notes: Optional[str] = ""
    tags: Optional[str] = None
    model_config = ConfigDict(from_attributes=True)


class BulkMetconIn(BaseModel):
    metcons: List[MetconIn]


class BulkMetconOut(BaseModel):
    saved: int
    ids: List[int]


class MetconPROut(BaseModel):
    name: str
    workout_type: str
    best_time_seconds: Optional[int] = None
    best_time_display: Optional[str] = None
    best_rounds: Optional[int] = None
    best_reps: Optional[int] = None
    best_score_display: Optional[str] = None
    rx: Optional[str] = None
    date: Optional[str] = None
    metcon_id: Optional[int] = None


class MetconTimelinePointOut(BaseModel):
    date: str
    score_time_seconds: Optional[int] = None
    score_rounds: Optional[int] = None
    score_reps: Optional[int] = None
    score_display: Optional[str] = None
    rx: Optional[str] = None


class MetconTimelineOut(BaseModel):
    name: str
    workout_type: Optional[str] = None
    timeline: List[MetconTimelinePointOut] = Field(default_factory=list)


class BenchmarkListOut(BaseModel):
    benchmarks: List[MetconPROut] = Field(default_factory=list)


# ── Analytics schemas ────────────────────────────────────────────────────────

class StatsOut(BaseModel):
    total_sessions: int
    total_metcons: int
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


class ConsistencyWeekOut(BaseModel):
    cycle: int
    week: int


class ConsistencyOut(BaseModel):
    completed_weeks: List[ConsistencyWeekOut] = Field(default_factory=list)
    total: int = 0


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


class DaySummaryOut(BaseModel):
    """Everything logged for a specific cycle/week/day — workouts + metcons."""
    cycle: int
    week: int
    day: int
    workouts: List[WorkoutOut] = Field(default_factory=list)
    metcons: List[MetconOut] = Field(default_factory=list)
    total_workout_sets: int = 0
    total_metcons: int = 0


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    await _init_db()
    yield
    await engine.dispose()


app = FastAPI(
    title="CF-Log API",
    description="CrossFit Training Log & Analytics API. Strength sets + metcon/benchmark tracking.",
    version="14.3.0",
    lifespan=lifespan,
)


# -----------------------------------------------------------------------------
# Global exception handler — log full traceback so Cloud Run logs show the cause
# -----------------------------------------------------------------------------
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    tb = traceback.format_exc()
    log.error(f"Unhandled error on {request.method} {request.url.path}: {exc}\n{tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {exc}"},
    )


# -----------------------------------------------------------------------------
# Rate limiting middleware
# -----------------------------------------------------------------------------
_rate_limit_store: Dict[str, List[float]] = defaultdict(list)
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "300"))
RATE_LIMIT_WINDOW = int(os.getenv("RATE_LIMIT_WINDOW", "60"))


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
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
    # Prune stale IPs to prevent memory leak
    if len(_rate_limit_store) > 1000:
        stale = [ip for ip, ts in _rate_limit_store.items()
                 if not ts or ts[-1] < window_start]
        for ip in stale:
            del _rate_limit_store[ip]
    response = await call_next(request)
    response.headers["X-RateLimit-Limit"] = str(RATE_LIMIT_REQUESTS)
    response.headers["X-RateLimit-Remaining"] = str(
        RATE_LIMIT_REQUESTS - len(_rate_limit_store[client_ip])
    )
    return response


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
_LIKE_ESCAPE_CHAR = "!"  # Use ! instead of \ to avoid PG backslash issues


def _safe_int(val) -> int | None:
    """Parse a reps value that may be '1+1' sum notation, plain int, or None."""
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    if '+' in s:
        try:
            return sum(int(p) for p in s.split('+'))
        except ValueError:
            pass
    return None


def _escape_like(s: str) -> str:
    return s.replace("!", "!!").replace("%", "!%").replace("_", "!_")


def _safe_like(column, term: str):
    escaped = _escape_like(term.strip().lower())
    pattern = f"%{escaped}%"
    return func.lower(column).like(pattern, escape=_LIKE_ESCAPE_CHAR)


def _row_to_out(w: Workout) -> WorkoutOut:
    return WorkoutOut(
        id=w.id, date=w.date, exercise=w.exercise, set_number=w.set_number,
        reps=_safe_int(w.reps),
        value=w.value, unit=w.unit, cycle=w.cycle, week=w.week,
        day=w.day, notes=w.notes, tags=w.tags,
    )


def _metcon_to_out(m: Metcon) -> MetconOut:
    return MetconOut(
        id=m.id, date=m.date, name=m.name, workout_type=m.workout_type,
        description=m.description, score_time_seconds=m.score_time_seconds,
        score_rounds=m.score_rounds, score_reps=m.score_reps,
        score_display=m.score_display, rx=m.rx,
        time_cap_seconds=m.time_cap_seconds, cycle=m.cycle, week=m.week,
        day=m.day, notes=m.notes, tags=m.tags,
    )


def _seconds_to_display(secs: int) -> str:
    if secs < 3600:
        return f"{secs // 60}:{secs % 60:02d}"
    h = secs // 3600
    remainder = secs % 3600
    return f"{h}:{remainder // 60:02d}:{remainder % 60:02d}"


def _db_type() -> str:
    if _cloud_sql:
        return f"Cloud SQL PostgreSQL ({_cloud_sql})"
    return "SQLite"


# -----------------------------------------------------------------------------
# Duplicate detection
# -----------------------------------------------------------------------------
async def _check_duplicates(session: AsyncSession, w: WorkoutIn) -> List[int]:
    if w.set_number is None:
        return []
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


# =============================================================================
# ENDPOINTS — Health / Root / Debug
# =============================================================================
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
        ok=db_connected, db_connected=db_connected,
        db_type=_db_type(), timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get("/", response_model=GenericResponse)
async def root() -> GenericResponse:
    return GenericResponse(message="CF-Log API v14.3 is running")


@app.get("/debug/dbinfo", response_model=DBInfoOut)
async def dbinfo() -> DBInfoOut:
    async with async_session() as s:
        wr = await s.execute(text("SELECT COUNT(*) FROM workout"))
        workout_rows = wr.scalar_one()
        try:
            mr = await s.execute(text("SELECT COUNT(*) FROM metcon"))
            metcon_rows = mr.scalar_one()
        except Exception:
            metcon_rows = 0
    return DBInfoOut(db_type=_db_type(), workout_rows=int(workout_rows), metcon_rows=int(metcon_rows))


@app.get("/debug/exercises", response_model=List[ExerciseCountOut])
async def debug_exercises(
    q: Optional[str] = Query(None, description="Filter by exercise name"),
    limit: int = Query(50, ge=1, le=500),
) -> List[ExerciseCountOut]:
    async with async_session() as s:
        stmt = (
            select(Workout.exercise, func.count().label("cnt"))
            .group_by(Workout.exercise)
        )
        if q:
            stmt = stmt.where(_safe_like(Workout.exercise, q))
        stmt = stmt.order_by(desc(func.count())).limit(limit)
        result = await s.execute(stmt)
        rows = result.all()
    return [ExerciseCountOut(exercise=e or "", count=int(c)) for e, c in rows]


# =============================================================================
# ENDPOINTS — Workouts (strength sets)
# =============================================================================
@app.post("/workouts", response_model=WorkoutOut)
async def add_workout(
    w: WorkoutIn, force: bool = Query(False, description="Skip duplicate check"),
) -> WorkoutOut:
    async with async_session() as s:
        if not force:
            dups = await _check_duplicates(s, w)
            if dups:
                raise HTTPException(409, detail=f"Possible duplicate. Existing IDs: {dups}. Use force=true to save anyway.")
        obj = Workout(**w.model_dump())
        s.add(obj)
        await s.flush()
        result = _row_to_out(obj)
        await s.commit()
    return result


@app.post("/workouts/bulk", response_model=BulkWorkoutOut)
async def add_workouts_bulk(
    body: BulkWorkoutIn, force: bool = Query(False, description="Skip duplicate check"),
) -> BulkWorkoutOut:
    if not body.workouts:
        raise HTTPException(400, "Empty workout list")
    ids: List[int] = []
    async with async_session() as s:
        if not force:
            all_dups: List[int] = []
            for w in body.workouts:
                all_dups.extend(await _check_duplicates(s, w))
            if all_dups:
                raise HTTPException(409, detail=f"Possible duplicates. Existing IDs: {all_dups}. Use force=true to save anyway.")
        for w in body.workouts:
            obj = Workout(**w.model_dump())
            s.add(obj)
            await s.flush()
            ids.append(obj.id)
        await s.commit()
    return BulkWorkoutOut(saved=len(ids), ids=ids)


# IMPORTANT: static paths before /workouts/{workout_id}
@app.get("/workouts/search", response_model=List[WorkoutOut])
async def search_workouts(
    q: str = Query(..., min_length=1, description="exercise name or part of it"),
    cycle: Optional[int] = Query(None, ge=0), week: Optional[int] = Query(None, ge=0),
    day: Optional[int] = Query(None, ge=0),
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
        if cycle is not None: stmt = stmt.where(Workout.cycle == cycle)
        if week is not None: stmt = stmt.where(Workout.week == week)
        if day is not None: stmt = stmt.where(Workout.day == day)
        if start: stmt = stmt.where(Workout.date >= start)
        if end: stmt = stmt.where(Workout.date <= end)
        if tag: stmt = stmt.where(_safe_like(Workout.tags, tag))
        stmt = stmt.order_by(asc(Workout.date), asc(Workout.id)).limit(500)
        result = await s.execute(stmt)
        rows = result.scalars().all()
    return [_row_to_out(w) for w in rows]


@app.get("/workouts", response_model=List[WorkoutOut])
async def query_workouts(
    exercise: Optional[str] = None, cycle: Optional[int] = None,
    week: Optional[int] = None,
    day: Optional[int] = None, start: Optional[str] = None,
    end: Optional[str] = None, tag: Optional[str] = None,
    limit: int = Query(200, ge=1, le=500, description="Max rows to return"),
) -> List[WorkoutOut]:
    stmt = select(Workout)
    if exercise: stmt = stmt.where(_safe_like(Workout.exercise, exercise))
    if cycle is not None: stmt = stmt.where(Workout.cycle == cycle)
    if week is not None: stmt = stmt.where(Workout.week == week)
    if day is not None: stmt = stmt.where(Workout.day == day)
    if start: stmt = stmt.where(Workout.date >= start)
    if end: stmt = stmt.where(Workout.date <= end)
    if tag: stmt = stmt.where(_safe_like(Workout.tags, tag))
    stmt = stmt.order_by(asc(Workout.date), asc(Workout.id)).limit(limit)
    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()
    return [_row_to_out(w) for w in rows]


@app.get("/workouts/by_cwd", response_model=List[WorkoutOut])
async def workouts_by_cycle_week_day(
    cycle: int = Query(..., ge=0), week: int = Query(..., ge=0), day: int = Query(..., ge=0),
) -> List[WorkoutOut]:
    async with async_session() as s:
        result = await s.execute(
            select(Workout).where(Workout.cycle == cycle, Workout.week == week, Workout.day == day)
            .order_by(asc(Workout.date), asc(Workout.id))
        )
        rows = result.scalars().all()
    return [_row_to_out(w) for w in rows]


@app.get("/workouts/last", response_model=WorkoutOut)
async def last_workout(exercise: str = Query(..., min_length=1)) -> WorkoutOut:
    async with async_session() as s:
        result = await s.execute(
            select(Workout).where(_safe_like(Workout.exercise, exercise))
            .order_by(desc(Workout.date), desc(Workout.id)).limit(1)
        )
        row = result.scalar()
        if not row:
            raise HTTPException(404, "No workout found for that exercise")
        return _row_to_out(row)


async def _do_verify(date: str, exercise: Optional[str], expected_sets: Optional[int],
                     cycle: Optional[int], week: Optional[int], day: Optional[int]) -> VerifyOut:
    stmt = select(Workout).where(Workout.date == date)
    if exercise: stmt = stmt.where(_safe_like(Workout.exercise, exercise))
    if cycle is not None: stmt = stmt.where(Workout.cycle == cycle)
    if week is not None: stmt = stmt.where(Workout.week == week)
    if day is not None: stmt = stmt.where(Workout.day == day)
    stmt = stmt.order_by(asc(Workout.set_number), asc(Workout.id))
    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()
    actual = len(rows)
    match = expected_sets == actual if expected_sets is not None else actual > 0
    return VerifyOut(date=date, exercise=exercise, expected_sets=expected_sets,
                     actual_sets=actual, match=match, logged=[_row_to_out(w) for w in rows])


@app.get("/workouts/verify", response_model=VerifyOut)
async def verify_workouts(
    date: str = Query(..., description="YYYY-MM-DD"),
    exercise: Optional[str] = Query(None),
    expected_sets: Optional[int] = Query(None, ge=1),
    cycle: Optional[int] = Query(None, ge=0),
    week: Optional[int] = Query(None, ge=0),
    day: Optional[int] = Query(None, ge=0),
) -> VerifyOut:
    return await _do_verify(date, exercise, expected_sets, cycle, week, day)


class VerifyIn(BaseModel):
    date: str
    exercise: Optional[str] = None
    expected_sets: Optional[int] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    day: Optional[int] = None


@app.post("/workouts/verify", response_model=VerifyOut)
async def verify_workouts_post(body: VerifyIn = Body(...)) -> VerifyOut:
    return await _do_verify(body.date, body.exercise, body.expected_sets,
                            body.cycle, body.week, body.day)


# undo_last MUST be before {workout_id}
@app.delete("/workouts/undo_last", response_model=UndoOut)
async def undo_last(
    count: int = Query(..., ge=1, le=50, description="Number of most recent rows to delete"),
) -> UndoOut:
    async with async_session() as s:
        result = await s.execute(select(Workout).order_by(desc(Workout.id)).limit(count))
        rows = result.scalars().all()
        if not rows:
            raise HTTPException(404, "No workouts to undo")
        ids = [w.id for w in rows]
        for w in rows:
            await s.delete(w)
        await s.commit()
    return UndoOut(deleted=len(ids), ids=ids)


@app.put("/workouts/{workout_id}", response_model=WorkoutOut)
async def edit_workout(workout_id: int = FPath(..., ge=1), body: WorkoutIn = Body(...)) -> WorkoutOut:
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


# =============================================================================
# ENDPOINTS — Metcons / Conditioning / Benchmarks
# =============================================================================
@app.post("/metcons", response_model=MetconOut)
async def add_metcon(m: MetconIn) -> MetconOut:
    async with async_session() as s:
        obj = Metcon(**m.model_dump())
        s.add(obj)
        await s.flush()
        result = _metcon_to_out(obj)
        await s.commit()
    return result


@app.post("/metcons/bulk", response_model=BulkMetconOut)
async def add_metcons_bulk(body: BulkMetconIn) -> BulkMetconOut:
    if not body.metcons:
        raise HTTPException(400, "Empty metcon list")
    ids: List[int] = []
    async with async_session() as s:
        for m in body.metcons:
            obj = Metcon(**m.model_dump())
            s.add(obj)
            await s.flush()
            ids.append(obj.id)
        await s.commit()
    return BulkMetconOut(saved=len(ids), ids=ids)


@app.get("/metcons/search", response_model=List[MetconOut])
async def search_metcons(
    q: str = Query(..., min_length=1, description="metcon name or part of it"),
    workout_type: Optional[str] = Query(None), rx: Optional[str] = Query(None),
    start: Optional[str] = Query(None), end: Optional[str] = Query(None),
    tag: Optional[str] = Query(None),
) -> List[MetconOut]:
    async with async_session() as s:
        stmt = select(Metcon).where(_safe_like(Metcon.name, q))
        if workout_type: stmt = stmt.where(Metcon.workout_type == workout_type.lower())
        if rx: stmt = stmt.where(Metcon.rx == rx.lower())
        if start: stmt = stmt.where(Metcon.date >= start)
        if end: stmt = stmt.where(Metcon.date <= end)
        if tag: stmt = stmt.where(_safe_like(Metcon.tags, tag))
        stmt = stmt.order_by(asc(Metcon.date), asc(Metcon.id)).limit(500)
        result = await s.execute(stmt)
        rows = result.scalars().all()
    return [_metcon_to_out(m) for m in rows]


@app.get("/metcons", response_model=List[MetconOut])
async def query_metcons(
    name: Optional[str] = None, workout_type: Optional[str] = None,
    rx: Optional[str] = None, cycle: Optional[int] = None,
    week: Optional[int] = None, day: Optional[int] = None,
    start: Optional[str] = None, end: Optional[str] = None,
    tag: Optional[str] = None,
    limit: int = Query(200, ge=1, le=500, description="Max rows to return"),
) -> List[MetconOut]:
    stmt = select(Metcon)
    if name: stmt = stmt.where(_safe_like(Metcon.name, name))
    if workout_type: stmt = stmt.where(Metcon.workout_type == workout_type.lower())
    if rx: stmt = stmt.where(Metcon.rx == rx.lower())
    if cycle is not None: stmt = stmt.where(Metcon.cycle == cycle)
    if week is not None: stmt = stmt.where(Metcon.week == week)
    if day is not None: stmt = stmt.where(Metcon.day == day)
    if start: stmt = stmt.where(Metcon.date >= start)
    if end: stmt = stmt.where(Metcon.date <= end)
    if tag: stmt = stmt.where(_safe_like(Metcon.tags, tag))
    stmt = stmt.order_by(asc(Metcon.date), asc(Metcon.id)).limit(limit)
    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()
    return [_metcon_to_out(m) for m in rows]


@app.get("/metcons/last", response_model=MetconOut)
async def last_metcon(name: str = Query(..., min_length=1)) -> MetconOut:
    async with async_session() as s:
        result = await s.execute(
            select(Metcon).where(_safe_like(Metcon.name, name))
            .order_by(desc(Metcon.date), desc(Metcon.id)).limit(1)
        )
        row = result.scalar()
        if not row:
            raise HTTPException(404, "No metcon found with that name")
        return _metcon_to_out(row)


@app.put("/metcons/{metcon_id}", response_model=MetconOut)
async def edit_metcon(metcon_id: int = FPath(..., ge=1), body: MetconIn = Body(...)) -> MetconOut:
    async with async_session() as s:
        m = await s.get(Metcon, metcon_id)
        if not m:
            raise HTTPException(404, "Metcon not found")
        for k, v in body.model_dump().items():
            setattr(m, k, v)
        await s.commit()
        await s.refresh(m)
        return _metcon_to_out(m)


@app.delete("/metcons/{metcon_id}", response_model=GenericResponse)
async def delete_metcon(metcon_id: int = FPath(..., ge=1)) -> GenericResponse:
    async with async_session() as s:
        m = await s.get(Metcon, metcon_id)
        if not m:
            raise HTTPException(404, "Metcon not found")
        await s.delete(m)
        await s.commit()
    return GenericResponse(message="Metcon deleted")


# ── Metcon Analytics ─────────────────────────────────────────────────────────

@app.get("/analytics/metcon_prs", response_model=BenchmarkListOut)
async def metcon_prs(name: Optional[str] = None, rx: Optional[str] = None) -> BenchmarkListOut:
    """Best scores for each benchmark. For timed WODs = fastest, for AMRAP = highest rounds+reps."""
    async with async_session() as s:
        stmt = select(Metcon.name).group_by(Metcon.name)
        if name: stmt = stmt.where(_safe_like(Metcon.name, name))
        name_result = await s.execute(stmt)
        names = [n for (n,) in name_result.all()]

        prs: List[MetconPROut] = []
        for n in names:
            q = select(Metcon).where(func.lower(Metcon.name) == n.lower())
            if rx: q = q.where(Metcon.rx == rx.lower())
            result = await s.execute(q)
            rows = result.scalars().all()
            if not rows:
                continue

            wt = rows[0].workout_type
            best = None

            if wt == "for_time":
                timed = [r for r in rows if r.score_time_seconds is not None]
                if timed:
                    best = min(timed, key=lambda r: r.score_time_seconds)
            elif wt == "amrap":
                amrap = [r for r in rows if r.score_rounds is not None]
                if amrap:
                    best = max(amrap, key=lambda r: (r.score_rounds or 0, r.score_reps or 0))
            else:
                timed = [r for r in rows if r.score_time_seconds is not None]
                if timed:
                    best = min(timed, key=lambda r: r.score_time_seconds)
                elif rows:
                    best = rows[-1]

            if best:
                prs.append(MetconPROut(
                    name=n, workout_type=wt,
                    best_time_seconds=best.score_time_seconds,
                    best_time_display=_seconds_to_display(best.score_time_seconds) if best.score_time_seconds else None,
                    best_rounds=best.score_rounds, best_reps=best.score_reps,
                    best_score_display=best.score_display, rx=best.rx,
                    date=best.date, metcon_id=best.id,
                ))
    return BenchmarkListOut(benchmarks=prs)


@app.get("/analytics/metcon_timeline", response_model=MetconTimelineOut)
async def metcon_timeline(
    name: str = Query(..., min_length=1),
    rx: Optional[str] = None,
    start: Optional[str] = None, end: Optional[str] = None,
) -> MetconTimelineOut:
    """History of a named metcon over time — for tracking progress on benchmarks."""
    stmt = select(Metcon).where(_safe_like(Metcon.name, name))
    if rx: stmt = stmt.where(Metcon.rx == rx.lower())
    if start: stmt = stmt.where(Metcon.date >= start)
    if end: stmt = stmt.where(Metcon.date <= end)
    stmt = stmt.order_by(asc(Metcon.date), asc(Metcon.id))
    async with async_session() as s:
        result = await s.execute(stmt)
        rows = result.scalars().all()
    wt = rows[0].workout_type if rows else None
    timeline = [
        MetconTimelinePointOut(
            date=m.date, score_time_seconds=m.score_time_seconds,
            score_rounds=m.score_rounds, score_reps=m.score_reps,
            score_display=m.score_display, rx=m.rx,
        ) for m in rows
    ]
    return MetconTimelineOut(name=name, workout_type=wt, timeline=timeline)


# =============================================================================
# ENDPOINTS — Workout Analytics (unchanged except RPE removed)
# =============================================================================
@app.get("/stats", response_model=StatsOut)
async def stats() -> StatsOut:
    async with async_session() as s:
        total_r = await s.execute(text("SELECT COUNT(*) FROM workout"))
        total = total_r.scalar_one()
        try:
            metcon_r = await s.execute(text("SELECT COUNT(*) FROM metcon"))
            metcon_total = metcon_r.scalar_one()
        except Exception:
            metcon_total = 0
        last_r = await s.execute(select(Workout).order_by(desc(Workout.date), desc(Workout.id)).limit(1))
        last_row = last_r.scalar()
        bests_r = await s.execute(
            select(Workout.exercise, func.max(Workout.value))
            .where(Workout.value.is_not(None)).group_by(Workout.exercise)
        )
        bests_rows = bests_r.all()
    bests = {e: float(v) for e, v in bests_rows if e}
    last = _row_to_out(last_row) if last_row else None
    return StatsOut(total_sessions=int(total), total_metcons=int(metcon_total), bests=bests, last=last)


@app.get("/analytics/prs", response_model=PRsOut)
async def analytics_prs(exercise: Optional[str] = None) -> PRsOut:
    async with async_session() as s:
        stmt = select(Workout.exercise, func.max(Workout.value)).where(Workout.value.is_not(None))
        if exercise: stmt = stmt.where(func.lower(Workout.exercise) == exercise.lower())
        stmt = stmt.group_by(Workout.exercise)
        result = await s.execute(stmt)
        rows = result.all()
    return PRsOut(prs={e: float(v) for e, v in rows if e})


@app.get("/analytics/weekly_summary", response_model=WeeklySummaryOut)
async def analytics_weekly_summary(cycle: int, week: Optional[int] = None) -> WeeklySummaryOut:
    async with async_session() as s:
        stmt = select(Workout.day, func.count()).where(Workout.cycle == cycle)
        if week is not None: stmt = stmt.where(Workout.week == week)
        stmt = stmt.group_by(Workout.day)
        result = await s.execute(stmt)
        rows = result.all()
        total_stmt = select(func.count()).select_from(Workout).where(Workout.cycle == cycle)
        if week is not None: total_stmt = total_stmt.where(Workout.week == week)
        total = (await s.execute(total_stmt)).scalar_one()
    days = {int(d): int(c) for d, c in rows if d is not None}
    return WeeklySummaryOut(cycle=cycle, week=week, days_logged=days, total_sets=int(total))


@app.get("/analytics/progress_compare", response_model=ProgressCompareOut)
async def analytics_progress_compare(exercise: str, cycle: int, week1: int, week2: int) -> ProgressCompareOut:
    async with async_session() as s:
        w1 = (await s.execute(select(func.count()).select_from(Workout).where(
            Workout.cycle == cycle, Workout.week == week1, func.lower(Workout.exercise) == exercise.lower()
        ))).scalar_one()
        w2 = (await s.execute(select(func.count()).select_from(Workout).where(
            Workout.cycle == cycle, Workout.week == week2, func.lower(Workout.exercise) == exercise.lower()
        ))).scalar_one()
    return ProgressCompareOut(exercise=exercise, cycle=cycle, week1=week1, week2=week2,
                              week1_sets=int(w1), week2_sets=int(w2), delta=int(w2) - int(w1))


@app.get("/analytics/repmax", response_model=RepmaxOut)
async def analytics_repmax(exercise: str, mode: str = "max") -> RepmaxOut:
    allowed = {"max", "min", "auto"}
    if mode not in allowed:
        raise HTTPException(400, f"mode must be one of {sorted(allowed)}")
    async with async_session() as s:
        result = await s.execute(select(Workout).where(
            func.lower(Workout.exercise) == exercise.lower(), Workout.value.is_not(None),
        ))
        rows = result.scalars().all()
    if not rows:
        return RepmaxOut(exercise=exercise)
    effective = mode
    if mode == "auto":
        timey = {"sec", "s", "seconds", "min", "minutes"}
        effective = "min" if any((w.unit or "").strip().lower() in timey for w in rows) else "max"
    key = lambda w: float(w.value)
    best = max(rows, key=key) if effective == "max" else min(rows, key=key)
    return RepmaxOut(exercise=exercise, best_value=float(best.value), unit=best.unit, workout_id=best.id)


@app.get("/analytics/consistency", response_model=ConsistencyOut)
async def analytics_consistency(cycle: Optional[int] = None) -> ConsistencyOut:
    REQUIRED_DAYS = {1, 2, 3, 4}
    async with async_session() as s:
        stmt = select(Workout.cycle, Workout.week).group_by(Workout.cycle, Workout.week)
        if cycle is not None: stmt = stmt.where(Workout.cycle == cycle)
        weeks = (await s.execute(stmt)).all()
        completed: List[ConsistencyWeekOut] = []
        for cyc, wk in weeks:
            if cyc is None or wk is None: continue
            days = {d for (d,) in (await s.execute(
                select(Workout.day).where(Workout.cycle == cyc, Workout.week == wk)
            )).all() if d is not None}
            if REQUIRED_DAYS.issubset(days):
                completed.append(ConsistencyWeekOut(cycle=int(cyc), week=int(wk)))
    completed.sort(key=lambda x: (x.cycle, x.week))
    return ConsistencyOut(completed_weeks=completed, total=len(completed))


@app.get("/analytics/estimated_1rm", response_model=OneRMOut)
async def estimated_1rm(
    exercise: str, formula: str = Query("epley", description="epley or brzycki"),
) -> OneRMOut:
    if formula not in ("epley", "brzycki"):
        raise HTTPException(400, "formula must be 'epley' or 'brzycki'")
    async with async_session() as s:
        result = await s.execute(select(Workout).where(
            func.lower(Workout.exercise) == exercise.lower(),
            Workout.value.is_not(None), Workout.reps.is_not(None), Workout.reps > 0,
        ))
        rows = result.scalars().all()
    if not rows:
        return OneRMOut(exercise=exercise, formula=formula)
    best_1rm, best_row = 0.0, None
    for w in rows:
        weight, reps = float(w.value), _safe_int(w.reps) or 0
        if reps == 1: e1rm = weight
        elif formula == "epley": e1rm = weight * (1 + reps / 30)
        else: e1rm = weight * (36 / (37 - reps)) if reps < 37 else weight
        if e1rm > best_1rm: best_1rm, best_row = e1rm, w
    if best_row is None:
        return OneRMOut(exercise=exercise, formula=formula)
    return OneRMOut(exercise=exercise, estimated_1rm_kg=round(best_1rm, 1),
                    based_on_value=float(best_row.value), based_on_reps=_safe_int(best_row.reps) or 0, formula=formula)


@app.get("/analytics/volume", response_model=WeeklyVolumeOut)
async def analytics_volume(
    cycle: Optional[int] = None, week: Optional[int] = None,
    start: Optional[str] = None, end: Optional[str] = None,
) -> WeeklyVolumeOut:
    stmt = select(Workout).where(Workout.value.is_not(None), Workout.reps.is_not(None))
    if cycle is not None: stmt = stmt.where(Workout.cycle == cycle)
    if week is not None: stmt = stmt.where(Workout.week == week)
    if start: stmt = stmt.where(Workout.date >= start)
    if end: stmt = stmt.where(Workout.date <= end)
    async with async_session() as s:
        rows = (await s.execute(stmt)).scalars().all()
    exercises: Dict[str, dict] = {}
    for w in rows:
        nm = w.exercise
        if nm not in exercises: exercises[nm] = {"total_volume": 0.0, "total_sets": 0, "total_reps": 0, "unit": w.unit}
        reps = _safe_int(w.reps) or 0
        exercises[nm]["total_volume"] += (float(w.value) if w.value else 0.0) * reps
        exercises[nm]["total_sets"] += 1
        exercises[nm]["total_reps"] += reps
    exercise_list = [VolumeOut(exercise=nm, **d) for nm, d in sorted(exercises.items())]
    return WeeklyVolumeOut(cycle=cycle, week=week, exercises=exercise_list,
                           grand_total_volume=round(sum(e.total_volume for e in exercise_list), 1),
                           grand_total_sets=sum(e.total_sets for e in exercise_list))


@app.get("/analytics/timeline", response_model=ExerciseTimelineOut)
async def exercise_timeline(
    exercise: str = Query(..., min_length=1), start: Optional[str] = None, end: Optional[str] = None,
) -> ExerciseTimelineOut:
    stmt = select(Workout).where(func.lower(Workout.exercise) == exercise.lower(), Workout.value.is_not(None))
    if start: stmt = stmt.where(Workout.date >= start)
    if end: stmt = stmt.where(Workout.date <= end)
    stmt = stmt.order_by(asc(Workout.date), asc(Workout.id))
    async with async_session() as s:
        rows = (await s.execute(stmt)).scalars().all()
    by_date: Dict[str, dict] = {}
    for w in rows:
        d = w.date
        if d not in by_date: by_date[d] = {"best": 0.0, "unit": w.unit, "sets": 0, "reps": 0}
        val = float(w.value)
        if val > by_date[d]["best"]: by_date[d]["best"] = val; by_date[d]["unit"] = w.unit
        by_date[d]["sets"] += 1; by_date[d]["reps"] += _safe_int(w.reps) or 0
    return ExerciseTimelineOut(exercise=exercise, timeline=[
        TimelinePointOut(date=d, best_value=data["best"], unit=data["unit"],
                         total_sets=data["sets"], total_reps=data["reps"])
        for d, data in sorted(by_date.items())
    ])


@app.get("/search_exercise", response_model=SearchExerciseOut)
async def search_exercise(
    exercise: str, cycle: Optional[int] = None, week: Optional[int] = None,
    day: Optional[int] = None, tag: Optional[str] = None,
) -> SearchExerciseOut:
    async with async_session() as s:
        stmt = select(Workout).where(_safe_like(Workout.exercise, exercise))
        if cycle is not None: stmt = stmt.where(Workout.cycle == cycle)
        if week is not None: stmt = stmt.where(Workout.week == week)
        if day is not None: stmt = stmt.where(Workout.day == day)
        if tag: stmt = stmt.where(_safe_like(Workout.tags, tag))
        rows = (await s.execute(stmt.order_by(asc(Workout.date), asc(Workout.id)))).scalars().all()
    return SearchExerciseOut(exercise=exercise, logged=[_row_to_out(w) for w in rows])


@app.get("/export/csv", response_model=CsvExportOut)
async def export_csv() -> CsvExportOut:
    async with async_session() as s:
        rows = (await s.execute(select(Workout).order_by(asc(Workout.date), asc(Workout.id)))).scalars().all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "date", "exercise", "set_number", "reps", "value", "unit",
                      "cycle", "week", "day", "notes", "tags"])
    for w in rows:
        writer.writerow([w.id, w.date, w.exercise, w.set_number, w.reps, w.value, w.unit,
                          w.cycle, w.week, w.day, (w.notes or ""), (w.tags or "")])
    return CsvExportOut(filename="workouts.csv", rows=len(rows), csv=buf.getvalue())


@app.get("/export/metcons_csv", response_model=CsvExportOut)
async def export_metcons_csv() -> CsvExportOut:
    async with async_session() as s:
        rows = (await s.execute(select(Metcon).order_by(asc(Metcon.date), asc(Metcon.id)))).scalars().all()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["id", "date", "name", "workout_type", "description",
                      "score_time_seconds", "score_rounds", "score_reps", "score_display",
                      "rx", "time_cap_seconds", "cycle", "week", "day", "notes", "tags"])
    for m in rows:
        writer.writerow([m.id, m.date, m.name, m.workout_type, (m.description or ""),
                          m.score_time_seconds, m.score_rounds, m.score_reps,
                          (m.score_display or ""), (m.rx or ""), m.time_cap_seconds,
                          m.cycle, m.week, m.day, (m.notes or ""), (m.tags or "")])
    return CsvExportOut(filename="metcons.csv", rows=len(rows), csv=buf.getvalue())


@app.get("/day_summary", response_model=DaySummaryOut)
async def day_summary(
    cycle: int = Query(..., ge=0), week: int = Query(..., ge=0), day: int = Query(..., ge=0),
) -> DaySummaryOut:
    """Get everything logged for a training day — both strength sets and metcons."""
    async with async_session() as s:
        w_result = await s.execute(
            select(Workout).where(Workout.cycle == cycle, Workout.week == week, Workout.day == day)
            .order_by(asc(Workout.date), asc(Workout.id))
        )
        workouts = [_row_to_out(w) for w in w_result.scalars().all()]
        m_result = await s.execute(
            select(Metcon).where(Metcon.cycle == cycle, Metcon.week == week, Metcon.day == day)
            .order_by(asc(Metcon.date), asc(Metcon.id))
        )
        metcons = [_metcon_to_out(m) for m in m_result.scalars().all()]
    return DaySummaryOut(
        cycle=cycle, week=week, day=day,
        workouts=workouts, metcons=metcons,
        total_workout_sets=len(workouts), total_metcons=len(metcons),
    )
