# app.py
# =============================================================================
# CF-Log API v10 — Workouts & Analytics (FastAPI + SQLAlchemy 2.x, Pydantic v2)
# Results-only: no plan storage. One row per working set.
# Supports both SQLite (local) and Cloud SQL PostgreSQL (production).
# =============================================================================

from __future__ import annotations

import csv
import io
import os
import logging
from datetime import datetime
from pathlib import Path as OSPath
from typing import Dict, List, Optional

from fastapi import Body, FastAPI, HTTPException, Query
from fastapi import Path as FPath
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import (
    Float,
    Integer,
    String,
    asc,
    create_engine,
    desc,
    func,
    select,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger("cf-log-api")

# -----------------------------------------------------------------------------
# DB connection
# Cloud SQL (PostgreSQL) if CLOUD_SQL_CONNECTION_NAME is set; else SQLite.
# -----------------------------------------------------------------------------
CLOUD_SQL_CONNECTION_NAME = os.getenv("CLOUD_SQL_CONNECTION_NAME")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "cf_log")

if CLOUD_SQL_CONNECTION_NAME:
    # Cloud SQL via Unix socket (Cloud Run)
    socket_path = f"/cloudsql/{CLOUD_SQL_CONNECTION_NAME}"
    DB_URL = (
        f"postgresql+pg8000://{DB_USER}:{DB_PASSWORD}@/{DB_NAME}"
        f"?unix_sock={socket_path}/.s.PGSQL.5432"
    )
    DB_PATH = DB_URL  # for health endpoint display
    DB_TYPE = "postgresql"
    engine = create_engine(DB_URL, future=True, echo=False)
    log.info(f"Using Cloud SQL: {CLOUD_SQL_CONNECTION_NAME}")
else:
    # Local SQLite
    env_db = os.getenv("CFLOG_DB")
    candidates = [
        env_db,
        str((OSPath(__file__).parent / "data" / "cf_log.db").resolve()),
        str((OSPath(__file__).parent / "cf_log.db").resolve()),
    ]
    DB_PATH = next((p for p in candidates if p and OSPath(p).exists()), candidates[-1])
    DB_TYPE = "sqlite"
    engine = create_engine(f"sqlite:///{DB_PATH}", future=True, echo=False)
    log.info(f"Using SQLite: {DB_PATH}")

# -----------------------------------------------------------------------------
# SQLAlchemy model — workout table only (no plans)
# -----------------------------------------------------------------------------
class Base(DeclarativeBase):
    pass


class Workout(Base):
    __tablename__ = "workout"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    date: Mapped[str] = mapped_column(String, nullable=False)
    exercise: Mapped[str] = mapped_column(String, nullable=False)
    value: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    unit: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    sets: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reps: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    set_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rpe: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    cycle: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    iso_week: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    day: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    plan_day_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(String, nullable=True)


# Add set_number column to existing DB if it doesn't exist
def _ensure_set_number_column():
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT set_number FROM workout LIMIT 1"))
    except Exception:
        try:
            with engine.begin() as conn:
                conn.execute(text("ALTER TABLE workout ADD COLUMN set_number INTEGER"))
            log.info("Added set_number column to workout table")
        except Exception as e:
            log.warning(f"Could not add set_number column (may already exist): {e}")


Base.metadata.create_all(engine)
_ensure_set_number_column()

# -----------------------------------------------------------------------------
# Pydantic schemas
# -----------------------------------------------------------------------------
class HealthOut(BaseModel):
    ok: bool = True
    db_path: str
    db_type: str
    timestamp: str


class GenericResponse(BaseModel):
    message: str


class DBInfoOut(BaseModel):
    db_path: str
    db_type: str
    workout_rows: int


class ExerciseCountOut(BaseModel):
    exercise: str
    count: int


class WorkoutIn(BaseModel):
    date: str
    exercise: str
    value: Optional[float] = None
    unit: Optional[str] = None
    sets: Optional[str] = None
    reps: Optional[str] = None
    set_number: Optional[int] = None
    rpe: Optional[float] = None
    cycle: Optional[int] = None
    week: Optional[int] = None
    iso_week: Optional[int] = None
    day: Optional[int] = None
    plan_day_id: Optional[int] = None
    notes: Optional[str] = ""


class WorkoutOut(WorkoutIn):
    id: int
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
    days_logged: Dict[str, int] = Field(default_factory=dict)
    total_sessions: int


class ProgressCompareOut(BaseModel):
    exercise: str
    cycle: int
    week1: int
    week2: int
    week1_sessions: int
    week2_sessions: int
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


# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(
    title="CF-Log API",
    description="CrossFit Training Log & Analytics API. Results only — one row per working set.",
    version="10.0.0",
)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _like_patterns(q: str) -> List[str]:
    q = q.strip().lower()
    if not q:
        return []
    tokens = [t for t in q.replace(",", " ").split() if t]
    patterns = [f"%{q}%"]
    for t in tokens:
        patterns.append(f"%{t}%")
    return patterns


def _row_to_workout_out(w: Workout) -> WorkoutOut:
    return WorkoutOut.model_validate(w, from_attributes=True)

# -----------------------------------------------------------------------------
# Health / Root
# -----------------------------------------------------------------------------
@app.get("/health", response_model=HealthOut)
def health() -> HealthOut:
    return HealthOut(
        ok=True,
        db_path=str(DB_PATH),
        db_type=DB_TYPE,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/", response_model=GenericResponse)
def root() -> GenericResponse:
    return GenericResponse(message="CF-Log API v10 is running")

# -----------------------------------------------------------------------------
# Debug
# -----------------------------------------------------------------------------
@app.get("/debug/dbinfo", response_model=DBInfoOut)
def dbinfo() -> DBInfoOut:
    with Session(engine) as s:
        workout_rows = s.execute(text("SELECT COUNT(*) FROM workout")).scalar_one()
    return DBInfoOut(
        db_path=str(DB_PATH),
        db_type=DB_TYPE,
        workout_rows=int(workout_rows),
    )


@app.get("/debug/exercises", response_model=List[ExerciseCountOut])
def debug_exercises(limit: int = 50) -> List[ExerciseCountOut]:
    with Session(engine) as s:
        rows = s.execute(
            select(Workout.exercise, func.count())
            .group_by(Workout.exercise)
            .order_by(desc(func.count()))
            .limit(limit)
        ).all()
    return [ExerciseCountOut(exercise=e or "", count=int(c)) for e, c in rows]

# -----------------------------------------------------------------------------
# Workouts
# -----------------------------------------------------------------------------
@app.post("/workouts", response_model=GenericResponse)
def add_workout(w: WorkoutIn) -> GenericResponse:
    with Session(engine) as s:
        obj = Workout(**w.model_dump())
        s.add(obj)
        s.commit()
    return GenericResponse(message="Workout saved")


@app.post("/workouts/bulk", response_model=BulkWorkoutOut)
def add_workouts_bulk(body: BulkWorkoutIn) -> BulkWorkoutOut:
    """Log all working sets of an exercise in one call. Returns saved count + IDs for verification."""
    ids: List[int] = []
    with Session(engine) as s:
        for w in body.workouts:
            obj = Workout(**w.model_dump())
            s.add(obj)
            s.flush()
            ids.append(obj.id)
        s.commit()
    return BulkWorkoutOut(saved=len(ids), ids=ids)


# IMPORTANT: define /workouts/search BEFORE any dynamic /workouts/{...}
@app.get("/workouts/search", response_model=List[WorkoutOut])
def search_workouts(
    q: str = Query(..., min_length=1, description="exercise name or part of it"),
    cycle: Optional[int] = Query(None, ge=0),
    week: Optional[int] = Query(None, ge=0),
    day: Optional[int] = Query(None, ge=0),
    iso_week: Optional[int] = Query(None, ge=0),
    start: Optional[str] = Query(None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> List[WorkoutOut]:
    """Case-insensitive LIKE search with optional filters. Returns [] on any internal error."""
    try:
        patterns = _like_patterns(q)
        if not patterns:
            return []

        with Session(engine) as s:
            cond = None
            for ptn in patterns:
                like = func.lower(Workout.exercise).like(ptn)
                cond = like if cond is None else (cond | like)

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

            stmt = stmt.order_by(asc(Workout.date), asc(Workout.id)).limit(500)
            rows = s.scalars(stmt).all()
        return [_row_to_workout_out(w) for w in rows]
    except Exception as e:
        log.error(f"/workouts/search error: {e}")
        return []


@app.get("/workouts/by_cwd", response_model=List[WorkoutOut])
def workouts_by_cycle_week_day(
    cycle: int = Query(..., ge=0),
    week: int = Query(..., ge=0),
    day: int = Query(..., ge=0),
) -> List[WorkoutOut]:
    """Convenience helper: Cycle -> Week -> Day."""
    try:
        with Session(engine) as s:
            stmt = (
                select(Workout)
                .where(Workout.cycle == cycle, Workout.week == week, Workout.day == day)
                .order_by(asc(Workout.date), asc(Workout.id))
            )
            rows = s.scalars(stmt).all()
        return [_row_to_workout_out(w) for w in rows]
    except Exception as e:
        log.error(f"/workouts/by_cwd error: {e}")
        return []


@app.get("/workouts/last", response_model=WorkoutOut)
def last_workout(exercise: str = Query(..., min_length=1)) -> WorkoutOut:
    with Session(engine) as s:
        row = s.scalar(
            select(Workout)
            .where(func.lower(Workout.exercise).like(f"%{exercise.lower()}%"))
            .order_by(desc(Workout.date), desc(Workout.id))
            .limit(1)
        )
        if not row:
            raise HTTPException(404, "No workout found for that exercise")
        return _row_to_workout_out(row)


@app.get("/workouts", response_model=List[WorkoutOut])
def query_workouts(
    exercise: Optional[str] = None,
    cycle: Optional[int] = None,
    week: Optional[int] = None,
    iso_week: Optional[int] = None,
    day: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[WorkoutOut]:
    stmt = select(Workout)
    if exercise:
        stmt = stmt.where(func.lower(Workout.exercise).like(f"%{exercise.lower()}%"))
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

    stmt = stmt.order_by(asc(Workout.date), asc(Workout.id))
    with Session(engine) as s:
        rows = s.scalars(stmt).all()
    return [_row_to_workout_out(w) for w in rows]


@app.put("/workouts/{workout_id}", response_model=WorkoutOut)
def edit_workout(workout_id: int = FPath(..., ge=1), body: WorkoutIn = Body(...)) -> WorkoutOut:
    with Session(engine) as s:
        w = s.get(Workout, workout_id)
        if not w:
            raise HTTPException(404, "Workout not found")
        for k, v in body.model_dump().items():
            setattr(w, k, v)
        s.commit()
        s.refresh(w)
        return _row_to_workout_out(w)


@app.delete("/workouts/{workout_id}", response_model=GenericResponse)
def delete_workout(workout_id: int = FPath(..., ge=1)) -> GenericResponse:
    with Session(engine) as s:
        w = s.get(Workout, workout_id)
        if not w:
            raise HTTPException(404, "Workout not found")
        s.delete(w)
        s.commit()
    return GenericResponse(message="Workout deleted")

# -----------------------------------------------------------------------------
# Stats & Analytics
# -----------------------------------------------------------------------------
@app.get("/stats", response_model=StatsOut)
def stats() -> StatsOut:
    with Session(engine) as s:
        total = s.execute(text("SELECT COUNT(*) FROM workout")).scalar_one()
        last_row = s.scalar(select(Workout).order_by(desc(Workout.date), desc(Workout.id)).limit(1))
        bests_rows = s.execute(
            select(Workout.exercise, func.max(Workout.value))
            .where(Workout.value.is_not(None))
            .group_by(Workout.exercise)
        ).all()

    bests = {e: float(v) for e, v in bests_rows if e}
    last = _row_to_workout_out(last_row) if last_row else None
    return StatsOut(total_sessions=int(total), bests=bests, last=last)


@app.get("/analytics/prs", response_model=PRsOut)
def analytics_prs(exercise: Optional[str] = None) -> PRsOut:
    with Session(engine) as s:
        stmt = select(Workout.exercise, func.max(Workout.value)).where(Workout.value.is_not(None))
        if exercise:
            stmt = stmt.where(func.lower(Workout.exercise) == exercise.lower())
        stmt = stmt.group_by(Workout.exercise)
        rows = s.execute(stmt).all()
    prs = {e: float(v) for e, v in rows if e}
    return PRsOut(prs=prs)


@app.get("/analytics/weekly_summary", response_model=WeeklySummaryOut)
def analytics_weekly_summary(cycle: int, week: Optional[int] = None) -> WeeklySummaryOut:
    try:
        with Session(engine) as s:
            stmt = select(Workout.day, func.count()).where(Workout.cycle == cycle)
            if week is not None:
                stmt = stmt.where(Workout.week == week)
            stmt = stmt.group_by(Workout.day)
            rows = s.execute(stmt).all()

            total_stmt = select(func.count()).where(Workout.cycle == cycle)
            if week is not None:
                total_stmt = total_stmt.where(Workout.week == week)
            total = s.execute(total_stmt).scalar_one()
        days = {str(d): int(c) for d, c in rows if d is not None}
        return WeeklySummaryOut(cycle=cycle, week=week, days_logged=days, total_sessions=int(total))
    except Exception as e:
        log.error(f"/analytics/weekly_summary error: {e}")
        return WeeklySummaryOut(cycle=cycle, week=week, days_logged={}, total_sessions=0)


@app.get("/analytics/progress_compare", response_model=ProgressCompareOut)
def analytics_progress_compare(exercise: str, cycle: int, week1: int, week2: int) -> ProgressCompareOut:
    with Session(engine) as s:
        w1 = s.execute(
            select(func.count()).where(
                Workout.cycle == cycle, Workout.week == week1, func.lower(Workout.exercise) == exercise.lower()
            )
        ).scalar_one()
        w2 = s.execute(
            select(func.count()).where(
                Workout.cycle == cycle, Workout.week == week2, func.lower(Workout.exercise) == exercise.lower()
            )
        ).scalar_one()
    return ProgressCompareOut(
        exercise=exercise, cycle=cycle, week1=week1, week2=week2,
        week1_sessions=int(w1), week2_sessions=int(w2), delta=int(w2) - int(w1),
    )


@app.get("/analytics/repmax", response_model=RepmaxOut)
def analytics_repmax(exercise: str) -> RepmaxOut:
    with Session(engine) as s:
        rows = s.scalars(
            select(Workout).where(func.lower(Workout.exercise) == exercise.lower(), Workout.value.is_not(None))
        ).all()

    if not rows:
        return RepmaxOut(exercise=exercise)

    # Auto-detect: if any row has time-based unit, pick minimum; else pick maximum
    timey = {"sec", "s", "seconds", "min", "minutes"}
    if any((w.unit or "").strip().lower() in timey for w in rows):
        best = min(rows, key=lambda w: float(w.value))
    else:
        best = max(rows, key=lambda w: float(w.value))

    return RepmaxOut(
        exercise=exercise, best_value=float(best.value), unit=best.unit, workout_id=best.id,
    )


@app.get("/analytics/consistency", response_model=ConsistencyOut)
def analytics_consistency(cycle: Optional[int] = None) -> ConsistencyOut:
    REQUIRED_DAYS = {1, 2, 3, 4}
    with Session(engine) as s:
        stmt = select(Workout.cycle, Workout.week).group_by(Workout.cycle, Workout.week)
        if cycle is not None:
            stmt = stmt.where(Workout.cycle == cycle)
        weeks = s.execute(stmt).all()

    full: List[int] = []
    with Session(engine) as s:
        for cyc, wk in weeks:
            if cyc is None or wk is None:
                continue
            days = {
                d for (d,) in s.execute(
                    select(Workout.day).where(Workout.cycle == cyc, Workout.week == wk)
                ).all()
                if d is not None
            }
            if REQUIRED_DAYS.issubset(days):
                full.append(int(wk))
    return ConsistencyOut(weeks_completed_100=sorted(full))

# -----------------------------------------------------------------------------
# Search exercise (logs only, no plans)
# -----------------------------------------------------------------------------
@app.get("/search_exercise", response_model=SearchExerciseOut)
def search_exercise(
    exercise: str,
    cycle: Optional[int] = None,
    week: Optional[int] = None,
    iso_week: Optional[int] = None,
    day: Optional[int] = None,
) -> SearchExerciseOut:
    with Session(engine) as s:
        stmt = select(Workout).where(func.lower(Workout.exercise).like(f"%{exercise.lower()}%"))
        if cycle is not None:
            stmt = stmt.where(Workout.cycle == cycle)
        if week is not None:
            stmt = stmt.where(Workout.week == week)
        if iso_week is not None:
            stmt = stmt.where(Workout.iso_week == iso_week)
        if day is not None:
            stmt = stmt.where(Workout.day == day)

        logged_rows = s.scalars(stmt.order_by(asc(Workout.date), asc(Workout.id))).all()
        logged = [_row_to_workout_out(w) for w in logged_rows]

    return SearchExerciseOut(exercise=exercise, logged=logged)

# -----------------------------------------------------------------------------
# Export CSV
# -----------------------------------------------------------------------------
@app.get("/export/csv", response_model=CsvExportOut)
def export_csv() -> CsvExportOut:
    with Session(engine) as s:
        rows = s.scalars(select(Workout).order_by(asc(Workout.date), asc(Workout.id))).all()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "date", "exercise", "value", "unit", "sets", "reps",
        "set_number", "rpe", "cycle", "week", "iso_week", "day",
        "plan_day_id", "notes",
    ])
    for w in rows:
        writer.writerow([
            w.id, w.date, w.exercise, w.value, w.unit, w.sets, w.reps,
            w.set_number, w.rpe, w.cycle, w.week, w.iso_week, w.day,
            w.plan_day_id, (w.notes or ""),
        ])

    return CsvExportOut(filename="workouts.csv", rows=len(rows), csv=buf.getvalue())
