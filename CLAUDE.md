# CLAUDE.md — CF Training Companion

## Project Overview

FastAPI backend for logging and analyzing CrossFit training data. Single file API (`app.py`) with 45 endpoints covering strength sets, metcon/conditioning results, personal records, and analytics.

**Stack:** Python 3.11, FastAPI, SQLAlchemy 2.x (async), SQLite (local) / Cloud SQL PostgreSQL (production), Docker, Google Cloud Run.

---

## Running Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Copy env file (SQLite works with no changes)
cp .env.example .env

# Start dev server (SQLite, auto-reload)
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API docs available at `http://localhost:8000/docs`.

## Running Tests

```bash
pytest test_app.py -v
```

Tests use an in-memory SQLite DB — no `.env` required.

---

## Architecture

- **Single file:** All logic lives in `app.py`. No separate routers or service layers.
- **Two DB tables:** `workout` (strength sets) and `metcon` (conditioning results). No foreign key relationships between them.
- **DB selection:** If `CLOUD_SQL_CONNECTION_NAME` env var is set → Cloud SQL PostgreSQL via `asyncpg`. Otherwise → SQLite via `aiosqlite`.
- **Schema migrations:** Handled inline at startup via `ALTER TABLE` wrapped in try/except (additive columns only).

## Key Environment Variables

| Variable | Default | Notes |
|----------|---------|-------|
| `CLOUD_SQL_CONNECTION_NAME` | _(none)_ | Set to use Cloud SQL; omit for SQLite |
| `DB_USER` | `postgres` | Cloud SQL only |
| `DB_PASSWORD` | _(empty)_ | Must be set explicitly in production |
| `DB_NAME` | `cf_log` | Cloud SQL only |
| `CFLOG_DB` | `./cf_log.db` | SQLite path (local only) |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins |
| `RATE_LIMIT_REQUESTS` | `300` | Requests per IP per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |

---

## Deployment (Cloud Run)

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cf-log-api

# Deploy
gcloud run deploy cf-log-api \
  --image gcr.io/YOUR_PROJECT_ID/cf-log-api \
  --platform managed \
  --region us-central1 \
  --add-cloudsql-instances PROJECT:REGION:INSTANCE \
  --set-env-vars CLOUD_SQL_CONNECTION_NAME=project:region:instance,...
```

Or via Docker locally:

```bash
docker build -t cf-log-api .
docker run -p 8080:8080 cf-log-api
```

---

## Endpoint Groups

| Group | Count | Paths |
|-------|-------|-------|
| Workouts (strength) | 11 | `/workouts*` |
| Metcons (conditioning) | 7 | `/metcons*` |
| Workout analytics | 11 | `/analytics/*`, `/stats` |
| Metcon analytics | 2 | `/analytics/metcon_*` |
| Advanced analytics | 5 | `/analytics/movement_frequency`, `/intensity_distribution`, etc. |
| Export | 2 | `/export/csv`, `/export/metcons_csv` |
| Utility | 7 | `/health`, `/`, `/debug/*`, `/day_summary`, `/search_exercise` |

Full endpoint reference: see `DASHBOARD_API_REFERENCE.md`.

---

## Known Limitations / Design Notes

- No UNIQUE DB constraint on `(date, exercise, set_number)` — duplicate detection is application-level only. Two concurrent POSTs can bypass it.
- Global exception handler catches `HTTPException`, returning 500 instead of the correct 4xx status in some edge cases.
- No pagination (offset/cursor) — large queries return up to hardcoded limits (200–500 rows).
- `requirements.txt` has no version pins — pin before any major dependency upgrades.
- Dockerfile runs as root and has no `HEALTHCHECK` instruction.
