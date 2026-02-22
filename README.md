# CF Training Companion

A FastAPI backend for logging and analyzing CrossFit training data — strength sets, conditioning workouts (metcons), personal records, and long-term progression trends.

---

## Features

- Log strength sets with exercise, reps, weight, unit, and training cycle metadata
- Log metcon/benchmark results (Fran, Helen, AMRAPs, EMOMs, etc.)
- Personal record tracking and estimated 1RM calculation
- Analytics: trends, week-over-week progression, volume, intensity distribution, movement frequency
- Export all data to CSV
- Duplicate detection, bulk logging, and undo support
- Rate limiting and structured error handling

---

## Tech Stack

- **Python 3.11** + **FastAPI**
- **SQLAlchemy 2.x** (async) with **SQLite** (local) or **Cloud SQL PostgreSQL** (production)
- **Uvicorn** ASGI server
- **Docker** + **Google Cloud Run** for deployment

---

## Getting Started

### Local Development (SQLite)

```bash
# 1. Clone the repo
git clone <repo-url>
cd cf-training-companion

# 2. Create a virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy and configure environment variables
cp .env.example .env
# Edit .env if needed (SQLite works with no changes)

# 5. Run the dev server
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

API is now available at:
- Root: `http://localhost:8000/`
- Swagger docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Docker

```bash
docker build -t cf-log-api .
docker run -p 8080:8080 cf-log-api
```

### Production (Cloud SQL + Cloud Run)

```bash
# Build and push to Container Registry
gcloud builds submit --tag gcr.io/YOUR_PROJECT_ID/cf-log-api

# Deploy to Cloud Run
gcloud run deploy cf-log-api \
  --image gcr.io/YOUR_PROJECT_ID/cf-log-api \
  --platform managed \
  --region us-central1 \
  --set-env-vars CLOUD_SQL_CONNECTION_NAME=project:region:instance,DB_USER=postgres,DB_PASSWORD=secret,DB_NAME=cf_log,ALLOWED_ORIGINS=https://your-dashboard.com
```

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `CLOUD_SQL_CONNECTION_NAME` | _(none)_ | If set, connects to Cloud SQL PostgreSQL. Otherwise uses SQLite. |
| `DB_USER` | `postgres` | PostgreSQL username |
| `DB_PASSWORD` | _(empty)_ | PostgreSQL password |
| `DB_NAME` | `cf_log` | Database name |
| `CFLOG_DB` | _(auto)_ | SQLite file path (local only) |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | Comma-separated CORS origins for the dashboard |
| `RATE_LIMIT_REQUESTS` | `300` | Requests per window per IP |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window in seconds |

---

## API Overview

### Workouts (Strength)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/workouts` | Log a single set |
| `POST` | `/workouts/bulk` | Log multiple sets at once |
| `GET` | `/workouts` | Query sets with filters (exercise, cycle, week, date range, tag) |
| `GET` | `/workouts/search` | Full-text search on exercise names |
| `GET` | `/workouts/last` | Most recent set for an exercise |
| `PUT` | `/workouts/{id}` | Edit a set |
| `DELETE` | `/workouts/{id}` | Delete a set |
| `DELETE` | `/workouts/undo_last` | Delete the last N sets |

### Metcons (Conditioning)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/metcons` | Log a metcon result |
| `POST` | `/metcons/bulk` | Log multiple metcon results |
| `GET` | `/metcons` | Query metcons with filters |
| `GET` | `/metcons/search` | Full-text search on benchmark names |
| `GET` | `/metcons/last` | Most recent result for a benchmark |
| `PUT` | `/metcons/{id}` | Edit a metcon |
| `DELETE` | `/metcons/{id}` | Delete a metcon |

### Analytics

| Path | Description |
|------|-------------|
| `/analytics/prs` | Personal records (max value per exercise) |
| `/analytics/estimated_1rm` | Estimated 1RM (Epley or Brzycki) |
| `/analytics/timeline` | Best value per session over time |
| `/analytics/trend` | Linear regression trend (increasing/stable/decreasing) |
| `/analytics/week_over_week` | Per-week best, avg, volume, sets |
| `/analytics/volume` | Total volume by exercise for a period |
| `/analytics/intensity_distribution` | Sets broken down by rep bracket |
| `/analytics/movement_frequency` | Session and set count per exercise |
| `/analytics/metcon_prs` | Best scores per benchmark |
| `/analytics/metcon_timeline` | Benchmark progression over time |
| `/analytics/weekly_summary` | Sets logged per day in a cycle/week |
| `/analytics/consistency` | Which weeks had 4+ training days logged |

### Other

| Path | Description |
|------|-------------|
| `GET /stats` | Summary: sessions, metcons, PRs, last workout date |
| `GET /day_summary` | Workouts + metcons for a training day |
| `GET /health` | Health check with DB connection status |
| `GET /export/csv` | Export all workouts as CSV |
| `GET /export/metcons_csv` | Export all metcons as CSV |

---

## Testing

```bash
pytest test_app.py -v
```

---

## Dashboard

The companion Next.js dashboard consumes this API. See [`DASHBOARD_API_REFERENCE.md`](DASHBOARD_API_REFERENCE.md) for the full endpoint reference, TypeScript types, and recommended project setup.
