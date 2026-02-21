# CF Training Companion — Dashboard API Reference

This document is written for the **Next.js dashboard** (`cf-dashboard`).
It describes every endpoint the frontend can call, the expected shapes, and
recommended pages to build.

---

## Base URL

| Environment | URL |
|-------------|-----|
| Local dev | `http://localhost:8000` |
| Production (Cloud Run) | Set in dashboard `.env.local` as `NEXT_PUBLIC_API_URL` |

---

## CORS

The API allows requests from `http://localhost:3000` by default.
For production, set `ALLOWED_ORIGINS` on the API to your deployed dashboard URL.

---

## Common Types

```ts
type WorkoutOut = {
  id: number
  date: string              // "YYYY-MM-DD"
  exercise: string
  set_number: number | null
  reps: number | null
  value: number | null
  unit: string | null       // "kg" | "lb" | "sec" | "cal" | ...
  cycle: number | null
  week: number | null
  iso_week: number | null
  day: number | null
  notes: string | null
  tags: string | null       // comma-separated, lowercase
}

type MetconOut = {
  id: number
  date: string
  name: string
  workout_type: "for_time" | "amrap" | "emom" | "chipper" | "interval" | "other"
  description: string | null
  score_time_seconds: number | null
  score_rounds: number | null
  score_reps: number | null
  score_display: string | null   // "4:05" | "12+8" | ...
  rx: "rx" | "scaled" | "rx_plus" | null
  time_cap_seconds: number | null
  cycle: number | null
  week: number | null
  iso_week: number | null
  day: number | null
  notes: string | null
  tags: string | null
}
```

---

## Endpoints by Dashboard Page

### 1. Home / Stats Overview

**`GET /stats`**
```json
{
  "total_workout_sessions": 120,
  "total_metcon_sessions": 45,
  "total_prs": 18,
  "last_workout_date": "2026-02-20"
}
```

**`GET /health`**
```json
{ "status": "ok", "db": "connected" }
```

---

### 2. PR Board

**`GET /analytics/prs`** — All PRs (max value per exercise)
```json
{
  "prs": [
    { "exercise": "Back Squat", "best_value": 120.0, "unit": "kg", "reps": 1, "date": "2026-01-10" }
  ]
}
```

**`GET /analytics/prs?exercise=Back+Squat`** — PR for one exercise

**`GET /analytics/estimated_1rm?exercise=Back+Squat&formula=epley`**
```json
{
  "exercise": "Back Squat",
  "estimated_1rm": 135.5,
  "formula": "epley",
  "based_on_value": 120.0,
  "based_on_reps": 5
}
```

---

### 3. Lift Progress (Timeline Chart)

**`GET /analytics/timeline?exercise=Back+Squat`**
```json
{
  "exercise": "Back Squat",
  "data": [
    { "date": "2026-01-05", "best_value": 100.0, "unit": "kg" },
    { "date": "2026-01-12", "best_value": 105.0, "unit": "kg" }
  ]
}
```
Optional params: `start=YYYY-MM-DD`, `end=YYYY-MM-DD`

**`GET /analytics/trend?exercise=Back+Squat`**
```json
{
  "exercise": "Back Squat",
  "slope": 1.2,
  "direction": "increasing",   // "increasing" | "decreasing" | "stable"
  "recent_avg": 110.0,
  "overall_avg": 98.5
}
```

**`GET /analytics/week_over_week?exercise=Back+Squat`**
```json
{
  "exercise": "Back Squat",
  "weeks": [
    { "cycle": 1, "week": 1, "best": 100.0, "avg": 97.5, "volume": 487.5, "sets": 5 }
  ]
}
```

---

### 4. Weekly Overview

**`GET /analytics/weekly_summary?cycle=1&week=3`**
```json
{
  "cycle": 1,
  "week": 3,
  "days": [
    { "day": 1, "set_count": 15 },
    { "day": 2, "set_count": 12 }
  ]
}
```

**`GET /analytics/volume?cycle=1&week=3`**
```json
{
  "period": { "cycle": 1, "week": 3 },
  "exercises": [
    { "exercise": "Back Squat", "total_volume": 2500.0, "unit": "kg" }
  ]
}
```

**`GET /analytics/training_density?cycle=1&week=3`**
```json
{
  "days": [
    { "date": "2026-02-10", "sets": 15, "total_volume": 1200.0 }
  ]
}
```

**`GET /analytics/consistency`**
```json
{
  "weeks": [
    { "cycle": 1, "week": 1, "complete": true, "days_logged": 4 }
  ]
}
```

---

### 5. Metcon / WOD Results

**`GET /metcons`** — All metcons (supports filters)
Query params: `name`, `workout_type`, `rx`, `start`, `end`, `tag`, `limit` (default 200)

**`GET /analytics/metcon_prs`** — Best score per benchmark
```json
{
  "benchmarks": [
    {
      "name": "Fran",
      "workout_type": "for_time",
      "best_time_seconds": 185,
      "best_score_display": "3:05",
      "rx": "rx",
      "date": "2026-01-15"
    }
  ]
}
```

**`GET /analytics/metcon_prs?name=Fran`** — PR for one benchmark

**`GET /analytics/metcon_timeline?name=Fran`**
```json
{
  "name": "Fran",
  "data": [
    { "date": "2025-06-01", "score_time_seconds": 245, "score_display": "4:05", "rx": "rx" },
    { "date": "2026-01-15", "score_time_seconds": 185, "score_display": "3:05", "rx": "rx" }
  ]
}
```

---

### 6. Exercise Explorer / Search

**`GET /debug/exercises`**
```json
[
  { "exercise": "Back Squat", "count": 45 },
  { "exercise": "Bench Press", "count": 30 }
]
```

**`GET /workouts/search?q=squat`** — Search exercises by name

**`GET /analytics/movement_frequency`**
```json
{
  "movements": [
    { "exercise": "Back Squat", "sessions": 12, "sets": 60 }
  ]
}
```

**`GET /analytics/intensity_distribution?exercise=Back+Squat`**
```json
{
  "exercise": "Back Squat",
  "brackets": {
    "heavy_1_3": 10,
    "strength_4_6": 25,
    "hypertrophy_7_12": 8,
    "endurance_13_plus": 2
  }
}
```

---

### 7. Day Detail

**`GET /day_summary?cycle=1&week=3&day=1`**
```json
{
  "cycle": 1,
  "week": 3,
  "day": 1,
  "workouts": [ /* WorkoutOut[] */ ],
  "metcons": [ /* MetconOut[] */ ]
}
```

---

## Recommended Next.js Project Setup

```bash
npx create-next-app@latest cf-dashboard \
  --typescript --tailwind --eslint --app --src-dir

# Install recommended packages
npm install @tanstack/react-query recharts lucide-react
```

**`.env.local`:**
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**Suggested folder structure:**
```
src/
  app/
    page.tsx                  # Home / Stats
    prs/page.tsx              # PR Board
    progress/[exercise]/page.tsx  # Lift Timeline
    weekly/page.tsx           # Weekly Overview
    metcons/page.tsx          # WOD Results
    exercises/page.tsx        # Exercise Explorer
  lib/
    api.ts                    # fetch wrapper using NEXT_PUBLIC_API_URL
  components/
    charts/                   # Recharts wrappers
    ui/                       # Shared UI components
```

**`src/lib/api.ts` starter:**
```ts
const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000"

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, init)
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`)
  return res.json() as Promise<T>
}
```

---

## Notes for the New Session

- API docs (Swagger): `http://localhost:8000/docs`
- All dates are `YYYY-MM-DD` strings
- Exercise and metcon names are Title-Cased by the API
- Tags are lowercase, comma-separated strings
- `cycle/week/day` is the training structure (not calendar week — use `iso_week` for calendar)
- For timed metcons, scores are stored in **seconds** (`score_time_seconds`); use `score_display` for UI
