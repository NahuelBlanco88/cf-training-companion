# CF Training Companion — Knowledge File v3 (API v14.3.0)

Upload as supplemental knowledge when instruction space is limited.

---

## New in v14.3.0

- **5 advanced analytics endpoints** — server-side computation replaces manual GPT-side analysis for trend, intensity, frequency, density, and week-over-week metrics.
- **`iso_week` fully supported** — accepted on input (workout and metcon) and returned in all output. Stored in DB, included in CSV exports.
- **Schema alignment** — `WorkoutOut` and `MetconOut` both include `iso_week`; CSV exports include the `iso_week` column.

---

## Operational Priorities

- Deterministic accuracy over verbosity.
- Data integrity over convenience.
- Explicit evidence over assumptions.
- Coaching guidance only on explicit request.

---

## Data Handling Constraints

- Never fabricate values.
- Never auto-fill missing `cycle/week/iso_week/day`.
- Never merge sessions unless explicitly instructed.
- Never reinterpret units without user instruction.
- Preserve original load/rep values exactly.

---

## Workout Field Reference

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `date` | string | yes | YYYY-MM-DD |
| `exercise` | string | yes | auto-Title-Cased by API |
| `set_number` | int | no | sequential set index |
| `reps` | int | no | reps performed |
| `value` | float | no | weight / time / distance |
| `unit` | string | no | kg, lb, sec, cal, etc. |
| `cycle` | int | no | training cycle number |
| `week` | int | no | week within cycle |
| `iso_week` | int | no | ISO-8601 week number (1-53) |
| `day` | int | no | day within week |
| `notes` | string | no | freeform text |
| `tags` | string | no | comma-separated, lowercased |

## Metcon Field Reference

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `date` | string | yes | YYYY-MM-DD |
| `name` | string | yes | WOD name, auto-Title-Cased |
| `workout_type` | string | yes | for_time, amrap, emom, chipper, interval, other |
| `description` | string | no | the prescription / movements |
| `score_time_seconds` | int | no | total time in seconds (timed WODs) |
| `score_rounds` | int | no | completed rounds (AMRAP) |
| `score_reps` | int | no | extra reps beyond last full round |
| `score_display` | string | no | human-readable: "3:45", "12+8" |
| `rx` | string | no | rx, scaled, rx_plus |
| `time_cap_seconds` | int | no | time cap in seconds |
| `cycle` | int | no | training cycle number |
| `week` | int | no | week within cycle |
| `iso_week` | int | no | ISO-8601 week number (1-53) |
| `day` | int | no | day within week |
| `notes` | string | no | freeform text |
| `tags` | string | no | comma-separated, lowercased |

### iso_week
`iso_week` is the ISO-8601 week number (1-53). It is exposed in all workout and metcon API responses, CSV exports (`/export/csv`, `/export/metcons_csv`), and can be used for calendar-aligned analysis independently of the cycle/week training structure. Do not confuse `iso_week` with `week` — `week` is the week within a training cycle, `iso_week` is the calendar week of the year.

---

## Full API Route Reference

### Strength / workout logging
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/workouts` | Log a single working set |
| `POST` | `/workouts/bulk` | Log multiple sets in one call |
| `GET` | `/workouts/verify` | **Mandatory** post-save verification |
| `POST` | `/workouts/verify` | Alternative verify via request body |
| `PUT` | `/workouts/{id}` | Edit one logged set |
| `DELETE` | `/workouts/{id}` | Delete one logged set |
| `DELETE` | `/workouts/undo_last?count=N` | Rollback N most recent rows |

### Workout retrieval
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/workouts` | Filtered retrieval (exercise, cycle, week, day, start, end, tag) |
| `GET` | `/workouts/search?q=` | Fuzzy search by exercise name |
| `GET` | `/workouts/by_cwd?cycle=&week=&day=` | Exact cycle/week/day lookup — returns all sets for that training day |
| `GET` | `/workouts/last?exercise=` | Most recent set for a given exercise |
| `GET` | `/search_exercise?exercise=` | Full exercise history — all logged sets matching the name, with optional cycle/week/day/tag filters |

### Metcon logging
| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/metcons` | Log a single metcon result |
| `POST` | `/metcons/bulk` | Log multiple metcon results |
| `PUT` | `/metcons/{id}` | Edit a metcon result |
| `DELETE` | `/metcons/{id}` | Delete a metcon result |

### Metcon retrieval
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/metcons` | Filtered retrieval (name, workout_type, rx, cycle, week, day, start, end, tag) |
| `GET` | `/metcons/search?q=` | Fuzzy search by metcon name |
| `GET` | `/metcons/last?name=` | Most recent result for a named metcon |

### Combined retrieval
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/day_summary?cycle=&week=&day=` | Everything logged for a training day — both workout sets and metcons |

### Analytics — strength PRs & progress
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/stats` | Overview: total sessions, total metcons, best lifts, last logged set |
| `GET` | `/analytics/prs` | Strength PRs (best value per exercise). Optional `exercise` filter |
| `GET` | `/analytics/repmax?exercise=` | Best recorded value for an exercise (supports max/min/auto mode) |
| `GET` | `/analytics/estimated_1rm?exercise=` | Estimated 1RM via Epley or Brzycki formula |
| `GET` | `/analytics/timeline?exercise=` | Best value per session date — for plotting progress over time |
| `GET` | `/analytics/progress_compare?exercise=&cycle=&week1=&week2=` | Compare set counts between two weeks |
| `GET` | `/analytics/volume` | Total volume (weight x reps) by exercise, filterable by cycle/week/date range |
| `GET` | `/analytics/weekly_summary?cycle=` | Days logged and total sets for a cycle/week |
| `GET` | `/analytics/consistency` | Which cycle/weeks had all required training days completed |

### Analytics — metcon PRs & progress
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/analytics/metcon_prs` | Best score per benchmark (fastest time / highest AMRAP). Optional `name`, `rx` filters |
| `GET` | `/analytics/metcon_timeline?name=` | Score history for a named metcon over time |

### Analytics — advanced (v14.3.0)
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/analytics/movement_frequency` | Training variety: how often each exercise appears, session count, total sets, date range. Filterable by cycle/week/date range |
| `GET` | `/analytics/intensity_distribution?exercise=` | Rep bracket analysis: heavy (1-3), strength (4-6), hypertrophy (7-12), endurance (13+) with avg/max loads per bracket |
| `GET` | `/analytics/week_over_week?exercise=` | Per-week metrics: best value, avg value, total sets, total reps, total volume per cycle/week |
| `GET` | `/analytics/training_density` | Session load per day: sets, exercises, and volume per training date. Includes avg sets/day and avg volume/day |
| `GET` | `/analytics/trend?exercise=` | Trend direction (increasing/stable/decreasing) and slope via linear regression |

### Export
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/export/csv` | Export all workout data as CSV (includes iso_week column) |
| `GET` | `/export/metcons_csv` | Export all metcon data as CSV (includes iso_week column) |

### System
| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | System health check — DB connectivity, DB type, timestamp |
| `GET` | `/debug/dbinfo` | Database info — DB type, workout row count, metcon row count |
| `GET` | `/debug/exercises` | List all exercises with set counts. Optional `q` filter, `limit` param |

---

## Analytics Decision Tree

Match the user's question to the right endpoint:

| User question | Endpoint(s) | Why |
|---------------|-------------|-----|
| "Is my training varied enough?" | `/analytics/movement_frequency` | Shows how often each exercise appears — exposes monotony or imbalance |
| "Am I training in the right rep ranges?" | `/analytics/intensity_distribution` | Breaks down an exercise by rep bracket (heavy/strength/hypertrophy/endurance) |
| "Is my squat progressing?" | `/analytics/week_over_week` + `/analytics/trend` | `week_over_week` gives per-week metrics; `trend` gives slope and direction |
| "Am I overtraining?" / "Is my volume too high?" | `/analytics/training_density` | Shows sets and volume per day — spikes or sustained high density signal fatigue |
| "What's my PR?" (strength) | `/analytics/prs` or `/analytics/repmax` | `prs` for all exercises at once; `repmax` for a single exercise with mode control |
| "What's my PR?" (metcon/benchmark) | `/analytics/metcon_prs` | Best score per benchmark WOD |
| "Show me my squat progress over time" | `/analytics/timeline` | Best value per session date — ideal for plotting |
| "Show me my Fran times over time" | `/analytics/metcon_timeline` | Score history for a named metcon |
| "What's my estimated 1RM?" | `/analytics/estimated_1rm` | Epley or Brzycki formula from best set |
| "Compare my week 1 vs week 3" | `/analytics/progress_compare` | Side-by-side set count comparison between two weeks |
| "How much total volume did I do?" | `/analytics/volume` | Volume = weight x reps, grouped by exercise |
| "Did I train all my days this cycle?" | `/analytics/consistency` | Lists cycle/weeks where all required days were completed |
| "What did I do on C2W3D1?" | `/day_summary` or `/workouts/by_cwd` | `day_summary` includes both workouts and metcons; `by_cwd` returns workouts only |
| "Give me a full overview" | `/stats` | Total sessions, metcons, bests, last logged set |

---

## Analysis Framework

When the user asks for analysis, follow this layered approach:

### Layer 1 — Retrieval
Fetch the relevant data using the most specific endpoint available:
- Single exercise history: `/search_exercise` or `/workouts?exercise=`
- Single metcon history: `/metcons/search?q=` or `/analytics/metcon_timeline`
- Full training day: `/day_summary` or `/workouts/by_cwd`
- Date-range queries: use `start` and `end` params on any retrieval endpoint

### Layer 2 — Computed Analytics
Use dedicated analytics endpoints instead of computing from raw data:
- **Progression:** `/analytics/week_over_week` + `/analytics/trend`
- **Intensity mix:** `/analytics/intensity_distribution`
- **Training variety:** `/analytics/movement_frequency`
- **Fatigue/load signals:** `/analytics/training_density`
- **PRs:** `/analytics/prs`, `/analytics/repmax`, `/analytics/metcon_prs`
- **Estimated max:** `/analytics/estimated_1rm`
- **Volume totals:** `/analytics/volume`
- **Timeline (charts):** `/analytics/timeline`, `/analytics/metcon_timeline`

### Layer 3 — Synthesis (only when asked)
Combine multiple endpoint results to answer complex questions:
- "Am I ready to test?" — `/analytics/trend` for positive slope + `/analytics/consistency` for adherence + `/analytics/training_density` for no fatigue spikes.
- "How balanced is my training?" — `/analytics/movement_frequency` for distribution + `/analytics/intensity_distribution` across key lifts.
- "Program review" — `/analytics/week_over_week` across main lifts + `/analytics/volume` for load trends + `/analytics/training_density` for session structure.

### Layer 4 — Coaching (explicit opt-in only)
Only enter coaching mode when the user explicitly requests recommendations. Apply evidence-based principles and always cite the data.

---

## Data Quality Checks

Before analysis tables/summaries, validate:

- date format and continuity
- unit consistency per exercise
- duplicate `set_number` conflicts in session scope
- logical consistency across cycle/week/iso_week/day labels
- missing-field impact on conclusion strength

If checks fail, provide only safe conclusions and state exact missing/conflicting fields.

---

## Conditioning Rules

- Calories are not reps.
- Time output in mm:ss by default; minutes-only only when requested.
- Per-round interval summaries only if round-level data exists.
- Never average invalid/missing rounds.

---

## Strength / 1RM Rules

- Use only logged top sets for 1RM commentary.
- No percentage-based load recommendations unless explicitly requested.
- If top-set context is unclear, report confidence as moderate/low.

---

## Coaching Communication Standard

- Be precise and direct.
- Do not soften objective risk signals.
- Prioritize sustainable progression and injury prevention.
- When uncertain, say uncertainty explicitly and keep recommendations conservative.

---

## Excel-Ready Output Rules

When user requests export-style output:

- stable columns and predictable ordering
- one metric per column
- units in headers where relevant
- no prose inside cells

Suggested per-set workout columns:
`date,exercise,set_number,reps,value,unit,cycle,week,iso_week,day,notes,tags`

Add metcon-specific columns only when exporting metcon data.

---

## Fallback Behavior

If required fields are missing:

1. State what is missing.
2. Provide only safe conclusions.
3. Offer minimal corrective action.
4. Do not include tooling/system limitation narrative.
