---
name: training-log
description: Use CF Training Companion MCP tools to log performed CrossFit strength sets and metcon results, retrieve training history, analyze performance, or correct an identified logged record. Use when a user asks about their workout log or wants a completed session saved.
---

# CF training log

Use the connected CF Training Companion MCP tools for personal training results. The API's authoritative tables are `public.workout` and `public.metcon` in the user's `cf_log` database. Do not use SQL or infer anything from similarly named plural tables. Report only records actually returned by the tools.

## Preserve the user's data

- One performed working set is one strength row. One performed metcon result is one metcon row. Ignore warm-ups unless the user explicitly asks to log them.
- Require a known training date for logging. Do not fill in an unknown cycle, week, ISO week, day, result, score, or unit. Ask only for the minimum missing fact needed to save the result.
- Preserve the user's units and performed values. Do not convert units, create estimated sets, merge separate sessions, or invent results.
- Treat duplicate conflicts and uncertain write outcomes as reasons to inspect stored records. Never force, silently retry, undo, or delete a record.

## Read and analyze

- Use `get_training_day` when cycle, week, and day are known. Use `find_workouts` or `find_metcons` for date ranges and filters. Their 500-row limit and oldest-first ordering mean a broad result may be incomplete; narrow the query before claiming full history.
- Use `get_last_workout` or `get_last_metcon` for the most recent performance. `find_workouts` with `limit=1` is oldest first, so it does not identify the latest result.
- Use `analyze_training` for PRs, trends, volume, estimated 1RM, frequency, intensity, training density, and metcon history. Keep units and RX levels separate; label estimates as estimates. Do not diagnose overtraining from a single trend or density metric.
- For coaching, distinguish observations supported by logged data from advice. State the dates and scope of the records used.

## Log completed sessions

1. Extract the date, exercise or metcon name, and only the values the user supplied. If logging several working sets, send one `WorkoutInput` per set with the actual set number to `log_workout_sets`.
2. Send a single completed metcon to `log_metcon`, with timed results in seconds or AMRAP rounds and extra reps as supplied.
3. Check the returned IDs and `verified` flag. Confirm success only for verified rows. When verification is false or the call outcome is unknown, read by ID or search the date before any retry.
4. On a duplicate conflict, show the existing record ID when available and ask whether the user intends a correction. Do not force a second insert.

## Correct a result

1. Resolve the exact row ID. Call `get_workout_by_id` or `get_metcon_by_id` and compare its date, name, and relevant score with the user's description.
2. For an explicit correction, send its `version` and only the fields to change to `correct_workout_set` or `correct_metcon`. An omitted field must remain unchanged.
3. A stale-version response means another change happened. Read the row again and confirm what the user wants; do not repeat the old correction automatically.
4. Check `verified` after the correction. No delete or bulk undo tool is available; do not claim that deletion was performed.

Keep API credentials and complete private exports out of responses. If the MCP connection is unavailable, say that logging or retrieval could not be completed.
