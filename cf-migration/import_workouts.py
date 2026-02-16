#!/usr/bin/env python3
"""Import workouts from CSV into CF-Log API v12.

Handles both old-format CSVs (with sets/rpe/plan_day_id columns) and
new-format CSVs (with set_number column). Sends only v12-compatible fields.

Usage: python3 import_workouts.py <csv_file> <api_url>
"""
import sys, csv, requests, json
from time import sleep


def _parse_reps(raw: str) -> int | None:
    """Parse reps from CSV — handles integers, '1+1+1' sums, '0:20' times."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    # Plain integer
    try:
        return int(raw)
    except ValueError:
        pass
    # Sum notation like "1+1+1"
    if '+' in raw:
        try:
            return sum(int(p) for p in raw.split('+'))
        except ValueError:
            pass
    # Time notation like "0:20" — store as seconds
    if ':' in raw:
        try:
            parts = raw.split(':')
            return int(parts[0]) * 60 + int(parts[1])
        except (ValueError, IndexError):
            pass
    return None


def _parse_int(raw: str) -> int | None:
    if not raw or not raw.strip():
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def import_workouts(csv_file, api_url):
    if not api_url.startswith('http'):
        api_url = f'https://{api_url}'
    api_url = api_url.rstrip('/')

    print(f"Reading CSV: {csv_file}")
    print(f"Target API: {api_url}")

    try:
        health = requests.get(f"{api_url}/health", timeout=10)
        health.raise_for_status()
        print(f"API is healthy\n")
    except Exception as e:
        print(f"Cannot connect: {e}")
        sys.exit(1)

    workouts = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        has_set_number = 'set_number' in headers

        for row in reader:
            workout = {
                'date': row['date'],
                'exercise': row['exercise'],
                'value': float(row['value']) if row.get('value') else None,
                'unit': row.get('unit') if row.get('unit') else None,
                'reps': _parse_reps(row.get('reps', '')),
                'cycle': _parse_int(row.get('cycle', '')),
                'week': _parse_int(row.get('week', '')),
                'day': _parse_int(row.get('day', '')),
                'notes': row.get('notes', '') or '',
                'tags': row.get('tags', '') or None,
            }
            if has_set_number:
                workout['set_number'] = _parse_int(row.get('set_number', ''))
            # Remove None values to keep payload clean
            workout = {k: v for k, v in workout.items() if v is not None}
            workouts.append(workout)

    print(f"Found {len(workouts)} workouts\n")

    success = 0
    failed = 0

    for i, workout in enumerate(workouts, 1):
        try:
            response = requests.post(
                f"{api_url}/workouts?force=true",
                json=workout, timeout=10,
            )
            response.raise_for_status()
            success += 1
            if i % 50 == 0:
                print(f"Progress: {i}/{len(workouts)}...")
        except Exception as e:
            failed += 1
            print(f"Failed workout {i}: {e} — {workout.get('exercise', '?')}")
            if failed > 50:
                print("Too many failures, stopping")
                break
        if i % 10 == 0:
            sleep(0.5)

    print(f"\n{'='*60}")
    print(f"Imported: {success}")
    print(f"Failed: {failed}")
    print(f"{'='*60}\n")

    try:
        dbinfo = requests.get(f"{api_url}/debug/dbinfo", timeout=10)
        info = dbinfo.json()
        print(f"Database now has: {info['workout_rows']} workouts, {info['metcon_rows']} metcons")
        print(f"Type: {info['db_type']}")
    except Exception as e:
        print(f"Could not verify: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 import_workouts.py <csv_file> <api_url>")
        sys.exit(1)
    import_workouts(sys.argv[1], sys.argv[2])
