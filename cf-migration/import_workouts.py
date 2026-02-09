#!/usr/bin/env python3
import sys, csv, requests, json
from time import sleep

def import_workouts(csv_file, api_url):
    if not api_url.startswith('http'):
        api_url = f'https://{api_url}'
    api_url = api_url.rstrip('/')
    
    print(f"📂 Reading CSV: {csv_file}")
    print(f"🌐 Target API: {api_url}")
    
    try:
        health = requests.get(f"{api_url}/health", timeout=10)
        health.raise_for_status()
        print(f"✅ API is healthy\n")
    except Exception as e:
        print(f"❌ Cannot connect: {e}")
        sys.exit(1)
    
    workouts = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            workout = {
                'date': row['date'],
                'exercise': row['exercise'],
                'value': float(row['value']) if row['value'] else None,
                'unit': row['unit'] if row['unit'] else None,
                'sets': row['sets'] if row['sets'] else None,
                'reps': row['reps'] if row['reps'] else None,
                'rpe': float(row['rpe']) if row['rpe'] else None,
                'cycle': int(row['cycle']) if row['cycle'] else None,
                'week': int(row['week']) if row['week'] else None,
                'iso_week': int(row['iso_week']) if row['iso_week'] else None,
                'day': int(row['day']) if row['day'] else None,
                'plan_day_id': int(row['plan_day_id']) if row['plan_day_id'] else None,
                'notes': row['notes'] if row['notes'] else ""
            }
            workouts.append(workout)
    
    print(f"📊 Found {len(workouts)} workouts\n")
    
    success = 0
    failed = 0
    
    for i, workout in enumerate(workouts, 1):
        try:
            response = requests.post(f"{api_url}/workouts", json=workout, timeout=10)
            response.raise_for_status()
            success += 1
            if i % 50 == 0:
                print(f"Progress: {i}/{len(workouts)}...")
        except Exception as e:
            failed += 1
            print(f"❌ Failed workout {i}: {e}")
            if failed > 10:
                print("Too many failures")
                break
        if i % 10 == 0:
            sleep(0.5)
    
    print(f"\n{'='*60}")
    print(f"✅ Imported: {success}")
    print(f"❌ Failed: {failed}")
    print(f"{'='*60}\n")
    
    try:
        dbinfo = requests.get(f"{api_url}/debug/dbinfo", timeout=10)
        info = dbinfo.json()
        print(f"📊 Database now has: {info['workout_rows']} workouts")
        print(f"💾 Type: {info['db_type']}")
    except Exception as e:
        print(f"⚠️  Could not verify: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 import_workouts.py <csv_file> <api_url>")
        sys.exit(1)
    import_workouts(sys.argv[1], sys.argv[2])
