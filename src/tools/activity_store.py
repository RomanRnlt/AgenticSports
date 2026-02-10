"""Activity storage: persist and retrieve parsed training activities."""

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
ACTIVITIES_DIR = DATA_DIR / "activities"


def store_activity(activity_data: dict, storage_dir: str | Path | None = None) -> Path:
    """Store parsed activity as JSON. Returns the file path.

    Filename is based on start_time and sport for easy identification.
    """
    dest = Path(storage_dir) if storage_dir else ACTIVITIES_DIR
    dest.mkdir(parents=True, exist_ok=True)

    start_time = activity_data.get("start_time", "")
    sport = activity_data.get("sport", "unknown")

    # Create a sortable filename from start time
    if start_time:
        try:
            dt = datetime.fromisoformat(start_time)
            ts = dt.strftime("%Y-%m-%d_%H%M%S")
        except ValueError:
            ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    else:
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    filename = f"{ts}_{sport}.json"
    path = dest / filename
    path.write_text(json.dumps(activity_data, indent=2))
    return path


def list_activities(
    storage_dir: str | Path | None = None,
    sport: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """List stored activities with optional filters.

    Args:
        storage_dir: Directory to read from (default: data/activities/)
        sport: Filter by sport type (e.g., "running")
        after: Only include activities after this ISO date string
        before: Only include activities before this ISO date string

    Returns:
        List of activity dicts, sorted by start_time ascending.
    """
    src = Path(storage_dir) if storage_dir else ACTIVITIES_DIR
    if not src.exists():
        return []

    activities = []
    for path in sorted(src.glob("*.json")):
        data = json.loads(path.read_text())

        # Filter by sport
        if sport and data.get("sport") != sport:
            continue

        # Filter by date range
        start = data.get("start_time", "")
        if after and start and start < after:
            continue
        if before and start and start >= before:
            continue

        activities.append(data)

    return activities


def get_weekly_summary(activities: list[dict]) -> dict:
    """Summarize a collection of activities.

    Returns:
        dict with total_sessions, total_duration_minutes, total_distance_km,
        avg_hr, sessions_by_sport, total_trimp (if available).
    """
    if not activities:
        return {
            "total_sessions": 0,
            "total_duration_minutes": 0,
            "total_distance_km": 0.0,
            "avg_hr": None,
            "sessions_by_sport": {},
        }

    total_duration_sec = 0
    total_distance_m = 0
    hr_sum = 0
    hr_count = 0
    sessions_by_sport: dict[str, int] = {}

    for act in activities:
        # Duration
        dur = act.get("duration_seconds")
        if dur:
            total_duration_sec += dur

        # Distance
        dist = act.get("distance_meters")
        if dist:
            total_distance_m += dist

        # Heart rate
        hr_data = act.get("heart_rate")
        if hr_data and hr_data.get("avg"):
            hr_sum += hr_data["avg"]
            hr_count += 1

        # Sport counts
        sport = act.get("sport", "unknown")
        sessions_by_sport[sport] = sessions_by_sport.get(sport, 0) + 1

    return {
        "total_sessions": len(activities),
        "total_duration_minutes": round(total_duration_sec / 60, 1),
        "total_distance_km": round(total_distance_m / 1000, 2),
        "avg_hr": round(hr_sum / hr_count) if hr_count > 0 else None,
        "sessions_by_sport": sessions_by_sport,
    }
