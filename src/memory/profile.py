"""Athlete profile creation and persistence."""

import json
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROFILE_PATH = DATA_DIR / "athlete" / "profile.json"


def create_profile(
    sports: list[str],
    event: str,
    target_date: str,
    target_time: str,
    training_days_per_week: int,
    max_session_minutes: int,
) -> dict:
    """Create an athlete profile dict from onboarding inputs."""
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "name": "Athlete",
        "sports": sports,
        "goal": {
            "event": event,
            "target_date": target_date,
            "target_time": target_time,
        },
        "fitness": {
            "estimated_vo2max": None,
            "threshold_pace_min_km": None,
            "weekly_volume_km": None,
            "trend": "unknown",
        },
        "constraints": {
            "training_days_per_week": training_days_per_week,
            "max_session_minutes": max_session_minutes,
            "available_sports": sports,
        },
        "created_at": now,
        "updated_at": now,
    }


def save_profile(profile: dict) -> Path:
    """Save athlete profile to data/athlete/profile.json. Returns the path."""
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))
    return PROFILE_PATH


def load_profile() -> dict:
    """Load athlete profile from disk. Raises FileNotFoundError if not found."""
    if not PROFILE_PATH.exists():
        raise FileNotFoundError(f"No profile found at {PROFILE_PATH}")
    return json.loads(PROFILE_PATH.read_text())
