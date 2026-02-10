"""FIT file parser: extracts structured activity data from Garmin/Wahoo FIT files."""

import json
from pathlib import Path

import fitdecode


def parse_fit_file(file_path: str) -> dict:
    """Parse a FIT file and return structured activity data.

    Returns a dict with sport, duration, HR, pace, distance, power, elevation.
    Also supports loading pre-parsed JSON fixtures for testing.
    """
    path = Path(file_path)

    # Support loading pre-parsed JSON fixtures directly
    if path.suffix == ".json":
        return json.loads(path.read_text())

    # Parse actual FIT file
    sport = "unknown"
    start_time = None
    duration_seconds = None
    distance_meters = None
    hr_values = []
    speed_values = []
    power_values = []
    altitude_values = []
    calories = None

    with fitdecode.FitReader(str(path)) as fit:
        for frame in fit:
            if frame.frame_type != fitdecode.FIT_FRAME_DATA:
                continue

            if frame.name == "sport":
                sport_field = frame.get_value("sport", fallback=None)
                if sport_field:
                    sport = _normalize_sport(str(sport_field))

            elif frame.name == "session":
                start_time = _get_field(frame, "start_time")
                duration_seconds = _get_field(frame, "total_elapsed_time")
                distance_meters = _get_field(frame, "total_distance")
                calories = _get_field(frame, "total_calories")
                # Session-level HR
                if not hr_values:
                    avg_hr = _get_field(frame, "avg_heart_rate")
                    max_hr = _get_field(frame, "max_heart_rate")
                    if avg_hr and max_hr:
                        hr_values = [avg_hr, max_hr]  # fallback

            elif frame.name == "record":
                hr = _get_field(frame, "heart_rate")
                if hr and isinstance(hr, (int, float)):
                    hr_values.append(hr)
                speed = _get_field(frame, "enhanced_speed") or _get_field(frame, "speed")
                if speed and isinstance(speed, (int, float)):
                    speed_values.append(speed)
                power = _get_field(frame, "power")
                if power and isinstance(power, (int, float)):
                    power_values.append(power)
                altitude = _get_field(frame, "enhanced_altitude") or _get_field(frame, "altitude")
                if altitude and isinstance(altitude, (int, float)):
                    altitude_values.append(altitude)

    # Build result
    result = {
        "sport": sport,
        "start_time": str(start_time) if start_time else None,
        "duration_seconds": round(duration_seconds) if duration_seconds else None,
        "distance_meters": round(distance_meters) if distance_meters else None,
        "heart_rate": _build_hr(hr_values),
        "pace": _build_pace(speed_values, sport),
        "power": _build_power(power_values),
        "elevation": _build_elevation(altitude_values),
        "calories": calories,
    }

    return result


def _get_field(frame, field_name):
    """Safely get a field value from a FIT frame."""
    try:
        return frame.get_value(field_name, fallback=None)
    except (KeyError, AttributeError):
        return None


def _normalize_sport(sport: str) -> str:
    """Normalize sport names to our standard types."""
    sport = sport.lower().strip()
    mapping = {
        "running": "running",
        "cycling": "cycling",
        "swimming": "swimming",
        "training": "strength",
        "generic": "strength",
        "strength_training": "strength",
    }
    return mapping.get(sport, sport)


def _build_hr(hr_values: list) -> dict | None:
    """Build heart rate summary from recorded values."""
    if not hr_values:
        return None
    return {
        "avg": round(sum(hr_values) / len(hr_values)),
        "max": round(max(hr_values)),
        "min": round(min(hr_values)),
    }


def _build_pace(speed_values: list, sport: str) -> dict | None:
    """Build pace summary (min/km) from speed values (m/s)."""
    if not speed_values or sport not in ("running",):
        return None
    # Filter out zeros
    valid_speeds = [s for s in speed_values if s > 0.1]
    if not valid_speeds:
        return None
    avg_speed = sum(valid_speeds) / len(valid_speeds)  # m/s
    max_speed = max(valid_speeds)
    return {
        "avg_min_per_km": round(1000 / (avg_speed * 60), 2) if avg_speed > 0 else None,
        "best_min_per_km": round(1000 / (max_speed * 60), 2) if max_speed > 0 else None,
    }


def _build_power(power_values: list) -> dict | None:
    """Build power summary from recorded values."""
    if not power_values:
        return None
    return {
        "avg_watts": round(sum(power_values) / len(power_values)),
        "max_watts": round(max(power_values)),
        "normalized_watts": None,  # Would need proper NP calculation
    }


def _build_elevation(altitude_values: list) -> dict | None:
    """Build elevation summary from altitude records."""
    if len(altitude_values) < 2:
        return None
    gain = 0.0
    loss = 0.0
    for i in range(1, len(altitude_values)):
        diff = altitude_values[i] - altitude_values[i - 1]
        if diff > 0:
            gain += diff
        else:
            loss += abs(diff)
    return {
        "gain_meters": round(gain),
        "loss_meters": round(loss),
    }
