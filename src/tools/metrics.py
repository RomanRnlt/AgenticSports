"""Training metrics: TRIMP, HR zones, pace zones."""

import math

# Default physiological values (can be overridden from athlete profile)
DEFAULT_REST_HR = 60
DEFAULT_MAX_HR = 190


def calculate_trimp(
    duration_minutes: float,
    avg_hr: int,
    rest_hr: int = DEFAULT_REST_HR,
    max_hr: int = DEFAULT_MAX_HR,
) -> float:
    """Calculate Training Impulse (Banister's TRIMP).

    TRIMP = duration * delta_hr_ratio * 0.64 * e^(1.92 * delta_hr_ratio)

    Returns a float representing training load. Typical values:
    - Easy 45min run: 50-150
    - Hard interval session: 150-300
    - Long endurance ride: 100-250
    """
    if max_hr <= rest_hr:
        raise ValueError(f"max_hr ({max_hr}) must be greater than rest_hr ({rest_hr})")
    if avg_hr < rest_hr:
        avg_hr = rest_hr  # clamp to avoid negative ratio

    delta_hr_ratio = (avg_hr - rest_hr) / (max_hr - rest_hr)
    trimp = duration_minutes * delta_hr_ratio * 0.64 * math.exp(1.92 * delta_hr_ratio)
    return round(trimp, 1)


def calculate_hr_zones(rest_hr: int = DEFAULT_REST_HR, max_hr: int = DEFAULT_MAX_HR) -> dict:
    """Calculate 5-zone HR model using Karvonen (Heart Rate Reserve) method.

    Zone 1: 50-60% HRR (Recovery)
    Zone 2: 60-70% HRR (Aerobic base)
    Zone 3: 70-80% HRR (Tempo)
    Zone 4: 80-90% HRR (Threshold)
    Zone 5: 90-100% HRR (VO2max)
    """
    hrr = max_hr - rest_hr
    zones = {}
    boundaries = [
        (1, 0.50, 0.60, "Recovery"),
        (2, 0.60, 0.70, "Aerobic"),
        (3, 0.70, 0.80, "Tempo"),
        (4, 0.80, 0.90, "Threshold"),
        (5, 0.90, 1.00, "VO2max"),
    ]
    for zone_num, low_pct, high_pct, name in boundaries:
        zones[zone_num] = {
            "name": name,
            "low_hr": round(rest_hr + hrr * low_pct),
            "high_hr": round(rest_hr + hrr * high_pct),
        }
    return zones


def classify_hr_zone(
    avg_hr: int, rest_hr: int = DEFAULT_REST_HR, max_hr: int = DEFAULT_MAX_HR
) -> int:
    """Return which zone (1-5) the average HR falls in.

    Returns 0 if below zone 1, 5 if above zone 5 upper bound.
    """
    zones = calculate_hr_zones(rest_hr, max_hr)
    for zone_num in range(1, 6):
        if avg_hr <= zones[zone_num]["high_hr"]:
            return zone_num
    return 5


def calculate_zone_distribution(
    hr_values: list[int],
    rest_hr: int = DEFAULT_REST_HR,
    max_hr: int = DEFAULT_MAX_HR,
    sample_interval_seconds: float = 1.0,
) -> dict:
    """Compute time in each HR zone from record-level heart rate data.

    This is the fallback for the ~5% of activity files that lack device-computed
    time_in_zone data. Each HR value is classified into a zone (1-5), and the
    count is multiplied by sample_interval_seconds to get seconds in each zone.

    Args:
        hr_values: List of heart rate values (one per sample).
        rest_hr: Resting heart rate for zone calculation.
        max_hr: Maximum heart rate for zone calculation.
        sample_interval_seconds: Time between samples (typically 1.0s for FIT records).

    Returns:
        Dict with zone_1_seconds through zone_5_seconds.
    """
    distribution = {f"zone_{i}_seconds": 0.0 for i in range(1, 6)}

    if not hr_values:
        return distribution

    zones = calculate_hr_zones(rest_hr, max_hr)

    for hr in hr_values:
        zone = classify_hr_zone(hr, rest_hr, max_hr)
        distribution[f"zone_{zone}_seconds"] += sample_interval_seconds

    return distribution


def calculate_pace_zones(threshold_pace_min_km: float) -> dict:
    """Calculate pace zones based on threshold (lactate threshold) pace.

    Zone 1: >130% of threshold pace (easy recovery)
    Zone 2: 115-130% of threshold pace (aerobic)
    Zone 3: 105-115% of threshold pace (tempo)
    Zone 4: 95-105% of threshold pace (threshold)
    Zone 5: <95% of threshold pace (VO2max intervals)
    """
    tp = threshold_pace_min_km
    return {
        1: {"name": "Recovery", "slow_min_km": None, "fast_min_km": round(tp * 1.30, 2)},
        2: {"name": "Aerobic", "slow_min_km": round(tp * 1.30, 2), "fast_min_km": round(tp * 1.15, 2)},
        3: {"name": "Tempo", "slow_min_km": round(tp * 1.15, 2), "fast_min_km": round(tp * 1.05, 2)},
        4: {"name": "Threshold", "slow_min_km": round(tp * 1.05, 2), "fast_min_km": round(tp * 0.95, 2)},
        5: {"name": "VO2max", "slow_min_km": round(tp * 0.95, 2), "fast_min_km": None},
    }
