"""Step 2 tests: FIT parsing, metrics, activity storage, and the full pipeline."""

import json
import math
from pathlib import Path

import pytest

from src.tools.fit_parser import parse_fit_file
from src.tools.metrics import (
    calculate_trimp,
    calculate_hr_zones,
    classify_hr_zone,
    calculate_pace_zones,
)
from src.tools.activity_store import store_activity, list_activities, get_weekly_summary

FIXTURES = Path(__file__).parent / "fixtures"


# ── FIT Parser (JSON fixtures) ──────────────────────────────────────────

class TestFitParser:
    def test_parse_easy_run(self):
        data = parse_fit_file(str(FIXTURES / "easy_run.json"))
        assert data["sport"] == "running"
        assert data["duration_seconds"] == 2700
        assert data["distance_meters"] == 7800
        assert data["heart_rate"]["avg"] == 135

    def test_parse_interval_run(self):
        data = parse_fit_file(str(FIXTURES / "interval_run.json"))
        assert data["sport"] == "running"
        assert data["heart_rate"]["avg"] == 162
        assert data["heart_rate"]["max"] == 185

    def test_parse_bike_ride(self):
        data = parse_fit_file(str(FIXTURES / "long_bike.json"))
        assert data["sport"] == "cycling"
        assert data["power"]["avg_watts"] == 180
        assert data["distance_meters"] == 45000

    def test_parse_gym_session(self):
        data = parse_fit_file(str(FIXTURES / "gym_session.json"))
        assert data["sport"] == "strength"
        assert data["distance_meters"] is None
        assert data["pace"] is None

    def test_parse_nonexistent_file(self):
        with pytest.raises(FileNotFoundError):
            parse_fit_file("/nonexistent/file.json")


# ── Metrics ──────────────────────────────────────────────────────────────

class TestMetrics:
    def test_trimp_easy_run(self):
        """45 min easy run at HR 135, should give TRIMP 50-150."""
        trimp = calculate_trimp(duration_minutes=45, avg_hr=135, rest_hr=60, max_hr=190)
        assert 50 <= trimp <= 150, f"TRIMP {trimp} not in expected range 50-150"

    def test_trimp_hard_interval(self):
        """50 min interval at HR 162 should give higher TRIMP than easy run."""
        easy = calculate_trimp(duration_minutes=45, avg_hr=135, rest_hr=60, max_hr=190)
        hard = calculate_trimp(duration_minutes=50, avg_hr=162, rest_hr=60, max_hr=190)
        assert hard > easy, f"Hard TRIMP {hard} should be > easy TRIMP {easy}"

    def test_trimp_zero_duration(self):
        trimp = calculate_trimp(duration_minutes=0, avg_hr=135, rest_hr=60, max_hr=190)
        assert trimp == 0.0

    def test_trimp_invalid_hr_range(self):
        with pytest.raises(ValueError):
            calculate_trimp(duration_minutes=45, avg_hr=135, rest_hr=190, max_hr=60)

    def test_hr_zones_five_zones(self):
        zones = calculate_hr_zones(rest_hr=60, max_hr=190)
        assert len(zones) == 5
        for i in range(1, 6):
            assert i in zones
            assert "low_hr" in zones[i]
            assert "high_hr" in zones[i]
            assert "name" in zones[i]

    def test_hr_zones_boundaries_ascending(self):
        zones = calculate_hr_zones(rest_hr=60, max_hr=190)
        for i in range(1, 5):
            assert zones[i]["high_hr"] <= zones[i + 1]["high_hr"]

    def test_hr_zones_zone1_starts_at_50pct_hrr(self):
        zones = calculate_hr_zones(rest_hr=60, max_hr=190)
        # 50% of HRR (130) + rest (60) = 125
        assert zones[1]["low_hr"] == 125

    def test_classify_hr_zone_easy(self):
        """HR 135 with rest 60 / max 190 => ~58% HRR => Zone 1."""
        zone = classify_hr_zone(135, rest_hr=60, max_hr=190)
        assert zone in (1, 2)  # 135 is right around the Z1/Z2 boundary

    def test_classify_hr_zone_hard(self):
        """HR 172 => ~86% HRR => Zone 4."""
        zone = classify_hr_zone(172, rest_hr=60, max_hr=190)
        assert zone == 4

    def test_classify_hr_zone_max(self):
        zone = classify_hr_zone(190, rest_hr=60, max_hr=190)
        assert zone == 5

    def test_pace_zones(self):
        zones = calculate_pace_zones(threshold_pace_min_km=5.0)
        assert len(zones) == 5
        # Zone 1 should be slowest (higher min/km), Zone 5 fastest
        assert zones[2]["slow_min_km"] > zones[4]["fast_min_km"]


# ── Activity Store ───────────────────────────────────────────────────────

class TestActivityStore:
    def test_store_and_list(self, tmp_path):
        activity = parse_fit_file(str(FIXTURES / "easy_run.json"))
        store_activity(activity, storage_dir=tmp_path)

        activities = list_activities(storage_dir=tmp_path)
        assert len(activities) == 1
        assert activities[0]["sport"] == "running"

    def test_store_multiple_and_filter_by_sport(self, tmp_path):
        for fixture in ["easy_run.json", "interval_run.json", "long_bike.json", "gym_session.json"]:
            activity = parse_fit_file(str(FIXTURES / fixture))
            store_activity(activity, storage_dir=tmp_path)

        all_acts = list_activities(storage_dir=tmp_path)
        assert len(all_acts) == 4

        runs = list_activities(storage_dir=tmp_path, sport="running")
        assert len(runs) == 2

        cycling = list_activities(storage_dir=tmp_path, sport="cycling")
        assert len(cycling) == 1

    def test_filter_by_date(self, tmp_path):
        for fixture in ["easy_run.json", "long_bike.json"]:
            activity = parse_fit_file(str(FIXTURES / fixture))
            store_activity(activity, storage_dir=tmp_path)

        # easy_run is 2026-02-03, long_bike is 2026-02-08
        recent = list_activities(storage_dir=tmp_path, after="2026-02-05")
        assert len(recent) == 1
        assert recent[0]["sport"] == "cycling"

    def test_weekly_summary(self, tmp_path):
        for fixture in ["easy_run.json", "interval_run.json", "long_bike.json", "gym_session.json"]:
            activity = parse_fit_file(str(FIXTURES / fixture))
            store_activity(activity, storage_dir=tmp_path)

        activities = list_activities(storage_dir=tmp_path)
        summary = get_weekly_summary(activities)

        assert summary["total_sessions"] == 4
        assert summary["total_duration_minutes"] > 0
        assert summary["total_distance_km"] > 0
        assert summary["avg_hr"] is not None
        assert "running" in summary["sessions_by_sport"]
        assert summary["sessions_by_sport"]["running"] == 2

    def test_empty_summary(self):
        summary = get_weekly_summary([])
        assert summary["total_sessions"] == 0


# ── Integration: Full Pipeline ───────────────────────────────────────────

class TestStep2Integration:
    def test_full_pipeline(self, tmp_path):
        """Parse all fixtures -> compute TRIMP -> store -> list -> summarize."""
        fixtures = ["easy_run.json", "interval_run.json", "long_bike.json", "gym_session.json"]

        for fixture in fixtures:
            activity = parse_fit_file(str(FIXTURES / fixture))

            # Calculate TRIMP for activities with HR data
            hr = activity.get("heart_rate")
            if hr and hr.get("avg"):
                dur_min = activity["duration_seconds"] / 60
                trimp = calculate_trimp(dur_min, hr["avg"])
                zone = classify_hr_zone(hr["avg"])
                activity["trimp"] = trimp
                activity["hr_zone"] = zone

            store_activity(activity, storage_dir=tmp_path)

        # Retrieve and summarize
        activities = list_activities(storage_dir=tmp_path)
        assert len(activities) == 4

        summary = get_weekly_summary(activities)
        assert summary["total_sessions"] == 4
        # 2700 + 3000 + 5400 + 3600 = 14700 sec = 245 min
        assert summary["total_duration_minutes"] == 245.0
        # 7800 + 9200 + 45000 + 0 = 62000m = 62 km
        assert summary["total_distance_km"] == 62.0
        assert summary["avg_hr"] is not None
        assert summary["sessions_by_sport"] == {
            "running": 2, "cycling": 1, "strength": 1
        }

        # Verify TRIMP was calculated and stored
        for act in activities:
            if act.get("heart_rate"):
                assert "trimp" in act
                assert act["trimp"] > 0
