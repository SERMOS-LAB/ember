"""Tests for ember.activities – incremental clustering and origin inference."""

import pandas as pd
import pytest

from ember.activities import ActivityCluster, haversine_m, incremental_cluster, find_origin


# ---------------------------------------------------------------------------
# haversine_m
# ---------------------------------------------------------------------------

def test_haversine_m_same_point():
    assert haversine_m(34.0, -118.0, 34.0, -118.0) == 0.0


def test_haversine_m_known_distance():
    # Los Angeles to San Francisco ≈ 559 km
    d = haversine_m(34.0522, -118.2437, 37.7749, -122.4194)
    assert 550_000 < d < 570_000


# ---------------------------------------------------------------------------
# incremental_cluster
# ---------------------------------------------------------------------------

def _make_pings(coords_with_times):
    """Helper: list of (lat, lon, datetime_str) → DataFrame."""
    rows = [
        {"latitude": lat, "longitude": lon, "datetime": pd.Timestamp(t)}
        for lat, lon, t in coords_with_times
    ]
    return pd.DataFrame(rows)


def test_cluster_single_activity():
    """Points within R_a that last >= T_a → one cluster."""
    pings = _make_pings([
        (34.05, -118.25, "2025-01-07 09:00:00"),
        (34.05, -118.25, "2025-01-07 09:03:00"),
        (34.05, -118.25, "2025-01-07 09:06:00"),  # 6 min total
    ])
    clusters = incremental_cluster(pings, R_a=200, T_a="5min")
    assert len(clusters) == 1
    assert clusters[0].n_points == 3


def test_cluster_below_duration_threshold():
    """Points within R_a but duration < T_a → no clusters."""
    pings = _make_pings([
        (34.05, -118.25, "2025-01-07 09:00:00"),
        (34.05, -118.25, "2025-01-07 09:02:00"),  # only 2 min
    ])
    clusters = incremental_cluster(pings, R_a=200, T_a="5min")
    assert len(clusters) == 0


def test_cluster_two_separate_activities():
    """Two groups of points far apart → two clusters."""
    pings = _make_pings([
        # Activity 1 — downtown LA
        (34.05, -118.25, "2025-01-07 08:00:00"),
        (34.05, -118.25, "2025-01-07 08:10:00"),
        # Activity 2 — Santa Monica (far away)
        (34.02, -118.50, "2025-01-07 10:00:00"),
        (34.02, -118.50, "2025-01-07 10:10:00"),
    ])
    clusters = incremental_cluster(pings, R_a=200, T_a="5min")
    assert len(clusters) == 2


def test_cluster_empty_input():
    clusters = incremental_cluster(pd.DataFrame(columns=["latitude", "longitude", "datetime"]))
    assert clusters == []


# ---------------------------------------------------------------------------
# find_origin
# ---------------------------------------------------------------------------

def test_find_origin_at_home():
    """Evacuee was at home during fire → origin = home."""
    home_lat, home_lon = 34.05, -118.25
    pings = _make_pings([
        (34.05, -118.25, "2025-01-07 10:00:00"),  # at home after fire
        (34.05, -118.25, "2025-01-07 10:05:00"),
    ])
    lat, lon, origin_type = find_origin(
        pings, home_lat, home_lon, fire_start=pd.Timestamp("2025-01-07 09:00:00")
    )
    assert origin_type == "home"
    assert lat == home_lat


def test_find_origin_at_activity():
    """Evacuee was NOT at home → origin = activity location."""
    home_lat, home_lon = 34.05, -118.25
    # Pings are 10+ km from home (Santa Monica area)
    pings = _make_pings([
        (34.02, -118.50, "2025-01-07 08:50:00"),
        (34.02, -118.50, "2025-01-07 08:56:00"),  # 6 min pre-fire activity
        (34.02, -118.50, "2025-01-07 09:10:00"),  # spans fire start
    ])
    lat, lon, origin_type = find_origin(
        pings, home_lat, home_lon,
        fire_start=pd.Timestamp("2025-01-07 09:00:00"),
        home_radius=200.0,
    )
    assert origin_type == "activity"
    assert abs(lat - 34.02) < 0.01
