"""Tests for trip-chain intermediate stop tracking."""

import pandas as pd

from ember.mobility.trip_chain import TripChainConfig, build_trip_chain_for_user
from ember.mobility.clustering import IncrementalClusterConfig


def _pings(rows):
    return pd.DataFrame(
        [{"latitude": lat, "longitude": lon, "datetime": pd.Timestamp(ts)} for lat, lon, ts in rows]
    )


def test_trip_chain_roles_intermediate_and_overnight():
    pings = _pings(
        [
            # intermediate stop around midday
            (34.1000, -118.3000, "2025-01-07 12:00:00"),
            (34.1001, -118.3001, "2025-01-07 12:20:00"),
            # overnight stop
            (34.2000, -118.4000, "2025-01-07 21:00:00"),
            (34.2001, -118.4001, "2025-01-07 23:00:00"),
        ]
    )
    chain = build_trip_chain_for_user(
        pings,
        user_id="u1",
        home_lat=34.05,
        home_lon=-118.25,
        clustering_config=IncrementalClusterConfig(radius_m=200, min_dwell_s=300),
        home_radius_m=300,
        overnight_start_hour=20,
        overnight_end_hour=7,
    )
    assert len(chain) == 2
    assert set(chain["stop_role"]) == {"intermediate", "overnight"}


def test_trip_chain_detects_return_stop():
    pings = _pings(
        [
            (34.0500, -118.2500, "2025-01-07 09:00:00"),
            (34.0501, -118.2499, "2025-01-07 09:25:00"),
            (34.1500, -118.3500, "2025-01-07 12:00:00"),
            (34.1502, -118.3502, "2025-01-07 13:00:00"),
            (34.0500, -118.2500, "2025-01-08 10:00:00"),
            (34.0501, -118.2499, "2025-01-08 10:35:00"),
        ]
    )
    chain = build_trip_chain_for_user(
        pings,
        user_id="u2",
        home_lat=34.05,
        home_lon=-118.25,
        clustering_config=IncrementalClusterConfig(radius_m=250, min_dwell_s=300),
        rules=TripChainConfig(classify_return=True),
    )
    assert "return" in set(chain["stop_role"])


def test_trip_chain_merges_jittered_stops():
    pings = _pings(
        [
            (34.1000, -118.3000, "2025-01-07 12:00:00"),
            (34.1001, -118.3001, "2025-01-07 12:12:00"),
            (34.1010, -118.3010, "2025-01-07 12:20:00"),
            (34.1011, -118.3011, "2025-01-07 12:34:00"),
        ]
    )
    chain = build_trip_chain_for_user(
        pings,
        user_id="u3",
        home_lat=34.0,
        home_lon=-118.0,
        clustering_config=IncrementalClusterConfig(radius_m=50, min_dwell_s=300),
        rules=TripChainConfig(
            merge_nearby_stops=True,
            merge_distance_m=200,
            merge_gap_s=1800,
        ),
    )
    assert len(chain) == 1

