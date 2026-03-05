"""Tests for clustering event stream API."""

import pandas as pd

from ember.mobility.clustering import (
    IncrementalClusterConfig,
    cluster_points,
    cluster_points_event_stream,
)


def test_event_stream_emits_lifecycle_events():
    df = pd.DataFrame(
        [
            {"latitude": 34.1000, "longitude": -118.3000, "datetime": "2025-01-07 12:00:00"},
            {"latitude": 34.1001, "longitude": -118.3001, "datetime": "2025-01-07 12:20:00"},
            {"latitude": 34.2000, "longitude": -118.4000, "datetime": "2025-01-07 13:00:00"},
            {"latitude": 34.2001, "longitude": -118.4001, "datetime": "2025-01-07 13:20:00"},
        ]
    )
    events = cluster_points_event_stream(
        df,
        config=IncrementalClusterConfig(radius_m=200, min_dwell_s=300),
    )
    event_types = [e.event_type for e in events]
    assert "cluster_started" in event_types
    assert "cluster_closed" in event_types


def test_event_closed_matches_batch_output():
    df = pd.DataFrame(
        [
            {"latitude": 34.1000, "longitude": -118.3000, "datetime": "2025-01-07 12:00:00"},
            {"latitude": 34.1001, "longitude": -118.3001, "datetime": "2025-01-07 12:20:00"},
        ]
    )
    summaries = cluster_points(df, config=IncrementalClusterConfig(radius_m=200, min_dwell_s=300))
    events = cluster_points_event_stream(df, config=IncrementalClusterConfig(radius_m=200, min_dwell_s=300))
    closed = [e.summary for e in events if e.event_type == "cluster_closed" and e.summary is not None]
    assert len(closed) == len(summaries)
    assert closed[0].n_points == summaries[0].n_points
