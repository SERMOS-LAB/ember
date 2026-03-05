"""Tests for route-modeling scaffold interfaces."""

import pandas as pd

from ember.mobility.routes import build_trip_legs, infer_route_candidates


def _chain():
    return pd.DataFrame(
        [
            {
                "user_id": "u1",
                "seq_idx": 1,
                "stop_id": 1,
                "start_ts": "2025-01-07 09:00:00+00:00",
                "end_ts": "2025-01-07 10:00:00+00:00",
                "dwell_s": 3600.0,
                "lat": 34.05,
                "lon": -118.25,
                "distance_from_home_m": 0.0,
                "distance_from_prev_m": None,
                "is_overnight": False,
                "stop_role": "home",
            },
            {
                "user_id": "u1",
                "seq_idx": 2,
                "stop_id": 2,
                "start_ts": "2025-01-07 11:00:00+00:00",
                "end_ts": "2025-01-07 12:00:00+00:00",
                "dwell_s": 3600.0,
                "lat": 34.10,
                "lon": -118.30,
                "distance_from_home_m": 7000.0,
                "distance_from_prev_m": 7000.0,
                "is_overnight": False,
                "stop_role": "destination",
            },
        ]
    )


def test_build_trip_legs():
    legs = build_trip_legs(_chain())
    assert len(legs) == 1
    assert legs.iloc[0]["displacement_m"] > 0


def test_infer_route_candidates_unresolved_default():
    legs = build_trip_legs(_chain())
    routes = infer_route_candidates(legs)
    assert "route_status" in routes.columns
    assert routes.iloc[0]["route_status"] == "unresolved_strategy"


def test_infer_route_candidates_custom_strategy():
    legs = build_trip_legs(_chain())

    def _strategy(df, _graph):
        out = df[["user_id", "from_seq", "to_seq"]].copy()
        out["route_status"] = "ok_custom"
        out["route_length_m"] = 1234.0
        return out

    routes = infer_route_candidates(legs, strategy_name="custom_stub", strategy_fn=_strategy)
    assert routes.iloc[0]["route_status"] == "ok_custom"
    assert routes.iloc[0]["strategy"] == "custom_stub"
