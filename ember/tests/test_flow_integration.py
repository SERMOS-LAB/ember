"""Integration and stress tests for chain-first flow."""

import pandas as pd

from ember import departure, destination, pipeline


def test_golden_chain_first_flow():
    homes = pd.DataFrame(
        [{"ID": "u1", "home_lat_4326": 34.0500, "home_lon_4326": -118.2500}]
    )
    dau = pd.DataFrame(
        [
            {"ID": "u1", "LAT": 34.0500, "LONG": -118.2500, "datetime": "2025-01-01 08:00:00+00:00"},
            {"ID": "u1", "LAT": 34.0501, "LONG": -118.2499, "datetime": "2025-01-01 09:00:00+00:00"},
            {"ID": "u1", "LAT": 34.2000, "LONG": -118.4000, "datetime": "2025-01-02 08:00:00+00:00"},
            {"ID": "u1", "LAT": 34.2001, "LONG": -118.4001, "datetime": "2025-01-02 10:00:00+00:00"},
            {"ID": "u1", "LAT": 34.2200, "LONG": -118.4200, "datetime": "2025-01-03 08:00:00+00:00"},
            {"ID": "u1", "LAT": 34.2201, "LONG": -118.4201, "datetime": "2025-01-03 10:00:00+00:00"},
        ]
    )
    chain = pipeline.compute_trip_chain(dau, homes, destination_min_dwell_s=300)
    assert not chain.empty

    dests = destination.infer_destinations(chain, homes)
    assert not dests.empty
    assert dests.iloc[0]["ID"] == "u1"

    dep, ret, origin_type = departure.infer(pd.DataFrame(), trip_chain=chain)
    assert dep is not None
    assert ret is not None
    assert origin_type == "activity"


def test_sparse_ping_does_not_create_false_clusters():
    homes = pd.DataFrame(
        [{"ID": "u2", "home_lat_4326": 34.0500, "home_lon_4326": -118.2500}]
    )
    dau = pd.DataFrame(
        [
            {"ID": "u2", "LAT": 34.0500, "LONG": -118.2500, "datetime": "2025-01-01 08:00:00+00:00"},
            {"ID": "u2", "LAT": 34.3000, "LONG": -118.6000, "datetime": "2025-01-01 08:03:00+00:00"},
            {"ID": "u2", "LAT": 34.0501, "LONG": -118.2499, "datetime": "2025-01-01 08:06:00+00:00"},
        ]
    )
    chain = pipeline.compute_trip_chain(
        dau,
        homes,
        destination_min_dwell_s=600,  # require sustained stop
        merge_nearby_stops=True,
    )
    # With sparse jitter-like hops, no valid dwell cluster should survive.
    assert chain.empty or chain["dwell_s"].max() < 600
