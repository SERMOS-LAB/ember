"""Tests for ember.destination – destination inference and classification."""

import pandas as pd
import numpy as np
import pytest

from ember.destination import infer_destinations, classify_destinations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_stops(records):
    """Build a stops DataFrame from list of dicts."""
    return pd.DataFrame(records)


def _make_homes(records):
    """Build a homes DataFrame from list of dicts."""
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# infer_destinations
# ---------------------------------------------------------------------------

class TestInferDestinations:

    def test_basic_single_destination(self):
        """Evacuee with one overnight stop away from home → one destination."""
        stops = _make_stops([
            {"ID": "A", "stop_lat_4326": 34.10, "stop_lon_4326": -118.30,
             "stop_date": "2025-01-08", "duration": 480},
        ])
        homes = _make_homes([
            {"ID": "A", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
        ])
        result = infer_destinations(stops, homes)
        assert len(result) == 1
        assert result.iloc[0]["ID"] == "A"
        assert result.iloc[0]["eu_distance_km"] > 0

    def test_home_buffer_filtering(self):
        """Stop very close to home should be excluded."""
        stops = _make_stops([
            {"ID": "A", "stop_lat_4326": 34.0501, "stop_lon_4326": -118.2501,
             "stop_date": "2025-01-08", "duration": 480},
        ])
        homes = _make_homes([
            {"ID": "A", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
        ])
        result = infer_destinations(stops, homes, home_buffer_m=1000)
        assert len(result) == 0

    def test_stop_merging(self):
        """Two consecutive stops within merge_distance → merged to one."""
        stops = _make_stops([
            {"ID": "A", "stop_lat_4326": 34.10, "stop_lon_4326": -118.30,
             "stop_date": "2025-01-08", "duration": 480},
            {"ID": "A", "stop_lat_4326": 34.1001, "stop_lon_4326": -118.3001,
             "stop_date": "2025-01-09", "duration": 480},
        ])
        homes = _make_homes([
            {"ID": "A", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
        ])
        result = infer_destinations(stops, homes, merge_distance_km=0.4)
        assert len(result) == 1

    def test_stops_far_apart_not_merged(self):
        """Two consecutive stops far apart → two destinations."""
        stops = _make_stops([
            {"ID": "A", "stop_lat_4326": 34.10, "stop_lon_4326": -118.30,
             "stop_date": "2025-01-08", "duration": 480},
            {"ID": "A", "stop_lat_4326": 34.40, "stop_lon_4326": -118.60,
             "stop_date": "2025-01-09", "duration": 480},
        ])
        homes = _make_homes([
            {"ID": "A", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
        ])
        result = infer_destinations(stops, homes, merge_distance_km=0.4)
        assert len(result) == 2
        assert list(result["dest_order"]) == [1, 2]

    def test_empty_stops(self):
        """No stops → empty result."""
        stops = _make_stops([])
        homes = _make_homes([
            {"ID": "A", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
        ])
        # Need to add expected columns to empty stops
        for c in ["ID", "stop_lat_4326", "stop_lon_4326", "stop_date"]:
            if c not in stops.columns:
                stops[c] = pd.Series(dtype="object")
        result = infer_destinations(stops, homes)
        assert len(result) == 0

    def test_multiple_evacuees(self):
        """Two different evacuees → destinations for each."""
        stops = _make_stops([
            {"ID": "A", "stop_lat_4326": 34.10, "stop_lon_4326": -118.30,
             "stop_date": "2025-01-08", "duration": 480},
            {"ID": "B", "stop_lat_4326": 34.20, "stop_lon_4326": -118.40,
             "stop_date": "2025-01-08", "duration": 480},
        ])
        homes = _make_homes([
            {"ID": "A", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
            {"ID": "B", "home_lat_4326": 34.05, "home_lon_4326": -118.25},
        ])
        result = infer_destinations(stops, homes)
        assert len(result) == 2
        assert set(result["ID"]) == {"A", "B"}


# ---------------------------------------------------------------------------
# classify_destinations — without parcel data
# ---------------------------------------------------------------------------

class TestClassifyDestinations:

    def test_no_parcels_returns_nan(self):
        """Without parcel data, dest_type should be NaN."""
        destinations = pd.DataFrame({
            "ID": ["A"],
            "dest_lat": [34.10],
            "dest_lon": [-118.30],
            "dest_date": ["2025-01-08"],
            "eu_distance_km": [8.5],
            "dest_order": [1],
        })
        result = classify_destinations(destinations, parcels=None)
        assert "dest_type" in result.columns
        assert pd.isna(result.iloc[0]["dest_type"])
