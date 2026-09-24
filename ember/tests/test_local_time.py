"""Hour-of-day and calendar-day rules read the study area's local clock."""

import pandas as pd
import pytest

from ember import pipeline
from ember.destination import infer_destinations_from_chain
from ember.ghost import infer_homes
from ember.mobility.clustering import IncrementalClusterConfig
from ember.mobility.trip_chain import build_trip_chain_for_user
from ember.timeutil import local_clock

LA = "America/Los_Angeles"
HOME = (34.0500, -118.2500)
AWAY = (34.2000, -118.4000)
CLUSTER = IncrementalClusterConfig(radius_m=200, min_dwell_s=300)


def _pings(stamps, place=AWAY):
    return pd.DataFrame(
        [{"latitude": place[0], "longitude": place[1], "datetime": pd.Timestamp(ts)} for ts in stamps]
    )


def _chain(pings, **kw):
    return build_trip_chain_for_user(
        pings, user_id="u1", home_lat=HOME[0], home_lon=HOME[1], clustering_config=CLUSTER,
        home_radius_m=300, overnight_start_hour=20, overnight_end_hour=7, **kw,
    )


def test_local_clock_converts_aware_and_epoch_times():
    aware = pd.Series(pd.to_datetime(["2025-01-07 18:30"], utc=True))
    assert local_clock(aware, LA).iloc[0] == pd.Timestamp("2025-01-07 10:30")
    epoch = pd.Series(pd.to_datetime([1736274600000], unit="ms"))           # 2025-01-07 18:30 UTC, parsed naive
    assert local_clock(epoch, LA, assume_utc=True).iloc[0] == pd.Timestamp("2025-01-07 10:30")
    naive_local = pd.Series(pd.to_datetime(["2025-01-07 10:30"]))
    assert local_clock(naive_local, LA).iloc[0] == pd.Timestamp("2025-01-07 10:30")
    assert local_clock(pd.Timestamp("2025-01-07 18:30", tz="UTC"), LA) == pd.Timestamp("2025-01-07 10:30")


def test_afternoon_stop_in_utc_is_not_overnight_with_local_tz():
    pings = _pings(["2025-01-07 21:00:00+00:00", "2025-01-07 23:00:00+00:00"])   # 13:00 to 15:00 in Los Angeles
    assert "overnight" not in set(_chain(pings, local_tz=LA)["stop_role"])
    with pytest.warns(UserWarning, match="local_tz"):
        assert set(_chain(pings)["stop_role"]) == {"overnight"}                 # UTC clock, kept as before


def test_overnight_window_uses_the_pings_own_zone():
    pings = _pings(["2025-01-07 13:00:00-08:00", "2025-01-07 15:00:00-08:00"])
    pings["datetime"] = pd.to_datetime(pings["datetime"], utc=True).dt.tz_convert(LA)
    assert "overnight" not in set(_chain(pings)["stop_role"])


def test_evening_stop_in_utc_is_overnight_with_local_tz():
    pings = _pings(["2025-01-08 05:00:00+00:00", "2025-01-08 07:00:00+00:00"])   # 21:00 to 23:00 on 7 January
    assert set(_chain(pings, local_tz=LA)["stop_role"]) == {"overnight"}


def test_destination_date_is_the_local_day():
    pings = _pings(["2025-01-08 06:00:00+00:00", "2025-01-08 07:30:00+00:00"])   # 22:00 to 23:30 on 7 January
    chain = _chain(pings, local_tz=LA)
    homes = pd.DataFrame({"ID": ["u1"], "home_lat_4326": [HOME[0]], "home_lon_4326": [HOME[1]]})
    dest = infer_destinations_from_chain(chain, homes, local_tz=LA)
    assert list(dest["dest_date"]) == ["2025-01-07"]


def test_infer_metrics_aligns_utc_stops_with_local_order_times():
    stamps = ["2025-01-07 19:30", "2025-01-08 20:00", "2025-01-09 20:00"]     # 11:30, 12:00, 12:00 in Los Angeles
    stops = pd.DataFrame({
        "ID": "r1",
        "stop_date": pd.to_datetime(stamps, utc=True),
        "is_outside_zone": True,
        "eu_distance": 5.0,
        "ZoneType": "Order",
        "OrderStart": "2025-01-07 10:30",
        "FireEvent": "Palisades",
        "home_lon": 0.0,
    })
    out = pipeline.infer_metrics(stops, {"Palisades": "2025-01-07 10:30"}, fire_end="2025-01-14", local_tz=LA,
                                 verbose=False)
    row = out.iloc[0]
    assert row["DepartureDate"] == pd.Timestamp("2025-01-07 11:30")
    assert row["DelayHours"] == pytest.approx(1.0)
    assert row["is_evacuee"]


def test_ghost_night_hours_are_local():
    rows = []
    for day in range(1, 7):
        for local in ("22:00", "23:30"):                     # home, late evening
            ts = pd.Timestamp(f"2025-01-{day:02d} {local}", tz=LA).tz_convert("UTC")
            rows.append(("u1", *HOME, ts))
        for local in ("12:00", "13:00", "14:00"):            # workplace, midday
            ts = pd.Timestamp(f"2025-01-{day:02d} {local}", tz=LA).tz_convert("UTC")
            rows.append(("u1", 34.1000, -118.3000, ts))
    pings = pd.DataFrame(rows, columns=["user_id", "latitude", "longitude", "datetime"])
    local = infer_homes(pings.copy(), min_nights=3, local_tz=LA)
    assert abs(local["home_lat_4326"].iloc[0] - HOME[0]) < 1e-3
    with pytest.warns(UserWarning, match="local_tz"):
        utc = infer_homes(pings.copy(), min_nights=3)
    assert abs(utc["home_lat_4326"].iloc[0] - 34.1000) < 1e-3                # UTC clock picks the workplace
