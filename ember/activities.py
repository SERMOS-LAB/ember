"""
Activity inference via incremental clustering.

Implements an incremental clustering method used by both:

- **Activity-Based Origin**: clustering *pre-fire* pings to find
  where the evacuee actually was when the fire started.
- **Destination Inference**: clustering *post-fire* nightly stops
  to identify overnight destinations.
"""

from __future__ import annotations

from collections import namedtuple
from typing import List, Tuple

import pandas as pd

from .mobility.clustering import (
    IncrementalClusterConfig,
    cluster_points,
    haversine_m,
)
from .mobility.trip_chain import build_trip_chain_for_user


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ActivityCluster = namedtuple(
    "ActivityCluster",
    ["centroid_lat", "centroid_lon", "start_time", "end_time", "n_points"],
)


# ---------------------------------------------------------------------------
# Core incremental clustering (compatibility wrapper)
# ---------------------------------------------------------------------------

def incremental_cluster(
    pings: pd.DataFrame,
    *,
    R_a: float = 200.0,
    T_a: str = "5min",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
) -> List[ActivityCluster]:
    """
    Cluster a chronological sequence of GPS pings into activity locations.

    Algorithm:

    1. Start a new cluster C₀ with the first point p₀.
    2. For each subsequent point pₖ, compute distance to the current
       cluster centroid.  If d ≤ R_a, add pₖ to the cluster and update
       the centroid.  Otherwise, close the current cluster and start a
       new one.
    3. After processing all points, keep only clusters whose duration
       (last − first timestamp) ≥ T_a.

    Parameters
    ----------
    pings : DataFrame
        Must contain ``lat_col``, ``lon_col``, and ``time_col`` columns,
        sorted chronologically.
    R_a : float
        Spatial radius threshold in **metres** (default 200 m).
    T_a : str or Timedelta
        Minimum duration for a cluster to qualify as an activity
        (default ``"5min"``).
    lat_col, lon_col, time_col : str
        Column name overrides.

    Returns
    -------
    list[ActivityCluster]
        Chronologically ordered list of identified activity clusters.
    """
    config = IncrementalClusterConfig(
        radius_m=R_a,
        min_dwell_s=pd.Timedelta(T_a).total_seconds(),
    )
    summaries = cluster_points(
        pings,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        config=config,
    )
    return [
        ActivityCluster(
            centroid_lat=s.centroid_lat,
            centroid_lon=s.centroid_lon,
            start_time=s.start_time,
            end_time=s.end_time,
            n_points=s.n_points,
        )
        for s in summaries
    ]


# ---------------------------------------------------------------------------
# Activity-based origin inference
# ---------------------------------------------------------------------------

def find_origin(
    pings: pd.DataFrame,
    home_lat: float,
    home_lon: float,
    *,
    fire_start: pd.Timestamp,
    R_a: float = 200.0,
    T_a: str = "5min",
    home_radius: float = 200.0,
    max_origin_distance_m: float = 50_000.0,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
) -> Tuple[float, float, str]:
    """
    Determine the evacuation origin for a single resident.

    Uses incremental clustering on the resident's pings around the
    fire-start time.  If the resident was at home, origin = home.
    Otherwise, origin = the centroid of the activity cluster they
    were in when the fire started.

    Algorithm:

    1. Check whether the evacuee had any pings near home around or
       shortly before fire_start.  If yes → origin = home.
    2. If not → cluster *pre-fire* pings and find the cluster that
       overlaps with fire_start (start ≤ fire_start ≤ end), or the
       last cluster before fire_start.  That cluster centroid is the
       activity-based origin.
    3. Reject activity origins farther than ``max_origin_distance_m``
       from home (likely travel/transit pings, not a real activity).

    Parameters
    ----------
    pings : DataFrame
        Chronological pings for a SINGLE user.
    home_lat, home_lon : float
        Proxy home location (EPSG:4326).
    fire_start : Timestamp
        Time the fire ignition / first order was issued.
    R_a : float
        Clustering spatial radius in metres (default 200).
    T_a : str
        Minimum activity duration (default '5min').
    home_radius : float
        Buffer around home in metres (default 200).
    max_origin_distance_m : float
        Maximum distance from home for an activity origin to be
        considered valid (default 50 km). Origins beyond this
        distance fall back to home.

    Returns
    -------
    (origin_lat, origin_lon, origin_type)
        origin_type is ``'home'`` or ``'activity'``.
    """
    if pings.empty:
        return home_lat, home_lon, "home"

    df = pings.copy()
    if time_col not in df.columns and "timestamp_ms" in df.columns:
        df[time_col] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)

    # Ensure time column is tz-aware (UTC) for consistent comparisons
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], format="mixed", utc=True)
    elif df[time_col].dt.tz is None:
        df[time_col] = df[time_col].dt.tz_localize("UTC")
    else:
        df[time_col] = df[time_col].dt.tz_convert("UTC")
    df = df.sort_values(time_col).reset_index(drop=True)

    # Normalize fire_start to UTC
    fire_ts = pd.Timestamp(fire_start)
    if fire_ts.tzinfo is None:
        fire_ts = fire_ts.tz_localize("UTC")
    else:
        fire_ts = fire_ts.tz_convert("UTC")

    # ---- Step 1: Check if evacuee was at/near home around fire start ----
    # Look at pings within a 2-hour window around fire start
    window_start = fire_ts - pd.Timedelta("2h")
    window_end = fire_ts + pd.Timedelta("1h")
    window_pings = df[(df[time_col] >= window_start) & (df[time_col] <= window_end)]

    if not window_pings.empty:
        for _, row in window_pings.iterrows():
            d = haversine_m(home_lat, home_lon, row[lat_col], row[lon_col])
            if d <= home_radius:
                return home_lat, home_lon, "home"

    # ---- Step 2: Build pre-fire stop chain and identify active stop ----
    lookback = fire_ts - pd.Timedelta("12h")
    pre_fire = df[(df[time_col] >= lookback) & (df[time_col] <= fire_ts)]

    if pre_fire.empty:
        return home_lat, home_lon, "home"

    chain = build_trip_chain_for_user(
        pre_fire,
        user_id="user",
        home_lat=home_lat,
        home_lon=home_lon,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        clustering_config=IncrementalClusterConfig(
            radius_m=R_a,
            min_dwell_s=pd.Timedelta(T_a).total_seconds(),
        ),
        home_radius_m=home_radius,
    )
    if chain.empty:
        return home_lat, home_lon, "home"

    chain = chain.sort_values("start_ts").reset_index(drop=True)
    candidate = None
    for _, row in chain.iterrows():
        c_start = pd.Timestamp(row["start_ts"])
        c_end = pd.Timestamp(row["end_ts"])
        if c_start <= fire_ts <= c_end:
            candidate = row
            break
        if c_end <= fire_ts:
            candidate = row

    if candidate is None:
        return home_lat, home_lon, "home"

    d_from_home = haversine_m(
        home_lat, home_lon,
        float(candidate["lat"]), float(candidate["lon"]),
    )
    if d_from_home <= home_radius or d_from_home > max_origin_distance_m:
        return home_lat, home_lon, "home"
    return float(candidate["lat"]), float(candidate["lon"]), "activity"

