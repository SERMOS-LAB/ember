"""
Activity inference via incremental clustering.

Implements an incremental clustering method used by both:

- **Activity-Based Origin**: clustering *pre-fire* pings to find
  where the evacuee actually was when the fire started.
- **Destination Inference**: clustering *post-fire* nightly stops
  to identify overnight destinations.
"""

from __future__ import annotations

import math
from collections import namedtuple
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

ActivityCluster = namedtuple(
    "ActivityCluster",
    ["centroid_lat", "centroid_lon", "start_time", "end_time", "n_points"],
)


# ---------------------------------------------------------------------------
# Haversine utility (metres)
# ---------------------------------------------------------------------------

def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in **metres**."""
    R = 6_371_000  # Earth radius in metres
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Core incremental clustering
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
    if pings.empty:
        return []

    T_a_td = pd.Timedelta(T_a)

    # Ensure sorted
    df = pings.sort_values(time_col).reset_index(drop=True)
    lats = df[lat_col].values
    lons = df[lon_col].values
    times = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(times):
        times = pd.to_datetime(times, format="mixed", utc=True)
    elif times.dt.tz is None:
        times = times.dt.tz_localize("UTC")
    else:
        times = times.dt.tz_convert("UTC")

    clusters: List[ActivityCluster] = []

    # Running cluster state
    c_lats = [lats[0]]
    c_lons = [lons[0]]
    c_start = times.iloc[0]
    c_end = times.iloc[0]

    for i in range(1, len(df)):
        centroid_lat = np.mean(c_lats)
        centroid_lon = np.mean(c_lons)
        d = haversine_m(centroid_lat, centroid_lon, lats[i], lons[i])

        if d <= R_a:
            # Add to current cluster
            c_lats.append(lats[i])
            c_lons.append(lons[i])
            c_end = times.iloc[i]
        else:
            # Close current cluster if duration qualifies
            if c_end - c_start >= T_a_td:
                clusters.append(
                    ActivityCluster(
                        centroid_lat=np.mean(c_lats),
                        centroid_lon=np.mean(c_lons),
                        start_time=c_start,
                        end_time=c_end,
                        n_points=len(c_lats),
                    )
                )
            # Start new cluster
            c_lats = [lats[i]]
            c_lons = [lons[i]]
            c_start = times.iloc[i]
            c_end = times.iloc[i]

    # Final cluster
    if c_end - c_start >= T_a_td:
        clusters.append(
            ActivityCluster(
                centroid_lat=np.mean(c_lats),
                centroid_lon=np.mean(c_lons),
                start_time=c_start,
                end_time=c_end,
                n_points=len(c_lats),
            )
        )

    return clusters


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

    # ---- Step 2: Cluster pre-fire pings to find activity location ----
    # Use pings from up to 12 hours before fire start
    lookback = fire_ts - pd.Timedelta("12h")
    pre_fire = df[(df[time_col] >= lookback) & (df[time_col] <= fire_ts)]

    if pre_fire.empty:
        # No pre-fire data → default to home
        return home_lat, home_lon, "home"

    clusters = incremental_cluster(
        pre_fire, R_a=R_a, T_a=T_a,
        lat_col=lat_col, lon_col=lon_col, time_col=time_col
    )

    if not clusters:
        return home_lat, home_lon, "home"

    # Find cluster that contains fire_start, or the last cluster before it
    best_cluster = None
    for c in clusters:
        c_start = pd.Timestamp(c.start_time)
        c_end = pd.Timestamp(c.end_time)
        if c_start.tzinfo is None:
            c_start = c_start.tz_localize("UTC")
        if c_end.tzinfo is None:
            c_end = c_end.tz_localize("UTC")

        if c_start <= fire_ts <= c_end:
            # Evacuee was in this cluster at fire start
            best_cluster = c
            break
        elif c_end <= fire_ts:
            # This cluster ended before fire — keep as candidate
            best_cluster = c

    if best_cluster is None:
        return home_lat, home_lon, "home"

    # ---- Step 3: Distance sanity check ----
    d_from_home = haversine_m(
        home_lat, home_lon,
        best_cluster.centroid_lat, best_cluster.centroid_lon
    )

    if d_from_home <= home_radius:
        # The "activity" is actually at home
        return home_lat, home_lon, "home"

    if d_from_home > max_origin_distance_m:
        # Too far from home — likely a travel/transit artifact
        return home_lat, home_lon, "home"

    return best_cluster.centroid_lat, best_cluster.centroid_lon, "activity"

