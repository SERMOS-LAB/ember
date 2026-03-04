"""
Activity inference via incremental clustering.

Implements the incremental clustering method (Zhang et al. 2023;
Alexander et al. 2015; Wang & Chen 2018) used by both:

- **Activity-Based Origin**: clustering *pre-fire* pings to find
  where the evacuee actually was when the fire started.
- **Destination Inference**: clustering *post-fire* nightly stops
  to identify overnight destinations.

References
----------
Nima et al. (2025) — Generalized algorithm for inferring wildfire
evacuation decisions using large-scale mobile location data.
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

    Algorithm (Zhang et al. 2023 / Alexander et al. 2015):

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
    times = pd.to_datetime(df[time_col], utc=True)

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
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
) -> Tuple[float, float, str]:
    """
    Determine the evacuation origin for a single resident.

    Uses incremental clustering on the resident's pings around the
    fire-start time.  If the resident was at home, origin = home.
    Otherwise, origin = the centroid of the first post-fire activity.

    Algorithm (Nima et al. 2025, Assumption 3 & 4):

    1. Check whether the evacuee was ever at their proxy home *during*
       the evacuation window.  If yes → origin = home, departure =
       last ping within home buffer.
    2. If not → find the first activity cluster whose end_time is
       *after* fire_start.  The cluster centroid is the origin; the
       point following this cluster is the departure time.

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
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values(time_col).reset_index(drop=True)

    # ---- Check if evacuee was ever at home during evacuation window ----
    fire_ts = pd.Timestamp(fire_start)
    if fire_ts.tzinfo is None:
        fire_ts = fire_ts.tz_localize("UTC")
    else:
        fire_ts = fire_ts.tz_convert("UTC")
    evac_pings = df[df[time_col] >= fire_ts]

    if not evac_pings.empty:
        for _, row in evac_pings.iterrows():
            d = haversine_m(home_lat, home_lon, row[lat_col], row[lon_col])
            if d <= home_radius:
                # They were home at some point → home-based origin
                return home_lat, home_lon, "home"

    # ---- Not at home → find first post-fire activity cluster ----
    clusters = incremental_cluster(
        df, R_a=R_a, T_a=T_a, lat_col=lat_col, lon_col=lon_col, time_col=time_col
    )

    for c in clusters:
        c_end = pd.Timestamp(c.end_time)
        if c_end.tzinfo is None:
            c_end = c_end.tz_localize("UTC")
        if c_end >= fire_ts:
            return c.centroid_lat, c.centroid_lon, "activity"

    # Fallback: no qualifying activity found → default to home
    return home_lat, home_lon, "home"
