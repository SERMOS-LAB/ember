"""
Departure and return time inference.
"""

import pandas as pd
from typing import Optional, Tuple, Literal

from .activities import find_origin as _find_origin, haversine_m


def _get_consecutive_intervals(dates: list) -> list:
    """Group dates that are within 1 day of each other into intervals."""
    if not dates:
        return []

    intervals = []
    # Ensure dates are unique and sorted
    dates = sorted(list(set(dates)))
    current = [dates[0]]

    for i in range(1, len(dates)):
        # Calculate diff in days directly from date objects
        diff_days = (dates[i] - current[-1]).days
        if diff_days <= 1:
            current.append(dates[i])
        else:
            intervals.append(current)
            current = [dates[i]]
            
    intervals.append(current)
    return intervals


def infer(
    pings: pd.DataFrame,
    *,
    home_radius: float = 200.0,
    away_radius: float = 1000.0,
    min_hours_away: float = 2.0,
    order_start: Optional[pd.Timestamp] = None,
    method: Literal["home_based", "activity_based", "auto"] = "home_based",
    home_lat: Optional[float] = None,
    home_lon: Optional[float] = None,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp], str]:
    """
    Infer departure and return times for a single resident from raw pings.
    
    Supports three modes via the ``method`` parameter:

    - ``'home_based'`` (default): Uses the proxy home location as the
      evacuation origin.  This is the original Zhao et al. (2022) approach.
    - ``'activity_based'``: Uses incremental clustering (Nima et al. 2025)
      to find where the evacuee actually was at fire start.  Requires
      ``home_lat``/``home_lon`` and ``order_start``.
    - ``'auto'``: Tries activity-based first; if the evacuee was at home,
      falls back to home-based.

    Parameters
    ----------
    pings : pd.DataFrame
        Pings for a SINGLE user.  Must have ``distance_to_home`` for
        home-based mode, or ``latitude``/``longitude``/``datetime``
        columns for activity-based mode.
    home_radius : float
        Buffer around home in metres.
    away_radius : float
        Distance threshold for "away from origin".
    order_start : pd.Timestamp, optional
        Used to filter intervals and as fire_start for activity clustering.
    method : {'home_based', 'activity_based', 'auto'}
        Origin inference method.
    home_lat, home_lon : float, optional
        Home coordinates (EPSG:4326).  Required for activity-based mode.
        
    Returns
    -------
    tuple
        ``(T_dep, T_ret, origin_type)`` where ``origin_type`` is
        ``'home'`` or ``'activity'``.  Returns ``(None, None, 'home')``
        if no evacuation detected.
    """
    origin_type = "home"

    # ---- Activity-based origin resolution ----
    if method in ("activity_based", "auto"):
        if home_lat is not None and home_lon is not None and order_start is not None:
            _, _, origin_type = _find_origin(
                pings,
                home_lat,
                home_lon,
                fire_start=order_start,
                home_radius=home_radius,
            )
            # If auto and origin is home, fall through to home-based logic
            if method == "activity_based" or origin_type == "activity":
                # Compute distance_to_origin for each ping
                if "latitude" in pings.columns and "longitude" in pings.columns:
                    o_lat, o_lon, origin_type = _find_origin(
                        pings,
                        home_lat,
                        home_lon,
                        fire_start=order_start,
                        home_radius=home_radius,
                    )
                    pings = pings.copy()
                    pings["distance_to_home"] = pings.apply(
                        lambda r: haversine_m(o_lat, o_lon, r["latitude"], r["longitude"]),
                        axis=1,
                    )

    # ---- Standard home-based departure inference ----
    if pings.empty or "distance_to_home" not in pings.columns:
        return None, None, origin_type
        
    # Identify away pings
    away_pings = pings[pings["distance_to_home"] > away_radius].copy()
    if away_pings.empty:
        return None, None, origin_type
        
    # Get distinct dates where they were away
    if "datetime" not in away_pings.columns:
        away_pings["datetime"] = pd.to_datetime(
            away_pings["timestamp_ms"], unit="ms", utc=True
        )
        
    away_dates = away_pings["datetime"].dt.date.tolist()
    
    intervals = _get_consecutive_intervals(away_dates)
    
    if not intervals:
        return None, None, origin_type
        
    # Select interval
    selected = None
    if order_start:
        ref_date = order_start.date()
        for interval in intervals:
            if interval[-1] >= ref_date:
                selected = interval
                break
    else:
        selected = intervals[0]
        
    if not selected:
        return None, None, origin_type
        
    # Overnight check: reject same day trips
    if selected[-1] <= selected[0]:
        return None, None, origin_type
        
    # Departure: earliest timestamp on the first day of the interval
    mask_start = away_pings["datetime"].dt.date == selected[0]
    dep_time = away_pings.loc[mask_start, "datetime"].min()
    
    # Return: latest timestamp on the last day of the interval
    mask_end = away_pings["datetime"].dt.date == selected[-1]
    ret_time = away_pings.loc[mask_end, "datetime"].max()
    
    return dep_time, ret_time, origin_type


def delay_hours(
    departure: pd.Series,
    order_start: pd.Series,
) -> pd.Series:
    """
    Compute delay in hours between departure and order/warning issuance.
    
    Negative delays are typically clamped to 0 in standard analytics, 
    but we return the raw negative delay here to allow distinguishing 
    pre-emptive evacuations physically.
    """
    dep = pd.to_datetime(departure, errors="coerce")
    order = pd.to_datetime(order_start, errors="coerce")
    
    diff = (dep - order).dt.total_seconds() / 3600.0
    return diff

