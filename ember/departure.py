"""
Departure and return time inference.
"""

import pandas as pd
from typing import Optional, Tuple, Literal


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
    method: Literal["home_based", "activity_based", "auto"] = "auto"
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Infer departure and return times for a single resident from raw pings.
    
    This is an advanced, flexible implementation. In practice, EMBER expects 
    stops/stay-points to be extracted first (as the original toolkit does).
    For now, this implements a simplified proxy:
    1. Identify days where the user was observed taking a trip >`away_radius` from home.
    2. Group consecutive away-days into intervals.
    3. If `order_start` is provided, find the first interval after it.
    4. Reject same-day trips (require overnight).
    5. Find first departure ping and last return ping for that interval.
    
    Parameters
    ----------
    pings : pd.DataFrame
        Pings for a SINGLE user, with distance_to_home pre-calculated.
    order_start : pd.Timestamp, optional
        Used to filter intervals. If none, takes the first valid evacuation trip.
        
    Returns
    -------
    tuple
        (T_dep, T_ret) timestamps. Returns (None, None) if no evacuation.
    """
    
    if pings.empty or 'distance_to_home' not in pings.columns:
        return None, None
        
    # Identify away pings
    away_pings = pings[pings['distance_to_home'] > away_radius].copy()
    if away_pings.empty:
        return None, None
        
    # Get distinct dates where they were away
    # Assuming 'datetime' column exists and is local tz
    if 'datetime' not in away_pings.columns:
        away_pings['datetime'] = pd.to_datetime(away_pings['timestamp_ms'], unit='ms', utc=True)
        
    away_dates = away_pings['datetime'].dt.date.tolist()
    
    intervals = _get_consecutive_intervals(away_dates)
    
    if not intervals:
        return None, None
        
    # Select interval
    selected = None
    if order_start:
        ref_date = order_start.date()
        for interval in intervals:
            if interval[-1] >= ref_date:
                selected = interval
                break
    else:
        selected = intervals[0] # Just take the first away trip
        
    if not selected:
        return None, None
        
    # Overnight check: reject same day trips
    if selected[-1] <= selected[0]:
        return None, None
        
    # Departure: earliest timestamp on the first day of the interval
    mask_start = away_pings['datetime'].dt.date == selected[0]
    dep_time = away_pings.loc[mask_start, 'datetime'].min()
    
    # Return: latest timestamp on the last day of the interval
    mask_end = away_pings['datetime'].dt.date == selected[-1]
    ret_time = away_pings.loc[mask_end, 'datetime'].max()
    
    return dep_time, ret_time


def delay_hours(
    departure: pd.Series,
    order_start: pd.Series
) -> pd.Series:
    """
    Compute delay in hours between departure and order/warning issuance.
    
    Negative delays are typically clamped to 0 in standard analytics, 
    but we return the raw negative delay here to allow distinguishing 
    pre-emptive evacuations physically.
    """
    dep = pd.to_datetime(departure, errors='coerce')
    order = pd.to_datetime(order_start, errors='coerce')
    
    diff = (dep - order).dt.total_seconds() / 3600.0
    return diff
