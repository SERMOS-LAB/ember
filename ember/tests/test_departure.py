import pytest
import pandas as pd
from datetime import datetime, timezone, timedelta
from ember.departure import infer, delay_hours


def test_infer_departure_return():
    # User stays home, goes away for 3 days, comes back
    base_time = pd.Timestamp("2025-01-01 12:00:00", tz='UTC')
    
    records = []
    # Home days (1st and 2nd)
    for i in range(2):
        records.append({'timestamp_ms': int((base_time + timedelta(days=i)).timestamp() * 1000), 'distance_to_home': 50})
        
    # Evac days (3rd, 4th, 5th)
    for i in range(2, 5):
        records.append({'timestamp_ms': int((base_time + timedelta(days=i)).timestamp() * 1000), 'distance_to_home': 5000}) # 5km
        
    # Return days (6th, 7th)
    for i in range(5, 7):
        records.append({'timestamp_ms': int((base_time + timedelta(days=i)).timestamp() * 1000), 'distance_to_home': 100})
        
    df = pd.DataFrame(records)
    
    t_dep, t_ret = infer(df, away_radius=1000.0)
    
    assert t_dep is not None
    assert t_ret is not None
    
    # Departure should be on the 3rd day (2025-01-03 12:00)
    assert t_dep.date() == pd.Timestamp("2025-01-03").date()
    # Return should be the last away day ping (2025-01-05 12:00)
    assert t_ret.date() == pd.Timestamp("2025-01-05").date()


def test_delay_hours():
    deps = pd.Series([
        '2025-01-02 12:00:00', # 24h delay
        '2025-01-01 10:00:00', # -2h delay (early)
        pd.NaT
    ])
    orders = pd.Series([
        '2025-01-01 12:00:00',
        '2025-01-01 12:00:00',
        '2025-01-01 12:00:00'
    ])
    
    delay = delay_hours(deps, orders)
    
    assert delay.iloc[0] == 24.0
    assert delay.iloc[1] == -2.0
    assert pd.isna(delay.iloc[2])
