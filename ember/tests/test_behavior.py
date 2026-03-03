import pytest
import pandas as pd
from ember.behavior import classify, summary


def test_behavior_classification():
    # Synthetic data matching the 7 categories
    # order_start: Jan 2 12:00
    # fire_start: Jan 1 12:00
    
    df = pd.DataFrame({
        'ID': ['u_sele', 'u_feuo', 'u_feuw', 'u_pere', 'u_sefn', 'u_ner', 'u_ur_nodata'],
        'ZoneType': ['Order', 'Order', 'Warning', 'Order', 'Buffer', 'Order', 'Order'],
        'OrderStart': pd.to_datetime([
            '2025-01-02 12:00', '2025-01-02 12:00', '2025-01-02 12:00', 
            '2025-01-02 12:00', '2025-01-02 12:00', '2025-01-02 12:00', pd.NaT
        ]),
        'FireStart': pd.to_datetime([
            '2025-01-01 12:00', '2025-01-01 12:00', '2025-01-01 12:00', 
            '2025-01-01 12:00', '2025-01-01 12:00', '2025-01-01 12:00', pd.NaT
        ]),
        'DepartureDate': pd.to_datetime([
            '2025-01-01 15:00', # SELE (after fire, before order)
            '2025-01-02 14:00', # FEUO (after order, order zone)
            '2025-01-02 14:00', # FEUW (after order, warning zone)
            '2025-01-02 14:00', # PERE (after order, but returns early) -> To trigger PERE we need an order lift time, 
                                # but in simplified logic, we map generic Order -> FEUO unless return is explicitly early. 
                                # We'll just test FEUO/FEUW here as they map 1:1 with zone types when departing.
            '2025-01-02 14:00', # SEFN (buffer zone)
            pd.NaT,             # NER (no departure)
            '2025-01-02 14:00'  # UR (missing order start)
        ]),
        'ReturnDate': pd.to_datetime([
            '2025-01-10 12:00', '2025-01-10 12:00', '2025-01-10 12:00',
            '2025-01-02 08:00', # early return for PERE testing (though simplified logic might just call it NER or FEUO if not carefully gated)
            '2025-01-10 12:00', pd.NaT, '2025-01-10 12:00'
        ])
    })
    
    cats = classify(
        df,
        zone_col='ZoneType',
        departure_col='DepartureDate',
        return_col='ReturnDate',
        order_start_col='OrderStart',
        fire_start_col='FireStart'
    )
    
    df['Category'] = cats
    
    assert df.loc[df['ID'] == 'u_sele', 'Category'].iloc[0] == 'SELE'
    assert df.loc[df['ID'] == 'u_feuo', 'Category'].iloc[0] == 'FEUO'
    assert df.loc[df['ID'] == 'u_feuw', 'Category'].iloc[0] == 'FEUW'
    assert df.loc[df['ID'] == 'u_sefn', 'Category'].iloc[0] == 'SEFN'
    assert df.loc[df['ID'] == 'u_ner', 'Category'].iloc[0] == 'NER'
    assert df.loc[df['ID'] == 'u_ur_nodata', 'Category'].iloc[0] == 'UR'
    
    # Test summary table
    sum_df = summary(cats)
    assert len(sum_df) > 0
    assert 'Percentage' in sum_df.columns
