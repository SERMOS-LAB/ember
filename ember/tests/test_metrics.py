import pytest
import pandas as pd
from ember.metrics import compliance_rate, dedi, departure_curve


def test_compliance_rate():
    df = pd.DataFrame({
        'ZoneType': ['Order', 'Order', 'Order', 'Warning', 'Warning', 'Buffer'],
        'Category': ['FEUO', 'NER', 'SELE', 'FEUW', 'NER', 'SEFN']
    })
    
    res = compliance_rate(df)
    
    # Order: 2 evacuees (FEUO, SELE), 1 non-evacuee (NER) -> 2/3 = 66.6%
    order_rate = res[res['ZoneType'] == 'Order']['Compliance_Rate'].iloc[0]
    assert order_rate == pytest.approx(0.666, abs=1e-2)
    
    # Warning: 1 evacuee, 1 NER -> 1/2 = 50%
    warn_rate = res[res['ZoneType'] == 'Warning']['Compliance_Rate'].iloc[0]
    assert warn_rate == pytest.approx(0.50, abs=1e-2)


def test_departure_curve():
    df = pd.DataFrame({
        'ZoneType': ['Order', 'Order', 'Order', 'Order'],
        'DepartureDate': [
            '2025-01-01 10:00:00',
            '2025-01-01 14:00:00',
            '2025-01-02 09:00:00',
            pd.NaT # NER
        ]
    })
    
    curve = departure_curve(
        df,
        start='2025-01-01 00:00:00',
        end='2025-01-03 00:00:00',
        freq='1h',
        zone='Order'
    )
    
    assert not curve.empty
    
    # Before any departures
    assert curve.loc['2025-01-01 08:00:00', 'Cumulative_Count'] == 0
    
    # After first
    assert curve.loc['2025-01-01 12:00:00', 'Cumulative_Count'] == 1
    assert curve.loc['2025-01-01 12:00:00', 'Cumulative_Pct'] == 25.0 # 1 out of 4
    
    # After last
    assert curve.loc['2025-01-02 12:00:00', 'Cumulative_Count'] == 3
    assert curve.loc['2025-01-02 12:00:00', 'Cumulative_Pct'] == 75.0 # 3 out of 4 (1 stayed)
