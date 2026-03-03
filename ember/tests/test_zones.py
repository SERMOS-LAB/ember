import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
from ember.zones import classify_zones
from ember._constants import EPSG_WGS84


@pytest.fixture
def mock_homes():
    df = pd.DataFrame({
        'ID': ['h1', 'h2', 'h3', 'h4'],
        'home_lat_4326': [34.1, 34.2, 34.3, 34.4],
        'home_lon_4326': [-118.1, -118.2, -118.3, -118.4]
    })
    return gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df.home_lon_4326, df.home_lat_4326),
        crs=EPSG_WGS84
    )


@pytest.fixture
def mock_zones():
    # Zone 1: Order covers h1 (34.1, -118.1)
    p1 = Polygon([(-118.15, 34.05), (-118.05, 34.05), (-118.05, 34.15), (-118.15, 34.15)])
    
    # Zone 2: Warning covers h2 (34.2, -118.2)
    p2 = Polygon([(-118.25, 34.15), (-118.15, 34.15), (-118.15, 34.25), (-118.25, 34.25)])
    
    # h3 is just outside Zone 2, will be caught by 5km buffer
    # h4 is far away, will be Outside
    
    df = pd.DataFrame({
        'zoneId': ['z1', 'z2'],
        'most_extreme_status': ['Evacuation Order', 'Evacuation Warning'],
        'incident_name': ['FireA', 'FireA'],
        'datetime': ['2025-01-01T12:00:00', '2025-01-01T10:00:00']
    })
    
    return gpd.GeoDataFrame(df, geometry=[p1, p2], crs=EPSG_WGS84)


def test_classify_zones(mock_homes, mock_zones):
    # Using a 15km buffer (15000m) to easily catch h3 (which is roughly ~11km diagonally from h2's box)
    result = classify_zones(
        mock_homes,
        mock_zones,
        buffer_distance=20000.0, # 20km
        order_status='Evacuation Order',
        warning_status='Evacuation Warning'
    )
    
    assert len(result) == 4
    
    # Check assignments
    h1_class = result[result['ID'] == 'h1']['ZoneType'].iloc[0]
    h2_class = result[result['ID'] == 'h2']['ZoneType'].iloc[0]
    h3_class = result[result['ID'] == 'h3']['ZoneType'].iloc[0]
    h4_class = result[result['ID'] == 'h4']['ZoneType'].iloc[0]
    
    assert h1_class == 'Order'
    assert h2_class == 'Warning'
    
    # h3 should be Buffer because of the 20km buffer
    assert h3_class == 'Buffer'
    
    # h4 is very far away (~30-40km)
    assert h4_class == 'Outside'
    
    # Check metadata was preserved
    assert 'OrderStart' in result.columns
    assert 'zoneId' in result.columns
    
    h1_start = result[result['ID'] == 'h1']['OrderStart'].iloc[0]
    assert pd.to_datetime(h1_start) == pd.to_datetime('2025-01-01T12:00:00')
