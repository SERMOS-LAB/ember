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
        'incident_name': ['Eaton', 'Eaton'],
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

    # In-zone homes carry their zone's FireEvent; the buffer home inherits the
    # nearest active zone's fire (here the only fire, Eaton).
    assert result[result['ID'] == 'h1']['FireEvent'].iloc[0] == 'Eaton'
    assert result[result['ID'] == 'h3']['FireEvent'].iloc[0] == 'Eaton'


@pytest.fixture
def mock_homes_two_fires():
    # hA: inside FireA order; hB_buf: a shadow evacuee OUTSIDE every zone but nearest
    # to FireB (far east); hFar: outside everything.
    df = pd.DataFrame({
        'ID': ['hA', 'hB_buf', 'hFar'],
        'home_lat_4326': [34.10, 34.10, 34.10],
        'home_lon_4326': [-118.10, -117.40, -119.50],
    })
    return gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df.home_lon_4326, df.home_lat_4326), crs=EPSG_WGS84
    )


@pytest.fixture
def mock_zones_two_fires():
    # FireA far west (~-118.1), FireB far east (~-117.5); ~55 km apart.
    pa = Polygon([(-118.15, 34.05), (-118.05, 34.05), (-118.05, 34.15), (-118.15, 34.15)])
    pb = Polygon([(-117.55, 34.05), (-117.45, 34.05), (-117.45, 34.15), (-117.55, 34.15)])
    df = pd.DataFrame({
        'zoneId': ['za', 'zb'],
        'most_extreme_status': ['Evacuation Order', 'Evacuation Order'],
        'incident_name': ['Palisades', 'Eaton'],
        'datetime': ['2025-01-01T12:00:00', '2025-01-01T18:00:00'],
    })
    return gpd.GeoDataFrame(df, geometry=[pa, pb], crs=EPSG_WGS84)


def test_buffer_home_gets_nearest_fire(mock_homes_two_fires, mock_zones_two_fires):
    """Regression: a shadow evacuee outside every zone must inherit the NEAREST fire,
    not a global/longitude-split default. hB_buf sits just east of the Eaton zone and
    must be attributed to Eaton even though the Palisades fire exists in the same study."""
    result = classify_zones(
        mock_homes_two_fires, mock_zones_two_fires,
        buffer_distance=10000.0,  # 10 km: catches hB_buf near Eaton, not hFar
        order_status='Evacuation Order', warning_status='Evacuation Warning',
    )
    by_id = result.set_index('ID')
    assert by_id.loc['hA', 'ZoneType'] == 'Order'
    assert by_id.loc['hA', 'FireEvent'] == 'Palisades'
    assert by_id.loc['hB_buf', 'ZoneType'] == 'Buffer'
    assert by_id.loc['hB_buf', 'FireEvent'] == 'Eaton'   # <- the fix (was NaN before)
    assert by_id.loc['hFar', 'ZoneType'] == 'Outside'
