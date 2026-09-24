"""
GHOST algorithm for nighttime home location inference.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pyproj import Transformer
from typing import Optional

from ._constants import EPSG_WGS84, EPSG_UTM_11N
from .timeutil import local_clock, resolve_tz


def infer_homes(
    pings: pd.DataFrame,
    *,
    nighttime_start: int = 20,   # 8 PM
    nighttime_end: int = 7,      # 7 AM
    grid_cell_size: float = 50.0, # meters
    min_nights: int = 14,
    min_stay_time: float = 0.0,
    crs_meters: int = EPSG_UTM_11N,
    local_tz: Optional[str] = None,
) -> gpd.GeoDataFrame:
    """
    Infer proxy home locations using the GHOST algorithm on raw GPS pings.
    
    Parameters
    ----------
    pings : pd.DataFrame
        Must contain columns: ['user_id', 'latitude', 'longitude', 'datetime']
        where datetime is a tz-aware or local timestamp.
    nighttime_start : int
        Hour of day considered start of night (0-23).
    nighttime_end : int
        Hour of day considered end of night (0-23).
    grid_cell_size : float
        Grid resolution in meters for snapping locations.
    min_nights : int
        Minimum number of distinct nights a user must be observed in the winning cell 
        to be classified as a local resident.
    min_stay_time : float
        Minimum stay duration in seconds (difference between last and first ping 
        in a given cell). Default 0 (only relies on days).
    crs_meters : int
        EPSG code for a local projected CRS (in meters) to use for snapping. 
        Default is EPSG:26911 (UTM Zone 11N relative to LA).
        
    Returns
    -------
    gpd.GeoDataFrame
        Point locations of inferred homes in EPSG:4326.
    """
    
    # Validate
    req_cols = {'user_id', 'latitude', 'longitude', 'datetime'}
    if not req_cols.issubset(pings.columns):
        raise ValueError(f"pings must contain columns: {req_cols}")
        
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(pings['datetime']):
        pings['datetime'] = pd.to_datetime(pings['datetime'])
        
    # Extract temporal features on the study area's clock
    clock = local_clock(pings['datetime'], resolve_tz(pings['datetime'], local_tz))
    hours = clock.dt.hour
    
    # Filter to nighttime pings
    if nighttime_start > nighttime_end:
        # e.g. 20 (8 PM) to 7 (7 AM)
        night_mask = (hours >= nighttime_start) | (hours < nighttime_end)
    else:
        night_mask = (hours >= nighttime_start) & (hours < nighttime_end)
        
    night_pings = pings[night_mask].copy()
    
    if night_pings.empty:
        return gpd.GeoDataFrame()
        
    # Get distinct 'night dates'. Shift hours before midnight to the previous day so
    # that 11 PM Tuesday and 2 AM Wednesday count as the same 'night'.
    offset_pings = clock[night_mask] - pd.Timedelta(hours=nighttime_end)
    night_pings['night_date'] = offset_pings.dt.date
    night_pings['ts_sec'] = night_pings['datetime'].astype('int64') // 10**9
    
    # Project to local CRS for metric grid snapping
    transformer_to_proj = Transformer.from_crs(
        f"epsg:{EPSG_WGS84}", f"epsg:{crs_meters}", always_xy=True
    )
    
    xx, yy = transformer_to_proj.transform(
        night_pings['longitude'].values, 
        night_pings['latitude'].values
    )
    
    # Grid Snap
    night_pings['gx'] = np.round(xx / grid_cell_size) * grid_cell_size
    night_pings['gy'] = np.round(yy / grid_cell_size) * grid_cell_size
    
    res_list = []
    
    # Group by user and find the cells
    for uid, user_group in night_pings.groupby('user_id'):
        best_cell = None
        max_stay = -1.0
        max_num_nights = 0
        
        # Group by cell
        for (cx, cy), cell_group in user_group.groupby(['gx', 'gy']):
            cell_group = cell_group.sort_values('ts_sec')
            
            stay_time = 0.0
            if len(cell_group) > 1:
                stay_time = float(cell_group['ts_sec'].iloc[-1] - cell_group['ts_sec'].iloc[0])
                
            num_nights = cell_group['night_date'].nunique()
            
            # GHOST Optimization Logic: Maximize StayTime, tiebreak on NumNights
            if stay_time > max_stay:
                max_stay = stay_time
                max_num_nights = num_nights
                best_cell = (cx, cy)
            elif stay_time == max_stay:
                if num_nights > max_num_nights:
                    max_num_nights = num_nights
                    best_cell = (cx, cy)
                    
        # Apply strict residency filter
        if best_cell and max_num_nights >= min_nights and max_stay >= min_stay_time:
            res_list.append({
                'ID': uid,
                'utm_x': best_cell[0],
                'utm_y': best_cell[1],
                'num_nights': max_num_nights,
                'stay_time': max_stay
            })
            
    if not res_list:
        return gpd.GeoDataFrame()
        
    df_homes = pd.DataFrame(res_list)
    
    # Reverse transform back to WGS84
    transformer_to_wgs = Transformer.from_crs(
        f"epsg:{crs_meters}", f"epsg:{EPSG_WGS84}", always_xy=True
    )
    
    wgs_lon, wgs_lat = transformer_to_wgs.transform(
        df_homes['utm_x'].values, df_homes['utm_y'].values
    )
    
    df_homes['home_lon_4326'] = wgs_lon
    df_homes['home_lat_4326'] = wgs_lat
    
    gdf = gpd.GeoDataFrame(
        df_homes,
        geometry=gpd.points_from_xy(df_homes['home_lon_4326'], df_homes['home_lat_4326']),
        crs=EPSG_WGS84
    )
    
    return gdf
