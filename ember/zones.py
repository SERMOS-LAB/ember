"""
Spatial matching of home locations to evacuation zones.
"""

import pandas as pd
import geopandas as gpd

from ._constants import ZONE_PRIORITY


def classify_zones(
    homes: gpd.GeoDataFrame,
    fire_zones: gpd.GeoDataFrame,
    *,
    buffer_distance: float = 2000.0,   # meters
    order_status: str = "Evacuation Order",
    warning_status: str = "Evacuation Warning",
) -> gpd.GeoDataFrame:
    """
    Assign each home to an evacuation zone (Order, Warning, Buffer, Outside).
    
    Parameters
    ----------
    homes : gpd.GeoDataFrame
        Home locations from `ember.io.load_homes` or `ember.ghost.infer_homes`.
        Must be in EPSG:4326.
    fire_zones : gpd.GeoDataFrame
        Evacuation zones from `ember.io.load_fire_timeline`.
        Must be in EPSG:4326 and contain 'most_extreme_status'.
    buffer_distance : float
        Distance in meters to expand Order/Warning zones to create
        a Buffer (shadow evacuation) zone. Default is 2000 m.
        Should be adjusted per region: e.g. 1000 m for dense urban
        areas, up to 6000 m for rural/suburban contexts.
    order_status : str
        The string in 'most_extreme_status' that indicates a mandatory order.
    warning_status : str
        The string in 'most_extreme_status' that indicates a voluntary warning.
        
    Returns
    -------
    gpd.GeoDataFrame
        The `homes` GeoDataFrame with new columns:
        - ZoneType: 'Order', 'Warning', 'Buffer', or 'Outside'
        - OrderStart: The earliest known datetime of the assigned status
        - zoneId: The ID of the matched zone
        - FireEvent: The incident name of the matched zone. For 'Buffer' homes
          (which match only the dissolved buffer, not an individual zone) this is
          the incident name of the NEAREST active zone, so shadow evacuees are still
          attributed to a fire. 'Outside' homes have no FireEvent.
    """
    
    # 1. Prepare Homes
    # Best practice for buffer math is a projected CRS
    # We will use Web Mercator 3857 for convenience or UTM if precision is critical,
    # but 3857 is computationally easy for standard buffer approximations.
    homes_proj = homes.to_crs(epsg=3857)
    zones_proj = fire_zones.to_crs(epsg=3857)
    
    # 2. Add 'priority' to zones so we can sort overlapping zones
    def get_priority(status: str) -> int:
        if order_status in str(status):
            return 2
        elif warning_status in str(status):
            return 1
        return 0
        
    if 'most_extreme_status' in zones_proj.columns:
        zones_proj['priority'] = zones_proj['most_extreme_status'].apply(get_priority)
    else:
        # Fallback if no status column, assume everything is an order
        zones_proj['priority'] = 2
        
    # Standardize ZoneType label
    def get_zonetype(prio: int) -> str:
        if prio == 2: return "Order"
        if prio == 1: return "Warning"
        return "Unknown"
        
    zones_proj['ZoneType'] = zones_proj['priority'].apply(get_zonetype)
    
    # Sort zones so that highest priority is first (Order > Warning)
    zones_proj = zones_proj.sort_values(by='priority', ascending=False)
    
    # 3. Spatial Join for Order/Warning
    # inner/left? We want a left join to keep all homes.
    joined = gpd.sjoin(homes_proj, zones_proj, how='left', predicate='intersects')
    
    # Deduplicate: if a point falls in multiple overlapping zones, the sort order
    # above ensures the 'first' match we keep is the highest priority.
    # If same priority, could keep the one with earlier datetime. 
    if 'datetime' in joined.columns:
        joined['datetime'] = pd.to_datetime(joined['datetime'])
        joined = joined.sort_values(['priority', 'datetime'], ascending=[False, True])
        joined = joined.drop_duplicates(subset=['ID'], keep='first') if 'ID' in joined.columns else joined[~joined.index.duplicated(keep='first')]
    else:
        joined = joined.drop_duplicates(subset=['ID'], keep='first') if 'ID' in joined.columns else joined[~joined.index.duplicated(keep='first')]
    
    # Fill non-matches with 'Outside'
    joined['ZoneType'] = joined['ZoneType'].fillna('Outside')
    
    # 4. Process Buffer Zone
    # For homes that are 'Outside', check if they are within `buffer_distance` of ANY Order/Warning zone.
    outside_mask = joined['ZoneType'] == 'Outside'
    
    if outside_mask.any() and not zones_proj.empty:
        # Dissolve all active zones into a single multipolygon to speed up distance check
        all_active_zones = zones_proj[zones_proj['priority'] > 0]
        if not all_active_zones.empty:
            unified_hazard_area = all_active_zones.union_all()
            buffered_area = unified_hazard_area.buffer(buffer_distance)

            # Check intersection
            is_in_buffer = joined.loc[outside_mask, 'geometry'].intersects(buffered_area)
            joined.loc[outside_mask & is_in_buffer, 'ZoneType'] = 'Buffer'

            # Buffer homes matched only the dissolved-union buffer, never an individual
            # zone, so the intersects-join left their incident_name (-> FireEvent) null.
            # In a multi-fire study that means shadow evacuees have no fire attributed,
            # and downstream per-fire timing falls back to a global default. Attribute
            # each buffer home to its NEAREST active zone so FireEvent (and the per-fire
            # start time keyed off it) is correct for shadow evacuees too.
            buf_mask = joined['ZoneType'] == 'Buffer'
            if buf_mask.any() and 'incident_name' in all_active_zones.columns:
                nearest = gpd.sjoin_nearest(
                    joined.loc[buf_mask, ['geometry']],
                    all_active_zones[['geometry', 'incident_name']],
                    how='left',
                )
                # ties (equidistant zones) can duplicate rows; keep the first per home
                nearest = nearest[~nearest.index.duplicated(keep='first')]
                joined.loc[buf_mask, 'incident_name'] = nearest['incident_name']

    # Clean up output
    rename_cols = {}
    if 'datetime' in joined.columns:
        rename_cols['datetime'] = 'OrderStart'
    if 'incident_name' in joined.columns:
        valid_mask = joined['incident_name'].notna()
        joined.loc[valid_mask, 'incident_name'] = joined.loc[valid_mask, 'incident_name'].astype(str).str.title()
        rename_cols['incident_name'] = 'FireEvent'
        
    joined = joined.rename(columns=rename_cols)
    
    # Keep desired columns and convert back to 4326
    keep_cols = list(homes.columns) + ['ZoneType']
    for extra in ['OrderStart', 'OrderEnd', 'zoneId', 'FireEvent', 'most_extreme_status']:
        if extra in joined.columns:
            keep_cols.append(extra)
            
    final_gdf = joined[keep_cols].to_crs(epsg=4326)
    
    return final_gdf
