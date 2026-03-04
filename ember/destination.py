"""
Destination inference for wildfire evacuees.

Implements the destination inference algorithm from Cova et al. (2024):
overnight stops during the fire → filter within home buffer → merge
nearby stops → destination list per evacuee → optional land-use
classification via parcel spatial join.

References
----------
Cova, T. J., et al. (2024). Destination unknown: Examining wildfire
evacuee trips using GPS data. *Transportation Research Part D*.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

try:
    import geopandas as gpd
except ImportError:
    gpd = None

from .activities import haversine_m


# ---------------------------------------------------------------------------
# Haversine utility — km scale (delegates to activities.haversine_m)
# ---------------------------------------------------------------------------

def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in kilometres."""
    return haversine_m(lat1, lon1, lat2, lon2) / 1000.0


# ---------------------------------------------------------------------------
# Core destination inference
# ---------------------------------------------------------------------------

def infer_destinations(
    stops: pd.DataFrame,
    homes: pd.DataFrame,
    *,
    merge_distance_km: float = 0.4,
    home_buffer_m: float = 400.0,
    nighttime_start: int = 20,
    nighttime_end: int = 7,
    lat_col: str = "stop_lat_4326",
    lon_col: str = "stop_lon_4326",
    date_col: str = "stop_date",
    duration_col: str = "duration",
    id_col: str = "ID",
    home_lat_col: str = "home_lat_4326",
    home_lon_col: str = "home_lon_4326",
) -> pd.DataFrame:
    """
    Infer evacuation destinations from nightly stop data.

    Algorithm (Cova et al. 2024, Section 3.2–3.3):

    1. For each evacuee, extract nightly stops (``nighttime_start``
       to ``nighttime_end``).
    2. Pick the longest-duration stop per night as the overnight
       destination.
    3. Filter out stops within ``home_buffer_m`` of the evacuee's
       home location.
    4. Merge consecutive stops where haversine(i, j) < ``merge_distance_km``
       into a single destination.
    5. Compute ``eu_distance_km`` from home to each destination.

    Parameters
    ----------
    stops : DataFrame
        Per-user per-night stop records.  Must contain ``id_col``,
        ``lat_col``, ``lon_col``, ``date_col``.  Optionally
        ``duration_col`` (defaults to equal weighting if absent).
    homes : DataFrame
        Home locations with ``id_col``, ``home_lat_col``, ``home_lon_col``.
    merge_distance_km : float
        Distance threshold to merge successive stops into one
        destination (default 0.4 km = 0.25 mi).
    home_buffer_m : float
        Stops within this distance of home are dropped (default 400 m).
    nighttime_start, nighttime_end : int
        Hours defining the overnight window (default 20:00–07:00).

    Returns
    -------
    DataFrame
        Columns: ``ID``, ``dest_lat``, ``dest_lon``, ``dest_date``,
        ``eu_distance_km``, ``dest_order``.
    """
    # Merge home coordinates onto stops
    home_cols = [id_col, home_lat_col, home_lon_col]
    if not all(c in homes.columns for c in home_cols):
        raise ValueError(
            f"homes must contain columns {home_cols}. "
            f"Found: {list(homes.columns)}"
        )

    df = stops.merge(
        homes[[id_col, home_lat_col, home_lon_col]].drop_duplicates(subset=[id_col]),
        on=id_col,
        how="left",
    )

    # --- 1. Nighttime filter (if we have timestamps, not just dates) ---
    # If stop data only has dates (no hour), skip the hourly filter
    has_duration = duration_col in df.columns

    # --- 2. Pick longest stop per night per user ---
    if has_duration:
        idx = df.groupby([id_col, date_col])[duration_col].idxmax()
        df = df.loc[idx].reset_index(drop=True)
    else:
        # If no duration, just deduplicate per user per night
        df = df.drop_duplicates(subset=[id_col, date_col], keep="first").reset_index(
            drop=True
        )

    # --- 3. Filter stops within home buffer ---
    df["_dist_home_m"] = df.apply(
        lambda r: haversine_m(
            r[home_lat_col], r[home_lon_col], r[lat_col], r[lon_col]
        ),
        axis=1,
    )
    df = df[df["_dist_home_m"] > home_buffer_m].reset_index(drop=True)

    if df.empty:
        return pd.DataFrame(
            columns=["ID", "dest_lat", "dest_lon", "dest_date", "eu_distance_km", "dest_order"]
        )

    # --- 4. Merge close successive stops ---
    results = []
    for uid, grp in df.groupby(id_col):
        grp = grp.sort_values(date_col).reset_index(drop=True)
        destinations = [
            {
                "lat": grp.loc[0, lat_col],
                "lon": grp.loc[0, lon_col],
                "date": grp.loc[0, date_col],
            }
        ]
        for i in range(1, len(grp)):
            prev = destinations[-1]
            d_km = _haversine_km(
                prev["lat"], prev["lon"], grp.loc[i, lat_col], grp.loc[i, lon_col]
            )
            if d_km > merge_distance_km:
                destinations.append(
                    {
                        "lat": grp.loc[i, lat_col],
                        "lon": grp.loc[i, lon_col],
                        "date": grp.loc[i, date_col],
                    }
                )
            # else: merge (keep existing destination, skip this stop)

        for order, dest in enumerate(destinations, start=1):
            eu_km = _haversine_km(
                grp.loc[0, home_lat_col],
                grp.loc[0, home_lon_col],
                dest["lat"],
                dest["lon"],
            )
            results.append(
                {
                    "ID": uid,
                    "dest_lat": dest["lat"],
                    "dest_lon": dest["lon"],
                    "dest_date": dest["date"],
                    "eu_distance_km": round(eu_km, 3),
                    "dest_order": order,
                }
            )

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Destination type classification (optional parcel data)
# ---------------------------------------------------------------------------

def classify_destinations(
    destinations: pd.DataFrame,
    parcels=None,
    *,
    buffer_m: float = 50.0,
    type_col: str = "GENERALIZE",
    dest_lat_col: str = "dest_lat",
    dest_lon_col: str = "dest_lon",
) -> pd.DataFrame:
    """
    Classify destination types via spatial join with parcel/land-use data.

    If ``parcels`` is ``None``, returns the destinations DataFrame
    unchanged with a ``dest_type`` column filled with NaN — the module
    works fully without parcel data.

    Parameters
    ----------
    destinations : DataFrame
        Output of :func:`infer_destinations`.
    parcels : GeoDataFrame or None
        County parcel polygons with a ``type_col`` column describing
        land-use (e.g. 'residential', 'commercial').
    buffer_m : float
        Buffer around destination points for nearest-parcel fallback.
    type_col : str
        Column in ``parcels`` containing the land-use type string.

    Returns
    -------
    DataFrame
        Input with an appended ``dest_type`` column.
    """
    df = destinations.copy()

    if parcels is None:
        df["dest_type"] = np.nan
        return df

    if gpd is None:
        raise ImportError(
            "geopandas is required for parcel classification. "
            "Install with: pip install geopandas"
        )

    # Build destination GeoDataFrame
    dest_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[dest_lon_col], df[dest_lat_col]),
        crs="EPSG:4326",
    )

    # Ensure parcels are in the same CRS
    if parcels.crs is None or parcels.crs.to_epsg() != 4326:
        parcels = parcels.to_crs("EPSG:4326")

    # Spatial join — direct containment
    joined = gpd.sjoin(dest_gdf, parcels[[type_col, "geometry"]], how="left", predicate="within")
    # A point may fall in overlapping polygons → keep first match per dest
    joined = joined[~joined.index.duplicated(keep="first")]

    if type_col in joined.columns:
        df["dest_type"] = joined[type_col].values
    else:
        df["dest_type"] = np.nan

    # --- Nearest-parcel fallback for unmatched points ---
    unmatched = df["dest_type"].isna()
    if unmatched.any() and buffer_m > 0:
        # Project to a metric CRS for buffering
        unmatched_gdf = dest_gdf[unmatched].to_crs("EPSG:3857")
        unmatched_gdf["geometry"] = unmatched_gdf.geometry.buffer(buffer_m)
        unmatched_gdf = unmatched_gdf.to_crs("EPSG:4326")

        joined2 = gpd.sjoin(
            unmatched_gdf, parcels[[type_col, "geometry"]], how="left", predicate="intersects"
        )
        # Drop duplicate matches — keep first per destination
        joined2 = joined2[~joined2.index.duplicated(keep="first")]
        if type_col in joined2.columns:
            df.loc[unmatched, "dest_type"] = joined2[type_col].values

    # Standardize categories
    df["dest_type"] = _standardize_type(df["dest_type"])

    return df


# ---------------------------------------------------------------------------
# Category standardization
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    # Residential
    "residential": "residential",
    "single family": "residential",
    "multi family": "residential",
    "multifamily": "residential",
    "mobile home": "residential",
    # Hotel / Motel
    "hotel": "hotel/motel",
    "motel": "hotel/motel",
    "lodging": "hotel/motel",
    # Commercial
    "commercial": "commercial",
    "shopping": "commercial",
    "store": "commercial",
    "restaurant": "commercial",
    "retail": "commercial",
    "office": "commercial",
    "mixed use": "commercial",
    # Industrial
    "industrial": "industrial",
    "manufacturing": "industrial",
    "warehouse": "industrial",
    # Public / Institutional
    "federal property": "public",
    "public": "public",
    "school": "public",
    "educational": "public",
    "hospital": "public",
    "religious": "public",
    "church": "public",
    "shelter": "public",
    "government": "public",
    "institutional": "public",
    "civic": "public",
    # Open Space / Recreation
    "park": "open space",
    "recreation": "open space",
    "open space": "open space",
    "agriculture": "open space",
    "rural": "open space",
    # Road / Transportation
    "road": "road",
    "highway": "road",
    "transportation": "road",
    "right of way": "road",
}


def _standardize_type(series: pd.Series) -> pd.Series:
    """Map raw parcel descriptions to standardized categories."""
    def _map(val):
        if pd.isna(val):
            return np.nan
        val_lower = str(val).lower()
        for keyword, category in _TYPE_MAP.items():
            if keyword in val_lower:
                return category
        return "other"

    return series.apply(_map)
