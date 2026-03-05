"""
Destination inference for wildfire evacuees.

Implements a destination inference algorithm:
overnight stops during the fire → filter within home buffer → merge
nearby stops → destination list per evacuee → optional land-use
classification via parcel spatial join.
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
from .contracts import TRIP_CHAIN_SCHEMA, validate_columns


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
    include_intermediate: bool = False,
    include_return: bool = False,
) -> pd.DataFrame:
    """
    Infer evacuation destinations from nightly stop data.

    Algorithm:

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
    # Chain-first path (preferred)
    chain_required = set(TRIP_CHAIN_SCHEMA.required_columns)
    if chain_required.issubset(set(stops.columns)):
        return infer_destinations_from_chain(
            stops,
            homes,
            merge_distance_km=merge_distance_km,
            home_buffer_m=home_buffer_m,
            include_intermediate=include_intermediate,
            include_return=include_return,
        )

    # Backward-compatible stop table path
    home_cols = [id_col, home_lat_col, home_lon_col]
    if not all(c in homes.columns for c in home_cols):
        raise ValueError(
            f"homes must contain columns {home_cols}. "
            f"Found: {list(homes.columns)}"
        )

    home_lookup = homes[[id_col, home_lat_col, home_lon_col]].drop_duplicates(subset=[id_col])
    df = stops.copy()
    if "user_id" in df.columns and id_col not in df.columns:
        df = df.rename(columns={"user_id": id_col})
    if "lat" in df.columns and lat_col not in df.columns:
        df = df.rename(columns={"lat": lat_col})
    if "lon" in df.columns and lon_col not in df.columns:
        df = df.rename(columns={"lon": lon_col})
    if "start_ts" in df.columns and date_col not in df.columns:
        df[date_col] = pd.to_datetime(df["start_ts"], utc=True).dt.date.astype(str)
    if "dwell_s" in df.columns and duration_col not in df.columns:
        df[duration_col] = pd.to_numeric(df["dwell_s"], errors="coerce") / 60.0

    if "stop_role" in df.columns:
        if include_intermediate:
            allowed = ["intermediate", "overnight", "destination"]
        else:
            allowed = ["overnight", "destination"]
        if include_return:
            allowed.append("return")
        df = df[df["stop_role"].isin(allowed)]
    df = df.merge(home_lookup, on=id_col, how="left")

    has_duration = duration_col in df.columns and df[duration_col].notna().any()
    if has_duration and date_col in df.columns:
        idx = df.groupby([id_col, date_col])[duration_col].idxmax()
        df = df.loc[idx].reset_index(drop=True)
    elif date_col in df.columns:
        df = df.drop_duplicates(subset=[id_col, date_col], keep="first").reset_index(drop=True)

    # Filter stops within home buffer
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

    # Merge close successive stops
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
                    "source_role": "intermediate_or_overnight" if include_intermediate else "overnight",
                }
            )

    return pd.DataFrame(results)


def infer_destinations_from_chain(
    trip_chain: pd.DataFrame,
    homes: pd.DataFrame,
    *,
    merge_distance_km: float = 0.4,
    home_buffer_m: float = 400.0,
    include_intermediate: bool = False,
    include_return: bool = False,
    id_col: str = "ID",
    home_lat_col: str = "home_lat_4326",
    home_lon_col: str = "home_lon_4326",
) -> pd.DataFrame:
    """Infer destinations directly from canonical trip-chain artifacts."""
    if trip_chain.empty:
        return pd.DataFrame(
            columns=["ID", "dest_lat", "dest_lon", "dest_date", "eu_distance_km", "dest_order"]
        )
    validate_columns(trip_chain, TRIP_CHAIN_SCHEMA)
    home_cols = [id_col, home_lat_col, home_lon_col]
    if not all(c in homes.columns for c in home_cols):
        raise ValueError(
            f"homes must contain columns {home_cols}. "
            f"Found: {list(homes.columns)}"
        )

    allowed_roles = {"overnight", "destination"}
    if include_intermediate:
        allowed_roles.add("intermediate")
    if include_return:
        allowed_roles.add("return")

    home_lookup = homes[[id_col, home_lat_col, home_lon_col]].drop_duplicates(subset=[id_col])
    df = trip_chain.copy().rename(columns={"user_id": id_col, "lat": "dest_lat", "lon": "dest_lon"})
    df = df[df["stop_role"].isin(allowed_roles)].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["ID", "dest_lat", "dest_lon", "dest_date", "eu_distance_km", "dest_order"]
        )

    df["dest_date"] = pd.to_datetime(df["start_ts"], utc=True).dt.date.astype(str)
    df = df.merge(home_lookup, on=id_col, how="left")

    # Keep stops beyond home buffer
    df["_dist_home_m"] = df.apply(
        lambda r: haversine_m(r[home_lat_col], r[home_lon_col], r["dest_lat"], r["dest_lon"]),
        axis=1,
    )
    df = df[df["_dist_home_m"] > home_buffer_m].copy()
    if df.empty:
        return pd.DataFrame(
            columns=["ID", "dest_lat", "dest_lon", "dest_date", "eu_distance_km", "dest_order"]
        )

    # Role precedence for choosing representative stop per date
    role_rank = {"overnight": 3, "destination": 2, "intermediate": 1, "return": 0}
    df["_role_rank"] = df["stop_role"].map(role_rank).fillna(0)
    df = df.sort_values([id_col, "dest_date", "_role_rank", "dwell_s"], ascending=[True, True, False, False])
    df = df.drop_duplicates(subset=[id_col, "dest_date"], keep="first").reset_index(drop=True)

    results = []
    for uid, grp in df.groupby(id_col):
        grp = grp.sort_values("dest_date").reset_index(drop=True)
        destinations = [{"lat": grp.loc[0, "dest_lat"], "lon": grp.loc[0, "dest_lon"], "date": grp.loc[0, "dest_date"]}]
        for i in range(1, len(grp)):
            prev = destinations[-1]
            d_km = _haversine_km(prev["lat"], prev["lon"], grp.loc[i, "dest_lat"], grp.loc[i, "dest_lon"])
            if d_km > merge_distance_km:
                destinations.append(
                    {"lat": grp.loc[i, "dest_lat"], "lon": grp.loc[i, "dest_lon"], "date": grp.loc[i, "dest_date"]}
                )
        for order, dest in enumerate(destinations, start=1):
            eu_km = _haversine_km(grp.loc[0, home_lat_col], grp.loc[0, home_lon_col], dest["lat"], dest["lon"])
            results.append(
                {
                    "ID": uid,
                    "dest_lat": dest["lat"],
                    "dest_lon": dest["lon"],
                    "dest_date": dest["date"],
                    "eu_distance_km": round(eu_km, 3),
                    "dest_order": order,
                    "source_role": "trip_chain",
                }
            )
    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Destination type classification (optional parcel data)
# ---------------------------------------------------------------------------

def classify_destinations(
    destinations: pd.DataFrame,
    parcels=None,
    roads=None,
    *,
    buffer_m: float = 50.0,
    road_buffer_m: float = 20.0,
    type_col: str = "UseType",
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
        land-use (e.g. 'Residential', 'Commercial').
    roads : GeoDataFrame or None
        Road centreline geometry (e.g. TIGER/Line roads).  Points
        within ``road_buffer_m`` of a road that were not matched to a
        parcel are labelled ``"road"``.
    buffer_m : float
        Buffer around destination points for nearest-parcel fallback.
    road_buffer_m : float
        Buffer around road centrelines for road classification (metres).
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

    # --- Road classification using actual road geometry ---
    still_unmatched = df["dest_type"].isna()
    if still_unmatched.any() and roads is not None:
        if roads.crs is None or roads.crs.to_epsg() != 3857:
            roads_proj = roads.to_crs("EPSG:3857")
        else:
            roads_proj = roads
        road_buffer = gpd.GeoDataFrame(
            geometry=roads_proj.geometry.buffer(road_buffer_m)
        ).dissolve().to_crs("EPSG:4326")

        unmatched_pts = dest_gdf[still_unmatched]
        road_join = gpd.sjoin(
            unmatched_pts, road_buffer, how="left", predicate="within"
        )
        is_road = ~road_join["index_right"].isna()
        is_road = is_road[~is_road.index.duplicated(keep="first")]
        road_mask = still_unmatched.copy()
        road_indices = is_road[is_road].index
        df.loc[road_indices, "dest_type"] = "road"

    # Destinations still unmatched after parcel + road joins → unknown
    still_unknown = df["dest_type"].isna()
    if still_unknown.any():
        df.loc[still_unknown, "dest_type"] = "unknown"

    # Standardize categories
    df["dest_type"] = _standardize_type(df["dest_type"])

    return df


# ---------------------------------------------------------------------------
# Category standardization
# ---------------------------------------------------------------------------

_TYPE_MAP = {
    # Residential — LA County UseType values
    "residential": "residential",
    "single family": "residential",
    "single-family": "residential",
    "multi family": "residential",
    "multi-family": "residential",
    "multifamily": "residential",
    "mobile home": "residential",
    "condominium": "residential",
    "duplex": "residential",
    "triplex": "residential",
    "apartment": "residential",
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
    "parking": "commercial",
    "service station": "commercial",
    "gas station": "commercial",
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
    "exempt": "public",
    # Open Space / Recreation
    "park": "open space",
    "recreation": "open space",
    "open space": "open space",
    "agriculture": "open space",
    "rural": "open space",
    "vacant": "open space",
    # Road / Transportation — only from actual road geometry
    "road": "road",
    "highway": "road",
    "transportation": "road",
    "right of way": "road",
    # Unknown / catch-all
    "unknown": "unknown",
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
