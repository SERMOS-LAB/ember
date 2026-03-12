"""
End-to-end evacuation pipeline.

Ports the core inference logic from the standalone
``run_evacuation_inference.py`` script into EMBER-native functions
so that the entire workflow can run inside a notebook.

Typical usage
-------------
>>> import ember.io as eio
>>> import ember.zones as ezones
>>> import ember.pipeline as epipe
>>>
>>> homes   = eio.load_homes('data/home.csv')
>>> zones   = eio.load_fire_timeline('data/Fire_Timeline')
>>> zoned   = ezones.classify_zones(homes, zones, buffer_distance=2000)
>>> dau     = epipe.load_gps_csv('data/dau_during_fire.csv')
>>> stops   = epipe.compute_stops(dau, zoned, zones)
>>> metrics = epipe.infer_metrics(stops, fire_starts={'Palisades': '2025-01-07 10:30',
...                                                    'Eaton':     '2025-01-07 18:18'})
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
from pyproj import Transformer
from datetime import timedelta
from typing import Dict, Optional
from .contracts import PINGS_SCHEMA, canonicalize_columns, validate_columns
from .mobility.trip_chain import TripChainConfig, build_trip_chain


# ---------------------------------------------------------------------------
# 1 · LOAD RAW GPS ----------------------------------------------------------
# ---------------------------------------------------------------------------

def load_gps_csv(path: str, id_col: str = "GRID", *, verbose: bool = True) -> pd.DataFrame:
    """
    Load a raw GPS trajectory CSV (``dau_during_fire.csv`` or similar).

    Normalises the user-ID column to ``ID`` and parses timestamps.
    """
    if verbose:
        print(f"Loading GPS data from {path} …")
    df = pd.read_csv(path, low_memory=False)
    if id_col in df.columns and id_col != "ID":
        df.rename(columns={id_col: "ID"}, inplace=True)

    # Ensure a datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])
    elif "TIMESTAMP" in df.columns:
        df["datetime"] = pd.to_datetime(df["TIMESTAMP"], unit="ms", utc=True)
    elif "Date" in df.columns:
        df["datetime"] = pd.to_datetime(df["Date"])

    if verbose:
        print(f"  → {len(df):,} pings, {df['ID'].nunique():,} unique devices")
    return df


# ---------------------------------------------------------------------------
# 2 · COMPUTE STOPS ---------------------------------------------------------
# ---------------------------------------------------------------------------

def _check_outside_zones(stops: pd.DataFrame,
                         zone_union_3857) -> pd.DataFrame:
    """Point-in-polygon test: True if the ping is outside the buffered zone."""
    if stops.empty:
        stops["is_outside_zone"] = False
        return stops

    unique_pts = stops[["lon_grid", "lat_grid"]].drop_duplicates()
    pts_geo = gpd.GeoDataFrame(
        unique_pts,
        geometry=gpd.points_from_xy(unique_pts.lon_grid, unique_pts.lat_grid),
        crs="EPSG:4326",
    )
    # Project the zone polygon back to 4326 for the sjoin
    zone_gdf = gpd.GeoDataFrame({"geometry": [zone_union_3857]}, crs="EPSG:3857").to_crs("EPSG:4326")
    zone_poly_4326 = zone_gdf.iloc[0].geometry

    union_gdf = gpd.GeoDataFrame({"geometry": [zone_poly_4326]}, crs="EPSG:4326")
    inside = gpd.sjoin(pts_geo, union_gdf, how="inner", predicate="within")
    unique_pts["is_outside_zone"] = ~unique_pts.index.isin(inside.index.unique())

    stops = pd.merge(stops, unique_pts[["lon_grid", "lat_grid", "is_outside_zone"]],
                     on=["lon_grid", "lat_grid"], how="left")
    return stops


def compute_stops(
    dau: pd.DataFrame,
    zoned_homes: gpd.GeoDataFrame,
    fire_zones: gpd.GeoDataFrame,
    *,
    buffer_meters: float = 2000,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Join GPS pings with home locations, compute distances, and flag zone exit.

    Parameters
    ----------
    dau : DataFrame
        Raw GPS pings from :func:`load_gps_csv`.
    zoned_homes : GeoDataFrame
        Output of :func:`ember.zones.classify_zones` (needs ``ID``,
        ``ZoneType``, ``OrderStart``, ``FireEvent``).
    fire_zones : GeoDataFrame
        The original fire-zone polygons (used to build the buffer polygon).
    buffer_meters : float
        Buffer distance in metres for the outside-zone check.

    Returns
    -------
    DataFrame
        One row per ping for each study-area resident, with ``eu_distance``,
        ``is_outside_zone``, and zone metadata.
    """
    # Keep only study-area residents (not Outside)
    study = zoned_homes[zoned_homes["ZoneType"] != "Outside"].copy()
    target_ids = study["ID"].unique()
    stops = dau[dau["ID"].isin(target_ids)].copy()

    if stops.empty:
        return pd.DataFrame()

    if verbose:
        print(f"Computing stops for {len(target_ids):,} residents ({len(stops):,} pings) …")

    # Prepare home info in EPSG:3857 for distance calc
    study_proj = study.to_crs(epsg=3857)
    study_proj["home_lon"] = study_proj.geometry.x
    study_proj["home_lat"] = study_proj.geometry.y

    merge_cols = ["ID", "home_lon", "home_lat", "ZoneType", "OrderStart"]
    if "FireEvent" in study_proj.columns:
        merge_cols.append("FireEvent")
    if "OrderEnd" in study_proj.columns:
        merge_cols.append("OrderEnd")


    # Deduplicate merge source on ID
    merge_src = study_proj[merge_cols].drop_duplicates(subset=["ID"])
    stops = pd.merge(stops, merge_src, on="ID", how="inner")

    # Parse ping times
    if "datetime" in stops.columns:
        stops["stop_date"] = pd.to_datetime(stops["datetime"])
    elif "TIMESTAMP" in stops.columns:
        stops["stop_date"] = pd.to_datetime(stops["TIMESTAMP"], unit="ms")
    else:
        stops["stop_date"] = pd.to_datetime(stops["Date"])

    # Distance calculation in EPSG:3857
    to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

    stops["lat_grid"] = pd.to_numeric(stops["LAT"], errors="coerce").clip(-85, 85)
    stops["lon_grid"] = pd.to_numeric(stops["LONG"], errors="coerce")

    sx, sy = to_3857.transform(stops["lon_grid"].values, stops["lat_grid"].values)
    stops["dist_meters"] = np.sqrt((sx - stops["home_lon"]) ** 2 +
                                   (sy - stops["home_lat"]) ** 2)
    stops["eu_distance"] = stops["dist_meters"] * 0.000621371  # to miles

    # Build combined zone polygon in 3857 for the outside-zone check
    zones_proj = fire_zones.to_crs(epsg=3857)
    zone_union = zones_proj.union_all()
    buffer_poly = zone_union.buffer(buffer_meters)

    stops = _check_outside_zones(stops, buffer_poly)

    # Impute missing distances
    stops = _impute_distances(stops)

    if verbose:
        print(f"  → {len(stops):,} stop records ready")
    return stops


def compute_trip_chain(
    dau: pd.DataFrame,
    homes: pd.DataFrame,
    *,
    id_col: str = "ID",
    lat_col: str = "LAT",
    lon_col: str = "LONG",
    time_col: str = "datetime",
    home_id_col: str = "ID",
    home_lat_col: str = "home_lat_4326",
    home_lon_col: str = "home_lon_4326",
    home_radius_m: float = 400.0,
    overnight_start_hour: int = 20,
    overnight_end_hour: int = 7,
    destination_min_dwell_s: float = 1800.0,
    merge_nearby_stops: bool = True,
    merge_distance_m: float = 120.0,
    merge_gap_s: float = 3600.0,
    classify_return: bool = True,
) -> pd.DataFrame:
    """Build user stop chains with intermediate-stop tracking."""
    points = canonicalize_columns(dau, PINGS_SCHEMA)
    homes_df = homes.copy()
    resolved_id_col = id_col
    resolved_home_id_col = home_id_col
    if id_col in points.columns and id_col != "ID":
        points = points.rename(columns={id_col: "ID"})
        resolved_id_col = "ID"
    if home_id_col in homes_df.columns and home_id_col != "ID":
        homes_df = homes_df.rename(columns={home_id_col: "ID"})
        resolved_home_id_col = "ID"

    if lat_col in points.columns and "latitude" not in points.columns:
        points = points.rename(columns={lat_col: "latitude"})
    if lon_col in points.columns and "longitude" not in points.columns:
        points = points.rename(columns={lon_col: "longitude"})
    if time_col not in points.columns and "TIMESTAMP" in points.columns:
        points[time_col] = pd.to_datetime(points["TIMESTAMP"], unit="ms", utc=True)
    if "datetime" not in points.columns and time_col in points.columns:
        points["datetime"] = points[time_col]
    validate_columns(points, PINGS_SCHEMA)
    required_home_cols = [resolved_home_id_col, home_lat_col, home_lon_col]
    missing_home_cols = [c for c in required_home_cols if c not in homes_df.columns]
    if missing_home_cols:
        raise ValueError(
            f"homes must contain columns {required_home_cols}. "
            f"Missing: {missing_home_cols}"
        )

    rules = TripChainConfig(
        home_radius_m=home_radius_m,
        overnight_start_hour=overnight_start_hour,
        overnight_end_hour=overnight_end_hour,
        destination_min_dwell_s=destination_min_dwell_s,
        merge_nearby_stops=merge_nearby_stops,
        merge_distance_m=merge_distance_m,
        merge_gap_s=merge_gap_s,
        classify_return=classify_return,
    )
    return build_trip_chain(
        points,
        id_col=resolved_id_col,
        lat_col="latitude",
        lon_col="longitude",
        time_col=time_col,
        homes=homes_df,
        home_id_col=resolved_home_id_col,
        home_lat_col=home_lat_col,
        home_lon_col=home_lon_col,
        home_radius_m=home_radius_m,
        overnight_start_hour=overnight_start_hour,
        overnight_end_hour=overnight_end_hour,
        destination_min_dwell_s=destination_min_dwell_s,
        merge_nearby_stops=merge_nearby_stops,
        merge_distance_m=merge_distance_m,
        merge_gap_s=merge_gap_s,
        classify_return=classify_return,
        rules=rules,
    )


def _impute_distances(stops: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill missing ``eu_distance`` within each user's timeline."""
    clean = []
    for _, grp in stops.groupby("ID"):
        g = grp.sort_values("stop_date").copy()
        if g["eu_distance"].isna().mean() > 0.5:
            continue
        if np.isnan(g.iloc[0]["eu_distance"]):
            g.iloc[0, g.columns.get_loc("eu_distance")] = 0
        g["eu_distance"] = g["eu_distance"].ffill()
        clean.append(g)
    return pd.concat(clean, ignore_index=True) if clean else pd.DataFrame(columns=stops.columns)


# ---------------------------------------------------------------------------
# 3 · INFER METRICS ---------------------------------------------------------
# ---------------------------------------------------------------------------

def _consecutive_intervals(dates: list) -> list:
    """Group dates within 1 day of each other into intervals."""
    if not dates:
        return []
    dates = sorted(set(dates))
    intervals, cur = [], [dates[0]]
    for i in range(1, len(dates)):
        if (dates[i] - cur[-1]).days <= 1:
            cur.append(dates[i])
        else:
            intervals.append(cur)
            cur = [dates[i]]
    intervals.append(cur)
    return intervals


def _select_interval(intervals: list, start_ts: pd.Timestamp):
    """First interval whose last day is >= the order date."""
    if not intervals:
        return None
    if pd.isna(start_ts):
        return intervals[0] if intervals else None
    ref = start_ts.date()
    for iv in intervals:
        if iv[-1] >= ref:
            return iv
    return None


def infer_metrics(
    stops: pd.DataFrame,
    fire_starts: Dict[str, str],
    *,
    fire_end: str = "2025-01-14",
    min_dist_miles: float = 0.25,
    min_evac_days: int = 2,
    min_evac_days_buffer: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Classify each resident's evacuation behaviour from processed stop data.

    Parameters
    ----------
    stops : DataFrame
        Output of :func:`compute_stops`.  May optionally contain an
        ``OrderEnd`` column with per-zone order/warning lift dates.
    fire_starts : dict
        ``{FireEvent: timestamp_str}`` mapping, e.g.
        ``{'Palisades': '2025-01-07 10:30', 'Eaton': '2025-01-07 18:18'}``.
    fire_end : str
        Date string for the end of the study window (global fallback
        when no per-zone ``OrderEnd`` is available).
    min_dist_miles : float
        Minimum distance from home (miles) to count as "away".
    min_evac_days : int
        Minimum number of consecutive calendar days a resident must
        be away from home to be considered an evacuee (in-zone
        residents, i.e. Order / Warning).  Default 2 (= "more than
        one day" per Sun et al.).
    min_evac_days_buffer : int
        Same threshold for buffer-zone residents.  Default 1 (N = 1
        for the Marshall Fire per Sun et al.).
    verbose : bool
        Print progress information.

    Returns
    -------
    DataFrame
        One row per resident with ``Category``, ``DepartureDate``,
        ``ReturnDate``, ``DelayHours``, and zone metadata.
    """
    fire_start_map = {k: pd.Timestamp(v) for k, v in fire_starts.items()}
    global_order_end = pd.to_datetime(fire_end) + timedelta(hours=23)

    # --- Timezone alignment ---------------------------------------------------
    # stop_date may be tz-aware (UTC) when parsed from epoch timestamps while
    # fire_starts / fire_end are typically tz-naive strings.  Align everything
    # to the same tz to avoid "Cannot subtract tz-naive and tz-aware" errors.
    _sample_tz = stops["stop_date"].dt.tz if not stops.empty else None
    if _sample_tz is not None:
        fire_start_map = {
            k: v.tz_localize(_sample_tz) if v.tzinfo is None else v.tz_convert(_sample_tz)
            for k, v in fire_start_map.items()
        }
        if global_order_end.tzinfo is None:
            global_order_end = global_order_end.tz_localize(_sample_tz)
    # --------------------------------------------------------------------------

    # Universal fallback: earliest known fire start (for Buffer zones with NaN FireEvent)
    earliest_fire = min(fire_start_map.values()) if fire_start_map else pd.NaT

    stops["is_evacuated"] = stops["is_outside_zone"] & (stops["eu_distance"] >= min_dist_miles)

    has_order_end_col = "OrderEnd" in stops.columns

    records = []
    total = stops["ID"].nunique()
    if verbose:
        print(f"Inferring metrics for {total:,} residents …")

    for uid, grp in stops.groupby("ID"):
        grp = grp.sort_values("stop_date")
        zone_type = grp.iloc[0]["ZoneType"]
        raw_order = grp.iloc[0]["OrderStart"]
        order_start = pd.to_datetime(raw_order) if not pd.isna(raw_order) else pd.NaT
        fire_event = grp.iloc[0].get("FireEvent", "Unknown")

        fire_start_ts = fire_start_map.get(fire_event, pd.NaT)
        if pd.isna(fire_start_ts):
            fire_start_ts = earliest_fire

        effective_start = order_start
        if pd.isna(effective_start) and zone_type in ("Buffer", "Outside"):
            effective_start = fire_start_ts

        # Per-zone order/warning end date, falling back to global fire_end
        if has_order_end_col:
            raw_end = grp.iloc[0]["OrderEnd"]
            if pd.isna(raw_end):
                order_end = global_order_end
            else:
                o_end = pd.to_datetime(raw_end)
                if _sample_tz is not None:
                    o_end = o_end.tz_localize(_sample_tz) if o_end.tzinfo is None else o_end.tz_convert(_sample_tz)
                order_end = o_end
        else:
            order_end = global_order_end

        evac = grp[grp["is_evacuated"]]

        cat, dep, ret, delay = "NER", None, None, None

        if not evac.empty:
            evac_dates = sorted(set(evac["stop_date"].dt.date.tolist()))
            intervals = _consecutive_intervals(evac_dates)
            sel = _select_interval(intervals, effective_start)

            # Configurable consecutive-day threshold
            threshold = min_evac_days_buffer if zone_type == "Buffer" else min_evac_days
            if sel and len(sel) >= threshold:
                dep = evac[evac["stop_date"].dt.date == sel[0]]["stop_date"].min()
                ret = evac[evac["stop_date"].dt.date == sel[-1]]["stop_date"].max()

                if not pd.isna(effective_start):
                    diff_hrs = (dep - effective_start).total_seconds() / 3600.0
                    delay = max(0.0, diff_hrs)
                else:
                    delay = None

                if pd.isna(effective_start):
                    cat = "UR"
                elif dep < effective_start:
                    # Left before order/warning was issued
                    if ret < effective_start:
                        cat = "NER"   # returned before order — not an evacuee
                    else:
                        cat = "SELE"  # self-evacuee leaving early
                elif dep < order_end:
                    # Left during order/warning period
                    if zone_type == "Buffer":
                        cat = "SEFN"
                    elif zone_type in ("Warning", "Order"):
                        if ret > order_end:
                            cat = "FEUW" if zone_type == "Warning" else "FEUO"
                        else:
                            cat = "PERE"
                    else:
                        cat = "UR"
                else:
                    # Left after order/warning was lifted
                    cat = "UR"

        records.append({
            "ID": uid,
            "Category": cat,
            "DepartureDate": dep,
            "ReturnDate": ret,
            "DelayHours": delay,
            "ZoneType": zone_type,
            "OrderStart": effective_start if not pd.isna(effective_start) else order_start,
            "OrderEnd": order_end,
            "FireEvent": fire_event,

            "fire_start": fire_start_ts,
            "home_lon": grp.iloc[0].get("home_lon"),
            "is_evacuee": cat not in ("NER",),
        })

    metrics_df = pd.DataFrame(records)
    n_evac = metrics_df["is_evacuee"].sum()
    if verbose:
        print(f"  → {n_evac:,} evacuees / {len(metrics_df):,} total")
        print(metrics_df["Category"].value_counts().to_string())
    return metrics_df
