"""Trip-chain extraction with intermediate-stop tracking."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import pandas as pd

from ..contracts import TRIP_CHAIN_SCHEMA, validate_columns
from .clustering import ClusterSummary, IncrementalClusterConfig, cluster_points, haversine_m


@dataclass(frozen=True)
class TripChainConfig:
    """Rule configuration for stop role derivation and chain post-processing."""

    home_radius_m: float = 400.0
    overnight_start_hour: int = 20
    overnight_end_hour: int = 7
    destination_min_dwell_s: float = 1800.0
    merge_nearby_stops: bool = True
    merge_distance_m: float = 120.0
    merge_gap_s: float = 3600.0
    classify_return: bool = True


def _is_overnight(start: pd.Timestamp, end: pd.Timestamp, night_start: int, night_end: int) -> bool:
    """Return True when a stop overlaps the configured night window."""
    hours = pd.date_range(start.floor("h"), end.ceil("h"), freq="h")
    for ts in hours:
        h = ts.hour
        if night_start > night_end:
            if h >= night_start or h < night_end:
                return True
        else:
            if night_start <= h < night_end:
                return True
    return False


def _merge_adjacent_clusters(
    clusters: List[ClusterSummary],
    *,
    merge_distance_m: float,
    merge_gap_s: float,
) -> List[ClusterSummary]:
    """Merge nearby consecutive clusters to reduce GPS jitter splits."""
    if not clusters:
        return []

    merged = [clusters[0]]
    next_id = clusters[0].cluster_id
    for cluster in clusters[1:]:
        prev = merged[-1]
        gap_s = (cluster.start_time - prev.end_time).total_seconds()
        dist_m = haversine_m(prev.centroid_lat, prev.centroid_lon, cluster.centroid_lat, cluster.centroid_lon)
        if gap_s <= merge_gap_s and dist_m <= merge_distance_m:
            next_id += 1
            merged[-1] = ClusterSummary(
                cluster_id=next_id,
                centroid_lat=(prev.centroid_lat * prev.n_points + cluster.centroid_lat * cluster.n_points)
                / (prev.n_points + cluster.n_points),
                centroid_lon=(prev.centroid_lon * prev.n_points + cluster.centroid_lon * cluster.n_points)
                / (prev.n_points + cluster.n_points),
                start_time=prev.start_time,
                end_time=cluster.end_time,
                n_points=prev.n_points + cluster.n_points,
                dwell_s=(cluster.end_time - prev.start_time).total_seconds(),
            )
        else:
            merged.append(cluster)
            next_id = cluster.cluster_id
    return merged


def build_trip_chain_for_user(
    pings: pd.DataFrame,
    *,
    user_id: str,
    home_lat: Optional[float] = None,
    home_lon: Optional[float] = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
    clustering_config: Optional[IncrementalClusterConfig] = None,
    home_radius_m: float = 400.0,
    overnight_start_hour: int = 20,
    overnight_end_hour: int = 7,
    destination_min_dwell_s: float = 1800.0,
    merge_nearby_stops: bool = True,
    merge_distance_m: float = 120.0,
    merge_gap_s: float = 3600.0,
    classify_return: bool = True,
    rules: Optional[TripChainConfig] = None,
) -> pd.DataFrame:
    """Build ordered stop-chain for one user."""
    if pings.empty:
        return pd.DataFrame()

    clusters: List[ClusterSummary] = cluster_points(
        pings,
        lat_col=lat_col,
        lon_col=lon_col,
        time_col=time_col,
        config=clustering_config,
    )
    if not clusters:
        return pd.DataFrame()

    cfg = rules or TripChainConfig(
        home_radius_m=home_radius_m,
        overnight_start_hour=overnight_start_hour,
        overnight_end_hour=overnight_end_hour,
        destination_min_dwell_s=destination_min_dwell_s,
        merge_nearby_stops=merge_nearby_stops,
        merge_distance_m=merge_distance_m,
        merge_gap_s=merge_gap_s,
        classify_return=classify_return,
    )
    if cfg.merge_nearby_stops:
        clusters = _merge_adjacent_clusters(
            clusters,
            merge_distance_m=cfg.merge_distance_m,
            merge_gap_s=cfg.merge_gap_s,
        )

    rows: List[Dict] = []
    prev_lat = None
    prev_lon = None
    seen_away = False

    for idx, cluster in enumerate(clusters, start=1):
        dist_home_m = None
        if home_lat is not None and home_lon is not None:
            dist_home_m = haversine_m(
                home_lat,
                home_lon,
                cluster.centroid_lat,
                cluster.centroid_lon,
            )

        dist_prev_m = None
        if prev_lat is not None and prev_lon is not None:
            dist_prev_m = haversine_m(
                prev_lat,
                prev_lon,
                cluster.centroid_lat,
                cluster.centroid_lon,
            )

        overnight = _is_overnight(
            cluster.start_time,
            cluster.end_time,
            cfg.overnight_start_hour,
            cfg.overnight_end_hour,
        )
        if dist_home_m is not None and dist_home_m <= cfg.home_radius_m:
            role = "return" if cfg.classify_return and seen_away else "home"
        elif overnight:
            role = "overnight"
            seen_away = True
        elif cluster.dwell_s >= cfg.destination_min_dwell_s:
            role = "destination"
            seen_away = True
        else:
            role = "intermediate"
            seen_away = True

        rows.append(
            {
                "user_id": user_id,
                "seq_idx": idx,
                "stop_id": cluster.cluster_id,
                "start_ts": cluster.start_time,
                "end_ts": cluster.end_time,
                "dwell_s": cluster.dwell_s,
                "lat": cluster.centroid_lat,
                "lon": cluster.centroid_lon,
                "distance_from_home_m": dist_home_m,
                "distance_from_prev_m": dist_prev_m,
                "is_overnight": overnight,
                "stop_role": role,
            }
        )
        prev_lat = cluster.centroid_lat
        prev_lon = cluster.centroid_lon

    return pd.DataFrame(rows)


def build_trip_chain(
    pings: pd.DataFrame,
    *,
    id_col: str = "ID",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
    homes: Optional[pd.DataFrame] = None,
    home_id_col: str = "ID",
    home_lat_col: str = "home_lat_4326",
    home_lon_col: str = "home_lon_4326",
    clustering_config: Optional[IncrementalClusterConfig] = None,
    home_radius_m: float = 400.0,
    overnight_start_hour: int = 20,
    overnight_end_hour: int = 7,
    destination_min_dwell_s: float = 1800.0,
    merge_nearby_stops: bool = True,
    merge_distance_m: float = 120.0,
    merge_gap_s: float = 3600.0,
    classify_return: bool = True,
    rules: Optional[TripChainConfig] = None,
) -> pd.DataFrame:
    """Build trip-chain DataFrame for all users in pings."""
    if pings.empty:
        return pd.DataFrame()
    required_ping_cols = [id_col, lat_col, lon_col, time_col]
    missing_ping_cols = [c for c in required_ping_cols if c not in pings.columns]
    if missing_ping_cols:
        raise ValueError(
            f"pings must contain columns {required_ping_cols}. "
            f"Missing: {missing_ping_cols}"
        )

    home_lookup = {}
    if homes is not None and not homes.empty:
        required_home_cols = [home_id_col, home_lat_col, home_lon_col]
        missing_home_cols = [c for c in required_home_cols if c not in homes.columns]
        if missing_home_cols:
            raise ValueError(
                f"homes must contain columns {required_home_cols}. "
                f"Missing: {missing_home_cols}"
            )
        for _, row in homes[[home_id_col, home_lat_col, home_lon_col]].drop_duplicates(subset=[home_id_col]).iterrows():
            home_lookup[row[home_id_col]] = (
                float(row[home_lat_col]),
                float(row[home_lon_col]),
            )

    chains: List[pd.DataFrame] = []
    for uid, grp in pings.groupby(id_col):
        home_lat, home_lon = home_lookup.get(uid, (None, None))
        user_chain = build_trip_chain_for_user(
            grp,
            user_id=uid,
            home_lat=home_lat,
            home_lon=home_lon,
            lat_col=lat_col,
            lon_col=lon_col,
            time_col=time_col,
            clustering_config=clustering_config,
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
        if not user_chain.empty:
            chains.append(user_chain)

    if not chains:
        return pd.DataFrame()
    out = pd.concat(chains, ignore_index=True)
    validate_columns(out, TRIP_CHAIN_SCHEMA)
    return out
