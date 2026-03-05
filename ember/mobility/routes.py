"""General route-integration interfaces built on stop-chain artifacts."""

from __future__ import annotations

from typing import Callable, Optional

import pandas as pd

from ..contracts import TRIP_CHAIN_SCHEMA, validate_columns
from .clustering import haversine_m


def build_trip_legs(
    stop_chain_df: pd.DataFrame,
    pings_df: Optional[pd.DataFrame] = None,
    *,
    user_col: str = "user_id",
) -> pd.DataFrame:
    """Build sequential trip legs between adjacent stops.

    This is a stable interface for downstream routing work.
    It only derives transition-level leg metadata from the stop chain.
    """
    if stop_chain_df.empty:
        return pd.DataFrame(
            columns=[
                user_col,
                "from_seq",
                "to_seq",
                "depart_ts",
                "arrive_ts",
                "from_lat",
                "from_lon",
                "to_lat",
                "to_lon",
                "displacement_m",
            ]
        )

    validate_columns(stop_chain_df, TRIP_CHAIN_SCHEMA)
    rows = []
    for uid, grp in stop_chain_df.groupby(user_col):
        ordered = grp.sort_values("seq_idx").reset_index(drop=True)
        for i in range(len(ordered) - 1):
            a = ordered.iloc[i]
            b = ordered.iloc[i + 1]
            rows.append(
                {
                    user_col: uid,
                    "from_seq": int(a["seq_idx"]),
                    "to_seq": int(b["seq_idx"]),
                    "depart_ts": a["end_ts"],
                    "arrive_ts": b["start_ts"],
                    "from_lat": float(a["lat"]),
                    "from_lon": float(a["lon"]),
                    "to_lat": float(b["lat"]),
                    "to_lon": float(b["lon"]),
                    # Direct stop-to-stop displacement; not a routed path length.
                    "displacement_m": haversine_m(
                        float(a["lat"]),
                        float(a["lon"]),
                        float(b["lat"]),
                        float(b["lon"]),
                    ),
                }
            )
    return pd.DataFrame(rows)


def infer_route_candidates(
    trip_legs_df: pd.DataFrame,
    network_graph=None,
    *,
    strategy_name: Optional[str] = None,
    strategy_fn: Optional[Callable[[pd.DataFrame, object], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Return route candidates via a pluggable strategy callback.

    By default (no strategy callback), this returns a neutral placeholder
    status for each leg and does not assume any routing model.
    """
    if trip_legs_df.empty:
        return pd.DataFrame(
            columns=["user_id", "from_seq", "to_seq", "strategy", "route_status", "route_length_m"]
        )

    base = trip_legs_df.copy()
    name = strategy_name or "unspecified"

    if strategy_fn is None:
        out = base[["user_id", "from_seq", "to_seq"]].copy()
        out["strategy"] = name
        out["route_status"] = "unresolved_strategy"
        out["route_length_m"] = pd.NA
        return out

    routed = strategy_fn(base, network_graph)
    if not isinstance(routed, pd.DataFrame):
        raise TypeError("strategy_fn must return a pandas DataFrame")
    for col in ("user_id", "from_seq", "to_seq"):
        if col not in routed.columns:
            raise ValueError(f"strategy_fn output missing required column: {col}")
    if "strategy" not in routed.columns:
        routed["strategy"] = name
    return routed
