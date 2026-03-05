"""Canonical tabular schemas and lightweight validators for EMBER."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import pandas as pd


@dataclass(frozen=True)
class SchemaSpec:
    """Column contract for a DataFrame-like artifact."""

    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    aliases: Mapping[str, tuple[str, ...]] = None


PINGS_SCHEMA = SchemaSpec(
    name="pings",
    required_columns=("ID", "datetime"),
    optional_columns=("latitude", "longitude", "LAT", "LONG", "timestamp_ms", "TIMESTAMP"),
    aliases={
        "latitude": ("LAT",),
        "longitude": ("LONG",),
    },
)

HOMES_SCHEMA = SchemaSpec(
    name="homes",
    required_columns=("ID", "home_lat_4326", "home_lon_4326"),
)

TRIP_CHAIN_SCHEMA = SchemaSpec(
    name="trip_chain",
    required_columns=(
        "user_id",
        "seq_idx",
        "start_ts",
        "end_ts",
        "dwell_s",
        "lat",
        "lon",
        "distance_from_home_m",
        "distance_from_prev_m",
        "stop_role",
    ),
    optional_columns=("stop_id", "is_overnight"),
)

DESTINATION_SCHEMA = SchemaSpec(
    name="destinations",
    required_columns=("ID", "dest_lat", "dest_lon", "dest_date", "eu_distance_km", "dest_order"),
    optional_columns=("source_role", "dest_type"),
)


def validate_columns(df: pd.DataFrame, spec: SchemaSpec, *, allow_aliases: bool = True) -> None:
    """Raise ValueError if required columns are missing."""
    cols = set(df.columns)
    missing: list[str] = []

    for required in spec.required_columns:
        if required in cols:
            continue
        if allow_aliases and spec.aliases and required in spec.aliases:
            if any(alias in cols for alias in spec.aliases[required]):
                continue
        missing.append(required)

    if missing:
        raise ValueError(
            f"{spec.name} schema missing required columns: {missing}. "
            f"Found columns: {sorted(cols)}"
        )


def canonicalize_columns(df: pd.DataFrame, spec: SchemaSpec) -> pd.DataFrame:
    """Return a copy with alias columns renamed to canonical names when needed."""
    out = df.copy()
    if not spec.aliases:
        return out

    for canonical, aliases in spec.aliases.items():
        if canonical in out.columns:
            continue
        for alias in aliases:
            if alias in out.columns:
                out = out.rename(columns={alias: canonical})
                break
    return out


def optional_missing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return optional columns that are currently missing."""
    present = set(df.columns)
    return [c for c in columns if c not in present]
