"""Generalized incremental clustering for mobility trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import List, Literal, Optional

import pandas as pd


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two (lat, lon) points in metres."""
    radius_m = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    )
    return radius_m * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@dataclass(frozen=True)
class ClusterPoint:
    """Single trajectory point used by the incremental engine."""

    latitude: float
    longitude: float
    timestamp: pd.Timestamp


@dataclass(frozen=True)
class ClusterSummary:
    """Closed-cluster output record with stable metadata."""

    cluster_id: int
    centroid_lat: float
    centroid_lon: float
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    n_points: int
    dwell_s: float


@dataclass(frozen=True)
class ClusterEvent:
    """Event emitted by the clustering engine for stream-style consumers."""

    event_type: Literal["cluster_started", "cluster_updated", "cluster_closed"]
    cluster_id: int
    timestamp: pd.Timestamp
    summary: Optional[ClusterSummary] = None


@dataclass(frozen=True)
class IncrementalClusterConfig:
    """Configurable thresholds for clustering behavior."""

    radius_m: float = 200.0
    min_dwell_s: float = 300.0
    max_gap_s: Optional[float] = None


class IncrementalClusterEngine:
    """Stateful incremental clustering engine for streaming/iterative use."""

    def __init__(self, config: IncrementalClusterConfig) -> None:
        self.config = config
        self._points: List[ClusterPoint] = []
        self._active_cluster_id = 0

    def update(self, point: ClusterPoint) -> Optional[ClusterSummary]:
        """Add one point and optionally emit a closed cluster summary."""
        events = self.update_with_events(point)
        for event in events:
            if event.event_type == "cluster_closed":
                return event.summary
        return None

    def update_with_events(self, point: ClusterPoint) -> List[ClusterEvent]:
        """Add one point and return emitted events for this update."""
        events: List[ClusterEvent] = []
        if not self._points:
            self._active_cluster_id += 1
            self._points = [point]
            events.append(
                ClusterEvent(
                    event_type="cluster_started",
                    cluster_id=self._active_cluster_id,
                    timestamp=point.timestamp,
                )
            )
            return events

        centroid_lat = sum(p.latitude for p in self._points) / len(self._points)
        centroid_lon = sum(p.longitude for p in self._points) / len(self._points)
        last_time = self._points[-1].timestamp
        gap_s = (point.timestamp - last_time).total_seconds()

        gap_break = self.config.max_gap_s is not None and gap_s > self.config.max_gap_s
        dist_break = (
            haversine_m(
                centroid_lat,
                centroid_lon,
                point.latitude,
                point.longitude,
            )
            > self.config.radius_m
        )

        if gap_break or dist_break:
            closed = self._close_current_cluster()
            if closed is not None:
                events.append(
                    ClusterEvent(
                        event_type="cluster_closed",
                        cluster_id=closed.cluster_id,
                        timestamp=closed.end_time,
                        summary=closed,
                    )
                )

            self._active_cluster_id += 1
            self._points = [point]
            events.append(
                ClusterEvent(
                    event_type="cluster_started",
                    cluster_id=self._active_cluster_id,
                    timestamp=point.timestamp,
                )
            )
            return events

        self._points.append(point)
        events.append(
            ClusterEvent(
                event_type="cluster_updated",
                cluster_id=self._active_cluster_id,
                timestamp=point.timestamp,
            )
        )
        return events

    def flush(self) -> Optional[ClusterSummary]:
        """Force-close and emit current cluster, if it passes min dwell."""
        return self._close_current_cluster()

    def flush_with_events(self) -> List[ClusterEvent]:
        """Force-close current cluster and return any emitted close event."""
        closed = self._close_current_cluster()
        if closed is None:
            return []
        return [
            ClusterEvent(
                event_type="cluster_closed",
                cluster_id=closed.cluster_id,
                timestamp=closed.end_time,
                summary=closed,
            )
        ]

    def _close_current_cluster(self) -> Optional[ClusterSummary]:
        if not self._points:
            return None

        start_time = self._points[0].timestamp
        end_time = self._points[-1].timestamp
        dwell_s = (end_time - start_time).total_seconds()
        if dwell_s < self.config.min_dwell_s:
            self._points = []
            return None

        summary = ClusterSummary(
            cluster_id=self._active_cluster_id,
            centroid_lat=sum(p.latitude for p in self._points) / len(self._points),
            centroid_lon=sum(p.longitude for p in self._points) / len(self._points),
            start_time=start_time,
            end_time=end_time,
            n_points=len(self._points),
            dwell_s=dwell_s,
        )
        self._points = []
        return summary


def _to_timestamp_utc(value: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def cluster_points(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
    config: Optional[IncrementalClusterConfig] = None,
) -> List[ClusterSummary]:
    """Batch helper for clustering a DataFrame of trajectory points."""
    if df.empty:
        return []

    cfg = config or IncrementalClusterConfig()
    ordered = df.sort_values(time_col).reset_index(drop=True).copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], format="mixed", utc=True)

    engine = IncrementalClusterEngine(cfg)
    summaries: List[ClusterSummary] = []

    for _, row in ordered.iterrows():
        point = ClusterPoint(
            latitude=float(row[lat_col]),
            longitude=float(row[lon_col]),
            timestamp=_to_timestamp_utc(row[time_col]),
        )
        closed = engine.update(point)
        if closed is not None:
            summaries.append(closed)

    tail = engine.flush()
    if tail is not None:
        summaries.append(tail)
    return summaries


def cluster_points_event_stream(
    df: pd.DataFrame,
    *,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    time_col: str = "datetime",
    config: Optional[IncrementalClusterConfig] = None,
) -> List[ClusterEvent]:
    """Batch helper returning stream-style cluster events."""
    if df.empty:
        return []

    cfg = config or IncrementalClusterConfig()
    ordered = df.sort_values(time_col).reset_index(drop=True).copy()
    ordered[time_col] = pd.to_datetime(ordered[time_col], format="mixed", utc=True)

    engine = IncrementalClusterEngine(cfg)
    events: List[ClusterEvent] = []
    for _, row in ordered.iterrows():
        point = ClusterPoint(
            latitude=float(row[lat_col]),
            longitude=float(row[lon_col]),
            timestamp=_to_timestamp_utc(row[time_col]),
        )
        events.extend(engine.update_with_events(point))
    events.extend(engine.flush_with_events())
    return events
