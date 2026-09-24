"""Local clock time for hour-of-day and calendar-day rules.

Night windows (for example 20:00 to 07:00) and calendar days are local quantities, while EMBER stores timestamps in
UTC. These helpers read timestamps on the study area's clock. Pass ``local_tz`` (an IANA name such as
``"America/Los_Angeles"``) wherever a function takes it; naive timestamps are taken to be local clock time already.
"""

from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd

UTC_WARNING = (
    "timestamps are in UTC and no local_tz was given, so hour-of-day and calendar-day rules read UTC clock time; "
    "pass local_tz, for example 'America/Los_Angeles'"
)


def _tz_of(values):
    if isinstance(values, pd.Series):
        return values.dt.tz if isinstance(values.dtype, pd.DatetimeTZDtype) else None
    if isinstance(values, pd.Timestamp):
        return values.tzinfo
    return None


def _is_utc(tz) -> bool:
    return tz is not None and str(tz).upper() in {"UTC", "ETC/UTC", "GMT", "ETC/GMT", "UTC+00:00"}


def resolve_tz(values, local_tz: Optional[str] = None) -> Optional[str]:
    """The zone to read clock time in: ``local_tz`` if given, else the values' own zone unless that zone is UTC.

    UTC values with no ``local_tz`` raise a warning and return None, which keeps UTC clock time."""
    if local_tz is not None:
        return local_tz
    tz = _tz_of(values)
    if tz is None:
        return None
    if _is_utc(tz):
        warnings.warn(UTC_WARNING, UserWarning, stacklevel=3)
        return None
    return str(tz)


def local_clock(values, local_tz: Optional[str] = None, *, assume_utc: bool = False):
    """``values`` (a Series or a Timestamp) as naive clock times in ``local_tz``.

    Tz-aware values are converted to ``local_tz``. Naive values are already local, unless ``assume_utc`` marks them as
    UTC wall clock (epoch timestamps parsed without a zone). Without ``local_tz``, tz-aware values keep their own
    zone's clock, which is what ``.dt.hour`` and ``.dt.date`` already return for them."""
    if isinstance(values, pd.Series):
        s = pd.to_datetime(values)
        aware = isinstance(s.dtype, pd.DatetimeTZDtype)
        if not aware and assume_utc and local_tz is not None:
            s, aware = s.dt.tz_localize("UTC"), True
        if not aware:
            return s
        if local_tz is not None:
            s = s.dt.tz_convert(local_tz)
        return s.dt.tz_localize(None)
    ts = pd.Timestamp(values)
    if pd.isna(ts):
        return ts
    if ts.tzinfo is None and assume_utc and local_tz is not None:
        ts = ts.tz_localize("UTC")
    if ts.tzinfo is None:
        return ts
    if local_tz is not None:
        ts = ts.tz_convert(local_tz)
    return ts.tz_localize(None)
