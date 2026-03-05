"""Tests for schema contracts and validators."""

import pandas as pd
import pytest

from ember.contracts import PINGS_SCHEMA, canonicalize_columns, validate_columns


def test_validate_columns_missing_required():
    with pytest.raises(ValueError):
        validate_columns(pd.DataFrame({"LAT": [1.0]}), PINGS_SCHEMA)


def test_canonicalize_alias_columns():
    df = pd.DataFrame({"ID": ["u1"], "datetime": ["2025-01-01"], "LAT": [34.0], "LONG": [-118.0]})
    out = canonicalize_columns(df, PINGS_SCHEMA)
    assert "latitude" in out.columns
    assert "longitude" in out.columns
