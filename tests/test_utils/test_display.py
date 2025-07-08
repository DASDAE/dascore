"""Tests for displaying dascore objects."""

from __future__ import annotations

import numpy as np
import pandas as pd

import dascore as dc
from dascore.utils.display import get_nice_text


class TestGetNiceText:
    """Tests for converting coordinate to nice looking rich Text."""

    def test_simple_datetime(self):
        """Ensure the process works for datetime objects."""
        dt = dc.to_datetime64("2023-10-01")
        # YMD should just show YMD
        txt1 = get_nice_text(dt)
        assert str(txt1) == "2023-10-01"
        # Unless YMD is 1970-01-01
        txt2 = get_nice_text(dc.to_datetime64(0))
        assert str(txt2) == "00:00:00"
        # Decimals are displayed if present
        txt3 = get_nice_text(dc.to_datetime64(1.111111111))
        assert str(txt3).endswith(".111111111")

    def test_nat(self):
        """Tests for NaT."""
        dt = np.datetime64("NaT")
        txt = get_nice_text(dt)
        assert str(txt) == "NaT"

    def test_timestamp(self):
        """Tests for pandas timestamps."""
        ts = pd.Timestamp("2012-01-10")
        txt = get_nice_text(ts)
        assert str(txt) == "2012-01-10"
