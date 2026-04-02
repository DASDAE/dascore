"""Tests for displaying dascore objects."""

from __future__ import annotations

import numpy as np
import pandas as pd

import dascore as dc
from dascore.config import set_config
from dascore.utils.display import array_to_text, get_nice_text
from dascore.utils.patch import _format_values


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

    def test_float_precision_config(self):
        """Float display precision should come from runtime config."""
        with set_config(display_float_precision=1):
            txt = get_nice_text(1.234)
        assert str(txt) == "1.2"


class TestArrayFormatting:
    """Tests for config-backed array formatting behavior."""

    def test_array_threshold_config(self):
        """Array display truncation threshold should be configurable."""
        data = np.arange(10)
        with set_config(display_array_threshold=3):
            txt = array_to_text(data)
        assert "..." in str(txt)

    def test_patch_history_threshold_config(self):
        """Patch history formatting should use the configured threshold."""
        data = np.arange(10)
        with set_config(
            display_float_precision=0,
            display_patch_history_array_threshold=3,
        ):
            out = _format_values(data)
        assert "..." in out
