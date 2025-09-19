"""Tests for TDMS utilities to improve coverage."""

from __future__ import annotations

import pytest

from dascore.io.tdms.utils import parse_time_stamp, type_not_supported


class TestTDMSUtils:
    """Tests for TDMS utility functions."""

    def test_type_not_supported(self):
        """Test that type_not_supported raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="Reading of this tdsDataType is not implemented"
        ):
            type_not_supported("any_input")

    def test_parse_time_stamp_none(self):
        """Test that parse_time_stamp returns None for invalid input."""
        # Test with invalid timestamp values that should return None
        result = parse_time_stamp(0, 0)  # epoch start should return None
        assert result is None

        # Test with None values
        result = parse_time_stamp(None, 100)
        assert result is None

        result = parse_time_stamp(100, None)
        assert result is None

        result = parse_time_stamp(None, None)
        assert result is None

    def test_parse_time_stamp_valid(self):
        """Test that parse_time_stamp works with valid input."""
        # Test with valid timestamp - using a reasonable epoch timestamp
        # LabVIEW epoch starts at 1904-01-01, so we need a positive value
        import datetime

        # Use a timestamp that represents a valid date after 1904
        seconds = 365 * 24 * 3600 * 100  # 100 years after 1904
        fractions = 0.5  # Some fractional seconds

        result = parse_time_stamp(fractions, seconds)
        assert isinstance(result, datetime.datetime)
        assert result.year >= 1904
