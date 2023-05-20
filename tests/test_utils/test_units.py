"""
Module for testing units.
"""
import numpy as np

import dascore as dc
from dascore.utils.units import get_conversion_factor, invert_unit


class TestUnitInit:
    """Ensure units can be initialized."""

    def test_time(self):
        """Tests for time units"""
        sec = dc.Unit("s")
        assert str(sec.dimensionality) == "[time]"

    def test_conversion_factor(self):
        """Ensure conversion factors are calculated correctly."""
        assert get_conversion_factor("s", "ms") == 1000.0
        assert get_conversion_factor("ms", "s") == 1 / 1000.0
        assert np.round(get_conversion_factor("ft", "m"), 5) == 0.3048

    def test_invert(self):
        """Ensure a unit can be inverted."""
        unit = dc.Unit("s")
        inverted = invert_unit(unit)
        reverted = invert_unit(inverted)
        assert unit == reverted
