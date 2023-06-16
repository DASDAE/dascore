"""
Module for testing units.
"""
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnitError
from dascore.utils.units import get_conversion_factor, invert_unit, validate_units


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

    def test_conversion_factor_none(self):
        """If either unit is None it should return 1."""
        assert get_conversion_factor(None, None) == 1
        assert get_conversion_factor(None, "m") == 1
        assert get_conversion_factor("m", None) == 1


class TestValidateUnits:
    """Ensure units can be validated."""

    valid = ("m", "s", "Hz", "1/s", "1/m", "feet", "furlongs", "km", "fortnight")
    invalid = ("bob", "gerbil inches", "farsee")

    @pytest.mark.parametrize("in_str", valid)
    def test_validate_units_good_input(self, in_str):
        """Ensure units can be validated from various inputs."""
        assert validate_units(in_str)

    @pytest.mark.parametrize("in_str", invalid)
    def test_validate_units_bad_input(self, in_str):
        """Ensure units can be validated from various inputs."""
        with pytest.raises(UnitError):
            validate_units(in_str)

    def test_none(self):
        """Ensure none and empty str also works."""
        assert validate_units(None) is None
        assert validate_units("") == ""
