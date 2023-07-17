"""
Module for testing units.
"""
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnitError
from dascore.units import (
    get_conversion_factor,
    get_factor_and_unit,
    get_filter_units,
    get_quantity,
    get_unit,
    invert_quantity,
    validate_quantity,
)


class TestUnitInit:
    """Ensure units can be initialized."""

    def test_time(self):
        """Tests for time units"""
        sec = dc.get_unit("s")
        assert str(sec.dimensionality) == "[time]"

    def test_conversion_factor(self):
        """Ensure conversion factors are calculated correctly."""
        assert get_conversion_factor("s", "ms") == 1000.0
        assert get_conversion_factor("ms", "s") == 1 / 1000.0
        assert np.round(get_conversion_factor("ft", "m"), 5) == 0.3048

    def test_invert(self):
        """Ensure a unit can be inverted."""
        unit = dc.get_unit("s")
        inverted = invert_quantity(unit)
        reverted = invert_quantity(inverted)
        assert unit == reverted
        assert invert_quantity(None) is None

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
        assert validate_quantity(in_str)

    @pytest.mark.parametrize("in_str", invalid)
    def test_validate_units_bad_input(self, in_str):
        """Ensure units can be validated from various inputs."""
        with pytest.raises(UnitError):
            validate_quantity(in_str)

    def test_none(self):
        """Ensure none and empty str also works."""
        assert validate_quantity(None) is None
        assert validate_quantity("") is None


class TestUnitAndFactor:
    """tests for returning units and scaling factor"""

    def test_quantx_units(self):
        """tests for the quantx unit str."""
        mag, ustr = get_factor_and_unit("rad * 2pi/2^16")
        assert ustr == "pi * radian"
        assert np.isclose(mag, (2 / (2**16)))

    def test_simplify_units(self):
        """test for reducing units."""
        mag, ustr = get_factor_and_unit("rad * (km/m)", simplify=True)
        assert get_unit(ustr) == get_unit("radian")
        assert np.isclose(mag, 1000)

    def test_none(self):
        """Ensure none returns a None and string"""
        factor, unit = get_factor_and_unit(None)
        assert factor == 1
        assert unit is None


class TestGetQuantity:
    """Tests for getting a quantity."""

    def test_quantity_identity(self):
        """Get quantity should always return the same quantity."""
        quant1 = get_quantity("1/s")
        quant2 = get_quantity("1 Hz")
        assert quant1 == get_quantity(quant1)
        assert quant1 is get_quantity(quant1)
        assert quant2 == get_quantity(quant2)
        assert quant2 is get_quantity(quant2)


class TestConvenientImport:
    """Tests for conveniently importing units for dascore.units"""

    def test_import_common(self):
        """Ensure common units are importable"""
        from dascore.units import Hz, ft, km, m, miles  # noqa

        assert m == get_unit("m")
        assert ft == get_unit("ft")
        assert miles == get_unit("miles")
        assert km == get_unit("km")
        assert Hz == get_unit("Hz")

    def test_bad_import_error_msg(self):
        """An import error should be raised if the unit isn't valid."""
        with pytest.raises(ImportError):
            from dascore.utils import bob  # noqa


class TestGetFilterUnits:
    """Tests for getting units that can be used for filtering."""

    def test_no_units(self):
        """Tests for when no units are specified."""
        assert get_filter_units(1, 10, "m") == (1.0, 10.0)
        assert get_filter_units(None, 10, "s") == (None, 10.0)
        assert get_filter_units(1, None, "s") == (1.0, None)

    def test_filter_units(self):
        """Tests for when filter units are already those selected."""
        Hz = get_unit("Hz")
        s = get_unit("s")
        assert get_filter_units(1.0 * Hz, 10.0 * Hz, s) == (1.0, 10.0)
        assert get_filter_units(None, 10.0 * Hz, s) == (None, 10.0)
        assert get_filter_units(1.0 * Hz, 10.0 * Hz, s) == (1.0, 10.0)

    def test_same_units(self):
        """Tests for when filter units are already those selected."""
        s = get_unit("s")
        assert get_filter_units(1.0 * s, 10.0 * s, s) == (0.1, 1.0)
        assert get_filter_units(None, 10.0 * s, s) == (None, 0.1)
        assert get_filter_units(10.0 * s, None, s) == (0.1, None)

    def test_different_units_raises(self):
        """The units must be the same or it should raise."""
        s, Hz = get_unit("s"), get_unit("Hz")

        with pytest.raises(UnitError):
            get_filter_units(1.0, 10.0 * s, s)

        with pytest.raises(UnitError):
            get_filter_units(1.0 * s, 10.0 * Hz, s)

    def test_incompatible_units_raise(self):
        """The units must be the same or it should raise."""
        s, m = get_unit("s"), get_unit("m")
        match = "Cannot convert from"
        with pytest.raises(UnitError, match=match):
            get_filter_units(1.0 * s, 10.0 * s, m)

        with pytest.raises(UnitError, match=match):
            get_filter_units(1.0 * m, 10.0 * m, s)
