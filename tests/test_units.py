"""Module for testing units."""
from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import UnitError
from dascore.units import (
    Quantity,
    assert_dtype_compatible_with_units,
    convert_units,
    get_factor_and_unit,
    get_filter_units,
    get_quantity,
    get_quantity_str,
    get_unit,
    invert_quantity,
)


class TestUnitInit:
    """Ensure units can be initialized."""

    def test_time(self):
        """Tests for time units."""
        sec = dc.get_unit("s")
        assert str(sec.dimensionality) == "[time]"

    def test_invert(self):
        """Ensure a unit can be inverted."""
        unit = dc.get_unit("s")
        inverted = invert_quantity(unit)
        reverted = invert_quantity(inverted)
        assert unit == reverted
        assert invert_quantity(None) is None


class TestGetQuantStr:
    """Ensure units can be validated."""

    valid = ("m", "s", "Hz", "1/s", "1/m", "feet", "furlongs", "km", "fortnight")
    invalid = ("bob", "gerbil inches", "farsee")

    @pytest.mark.parametrize("in_str", valid)
    def test_validate_units_good_input(self, in_str):
        """Ensure units can be validated from various inputs."""
        assert get_quantity_str(in_str)

    @pytest.mark.parametrize("in_str", invalid)
    def test_validate_units_bad_input(self, in_str):
        """Ensure units can be validated from various inputs."""
        with pytest.raises(UnitError):
            get_quantity_str(in_str)

    def test_none(self):
        """Ensure none and empty str also works."""
        assert get_quantity_str(None) is None
        assert get_quantity_str("") is None

    def test_quantity(self):
        """Ensure a quantity works."""
        # with no magnitude the string should be simple units
        quant = get_quantity("m/s")
        out = get_quantity_str(quant)
        assert out == "m / s"
        # with magnitude it should be included.
        quant = get_quantity("10 m /s")
        out = get_quantity_str(quant)
        assert "10.0" in out


class TestUnitAndFactor:
    """tests for returning units and scaling factor."""

    def test_quantx_units(self):
        """Tests for the quantx unit str."""
        mag, ustr = get_factor_and_unit("rad * 2pi/2^16")
        assert ustr == "rad * π"
        assert np.isclose(mag, (2 / (2**16)))

    def test_simplify_units(self):
        """Test for reducing units."""
        mag, ustr = get_factor_and_unit("rad * (km/m)", simplify=True)
        assert get_unit(ustr) == get_unit("radian")
        assert np.isclose(mag, 1000)

    def test_none(self):
        """Ensure none returns a None and string."""
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

    def test_get_temp(self):
        """Get quantity should work with temperatures."""
        quant1 = get_quantity("degC")
        assert "°C" in str(quant1)


class TestConvenientImport:
    """Tests for conveniently importing units for dascore.units."""

    def test_import_common(self):
        """Ensure common units are importable."""
        from dascore.units import Hz, ft, km, m, miles

        assert m == get_quantity("m")
        assert ft == get_quantity("ft")
        assert miles == get_quantity("miles")
        assert km == get_quantity("km")
        assert Hz == get_quantity("Hz")

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
        hz = get_unit("Hz")
        s = get_unit("s")
        assert get_filter_units(1.0 * hz, 10.0 * hz, s) == (1.0, 10.0)
        assert get_filter_units(None, 10.0 * hz, s) == (None, 10.0)
        assert get_filter_units(1.0 * hz, 10.0 * hz, s) == (1.0, 10.0)

    def test_same_units(self):
        """Tests for when filter units are already those selected."""
        s = get_unit("s")
        assert get_filter_units(1.0 * s, 10.0 * s, s) == (0.1, 1.0)
        assert get_filter_units(None, 10.0 * s, s) == (None, 0.1)
        assert get_filter_units(10.0 * s, None, s) == (0.1, None)

    def test_different_units_raises(self):
        """The units must be the same or it should raise."""
        s, hz = get_unit("s"), get_unit("Hz")

        with pytest.raises(UnitError):
            get_filter_units(1.0, 10.0 * s, s)

        with pytest.raises(UnitError):
            get_filter_units(1.0 * s, 10.0 * hz, s)

    def test_incompatible_units_raise(self):
        """The units must be the same or it should raise."""
        s, m = get_unit("s"), get_unit("m")
        match = "Cannot convert from"
        with pytest.raises(UnitError, match=match):
            get_filter_units(1.0 * s, 10.0 * s, m)

        with pytest.raises(UnitError, match=match):
            get_filter_units(1.0 * m, 10.0 * m, s)

    def test_specifying_units_unitless_dimension_raises(self):
        """Check an error is raised when units are used on a unitless dimension."""
        msg = "Cannot use units on dimension"
        m = dc.get_unit("m")
        with pytest.raises(UnitError, match=msg):
            get_filter_units(1 * m, 2 * m, None)


class TestDTypeCompatible:
    """Ensure dtype compatibility check works."""

    quants = ("degC", "m/s", get_quantity("kg"))
    non_dt_dtypes = (np.float_, np.int_, np.float32)

    def test_non_datetime(self):
        """Any non-datetime should be compatible."""
        for quant in self.quants:
            for dtype in self.non_dt_dtypes:
                out = assert_dtype_compatible_with_units(dtype, quant)
                assert isinstance(out, Quantity)

    def test_bad_dim_raises(self):
        """Ensure a bad dimension of quantity raises."""
        for quant in self.quants:
            with pytest.raises(UnitError):
                assert_dtype_compatible_with_units(np.datetime64, quant)
            with pytest.raises(UnitError):
                assert_dtype_compatible_with_units(np.timedelta64, quant)

    def test_non_s_raises(self):
        """Only 's' should work, no other increment of time."""
        match = "only allowable units are s"
        with pytest.raises(UnitError, match=match):
            assert_dtype_compatible_with_units(np.datetime64, "ms")

    def test_s_works(self):
        """Seconds should work fine."""
        out = assert_dtype_compatible_with_units(np.datetime64, "s")
        assert out == get_quantity("s")


class TestConvertUnits:
    """Test suite for converting units."""

    def test_simple(self):
        """Simple units to simple units."""
        out = convert_units(1, "m", "ft")
        assert np.isclose(out, 0.3048)

    def test_temperature(self):
        """Ensure temperature can be converted."""
        out = convert_units(1, "m", "ft")
        assert np.isclose(out, 0.3048)

    def test_convert_offset_units(self):
        """Test simple offset units."""
        array = np.arange(10)
        f_array = array * (9 / 5) + 32.0
        out = convert_units(array, from_units="degC", to_units="degF")
        assert np.allclose(f_array, out)

    def test_convert_offset_units_with_mag(self):
        """Ensure units can be converted/set for offset units when non-1 magnitudes."""
        # One non-1 quantity
        array = np.arange(10)
        f_array = 2 * array * (9 / 5) + 32.0
        out = convert_units(array, from_units="2*degC", to_units="degF")
        assert np.allclose(f_array, out)

    def test_convert_offset_units_multiple_mags(self):
        """Ensure if both units have non-1 offsets conversion still works."""
        # Multiple non-1 quants
        array = np.arange(10)
        f_array = (array * (18 / 5) + 32.0) / 2
        out = convert_units(array, from_units="2*degC", to_units="2*degF")
        assert np.allclose(f_array, out)
        # non equal quants
        f_array = (array * (9 * 2.5 / 5) + 32.0) / 6
        out = convert_units(array, from_units="2.5*degC", to_units="6*degF")
        assert np.allclose(f_array, out)
