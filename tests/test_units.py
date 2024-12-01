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
    quant_sequence_to_quant_array,
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
        # with magnitude, it should be included.
        quant = get_quantity("10 m /s")
        out = get_quantity_str(quant)
        assert "10.0" in out

    def test_timedelta_to_quantity(self):
        """Ensure a timedelta can be converted to a quantity."""
        dt = dc.to_timedelta64(20)
        quant = dc.get_quantity(dt)
        assert quant == (20 * dc.get_unit("s"))

    def test_datetime_to_quantity(self):
        """Ensure a datetime can be converted to a quantity."""
        td = dc.to_datetime64("1970-01-01T00:00:20")
        quant = dc.get_quantity(td)
        assert quant == (20 * dc.get_unit("s"))


class TestUnitAndFactor:
    """tests for returning units and scaling factor."""

    def test_quantx_units(self):
        """Tests for the quantx unit str."""
        mag, ustr = get_factor_and_unit("rad * 2pi/2^16")
        # sometimes it is "rad * π" other times "π * rad", so just use set.
        assert set(ustr) == set("rad * π")
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

    def test_timedelta64(self):
        """Ensure timedeltas can be separated."""
        td = dc.to_timedelta64(20)
        (factor, unit) = get_factor_and_unit(td)
        assert factor == 20.00
        assert unit == "s"

    def test_datetime64(self):
        """Ensure datetime64 can be separated."""
        td = dc.to_datetime64(20)
        (factor, unit) = get_factor_and_unit(td)
        assert factor == 20.00
        assert unit == "s"


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

    def test_timedelta64(self):
        """Ensure time deltas can be converted to quantity"""
        quant = get_quantity(dc.to_timedelta64(20))
        assert quant == (20 * dc.get_unit("s"))

    def test_datetime64(self):
        """Ensure time deltas can be converted to quantity"""
        quant = get_quantity(dc.to_datetime64(20))
        assert quant == (20 * dc.get_unit("s"))


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
    non_dt_dtypes = (np.float64, np.int_, np.float32)

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

    def test_not_output_units_raises(self):
        """Ensure an error is raised if output units are None."""
        msg = "are not specified"
        with pytest.raises(UnitError, match=msg):
            convert_units(1, from_units="m", to_units=None)

    def test_array_quantity(self):
        """Test that an array quantity works."""
        array = np.arange(10) * get_quantity("m")
        out = convert_units(array, to_units="ft")
        np.allclose(array.magnitude, out * 3.28084)


class TestQuantSequenceToQuantArray:
    """Ensure we can convert a quantity sequence to an array."""

    def test_valid_sequence_same_units(self):
        """Test with a valid sequence of quantities with the same units."""
        meter = get_quantity("m")
        sequence = [1 * meter, 2 * meter, 3 * meter]
        result = quant_sequence_to_quant_array(sequence)
        expected = np.array([1, 2, 3]) * meter
        np.testing.assert_array_equal(result.magnitude, expected.magnitude)
        assert result.units == expected.units

    def test_valid_sequence_different_units(self):
        """Test sequence of quantities with compatible but different units."""
        m, cm, km = get_quantity("m"), get_quantity("cm"), get_quantity("km")

        sequence = [1 * m, 100 * cm, 0.001 * km]
        result = quant_sequence_to_quant_array(sequence)
        expected = np.array([1, 1, 1]) * m
        assert np.allclose(result.magnitude, expected.magnitude)
        assert result.units == expected.units

    def test_incompatible_units(self):
        """Test with a sequence of quantities with incompatible units."""
        sequence = [1 * get_quantity("m"), 1 * get_quantity("s")]
        msg = "Not all values in sequence have compatible units."
        with pytest.raises(UnitError, match=msg):
            quant_sequence_to_quant_array(sequence)

    def test_non_quantity_elements(self):
        """Test with a sequence containing non-quantity elements."""
        sequence = [1 * get_quantity("m"), 5]
        msg = "Not all values in sequence are quantities."
        with pytest.raises(UnitError, match=msg):
            quant_sequence_to_quant_array(sequence)

    def test_empty_sequence(self):
        """Test with an empty sequence."""
        sequence = []
        out = quant_sequence_to_quant_array(sequence)
        assert isinstance(out, Quantity)

    def test_numpy_array_input(self):
        """Test with a numpy array input."""
        sequence = np.array([1, 2, 3])
        out = quant_sequence_to_quant_array(sequence)
        assert isinstance(out, Quantity)
