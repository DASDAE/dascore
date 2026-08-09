"""Module for testing units."""

from __future__ import annotations

import numpy as np
import pint
import pytest

import dascore as dc
import dascore.units as units_module
from dascore.exceptions import UnitError
from dascore.units import (
    Quantity,
    assert_dtype_compatible_with_units,
    convert_units,
    get_byte_count,
    get_factor_and_unit,
    get_filter_units,
    get_quantity,
    get_quantity_str,
    get_unit,
    invert_quantity,
    is_data_size,
    maybe_convert_percent_to_fraction,
    quant_sequence_to_quant_array,
)
from dascore.utils.time import to_float


class TestUnitInit:
    """Ensure units can be initialized."""

    def test_stale_pint_cache_falls_back(self, monkeypatch):
        """Stale Pint disk cache should not prevent registry creation."""
        from dascore import units  # noqa: PLC0415

        class FakeRegistry:
            pass

        cache_folders = []
        removed_paths = []
        cache_path = "pint-cache"
        fake_registry = FakeRegistry()

        def unit_registry(*args, **kwargs):
            cache_folders.append(kwargs.get("cache_folder"))
            if len(cache_folders) == 1 and kwargs.get("cache_folder") == ":auto:":
                raise FileNotFoundError("stale pint cache")
            return fake_registry

        def rmtree(path, ignore_errors):
            removed_paths.append((path, ignore_errors))

        monkeypatch.setattr(units.pint, "UnitRegistry", unit_registry)
        monkeypatch.setattr(
            units,
            "user_cache_path",
            lambda appname, appauthor: cache_path,
        )
        monkeypatch.setattr(units.shutil, "rmtree", rmtree)

        registry = units._get_unit_registry()

        assert cache_folders == [":auto:", ":auto:"]
        assert removed_paths == [(cache_path, True)]
        assert registry is fake_registry

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

    def test_invert_empty_string(self):
        """An empty unit string has no quantity to invert; return None."""
        assert invert_quantity("") is None


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

    def test_equal_quantities_keep_distinct_strings(self):
        """
        Quantities that compare equal (e.g. 1 m and 100 cm) must still
        return their own string representations. This guards against
        caching results keyed on quantity equality.
        """
        quant_m, quant_cm = get_quantity("1 m"), get_quantity("100 cm")
        assert quant_m == quant_cm  # sanity check: pint treats these as equal
        str_m, str_cm = get_quantity_str(quant_m), get_quantity_str(quant_cm)
        assert str_m != str_cm
        assert "m" in str_m
        assert "c" in str_cm  # cm or centimeter

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
        assert ustr is not None
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
        from dascore.units import Hz, ft, km, m, miles  # noqa: PLC0415

        assert m == get_quantity("m")
        assert ft == get_quantity("ft")
        assert miles == get_quantity("miles")
        assert km == get_quantity("km")
        assert Hz == get_quantity("Hz")

    def test_bad_import_error_msg(self):
        """An import error should be raised if the unit isn't valid."""
        with pytest.raises(ImportError):
            from dascore.utils import bob  # noqa

    def test_empty_name_raises(self):
        """The empty string is the one name get_quantity maps to None."""
        import dascore.units  # noqa: PLC0415

        with pytest.raises(AttributeError):
            getattr(dascore.units, "")


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

    def test_impure_to_unit_raises(self):
        """to_unit names a unit; a scaled quantity has no filter meaning."""
        s = get_unit("s")
        match = "must be a unit of magnitude 1"
        with pytest.raises(UnitError, match=match):
            get_filter_units(1.0 * s, 10.0 * s, 2 * s)

        with pytest.raises(UnitError, match=match):
            get_filter_units(1.0 * s, 10.0 * s, "")

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
        array = get_quantity("m") * np.arange(10)
        out = convert_units(array, to_units="ft")
        np.allclose(array.magnitude, out * 3.28084)


class TestQuantSequenceToQuantArray:
    """Ensure we can convert a quantity sequence to an array."""

    def test_valid_sequence_same_units(self):
        """Test with a valid sequence of quantities with the same units."""
        meter = get_quantity("m")
        sequence = [1 * meter, 2 * meter, 3 * meter]
        result = quant_sequence_to_quant_array(sequence)
        expected = meter * np.array([1, 2, 3])
        np.testing.assert_array_equal(result.magnitude, expected.magnitude)
        assert result.units == expected.units

    def test_valid_sequence_different_units(self):
        """Test sequence of quantities with compatible but different units."""
        m, cm, km = get_quantity("m"), get_quantity("cm"), get_quantity("km")

        sequence = [1 * m, 100 * cm, 0.001 * km]
        result = quant_sequence_to_quant_array(sequence)
        expected = m * np.array([1, 1, 1])
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


class TestMaybeConvertPercentToFraction:
    """Tests for converting percentages to fractions."""

    def test_single_percentage(self):
        """Test converting a single percentage value."""
        result = maybe_convert_percent_to_fraction(get_quantity("50%"))
        assert len(result) == 1
        assert result[0] == 0.5

    def test_list_of_percentages(self):
        """Test converting a list of percentages."""
        result = maybe_convert_percent_to_fraction(
            [get_quantity("25%"), get_quantity("75%"), get_quantity("100%")]
        )
        assert len(result) == 3
        assert result[0] == 0.25
        assert result[1] == 0.75
        assert result[2] == 1.0

    def test_non_percentage_quantity_unchanged(self):
        """Test that non-percentage quantities remain unchanged."""
        meter = get_quantity("10 m")
        hz = get_quantity("5 Hz")
        result = maybe_convert_percent_to_fraction([meter, hz])
        assert len(result) == 2
        assert result[0] == meter
        assert result[1] == hz

    def test_plain_numeric_values(self):
        """Test that plain numeric values without units are unchanged."""
        result = maybe_convert_percent_to_fraction([1, 2.5, 0])
        assert result == [1, 2.5, 0]

    def test_mixed_values(self):
        """Test a mix of percentages, quantities, and plain values."""
        percent_val = get_quantity("50%")
        meter_val = get_quantity("10 m")
        plain_val = 0.5
        result = maybe_convert_percent_to_fraction([percent_val, plain_val, meter_val])
        assert len(result) == 3
        assert result[0] == 0.5  # 50% converted to fraction
        assert result[1] == 0.5  # plain value unchanged
        assert result[2] == meter_val  # quantity unchanged

    def test_zero_percent(self):
        """Test that 0% converts correctly."""
        result = maybe_convert_percent_to_fraction(get_quantity("0%"))
        assert len(result) == 1
        assert result[0] == 0.0

    def test_large_percentage(self):
        """Test that percentages over 100% convert correctly."""
        result = maybe_convert_percent_to_fraction(get_quantity("250%"))
        assert len(result) == 1
        assert result[0] == 2.5

    def test_fractional_percentage(self):
        """Test that fractional percentages convert correctly."""
        result = maybe_convert_percent_to_fraction(get_quantity("12.5%"))
        assert len(result) == 1
        assert np.isclose(result[0], 0.125)


class TestUnitConcurrency:
    """The pint registry must initialize once and parse safely in threads."""

    @pytest.mark.concurrency
    def test_registry_created_once(self, monkeypatch, run_in_threads):
        """Racing threads all get the same registry instance."""
        # The registry is process-wide; restore it so quantities created by
        # other tests keep belonging to the active registry.
        monkeypatch.setattr(units_module, "_UNIT_REGISTRY", None)
        original = pint.get_application_registry().get()
        try:
            results = run_in_threads(lambda _: units_module.get_registry())
        finally:
            pint.set_application_registry(original)
        assert len({id(x) for x in results}) == 1

    @pytest.mark.concurrency
    def test_concurrent_parsing(self, run_in_threads):
        """Parsing distinct quantities in threads returns correct values."""
        strings = ["m/s", "1/s", "furlong/fortnight", "strain"]
        results = run_in_threads(lambda index: get_quantity(strings[index]))
        assert results == [get_quantity(x) for x in strings]

    def test_fork_handler_replaces_held_lock(self):
        """A lock held at fork time is replaced so the child cannot deadlock."""
        old_lock = units_module._UNIT_LOCK
        try:
            with old_lock:
                units_module._reinit_unit_lock()
                new_lock = units_module._UNIT_LOCK
                # The replacement is free even while the old lock is held.
                assert new_lock.acquire(blocking=False)
                new_lock.release()
            assert new_lock is not old_lock
        finally:
            units_module._UNIT_LOCK = old_lock


class TestDataSize:
    """Tests for identifying and measuring data size quantities."""

    sizes = ("1 byte", "1 bit", "25 MB", "3 kB", "1 MiB", "2 GiB")
    not_sizes = ("1 m", "10 s", "50%", "1 strain", "1 dimensionless")

    @pytest.mark.parametrize("value", sizes)
    def test_sizes_detected(self, value):
        """Quantities of information are data sizes."""
        assert is_data_size(get_quantity(value))

    @pytest.mark.parametrize("value", not_sizes)
    def test_non_sizes_rejected(self, value):
        """Percents and dimensionless quantities are not data sizes."""
        assert not is_data_size(get_quantity(value))

    @pytest.mark.parametrize("value", (25, 1.0, None, "25 MB"))
    def test_non_quantities_rejected(self, value):
        """Only quantities can be data sizes."""
        assert not is_data_size(value)

    def test_millibarn_is_not_megabytes(self):
        """`mb` is millibarn (an area) in pint; only `MB` is megabytes."""
        assert not is_data_size(dc.units.mb)
        assert is_data_size(dc.units.MB)

    def test_undefined_byte_alias_raises(self):
        """`KB` is not a pint unit; the kilobyte spelling is `kB`."""
        with pytest.raises(pint.UndefinedUnitError):
            dc.units.KB

    @pytest.mark.parametrize(
        "value,expected",
        (
            ("25 MB", 25_000_000),
            ("1 MiB", 1_048_576),
            ("1 kB", 1_000),
            ("8 bit", 1),
            ("1 byte", 1),
        ),
    )
    def test_byte_count(self, value, expected):
        """Byte counts follow pint's decimal/binary prefixes."""
        assert get_byte_count(get_quantity(value)) == expected

    @pytest.mark.parametrize("value", not_sizes)
    def test_byte_count_requires_size(self, value):
        """Non-sizes cannot be measured in bytes."""
        with pytest.raises(UnitError, match="data size"):
            get_byte_count(get_quantity(value))

    def test_byte_count_is_not_to_float(self):
        """
        Guard the bits trap.

        pint converts a dimensionless quantity to base units, and
        information's base unit is the bit, so routing a size through
        `float()` makes it eight times too large. Assert the byte count
        directly rather than the wrong value, so this holds however
        `to_float` treats quantities.
        """
        quant = get_quantity("25 MB")
        assert get_byte_count(quant) == 25_000_000
        # the trap: what a bare float() conversion would have produced
        assert quant.to_base_units().magnitude == 8 * 25_000_000
        # to_float is not a byte converter; it either raises or answers
        # in seconds, but must never be mistaken for get_byte_count
        try:
            assert to_float(quant) != get_byte_count(quant)
        except UnitError:
            pass
