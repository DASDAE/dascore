"""
Tests for patch ufuncs.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager

import numpy as np
import pytest
from pint import DimensionalityError

import dascore as dc
import dascore.proc.coords
import dascore.utils.array as array_utils
from dascore import get_quantity
from dascore.exceptions import (
    IncompatiblePatchError,
    ParameterError,
    PatchCoordinateError,
    UnitError,
)
from dascore.units import furlongs, m, s
from dascore.utils.array import (
    UFUNC_NAMES,
    PatchUFunc,
    _BoundPatchUFunc,
    _is_offset_unit,
    apply_array_func,
    apply_ufunc,
    convert_bytes_to_strings,
    convert_strings_to_bytes,
    hash_array,
    is_string_byte_serializable_array,
)
from dascore.utils.array_api import array_namespace, backend_name
from dascore.utils.misc import suppress_warnings
from dascore.warnings import NumpyFallbackWarning


def to_backend_array(patch, array):
    """Return the array on the same backend as the patch's data."""
    return array_namespace(patch.data).asarray(array)


class _OtherBackendArray:
    """An array which claims to belong to a different array API backend."""

    def __init__(self, array):
        self._array = array
        self.shape = array.shape
        self.dtype = array.dtype

    def __array_namespace__(self, api_version=None):
        """Return a namespace which is not the one under test."""
        return np


@contextmanager
def warnings_as_errors():
    """A context manager which raises rather than warns."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield


class TestApplyUfunc:
    """Tests for applying various ufunc-type operators."""

    @pytest.mark.parametrize("func", (np.abs, np.tan, np.isfinite, np.exp))
    def test_unary_ufuncs(self, func, random_patch):
        """Ensure ufuncs that take a single input also work."""
        out = func(random_patch)
        assert isinstance(out, dc.Patch)
        if func is np.isfinite:
            assert out.data.dtype == np.bool_

    def test_scalar(self, random_patch):
        """Test for a single scalar."""
        new = apply_ufunc(np.multiply, random_patch, 10)
        assert np.allclose(new.data, random_patch.data * 10)

    def test_array_like(self, random_patch):
        """Ensure array-like operations work."""
        ones = np.ones(random_patch.shape)
        new = apply_ufunc(np.add, random_patch, ones)
        assert np.allclose(new.data, ones + random_patch.data)

    def test_reversed_scalar(self, random_patch):
        """Ensure reversed scalar works on patch."""
        out = 10 + random_patch  # np.add with Patch on RHS
        assert isinstance(out, dc.Patch)
        assert np.allclose(out.data, random_patch.data + 10)

    def test_reversed_array_like(self, random_patch):
        """Test reversed array works on patch."""
        ones = np.ones(random_patch.shape)
        out = ones * random_patch
        assert np.allclose(out.data, ones * random_patch.data)

    @pytest.mark.xfail(raises=(UnitError, DimensionalityError))
    def test_reversed_unit_and_quantity(self, random_patch):
        """
        Ensure reversed quantity works on patch.

        Currently, there is no way to make this pass without relying on internal
        Pint implementation details. This is because pint first handles the
        operation but doesn't know how to treat a Patch.
        """
        pa = random_patch.set_units("m/s")
        out1 = (m / s) + pa  # unit on LHS
        out2 = (10 * m / s) + pa  # quantity on LHS
        assert np.allclose(out1.data, random_patch.data + 1)
        assert np.allclose(out2.data, random_patch.data + 10)

    def test_disjoint_coords_raise(self, random_patch):
        """Un-alignable coords are a conflict, not an empty answer."""
        time = random_patch.get_coord("time")
        new_time = time.max() + time.step
        new = random_patch.update_coords(time_min=new_time)
        with pytest.raises(PatchCoordinateError, match="share no values"):
            apply_ufunc(np.multiply, new, random_patch)

    def test_quantity_scalar(self, random_patch):
        """Ensure operators work with quantities."""
        patch = random_patch.set_units("m/s")
        other = 10 * m / s
        # first try multiply
        new = apply_ufunc(np.multiply, patch, other)
        new_units = get_quantity("m/s") * get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert isinstance(new.data, np.ndarray)
        # try add
        new = apply_ufunc(np.add, patch, other)
        new_units = get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert isinstance(new.data, np.ndarray)
        # and divide
        new = apply_ufunc(np.divide, patch, other)
        # Dimensionless result: units may be omitted; accept None or "1".
        q = get_quantity(new.attrs.data_units)
        assert q is None or q == get_quantity("1")
        assert isinstance(new.data, np.ndarray)

    def test_unit(self, random_patch):
        """Units should only affect the unit attr."""
        patch = random_patch.set_units("m/s")
        other = m / s
        # first try multiply
        new = apply_ufunc(np.multiply, patch, other)
        new_units = get_quantity("m/s") * get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert np.allclose(new.data, random_patch.data)
        # try add
        new = apply_ufunc(np.add, patch, other)
        new_units = get_quantity("m/s")
        assert get_quantity(new.attrs.data_units) == new_units
        assert np.allclose(new.data, random_patch.data + 1)
        # and divide
        new = apply_ufunc(np.divide, patch, other)
        new_units = get_quantity("m/s") / get_quantity("m/s")
        assert new.attrs.data_units is None or new.attrs.data_units == new_units
        assert np.allclose(new.data, random_patch.data)

    def test_patch_with_units(self, random_patch):
        """Ensure when patch units are set they are applied as well."""
        # test add
        pa1 = random_patch.set_units("m/s")
        out1 = apply_ufunc(np.add, pa1, pa1)
        assert get_quantity(out1.attrs.data_units) == get_quantity("m/s")
        # test multiply
        out2 = apply_ufunc(
            np.multiply,
            pa1,
            pa1,
        )
        assert get_quantity(out2.attrs.data_units) == get_quantity("m**2/s**2")

    def test_array_with_units(self, random_patch):
        """Ensure an array with units works for multiplication."""
        patch1 = random_patch
        ones = np.ones(patch1.shape) * furlongs
        out1 = patch1 * ones
        assert get_quantity(out1.attrs.data_units) == get_quantity("furlongs")
        # test division with units
        patch2 = random_patch.set_units("m")
        out2 = patch2 / ones
        expected = get_quantity("m/furlongs")
        assert get_quantity(out2.attrs.data_units) == expected

    def test_incompatible_units(self, random_patch):
        """Ensure incompatible units raise."""
        pa1 = random_patch.set_units("m/s")
        other = 10 * get_quantity("m")
        with pytest.raises(UnitError):
            apply_ufunc(np.add, pa1, other)

    def test_different_kind_raises(self, random_patch):
        """Patches of different kinds are never operated together."""
        other = random_patch.update_attrs(tag="other")
        with pytest.raises(IncompatiblePatchError, match="not the same kind"):
            apply_ufunc(np.add, random_patch, other)

    def test_units_take_part_rather_than_gate(self, random_patch):
        """Differing data units go through the operator, not a kind check."""
        pa1 = random_patch.set_units("m")
        pa2 = random_patch.set_units("s")
        out = apply_ufunc(np.multiply, pa1, pa2)
        assert get_quantity(out.attrs.data_units) == get_quantity("m * s")
        with pytest.raises(UnitError):
            apply_ufunc(np.add, pa1, pa2)
        # Convertible units are converted to the first patch's.
        pa3 = random_patch.set_units("km")
        out = apply_ufunc(np.add, pa1, pa3)
        assert get_quantity(out.attrs.data_units) == get_quantity("m")
        assert np.allclose(out.data, pa1.data + 1000 * pa3.data)

    @pytest.mark.parametrize("kind", ["patch", "array", "scalar"])
    def test_missing_units_conflict_with_nothing(self, random_patch, kind):
        """A unitless operand: dimensionless for products, adopts for sums."""
        metres = random_patch.set_units("m")
        bare = {
            "patch": random_patch.set_units(None),
            "array": np.ones(random_patch.shape),
            "scalar": 2,
        }[kind]
        m, per_m = get_quantity("m"), get_quantity("1/m")
        assert get_quantity((metres + bare).attrs.data_units) == m
        assert get_quantity((bare + metres).attrs.data_units) == m
        assert get_quantity((metres * bare).attrs.data_units) == m
        assert get_quantity((metres / bare).attrs.data_units) == m
        assert get_quantity((bare / metres).attrs.data_units) == per_m
        # Comparisons need equal units too, so they adopt rather than raise.
        assert (metres > bare).data.dtype == np.bool_

    def test_ufuncs_outside_the_unit_registry(self, random_patch):
        """A ufunc the registry lacks still works; units are left as they were."""
        ints = random_patch.new(data=(random_patch.data * 100).astype(np.int64))
        metres = ints.set_units("m")
        out = apply_ufunc(np.bitwise_and, metres, 3)
        assert get_quantity(out.attrs.data_units) == get_quantity("m")
        assert np.array_equal(out.data, ints.data & 3)
        out = apply_ufunc(np.logical_and, random_patch.set_units("m"), True)
        assert out.data.dtype == np.bool_
        # Two unitful operands take the same fallback, other in patch units.
        out = apply_ufunc(np.bitwise_and, metres, metres)
        assert get_quantity(out.attrs.data_units) == get_quantity("m")
        assert np.array_equal(out.data, ints.data & ints.data)
        with pytest.raises(UnitError):
            apply_ufunc(np.bitwise_and, metres, ints.set_units("s"))
        # convertible units are converted into the patch's for the fallback
        out = apply_ufunc(np.logical_and, metres, ints.set_units("km"))
        assert out.data.dtype == np.bool_

    def test_scaled_units_keep_their_scale(self, random_patch):
        """Data in "100 cm" stay as they are; the scale rides on the units."""
        scaled = random_patch.set_units("100 cm")
        out = scaled * 2
        assert np.allclose(out.data, 2 * random_patch.data)
        assert get_quantity(out.attrs.data_units) == get_quantity("100 cm")
        out = scaled - 1
        assert np.allclose(out.data, random_patch.data - 1)
        assert get_quantity(out.attrs.data_units) == get_quantity("100 cm")
        out = scaled**2
        assert get_quantity(out.attrs.data_units) == get_quantity("10000 cm**2")

    def test_unit_and_unit_string_operands(self, random_patch):
        """A Unit or a unit string names units, unlike a bare number."""
        patch = random_patch.set_units("m")
        assert get_quantity((patch * m.units).attrs.data_units) == get_quantity("m**2")
        assert get_quantity((patch / "s").attrs.data_units) == get_quantity("m/s")

    def test_no_units_fit_raises(self, random_patch):
        """An operation no assignment of units can satisfy raises UnitError."""
        patch = random_patch.set_units("m")
        # dimensionless ** metres and metres ** metres both fail
        with pytest.raises(UnitError, match="failed with units"):
            apply_ufunc(np.power, 2.0, patch)

    def test_adopting_scaled_units_is_symmetric(self, random_patch):
        """A unitless patch adopts "100 cm" whole, whichever side it is on."""
        bare = random_patch.set_units(None)
        scaled = random_patch.set_units("100 cm")
        expected = 2 * random_patch.data
        for out in (bare + scaled, scaled + bare):
            assert np.allclose(out.data, expected)
            assert get_quantity(out.attrs.data_units) == get_quantity("100 cm")
        out = bare / scaled
        units = get_quantity(out.attrs.data_units)
        assert units.units == get_quantity("1 / cm").units
        assert np.isclose(units.magnitude, 0.01)

    def test_multiply_by_zero_keeps_units(self, random_patch):
        """A result of all zeros is still in the patch's units."""
        metres = random_patch.set_units("m")
        out = metres * 0
        assert np.all(out.data == 0)
        assert get_quantity(out.attrs.data_units) == get_quantity("m")
        out = metres.set_units("100 cm") - metres.data
        assert get_quantity(out.attrs.data_units) == get_quantity("100 cm")

    def test_offset_units_are_kept(self, random_patch):
        """Temperatures are not scaled by a probe: degC stays degC."""
        temp = random_patch.set_units("degC")
        out = temp + 1
        assert np.allclose(out.data, random_patch.data + 1)
        assert get_quantity(out.attrs.data_units) == get_quantity("degC")
        assert (temp > 20).attrs.data_units is None
        out = random_patch.set_units(None) + temp
        assert get_quantity(out.attrs.data_units) == get_quantity("degC")
        # a reciprocal or a power of a temperature has no offset unit
        for bad in (lambda: 1 / temp, lambda: temp**2, lambda: temp * 2):
            with pytest.raises(UnitError, match="offset units"):
                bad()
        # degrees may be taken from a temperature, not a temperature from a number
        assert get_quantity((temp - 1).attrs.data_units) == get_quantity("degC")
        with pytest.raises(UnitError, match="Cannot subtract a temperature"):
            1 - temp
        with pytest.raises(UnitError, match="Cannot subtract a temperature"):
            random_patch.set_units(None) - temp

    def test_offset_unit_detection(self):
        """Offset units are told apart by behaviour, not a registry attribute."""
        assert _is_offset_unit(get_quantity("degC"))
        assert _is_offset_unit(get_quantity("degF"))
        assert not _is_offset_unit(get_quantity("kelvin"))
        assert not _is_offset_unit(get_quantity("100 cm"))

    def test_empty_unit_string_raises(self, random_patch):
        """A string operand must name units."""
        with pytest.raises(UnitError, match="names no units"):
            apply_ufunc(np.multiply, random_patch, "")

    def test_two_offset_unit_patches(self, random_patch):
        """Two temperatures differ by a delta; their sum has no meaning."""
        warm = random_patch.new(data=np.full(random_patch.shape, 20.0)).set_units(
            "degC"
        )
        cool = random_patch.new(data=np.full(random_patch.shape, 5.0)).set_units("degC")
        out = warm - cool
        assert np.allclose(out.data, 15.0)
        assert get_quantity(out.attrs.data_units) == get_quantity("delta_degC")
        with pytest.raises(UnitError, match="offset units"):
            warm + cool
        assert np.all((warm > cool).data)
        hottest = np.maximum(warm, cool)
        assert get_quantity(hottest.attrs.data_units) == get_quantity("degC")
        with pytest.raises(UnitError, match="failed with units"):
            warm - random_patch.set_units("m")

    def test_temperature_and_difference(self, random_patch):
        """A temperature plus or minus a difference is a temperature."""
        shape = random_patch.shape
        warm = random_patch.new(data=np.full(shape, 20.0)).set_units("degC")
        step = random_patch.new(data=np.full(shape, 5.0)).set_units("delta_degC")
        for out in (warm + step, step + warm):
            assert np.allclose(out.data, 25.0)
            assert get_quantity(out.attrs.data_units) == get_quantity("degC")
        out = warm - step
        assert np.allclose(out.data, 15.0)
        assert get_quantity(out.attrs.data_units) == get_quantity("degC")
        # a kelvin difference converts; a difference minus a temperature does not exist
        kelvin_step = step.set_units("kelvin")
        assert np.allclose((warm + kelvin_step).data, 25.0)
        with pytest.raises(UnitError, match="offset units"):
            step - warm
        with pytest.raises(UnitError, match="offset units"):
            warm * step
        with pytest.raises(UnitError, match="failed with units"):
            warm + random_patch.set_units("m")

    def test_generalized_ufunc_with_units(self):
        """A gufunc such as matmul cannot be probed on scalars; numpy runs it."""
        square = dc.get_example_patch(shape=(10, 10)).set_units("m")
        out = np.matmul(square, np.eye(10))
        assert np.allclose(out.data, square.data)
        assert get_quantity(out.attrs.data_units) == get_quantity("m")
        # the units are kept from whichever side had them
        bare = square.set_units(None)
        out = np.matmul(bare, square)
        assert get_quantity(out.attrs.data_units) == get_quantity("m")

    def test_scalar_units_need_no_array_wrapping(self, random_patch):
        """Scalars settle units on a probe; comparisons and powers behave."""
        metres = random_patch.set_units("m")
        assert (metres > 5).data.dtype == np.bool_
        assert get_quantity((metres**2).attrs.data_units) == get_quantity("m**2")
        with pytest.raises(UnitError, match="scalar exponent"):
            metres ** np.ones(metres.shape)

    def test_other_attrs_keep_first(self, random_patch):
        """Attrs outside the kind keep the first patch's value."""
        pa1 = random_patch.update_attrs(foo="a")
        pa2 = random_patch.update_attrs(foo="b")
        assert apply_ufunc(np.add, pa1, pa2).attrs.foo == "a"
        assert apply_ufunc(np.add, pa2, pa1).attrs.foo == "b"

    def test_patches_non_coords_len_1(self, random_patch):
        """Ensure patches with non-coords also work."""
        mean_patch = random_patch.mean("distance")
        out = mean_patch / mean_patch
        assert np.allclose(out.data, 1)

    def test_patches_non_coords_different_len(self, random_patch):
        """Ensure patches with non-coords of different lengths work."""
        patch_1 = random_patch.mean("distance")
        dist_ind = random_patch.get_axis("distance")
        old_shape = list(patch_1.shape)
        old_shape[dist_ind] = old_shape[dist_ind] + 2
        patch_2 = patch_1.make_broadcastable_to(tuple(old_shape))
        out = patch_1 / patch_2
        assert np.allclose(out.data, 1)
        assert out.shape == patch_2.shape

    def test_non_dim_coords(self, random_dft_patch):
        """Ensure ufuncs can still be applied to coords with non dim coords."""
        out = random_dft_patch * random_dft_patch
        out_coord_keys = set(out.coords.coord_map.keys())
        input_coord_keys = set(random_dft_patch.coords.coord_map.keys())
        assert out_coord_keys == input_coord_keys
        assert set(out.coords.coord_map) == set(random_dft_patch.coords.coord_map)

    @pytest.mark.parametrize(
        "op, other, expected_units, expected_data",
        [
            (np.multiply, 10 * m / s, "m**2/s**2", lambda d: d * 10),
            (np.add, 10 * m / s, "m/s", lambda d: d + 10),
            (np.divide, 10 * m / s, "1", lambda d: d / 10.0),
        ],
    )
    def test_quantity_ops_param(
        self, random_patch, op, other, expected_units, expected_data
    ):
        """Run several tests for quantities in various operations."""
        pa = random_patch.set_units("m/s")
        out = apply_ufunc(op, pa, other)
        quant = get_quantity(out.attrs.data_units)
        none_or_1 = quant is None and expected_units == "1"
        assert none_or_1 or quant == get_quantity(expected_units)
        assert np.allclose(out.data, expected_data(random_patch.data))

    def test_unsupported_raises(self, random_patch):
        """
        When ufuncs don't have the right number of input/output an error
        should be raised.
        """
        msg = "ufuncs with input/output"
        with pytest.raises(ParameterError, match=msg):
            apply_ufunc(np.frexp, random_patch)

    def test_apply_reduction(self, random_patch):
        """Ensure reductions also work."""
        out = np.multiply.reduce(random_patch, axis=1)
        out2 = np.multiply.reduce(random_patch, 1)
        assert isinstance(out, dc.Patch)
        assert out.shape[1] == 1
        assert out2.equals(out)


class TestPatchUFunc:
    """Tests for PatchUFunc class."""

    def test_basic_ufunc_usage(self, random_patch):
        """Test basic ufunc usage (patch + patch) from docstring example."""
        ufunc = PatchUFunc(np.add)
        result = ufunc(random_patch, random_patch)

        assert isinstance(result, dc.Patch)
        assert np.allclose(result.data, random_patch.data + random_patch.data)
        assert result.coords.equals(random_patch.coords)

    def test_accumulate_method(self, random_patch):
        """Test accumulate method with dimensions from docstring example."""
        ufunc = PatchUFunc(np.add)

        # Test accumulate along time dimension
        result = ufunc.accumulate(random_patch, dim="time")

        assert isinstance(result, dc.Patch)
        # Check that shape is preserved for accumulate
        assert result.shape == random_patch.shape
        # Verify it's actually doing cumulative sum
        axis = random_patch.get_axis("time")
        expected = np.cumsum(random_patch.data, axis=axis)
        assert np.allclose(result.data, expected)

    def test_reduce_method(self, random_patch):
        """Test reduce method with dimensions from docstring example."""
        ufunc = PatchUFunc(np.add)

        # Test reduce along distance dimension
        result = ufunc.reduce(random_patch, dim="distance")

        assert isinstance(result, dc.Patch)
        # Check that the distance dimension is reduced to size 1
        dist_axis = random_patch.get_axis("distance")
        expected_shape = list(random_patch.shape)
        expected_shape[dist_axis] = 1
        assert result.shape == tuple(expected_shape)


class TestStringArrayHelpers:
    """Tests for generic string-array serialization helpers."""

    def test_string_byte_serializable_array_policy(self):
        """Only true string-like arrays should take the byte-serialization path."""
        assert is_string_byte_serializable_array(np.array(["alpha"]))
        assert is_string_byte_serializable_array(np.array([b"alpha"], dtype="S5"))
        assert is_string_byte_serializable_array(np.array(["alpha"], dtype=object))
        assert not is_string_byte_serializable_array(np.array([1, 2], dtype=object))

    def test_convert_strings_to_bytes(self):
        """Unicode strings should convert to fixed-width bytes."""
        out = convert_strings_to_bytes(np.array(["alpha", "cafe", "北京"]))
        assert out.dtype.kind == "S"

    def test_convert_empty_strings_to_bytes(self):
        """Empty string arrays should still have a concrete byte dtype."""
        out = convert_strings_to_bytes(np.array([], dtype="U4"))
        assert out.dtype == np.dtype("S1")
        assert out.size == 0

    def test_convert_shaped_empty_strings_to_bytes_preserves_shape(self):
        """Shaped empty inputs should keep their original array shape."""
        data = np.empty((0, 3), dtype="U4")
        out = convert_strings_to_bytes(data)
        assert out.shape == data.shape
        assert out.dtype == np.dtype("S1")

    def test_convert_bytes_like_strings_to_bytes(self):
        """Existing byte-like entries should round-trip without repr mangling."""
        data = np.array(
            [b"alpha", np.bytes_(b"beta"), bytearray(b"gamma")], dtype=object
        )
        out = convert_strings_to_bytes(data)
        assert np.array_equal(out, np.array([b"alpha", b"beta", b"gamma"]))

    def test_convert_bytes_to_strings(self):
        """UTF-8 bytes should convert back to unicode strings."""
        data = np.array([b"alpha", "北京".encode()], dtype="S6")
        out = convert_bytes_to_strings(data, "<U8")
        assert out.dtype.kind == "U"
        assert np.array_equal(out, np.array(["alpha", "北京"]))

    def test_convert_empty_bytes_to_strings(self):
        """Empty byte arrays should decode to an empty unicode array."""
        out = convert_bytes_to_strings(np.array([], dtype="S1"), original_dtype=object)
        assert out.dtype == np.dtype("U1")
        assert out.size == 0

    def test_convert_shaped_empty_bytes_to_strings_preserves_shape(self):
        """Shaped empty byte arrays should keep their original array shape."""
        data = np.empty((0, 3), dtype="S1")
        out = convert_bytes_to_strings(data, original_dtype="<U4")
        assert out.shape == data.shape
        assert out.dtype == np.dtype("<U4")

    def test_convert_empty_bytes_to_unicode_strings(self):
        """Empty byte arrays should preserve explicit unicode dtypes."""
        out = convert_bytes_to_strings(np.array([], dtype="S1"), original_dtype="<U4")
        assert out.dtype == np.dtype("<U4")
        assert out.size == 0

    def test_convert_empty_bytes_to_fixed_width_bytes(self):
        """Empty byte arrays should preserve explicit byte dtypes."""
        out = convert_bytes_to_strings(np.array([], dtype="S1"), original_dtype="|S4")
        assert out.dtype == np.dtype("|S4")
        assert out.size == 0

    def test_convert_bytes_to_object_strings(self):
        """Object-backed string arrays should restore object dtype."""
        data = np.array([b"alpha", b"beta"], dtype="S5")
        out = convert_bytes_to_strings(data, "object")
        assert out.dtype == object
        assert np.array_equal(out, np.array(["alpha", "beta"], dtype=object))

    def test_convert_bytes_to_fixed_width_bytes(self):
        """Fixed-width byte dtypes should be preserved without decoding."""
        data = np.array([b"alpha", b"beta"], dtype="S5")
        out = convert_bytes_to_strings(data, "|S5")
        assert out.dtype == np.dtype("|S5")
        assert np.array_equal(out, data)

    def test_convert_bytes_to_default_unicode_strings(self):
        """Unknown source dtypes should fall back to unicode arrays."""
        data = np.array([b"alpha", b"beta"], dtype="S5")
        out = convert_bytes_to_strings(data)
        assert out.dtype.kind == "U"
        assert np.array_equal(out, np.array(["alpha", "beta"]))

    def test_introspection(self):
        """Test that generated ufunc has proper introspection."""
        ufunc = PatchUFunc(np.add)

        assert hasattr(ufunc, "__name__")
        assert ufunc.__name__ == "add"
        assert hasattr(ufunc, "__doc__")
        assert ufunc.__doc__ is not None

    def test_method_binding(self, random_patch):
        """Test that generated ufunc can be bound as a method."""
        ufunc = PatchUFunc(np.multiply)

        # Test descriptor protocol works
        bound_ufunc = ufunc.__get__(random_patch, type(random_patch))
        result = bound_ufunc(random_patch)

        assert isinstance(result, dc.Patch)
        assert np.allclose(result.data, random_patch.data * random_patch.data)

        # accumulation and reduction should also work.
        pa1 = bound_ufunc.reduce(dim="time")
        pa2 = ufunc.reduce(random_patch, dim="time")
        assert pa1.equals(pa2)

        pa1 = bound_ufunc.accumulate(dim="time")
        pa2 = ufunc.accumulate(random_patch, dim="time")
        assert pa1.equals(pa2)

    def test_different_ufuncs(self, random_patch):
        """Test PatchUFunc works with different numpy ufuncs."""
        # Test with multiply
        mul_ufunc = PatchUFunc(np.multiply)
        mul_result = mul_ufunc(random_patch, 2.0)
        assert np.allclose(mul_result.data, random_patch.data * 2.0)

        # Test with subtract
        sub_ufunc = PatchUFunc(np.subtract)
        sub_result = sub_ufunc(random_patch, random_patch)
        assert np.allclose(sub_result.data, 0.0)

    def test_bound_calls(self, random_patch):
        """Test bound method calls using descriptor protocol."""
        ufunc = PatchUFunc(np.multiply)

        # Test bound call via __get__
        bound_ufunc = ufunc.__get__(random_patch, type(random_patch))
        result = bound_ufunc(2.0)
        assert isinstance(result, dc.Patch)
        assert np.allclose(result.data, random_patch.data * 2.0)

        # Test bound reduce call
        bound_result = bound_ufunc.reduce("time")
        unbound_result = ufunc.reduce(random_patch, "time")
        assert bound_result.equals(unbound_result)

        # Test bound accumulate call
        bound_result = bound_ufunc.accumulate("distance")
        unbound_result = ufunc.accumulate(random_patch, "distance")
        assert bound_result.equals(unbound_result)

    def test_unary_ufunc(self, random_patch):
        """Test PatchUFunc with unary ufuncs."""
        ufunc = PatchUFunc(np.abs)
        result = ufunc(random_patch)

        assert isinstance(result, dc.Patch)
        assert np.allclose(result.data, np.abs(random_patch.data))
        assert ufunc.__name__ == "absolute"

    def test_reduce_positional_args(self, random_patch):
        """Test reduce method with positional arguments."""
        ufunc = PatchUFunc(np.add)

        # Test with positional dim argument
        result = ufunc.reduce(random_patch, "time")
        expected = ufunc.reduce(random_patch, dim="time")
        assert result.equals(expected)

    def test_accumulate_positional_args(self, random_patch):
        """Test accumulate method with positional arguments."""
        ufunc = PatchUFunc(np.add)

        # Test with positional dim argument
        result = ufunc.accumulate(random_patch, "distance")
        expected = ufunc.accumulate(random_patch, dim="distance")
        assert result.equals(expected)

    def test_reduce_with_none_dim(self, random_patch):
        """Test reduce method with dim=None."""
        ufunc = PatchUFunc(np.add)

        # Test with explicit None dim
        result = ufunc.reduce(random_patch, dim=None)
        assert isinstance(result, dc.Patch)

    def test_accumulate_with_none_dim(self, random_patch):
        """Test accumulate method with dim=None."""
        ufunc = PatchUFunc(np.add)

        # Test with explicit None dim
        result = ufunc.accumulate(random_patch, dim=None)
        assert isinstance(result, dc.Patch)

    def test_bound_reduce_with_none(self, random_patch):
        """Test bound reduce method with None argument."""
        ufunc = PatchUFunc(np.add)
        bound_ufunc = ufunc.__get__(random_patch, type(random_patch))

        # Test bound call with None
        result = bound_ufunc.reduce(None)
        assert isinstance(result, dc.Patch)

    def test_bound_accumulate_with_none(self, random_patch):
        """Test bound accumulate method with None argument."""
        ufunc = PatchUFunc(np.add)
        bound_ufunc = ufunc.__get__(random_patch, type(random_patch))

        # Test bound call with None
        result = bound_ufunc.accumulate(None)
        assert isinstance(result, dc.Patch)

    def test_ufunc_with_no_name_or_doc(self):
        """Test PatchUFunc with ufunc that has no __name__ or __doc__."""

        # Create a mock ufunc-like object without __name__ or __doc__
        class MockUfunc:
            nin = 2
            nout = 1

            def __call__(self, *args, **kwargs):
                return np.add(*args, **kwargs)

        mock_ufunc = MockUfunc()
        ufunc = PatchUFunc(mock_ufunc)

        # Should use defaults
        assert ufunc.__name__ == "patch_ufunc"
        assert ufunc.__doc__ is None

    def test_comprehensive_bound_unbound_equivalence(self, random_patch):
        """Test that bound and unbound calls produce equivalent results."""
        ufunc = PatchUFunc(np.multiply)

        # Create bound version
        bound_ufunc = ufunc.__get__(random_patch, type(random_patch))

        # Test basic call equivalence
        unbound_result = ufunc(random_patch, 3.0)
        bound_result = bound_ufunc(3.0)
        assert bound_result.equals(unbound_result)

        # Test reduce equivalence with positional args
        unbound_reduce = ufunc.reduce(random_patch, "time", dtype=np.float32)
        bound_reduce = bound_ufunc.reduce("time", dtype=np.float32)
        assert bound_reduce.equals(unbound_reduce)

        # Test accumulate equivalence with keyword args
        unbound_accum = ufunc.accumulate(random_patch, dim="distance")
        bound_accum = bound_ufunc.accumulate(dim="distance")
        assert bound_accum.equals(unbound_accum)

    def test_multiple_binding_levels(self, random_patch):
        """Test binding a generated ufunc multiple times."""
        ufunc = PatchUFunc(np.add)

        # Bind once
        bound_once = ufunc.__get__(random_patch, type(random_patch))

        # Verify the bound instance is a _BoundPatchUFunc

        assert isinstance(bound_once, _BoundPatchUFunc)

        # Test that the bound instance works correctly
        result1 = bound_once(random_patch)
        unbound_result = ufunc(random_patch, random_patch)
        assert result1.equals(unbound_result)

    def test_get_with_none_object(self):
        """Test __get__ method with None object returns self."""
        ufunc = PatchUFunc(np.add)
        result = ufunc.__get__(None, None)
        assert result is ufunc

    def test_patch_ufunc_class_properties(self, random_patch):
        """Test _BoundPatchUFunc class has proper attributes."""
        ufunc = PatchUFunc(np.multiply)
        bound_ufunc = ufunc.__get__(random_patch, type(random_patch))

        # Should be instance of _BoundPatchUFunc
        assert isinstance(bound_ufunc, _BoundPatchUFunc)

        # Should have proper attributes
        assert bound_ufunc.__name__ == "multiply"
        assert bound_ufunc.np_ufunc is np.multiply
        assert bound_ufunc.patch is random_patch

    def test_out_parameter_raises(self, random_patch):
        """Since patches are immutable, we cant support out. Raise if provided."""
        match = "cannot be used"
        with pytest.raises(ParameterError, match=match):
            apply_ufunc(np.add, random_patch, random_patch, out=random_patch)


class TestApplyArrayFunc:
    """Tests for apply array func."""

    def test_reduce_signature_fallback(self, monkeypatch, random_patch):
        """Fallback to dummy signatures when ufunc introspection fails."""
        real_signature = array_utils.inspect.signature

        def _signature(obj):
            if getattr(obj, "__name__", None) == "reduce":
                raise TypeError("simulated signature failure")
            return real_signature(obj)

        monkeypatch.setattr(array_utils.inspect, "signature", _signature)
        out = apply_array_func(np.add.reduce, random_patch, axis=1)
        assert isinstance(out, dc.Patch)
        assert out.shape[1] == 1

    def test_accumulate_signature_fallback(self, monkeypatch, random_patch):
        """Fallback to dummy signatures when accumulate introspection fails."""
        real_signature = array_utils.inspect.signature

        def _signature(obj):
            if getattr(obj, "__name__", None) == "accumulate":
                raise ValueError("simulated signature failure")
            return real_signature(obj)

        monkeypatch.setattr(array_utils.inspect, "signature", _signature)
        out = apply_array_func(np.add.accumulate, random_patch, axis=1)
        assert isinstance(out, dc.Patch)
        assert out.shape == random_patch.shape

    def test_unknown_signature_failure_reraises(self, monkeypatch, random_patch):
        """Unknown callables with introspection failure should re-raise."""

        def shape_changing_func(data):
            return data[0]

        shape_changing_func.__name__ = "not_a_ufunc_method"
        real_signature = array_utils.inspect.signature

        def _signature(obj):
            if getattr(obj, "__name__", None) == "not_a_ufunc_method":
                raise TypeError("simulated signature failure")
            return real_signature(obj)

        monkeypatch.setattr(array_utils.inspect, "signature", _signature)
        with pytest.raises(TypeError, match="simulated signature failure"):
            apply_array_func(shape_changing_func, random_patch)

    def test_function_without_axis_parameter_error(self, random_patch):
        """Test that functions without axis parameter that change shape raise error."""

        # Create a mock function that changes shape but has no axis parameter
        def shape_changing_func(data):
            # Return a different shape to trigger the error path
            return np.array([1, 2, 3])  # Always return same small array

        # Remove any axis-related attributes to ensure no axis parameter
        shape_changing_func.__name__ = "test_func"

        # This should trigger the ParameterError
        msg = "result of test_func without an axis parameter"
        with pytest.raises(ParameterError, match=msg):
            apply_array_func(shape_changing_func, random_patch)

    def test_function_without_axis_parameter_same_shape(self, random_patch):
        """Test functions without axis parameter but same shape work fine."""

        # Create a mock function that keeps the same shape
        def same_shape_func(data):
            # Return same shape but modified data
            return data * 2

        same_shape_func.__name__ = "same_shape_test"

        # This should work fine
        result = apply_array_func(same_shape_func, random_patch)

        assert isinstance(result, dc.Patch)
        assert result.shape == random_patch.shape
        assert np.allclose(result.data, random_patch.data * 2)

    def test_no_axis_signature_same_shape_success(self, random_patch):
        """Function without axis parameter, same shape returns success."""

        # Create a function that has no 'axis' in its signature and preserves shape
        def element_wise_func(data):
            """A simple element-wise function that preserves array shape."""
            return np.abs(data) + 1

        # Ensure the function has a name for error reporting
        element_wise_func.__name__ = "element_wise_func"

        result = apply_array_func(element_wise_func, random_patch)

        # Verify the result
        assert isinstance(result, dc.Patch)
        assert result.shape == random_patch.shape
        assert result.coords.equals(random_patch.coords)  # coords should be preserved
        # attrs should be preserved, apart from the id which says an array
        # function was applied -- which is the one thing that did happen.
        managed = ("processing_id",)
        assert result.attrs.drop(*managed) == random_patch.attrs.drop(*managed)
        assert result.attrs.processing_id != random_patch.attrs.processing_id
        assert np.allclose(result.data, np.abs(random_patch.data) + 1)


class TestHashArray:
    """Tests for hash_array."""

    def test_returns_hex_string_of_length_32(self):
        """Output is a 32-character hex string (16-byte digest)."""
        result = hash_array(np.array([1, 2, 3]))
        assert isinstance(result, str)
        assert len(result) == 32

    def test_copy_same_hash(self):
        """A copy of an array produces the same hash."""
        a = np.array([1.0, 2.0, 3.0])
        assert hash_array(a) == hash_array(a.copy())

    def test_different_values_different_hash(self):
        """Different data produces a different hash."""
        a = np.array([1, 2, 3])
        b = np.array([1, 2, 4])
        assert hash_array(a) != hash_array(b)

    def test_different_dtype_different_hash(self):
        """Same raw shape but different dtype produces a different hash."""
        a = np.array([1, 2, 3], dtype=np.int32)
        b = np.array([1, 2, 3], dtype=np.int64)
        assert hash_array(a) != hash_array(b)

    def test_different_shape_different_hash(self):
        """Same values but reshaped produce a different hash."""
        a = np.arange(6).reshape(2, 3)
        b = np.arange(6).reshape(3, 2)
        assert hash_array(a) != hash_array(b)

    def test_object_array_raises(self):
        """Object arrays are not supported."""
        with pytest.raises(ParameterError):
            hash_array(np.array([1, "a"], dtype=object))

    def test_non_contiguous_matches_contiguous(self):
        """A non-C-contiguous view hashes identically to its contiguous copy."""
        base = np.arange(12).reshape(3, 4)
        # Fortran-order (non-C-contiguous)
        non_contig = np.asfortranarray(base)
        assert not non_contig.flags.c_contiguous
        assert hash_array(base) == hash_array(non_contig)

    def test_datetime_array_hashes(self):
        """Datetime arrays should hash without special casing at call sites."""
        arr = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
        assert hash_array(arr) == hash_array(arr.copy())


class TestArrayBackends:
    """Tests for operators applied to patches backed by other array libraries."""

    @pytest.fixture(scope="class")
    def xps(self):
        """The reference implementation of the array API standard."""
        return pytest.importorskip("array_api_strict")

    @pytest.fixture(scope="class")
    def backend_patch(self, random_patch, to_backend) -> dc.Patch:
        """A patch whose data are backed by the array backend under test."""
        return to_backend(random_patch)

    @pytest.fixture(scope="class")
    def int_numpy_patch(self, random_patch) -> dc.Patch:
        """A patch with integer data."""
        data = (np.asarray(random_patch.data) * 10).astype("int32")
        return random_patch.new(data=data)

    @pytest.fixture(scope="class")
    def int_patch(self, int_numpy_patch, to_backend) -> dc.Patch:
        """A patch with integer data on the backend under test."""
        return to_backend(int_numpy_patch)

    def _assert_matches_numpy(self, out, expected, backend):
        """The output keeps its backend and matches the patch numpy returns."""
        assert backend_name(out.data) == backend
        array = np.asarray(out.data)
        assert array.dtype == expected.data.dtype
        assert out.dims == expected.dims
        assert out.coords == expected.coords
        assert out.attrs == expected.attrs
        assert np.allclose(array, np.asarray(expected.data), equal_nan=True)

    def test_scalar_operand(self, backend_patch, random_patch, backend):
        """Operations with python scalars stay on the patch's backend."""
        with warnings_as_errors():
            out = backend_patch / 10 + 1
        self._assert_matches_numpy(out, random_patch / 10 + 1, backend)

    def test_patch_operand(self, backend_patch, random_patch, backend):
        """So do operations between two patches."""
        with warnings_as_errors():
            out = backend_patch * backend_patch
        self._assert_matches_numpy(out, random_patch * random_patch, backend)

    def test_reversed_operand(self, backend_patch, random_patch, backend):
        """Reversed operators (scalar on the left) also work."""
        with warnings_as_errors():
            out = 1 - backend_patch
        self._assert_matches_numpy(out, 1 - random_patch, backend)

    def test_unary_ufunc(self, backend_patch, random_patch, backend):
        """Unary ufuncs use the backend's own implementation."""
        with warnings_as_errors():
            out = np.exp(backend_patch)
        self._assert_matches_numpy(out, np.exp(random_patch), backend)

    def test_numpy_array_operand(self, backend_patch, random_patch, backend):
        """A numpy array operand is converted to the patch's backend."""
        other = np.ones(backend_patch.shape)
        with warnings_as_errors():
            out = backend_patch + other
        self._assert_matches_numpy(out, random_patch + other, backend)

    def test_comparison_clears_units(self, backend_patch, backend):
        """Comparisons return bool data, which have no units."""
        with warnings_as_errors():
            out = backend_patch > 0
        assert backend_name(out.data) == backend
        assert array_namespace(out.data).isdtype(out.data.dtype, "bool")
        assert out.attrs.data_units is None

    def test_dtype_argument(self, backend_patch, backend):
        """A dtype argument is not mistaken for an array from the backend."""
        with pytest.warns(NumpyFallbackWarning):
            out = backend_patch.add.reduce(dim="time", dtype=np.float32)
        assert backend_name(out.data) == backend

    def test_reduce_falls_back(self, backend_patch, random_patch, backend):
        """Reductions have no array API equivalent, so they use numpy."""
        with pytest.warns(NumpyFallbackWarning, match="reduce"):
            out = backend_patch.add.reduce(dim="time")
        self._assert_matches_numpy(out, random_patch.add.reduce(dim="time"), backend)

    def test_array_function_falls_back(self, backend_patch, random_patch, backend):
        """So do numpy functions applied to a patch."""
        with pytest.warns(NumpyFallbackWarning, match="mean"):
            out = np.mean(backend_patch, axis=0)
        self._assert_matches_numpy(out, np.mean(random_patch, axis=0), backend)

    def test_units_fall_back(self, backend_patch, random_patch, backend):
        """Units are implemented with pint, which only wraps numpy arrays."""
        with pytest.warns(NumpyFallbackWarning):
            out = backend_patch * get_quantity("m")
        expected = random_patch * get_quantity("m")
        self._assert_matches_numpy(out, expected, backend)
        assert out.attrs.data_units == expected.attrs.data_units

    def test_ufunc_outside_the_standard(self, backend_patch, random_patch, backend):
        """Ufuncs the standard doesn't define still match numpy.

        Whether they need the numpy fallback depends on the backend, since
        array-api-compat's wrappers expose more than the standard requires.
        """
        with suppress_warnings(NumpyFallbackWarning):
            out = np.fmod(backend_patch, 2)
        self._assert_matches_numpy(out, np.fmod(random_patch, 2), backend)

    def test_foreign_array_operand_converted(
        self, backend_patch, random_patch, backend
    ):
        """An operand which is a bare array from the backend crosses too."""
        array = np.asarray(random_patch.data) + 1
        other = to_backend_array(backend_patch, array)
        with suppress_warnings(NumpyFallbackWarning):
            out = np.fmod(backend_patch, other)
        self._assert_matches_numpy(out, np.fmod(random_patch, array), backend)

    def test_reduction_dtype_without_equivalent(
        self, backend_patch, random_patch, backend
    ):
        """A dtype the standard cannot reduce is reduced by numpy."""
        array = np.asarray(random_patch.data) > 0.5
        numpy_patch = random_patch.new(data=array)
        patch = backend_patch.new(data=to_backend_array(backend_patch, array))
        # The standard has no minimum of a boolean array, but numpy does.
        with suppress_warnings(NumpyFallbackWarning):
            out = patch.min("time")
        self._assert_matches_numpy(out, numpy_patch.min("time"), backend)

    def test_aggregation_the_standard_lacks(self, backend_patch, random_patch):
        """An aggregation with no name in the standard goes straight to the func."""
        # aggregate promises nothing about the backend of its output, so
        # everything but the backend has to survive.
        out = backend_patch.aggregate(dim=None, method="median")
        expected = random_patch.aggregate(dim=None, method="median")
        array = np.asarray(out.data)
        assert array.dtype == expected.data.dtype
        assert out.dims == expected.dims
        assert out.coords == expected.coords
        assert out.attrs == expected.attrs
        assert np.allclose(array, np.asarray(expected.data))

    @pytest.mark.parametrize("dtype", ["int32", "bool"])
    @pytest.mark.parametrize(
        "name", ["min", "max", "sum", "mean", "std", "demean", "standardize"]
    )
    def test_dtypes_without_fractions(
        self, name, dtype, backend_patch, random_patch, backend
    ):
        """Data which can't hold a fraction still match numpy exactly."""
        array = (np.asarray(random_patch.data) * 10).astype(dtype)
        numpy_patch = random_patch.new(data=array)
        patch = backend_patch.new(data=to_backend_array(backend_patch, array))
        with suppress_warnings(NumpyFallbackWarning):
            out = getattr(patch, name)("time")
            expected = getattr(numpy_patch, name)("time")
        self._assert_matches_numpy(out, expected, backend)

    @pytest.mark.parametrize("norm", ["l1", "l2", "max", "bit"])
    def test_normalize_integer_data(self, norm, backend_patch, random_patch, backend):
        """Every normalization divides, so integer data must promote."""
        array = (np.asarray(random_patch.data) * 10).astype("int32")
        numpy_patch = random_patch.new(data=array)
        patch = backend_patch.new(data=to_backend_array(backend_patch, array))
        with suppress_warnings(NumpyFallbackWarning):
            out = patch.normalize("time", norm=norm)
            expected = numpy_patch.normalize("time", norm=norm)
        self._assert_matches_numpy(out, expected, backend)

    def test_no_equivalent_in_the_standard(self, xps):
        """A ufunc the standard lacks has no equivalent to look up."""
        array = xps.asarray([1.0, 2.0])
        assert array_utils._get_backend_ufunc(np.fmod, array) is None
        assert array_utils._get_backend_ufunc(np.power, array) is xps.pow

    @pytest.mark.parametrize("numpy_name,array_api_name", sorted(UFUNC_NAMES.items()))
    def test_renamed_ufuncs_exist(self, numpy_name, array_api_name, xps):
        """Each renamed ufunc exists under both names."""
        assert isinstance(getattr(np, numpy_name), np.ufunc)
        assert hasattr(xps, array_api_name)

    def test_numpy_scalar_operand(self, backend_patch, random_patch, backend):
        """Numpy scalars are converted; they are not python scalars."""
        with warnings_as_errors():
            out = backend_patch + np.float64(1)
        self._assert_matches_numpy(out, random_patch + np.float64(1), backend)

    def test_sequence_operand_falls_back(self, backend_patch, random_patch, backend):
        """Sequences have no dtype, and the standard won't promote them."""
        other = [1] * backend_patch.shape[-1]
        with pytest.warns(NumpyFallbackWarning):
            out = backend_patch + other
        self._assert_matches_numpy(out, random_patch + other, backend)

    def test_ufunc_keyword_falls_back(self, backend_patch, random_patch, backend):
        """Numpy-only ufunc keywords have no array API equivalent."""
        where = np.ones(backend_patch.shape, dtype=bool)
        with pytest.warns(NumpyFallbackWarning):
            out = np.add(backend_patch, 1, where=where)
        self._assert_matches_numpy(out, np.add(random_patch, 1, where=where), backend)

    def test_stored_units_fall_back(self, backend_patch, random_patch, backend):
        """Units stored on a patch become quantities when patches align."""
        with suppress_warnings(NumpyFallbackWarning):
            out = backend_patch.set_units("m") * backend_patch.set_units("m")
        expected = random_patch.set_units("m") * random_patch.set_units("m")
        self._assert_matches_numpy(out, expected, backend)
        assert out.attrs.data_units == expected.attrs.data_units

    def test_integer_data_falls_back(self, int_patch, int_numpy_patch, backend):
        """Numpy and the standard disagree most on integer data."""
        with pytest.warns(NumpyFallbackWarning):
            out = np.rint(int_patch)
        # numpy casts integers up to floats here, the standard does not.
        expected = np.rint(int_numpy_patch)
        assert np.asarray(out.data).dtype == expected.data.dtype
        assert backend_name(out.data) == backend

    def test_broadcast_keeps_backend(self, backend_patch, random_patch, backend):
        """Broadcasting a patch up to a shape doesn't convert its data."""
        collapsed = backend_patch.mean("time")
        with suppress_warnings(NumpyFallbackWarning):
            out = collapsed.make_broadcastable_to(backend_patch.shape)
        assert backend_name(out.data) == backend
        assert out.shape == backend_patch.shape

    def test_other_backend_operand(self, backend_patch, backend):
        """An array from a third backend is not passed to this one."""
        other = _OtherBackendArray(np.ones(backend_patch.shape))
        assert not array_utils._operand_can_apply(other, backend_patch.data)

    @pytest.mark.parametrize("ufunc", [np.absolute, np.power, np.arctan2, np.rint])
    def test_renamed_ufunc_values(self, ufunc, backend_patch, random_patch, backend):
        """Renamed ufuncs give the same answer as numpy does."""
        with warnings_as_errors():
            out = ufunc(backend_patch, 2) if ufunc.nin == 2 else ufunc(backend_patch)
        expected = ufunc(random_patch, 2) if ufunc.nin == 2 else ufunc(random_patch)
        self._assert_matches_numpy(out, expected, backend)
