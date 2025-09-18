"""
Tests for patch ufuncs.
"""

from __future__ import annotations

import numpy as np
import pytest
from pint import DimensionalityError

import dascore as dc
import dascore.proc.coords
from dascore import get_quantity
from dascore.exceptions import ParameterError, UnitError
from dascore.units import furlongs, m, s
from dascore.utils.array import PatchUFunc, apply_array_func, apply_ufunc


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

    def test_incompatible_coords(self, random_patch):
        """Ensure un-alignable coords returns degenerate patch."""
        time = random_patch.get_coord("time")
        new_time = time.max() + time.step
        new = random_patch.update_attrs(time_min=new_time)
        out = apply_ufunc(np.multiply, new, random_patch)
        assert 0 in set(out.shape)
        assert out.data.size == 0
        assert out.dims == random_patch.dims

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
        When ufuncs don't have the right number of input/ouput an error
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

        # Verify it's actually summing along the distance axis
        expected = np.expand_dims(
            np.sum(random_patch.data, axis=dist_axis), axis=dist_axis
        )
        assert np.allclose(result.data, expected)

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
        from dascore.utils.array import _BoundPatchUFunc

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
        from dascore.utils.array import _BoundPatchUFunc

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
        assert result.attrs == random_patch.attrs  # attrs should be preserved
        assert np.allclose(result.data, np.abs(random_patch.data) + 1)
