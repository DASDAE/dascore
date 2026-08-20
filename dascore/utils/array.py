"""
Utilities for working with patches and arrays.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np

import dascore as dc
from dascore.compat import array, is_array
from dascore.constants import PatchType
from dascore.exceptions import ParameterError, PatchBroadcastError, UnitError
from dascore.models import ArrayLike
from dascore.units import DimensionalityError, Quantity, Unit, get_quantity
from dascore.utils.array_api import (
    array_namespace,
    asarray_like,
    is_foreign,
    is_numpy,
    nan_reduce,
)
from dascore.utils.misc import iterate, suppress_warnings
from dascore.utils.patch import (
    _merge_aligned_coords,
    _merge_models,
    align_patch_coords,
    get_dim_axis_value,
    numpy_fallback,
    swap_kwargs_dim_to_axis,
)
from dascore.warnings import DASCoreWarning
from dascore.workflow.builtin import ArrayFunc, Ufunc
from dascore.workflow.identity import ids_enabled, stamp_combination
from dascore.workflow.processor import _PATCH_ARGUMENT

# Numpy reductions which skip nans, and the name they are known by in
# dascore.utils.array_api.nan_reduce.
NAN_REDUCTIONS = {
    np.nanmax: "max",
    np.nanmean: "mean",
    np.nanmin: "min",
    np.nanstd: "std",
    np.nansum: "sum",
}

# Numpy reductions which the array API standard defines under the same name.
REDUCTIONS = {np.all: "all", np.any: "any"}

# Numpy ufunc names which the array API standard spells differently.
UFUNC_NAMES = {
    "absolute": "abs",
    "arccos": "acos",
    "arccosh": "acosh",
    "arcsin": "asin",
    "arcsinh": "asinh",
    "arctan": "atan",
    "arctan2": "atan2",
    "arctanh": "atanh",
    "conjugate": "conj",
    "invert": "bitwise_invert",
    "left_shift": "bitwise_left_shift",
    "power": "pow",
    "right_shift": "bitwise_right_shift",
    "rint": "round",
}


def _dummy_accumulate(array, axis=0, dtype=None, out=None):
    """A dummy accumulate function since inspect.signature fails."""


def _dummy_reduce(array, axis=0, dtype=None, out=None, keepdims=False, **kwargs):
    """A dummy reduce function since inspect.signature fails."""


def _raise_on_out(kwargs):
    """DASCore doesn't support the out parameter. Check for it and raise."""
    if "out" in kwargs and kwargs["out"] is not None:
        msg = (
            "Since patches are immutable, the 'out' parameter cannot be "
            "used in patch functions."
        )
        raise ParameterError(msg)


def _clear_units_if_bool_dtype(patch):
    """Clear the units on the patch if it is a boolean."""
    # Every ufunc and array function path reassembles a patch first.
    assert hasattr(patch, "dtype"), "can only clear the units of a patch"
    if array_namespace(patch.data).isdtype(patch.dtype, "bool"):
        return patch.update_attrs(data_units=None)
    return patch


def _is_text_coercible_array(data: Any) -> bool:
    """Return True when an array can be normalized to text values.

    Arrays must expose ``dtype``; non-array inputs return ``False``. Unicode
    and fixed-width byte dtypes are accepted directly. Object arrays are only
    accepted if every flattened element is a string/bytes instance. Empty
    object arrays return ``False``.
    """
    if not hasattr(data, "dtype"):
        return False
    if data.dtype.kind in {"U", "S"}:
        return True
    if data.dtype.kind != "O":
        return False
    array = np.asarray(data)
    if not array.size:
        return False
    string_types = (str, bytes, np.str_, np.bytes_)
    flat = array.reshape(-1)
    return all(isinstance(value, string_types) for value in flat)


def _coerce_text_array(data: ArrayLike) -> np.ndarray:
    """Normalize text-coercible arrays to unicode numpy arrays."""
    array = np.asarray(data)
    if not _is_text_coercible_array(array):
        msg = "CoordString requires string-like data."
        raise ValueError(msg)
    if array.dtype.kind == "U":
        return array.astype(array.dtype, copy=False)
    if not array.size:
        return np.empty(array.shape, dtype=f"U{array.dtype.itemsize}")
    flat = array.reshape(-1)
    decoded = []
    for value in flat:
        if isinstance(value, bytes | bytearray | np.bytes_):
            decoded.append(bytes(value).decode("utf-8"))
        else:
            decoded.append(str(value))
    return np.asarray(decoded, dtype="U").reshape(array.shape)


def is_string_byte_serializable_array(data: ArrayLike) -> bool:
    """Return True when an array should use fixed-width string-byte storage."""
    data = np.asarray(data)
    return data.dtype.kind in {"U", "S"} or (
        data.dtype.kind == "O" and _is_text_coercible_array(data)
    )


def convert_strings_to_bytes(data: ArrayLike) -> np.ndarray:
    """Encode string-like arrays as fixed-width UTF-8 bytes."""
    data = np.asarray(data)
    if not data.size:
        return np.empty(data.shape, dtype="S1")
    # Storage backends that require one fixed-width bytes dtype need the
    # encoded values flattened first so the longest byte string can size it.
    flat = data.reshape(-1)
    encoded = []
    for value in flat:
        if isinstance(value, bytes | bytearray | np.bytes_):
            encoded.append(bytes(value))
        else:
            encoded.append(str(value).encode("utf-8"))
    max_len = max(len(item) for item in encoded)
    out = np.asarray(encoded, dtype=f"S{max_len}")
    return out.reshape(data.shape)


def convert_bytes_to_strings(data: ArrayLike, original_dtype="") -> np.ndarray:
    """Decode fixed-width bytes arrays back to the requested string form."""
    data = np.asarray(data)
    if not data.size:
        dtype = original_dtype if isinstance(original_dtype, str) else ""
        if dtype.startswith("|S") or dtype.startswith("S"):
            return np.empty(data.shape, dtype=dtype)
        if dtype.startswith("<U") or dtype.startswith("U"):
            return np.empty(data.shape, dtype=dtype)
        return np.empty(data.shape, dtype="U1")
    # Preserve byte-backed arrays when requested. Otherwise decode back to the
    # stored string representation using the recorded original dtype.
    if isinstance(original_dtype, str) and (
        original_dtype.startswith("|S") or original_dtype.startswith("S")
    ):
        return data.astype(original_dtype, copy=False).reshape(data.shape)
    flat = [value.decode("utf-8") for value in data.reshape(-1)]
    out = np.asarray(flat, dtype="U").reshape(data.shape)
    if original_dtype in {"object", "|O"}:
        return out.astype(object)
    if isinstance(original_dtype, str) and (
        original_dtype.startswith("<U") or original_dtype.startswith("U")
    ):
        return out.astype(original_dtype)
    return out


def _get_backend_ufunc(ufunc, array):
    """Get the array API equivalent of a numpy ufunc, or None if it has none."""
    name = UFUNC_NAMES.get(ufunc.__name__, ufunc.__name__)
    return getattr(array_namespace(array), name, None)


def _is_python_scalar(value):
    """Return True for a scalar every array API function accepts."""
    # Not isinstance; np.float64 subclasses float but is not a python scalar.
    return type(value) in (int, float, bool)


def _is_inexact(array):
    """Return True if the array holds real or complex floating point values."""
    xp = array_namespace(array)
    return xp.isdtype(array.dtype, ("real floating", "complex floating"))


def _as_backend(value, template):
    """Put a ufunc operand in the same array namespace as template."""
    # Python scalars are valid operands for any backend, as are arrays which
    # already belong to it.
    if _is_python_scalar(value) or array_namespace(value) is array_namespace(template):
        return value
    return asarray_like(value, template)


def _apply_operator(operator, *arrays, **kwargs):
    """
    Apply a ufunc to arrays, one of which may not be numpy backed.

    Numpy data are passed to the numpy ufunc, everything else uses the
    matching function from its own array API namespace.
    """
    template = next((x for x in arrays if is_foreign(x)), None)
    if template is None:
        return operator(*arrays, **kwargs)
    func = _get_backend_ufunc(operator, template)
    assert func is not None, f"{operator.__name__} has no array API equivalent."
    arrays = tuple(_as_backend(x, template) for x in arrays)
    return func(*arrays, **kwargs)


def _apply_unary_ufunc(operator: np.ufunc, patch, *args, **kwargs):
    """
    Create a patch from a unary ufunc.

    Parameters
    ----------
    operator
        The operator. Must be numpy ufunc-like.
    patch
        The patch instance.
    *args
        Arguments to pass to the operator, can include arrays, scalars,
        and patches.
    **kwargs
        Keyword arguments to pass to the operator.

    Notes
    -----
    We assume the shape of the array won't change.
    """
    out = _apply_operator(operator, patch.data, *args, **kwargs)
    # As for the binary case: a ufunc has no patch function to name it, so
    # `np.abs(patch)` would otherwise record that nothing happened.
    task = Ufunc(
        name=getattr(operator, "__name__", str(operator)),
        operands=tuple(args),
        kwargs=_without_patch_values(kwargs),
    )
    attrs = stamp_combination(patch.attrs, [patch.attrs], task.fingerprint)
    return patch.new(data=out, attrs=attrs)


def _apply_binary_ufunc(
    operator: np.ufunc,
    patch: PatchType | ArrayLike,
    other: PatchType | ArrayLike,
    *args: tuple[PatchType | ArrayLike, ...],
    **kwargs,
) -> PatchType:
    """
    Apply a binary ufunc-type operator to one or more patches.

    The input should be 2 and the output 1.

    This is used to implement a patch's operator overloading.

    Parameters
    ----------
    operator
        The operator. Must be numpy ufunc-like.
    patch
        The patch instance.
    other
        The other object to apply the operator element-wise. Must be either a
        non-patch which is broadcastable to the shape of the patch's data, or
        a patch of the same kind (see
        [`check_kind`](`dascore.utils.patch.check_kind`)) sharing at least
        one dimension; shared dimensions are aligned on the intersection
        of their coordinate values. Data units take part in the operation,
        so they need not match unless the operator requires it (adding
        metres to seconds raises, multiplying them does not). An operand
        without units — an array, a scalar, or a patch with no
        `data_units` — conflicts with nothing: it is dimensionless where
        that works (`metres * x` is metres, `x / metres` is 1/metres) and
        takes the other operand's units where the operation needs equal
        units (`metres + x` is metres).
    *args
        Arguments to pass to the operator, can include arrays, scalars,
        and patches.
    **kwargs
        Keyword arguments to pass to the operator.

    Notes
    -----
    See [numpy's ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html)
    """

    def _get_coords_attrs_from_patches(patch, other):
        """Deal with aligning two patches."""
        # Align patches so their coords are identical and data aligned.
        patch, other_patch = align_patch_coords(patch, other)
        coords = _merge_aligned_coords(patch.coords, other_patch.coords)
        # Get new attributes.
        attrs = _merge_models(patch.attrs, other_patch.attrs)
        other = other_patch.data
        if other_units := get_quantity(other_patch.attrs.data_units):
            other = other * other_units
        return patch, other, coords, attrs

    def _ensure_array_compatible(patch, other):
        """Deal with broadcasting a patch and an array."""
        # This handles warning from quantity.
        other = other.magnitude if hasattr(other, "magnitude") else other
        # Only the shape is needed here, so don't convert arrays which
        # already have one; that would materialize non-numpy data.
        if (shape := getattr(other, "shape", None)) is None:
            shape = np.asanyarray(other).shape
        if patch.shape == shape:
            return patch
        if (patch_ndims := patch.ndim) < (array_ndims := len(shape)):
            msg = f"Cannot broadcast patch/array {patch_ndims=} {array_ndims=}"
            raise PatchBroadcastError(msg)
        patch = patch.make_broadcastable_to(shape)
        return patch

    def _apply_op(array1, array2, operator, reversed=False):
        """Simply apply the operator, account for reversal."""
        if reversed:
            array1, array2 = array2, array1
        return _apply_operator(operator, array1, array2, *args, **kwargs)

    def _apply_op_unitless_other(patch, other, operator, attrs, reversed=False):
        """
        Apply the operation when the patch has units and `other` has none.

        `other` conflicts with nothing: the units of the result are
        settled on scalars first — `other` as dimensionless, and if the
        operator rejects that, as sharing the patch's units (sums and
        comparisons need equal units) — then the arrays are operated on
        bare. Nothing large is ever wrapped by the unit registry, and a
        ufunc the registry does not implement falls through to numpy
        with the units left as they were.
        """
        data_units = get_quantity(attrs.data_units)
        assert data_units is not None, "caller guarantees the patch has units"
        if operator in (np.power, np.float_power) and np.ndim(other) > 0:
            msg = f"{operator} with units {data_units} needs a scalar exponent."
            raise UnitError(msg)
        probe_other = other if np.ndim(other) == 0 else 1.0
        dimensionless = get_quantity("dimensionless")
        assert dimensionless is not None
        try:
            patch_q, other_q = 1.0 * data_units, probe_other * dimensionless
            if reversed:
                patch_q, other_q = other_q, patch_q
            probe = operator(patch_q, other_q)
        except DimensionalityError:
            adopted = probe_other * data_units
            pair = (
                (adopted, 1.0 * data_units) if reversed else (1.0 * data_units, adopted)
            )
            try:
                probe = operator(*pair)
            except DimensionalityError as er:
                msg = f"{operator} failed with units {data_units} and none"
                raise UnitError(msg) from er
        except TypeError:
            # The unit registry does not implement this ufunc (or cannot
            # hold this scalar, a bool say); numpy does, and the units are
            # whatever they were.
            return _apply_op(patch.data, other, operator, reversed), attrs
        new_data = _apply_op(patch.data, other, operator, reversed)
        if hasattr(probe, "units"):
            return new_data, attrs.update(data_units=str(probe.units))
        return new_data, attrs.update(data_units=None)

    def _apply_op_units(patch, other, operator, attrs, reversed=False):
        """
        Apply the operation with units on at least one side.

        Two known units are left to the unit registry to reconcile or
        reject. A patch without units beside a unitful operand is
        dimensionless first and, if the operator rejects that, adopts the
        operand's units — the same rule `_apply_op_unitless_other` applies
        the other way round.
        """
        data_units = get_quantity(attrs.data_units)
        # A bare number or array is an operand without units; only a unit
        # or a unit string names units.
        if isinstance(other, Unit):
            other = 1 * other
        elif isinstance(other, str):
            other = get_quantity(other)
        if not isinstance(other, Quantity):
            return _apply_op_unitless_other(patch, other, operator, attrs, reversed)
        dimensionless = get_quantity("dimensionless")
        patch_q = data_units if data_units is not None else dimensionless
        attempts = [(patch_q, other)]
        if data_units is None:
            attempts.append((other.units, other))
        error = None
        for patch_q, other_q in attempts:
            try:
                new_data_w_units = _apply_op(
                    patch.data * patch_q, other_q, operator, reversed
                )
                break
            except DimensionalityError as er:
                error = er
        else:
            msg = f"{operator} failed with units {data_units} and {other.units}"
            raise UnitError(msg) from error
        # Check if result has units (comparison operators return plain arrays)
        if hasattr(new_data_w_units, "units"):
            attrs = attrs.update(data_units=str(new_data_w_units.units))
            new_data = new_data_w_units.magnitude
        else:
            # Result is unitless (e.g., from boolean comparison)
            attrs = attrs.update(data_units=None)
            new_data = new_data_w_units
        return new_data, attrs

    # Count patch operands (we only support binary ops on patches).
    patch_is_patch = isinstance(patch, dc.Patch)
    other_is_patch = isinstance(other, dc.Patch)
    patch_count = int(patch_is_patch) + int(other_is_patch)
    # We only support binary ops on patch.
    assert patch_count >= 1, "apply_ufunc requires at least one Patch operand."
    # Make sure patch is a patch. Reverse if needed.
    reversed = False
    if other_is_patch and not patch_is_patch:
        patch, other = other, patch
        reversed = True

    # Taken before the operands are aligned and possibly replaced below:
    # what went in is what decides which data comes out.
    members = [x.attrs for x in (patch, other) if isinstance(x, dc.Patch)]
    if patch_count > 1:
        patch, other, coords, attrs = _get_coords_attrs_from_patches(patch, other)
    else:
        patch = _ensure_array_compatible(patch, other)
        coords, attrs = patch.coords, patch.attrs
    # Apply operation; only two unitless operands skip the unit registry.
    if isinstance(other, Quantity | Unit) or attrs.data_units is not None:
        new_data, attrs = _apply_op_units(patch, other, operator, attrs, reversed)
    else:
        new_data = _apply_op(patch.data, other, operator, reversed)
    # A ufunc is not a patch function, so nothing else names it. Without
    # this, `patch + 1` and `patch - (-1)` produce the same data and the
    # same id, though they are different operations. Guarded, so that a
    # process which has turned the ids off does not hash operands for a
    # value nothing will read.
    if ids_enabled():
        rest = () if other_is_patch else (other,)
        task = Ufunc(
            name=getattr(operator, "__name__", str(operator)),
            reversed=reversed,
            # `args` reaches the operator too, so two calls which differ
            # only in those are two operations.
            operands=_without_patch_values((*rest, *args)),
            kwargs=_without_patch_values(kwargs),
        )
        attrs = stamp_combination(attrs, members, _fingerprint_of(task))
    new = patch.new(data=new_data, coords=coords, attrs=attrs)
    return new


class _BoundPatchUFunc:
    """A ufunc bound to a specific patch instance."""

    def __init__(self, np_ufunc, patch):
        self.np_ufunc = np_ufunc
        self.patch = patch
        self.__name__ = getattr(np_ufunc, "__name__", "patch_ufunc")
        self.__doc__ = getattr(np_ufunc, "__doc__", None)

    def __call__(self, *args, **kwargs):
        """Call the ufunc with the bound patch."""
        return apply_ufunc(self.np_ufunc, self.patch, *args, **kwargs)

    def reduce(self, dim=None, dtype=None, **kwargs):
        """Apply ufunc reduction along specified dimensions."""
        return apply_ufunc(
            self.np_ufunc.reduce, self.patch, dim=dim, dtype=dtype, **kwargs
        )

    def accumulate(self, dim=None, dtype=None, **kwargs):
        """Apply ufunc accumulation along specified dimensions."""
        return apply_ufunc(
            self.np_ufunc.accumulate, self.patch, dim=dim, dtype=dtype, **kwargs
        )


class PatchUFunc:
    """
    A ufunc wrapper that can be applied to patches with dimension support.

    This class wraps numpy ufuncs to work seamlessly with DASCore patches,
    providing support for dimension-aware operations, coordinate preservation,
    and method binding.

    Parameters
    ----------
    np_ufunc : np.ufunc
        The numpy ufunc to wrap.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.utils.array import PatchUFunc
    >>>
    >>> # Create a patch ufunc from np.add
    >>> add_ufunc = PatchUFunc(np.add)
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Use it to operate on patches
    >>> result = add_ufunc(patch, patch)
    >>>
    >>> # Use accumulate and reduce methods
    >>> cumsum = add_ufunc.accumulate(patch, dim="time")
    >>> total = add_ufunc.reduce(patch, dim="distance")
    >>>
    >>> # Bind to patch instance for cleaner syntax
    >>> bound_ufunc = add_ufunc.__get__(patch, type(patch))
    >>> result2 = bound_ufunc.reduce(dim="time")  # No patch argument needed

    Notes
    -----
    - Automatically handles dimension-to-axis conversion when `dim` parameter is used
    - Preserves patch coordinates and attributes appropriately
    - Supports method binding via the descriptor protocol
    - Methods call `dascore.utils.array.apply_ufunc` under the hood
    """

    def __init__(self, np_ufunc):
        self.np_ufunc = np_ufunc
        self.__name__ = getattr(np_ufunc, "__name__", "patch_ufunc")
        self.__doc__ = getattr(np_ufunc, "__doc__", None)

    def __get__(self, obj, objtype=None):
        """Bind to a patch instance, returning a _BoundPatchUFunc."""
        if obj is None:
            return self
        return _BoundPatchUFunc(self.np_ufunc, obj)

    def __call__(self, patch, *args, **kwargs):
        """Call the ufunc with a patch as first argument."""
        return apply_ufunc(self.np_ufunc, patch, *args, **kwargs)

    def reduce(self, patch, dim=None, dtype=None, **kwargs):
        """Apply ufunc reduction along specified dimensions."""
        return apply_ufunc(self.np_ufunc.reduce, patch, dim=dim, dtype=dtype, **kwargs)

    def accumulate(self, patch, dim=None, dtype=None, **kwargs):
        """Apply ufunc accumulation along specified dimensions."""
        return apply_ufunc(
            self.np_ufunc.accumulate, patch, dim=dim, dtype=dtype, **kwargs
        )


def _apply_reduction(func, data, axis):
    """
    Apply a reduction to data, using its own array namespace if it can.

    nan_reduce handles the dtypes the standard cannot reduce.
    """
    if not is_numpy(data):
        if (name := NAN_REDUCTIONS.get(func)) is not None:
            return nan_reduce(name, data, axis=axis)
        if (name := REDUCTIONS.get(func)) is not None:
            return getattr(array_namespace(data), name)(data, axis=axis)
    # Numpy data, or an aggregation the standard has no name for: a median, or
    # a callable passed to aggregate. It gets the array as-is, exactly as
    # before dascore knew about other backends, and decides the output's
    # backend.
    return func(data, axis=axis)


def _apply_aggregator(patch, dim, func, dim_reduce="empty"):
    """Apply an aggregation operator to patch."""
    data = patch.data
    dims = tuple(iterate(patch.dims if dim is None else dim))
    dfo = get_dim_axis_value(patch, args=dims, allow_multiple=True)
    if dim_reduce == "squeeze" and {dim for dim, _, _ in dfo} == set(patch.dims):
        msg = "Cannot squeeze all dimensions; at least one dimension must remain."
        raise ParameterError(msg)
    # Iter all specified dimensions.
    for dim, _, _ in dfo:
        axis = patch.get_axis(dim)
        new_coord = patch.get_coord(dim).reduce_coord(dim_reduce=dim_reduce)
        if new_coord is None:
            coords = patch.coords.drop_coords(dim)[0]
            data = _apply_reduction(func, data, axis)
        else:
            coords = patch.coords.update(**{dim: new_coord})
            reduced = _apply_reduction(func, data, axis)
            data = array_namespace(reduced).expand_dims(reduced, axis=axis)
        attrs = patch.attrs.model_dump(exclude={"coords", "dims"}, exclude_unset=True)
        patch = patch.new(data=data, coords=coords, attrs=attrs)
    return patch


def _find_patches(args, kwargs):
    """Return any patches in args and kwargs."""
    patches1 = [x for x in args if isinstance(x, dc.Patch)]
    patches2 = [x for x in kwargs.values() if isinstance(x, dc.Patch)]
    return patches1 + patches2


def _strip_data_from_patch(patch):
    """Get data from patches."""
    return patch.data if isinstance(patch, dc.Patch) else patch


def _get_dims_and_inds_from_signature(
    patch, sig, args, kwargs
) -> tuple[tuple[str, ...], tuple]:
    """
    Get dimension over which function was applied, and indices to re-expand result.
    """

    def _normalize_axes(axis, ndim: int) -> tuple[int, ...]:
        """
        Normalize axis spec (int | tuple[int] | list[int] | None)
        to a tuple of positive ints.
        """
        if axis is None:
            return tuple(range(ndim))  # reduce over all axes
        if isinstance(axis, Iterable) and not isinstance(axis, str | bytes):
            axes = tuple(int(a) for a in axis)
        else:
            axes = (int(axis),)
        # normalize negatives and dedupe but preserve order
        seen = set()
        out = []
        for a in axes:
            a = a % ndim
            if a not in seen:
                seen.add(a)
                out.append(a)
        return tuple(out)

    def get_effective_axes(sig, args, kwargs, ndim: int) -> tuple[int, ...]:
        """
        Return the axes the function will operate on, normalized to positive indices.
        If the function has no 'axis' parameter, returns an empty tuple (unknown/no-op).
        """
        params = sig.parameters
        # bind *partially* so missing, defaulted params are allowed
        bound = sig.bind_partial(*args, **(kwargs or {}))
        assert "axis" in params, "axis must be defined in signature"
        axis = bound.arguments.get("axis", params["axis"].default)
        if axis is inspect._empty or axis is None:
            axis = None  # treat as default

        return _normalize_axes(axis, ndim)

    axes = get_effective_axes(sig, args, kwargs, patch.ndim)
    # get indexes needed to re-expand array
    inds = tuple(
        None if num in axes else slice(None, None) for num in range(patch.ndim)
    )
    dims = tuple(patch.dims[x] for x in axes)
    return dims, inds


def _reassemble_patch(result, patch, func, args, kwargs):
    """
    Method to put the patch back together.
    """
    # Simple case, data shape hasn't changed.
    if result.shape == patch.shape:
        return patch.new(data=result)
    # Otherwise, we have to do some detective work to figure out what happened
    # to the array so we can adjust the coords accordingly.
    try:
        sig = inspect.signature(func)
    # Some ufunc methods can fail introspection on certain numpy/python combos.
    except (ValueError, AttributeError, TypeError):
        ufunc_method_name = {"reduce": _dummy_reduce, "accumulate": _dummy_accumulate}
        name = getattr(func, "__name__", None)
        if name in ufunc_method_name:
            sig = inspect.signature(ufunc_method_name[name])
        else:
            raise

    if "axis" in sig.parameters:
        dims, inds = _get_dims_and_inds_from_signature(patch, sig, args, kwargs)
        # re-expand array.
        result = result[inds]
        new_coords = {x: patch.get_coord(x).reduce_coord() for x in dims}
        cm = patch.coords.update(**new_coords)
        return patch.new(data=result, coords=cm)
    else:
        # This case should have already been caught at the start of this function.
        assert result.shape != patch.shape
        func_name = getattr(func, "__name__", None)
        msg = f"Cannot reassemble result of {func_name} without an axis parameter."
        raise ParameterError(msg)


def apply_array_func(func, *args, **kwargs):
    """
    Apply an array function.

    Numpy functions have no array API equivalent, so patches from other
    backends are converted to numpy and the result converted back.
    """
    _raise_on_out(kwargs)
    if (data := _get_foreign_data(args, kwargs)) is not None:
        name = getattr(func, "__name__", "operation")
        runner = functools.partial(_apply_array_func, func)
        return numpy_fallback(name, data, runner, args, kwargs, stacklevel=2)
    return _apply_array_func(func, *args, **kwargs)


def _apply_array_func(func, *args, **kwargs):
    """Apply an array function to numpy backed patches."""
    # Only handle functions involving Patches
    patches = _find_patches(args, kwargs)
    assert len(patches), "No patches found in apply_array_func"
    first_patch = patches[0]

    # Convert dim to axis for numpy functions
    converted_kwargs = swap_kwargs_dim_to_axis(first_patch, kwargs)
    converted_args = args

    _args = tuple(_strip_data_from_patch(a) for a in converted_args)
    _kwargs = {k: _strip_data_from_patch(v) for k, v in converted_kwargs.items()}

    # Call the array function
    result = func(*_args, **_kwargs)
    # If we didn't get an array back, try to package it as an array
    if not is_array(result):
        result = array(result)
    # Then we need to put the array back into the patch, but account for
    # dimensions that may have changed.
    patch = _reassemble_patch(
        result, first_patch, func, converted_args, converted_kwargs
    )
    # An array function is not a patch function either, so nothing else
    # names it: without this `np.mean(patch, axis=0)` leaves the ids where
    # they were and claims nothing was done.
    if ids_enabled():
        task = ArrayFunc(
            name=_array_func_name(func),
            # The positional arguments say which reduction it was:
            # `np.mean(patch, 0)` and `np.mean(patch, 1)` are two.
            args=_without_patch_values(converted_args),
            kwargs=_without_patch_values(converted_kwargs),
        )
        attrs = stamp_combination(
            patch.attrs, [x.attrs for x in patches], _fingerprint_of(task)
        )
        patch = patch.new(attrs=attrs)
    return _clear_units_if_bool_dtype(patch)


def _array_func_name(func) -> str:
    """
    Return the name an array function is recorded under.

    A ufunc method arrives here as the bound `np.add.reduce`, whose
    `__name__` is only "reduce" -- so `np.add.reduce` and
    `np.multiply.reduce` would be one operation without the ufunc it
    belongs to.
    """
    name = getattr(func, "__name__", str(func))
    owner = getattr(getattr(func, "__self__", None), "__name__", None)
    return f"{owner}.{name}" if owner else name


def _fingerprint_of(task) -> str:
    """
    Return a task's fingerprint without complaining about the patch marker.

    The marker is a singleton, so hashing it by its type -- which is what
    the warning is about -- loses nothing. The warning is worth hearing
    for a value where it would.
    """
    with suppress_warnings(
        DASCoreWarning, message="A value of type .* has no encoding"
    ):
        return task.fingerprint


def _without_patch_values(values):
    """
    Return arguments with anything the fingerprint should not hold replaced.

    A patch is an *input*, not a parameter -- which one it was is said by
    the ids folded from the operands. A numpy dtype has no encoding of its
    own, so it is spelled out rather than hashed by its class, which would
    give every dtype one fingerprint and warn on every call.
    """

    def _plain(value):
        if isinstance(value, dc.Patch):
            return _PATCH_ARGUMENT
        if isinstance(value, np.dtype):
            return str(value)
        return value

    if isinstance(values, Mapping):
        return {key: _plain(value) for key, value in values.items()}
    return tuple(_plain(x) for x in values)


# Mapping of ufunc dispatches. Keys are method name or num input/num output.
UFUNC_MAP = {
    (2, 1): _apply_binary_ufunc,
    (1, 1): _apply_unary_ufunc,
    "reduce": apply_array_func,
    "accumulate": apply_array_func,
}


def _get_foreign_data(args, kwargs):
    """Return the data of the first non-numpy backed operand, or None."""
    values = (*args, *kwargs.values())
    patch_data = (x.data for x in values if isinstance(x, dc.Patch))
    return next((x for x in (*patch_data, *values) if is_foreign(x)), None)


def _operand_can_apply(value, data):
    """Determine if a ufunc operand can be used with the data's namespace."""
    if _is_python_scalar(value):
        return True
    if getattr(value, "dtype", None) is None:
        return False
    # Arrays from a third backend can't be mixed with the data's own.
    if is_foreign(value) and array_namespace(value) is not array_namespace(data):
        return False
    return _is_inexact(value)


def _backend_can_apply(ufunc, key, args, kwargs, data):
    """
    Determine if a ufunc can be applied in the data's own namespace.

    The numpy fallback is always correct, so this only answers yes where
    the standard is known to agree with numpy. The standard has narrower
    dtype domains and no value-based promotion, so eg numpy's rint casts
    integers up to floats while the standard's round leaves them alone.
    """
    # Only element-wise ufuncs have array API equivalents; reductions and
    # numpy functions (__array_function__) have to use numpy.
    if key not in ((1, 1), (2, 1)):
        return False
    # Numpy-only ufunc options (where, casting, dtype) have no equivalent.
    if kwargs:
        return False
    # Units are implemented with pint, which only wraps numpy arrays. A
    # second patch's units become a quantity while its coords are aligned,
    # after this runs, so they have to be caught here too.
    if any(isinstance(x, Quantity | Unit) for x in args):
        return False
    patches = [x for x in args if isinstance(x, dc.Patch)]
    if len(patches) > 1 and any(x.attrs.data_units for x in patches):
        return False
    # Integer and boolean data are where the two disagree most, so they use
    # numpy; patch data are floating point nearly always.
    if not _is_inexact(data):
        return False
    operands = (x.data if isinstance(x, dc.Patch) else x for x in args)
    if not all(_operand_can_apply(x, data) for x in operands):
        return False
    return _get_backend_ufunc(ufunc, data) is not None


def apply_ufunc(ufunc, *args, **kwargs):
    """
    Apply a ufunc to one or more patches.

    Parameters
    ----------
    ufunc
        The ufunc to use.
    *args
        Additional positional arguments, can contain patches.
    **kwargs
        Keyword arguments, can contain patches.

    Examples
    --------
    >>> from dascore.utils.array import apply_ufunc
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Call abs on a patch.
    >>> new1 = apply_ufunc(np.abs, patch)
    >>>
    >>> # Add two patches
    >>> new2 = apply_ufunc(np.add, patch, patch)
    >>>
    >>> # multiply the patch by 10
    >>> new3 = apply_ufunc(np.multiply, patch, 10)

    Notes
    -----
    See [numpy's ufunc docs](https://numpy.org/doc/stable/reference/ufuncs.html)
    """
    _raise_on_out(kwargs)
    is_ufunc = isinstance(ufunc, np.ufunc)
    if is_ufunc:
        key = (ufunc.nin, ufunc.nout)
    else:
        key = ufunc.__name__
    # Handle bad key.
    if (func := UFUNC_MAP.get(key, None)) is None:
        msg = (
            f"ufuncs with input/output numbers or method of {key} are not "
            f"supported for use with Patch. Use the patch.data array directly."
        )
        raise ParameterError(msg)
    data = _get_foreign_data(args, kwargs)
    if data is None or _backend_can_apply(ufunc, key, args, kwargs, data):
        out = func(ufunc, *args, **kwargs)
    else:
        name = getattr(ufunc, "__name__", "operation")
        runner = functools.partial(func, ufunc)
        out = numpy_fallback(name, data, runner, args, kwargs, stacklevel=2)
    return _clear_units_if_bool_dtype(out)


def patch_array_ufunc(self, ufunc, method, *inputs, **kwargs):
    """
    Called when a numpy ufunc is applied to a patch (__array_ufunc__).
    """
    method = ufunc if method == "__call__" else getattr(ufunc, method, ufunc)
    return apply_ufunc(method, *inputs, **kwargs)


def patch_array_function(self, func, types, args, kwargs):
    """
    Intercept NumPy functions for patch operations.

    Parameters
    ----------
    func : callable
        The NumPy function being called.
    types : tuple
        Types involved in the call.
    args : tuple
        Positional arguments to the function.
    kwargs : dict
        Keyword arguments to the function.
    """
    # Only handle functions involving Patches
    assert any(issubclass(t, dc.Patch) for t in types)
    return apply_array_func(func, *args, **kwargs)


def hash_array(arr: np.ndarray) -> str:
    """
    Return a stable hash for a NumPy array.

    Fast path:
        - zero-copy for C-contiguous arrays
    Fallback:
        - makes one contiguous copy for non-contiguous arrays

    The hash includes:
        - dtype
        - shape
        - raw array bytes

    Returns
    -------
    str
        32-character hex string (16-byte BLAKE2b digest).

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.array import hash_array
    >>> a = np.array([1.0, 2.0, 3.0])
    >>> h = hash_array(a)
    >>> assert isinstance(h, str) and len(h) == 32
    >>> # Same data always produces the same hash
    >>> assert hash_array(a) == hash_array(a.copy())
    >>> # Different dtype produces a different hash
    >>> assert hash_array(a) != hash_array(a.astype(np.float32))
    """
    arr = np.asarray(arr)
    if arr.dtype == object:
        msg = "hash_array does not support object arrays."
        raise ParameterError(msg)

    h = hashlib.blake2b(digest_size=16)

    # Include dtype and shape so arrays with identical raw bytes but different
    # interpretations do not collide.
    h.update(arr.dtype.str.encode("ascii"))
    h.update(np.asarray(arr.shape, dtype=np.int64).tobytes())

    # .data rather than memoryview(...) throughout: it is the same zero-copy
    # object, and numpy types it as a memoryview, whereas ndarray's own
    # __buffer__ is declared only for Python 3.12+ and so is invisible to a
    # checker resolving the 3.11 floor this project supports.
    if arr.flags.c_contiguous and arr.dtype.kind not in {"M", "m"}:
        # Zero-copy fast path
        h.update(arr.data.cast("B"))
    else:
        # Canonicalize layout; this also handles datetime/timedelta dtypes,
        # which do not expose a Python buffer directly. The uint8 view must
        # come before .data -- a datetime64 buffer cannot be exported.
        h.update(np.ascontiguousarray(arr).view(np.uint8).data)

    return h.hexdigest()
