"""Module for applying aggregations (reductions) along a specified axis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from dascore.constants import _AGG_FUNCS, DIM_REDUCE_DOCS
from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.utils.array import _apply_reduction, is_numpy
from dascore.utils.array_api import (
    array_namespace,
    asarray_like,
    backend_name,
    to_numpy,
    warn_numpy_fallback,
)
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import _get_nullish, iterate
from dascore.utils.patch import get_dim_axis_value
from dascore.utils.time import dtype_time_like

AGG_DOC_STR = f"""
patch
    The input Patch.
dim
    The dimension along which aggregations are to be performed.
    If None, apply aggregation to all dimensions sequentially.
    If a sequence, apply sequentially in order provided.
{DIM_REDUCE_DOCS}
"""

AGG_NOTES = """
Notes
-----
See [`Patch.aggregate`](`dascore.Patch.aggregate`) for examples
and more details.
"""

# The module's own `min`, `max`, `sum`, `any` and `all` are patch functions,
# so nothing below may use those builtins.


def _reduce_meta(meta, dim, dim_reduce):
    """Return metadata with dims reduced, and each one's axis and whether it stays."""
    dims = tuple(iterate(meta.dims if dim is None else dim))
    dfo = get_dim_axis_value(meta, args=dims, allow_multiple=True)
    if dim_reduce == "squeeze" and {x for x, _, _ in dfo} == set(meta.dims):
        msg = "Cannot squeeze all dimensions; at least one dimension must remain."
        raise ParameterError(msg)
    steps = []
    for name, _, _ in dfo:
        axis = meta.get_axis(name)
        new_coord = meta.get_coord(name).reduce_coord(dim_reduce=dim_reduce)
        if new_coord is None:
            coords = meta.coords.drop_coords(name)[0]
        else:
            coords = meta.coords.update(**{name: new_coord})
        attrs = meta.attrs.model_dump(exclude={"coords", "dims"}, exclude_unset=True)
        meta = meta.new(coords=coords, attrs=attrs)
        steps.append((axis, new_coord is not None))
    return meta, tuple(steps)


class _Reduction(PatchProcessor):
    """Reduce the data along dimensions with one aggregator."""

    name = None
    # The aggregator, for the shortcuts which fix one.
    reducer: ClassVar[Callable]

    def _reducer(self) -> Callable:
        """Return the aggregator to apply."""
        return self.reducer

    def get_metadata(self, meta):
        """Return the reduced metadata, and each reduced axis in turn."""
        values = self.kwargs
        out, steps = _reduce_meta(meta, values["dim"], values["dim_reduce"])
        return out, {"steps": steps}

    def kernel(self, data, *, steps):
        """Return the data reduced along each axis in turn."""
        func, original = self._reducer(), data
        for axis, keep in steps:
            data = _apply_reduction(func, data, axis)
            if keep:
                data = array_namespace(data).expand_dims(data, axis=axis)
        if is_numpy(original) or backend_name(data) == backend_name(original):
            return data
        # An aggregator the standard has no name for went through numpy.
        # Only a named subclass runs, so the name is set.
        assert self.name is not None
        warn_numpy_fallback(self.name, backend_name(original), skip_dascore=True)
        return asarray_like(data, original)


@compose_docstring(params=AGG_DOC_STR, options=sorted(_AGG_FUNCS))
class Aggregate(_Reduction):
    """
    Aggregate values along a specified dimension.

    Notes
    -----
    The output stays on the patch's array backend. A method the array API
    standard has no name for, such as a median or a callable, runs through
    NumPy and warns before the result is converted back; most shortcuts,
    such as [`Patch.mean`](`dascore.proc.aggregate.mean`), avoid that round
    trip, so prefer one where it fits.

    Parameters
    ----------
    {params}
    method
        The aggregation to apply along dimension. Options are:
            {options}

    See Also
    --------
    - See also the aggregation shortcut methods in the
      [aggregate module](`dascore.proc.aggregate`).

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc

    >>> patch = dc.get_example_patch()
    >>>
    >>> # Calculate mean along time axis
    >>> patch_time = patch.aggregate("time", method=np.nanmean)
    >>>
    >>> # Calculate median distance along distance dimension
    >>> patch_dist = patch.aggregate("distance", method=np.nanmedian)
    >>>
    >>> # Calculate the mean, and remove the associated dimension
    >>> patch_mean_no_dim = patch.aggregate(
    ...     "time", method="mean", dim_reduce="squeeze"
    ... )
    >>>
    >>> # Aggregate by the min value and keep the mean of the dimension
    >>> patch_mean_min = patch.aggregate(
    ...     "distance", method="min", dim_reduce="mean",
    ... )
    """

    # Loose here and below, so values are recorded as given.
    dim: Any = None
    method: Any = "mean"
    dim_reduce: Any = "empty"

    def _reducer(self) -> Callable:
        """Return the aggregator a method names, or the method itself."""
        return _AGG_FUNCS.get(self.method, self.method)


class _Shortcut(_Reduction):
    """An aggregation whose aggregator is fixed."""

    name = None
    dim: Any = None
    dim_reduce: Any = "empty"


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Min(_Shortcut):
    """
    Calculate the minimum along one or more dimensions.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get minimum along time dimension
    >>> min_patch = patch.min(dim='time')
    >>> assert min_patch.size < patch.size

    {notes}
    """

    reducer = staticmethod(np.nanmin)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Max(_Shortcut):
    """
    Calculate the maximum along one or more dimensions.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get maximum along distance dimension
    >>> max_patch = patch.max(dim='distance')
    >>> assert max_patch.size < patch.size

    {notes}
    """

    reducer = staticmethod(np.nanmax)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Mean(_Shortcut):
    """
    Calculate the mean along one or more dimensions.

    Parameters
    ----------
    {params}

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Get mean along time dimension
    >>> time_mean = patch.mean(dim='time')
    >>> assert time_mean.size < patch.size

    {notes}
    """

    reducer = staticmethod(np.nanmean)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Median(_Shortcut):
    """
    Calculate the median along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    reducer = staticmethod(np.nanmedian)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Std(_Shortcut):
    """
    Calculate the standard deviation along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    reducer = staticmethod(np.nanstd)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class First(_Shortcut):
    """
    Get the first value along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    reducer = staticmethod(_AGG_FUNCS["first"])


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Last(_Shortcut):
    """
    Get the last value along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    reducer = staticmethod(_AGG_FUNCS["last"])


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class Sum(_Shortcut):
    """
    Sum the values along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    reducer = staticmethod(np.nansum)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class AnyTrue(_Shortcut):
    """
    Perform boolean any operation along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    name = "any"
    data_type = ""
    reducer = staticmethod(np.any)


@compose_docstring(params=AGG_DOC_STR, notes=AGG_NOTES)
class AllTrue(_Shortcut):
    """
    Perform boolean all operation along one or more dimensions.

    Parameters
    ----------
    {params}

    {notes}
    """

    name = "all"
    data_type = ""
    reducer = staticmethod(np.all)


IDX_DOC_STR = f"""
patch
    The input Patch.
dim
    The name of the single dimension to reduce. None or a sequence,
    which the other aggregations accept, raises here.
{DIM_REDUCE_DOCS}
"""

IDX_NOTES = """
Notes
-----
- The data become coordinate values rather than the values found there,
  so they take the coordinate's dtype. Use
  [`Patch.max`](`dascore.proc.aggregate.max`) or
  [`Patch.min`](`dascore.proc.aggregate.min`) for the values themselves.

- NaN and NaT samples are skipped. A slice with none left has no
  coordinate to point at, so it yields a null; an integer coordinate is
  widened to float64 to hold one, which loses exactness above 2**53, and
  a coordinate which can hold no null, such as a string one, raises.

- Ties go to the first occurrence, as in NumPy.
"""


def _extreme_index(data, axis, want_max):
    """
    Return the index of the extreme along axis, and the all-missing slices.

    The index of an all-missing slice is arbitrary; the mask says which
    those are so the caller can null them out.
    """
    if dtype_time_like(data.dtype):
        # NaT does not compare, but its int64 view is the smallest int64,
        # so the view orders correctly once the missing are filled away.
        info = np.iinfo(np.int64)
        missing, values = np.isnat(data), data.view(np.int64)
        fill = info.min if want_max else info.max
    else:
        values, fill = data, (-np.inf if want_max else np.inf)
        # Integers and booleans have no value which means "missing".
        inexact = np.issubdtype(data.dtype, np.inexact)
        missing = np.isnan(data) if inexact else None
    if missing is None or not missing.any():
        return (np.argmax if want_max else np.argmin)(values, axis=axis), None
    filled = np.where(missing, fill, values)
    extreme = filled.max(axis=axis) if want_max else filled.min(axis=axis)
    # Match the extreme against the unfilled data rather than taking the
    # arg of the filled data: a real -inf equals the fill for a max, and a
    # first-wins tie would then answer with the missing sample. Nothing
    # missing can match here, since NaN equals nothing and NaT is only the
    # extreme when the whole slice is missing.
    hit = values == np.expand_dims(extreme, axis)
    return hit.argmax(axis=axis), missing.all(axis=axis)


def _fill_empty(values, empty):
    """Put a null where a slice had nothing to point at."""
    dtype = values.dtype
    if not (dtype_time_like(dtype) or np.issubdtype(dtype, np.number)):
        msg = (
            f"A slice with no valid sample has no {dtype} coordinate to "
            "point at. Drop or fill the empty slices first."
        )
        raise ParameterError(msg)
    # An integer coordinate cannot hold NaN, so where widens it to float.
    return np.where(empty, _get_nullish(dtype), values)


@compose_docstring(params=IDX_DOC_STR, notes=IDX_NOTES)
class Idxmax(PatchProcessor):
    """
    Return the coordinate value where the data are largest along a dimension.

    Parameters
    ----------
    {params}

    {notes}

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> patch = dc.get_example_patch()
    >>>
    >>> # The time of each channel's largest sample.
    >>> peak_time = patch.idxmax("time")
    >>>
    >>> # Drop the reduced dimension, as xarray's idxmax does.
    >>> squeezed = patch.idxmax("time", dim_reduce="squeeze")

    See Also
    --------
    - [`Patch.idxmin`](`dascore.proc.aggregate.idxmin`)
    - [`Patch.max`](`dascore.proc.aggregate.max`)
    """

    dim: Any
    dim_reduce: Any = "empty"

    data_type = ""
    # Whether the extreme is the largest.
    _want_max: ClassVar[bool] = True

    def get_metadata(self, meta):
        """Return the reduced metadata, and the axis and whether it stays."""
        name, dim = self.name, self.dim
        if not isinstance(dim, str):
            msg = f"{name} reduces a single dimension; dim must be its name."
            raise ParameterError(msg)
        coord = meta.get_coord(dim)
        if coord._partial:
            # The coord the default dim_reduce leaves behind holds no values,
            # so indexing it would quietly null the whole result.
            msg = (
                f"The '{dim}' coordinate holds no values for {name} to return; "
                "the dimension has already been reduced."
            )
            raise ParameterError(msg)
        out, ((axis, keep),) = _reduce_meta(meta, dim, self.dim_reduce)
        # A time coord's units describe its step, not its magnitude, so
        # labelling nanoseconds "s" would scale any unit maths by a billion.
        units = None if dtype_time_like(coord.dtype) else coord.units
        return out.update_attrs(data_units=units), {"axis": axis, "keep": keep}

    # The kernel finds indices; `reconcile` swaps in the coordinate values.
    def numpy_kernel(self, data, *, axis, keep):
        """Return the index of each extreme, and the empty slices or None."""
        out = _extreme_index(data, axis, self._want_max)
        if keep:
            out = tuple(x if x is None else np.expand_dims(x, axis) for x in out)
        return out

    def reconcile(self, data, out, meta):
        """Return the coordinate values at the extremes."""
        index, empty = data
        values = meta.get_coord(self.dim).values[to_numpy(index)]
        if empty is not None:
            values = _fill_empty(values, to_numpy(empty))
        if not is_numpy(index):
            values = asarray_like(values, index)
        return out.to_patch(values)


@compose_docstring(params=IDX_DOC_STR, notes=IDX_NOTES)
class Idxmin(Idxmax):
    """
    Return the coordinate value where the data are smallest along a dimension.

    Parameters
    ----------
    {params}

    {notes}

    Examples
    --------
    >>> import dascore as dc
    >>>
    >>> trough_time = dc.get_example_patch().idxmin("time")

    See Also
    --------
    - [`Patch.idxmax`](`dascore.proc.aggregate.idxmax`)
    - [`Patch.min`](`dascore.proc.aggregate.min`)
    """

    _want_max = False
