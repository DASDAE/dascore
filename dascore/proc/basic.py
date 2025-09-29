"""Basic operations for patches."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy.fft import next_fast_len

import dascore as dc
from dascore.compat import array
from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import ParameterError
from dascore.utils.array import _apply_binary_ufunc
from dascore.utils.misc import _get_nullish
from dascore.utils.models import ArrayLike
from dascore.utils.patch import (
    align_patch_coords,
    get_dim_axis_value,
    patch_function,
)


def set_dims(self: PatchType, **kwargs: str) -> PatchType:
    """
    Set dimension to non-dimensional coordinate.

    Parameters
    ----------
    **kwargs
        A mapping indicating old_dim: new_dim where new_dim refers to
        the name of a non-dimensional coordinate which will become a
        dimensional coordinate. The old dimensional coordinate will
        become a non-dimensional coordinate.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # add new coordinate, random numbers length of time dim
    >>> my_coord = np.random.random(patch.coord_shapes["time"])
    >>> out = (
    ...    patch.update_coords(my_coord=("time", my_coord))  # add my_coord
    ...    .set_dims(time="my_coord") # set mycoord as dim (rather than time)
    ... )
    >>> assert "my_coord" in out.dims
    """
    cm = self.coords.set_dims(**kwargs)
    return self.new(coords=cm)


def pipe(
    self: PatchType, func: Callable[[PatchType, ...], PatchType], *args, **kwargs
) -> PatchType:
    """
    Pipe the patch to a function.

    This is primarily useful for maintaining a chain of patch calls for
    a function.

    Parameters
    ----------
    func
        The function to pipe the patch. It must take a patch instance as
        the first argument followed by any number of positional or keyword
        arguments, then return a patch.
    *args
        Positional arguments that get passed to func.
    **kwargs
        Keyword arguments passed to func.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Define a custom function that squares the data
    >>> def square_data(patch):
    ...     return patch.new(data=patch.data ** 2)
    >>>
    >>> # Use pipe to apply the function
    >>> squared = patch.pipe(square_data)
    >>>
    >>> # Can also chain with other methods
    >>> result = patch.pipe(square_data).mean(dim="time")
    """
    return func(self, *args, **kwargs)


def update_attrs(self: PatchType, **attrs) -> PatchType:
    """
    Update attrs and return a new Patch.

    Parameters
    ----------
    **attrs
        attrs to add/update.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Update existing attributes
    >>> updated = patch.update_attrs(instrument_type="DAS")
    >>>
    >>> # Add new custom attributes
    >>> with_custom = patch.update_attrs(processing_date="2024-01-01")
    """

    def _fast_attr_update(self, attrs):
        """A fast method for just squashing the attrs and returning new patch."""
        new = self.__new__(self.__class__)
        new._data = self.data
        new._attrs = attrs
        new._coords = self.coords
        return new

    # since we update history so often, we make a fast track for it.
    new_attrs = self.attrs.model_dump(exclude_unset=True)
    new_attrs.update(attrs)
    # pop out coords so new coords has priority.
    if len(attrs) == 1 and "history" in attrs:
        return _fast_attr_update(self, PatchAttrs(**new_attrs))
    new_coords, new_attrs = self.coords.update_from_attrs(new_attrs)
    out = dict(coords=new_coords, attrs=new_attrs, dims=self.dims)
    return self.__class__(self.data, **out)


def equals(self: PatchType, other: Any, only_required_attrs=True, close=False) -> bool:
    """
    Determine if the current patch equals another.

    Parameters
    ----------
    other
        A Patch (could be equal) or some other type (not equal)
    only_required_attrs
        If True, only compare required attributes. This helps avoid issues
        with comparing histories or custom attrs of patches, for example.
    close
        If True, the data can be "close" using np.allclose, otherwise
        all data must be equal.

    Examples
    --------
    >>> import dascore as dc
    >>> patch1 = dc.get_example_patch()
    >>> patch2 = dc.get_example_patch()
    >>>
    >>> # Test equality
    >>> are_equal = patch1.equals(patch2)
    >>>
    >>> # Test with modified patch
    >>> modified = patch1.update_attrs(custom_attr="test")
    >>> equal_ignoring_custom = patch1.equals(modified, only_required_attrs=True)
    >>>
    >>> # Test with close comparison for numerical data
    >>> noisy = patch1.new(data=patch1.data + 1e-10)
    >>> close_equal = patch1.equals(noisy, close=True)
    """
    # different types are not equal
    if not isinstance(other, type(self)):
        return False
    # Different coords are not equal; can pop out coords from attrs
    if not self.coords == other.coords:
        return False
    if only_required_attrs:  # only include default fields
        attrs_to_compare = set(PatchAttrs.model_fields) - {"history", "coords"}
        attrs1 = self.attrs.model_dump(include=attrs_to_compare)
        attrs2 = other.attrs.model_dump(include=attrs_to_compare)
    else:  # include all fields but coords
        attrs1 = self.attrs.model_dump(exclude=["coords"])
        attrs2 = other.attrs.model_dump(exclude=["coords"])
    if set(attrs1) != set(attrs2):  # attrs don't have same keys; not equal
        return False
    if attrs1 != attrs2:
        # see if some values are NaNs, these should be counted equal
        not_equal = {
            x
            for x in attrs1
            if attrs1[x] != attrs2[x]
            and not (pd.isnull(attrs1[x]) and pd.isnull(attrs2[x]))
        }
        if not_equal:
            return False
    # Test data equality or proximity.
    if close and not np.allclose(self.data, other.data):
        return False
    elif not close and not np.equal(self.data, other.data).all():
        return False
    return True


def bool_patch(self: PatchType):
    """
    Get the boolean value of a patch.

    This follows the NumPy convention of raising a ValueError if the patch
    has more than one element. Otherwise, it returns the truthiness of the
    one element.
    """
    return bool(self.data)


def update(
    self: PatchType,
    data: ArrayLike | np.ndarray | None = None,
    coords: None | dict[str | Sequence[str], ArrayLike] | CoordManager = None,
    dims: Sequence[str] | None = None,
    attrs: Mapping | PatchAttrs | None = None,
) -> PatchType:
    """
    Return a copy of the Patch with updated data, coords, dims, or attrs.

    Parameters
    ----------
    data
        An array-like containing data, an xarray DataArray object, or a Patch.
    coords
        The coordinates, or dimensional labels for the data. These can be
        passed in three forms:
        {coord_name: data}
        {coord_name: ((dimensions,), data)}
        {coord_name: (dimensions, data)}
    dims
        A sequence of dimension strings. The first entry corresponds to the
        first axis of data, the second to the second dimension, and so on.
    attrs
        Optional attributes (non-coordinate metadata) passed as a dict.

    Notes
    -----
    - If both coords and attrs are defined, attrs will have priority.
    """
    data = data if data is not None else self.data
    coords = coords if coords is not None else self.coords
    if dims is None:
        dims = coords.dims if isinstance(coords, CoordManager) else self.dims
    coords = get_coord_manager(coords, dims)
    if attrs is not None:
        coords, attrs = coords.update_from_attrs(attrs)
    else:
        _attrs = dc.PatchAttrs.from_dict(attrs or self.attrs)
        attrs = _attrs.update(coords=coords)
    return self.__class__(data=data, coords=coords, attrs=attrs)


@patch_function()
def abs(patch: PatchType) -> PatchType:
    """
    Take the absolute value of the patch data.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> out = pa.abs() # take absolute value of generated example patch data
    """
    return patch.new(data=np.abs(patch.data))


@patch_function()
def conj(patch: PatchType) -> PatchType:
    """
    Apply the complex conjugate of the patch data.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>>
    >>> # Example 1
    >>> dft = pa.dft(None)  # multi-dim dft
    >>> conj = dft.conj()
    """
    return patch.new(data=np.conj(patch.data))


@patch_function()
def real(patch: PatchType) -> PatchType:
    """
    Return a new patch with the real part of the data array.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> out = pa.real()
    """
    return patch.new(data=np.real(patch.data))


@patch_function()
def imag(patch: PatchType) -> PatchType:
    """
    Return a new patch with the imaginary part of the data array.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> out = pa.imag()
    """
    return patch.new(data=np.imag(patch.data))


@patch_function()
def angle(patch: PatchType) -> PatchType:
    """
    Return a new patch with the phase angles from the data array.

    Examples
    --------
    >>> import dascore
    >>> pa = dascore.get_example_patch()
    >>> out = pa.angle()
    """
    return patch.new(data=np.angle(patch.data))


@patch_function()
def normalize(
    self: PatchType,
    dim: str,
    norm: Literal["l1", "l2", "max", "bit"] = "l2",
) -> PatchType:
    """
    Normalize a patch along a specified dimension.

    Parameters
    ----------
    dim
        The dimension along which the normalization takes place.
    norm
        Determines the value to divide each sample by along a given axis.
        Options are:
            l1 - divide each sample by the l1 of the axis.
            l2 - divide each sample by the l2 of the axis.
            max - divide each sample by the maximum of the absolute value of the axis.
            bit - sample-by-sample normalization (-1/+1)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # L2 normalization along time dimension
    >>> l2_norm = patch.normalize(dim="time", norm="l2")
    >>>
    >>> # Max normalization along distance dimension
    >>> max_norm = patch.normalize(dim="distance", norm="max")
    >>>
    >>> # Bit normalization (sign only)
    >>> bit_norm = patch.normalize(dim="time", norm="bit")
    """
    axis = self.get_axis(dim)
    data = self.data
    if norm in {"l1", "l2"}:
        order = int(norm[-1])
        norm_values = np.linalg.norm(self.data, axis=axis, ord=order)
    elif norm == "max":
        norm_values = np.max(data, axis=axis)
    elif norm == "bit":
        pass
    else:
        msg = (
            f"Norm value of {norm} is not supported. "
            f"Supported values are {('l1', 'l2', 'max', 'bit')}"
        )
        raise ValueError(msg)
    if norm == "bit":
        new_data = np.divide(
            data, np.abs(data), out=np.zeros_like(data), where=np.abs(data) != 0
        )
    else:
        expanded_norm = np.expand_dims(norm_values, axis=axis)
        new_data = np.divide(
            data, expanded_norm, out=np.zeros_like(data), where=expanded_norm != 0
        )
    return self.new(data=new_data)


@patch_function()
def standardize(
    self: PatchType,
    dim: str,
) -> PatchType:
    """
    Standardize data by removing the mean and scaling to unit variance.

    The standard score of a sample x is calculated as:

    z = (x - u) / s
    where u is the mean of the training samples or zero if with_mean=False,
    and s is the standard deviation of the training samples or one if with_std=False.

    Parameters
    ----------
    dim
        The dimension along which the normalization takes place.

    Examples
    --------
    ```{python}
    import dascore as dc

    patch = dc.get_example_patch()

    # standardize along the time axis
    standardized_time = patch.standardize('time')

    # standardize along the x axis
    standardized_distance = patch.standardize('distance')
    ```
    """
    axis = self.get_axis(dim)
    data = self.data
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    new_data = (data - mean) / std
    return self.new(data=new_data)


# This is left here to not break compatibility. It also forces `apply_ufunc`
# to be imported in to this module where it used to live. In the rare chance
# a user tries to access it directly from here it will still work.
apply_operator = _apply_binary_ufunc


@patch_function()
def dropna(
    patch: PatchType,
    dim,
    how: Literal["any", "all"] = "any",
    include_inf=True,
) -> PatchType:
    """
    Return a patch with nullish values dropped along dimension.

    Parameters
    ----------
    patch
        The patch which may contain nullish values.
    dim
        The dimension along which to drop nullish values.
    how
        "any" or "all". If "any" drop label if any null values.
        If "all" drop label if all values are nullish.
    include_inf
        If True, drop all non-finite values.

    Notes
    -----
    When include_inf is False, "nullish" is defined by `pandas.isnull`.
    When include_inf is True (default), "nullish" includes non-finite values
    (NaN, inf, -inf) as determined by `numpy.isfinite`

    Examples
    --------
    >>> import dascore as dc
    >>> # load an example patch which has some NaN values.
    >>> patch = dc.get_example_patch("patch_with_null")
    >>> # drop all time labels that have a single null value
    >>> out = patch.dropna("time", how="any")
    >>> # drop all distance labels that have all null values
    >>> out = patch.dropna("distance", how="all")
    """
    axis = patch.get_axis(dim)
    func = np.any if how == "any" else np.all
    if include_inf:
        to_drop = ~np.isfinite(patch.data)
    else:
        to_drop = pd.isnull(patch.data)
    # need to iterate each non-dim axis and collapse with func
    axes = set(range(len(patch.shape))) - {axis}
    to_drop = func(to_drop, axis=tuple(axes))
    to_keep = ~to_drop
    assert len(to_keep.shape) == 1
    assert to_keep.shape[0] == patch.data.shape[axis]
    # get slices for trimming data.
    slices = [slice(None)] * len(patch.dims)
    slices[axis] = to_keep
    new_data = patch.data[tuple(slices)]
    coord = patch.get_coord(dim)
    cm = patch.coords.update(**{dim: coord[to_keep]})
    attrs = patch.attrs.update(coords={})
    return patch.new(data=new_data, coords=cm, attrs=attrs)


@patch_function()
def fillna(patch: PatchType, value, include_inf=True) -> PatchType:
    """
    Return a patch with nullish values replaced by a value.

    Parameters
    ----------
    patch
        The patch which may contain nullish values.
    value
        The value to replace nullish values with.
    include_inf
        If True, also fill all non-finite values.

    Notes
    -----
    When include_inf is False, "nullish" is defined by `pandas.isnull`.
    When include_inf is True (default), "nullish" includes non-finite values
    (NaN, inf, -inf) as determined by `numpy.isfinite`

    Examples
    --------
    >>> import dascore as dc
    >>> # load an example patch which has some NaN values.
    >>> patch = dc.get_example_patch("patch_with_null")
    >>>
    >>> # Replace all occurrences of NaN with 0
    >>> out = patch.fillna(0)
    >>>
    >>> # Replace all occurrences of NaN with 5
    >>> out = patch.fillna(5)
    """
    if include_inf:
        to_replace = ~np.isfinite(patch.data)
    else:
        to_replace = pd.isnull(patch.data)
    new_data = patch.data.copy()
    new_data[to_replace] = value

    return patch.new(data=new_data)


@patch_function()
def pad(
    patch: PatchType,
    mode: Literal["constant"] = "constant",
    constant_values: Any = 0,
    expand_coords=True,
    samples=False,
    **kwargs,
) -> PatchType:
    """
    Pad the patch data along specified dimensions.

    Parameters
    ----------
    mode : str, optional
        The mode of padding, by default 'constant'.
    constant_values : scalar , optional
        A single scalar value used as the padding value across all dimensions.
        Defaults to 0.
    expand_coords : bool, optional
        Determines how coordinates are adjusted when padding is applied.
        If set to True, the coordinates will be expanded to maintain their
        order and even sampling (if evenly sampled), by extrapolating
        based on the coordinate's step size. If set to False, or coordinate
        is not evenly sampled, the new coordinates introduced by padding
        will be padded with NaN values.
    **kwargs:
        Used to specify dimension and number of elements,
        either an integer or a tuple (before, after).
        In addition, the following strings are supported:

        "fft" - pad to the next fast fft length along the given dimension by
        adding values to the end of the axis.

        "correlate" - prepare the coordinate for correlation/convolution in
        the frequency domain by padding to the next fast fft length after
        2*n - 1 where n is the current dimension length by adding values
        to the end of the axis.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Zero pad `time` dimension with 2 patch's time unit (e.g., sec)
    >>> # zeros before and 3 zeros after
    >>> padded_patch_1 = patch.pad(time=(2, 3))
    >>>
    >>> # Pad `distance` dimension with 1s 4 samples before and 4 after.
    >>> padded_patch_3 = patch.pad(distance=4, constant_values=1, samples=True)
    >>>
    >>> # Get patch ready for fast fft along time dimension.
    >>> padded_fft = patch.pad(time="fft")
    """

    def _get_pad_tuple(value, samples, coord):
        """
        Get a tuple, in samples, of (pad_to_start, pad_to_end).
        """
        if value in {"fft", "correlate"}:
            target_length = len(coord) if value == "fft" else 2 * len(coord) - 1
            # Determine value so that the output dim will be a fast length.
            value = (0, next_fast_len(target_length) - len(coord))
            samples = True  # ensure padding isn't interpreted as coord units.
        elif not isinstance(value, Sequence):
            value = (value, value)
        if not samples:  # Ensure values are in samples.
            value = tuple(coord.get_sample_count(x) for x in value)
        return value

    def _get_new_coord(coord, pad_tuple, expand_coords):
        """Get the new coordinate along the expanded axis."""
        if expand_coords and coord.evenly_sampled:
            new_start = coord.min() - pad_tuple[0] * coord.step
            new_end = coord.max() + (pad_tuple[1] + 1) * coord.step
            assert coord.evenly_sampled, "expand_coords requires evenly sampled."
            new_coord = get_coord(
                start=new_start, stop=new_end, step=coord.step, units=coord.units
            )
        else:
            old_values = coord.values
            # Need to convert ints to float so NaN can be used.
            if np.issubdtype(old_values.dtype, np.integer):
                old_values = old_values.astype(np.float64)
            null_value = _get_nullish(old_values.dtype)
            added_nan_values = np.pad(
                old_values, pad_width=pad_tuple, constant_values=null_value
            )
            new_coord = coord.update(data=added_nan_values)
        return new_coord

    if isinstance(constant_values, Sequence):
        raise ParameterError("constant_values must be a scalar, not a sequence.")

    pad_width = [(0, 0)] * len(patch.shape)
    dimfo = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    new_coords = {}

    for dim, axis, value in dimfo:
        coord = patch.get_coord(dim, require_evenly_sampled=not samples)
        pad_tuple = _get_pad_tuple(value, samples, coord)
        pad_width[axis] = pad_tuple
        new_coords[dim] = _get_new_coord(coord, pad_tuple, expand_coords)

    # Pad data, update coord manager, and return.
    new_data = np.pad(patch.data, pad_width, mode=mode, constant_values=constant_values)
    new_coords = patch.coords.update(**new_coords)
    return patch.new(data=new_data, coords=new_coords)


@patch_function()
def roll(patch, samples=False, update_coord=False, **kwargs):
    """
    Roll patch array elements along a given dimension.

    Parameters
    ----------
    patch
        input patch
    samples
        if True, value indicates coordinate or value of dimension
    update_coord
        if True, updates coord based on rolled amount
    **kwargs
        specifies dimension and number of elements to roll

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # roll time dimension 5 elements
    >>> rolled_patch = patch.roll(time=5, samples=True)
    >>>
    >>> # roll distance dimension 30 meters(or units of distance in patch)
    >>> rolled_patch2 = patch.roll(distance=30, samples=False)
    >>>
    >>> # roll time dimension 5 elements and update coordinates
    >>> rolled_patch3 = patch.roll(time=5, samples=True, update_coord=True)
    """
    dim, axis, input_value = get_dim_axis_value(patch, kwargs=kwargs)[0]
    arr = patch.data
    coord = patch.get_coord(dim)
    value = coord.get_sample_count(input_value, samples=samples)

    roll_arr = np.roll(arr, value, axis=axis)

    # update coords if True
    if update_coord:
        roll_coord_arr = np.roll(coord.values, value)
        new_coord = coord.update(values=roll_coord_arr)
        patch = patch.update_coords(**{dim: new_coord})

    return patch.new(data=roll_arr)


@patch_function()
def where(
    patch: PatchType, cond: ArrayLike | PatchType, other: Any | PatchType = np.nan
) -> PatchType:
    """
    Return elements from patch where condition is True, else fill with other.

    Parameters
    ----------
    patch
        The input patch
    cond
        Condition array. Should be a boolean array with the same shape as patch data,
        or a patch with boolean data that is broadcastable to the patch's shape.
    other
        Value to use for locations where cond is False. Can be a scalar value,
        array, or patch that is broadcastable to the patch's shape. Default is NaN.

    Returns
    -------
    PatchType
        A new patch with values from patch where cond is True, and other elsewhere.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Where data > 0 fill with original patch values else nan.
    >>> condition = patch.data > 0
    >>> out = patch.where(condition)
    >>>
    >>> # Use another patch as condition
    >>> threshold = patch.data.mean()
    >>> boolean_patch = patch.new(data=(patch.data > threshold))
    >>> out = patch.where(boolean_patch, other=0)
    >>>
    >>> # Replace values below threshold with 0
    >>> out = patch.where(patch.data > patch.data.mean(), other=0)
    """
    cls = patch.__class__  # Use this so it works with subclasses
    # Align patch and cond
    if isinstance(cond, cls):
        patch, cond = align_patch_coords(patch, cond)
    # Align patch and other, may need to re-align cond
    if isinstance(other, cls):
        patch, other = align_patch_coords(patch, other)
        if isinstance(cond, cls):
            patch, cond = align_patch_coords(patch, cond)

    cond_array, other_array = array(cond), array(other)

    # Ensure condition is boolean
    if not np.issubdtype(cond_array.dtype, np.bool_):
        msg = "Condition must be a boolean array or patch with boolean data"
        raise ValueError(msg)

    # Use numpy.where to apply condition
    new_data = np.where(cond_array, patch.data, other_array)
    return patch.new(data=new_data)


@patch_function()
def flip(patch, *dims, flip_coords=True):
    """
    Flip patch data and (optionally coords) along specified dimensions.

    Parameters
    ----------
    patch
        The patch to flip.
    *dims
        The dimensions over which to flip (mirror).
    flip_coords
        If True, also flip coords associated with dimensions, otherwise
        leave them unchanged.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Flip patch over time axis
    >>> out = patch.flip("time")
    >>> assert np.all(patch.get_array("time") == out.get_array("time")[::-1])
    >>>
    >>> # Flip patch over all dimensions.
    >>> out = patch.flip(*patch.dims)
    """
    if not dims:
        return patch  # no-op
    axes = tuple(patch.get_axis(name) for name in dims)
    data = np.flip(patch.data, axis=axes) if dims else patch.data
    coords = patch.coords.flip(*dims) if flip_coords else patch.coords
    return patch.new(data=data, coords=coords)
