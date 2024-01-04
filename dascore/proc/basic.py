"""Basic operations for patches."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs, merge_compatible_coords_attrs
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.exceptions import UnitError
from dascore.units import DimensionalityError, Quantity, Unit, get_quantity
from dascore.utils.models import ArrayLike
from dascore.utils.patch import patch_function


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
    """
    return func(self, *args, **kwargs)


def update_attrs(self: PatchType, **attrs) -> PatchType:
    """
    Update attrs and return a new Patch.

    Parameters
    ----------
    **attrs
        attrs to add/update.
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


def equals(self: PatchType, other: Any, only_required_attrs=True) -> bool:
    """
    Determine if the current patch equals another.

    Parameters
    ----------
    other
        A Patch (could be equal) or some other type (not equal)
    only_required_attrs
        If True, only compare required attributes. This helps avoid issues
        with comparing histories or custom attrs of patches, for example.
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

    return np.equal(self.data, other.data).all()


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
    if attrs:
        coords, attrs = coords.update_from_attrs(attrs)
    else:
        _attrs = dc.PatchAttrs.from_dict(attrs or self.attrs)
        attrs = _attrs.update(coords=coords, dims=coords.dims)
    return self.__class__(data=data, coords=coords, attrs=attrs, dims=coords.dims)


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


@patch_function(history=None)
def transpose(self: PatchType, *dims: str) -> PatchType:
    """
    Transpose the data array to any dimension order desired.

    Parameters
    ----------
    *dims
        Dimension names which define the new data axis order.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> # transpose the time and data array dimensions in the example patch
    >>> out = dascore.proc.transpose(pa,"time", "distance")
    """
    dims = tuple(dims)
    old_dims = self.coords.dims
    new_coord = self.coords.transpose(*dims)
    new_dims = new_coord.dims
    axes = tuple(old_dims.index(x) for x in new_dims)
    new_data = np.transpose(self.data, axes)
    return self.new(data=new_data, coords=new_coord)


@patch_function()
def squeeze(self: PatchType, dim=None) -> PatchType:
    """
    Return a new object with len one dimensions flattened.

    Parameters
    ----------
    dim
        Selects a subset of the length one dimensions. If a dimension
        is selected with length greater than one, an error is raised.
        If None, all length one dimensions are squeezed.
    """
    coords = self.coords.squeeze(dim)
    axis = None if dim is None else self.coords.dims.index(dim)
    data = np.squeeze(self.data, axis=axis)
    return self.new(data=data, coords=coords)


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
    """
    axis = self.dims.index(dim)
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
    axis = self.dims.index(dim)
    data = self.data
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.std(data, axis=axis, keepdims=True)
    new_data = (data - mean) / std
    return self.new(data=new_data)


@patch_function()
def apply_operator(patch: PatchType, other, operator) -> PatchType:
    """
    Apply a ufunc-type operator to a patch.

    This is used to implement a patch's operator overload.

    Parameters
    ----------
    patch
        The patch instance.
    other
        The other object to apply the operator element-wise. Must be either a
        non-patch which is broadcastable to the shape of the patch's data, or
        a patch which has compatible coordinates. If units are provided they
        must be compatible.
    operator
        The operator. Must be numpy ufunc-like.

    Examples
    --------
    >>> import numpy as np
    >>> import dascore as dc
    >>> from dascore.proc.basic import apply_operator
    >>> patch = dc.get_example_patch()
    >>> # multiply the patch by 10
    >>> new = apply_operator(patch, 10, np.multiply)
    >>> assert np.allclose(patch.data * 10, new.data)
    >>> # add a random value to each element of patch data
    >>> noise = np.random.random(patch.shape)
    >>> new = apply_operator(patch, noise, np.add)
    >>> assert np.allclose(new.data, patch.data + noise)
    >>> # subtract one patch from another. Coords and attrs must be compatible
    >>> new = apply_operator(patch, patch, np.subtract)
    >>> assert np.allclose(new.data, 0)
    """
    if isinstance(other, dc.Patch):
        coords, attrs = merge_compatible_coords_attrs(patch, other)
        other = other.data
        if other_units := get_quantity(attrs.data_units):
            other = other * other_units
    else:
        coords, attrs = patch.coords, patch.attrs
    # handle units of output
    if isinstance(other, Quantity | Unit):
        data_units = get_quantity(attrs.data_units)
        data = patch.data if data_units is None else patch.data * data_units
        # other is not numpy array wrapped w/ quantity, convert to quant
        if not hasattr(other, "shape"):
            other = get_quantity(other)
        try:
            new_data_w_units = operator(data, other)
        except DimensionalityError:
            msg = f"{operator} failed with units {data_units} and {other.units}"
            raise UnitError(msg)
        attrs = attrs.update(data_units=str(new_data_w_units.units))
        new_data = new_data_w_units.magnitude
    else:  # simpler case; no units.
        new_data = operator(patch.data, other)
    new = patch.new(data=new_data, coords=coords, attrs=attrs)
    return new


@patch_function()
def dropna(patch: PatchType, dim, how: Literal["any", "all"] = "any") -> PatchType:
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

    Notes
    -----
    "nullish" is defined by `pandas.isnull`.

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
    axis = patch.dims.index(dim)
    func = np.any if how == "any" else np.all
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
