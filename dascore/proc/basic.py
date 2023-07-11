"""
Basic operations for patches.
"""
from typing import Literal

import numpy as np

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import UnitError
from dascore.units import DimensionalityError, Quantity, Unit, get_quantity
from dascore.utils.patch import merge_compatible_coords_attrs, patch_function


@patch_function()
def abs(patch: PatchType) -> PatchType:
    """
    Take the absolute value of the patch data.

    Examples
    --------
    >>> import dascore # import dascore library
    >>> pa = dascore.get_example_patch() # generate example patch
    >>> out = dascore.proc.abs(pa) # take absolute value of generated example patch data
    """
    new_data = np.abs(patch.data)
    return patch.new(data=new_data)


@patch_function()
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
def rename_coords(self: PatchType, **kwargs) -> PatchType:
    """
    Rename coordinate of Patch.

    Parameters
    ----------
    **kwargs
        The mapping from old names to new names

    Examples
    --------
    >>> from dascore.examples import get_example_patch
    >>> pa = get_example_patch()
    >>> # rename dim "distance" to "fragrance"
    >>> pa2 = pa.rename_coords(distance='fragrance')
    >>> assert 'fragrance' in pa2.dims
    """
    new_coord = self.coords.rename_coord(**kwargs)
    attrs = self.attrs.rename_dimension(**kwargs)
    return self.new(coords=new_coord, dims=new_coord.dims, attrs=attrs)


@patch_function()
def update_coords(self: PatchType, **kwargs) -> PatchType:
    """
    Update the coordiantes of a patch.

    Will either add new coordinates, or update existing ones.

    Parameters
    ----------
    **kwargs
        The mapping from old names to new names

    Examples
    --------
    >>> from dascore.examples import get_example_patch
    >>> pa = get_example_patch()
    >>> # rename dim "distance" to "fragrance"
    >>> pa2 = pa.rename_coords(distance='fragrance')
    >>> assert 'fragrance' in pa2.dims
    """
    new_coord = self.coords.update_coords(**kwargs)
    attrs = self.attrs.rename_dimension(**kwargs)
    return self.new(coords=new_coord, dims=new_coord.dims, attrs=attrs)


@patch_function()
def normalize(
    self: PatchType,
    dim: str,
    norm: Literal["l1", "l2", "max"] = "l2",
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
    """
    axis = self.dims.index(dim)
    data = self.data
    if norm in {"l1", "l2"}:
        order = int(norm[-1])
        norm_values = np.linalg.norm(self.data, axis=axis, ord=order)
    elif norm == "max":
        norm_values = np.max(data, axis=axis)
    else:
        msg = (
            f"Norm value of {norm} is not supported. "
            f"Supported values are {('l1', 'l2', 'max')}"
        )
        raise ValueError(msg)
    new_data = data / np.expand_dims(norm_values, axis=axis)
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

    patch = dc.load_example_patch()

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
def integrate(patch: PatchType, dim: str) -> PatchType:
    """
    Integrate along a specified dimension.

    Parameters
    ----------
    patch
        Patch object for integration.
    dim
        The dimension along which to integrate.
    """


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
    >>> new = patch - patch
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
    if isinstance(other, (Quantity, Unit)):
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
