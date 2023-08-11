"""Module to transform a Patch into spectrograms."""
from __future__ import annotations

from operator import mul

from scipy.signal import spectrogram as scipy_spectrogram

from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.utils.misc import iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)
from dascore.utils.time import (
    is_datetime64,
    is_timedelta64,
    to_datetime64,
    to_timedelta64,
)
from dascore.utils.transformatter import FourierTransformatter


def _get_new_original_coord(old_coord, array):
    """Get a new coordinate for original axis (eg time)."""
    conversion_tuples = (
        (is_datetime64, to_datetime64),
        (is_timedelta64, to_timedelta64),
    )
    # ensure the output is in the right data format.
    for predicate, func in conversion_tuples:
        if predicate(old_coord.min()):
            array = func(array)

    out = get_coord(values=array, units=old_coord.units)
    return out


def _get_transformed_coord(coord, freqs):
    """Get the transformed coordinates."""
    units = 1 / coord.units if coord.units is not None else None
    return get_coord(values=freqs, units=units)


def _get_new_attrs(patch, cm, dim):
    """Update attributes."""
    new = dict(patch.attrs)
    new["dims"] = cm.dims
    new["data_units"] = _get_data_units_from_dims(patch, dim, mul)
    return PatchAttrs(**new)


def _get_new_dims(patch, dim, new_coord_name):
    """
    Get a new dimension tuple.

    The new dimension always takes the place of the transformed dimension,
    and the transformed dimension is appended to the end.
    EG, ("time", "distance") dimensions become ("ft_time", "distance", "time").
    """
    dims = list(patch.dims)
    dims[dims.index(dim)] = new_coord_name
    return tuple([*dims, dim])


@patch_function()
def spectrogram(patch: PatchType, dim: str = "time", **kwargs) -> PatchType:
    """
    Calculate a spectrogram from the patch data.

    The output patch will have one more dimensions than the input patch.

    Parameters
    ----------
    patch
        The input patch
    dim
        The dimension along which the spectrograms are calculated.
    **kwargs
        Passed to `scipy.signal.spectrogram` to control spectrogram options.
        See its documentation for options.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # calculate spectrogram along time axis
    >>> time_spec = patch.tran.spectrogram("time")
    >>> # note the new dimensions
    >>> print(time_spec.dims)
    ('distance', 'ft_time', 'time')
    >>> # perform fourier transforms along distance dimensions
    >>> dist_spec = patch.tran.spectrogram("distance")
    """
    assert len(iterate(dim)) == 1, "only one dimension allowed."
    coord = patch.get_coord(dim)
    dxs, axes = _get_dx_or_spacing_and_axes(patch, dim, require_evenly_spaced=True)
    new_coord_name = FourierTransformatter().rename_dims(dim)[0]
    out_coords = patch.coords._get_dim_array_dict()
    # returns frequency, new values for original dimension (eg time) and spectrogram
    freqs, original, spec = scipy_spectrogram(
        patch.data,
        fs=dxs[0],
        axis=axes[0],
        **kwargs,
    )
    # add new coordinates
    out_coords[dim] = _get_new_original_coord(coord, original)
    out_coords[new_coord_name] = _get_transformed_coord(coord, freqs)
    new_dims = _get_new_dims(patch, dim, new_coord_name)
    cm = get_coord_manager(out_coords, dims=tuple(new_dims))
    attrs = _get_new_attrs(patch, cm, dim)
    return patch.__class__(data=spec, attrs=attrs, coords=cm)
