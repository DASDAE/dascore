"""A spectrogram visualization."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram as scipy_spectrogram

from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_compatible_values, get_coord
from dascore.utils.misc import iterate
from dascore.utils.patch import patch_function
from dascore.utils.transformatter import FourierTransformatter


def _get_other_dim(dim, dims):
    if not isinstance(dim, str):
        raise TypeError(f"Expected 'dim' to be a string, got {type(dim).__name__}.")
    if dim not in dims:
        raise ValueError(f"The dimension '{dim}' is not in patch's dimensions {dims}.")
    if len(dims) == 1:
        return None
    else:
        return dims[0] if dims[1] == dim else dims[1]


def _get_new_original_coord(old_coord, array):
    """Get a new coordinate for original axis (eg time)."""
    old_min = get_compatible_values(old_coord.min(), array.dtype)
    is_dt = np.issubdtype(old_coord.dtype, np.datetime64)
    val_dtype = np.timedelta64 if is_dt else old_coord.dtype
    vals = get_compatible_values(array, val_dtype)
    return get_coord(data=old_min + vals, units=old_coord.units)


def _get_transformed_coord(coord, freqs):
    """Get the transformed coordinates."""
    units = 1 / coord.units if coord.units is not None else None
    return get_coord(data=freqs, units=units)


def _get_new_dims(patch, dim, new_coord_name):
    """Get the new dimension tuple."""
    dims = list(patch.dims)
    dims[dims.index(dim)] = new_coord_name
    return tuple([*dims, dim])


def _spectrogram_patch(patch: PatchType, dim: str = "time", **kwargs) -> PatchType:
    """Create a spectrogram patch for visualization."""
    from dascore.utils.patch import (
        _get_data_units_from_dims,
        _get_dx_or_spacing_and_axes,
    )

    assert len(iterate(dim)) == 1, "only one dimension allowed."
    coord = patch.get_coord(dim)
    dxs, axes = _get_dx_or_spacing_and_axes(patch, dim, require_evenly_spaced=True)
    new_coord_name = FourierTransformatter().rename_dims(dim)[0]
    out_coords = patch.coords._get_dim_array_dict()
    freqs, original, spec = scipy_spectrogram(
        patch.data,
        fs=1 / dxs[0],
        axis=axes[0],
        **kwargs,
    )
    out_coords[dim] = _get_new_original_coord(coord, original)
    out_coords[new_coord_name] = _get_transformed_coord(coord, freqs)
    new_dims = _get_new_dims(patch, dim, new_coord_name)
    cm = get_coord_manager(out_coords, dims=tuple(new_dims))
    attrs = dict(patch.attrs)
    attrs["dims"] = cm.dims
    attrs["data_units"] = _get_data_units_from_dims(patch, dim, np.multiply)
    return patch.__class__(data=spec, attrs=PatchAttrs(**attrs), coords=cm)


@patch_function()
def spectrogram(
    patch: PatchType,
    ax: plt.Axes | None = None,
    dim="time",
    aggr_domain="frequency",
    cmap="bwr",
    scale: float | Sequence[float] | None = None,
    scale_type: Literal["relative", "absolute"] = "relative",
    log=False,
    show=False,
    **kwargs,
) -> plt.Axes:
    """
    Plot a spectrogram of a patch.

    Parameters
    ----------
    patch : PatchType
        The Patch object.
    ax : matplotlib.axes.Axes or None, optional
        A matplotlib axis object. If None, creates a new axis.
    dim : str, optional
        Dimension along which the spectrogram is being plotted.
        Default is "time".
    aggr_domain : str, optional
        "time" or "frequency" in which the mean value of the other
        dimension is calculated. No need to specify if the other
        dimension's coordinate size is 1. Default is "frequency".
    cmap : str or matplotlib.colors.Colormap, optional
        A matplotlib colormap string or instance. Set to None to not plot the
        colorbar. Default is "bwr".
    scale : float, tuple of floats, or None, optional
        If not None, controls the saturation level of the colorbar.
        Values can be a single float or a length-2 tuple specifying upper
        and lower limits. See `scale_type` for more details.
    scale_type : {"relative", "absolute"}, optional
        Specifies the type of scaling:
            - "relative": Scale based on half the dynamic range in the patch.
            - "absolute": Scale based on absolute values provided to `scale`.
        Default is "relative".
    log : bool, optional
        If True, visualize the common logarithm of the absolute values of patch data.
    show : bool, optional
        If True, show the plot. Otherwise, just return the axis.
    **kwargs : dict, optional
        Passed to `scipy.signal.spectrogram` to control spectrogram options.
        See its documentation for options.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Create a spectrogram plot
    >>> ax = patch.viz.spectrogram(show=False)
    """
    dims = patch.dims
    if len(dims) > 2 or len(dims) < 1:
        raise ValueError("Can only make spectrogram of 1D or 2D patches.")

    other_dim = _get_other_dim(dim, dims)
    if other_dim is not None:
        if aggr_domain == "time":
            patch_aggr = patch.aggregate(other_dim, method="mean", dim_reduce="squeeze")
            spec = _spectrogram_patch(patch_aggr, dim, **kwargs)
        elif aggr_domain == "frequency":
            _spec = _spectrogram_patch(patch, dim, **kwargs).squeeze()
            spec = _spec.aggregate(other_dim, method="mean").squeeze()
        else:
            raise ValueError(
                f"The aggr_domain '{aggr_domain}' should be 'time' or 'frequency'."
            )
    else:
        spec = _spectrogram_patch(patch, dim, **kwargs)
    return spec.viz.waterfall(
        ax=ax, cmap=cmap, scale=scale, scale_type=scale_type, log=log, show=show
    )
