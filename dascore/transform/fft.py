"""
Module for Fourier transforms.

See the [FFT note](dascore.org/notes/fft_notes.html) details.
"""
from __future__ import annotations

import warnings
from operator import mul
from typing import Sequence

import numpy as np
import numpy.fft as nft

from dascore.constants import PatchType
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.core.schema import PatchAttrs
from dascore.units import get_quantity
from dascore.utils.misc import iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)
from dascore.utils.time import to_float
from dascore.utils.transformatter import FourierTransformatter


def _get_frequencies(patch, dxs, axes, real) -> list[np.ndarray, ...]:
    """Get the un-shifted output frequencies arrays"""
    assert len(dxs) > 0
    assert len(dxs) == len(axes)
    out = []
    shape = patch.shape
    for dx, ax in zip(dxs, axes):
        out.append(nft.fftfreq(shape[ax], dx))
    if real:
        out[-1] = nft.rfftfreq(shape[axes[-1]], dxs[-1])
    return out


def _maybe_invert_unit(units):
    units = get_quantity(units)
    if units is not None:
        units = 1 / units
    return units


def _get_dft_new_coords(patch, dxs, dims, axes, real):
    """Create coordinates based on dxs and patch shape."""

    def _get_fft_coord(x_len, dx, units):
        """Get coord for normal fft coord."""
        new_dx = 1.0 / (x_len * dx)
        stop = ((x_len - 1) // 2 + 1) * new_dx
        start = -(x_len // 2) * new_dx
        units = _maybe_invert_unit(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    def _get_rfft_coord(x_len, dx, units):
        """Get coord from real fft coord."""
        new_dx = 1.0 / (x_len * dx)
        start = 0
        stop = (x_len // 2 + 1) * new_dx
        units = _maybe_invert_unit(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    coord_map = dict(patch.coords.coord_map)
    ft = FourierTransformatter()
    old_to_new_dim = ft.rename_dims(dims)
    for i, dim in enumerate(dims):
        old_coord = coord_map.pop(dim)
        units = old_coord.units
        size = old_coord.shape[0]
        dx = dxs[i]
        if dim == real:
            coord = _get_rfft_coord(size, dx, units)
        else:
            coord = _get_fft_coord(size, dx, units)
        coord_map[old_to_new_dim[i]] = coord
    new_dims = ft.rename_dims(patch.dims, index=axes)
    cm = get_coord_manager(coord_map, dims=new_dims)
    return cm


def _get_dft_attrs(patch, dims, new_coords):
    """Get new attributes for transformed patch."""
    old_coords = patch.coords
    # add all {dim}_min to new coords to ensure reverse ft can restore dims.
    new = dict(patch.attrs)
    new["dims"] = new_coords.dims
    new["data_units"] = _get_data_units_from_dims(patch, dims, mul)
    for dim in dims:
        new[f"{dim}_min"] = old_coords.coord_map[dim].min()
    return PatchAttrs(**new)


@patch_function()
def dft(
    patch: PatchType, dim: str | None | Sequence[str], real: str | bool | None = None
) -> PatchType:
    """
    Perform the discrete Fourier transform (dft) on specified dimension(s).

    Parameters
    ----------
    patch
        Patch to transform.
    dim
        A single, or multiple dimensions over which to perform dft. If
        None, perform dft over all dimensions.
    real
        Either 1) The name of the axis over which to perform a rfft, 2)
        True, which means the last (possibly only) dimenson should have an
        rfft performed, or 3) None, meaning no rfft.

    Notes
    -----
    - See the [FFT note](dascore.org/notes/fft_notes.html) in Notes section
      of DASCore's documentation.

    - Simply uses numpy's fft module but outputs are scaled by the sample
      spacing along each transformed dimension and coordinates are shifted
      so they remain ordered.

    - Each transformed dimension is renamed with a preceding `ft_`. e.g.,
      `time` becomes `ft_time` (ft stands for fourier transform).

    - Each transformed dimension has units of 1/original units.

    - Output data units are the original data units multiplied by the units
      of each transformed dimension.

    - Non-dimensional coordiantes associated with transformed coordinates
      will be dropped in the output.

    See Also
    --------
    -[idft](`dascore.tran.fft.idft`)

    Examples
    --------
    >>> import dascore as dc
    """
    dims = list(iterate(dim if dim is not None else patch.dims))
    # re-arrange list so real dim is last (if provided)
    if isinstance(real, str):
        assert real in dims, "real must be in provided dimensions."
        dims.append(dims.pop(dims.index(real)))
    real = dims[-1] if real is True else real  # if true grab last dim
    # get axes and spacing along desired dimensions.
    dxs, axes = _get_dx_or_spacing_and_axes(patch, dims, require_evenly_spaced=True)
    func = nft.rfftn if real is not None else nft.fftn
    # scale by dx's as explained above and in notes, then shift
    fft_data = func(patch.data, axes=axes) * np.prod(dxs)
    shift_slice = slice(None) if real is None else slice(None, -1)
    data = nft.fftshift(fft_data, axes=axes[shift_slice])
    # get new coordinates
    new_coords = _get_dft_new_coords(patch, dxs, dims, axes, real)
    # get attributes
    attrs = _get_dft_attrs(patch, dims, new_coords)
    return patch.new(data=data, coords=new_coords, attrs=attrs)


@patch_function()
def idft(patch: PatchType, dim: str | None | Sequence[str] = None) -> PatchType:
    """
    Perform the inverse discrete Fourier transform (idft) on specified dimension(s).

    Parameters
    ----------
    patch
        Patch to transform.
    dim
        A single, or multiple dimensions over which to perform idft. If
        None, perform idft over all dimensions that have names starting
        with "ft_", which indicates they have already undergone a fourier
        transform.

    Notes
    -----
    - See the [FFT note](dascore.org/notes/fft_notes.html) in Notes section
      of DASCore's documentation.

    - First divides output by spacing of transformation axis.

    - Real transforms are determined by transformed coordinates which have
      no negative values.

    - The transformed coordinates don't carry information about the offset
      (e.g. we can't tell from frequencies the actual start time) so attibutes
      are searched to restore offset.

    See Also
    --------
    -[dft](`dascore.tran.fft.dft`)

    Examples
    --------
    >>> import dascore as dc

    """


@patch_function()
def rfft(patch: PatchType, dim="time") -> PatchType:
    """
    Perform a real fourier transform along the specified dimension.

    DEPRECATED FUNCTION: Use [dft](`dascore.tran.fft.dft`) instead.
    This function is not scaled as detailed in the dascore documentation.
    """
    msg = "The Patch transform rfft is deprecated. Use dft instead."
    warnings.warn(msg, DeprecationWarning)
    assert dim in patch.dims
    axis = patch.dims.index(dim)

    ft = FourierTransformatter()
    data = patch.data
    sr = 1 / to_float(patch.attrs[f"d_{dim}"])
    freqs = np.fft.rfftfreq(data.shape[axis], sr)
    new_data = np.fft.rfft(data, axis=axis)
    # get new dims and data units
    new_dims = ft.rename_dims(dim)
    new_data_units = _get_data_units_from_dims(patch, dim, mul)
    attrs = patch.attrs.update(data_units=new_data_units)
    dims = [x if i != axis else new_dims[0] for i, x in enumerate(patch.dims)]
    # get new coord
    units = get_quantity(patch.coords.coord_map[dim].units)
    coord = get_coord(values=freqs, units=None if units is None else 1 / units)
    new_coords = {new_dims[0]: coord}
    new_coords.update({x: patch.coords[x] for x in patch.dims if x != dim})
    return patch.__class__(data=new_data, coords=new_coords, dims=dims, attrs=attrs)
