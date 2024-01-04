"""
Module for Fourier transforms.

See the [FFT note](/notes/dft_notes.qmd) for discussion on the
implementation.
"""
from __future__ import annotations

from collections.abc import Sequence
from operator import mul

import numpy as np
import numpy.fft as nft

from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.units import invert_quantity
from dascore.utils.misc import iterate
from dascore.utils.patch import (
    _get_data_units_from_dims,
    _get_dx_or_spacing_and_axes,
    patch_function,
)
from dascore.utils.time import to_float
from dascore.utils.transformatter import FourierTransformatter


def _get_dft_new_coords(patch, dxs, dims, axes, real):
    """Create coordinates based on dxs and patch shape."""

    def _get_fft_coord(x_len, dx, units):
        """Get coord for normal fft coord."""
        new_dx = 1.0 / (x_len * dx)
        stop = ((x_len - 1) // 2 + 1) * new_dx
        start = -(x_len // 2) * new_dx
        units = invert_quantity(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    def _get_rfft_coord(x_len, dx, units):
        """Get coord from real fft coord."""
        new_dx = 1.0 / (x_len * dx)
        start = 0
        stop = (x_len // 2 + 1) * new_dx
        units = invert_quantity(units)
        return get_coord(start=start, stop=stop, step=new_dx, units=units)

    # first disassociate old coordinates. We do this rather than drop them
    # so the idft can find them and exactly restore old coords.
    old_cm = patch.coords.disassociate_coord(*dims)
    new_coords = old_cm.get_coord_tuple_map()
    ft = FourierTransformatter()
    for i, dim in enumerate(dims):
        old_coord = patch.get_coord(dim)
        units = old_coord.units
        size = old_coord.shape[0]
        dx = dxs[i]
        name = ft.rename_dims(dim)[0]
        if dim == real:
            coord = _get_rfft_coord(size, dx, units)
        else:
            coord = _get_fft_coord(size, dx, units)
        new_coords[name] = (name, coord)
    new_dims = ft.rename_dims(patch.dims, index=axes)
    cm = get_coord_manager(new_coords, dims=new_dims)
    return cm


def _get_dft_attrs(patch, dims, new_coords):
    """Get new attributes for transformed patch."""
    new = dict(patch.attrs)
    new["dims"] = new_coords.dims
    new["data_units"] = _get_data_units_from_dims(patch, dims, mul)
    return PatchAttrs(**new)


@patch_function()
def dft(
    patch: PatchType, dim: str | None | Sequence[str], *, real: str | bool | None = None
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
    - Simply uses numpy's fft module but outputs are scaled by the sample
      spacing along each transformed dimension and coordinates corresponding
      to frequency bins are shifted so they remain ordered.

    - Each transformed dimension is renamed with a preceding `ft_`. e.g.,
      `time` becomes `ft_time` (ft stands for fourier transform).

    - Each transformed dimension has units of 1/original units.

    - Output data units are the original data units multiplied by the units
      of each transformed dimension.

    - Non-dimensional coordiantes associated with transformed coordinates
      will be dropped in the output.

    - See the [FFT note](dascore.org/notes/fft_notes.html) in the Notes section
      of DASCore's documentation for more details.

    See Also
    --------
    -[idft](`dascore.transform.fourier.idft`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # perform dft (fft) on time axis
    >>> dft_time = patch.dft(dim="time")
    >>> # make it a real fft (no negative frequencies)
    >>> dft_time_real = patch.dft(dim="time", real=True)
    >>> # dft on specified dimensions, specify real dimension
    >>> dft_some_real = patch.dft(dim=("time", "distance"), real="time")
    """
    dims = list(iterate(dim if dim is not None else patch.dims))
    patch.assert_has_coords(dims)
    # re-arrange list so real dim is last (if provided)
    if isinstance(real, str):
        assert real in dims, "real must be in provided dimensions."
        dims.append(dims.pop(dims.index(real)))
    real = dims[-1] if real is True else real  # if true grab last dim
    # get axes and spacing along desired dimensions.
    dxs, axes = _get_dx_or_spacing_and_axes(patch, dims, require_evenly_spaced=True)
    func = nft.rfftn if real is not None else nft.fftn
    # scale by dx's as explained above and in notes, then shift
    scale_factor = np.prod(dxs)
    fft_data = func(patch.data, axes=axes) * scale_factor
    shift_slice = slice(None) if real is None else slice(None, -1)
    data = nft.fftshift(fft_data, axes=axes[shift_slice])
    # get new coordinates
    new_coords = _get_dft_new_coords(patch, dxs, dims, axes, real)
    # get attributes
    attrs = _get_dft_attrs(patch, dims, new_coords)
    return patch.new(data=data, coords=new_coords, attrs=attrs)


def _get_idft_dims_steps_axis(patch, dim):
    """
    Get the dimensions, step sizes as a float, axis numbers and if an
    rff should be performed.
    """
    ft = FourierTransformatter()
    if dim is None:
        dim = [x for x in patch.dims if x.startswith("ft_")]
    # try to get pre-transformed names if used. EG "time" might refer to
    # ft_time for brevity.
    current_dims = set(patch.dims)
    dims = [x if x in current_dims else ft.rename_dims(x)[0] for x in iterate(dim)]
    patch.assert_has_coords(dims)
    coords = [patch.get_coord(x, require_evenly_sampled=True) for x in dims]
    is_real = [1 if to_float(x.min()) == 0 else 0 for x in coords]
    real_sum = sum(is_real)
    assert real_sum <= 1, "only one real axis allowed."
    has_real = bool(real_sum)
    # we need to move the real dim to the end of the list
    if has_real:
        real_ind = is_real.index(1)
        dims.append(dims.pop(real_ind))
    steps, axis = _get_dx_or_spacing_and_axes(patch, dims)
    return dims, steps, axis, has_real


def _get_idft_coords_and_sizes(patch, dims, new_dims, axes, real):
    """Get the new coords for the idft and expected sizes to pass to numpy."""
    shapes = patch.shape
    coord_map = patch.coords.disassociate_coord(*dims).get_coord_tuple_map()
    sizes = []
    for old_dim, new_dim, ax in zip(dims, new_dims, axes):
        # if old dim is stored
        ax_len = shapes[ax]
        potential_coord = coord_map.get(new_dim, (None, None))[1]
        if potential_coord is None:
            msg = (
                "Currently, IDFT can only be performed on patches which have"
                " been transformed to Fourier domain with dft method."
            )
            raise NotImplementedError(msg)
        if (len(potential_coord) == ax_len) or (real and old_dim == dims[-1]):
            coord_map[new_dim] = (new_dim, potential_coord)
            sizes.append(len(potential_coord))
    ft = FourierTransformatter()
    new_dims = ft.rename_dims(patch.dims, index=axes, forward=False)
    cm = get_coord_manager(coord_map, dims=new_dims).drop_coords(*dims)[0]
    out_size = np.array(sizes) if len(sizes) else None
    return cm, out_size


def _get_idft_attrs(patch, dims, new_coords):
    """Get new attributes for transformed patch."""
    # add all {dim}_min to new coords to ensure reverse ft can restore dims.
    new = dict(patch.attrs)
    new["dims"] = new_coords.dims
    new["data_units"] = _get_data_units_from_dims(patch, dims, mul)
    return PatchAttrs(**new)


@patch_function()
def idft(patch: PatchType, dim: str | None | Sequence[str] = None) -> PatchType:
    """
    Perform the inverse discrete Fourier transform (idft) on specified dimension(s).

    Currently, only patches that have been transformed with
    [dft](`dascore.transform.fourier.dft`) can be used with this function.
    After transformation with dft, the transformed coordinates cannot change
    (e.g., with [select]('dascore.proc.basic.select`) otherwise idft won't
    work.

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
    - Real transforms are determined by transformed coordinates which have
      no negative values.

    - See the [FFT note](dascore.org/notes/fft_notes.html) in Notes section
      of DASCore's documentation.

    See Also
    --------
    -[dft](`dascore.transform.fourier.dft`)

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # perform dft (fft) on time axis
    >>> dft_time = patch.dft(dim="time")
    >>> # get inverse dft, transformed axis are ascertained automatically
    >>> idft = dft_time.idft()
    """
    dims, steps, axes, real = _get_idft_dims_steps_axis(patch, dim)
    new_dims = FourierTransformatter().rename_dims(dims, forward=False)
    func = nft.irfftn if real else nft.ifftn
    coords, sizes = _get_idft_coords_and_sizes(patch, dims, new_dims, axes, real)
    # now unshift data and undo scaling
    ax_slice = slice(None, -1) if real else slice(None)
    scale_factor = np.prod([to_float(coords.coord_map[x].step) for x in new_dims])
    _preped = nft.ifftshift(patch.data / scale_factor, axes=axes[ax_slice])
    data = func(_preped, s=sizes, axes=axes)
    attrs = _get_idft_attrs(patch, dims, coords)
    return patch.new(data=data, attrs=attrs, coords=coords)
