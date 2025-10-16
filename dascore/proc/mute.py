"""Processing for muting (zeroing) patch data in specified regions."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.patch import get_dim_axis_value, patch_function


def _get_mute_params(patch, kwargs):
    """
    Get (and validate) the muting parameters.

    Returns arrays of dims, axes, and values.
    """
    # Validate we have dimension specifications
    if not kwargs or len(kwargs) > 2:
        msg = (
            "Mute requires one or two keyword arguments must be provided. "
            f"You passed {kwargs}"
        )
        raise ParameterError(msg)
    dim_ax_vals = get_dim_axis_value(patch, kwargs=kwargs, allow_multiple=True)
    dims = [x[0] for x in dim_ax_vals]
    axes = np.array([x[1] for x in dim_ax_vals])
    # Handle single dimension Mute.
    if len(dims) == 1:
        vals = np.array([x[2] for x in dim_ax_vals])
        if vals.size != 2:
            msg = "Mute requires two boundaries when using a single dimension."
            raise ParameterError(msg)
    else:
        breakpoint()
        print()
    return dims, axes, vals


def _get_taper_vals(dims, taper, patch, samples):
    """Get the taper values ordered by dimension, in samples."""
    # Just broadcast taper to  same length as dims.
    if taper is None:
        return [0] * len(dims)
    if not isinstance(taper, Mapping):
        vals = [taper] * len(dims)
    else:
        # Otherwise each dimension's taper must be specified.
        if not set(dims) == set(taper):
            msg = (
                f"If a taper dictionary is used in Mute, it must have all "
                f"the same keys as the dimensions. Kwarg dims are {dims} and"
                f"taper keys are {list(taper)}."
            )
            raise ParameterError(msg)
        vals = [taper[dim] for dim in dims]
    out = []
    for dim, val in zip(dims, vals):
        coord = patch.get_coord(dim)
        if val is None:
            out.append(0)
        elif samples:
            out.append(val)
        else:
            coord = patch.get_coord(dim)
            coord.coord_range(val)
            out.append(val)
    return np.array(out)


def _mute_patch_1d(
    patch,
    dim,
    vals,
    taper_samps,
    window,
    samples,
    relative,
    invert,
):
    """Apply mute to 1D patch using patch.range_taper."""
    coord = patch.get_coord(dim)

    if not samples:
        # Get range represented by values. This is a bit ugly...
        sel = coord.select(tuple(*vals), relative=relative)[1]
        start = sel.start if sel.start is not None else 0
        # stop can be None (open interval) or an exclusive upper bound
        if sel.stop is None:
            stop = len(coord)
        else:
            stop = sel.stop
        trange = (start, stop)
    else:
        # vals is a 2D array with shape (1, 2), flatten to get (start, stop)
        trange = vals.flatten() if vals.ndim > 1 else vals

    # Handle edge case: if muting to the end with no taper, use 2-value form
    # because 4-value form can't express "mute to the very last sample"
    max_idx = len(coord) - 1
    if taper_samps == 0 and trange[1] >= len(coord):
        taper_dict = {dim: [trange[0], max_idx]}
    else:
        # Clamp taper boundaries to valid sample indices.
        # taper_range uses exclusive upper bounds in 4-value form [a,b,c,d]
        # which zeros indices [b, c).
        taper_dict = {
            dim: [
                max(0, trange[0] - taper_samps),
                trange[0],
                min(max_idx, trange[1]),
                min(max_idx, trange[1] + taper_samps),
            ]
        }
    out = patch.taper_range(
        invert=not invert,
        samples=True,
        window_type=window,
        **taper_dict,
    )
    return out


@patch_function()
def mute(
    patch: PatchType,
    *,
    taper: float | dict | None = None,
    window_type: str = "hann",
    invert: bool = False,
    relative: bool = True,
    samples: bool = False,
    **kwargs,
) -> PatchType:
    """
    Mute (zero out) data in a region specified by one or more lines.

    Each dimension accepts two boundary specifications that define
    the mute region. Boundaries can be scalar values or arrays of values.

    Parameters
    ----------
    patch
        The patch instance.
    taper
        Taper width at mute boundaries. Can be:
        - None: sharp mute (no taper)
        - float (0.0-1.0): fraction of dimension range (e.g., 0.05 = 5%)
          The number of samples in the mute is held constant across dimensions
          by calculating the samples the fraction represents for each dimension
          and using the minimum sample count. This helps avoid distorted mutes
          based on dimensions with vastly different lengths.
        - Quantity with units: absolute value (e.g., 0.02*dc.units.s)
          Note: If multiple dimensions specified, must use dict.
        - dict: (dim: taper_value) for dimension-specific taper
          Values can be floats (fractions) or Quantities (absolute)
        - Callable: custom function that receives and modifies envelope array.
    window_type
        Window function for tapering (only used for non-callable taper).
        Options:
            {sorted(WINDOW_FUNCTIONS)}.
    invert
        If True, invert the taper such that the values outside the defined region
        are set to 0.
    relative
        If True (default), values are relative to coordinate edges.
        Positive values are offsets from start, negative from end.
    samples
        If True, values specified in samples rather than coordinate values.
    **kwargs
        Dimension specifications as (boundary_1, boundary_2) pairs.
        Each boundary can be:
        - Scalar: constant value (defines plane perpendicular to dimension)
        - Array/list: coordinates defining line/curve/surface
        - None or ...: edge of coordinate (min or max)

    Examples
    --------
    >>> import dascore as dc
    >>> from scipy.ndimage import gaussian_filter
    >>>
    >>> patch = dc.get_example_patch("ricker_moveout")
    >>>
    >>>
    >>> # Mute first 0.5s (relative to start by default)
    >>> muted = patch.mute(time=(0, 0.5))
    >>>
    >>> # Mute everything except middle section
    >>> kept = patch.mute(time=(0.2, -0.2), mode="complement")
    >>>
    >>> # Taper with absolute units
    >>> muted = patch.mute(
    ...     time=(0.2, 0.8),
    ...     taper={'time': 0.02 * dc.units.s},
    ... )
    >>>
    >>> # Classic first break mute: mute early arrivals
    >>> # Line from (t=0, d=0) to (t=0.3, d=300) defines velocity=1000 m/s
    >>> muted = patch.mute(
    ...     time=(0, [0, 0.3]),
    ...     distance=(None, [0, 300]),
    ...     taper=0.02,
    ... )
    >>>
    >>> # Mute late arrivals: from velocity line to end
    >>> muted = patch.mute(
    ...     time=([0, 0.3], None),
    ...     distance=([0, 300], 0),
    ... )
    >>>
    >>> # Mute wedge between two velocity lines
    >>> muted = patch.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ... )
    >>>
    >>> # Mute wedge outside two velocity lines
    >>> muted = patch.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ...     invert=True,
    ... )
    >>>
    >>> # Apply custom tapering
    >>> ones = patch.full(1.0)
    >>> envelope = ones.mute(
    ...     time=([0, 0.375], [0, 0.25]),
    ...     distance=([0, 300], [0, 300]),
    ... )
    >>> # Knock down edges with gaussian filter.
    >>> smooth = envelope.gaussian_filter(time=5, distance=5, samples=True)
    >>> # Then multiply the two patches.
    >>> result = patch * smooth

    Notes
    -----
    - By relative=True (the default) means values are offsets from coordinate
      edges. Use relative=False for absolute coordinate values.

    - Currently, mute doesn't support more than 2 dimensions.

    - For more control over tapering, use a patch with one values then apply
      custom tapering/smooting before multiplying with the original patch.
      See example section for more details.

    See Also
    --------
    [`Patch.select`](`dascore.Patch.select`)
    [`Patch.taper_range`](`dascore.Patch.taper_range`)
    """
    dims, axes, values = _get_mute_params(patch, kwargs)
    taper_vals = _get_taper_vals(dims, taper, patch, samples)
    # Easy path for 1D mute.
    if len(dims) == 1:
        out = _mute_patch_1d(
            patch,
            dims[0],
            values,
            taper_vals[0],
            window=window_type if taper is not None else "boxcar",
            samples=samples,
            relative=relative,
            invert=invert,
        )
        return out

    breakpoint()
