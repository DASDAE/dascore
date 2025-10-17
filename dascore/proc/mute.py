"""Processing for muting (zeroing) patch data in specified regions."""

from __future__ import annotations

import dataclasses
from collections import defaultdict
from collections.abc import Mapping, Sized

import numpy as np
from scipy.ndimage import gaussian_filter

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import broadcast_for_index
from dascore.utils.patch import get_dim_axis_value, patch_function


@dataclasses.dataclass
class _OriginNLines:
    """Get a tuple of origin, line1, line2."""

    origin: np.ndarray
    line1: np.ndarray
    line2: np.ndarray

    @classmethod
    def _check_degenerate(self, line):
        """Ensure a line isn't degenerate else raise."""
        if len(np.unique(line, axis=0)) != 2:
            msg = f"Line specified for mute {line} is degenerate!"
            raise ParameterError(msg)

    @classmethod
    def from_point_list(cls, points):
        """"""
        cls._check_degenerate(points[:2])
        cls._check_degenerate(points[2:])

        # We need to ensure one of the points is duplicated; this is the orign
        unique, idx, counts = np.unique(
            points, axis=0, return_index=True, return_counts=True
        )
        if len(unique) == len(points):
            msg = (
                f"No common point found in {points}. For unambiguous mute "
                f"lines must share an origin."
            )
            raise ParameterError(msg)
        gt_1_counts = counts > 1
        origin = points[idx[gt_1_counts]]
        others = points[idx[~gt_1_counts]]
        l1, l2 = others
        return cls(origin=origin, line1=l1, line2=l2)


def _get_2d_mute_lines(vals, patch, dims, relative=True):
    """
    Return values which are two lines in absolute coordinate space,
    (expressed as floats)
    """

    def _get_coord_float_values(coord, vals, relative):
        """Get the coordinate float values relative to start of coords."""
        out = dc.to_float(np.array(vals))
        if not relative:
            out = out - dc.to_float(coord.min())
        return out

    out = []
    # Keep track of the axis that need to be filled in (eg None, paired w/ float)
    fill_inds = defaultdict(list)
    for ind, (dim, row) in enumerate(zip(dims, vals)):
        coord = patch.get_coord(dim)
        # In the case of single values (eg None, ..., or some other)
        # We need to just mark them and come back later.
        if not isinstance(row, Sized):
            # We need to get the actual value from the coord.
            if row is not None and row is not Ellipsis:
                row = _get_coord_float_values(coord, vals, relative=relative)
            fill_inds[dim].append(row)
            out.append(None)
        # Otherwise, we just run with it.
        vals = _get_coord_float_values(coord, np.array(row), relative=relative)
        out.append(vals)
    # Now we can ascertain the line intended by None.
    if fill_inds:
        breakpoint()
    # Then put the 4 points together. This gives us a len 4 array with rows
    # as points from first line, then points from second.
    points = np.stack(out, axis=-1).reshape(-1, 2)
    origin_n_lines = _OriginNLines.from_point_list(points)
    breakpoint()


def _get_mute_params(patch, kwargs, relative=True):
    """
    Get (and validate) the muting parameters.

    Returns arrays of dims, axes, and values. Values has different meaning
    based on the number of dimensions.
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
    val_list = [x[2] for x in dim_ax_vals]
    if len(dims) == 1:
        vals = np.array(val_list)
        if vals.size != 2:  # Handle single dimension Mute.
            msg = "Mute requires two boundaries when using a single dimension."
            raise ParameterError(msg)
    elif len(dims) > 1:  # Dealing with lines. .
        vals = _get_2d_mute_lines(val_list, patch, dims, relative)
    return dims, axes, vals


def _get_smooth_sigma(dims, smooth, patch):
    """Get sigma values, in samples, for the gaussian kernel."""

    def _broadcast_smooth_to_dims(dims, smooth):
        """Broadcast smooth samples to dims length."""
        # First, we need to get the smooth parameters to line up with the dims.
        # For a single value, just broadcast to dim length.
        if not isinstance(smooth, Mapping):
            vals = [smooth] * len(dims)
        else:
            # Otherwise each dimension's smooth must be specified.
            if not set(dims) == set(smooth):
                msg = (
                    f"If a taper dictionary is used in Mute, it must have all "
                    f"the same keys as the dimensions. Kwarg dims are {dims} and"
                    f"taper keys are {list(smooth)}."
                )
                raise ParameterError(msg)
            vals = [smooth[dim] for dim in dims]
        return vals

    def _convert_to_samples(smooth, dims, patch):
        """Convert the smooth parameter to number of samples."""
        out = []
        for dim, val in zip(dims, smooth):
            coord = patch.get_coord(dim)
            if val is None:
                out.append(0)
            elif isinstance(val, int | np.integer):
                out.append(val)
            elif isinstance(val, float | np.floating):
                if not 0 <= val <= 1:
                    msg = (
                        f"Mute's smooth parameter for {dim} must be between 0 "
                        f"and 1 when using a floating point value."
                    )
                    raise ParameterError(msg)
                out.append(int(np.round(len(coord) * val)))
            else:  # should capture quantities
                out.append(coord.get_sample_count(val))
        return out

    smooth_by_dims = _broadcast_smooth_to_dims(dims, smooth)
    smooth_ints = _convert_to_samples(smooth_by_dims, dims, patch)
    # Now finagle into input for scipy's gaussian smooth.
    return smooth_ints


def _line_smooth(data, dims, axes, values, patch):
    """
    Mute data between two lines.
    """
    breakpoint()


@patch_function()
def mute(
    patch: PatchType,
    *,
    smooth=None,
    invert: bool = False,
    relative: bool = True,
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
    smooth
        Parameter controlling smoothing of the mute evenlope. Defines the sigma
        Can be:
        - None: sharp mute
        - float (0.0-1.0): fraction of dimension range (e.g., 0.01 = 1%)
          which is applied independently to each dimension involved in the
          mute.
        - int: Indicates number of samples for each dimension.
        - Quantity with units, indicates values along a single dimension.
          Only applicable if a single dimension is specified.
        - dict: {dim: taper_value} for dimension-specific smooth
          values which can be any of the above.
    invert
        If True, invert the taper such that the values outside the defined region
        are set to 0.
    relative
        If True (default), values are relative to coordinate edges.
        Positive values are offsets from start, negative from end.
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
    >>> patch = dc.get_example_patch().full(1)
    >>>
    >>>
    >>> # Mute first 0.5s (relative to start by default)
    >>> muted = patch.mute(time=(0, 0.5))
    >>>
    >>> # Mute everything except middle section
    >>> kept = patch.mute(time=(0.2, -0.2), mode="complement")
    >>>
    >>> # 1D Mute with smoothed absolute units for time.
    >>> muted = patch.mute(time=(0.2, 0.8), smooth=0.02 * dc.units.s)
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
    >>> # Knock down edges with rolling mean along the time dimension.
    >>> smooth = envelope.rolling(time=5, samples=True).mean()
    >>> # Then multiply the two patches.
    >>> result = patch * smooth

    Notes
    -----
    - By relative=True (the default) means values are offsets from coordinate
      edges. Use relative=False for absolute coordinate values.

    - Currently, mute doesn't support more than 2 dimensions.

    - For more control over boundary smoothing, use a patch with one values
      then apply custom tapering/smooting before multiplying with the
      original patch. See example section for more details.

    See Also
    --------
    - [`Patch.select`](`dascore.Patch.select`)
    - [`Patch.taper_range`](`dascore.Patch.taper_range`)
    - [`Patch.gaussian_filter`](`dacore.proc.filter.gaussian_filter`)
    """
    dims, axes, values = _get_mute_params(patch, kwargs, relative)
    out = np.zeros_like(patch.data) if invert else np.ones_like(patch.data)
    fill_val = 1 if invert else 0
    # Easy path for 1D mute.
    if len(dims) == 1:
        coord = patch.get_coord(dims[0])
        args = (values[0][0], values[0][1])
        _, c_index = coord.select(args, relative=relative)
        index = broadcast_for_index(out.ndim, axes[0], c_index, slice(None))
        out[index] = fill_val
    else:
        out = _line_smooth(out, dims, axes, values, patch)
    # Apply smoothing.
    if smooth is not None:
        sigma = _get_smooth_sigma(dims, smooth, patch)
        out = gaussian_filter(out, sigma=sigma, axes=tuple(axes))
    return patch.update(data=patch.data * out)
