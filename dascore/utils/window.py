"""
One way to say "a window of X, overlapping by Y" along a patch's dimensions.

Every windowed patch function takes its windows as dimension keywords --
``time=0.5``, ``distance=32 * m``, ``time=16`` with ``samples=True`` -- and
some take an ``overlap`` or ``step`` between windows besides. Turning those
into sample counts is one job, and :func:`resolve_window` does it for all of
them, returning a :class:`Window`.

What differs between functions is policy, not conversion: whether an even
window is adjusted or refused, whether a window longer than the coordinate is
refused, what a missing overlap means. Each of those is a keyword here, stated
by the function which wants it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from dascore.exceptions import CoordError, ParameterError
from dascore.units import Quantity, is_percent
from dascore.utils.misc import get_parent_code_name
from dascore.utils.patch import get_dim_axis_value
from dascore.utils.time import is_timedelta64, to_float, to_timedelta64

# What a function means by an overlap nobody gave: a sample count, or a
# rule for one from the window's size in samples.
OverlapDefault = Callable[[int], int] | int | None

# Marks a dimension a mapping left out, which is not the same as one it
# set to None.
_UNGIVEN = object()


@dataclass(frozen=True)
class Window:
    """
    A window along some of a patch's dimensions, in samples.

    Parameters
    ----------
    dims
        The dimensions windowed, in the order the caller named them.
    axes
        The axis of each.
    size
        The window along each, in samples.
    overlap
        How far each window reaches into the next, in samples, or None if
        the caller gave no overlap, no step, and no default -- which each
        function reads its own way.
    ndim
        How many dimensions the patch has, so the window can be spelled
        out for every axis.
    """

    dims: tuple[str, ...]
    axes: tuple[int, ...]
    size: tuple[int, ...]
    overlap: tuple[int, ...] | None
    ndim: int

    @property
    def stride(self) -> tuple[int, ...] | None:
        """How far each window advances, in samples; None with no overlap."""
        if self.overlap is None:
            return None
        return tuple(size - over for size, over in zip(self.size, self.overlap))

    def full_size(self, fill: int = 1) -> tuple[int, ...]:
        """Return the window along every patch axis, `fill` where none was given."""
        out = [fill] * self.ndim
        for axis, size in zip(self.axes, self.size):
            out[axis] = size
        return tuple(out)


def _percent_to_samples(value: Any, size: int) -> tuple[Any, bool]:
    """Turn a percent of the window into samples of it; say whether it was one."""
    if not (was_percent := is_percent(value)):
        return value, False
    magnitude = value.magnitude
    if magnitude < 0 or magnitude > 100:
        msg = f"Percentage must be between 0 and 100, not {value}"
        raise ParameterError(msg)
    # Half rounds to even, as numpy rounds: 50% of 5 is 2 and of 7 is 4.
    return int(np.round(magnitude / 100 * size)), was_percent


def _check_not_negative(value: Any, name: str) -> None:
    """Refuse a step or overlap which retreats."""
    magnitude = value.magnitude if isinstance(value, Quantity) else value
    # A bare 0 would make numpy cast the timedelta to a generic unit.
    zero = to_timedelta64(0) if is_timedelta64(magnitude) else 0
    if magnitude is not None and magnitude < zero:
        msg = f"{name} must be non-negative"
        raise ParameterError(msg)


def _to_samples(
    coord,
    value: Any,
    *,
    samples: bool,
    enforce_lt_coord: bool,
    through_coord: bool = True,
) -> tuple[int, bool]:
    """
    Return a value along a coordinate as a sample count, and whether it was one.

    A quantity carries its own units, whatever the call said about samples.
    A bare sample count need not go through the coordinate at all, and a
    function which does not require even sampling reads it directly.
    """
    if isinstance(value, Quantity):
        samples = False
    if samples and not through_coord:
        return int(value), True
    count = coord.get_sample_count(
        value, samples=samples, enforce_lt_coord=enforce_lt_coord
    )
    return count, samples


def _spread(value: Any, dims: tuple[str, ...], name: str) -> dict[str, Any]:
    """Return one value per dimension from a scalar or a mapping."""
    if not isinstance(value, Mapping):
        return dict.fromkeys(dims, value)
    if extra := set(value) - set(dims):
        names = sorted(map(str, extra))
        msg = f"{name} contains dimensions not being windowed: {names}"
        raise ParameterError(msg)
    for dim, given in value.items():
        if given is None:
            msg = (
                f"{name} for {dim!r} is None; leave it out of the mapping "
                "to take the default."
            )
            raise ParameterError(msg)
    return {dim: value.get(dim, _UNGIVEN) for dim in dims}


def _too_small(name: str, coord, count: int, min_samples: int, samples: bool) -> str:
    """Say why a window is too short, and what to do about it."""
    if samples:
        hint = "Try increasing its value."
    else:
        # in seconds, which is what a time value would be given in
        step = coord.step
        step = f"{to_float(step)} s" if is_timedelta64(step) else step
        hint = (
            f"The value is in the units of {name}, which is sampled "
            f"every {step}; increase it, or use samples=True to give "
            "the window in samples."
        )
    return (
        f"Window must have at least {min_samples} samples along each "
        f"dimension. {name} has {count} samples. {hint}"
    )


def resolve_window(
    patch,
    kwargs: Mapping[str, Any],
    *,
    samples: bool = False,
    overlap: Any = None,
    step: Any = None,
    default_overlap: OverlapDefault = None,
    allow_multiple: bool = True,
    allow_empty: bool = False,
    require_evenly_sampled: bool = True,
    require_odd: bool = False,
    min_samples: int | None = 1,
    warn_above: int | None = None,
    enforce_lt_coord: bool = False,
) -> Window:
    """
    Return the window a call's dimension keywords describe, in samples.

    Parameters
    ----------
    patch
        The patch, or anything with its ``dims`` and ``coords`` -- a
        `PatchMeta` serves, so a processor can resolve a window without data.
    kwargs
        Dimension names and the window along each, in coordinate units or,
        with ``samples``, in samples. A `Quantity` keeps its units either way.
    samples
        If True, bare numbers are sample counts.
    overlap
        How far each window reaches into the next: one value for every
        dimension, or a mapping of dimension to value. Read like the windows
        are, except that a percent is a fraction of that dimension's window.
    step
        How far each window advances, spelled instead of ``overlap``; the
        two are exclusive.
    default_overlap
        What a dimension given no overlap and no step gets: a sample count,
        or a callable from the window's size in samples to one. None leaves
        `Window.overlap` None when nothing was given.
    allow_multiple
        Whether more than one dimension may be windowed.
    allow_empty
        Whether no dimension at all may be, giving an empty window whose
        `full_size` is all ones.
    require_evenly_sampled
        Whether a windowed coordinate must be evenly sampled. When it need
        not be, a window given in samples never consults the coordinate;
        one in units cannot be converted without an even step anyway.
    require_odd
        Whether the window must have an odd number of samples. Given in
        units, an even count is rounded up; given in samples, refused.
    min_samples
        The fewest samples a window may have along any dimension, or None
        for no floor at all.
    warn_above
        Warn when the window's total sample count -- its area -- exceeds this.
    enforce_lt_coord
        Whether a window, step, or overlap longer than its coordinate is refused.

    Raises
    ------
    ParameterError
        For a window which is too small, even when it must be odd, longer
        than its coordinate when that is refused, or an overlap or step which
        is negative, leaves no advance, or is given alongside the other.
    CoordError
        For a coordinate which is not evenly sampled when that is required.
    """
    if overlap is not None and step is not None:
        msg = "step and overlap are mutually exclusive."
        raise ParameterError(msg)
    if not kwargs and allow_empty:
        return Window((), (), (), None, len(patch.dims))
    dim_axis_values = get_dim_axis_value(
        patch, kwargs=kwargs, allow_multiple=allow_multiple
    )
    dims = tuple(x.dim for x in dim_axis_values)
    axes = tuple(x.axis for x in dim_axis_values)
    coords = {}
    for dim in dims:
        coord = patch.coords.get_coord(dim)
        if require_evenly_sampled and coord.step is None:
            extra = f"as required by {get_parent_code_name()}"
            msg = f"Coordinate {dim} is not evenly sampled {extra}"
            raise CoordError(msg)
        coords[dim] = coord

    sizes = []
    for dim, _, value in dim_axis_values:
        coord = coords[dim]
        count, in_samples = _to_samples(
            coord,
            value,
            samples=samples,
            enforce_lt_coord=enforce_lt_coord,
            through_coord=require_evenly_sampled,
        )
        if min_samples is not None and count < min_samples:
            msg = _too_small(dim, coord, count, min_samples, in_samples)
            raise ParameterError(msg)
        if require_odd and count % 2 != 1:
            if in_samples:
                msg = (
                    f"For clean median calculation, dimension windows must be odd "
                    f"but {dim} has a value of {count} samples."
                )
                raise ParameterError(msg)
            count += 1
        sizes.append(count)
    size = tuple(sizes)

    # Warn on the total window, not each dimension: the cost of a windowed
    # operation tracks the number of samples the window covers, so a 2D
    # window is as expensive as its area.
    total = math.prod(size)
    if warn_above is not None and total > warn_above:
        msg = (
            f"Large window size ({total} samples) may result in slow "
            "performance. Consider reducing the window size."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)

    overlaps = _resolve_overlap(
        coords,
        dims,
        size,
        overlap=overlap,
        step=step,
        default=default_overlap,
        samples=samples,
        enforce_lt_coord=enforce_lt_coord,
        through_coord=require_evenly_sampled,
    )
    return Window(dims, axes, size, overlaps, len(patch.dims))


def _resolve_overlap(
    coords: Mapping[str, Any],
    dims: tuple[str, ...],
    size: tuple[int, ...],
    *,
    overlap: Any,
    step: Any,
    default: OverlapDefault,
    samples: bool,
    enforce_lt_coord: bool,
    through_coord: bool,
) -> tuple[int, ...] | None:
    """Return the overlap along each dimension in samples, or None if none was given."""
    name = "step" if step is not None else "overlap"
    given = _spread(step if step is not None else overlap, dims, name)
    out: list[int | None] = []
    for dim, window in zip(dims, size):
        value = given[dim]
        kind, in_samples, via_coord = name, samples, through_coord
        if value is _UNGIVEN or value is None:
            if default is None:
                out.append(None)
                continue
            # A default is a sample count already, and is checked like a
            # given one: it must not retreat, and must leave an advance.
            value = default if isinstance(default, int) else default(window)
            kind, in_samples, via_coord = "overlap", True, False
        value, was_percent = _percent_to_samples(value, window)
        _check_not_negative(value, kind)
        count, _ = _to_samples(
            coords[dim],
            value,
            samples=in_samples or was_percent,
            enforce_lt_coord=enforce_lt_coord,
            through_coord=via_coord,
        )
        advance = count if kind == "step" else window - count
        if advance <= 0:
            msg = "Window step must be greater than zero."
            raise ParameterError(msg)
        out.append(window - advance)
    if all(value is None for value in out):
        return None
    if any(value is None for value in out):
        missing = [dim for dim, value in zip(dims, out) if value is None]
        msg = f"{name} was given for some dimensions but not {missing}."
        raise ParameterError(msg)
    return tuple(out)  # ty: ignore[invalid-return-type]
