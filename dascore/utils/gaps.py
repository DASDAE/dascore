"""
One gap contract: a tolerance, its predicate, the sweep, and mesh edges.

A gap is a spacing wider than a sampling step allows. Every place DASCore
asks the question (a coordinate's discontinuities, a spool's chunk plan
and gap report, a waterfall's painted seams) states its tolerance in its
own spelling, converts it here, and gets one verdict for one
``(spacing, step, tolerance)``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from dascore.exceptions import ParameterError, UnitError
from dascore.units import Quantity, is_data_size, is_percent
from dascore.utils.time import is_datetime64, is_timedelta64, to_float, to_timedelta64

# The default continuity tolerance, in samples; looser values warn when
# they force merges (#662).
DEFAULT_TOLERANCE = 1.5


def _check_tolerance_value(value, name, shown=None, *, allow_infinite=False):
    """
    Reject a tolerance no gap could be measured against.

    An infinite sample count is a coherent request (no boundary is ever a
    gap) but an infinite distance is not a distance, so only the count is
    allowed to be one.
    """
    shown = value if shown is None else shown
    # One tolerance, not one per patch: a one-element array passes every
    # test below and then broadcasts through the gap comparison.
    if np.asarray(value).ndim:
        msg = (
            f"The tolerance for {name!r} must be a single value, got an "
            f"array of {np.asarray(value).size}."
        )
        raise ParameterError(msg)
    try:
        null = bool(pd.isnull(value))
        # A bare 0 would make numpy cast the timedelta to a generic unit.
        zero = to_timedelta64(0) if is_timedelta64(value) else 0
        negative = not null and value < zero
    except TypeError:
        msg = (
            f"The tolerance for {name!r} must be a sample count, a quantity, "
            f"or a timedelta, got {shown!r}. A unit-bearing string becomes a "
            "quantity with dascore.get_quantity."
        )
        raise ParameterError(msg) from None
    if null or (not allow_infinite and not np.isfinite(value)):
        msg = f"The tolerance for {name!r} must be finite, got {shown}."
        raise ParameterError(msg)
    if negative:
        msg = f"The tolerance for {name!r} must not be negative, got {shown}."
        raise ParameterError(msg)


@dataclass(frozen=True)
class GapTolerance:
    """
    How far past one step a spacing may reach before it is a gap.

    Two constructors, one predicate. ``samples(k)`` allows ``k`` steps
    between neighbouring samples (``k = 1`` is contiguity); ``absolute(q)``
    allows one step plus ``q`` in the coordinate's own units, so the excess
    over the step is what is bounded. An absolute quantity or timedelta is
    converted into the coordinate's own scalar by the caller which knows
    the units (a coordinate's ``_gap_tolerance``, a chunk cell's
    ``_cell_tolerance``); until then `is_gap` cannot be asked of it.

    Examples
    --------
    >>> from dascore.utils.gaps import GapTolerance
    >>> tol = GapTolerance.samples(1.5)
    >>> bool(tol.is_gap(2.0, step=1.0)), bool(tol.is_gap(1.4, step=1.0))
    (True, False)
    >>> GapTolerance.absolute(0.5).is_gap(1.6, step=1.0)
    np.True_
    """

    # exactly one is set: the allowed spacing in steps, or in units
    count: float | None = None
    excess: Any = None

    def __post_init__(self):
        if (self.count is None) == (self.excess is None):
            msg = "A GapTolerance is a sample count or an absolute excess, not both."
            raise ParameterError(msg)
        if self.count is not None:
            _check_tolerance_value(self.count, "count", allow_infinite=True)
        else:
            excess = self.excess
            value = excess.magnitude if isinstance(excess, Quantity) else excess
            _check_tolerance_value(value, "excess", shown=excess)

    def __str__(self):
        if self.count is not None:
            return f"{self.count:g} samples"
        return f"{self.excess} excess"

    @classmethod
    def samples(cls, count) -> GapTolerance:
        """A spacing is a gap past ``count`` steps."""
        return cls(count=float(count))

    @classmethod
    def absolute(cls, excess) -> GapTolerance:
        """A spacing is a gap past one step plus ``excess``, in coordinate units."""
        return cls(excess=excess)

    @classmethod
    def from_user(cls, tolerance, name: str = "tolerance") -> GapTolerance:
        """
        Read a tolerance as the public entry points spell it.

        A number is a multiple of the sampling interval; a quantity or
        timedelta states an absolute excess in the coordinate's units. A
        dimensionless quantity is the multiple it spells out.
        """
        if isinstance(tolerance, GapTolerance):
            return tolerance
        if isinstance(tolerance, timedelta) or is_timedelta64(tolerance):
            return cls(excess=to_timedelta64(tolerance))
        if isinstance(tolerance, Quantity):
            if is_data_size(tolerance):
                msg = (
                    f"Cannot use a tolerance of {tolerance} for {name!r}: a data "
                    "size does not measure a gap along a coordinate."
                )
                raise UnitError(msg)
            if is_percent(tolerance):
                msg = (
                    f"Cannot use a tolerance of {tolerance} for {name!r}: a "
                    "percentage is neither a sample count nor a length. Pass the "
                    "count itself, or a length in the coordinate's units."
                )
                raise UnitError(msg)
            if not tolerance.dimensionless:
                return cls(excess=tolerance)
            tolerance = float(tolerance.m_as("dimensionless"))
        _check_tolerance_value(tolerance, name, allow_infinite=True)
        return cls(count=float(tolerance))

    def is_gap(self, delta, step):
        """
        Whether each spacing ``delta`` is a gap against sampling ``step``.

        Both are magnitudes or signed values in the coordinate's scalar
        (an absolute tolerance must already be resolved). An unknown step
        (NaN) is never a gap against a sample count, and against an
        absolute excess the step counts as nothing.
        """
        delta = np.abs(np.asarray(delta))
        step = np.abs(np.asarray(step))
        if self.count is not None:
            with np.errstate(invalid="ignore"):
                return delta > step * self.count
        margin = np.where(pd.isnull(step), 0, step) + self.excess
        return delta > margin


def gap_boundaries(start, stop, step, tolerance: GapTolerance):
    """
    Locate the gaps between value-ordered runs.

    Each row is a run: its first and last value and its step. Returns
    ``(order, reach, has_gap)`` over the start-ordered rows. ``reach`` is
    the furthest stop seen *before* each row, so an overlapping or fully
    nested row can never open a gap behind it, and ``has_gap`` marks each
    row whose start clears that reach by more than the tolerance allows.
    The first row has nothing behind it and never reports a gap.

    Arrays rather than a frame, and an explicit first-row mask rather
    than ``shift``, whose NaN fill would upcast integer values to float:
    ``reach`` is a reported value, not just a comparand.
    """
    start, stop, step = (np.asarray(x) for x in (start, stop, step))
    order = np.argsort(start)
    starts, stops = start[order], stop[order]
    # runs are value-ordered regardless of coordinate orientation, so
    # the continuity margin uses the step magnitude
    steps = np.abs(step[order])
    reach = np.empty_like(stops)
    reach[:1] = stops[:1]
    np.maximum.accumulate(stops[:-1], out=reach[1:])
    # Measure the distance from the reach rather than comparing against
    # `reach + step * tolerance`: a float margin promotes that sum, and
    # an integer coordinate past 2**53 rounds both endpoints together,
    # hiding the gap. Only rows past the reach are measured: a row which
    # starts at or before it cannot open a gap, and subtracting there
    # would wrap an unsigned value into an enormous phantom one.
    has_gap = np.zeros(len(starts), dtype=bool)
    ahead = starts > reach
    if ahead.any():
        has_gap[ahead] = tolerance.is_gap(starts[ahead] - reach[ahead], steps[ahead])
    has_gap[:1] = False
    return order, reach, has_gap


# --- mesh edges for plotting


def _to_numeric(values):
    """Convert timedeltas to seconds while retaining other numeric values."""
    values = np.asarray(values)
    return to_float(values) if is_timedelta64(values) else values


def _normalize_coord_values(values):
    """Normalize coordinate values for gap and edge calculations."""
    values = np.asarray(values)
    if is_datetime64(values):
        return values.astype("datetime64[ns]")
    return _to_numeric(values)


def is_monotonic_and_finite(values) -> bool:
    """Return True when values are finite and strictly monotonic."""
    values = _normalize_coord_values(values)
    if not np.all(np.isfinite(values)):
        return False
    diffs = _to_numeric(np.diff(values))
    return bool(not len(diffs) or np.all(diffs > 0) or np.all(diffs < 0))


def _cell_step(declared, values):
    """A declared step as the magnitude the normalized values are spaced in."""
    if declared is None or pd.isnull(declared):
        return None
    if is_datetime64(values):
        # datetime values keep their dtype, so their step stays a timedelta
        return np.abs(np.asarray(declared).astype("timedelta64[ns]"))[()]
    return np.abs(_to_numeric([declared])[0])


def get_gap_edges(coord, tolerance: GapTolerance | None = None):
    """
    Return cell edges and gap locations for drawing a coordinate as cells.

    Cells follow the centre convention: an edge sits halfway between
    neighbouring values, and across a gap each side gets a cell one step
    wide instead. The step is the coordinate's declared one when it has
    one and the median spacing otherwise. Timedelta coordinates are
    converted to seconds; datetimes are kept for plotting libraries which
    convert them.

    Parameters
    ----------
    coord
        A one-dimensional monotonic coordinate (or its values).
    tolerance
        Which spacings are gaps. None means none are.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Cell edges and a Boolean array marking a gap after each value.

    See Also
    --------
    [`Spool.get_gaps`](`dascore.core.spool.Spool.get_gaps`) answers the
    other gap question: where whole patches fail to meet, read from the
    index rather than from a coordinate's values.
    """
    values = _normalize_coord_values(getattr(coord, "values", coord))
    step = _cell_step(getattr(coord, "step", None), values)
    if len(values) == 1:
        if step is None:
            msg = "Singleton coordinate has no inferred cell width; using a default."
            warnings.warn(msg, UserWarning, stacklevel=2)
            if is_datetime64(values):
                step = np.asarray(np.timedelta64(1, "D")).astype("timedelta64[ns]")[()]
            else:
                step = 1
        return np.asarray([values[0] - step / 2, values[0] + step / 2]), np.zeros(
            0, dtype=bool
        )
    diffs = np.diff(values)
    numeric_diffs = _to_numeric(diffs)
    if step is None:
        step = np.median(np.abs(diffs))
    gap_mask = np.zeros(len(diffs), dtype=bool)
    if tolerance is not None:
        gap_mask = tolerance.is_gap(numeric_diffs, _to_numeric([step])[0])
    if not np.any(gap_mask):
        edges = np.concatenate(
            (
                [values[0] - diffs[0] / 2],
                values[:-1] + diffs / 2,
                [values[-1] + diffs[-1] / 2],
            )
        )
        return edges, gap_mask
    direction = 1 if numeric_diffs[0] > 0 else -1
    signed_step = direction * step
    first_step = signed_step if gap_mask[0] else diffs[0]
    last_step = signed_step if gap_mask[-1] else diffs[-1]
    edges = [values[0] - first_step / 2]
    for index, diff in enumerate(diffs):
        if gap_mask[index]:
            edges.extend(
                [
                    values[index] + signed_step / 2,
                    values[index + 1] - signed_step / 2,
                ]
            )
        else:
            edges.append(values[index] + diff / 2)
    edges.append(values[-1] + last_step / 2)
    return np.asarray(edges), gap_mask
