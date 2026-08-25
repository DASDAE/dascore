"""
Utilities for half-open intervals and the values they carry.

Intervals are ``[start, end)`` everywhere: an interval whose start equals
its end is a point marker which covers nothing. These helpers are shared
by anything which lays values along an axis -- the inventory's optical
distance tracks, and the annotation sets which describe patch data.
"""

from __future__ import annotations

import itertools
from functools import partial
from typing import Annotated

import numpy as np
from pydantic import BeforeValidator

from dascore.exceptions import ParameterError


def interval_masks(values, intervals) -> list[np.ndarray]:
    """
    Return, per interval, the mask of values that interval covers.

    Coverage is half-open, ``[start, end)``, with one exception: the end of
    a coverage run belongs to the interval ending there when no half-open
    interval claims it, so the last point of a run is not left out. Point
    markers (equal start and end) cover nothing.

    Parameters
    ----------
    values
        The values, along the interval axis, to test for coverage.
    intervals
        A sequence of (min, max) pairs.

    Examples
    --------
    >>> from dascore.utils.intervals import interval_masks
    >>> masks = interval_masks([0, 1, 2, 3], [(0, 2), (2, 3)])
    >>> masks[0]
    array([ True,  True, False, False])
    >>> masks[1]  # the run ends at 3, so 3 is not left uncovered
    array([False, False,  True,  True])
    """
    values = np.asarray(values, dtype=float)
    spans = [(lo, hi) for lo, hi in intervals]
    claimed = np.zeros(len(values), dtype=bool)
    for lo, hi in spans:
        if lo < hi:
            claimed |= (values >= lo) & (values < hi)
    out = []
    for lo, hi in spans:
        if lo >= hi:  # a point marker covers nothing
            out.append(np.zeros(len(values), dtype=bool))
            continue
        mask = (values >= lo) & (values < hi)
        out.append(mask | ((values == hi) & ~claimed))
    return out


def intervals_overlap(intervals) -> tuple | None:
    """
    Return the first overlapping pair of half-open intervals, or None.

    Empty (point) intervals cover nothing and cannot overlap.

    Parameters
    ----------
    intervals
        A sequence of (min, max) pairs.

    Examples
    --------
    >>> from dascore.utils.intervals import intervals_overlap
    >>> intervals_overlap([(0, 2), (2, 4)]) is None
    True
    >>> intervals_overlap([(0, 3), (2, 4)])
    ((0, 3), (2, 4))
    """
    ordered = sorted(x for x in intervals if x[0] < x[1])
    for first, second in itertools.pairwise(ordered):
        if second[0] < first[1]:
            return first, second
    return None


def clip_intervals(
    items,
    lo: float,
    hi: float,
    outer: float | None = None,
    min_field: str = "distance_min",
    max_field: str = "distance_max",
) -> list:
    """
    Clip interval items to [lo, hi), dropping those left with no coverage.

    Point markers cover nothing but are not nothing: they survive when they
    fall inside the clip, or on its outermost included endpoint.

    Parameters
    ----------
    items
        Models whose bounds are held by ``min_field`` and ``max_field``,
        and which support pydantic's ``model_copy``.
    lo
        The inclusive start of the clip.
    hi
        The exclusive end of the clip.
    outer
        The outermost value the clipped axis holds, if any. A point marker
        sitting exactly there survives, since nothing beyond it can claim it.
    min_field
        Name of the field holding each item's lower bound.
    max_field
        Name of the field holding each item's upper bound.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from dascore.utils.intervals import clip_intervals
    >>> class Span(BaseModel):
    ...     distance_min: float
    ...     distance_max: float
    >>> clipped = clip_intervals([Span(distance_min=0, distance_max=10)], 2, 6)
    >>> clipped[0].distance_min, clipped[0].distance_max
    (2, 6)
    """
    out = []
    for item in items:
        start, end = getattr(item, min_field), getattr(item, max_field)
        if start == end:
            if lo <= start < hi or (outer is not None and start == hi == outer):
                out.append(item)
            continue
        new_lo, new_hi = max(start, lo), min(end, hi)
        if new_hi <= new_lo:
            continue
        out.append(item.model_copy(update={min_field: new_lo, max_field: new_hi}))
    return out


def value_kind(value) -> str:
    """
    Return the value kind which decides an interval group's shape.

    An item with no value states membership, and membership groups may
    overlap; string and numeric groups are single valued where they
    project onto a coordinate. Booleans never arrive here: `normalize_value`
    refuses them.

    Examples
    --------
    >>> from dascore.utils.intervals import value_kind
    >>> value_kind(None), value_kind("car"), value_kind(1)
    ('membership', 'string', 'numeric')
    """
    if value is None:
        return "membership"
    if isinstance(value, str):
        return "string"
    return "numeric"


def normalize_value(value, error: type[Exception] = ParameterError):
    """
    Normalize an interval's value so its Python type survives validation.

    Numpy scalars are unwrapped: pydantic's smart union resolves every numpy
    scalar to float, which would write a numpy 3 back as 3.0. A boolean is
    refused: membership is stated by carrying no value, so true and false
    are not values.

    Parameters
    ----------
    value
        The value to normalize.
    error
        The exception raised when the value is a non-finite number.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.utils.intervals import normalize_value
    >>> normalize_value(np.int64(3))
    3
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        msg = (
            f"Got {value}, but true and false are not values. State "
            "membership by leaving the value unset, and an interval outside "
            "the group by giving it no row at all."
        )
        raise error(msg)
    if isinstance(value, float) and not np.isfinite(value):
        msg = f"A value must be finite; got {value}."
        raise error(msg)
    return value


def interval_value_type(error: type[Exception] = ParameterError):
    """
    Return the annotated type of the one value an interval may carry.

    An item states membership by carrying no value, so a value, when there
    is one, is text or a number, and its Python type must survive
    validation: a 1 must not be written back as 1.0.

    Parameters
    ----------
    error
        The exception raised for a value the interval cannot carry, so
        each caller refuses in the vocabulary of its own subsystem.

    Examples
    --------
    >>> from pydantic import BaseModel
    >>> from dascore.utils.intervals import interval_value_type
    >>> class Item(BaseModel):
    ...     value: interval_value_type() | None = None
    >>> Item(value=1).value
    1
    """
    return Annotated[
        str | int | float, BeforeValidator(partial(normalize_value, error=error))
    ]
