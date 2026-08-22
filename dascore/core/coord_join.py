"""
Predicting what joining coordinates will produce, from their summaries.

A lazy plan describes each output patch before any patch is loaded. What
that description says about a coordinate must be what assembly will
actually build, or the catalog and the patch tell different stories.

The way to guarantee that is to decide it *once*: this module rebuilds
each member's coordinate from the summary the index stored and runs the
same [`concat_coords`](`dascore.core.coords.concat_coords`) call assembly
runs, then states the result as a summary again. Nothing here reimplements
a joining rule; where a rule cannot be applied to summaries alone the
answer is None, which means "claim nothing", never a guess.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from dascore.core.coords import CoordSummary, concat_coords
from dascore.exceptions import CoordError
from dascore.units import get_quantity
from dascore.utils.misc import get_middle_value


def join_summaries(
    summaries: Sequence[CoordSummary],
    *,
    snap_tolerance: float | None = None,
) -> CoordSummary | None:
    """
    Return the summary of the coordinate joining these members would give.

    Parameters
    ----------
    summaries
        The members' coordinate summaries, in the order they will be
        joined. They are trusted to describe validated coordinates, as
        the index's do.
    snap_tolerance
        Multiplied by the step to bound how far
        [`simplify`](`dascore.core.coords.BaseCoord.simplify`) may move a
        value when absorbing a seam, matching what the assembler passes.
        None performs only exact simplifications.

    Returns
    -------
    The joined summary, or None when the join cannot be decided from
    summaries alone — a member which states no step (an array, a
    segmented or value-less coordinate, labels), members of different
    kinds or units, or values which overlap. The caller then states only
    what it can prove of its own accord.

    Examples
    --------
    >>> from dascore.core.coords import get_coord
    >>> from dascore.core.coord_join import join_summaries
    >>>
    >>> first = get_coord(start=0.0, stop=10.0, step=1.0)
    >>> second = get_coord(start=10.0, stop=20.0, step=1.0)
    >>> joined = join_summaries([first.to_summary(), second.to_summary()])
    >>> assert joined.min == 0.0 and joined.max == 19.0
    """
    if not summaries:
        return None
    if len(summaries) == 1:
        return summaries[0]
    if not all(x.is_range_like and x.len for x in summaries):
        # Values the summary does not carry cannot be joined without
        # reading them, and reading them is what laziness avoids.
        return None
    if len({get_quantity(x.units) for x in summaries}) > 1:
        # One physical coordinate spelled two ways: which spelling the
        # output speaks is assembly's choice, made on the values.
        return None
    coords = [x.to_coord(on_grid=True) for x in summaries]
    try:
        joined = concat_coords(*coords)
    except CoordError:
        # Overlapping, contradictory, or otherwise unjoinable members;
        # loading them will raise, and the row must not pretend otherwise.
        return None
    if snap_tolerance:
        step = joined.step if joined.step is not None else _middle_step(coords)
        if step is not None:
            joined = joined.simplify(snap_tolerance * np.abs(step))
    stated = joined.to_summary()
    if not _rebuilt_faithfully(summaries, coords):
        # A member which does not rebuild into the coordinate it was made
        # from — one written at a precision the index does not store, say
        # — cannot have the join's identity computed from it. The step
        # goes with the identity: a summary which still looked evenly
        # sampled would have the identity recomputed from the very values
        # which did not survive. The envelope holds either way.
        stated = stated.model_copy(update=dict(fingerprint=None, step=None, len=None))
    return stated


def _rebuilt_faithfully(summaries, coords) -> bool:
    """Whether every member came back as the coordinate it was made from."""
    return all(
        x.fingerprint is None or x.fingerprint == y.fingerprint()
        for x, y in zip(summaries, coords, strict=True)
    )


def _middle_step(coords):
    """
    The step a tolerance is scaled by when the join has none of its own.

    The middle of the members' steps, which is what the assembler scales
    by (`dascore.utils.patch._middle_step`); a wider one would absorb a
    seam here that the loaded patch keeps.
    """
    steps = [x.step for x in coords if x.step is not None]
    return get_middle_value(steps) if steps else None
