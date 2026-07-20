"""Shared utilities for IO implementations."""

import numpy as np

from dascore.core.coords import BaseCoord, CoordSegmented, get_coord
from dascore.exceptions import CoordError

# Stored coordinate arrays often carry sub-step jitter (e.g. GPS-stamped DAS
# time). ``CoordSegmented.from_array`` treats every isolated sampling change as
# a seam, so a jittery array explodes into roughly one short segment per
# sample. Past this fraction of segments the segmented form is slower to build,
# larger in memory, and no more exact than a plain monotonic coordinate, so we
# skip segmentation for arrays large enough for the cost to matter.
_MAX_SEGMENT_FRACTION = 0.1
_MIN_SEGMENT_GUARD_SIZE = 1_000


def get_exact_coord(values, units=None) -> BaseCoord:
    """Return an exact coordinate, including for non-monotonic values."""
    # atleast_1d matches get_coord(values=...): a squeezed single-sample
    # array (0-d) becomes a length-1 coordinate rather than a scalar.
    values = np.atleast_1d(np.asarray(values))
    if _is_over_segmented(values):
        return get_coord(data=values, units=units)
    try:
        return CoordSegmented.from_array(values, tolerance=0, units=units)
    except CoordError:
        return get_coord(data=values, units=units)


def _is_over_segmented(values) -> bool:
    """
    Cheaply predict whether ``from_array`` would explode into many segments.

    Mirrors ``CoordSegmented.from_array``'s run detection but stops at the
    segment count, so the degenerate path never materializes the segments.
    """
    if values.ndim != 1 or len(values) < _MIN_SEGMENT_GUARD_SIZE:
        return False
    diffs = np.diff(values)
    zero = diffs[0] - diffs[0]
    if not (np.all(diffs > zero) or np.all(diffs < zero)):
        return False  # non-monotonic; from_array handles its own fallback
    # A diff belongs to a run when it matches a neighbor; isolated diffs seam.
    eq_next = diffs[:-1] == diffs[1:]
    in_run = np.zeros(len(diffs), dtype=bool)
    in_run[1:] |= eq_next
    in_run[:-1] |= eq_next
    segment_count = int(np.count_nonzero(~in_run)) + 1
    return segment_count > _MAX_SEGMENT_FRACTION * len(values)
