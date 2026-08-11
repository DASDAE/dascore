"""Shared utilities for IO implementations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

import dascore as dc
from dascore.core.coordmanager import CoordManager
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


def build_patches(
    coords: CoordManager,
    data,
    attrs=None,
    *,
    attr_cls: type[dc.PatchAttrs] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> list[dc.Patch]:
    """
    Trim a data source to a selection and build the resulting patch list.

    Most single-patch readers share this tail: apply the caller's
    dimension selections, drop the patch if nothing is left, then attach
    attrs.

    Parameters
    ----------
    coords
        The coordinates of the untrimmed patch.
    data
        The patch data, often an unread node (eg an h5 dataset).
    attrs
        The patch attributes, or anything convertible to them.
    attr_cls
        The PatchAttrs subclass used by the format. Defaults to PatchAttrs.
    selection
        A mapping of {dimension_name: selection}, eg {"time": (t1, t2)}.
        Entries whose value is None are dropped, so a read with nothing
        to trim never touches the data source. Selections are passed to
        `CoordManager.select`, which ignores names it doesn't know.

    Returns
    -------
    A list with one patch, or an empty list if the selection removed
    all the data.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.io.utils import build_patches
    >>>
    >>> patch = dc.get_example_patch()
    >>> patches = build_patches(patch.coords, patch.data, patch.attrs)
    >>> assert len(patches) == 1
    """
    # A def-time default would need dc.PatchAttrs while dascore is still
    # importing this module, so the sentinel is resolved here instead.
    attr_cls = dc.PatchAttrs if attr_cls is None else attr_cls
    # Validate attrs before the selection can short-circuit, so bad metadata
    # still raises on a read which happens to select nothing.
    patch_attrs = attr_cls.from_dict(attrs)
    trim = {i: v for i, v in (selection or {}).items() if v is not None}
    if trim:
        coords, data = coords.select(array=data, **trim)
    if not data.size:
        return []
    # Ellipsis rather than a slice so 0d data (a scalar patch) also loads.
    return [dc.Patch(data=data[...], coords=coords, attrs=patch_attrs)]


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
