"""Shared utilities for IO implementations."""

from __future__ import annotations

import warnings

import numpy as np

from dascore.constants import INVENTORY_ATTRS
from dascore.core.coords import BaseCoord, CoordSegmented, get_coord
from dascore.exceptions import CoordError, UnitError
from dascore.units import convert_units, get_quantity_str
from dascore.utils.misc import unbyte

# Stored coordinate arrays often carry sub-step jitter (e.g. GPS-stamped DAS
# time). ``CoordSegmented.from_array`` treats every isolated sampling change as
# a seam, so a jittery array explodes into roughly one short segment per
# sample. Past this fraction of segments the segmented form is slower to build,
# larger in memory, and no more exact than a plain monotonic coordinate, so we
# skip segmentation for arrays large enough for the cost to matter.
_MAX_SEGMENT_FRACTION = 0.1
_MIN_SEGMENT_GUARD_SIZE = 1_000


def get_attr_names(attr_cls) -> set[str]:
    """
    Return the attr names a reader's attr class accepts from a file header.

    Dotted inventory names (``interrogator.serial_number``) name a nested
    inventory fact and so cannot be pydantic fields. Readers which keep only
    the keys their attr class declares filter through this instead of through
    ``model_fields`` alone, or the nested facts would be silently dropped.
    """
    return set(attr_cls.model_fields) | {x for x in INVENTORY_ATTRS if "." in x}


def convert_attr_units(attrs: dict, name: str, to_units: str, from_units="") -> dict:
    """
    Convert one attr to the units patch attrs use, dropping the file's units.

    Patch attrs record each physical quantity in the units the inventory
    documents, so a file's own unit declaration is spent here at the parse
    boundary rather than travelling beside the value as a companion attr.
    A file which declares no units keeps its value: the format's documented
    default is then the assumption, and the reader says so. A file which
    declares units that cannot be used -- unreadable, or of the wrong
    dimension -- has a value of unknown scale, so the value is dropped with
    a warning rather than passed off as canonical.

    Parameters
    ----------
    attrs
        The parsed attrs, modified in place and returned.
    name
        The attr to convert.
    to_units
        The units the patch attr uses.
    from_units
        The units the format documents when it states none of its own, for
        a header whose units live in the key name rather than beside it.
        A declared unit still wins.
    """
    raw_units = unbyte(attrs.pop(f"{name}_units", None)) or from_units
    value = attrs.get(name)
    if value is None or raw_units is None or raw_units == "":
        return attrs
    try:
        attrs[name] = convert_units(
            float(value), to_units=to_units, from_units=get_quantity_str(raw_units)
        )
    except (TypeError, ValueError, UnitError):
        msg = (
            f"Dropping {name}={value!r}: the file states units {raw_units!r}, "
            f"which cannot be converted to {to_units!r}, so the value's scale "
            "is unknown."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)
        attrs.pop(name)
    return attrs


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
