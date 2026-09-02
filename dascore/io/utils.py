"""Shared utilities for IO implementations."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

import dascore as dc
from dascore.constants import INVENTORY_ATTRS
from dascore.core.coordmanager import CoordManager
from dascore.core.coords import BaseCoord, CoordSegmented, get_coord
from dascore.exceptions import CoordError, ParameterError, UnitError
from dascore.models import ArrayLike
from dascore.units import convert_units, get_quantity_str
from dascore.utils.misc import _to_slice, _validate_sample_values, unbyte

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


def drop_blank_attrs(attrs: dict, names: Iterable[str]) -> dict:
    """
    Drop the named attrs whose value is blank, in place.

    A field the vendor left empty, or omitted, states nothing. Keeping it
    turns "the file does not say" into a value downstream code can match
    on, so the parse boundary drops it rather than passing the blank along.

    Parameters
    ----------
    attrs
        The parsed attrs, modified in place and returned.
    names
        The attrs to drop when blank. A name is blank when it is missing,
        None, or a string of only whitespace.
    """
    for name in names:
        value = attrs.get(name)
        if value is None or (isinstance(value, str) and not value.strip()):
            attrs.pop(name, None)
    return attrs


def build_patches(
    coords: CoordManager,
    data: ArrayLike,
    attrs: dc.PatchAttrs | Mapping[str, Any] | None = None,
    *,
    attr_cls: type[dc.PatchAttrs] | None = None,
    selection: Mapping[str, Any] | None = None,
) -> list[dc.Patch]:
    """
    Trim a data source to a selection and build the resulting patch list.

    This is the tail most single-patch readers share. It returns one
    patch, or nothing if the selection left no data.

    Parameters
    ----------
    coords
        The coordinates of the untrimmed patch.
    data
        The patch data, often an unread node (eg an h5 dataset).
    attrs
        The patch attributes, or anything convertible to them.
    attr_cls
        The format's PatchAttrs subclass. Defaults to PatchAttrs.
    selection
        A mapping of {dimension_name: selection}, eg {"time": (t1, t2)}.
        None values are dropped, so a read with nothing to trim never
        touches the data source. Passed to `CoordManager.select`, which
        ignores names it doesn't know.
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


def windows_to_slices(
    windows: Mapping[str, Any], dims: Sequence[str], shape: Sequence[int]
) -> tuple[slice, ...]:
    """
    Turn `FiberIO.read_array` windows into one slice per dimension.

    Each window is validated as `Patch.select` validates ``samples=True``
    values and resolved against its dimension's length, so every slice
    comes back with explicit non-negative bounds and ``start <= stop`` (a
    reversed window is empty); a dimension without a window is taken whole.

    Parameters
    ----------
    windows
        Dimension name to ``(start, stop)`` half-open sample indices.
    dims
        The dimensions in the array's stored order.
    shape
        The array's shape, in the same order.
    """
    if unknown := sorted(set(windows) - set(dims)):
        msg = f"Window dimensions {unknown} are not among patch dims {tuple(dims)}."
        raise ParameterError(msg)
    out = []
    for dim, size in zip(dims, shape, strict=True):
        if dim not in windows:
            out.append(slice(0, size))
            continue
        _validate_sample_values(windows[dim])
        span = range(size)[_to_slice(windows[dim])]
        out.append(slice(span.start, max(span.stop, span.start)))
    return tuple(out)


def slice_dataset(dataset, dims: Sequence[str], windows: Mapping[str, Any], shape=None):
    """
    Read the sample windows of an array stored in ``dims`` order.

    ``shape`` defaults to the dataset's own; pass it when the scan grid
    is shorter than the stored array (see `windows_to_slices`).
    """
    shape = dataset.shape if shape is None else shape
    return dataset[windows_to_slices(windows, dims, shape)]


def get_gridded_coord(values, units=None) -> BaseCoord:
    """
    Return a stored coordinate array forced onto an even grid.

    For axes the instrument samples on a fixed grid, where the stored values
    only restate that grid and any departure from it is representation noise.
    Such an array can jitter past the tolerance `get_coord` uses to recognize
    an even coordinate and leave a monotonic coord with no step.

    Parameters
    ----------
    values
        The stored coordinate values.
    units
        Units to attach to the returned coordinate.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.io.utils import get_gridded_coord
    >>> # an even grid restated in float32, as some formats store it
    >>> values = np.linspace(4000.0, 4009.9, 100, dtype=np.float32)
    >>> coord = get_gridded_coord(values.astype(np.float64), units="m")
    >>> float(round(coord.step, 4))
    0.1
    """
    coord = get_coord(data=np.atleast_1d(np.asarray(values)), units=units)
    # A lone sample states no spacing, and snap would invent a step of 1.
    return coord.snap() if len(coord) > 1 else coord


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
