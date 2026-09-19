"""Shared utilities for IO implementations."""

from __future__ import annotations

import warnings
from collections.abc import Iterable, Mapping, Sequence
from fractions import Fraction
from typing import Any, cast

import numpy as np

import dascore as dc
from dascore.constants import INVENTORY_ATTRS, snap_type
from dascore.core.coordmanager import CoordManager
from dascore.core.coords import BaseCoord, CoordSegmented, get_coord
from dascore.core.summary import normalize_source_patch_key
from dascore.exceptions import (
    CoordError,
    MissingPatchError,
    ParameterError,
    PatchAttributeError,
    UnitError,
)
from dascore.models import ArrayLike
from dascore.units import convert_units, get_quantity_str
from dascore.utils.misc import _to_slice, _validate_sample_values, iterate, unbyte
from dascore.utils.time import to_exact_fraction


def should_snap(snap: snap_type, name: str) -> bool:
    """Return whether the all/none or named-coordinate option enables snapping."""
    return snap if isinstance(snap, bool) else name in iterate(snap)


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
    if not dims and tuple(shape) == (0,):
        return (slice(0, 0),)  # Legacy empty Patch has no dims and one empty axis.
    out = []
    for dim, size in zip(dims, shape, strict=True):
        if dim not in windows:
            out.append(slice(0, size))
            continue
        _validate_sample_values(windows[dim])
        window = _to_slice(windows[dim])
        if window.step not in (None, 1):
            msg = f"A window is a contiguous range; {dim!r} asked for {windows[dim]!r}."
            raise ParameterError(msg)
        span = range(size)[window]
        out.append(slice(span.start, max(span.stop, span.start)))
    return tuple(out)


def resolve_keyed_source(
    sources: Mapping[str, Any] | Iterable[tuple[str, Any]],
    key,
    where: str = "the resource",
):
    """
    Return the one source a native logical key names.

    An empty resource is missing data. An unknown key, an ambiguous
    keyless resource, or a key naming multiple sources cannot be resolved.
    Native keys do not fall back to positional indices.

    ``sources`` maps each native key to whatever the caller needs back,
    and is read lazily, so an h5py group can be passed as it is. Pass
    ``(key, value)`` pairs instead where a resource can state one key
    twice, so the ambiguity is seen rather than silently resolved.
    """
    key = normalize_source_patch_key(key)
    if isinstance(sources, Mapping):
        # a mapping cannot hold a name twice, so it is read as it is: an
        # h5py group resolves a name without opening its siblings
        mapping = cast("Mapping[str, Any]", sources)
        if not len(mapping):
            raise MissingPatchError(f"No patches in {where}.")
        if key:
            if key not in mapping:
                raise PatchAttributeError(f"No patch named '{key}' in {where}.")
            return mapping[key]
        if len(mapping) > 1:
            msg = f"{where} holds several patches; pass an explicit key."
            raise PatchAttributeError(msg)
        return next(iter(mapping.values()))
    pairs = list(sources)
    if not pairs:
        raise MissingPatchError(f"No patches in {where}.")
    if key:
        found = [value for name, value in pairs if name == key]
        if not found:
            raise PatchAttributeError(f"No patch named '{key}' in {where}.")
        if len(found) > 1:
            msg = f"{where} names '{key}' more than once; it cannot be resolved."
            raise PatchAttributeError(msg)
        return found[0]
    if len(pairs) > 1:
        msg = f"{where} holds several patches; pass an explicit key."
        raise PatchAttributeError(msg)
    return pairs[0][1]


def slice_dataset(
    dataset: ArrayLike,
    dims: Sequence[str],
    windows: Mapping[str, Any],
    shape: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Read the sample windows of an array stored in ``dims`` order.

    ``shape`` defaults to the dataset's own; pass it when an axis of the
    grid `scan` reports is shorter than the stored one, as it is for a
    Terra15 file whose trailing rows were never written.
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
    """
    Return an exact coordinate, including for non-monotonic values.

    Monotonic values keep their runs (`CoordSegmented.from_array`, whose
    dense-array guard keeps a jittery array as one monotonic coordinate);
    anything else keeps its values as an array.
    """
    # atleast_1d matches get_coord(values=...): a squeezed single-sample
    # array (0-d) becomes a length-1 coordinate rather than a scalar.
    values = np.atleast_1d(np.asarray(values))
    try:
        return CoordSegmented.from_array(values, tolerance=0, units=units)
    except CoordError:
        return get_coord(data=values, units=units)


def step_from_rate(rate) -> Fraction | np.timedelta64:
    """
    The time step a file states as a sampling rate in Hz.

    A rate that is a simple number (1024.0, 3000.0, 62.5, 999.9) gives an
    exact `Fraction` step in seconds, which `get_coord` keeps on its grid;
    any other rate gives the nearest nanosecond timedelta, as before. A
    float32 rate recovers only the values it holds exactly, which is the
    honest reading of it.
    """
    frac = to_exact_fraction(rate)
    if frac is not None and frac > 0:
        return 1 / frac
    return dc.to_timedelta64(1 / float(rate))


def step_from_interval(seconds) -> Fraction | np.timedelta64:
    """
    The time step a file states as a sample interval in seconds.

    The exact `Fraction` when the interval is a simple fraction (``1 /
    3000``, ``0.004``), otherwise the nearest nanosecond timedelta, as
    before.
    """
    frac = to_exact_fraction(seconds)
    if frac is not None and frac > 0:
        return frac
    return dc.to_timedelta64(float(seconds))


def selection_windows(
    coords: CoordManager, indexers: Mapping[str, int | slice | np.ndarray]
) -> tuple[dict[str, tuple[int, int]], tuple[slice | np.ndarray, ...]]:
    """Return bounding array windows and residual coordinate indexers."""
    windows, residual = {}, []
    if not coords.dims:
        return windows, ()
    for dim, size in zip(coords.dims, coords.shape, strict=True):
        indexer = indexers.get(dim, slice(None))
        if isinstance(indexer, slice):
            span = range(size)[indexer]
            if not span:
                windows[dim] = (0, 0)
                residual.append(slice(None))
                continue
            start, stop = min(span[0], span[-1]), max(span[0], span[-1]) + 1
            leftover = (
                slice(None)
                if span.step == 1
                else slice(
                    span[0] - start, None if span.step < 0 else stop - start, span.step
                )
            )
        else:
            indices = np.atleast_1d(indexer)
            if not len(indices):
                windows[dim] = (0, 0)
                residual.append(slice(None))
                continue
            start, stop = int(indices.min()), int(indices.max()) + 1
            leftover = indices - start
        windows[dim] = (start, stop)
        residual.append(leftover)
    return windows, tuple(residual)
