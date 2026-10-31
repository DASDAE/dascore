"""Shared utilities for IO implementations."""

from __future__ import annotations

import math
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import INVENTORY_ATTRS, snap_type
from dascore.core.coordmanager import CoordManager
from dascore.core.coords import BaseCoord, _stored_float_dtype, get_coord
from dascore.core.summary import normalize_source_patch_key
from dascore.exceptions import (
    CoordError,
    InvalidFiberFileError,
    MissingPatchError,
    ParameterError,
    PatchAttributeError,
    UnitError,
)
from dascore.models import ArrayLike
from dascore.units import convert_units, get_quantity, get_quantity_str
from dascore.utils.misc import _maybe_unpack, _to_slice, iterate, unbyte
from dascore.utils.patch import _uneven_message
from dascore.utils.time import to_exact_fraction
from dascore.warnings import SnapWarning

_CF_CALENDARS = frozenset(("standard", "gregorian", "proleptic_gregorian"))
# Warnings point past DASCore to the caller which read the file.
_DASCORE_DIR = str(Path(dc.__file__).parent)


def cf_time_values(node, index=slice(None)):
    """Time labels, decoding CF units such as 'milliseconds since 2017-09-18'."""
    values = decode_cf_time(node.attrs, np.atleast_1d(node[index]))
    return values if isinstance(index, slice) else values[0]


def decode_cf_time(attrs, raw) -> np.ndarray:
    """Decode stored times by the CF `units` (and `calendar`) in `attrs`."""
    units = str(unbyte(_maybe_unpack(attrs.get("units", ""))))
    unit, *rest = re.split(r" since ", units, maxsplit=1, flags=re.IGNORECASE)
    origin = rest[0] if rest else ""
    if not origin:
        return dc.to_datetime64(raw)
    calendar = str(unbyte(_maybe_unpack(attrs.get("calendar", "standard"))))
    if calendar.lower() not in _CF_CALENDARS:
        msg = f"Time uses the unsupported CF calendar {calendar!r}."
        raise InvalidFiberFileError(msg)
    try:
        name = cast("Any", unit.strip())  # CF names a unit pandas also knows
        offsets = pd.to_timedelta(raw, unit=name)
        # pandas raises where numpy would wrap past the nanosecond range
        out = (pd.Timestamp(origin.strip()) + offsets).as_unit("ns")
    except (ValueError, OverflowError) as exc:
        msg = f"Cannot decode the time units {units!r}."
        raise InvalidFiberFileError(msg) from exc
    out = out.tz_convert(None) if out.tz is not None else out
    return out.to_numpy()


def should_snap(snap: snap_type, name: str) -> bool:
    """Return whether the all/none or named-coordinate option enables snapping."""
    return snap if isinstance(snap, bool) else name in iterate(snap)


def require_grid(patch, dims: str | Iterable[str], writer: str) -> None:
    """
    Refuse dimensions a format stores only as a grid, saying how to make one.

    Writers keep a coordinate's step and holes, or refuse (rule 13 of the
    coordinate guidelines, `notes/coordinates.qmd`); a format which stores
    a start and a step cannot hold holes, jitter or labels with no step.
    """
    for dim in iterate(dims):
        coord = patch.get_coord(dim)
        if not coord.evenly_sampled:
            raise CoordError(_uneven_message(dim, coord, f"the {writer} writer"))


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


def parse_units(raw_units) -> str | None:
    """Return the unit string if it parses as units, else None (scans don't crash)."""
    try:
        get_quantity(raw_units)
    except Exception:
        return None
    return raw_units


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


_WINDOW_BOUNDS = (int, np.integer, type(None), type(Ellipsis))


def validate_windows(windows: Sequence[Any]) -> tuple[Any, ...]:
    """
    Return `FiberIO.read_array` windows, refusing a spelling which is not one.

    A window is ``None`` (the whole axis), a slice, or a ``(start, stop)``
    pair of sample indices; ``windows`` holds one per axis, in order. A
    bare integer is not a window, so no spelling is read two ways.
    """
    if not isinstance(windows, Sequence) or isinstance(windows, str):
        msg = (
            f"Windows are one positional window per axis; got {windows!r}. "
            f"Use ((0, 10),) or slice(0, 10) to window the first axis."
        )
        raise ParameterError(msg)
    for window in windows:
        if window is None or isinstance(window, slice):
            continue
        pair = isinstance(window, Sequence) and not isinstance(window, str)
        if not (
            pair
            and len(window) == 2
            and all(isinstance(x, _WINDOW_BOUNDS) for x in window)
        ):
            msg = (
                f"Each window is None, a slice, or a (start, stop) pair of "
                f"sample indices; got {window!r} in {windows!r}. Use "
                f"((0, 10),) or slice(0, 10) to window the first axis."
            )
            raise ParameterError(msg)
    return tuple(windows)


def _in_bounds_slices(windows, shape) -> tuple[slice, ...] | None:
    """Fast path for in-bounds integer pairs, as lazy loads send; else None."""
    if type(windows) is not tuple or len(windows) > len(shape):
        return None
    out = []
    for window, size in zip(windows, shape):
        if type(window) is not tuple or len(window) != 2:
            return None
        start, stop = window
        if type(start) is not int or type(stop) is not int:
            return None
        if not 0 <= start <= stop <= size:
            return None
        out.append(slice(start, stop))
    return (*out, *(slice(0, size) for size in shape[len(out) :]))


def windows_to_slices(
    windows: Sequence[Any], shape: Sequence[int]
) -> tuple[slice, ...]:
    """
    Turn `FiberIO.read_array` windows into one slice per axis.

    Each window is checked by `validate_windows` and resolved against its
    axis's length, so every slice comes back with explicit non-negative
    bounds and ``start <= stop`` (a reversed window is empty); an axis
    without a window is taken whole.

    Parameters
    ----------
    windows
        A ``(start, stop)`` half-open sample range, a slice, or ``None``
        for a whole axis, one per axis in order. Trailing axes may be
        left out.
    shape
        The array's shape.
    """
    if (fast := _in_bounds_slices(windows, shape)) is not None:
        return fast
    windows = validate_windows(windows)
    if len(windows) > len(shape):
        msg = (
            f"Windows are one positional range per axis of {tuple(shape)}; "
            f"got {windows!r}."
        )
        raise ParameterError(msg)
    out = []
    for axis, size in enumerate(shape):
        if axis >= len(windows) or windows[axis] is None:
            out.append(slice(0, size))
            continue
        window = _to_slice(windows[axis])
        if window.step not in (None, 1):
            msg = (
                f"A window is a contiguous range; "
                f"axis {axis} asked for {windows[axis]!r}."
            )
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
    windows: Sequence[Any] = (),
    shape: Sequence[int] | None = None,
) -> np.ndarray:
    """
    Read the positional sample windows of a stored array.

    ``shape`` defaults to the dataset's own; pass it when an axis of the
    grid `scan` reports is shorter than the stored one, as it is for a
    Terra15 file whose trailing rows were never written.
    """
    shape = dataset.shape if shape is None else shape
    return dataset[windows_to_slices(windows, shape)]


def stored_coord(
    labels,
    snap: snap_type,
    *,
    name: str,
    where: Any = "",
    decode: Callable[[np.ndarray], np.ndarray] | None = None,
    start=None,
    step=None,
    count: int | None = None,
    units=None,
) -> BaseCoord:
    """
    Return a coordinate a file stores as labels, on a grid if `snap` names it.

    Every reader builds its snapped grids here (rule 3 of the coordinate
    guidelines, `notes/coordinates.qmd`). The grid comes from the start and
    step the file states when both are given, such as a start time and its
    acquisition rate; otherwise it spans the first and last labels in their
    count, so labels from a clock unlocked from sampling still come back
    evenly sampled. A spanning step is a whole number of ticks where that
    keeps the last label within its stored precision, else an exact
    fraction of one. The grid is checked against the end labels, and a grid
    spanning them against the quartile labels too. Where one sits further than
    `dascore.config.max_shift` steps from it, beyond what its stored
    precision resolves (a hole, or a clock running apart), the labels are
    returned as stored with a `SnapWarning` naming the file, as they are
    when they end in missing values or do not advance. Only the checked
    labels are read unless they are kept. A coordinate `snap` does not
    name, or which is not numeric or time-like, comes back exactly as
    stored.

    Parameters
    ----------
    labels
        The stored labels: an array, or anything indexed like one, such as
        an HDF5 dataset.
    snap
        The reader's `snap` option; see `should_snap`.
    name
        The coordinate's name.
    where
        The file, or the open resource, named in the warning.
    decode
        Turns stored labels into coordinate values, such as
        `dascore.to_datetime64` for epoch seconds.
    start
        The first label the file states, as a coordinate value.
    step
        The step the file states, such as `step_from_rate(rate)`.
    count
        How many leading labels are samples; all of them if None.
    units
        The coordinate's units.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.io.utils import stored_coord
    >>> # distances stored with a little noise
    >>> labels = np.arange(10.0) + np.array([0, 1, -1, 0, 2, 0, -2, 1, 0, 0]) / 100
    >>> exact = stored_coord(labels, False, name="distance")
    >>> grid = stored_coord(labels, ("distance",), name="distance")
    >>> assert grid.evenly_sampled and not exact.evenly_sampled
    """
    decode = np.asarray if decode is None else decode
    labels = labels if hasattr(labels, "dtype") else np.asarray(labels)
    count = len(labels) if count is None else count
    snappable = labels.ndim == 1 and np.dtype(labels.dtype).kind in "iufmM"
    if not (should_snap(snap, name) and snappable and count):
        return get_coord(data=decode(labels[:count]), units=units, snap=False)
    stated = start is not None and step is not None
    positions = _checked_positions(count, quartiles=not stated)
    stored = np.asarray(labels[positions.tolist()])
    values = decode(stored)
    if values.dtype.kind in "mM":
        values = values.astype(f"{values.dtype.kind}8[ns]")
    elif values.dtype.kind == "f":  # float32 step arithmetic splits equal spans
        values = values.astype(np.float64)
    resolution, reason = _resolution(stored, decode), None
    if values.dtype.kind in "iu":  # a descending or wide grid needs room
        if values.max() > np.iinfo(np.int64).max:
            reason = "exceed what int64 holds"
        values = values.astype(np.int64)
    if not stated and reason is None:
        start = values[0]
        if count == 1:
            return get_coord(data=values, step=step, units=units)
        if pd.isnull(values[[0, -1]]).any():
            reason = "end in missing values"
        elif not (step := _span_step(start, values[-1], count, resolution)):
            reason = "do not advance"
    if reason is None:
        ticks = _step_ticks(step, values.dtype.kind)
        shift = float(np.max(np.abs(_offsets(values, start, ticks, positions))))
        max_shift = dc.get_config().max_shift
        if not shift <= max_shift + resolution / abs(float(ticks)):  # NaN fails
            how = "stated start and step" if stated else "first and last labels"
            reason = (
                f"sit up to {shift:.2g} steps off the grid built from its {how}, "
                f"more than dascore.config.max_shift ({max_shift}); the file may "
                "hold a gap, or its clock may run apart from its sampling"
            )
    if reason is None:
        return get_coord(start=start, step=step, shape=(count,), units=units)
    _warn_snap(f"the stored {name} labels {reason}", where)
    return get_coord(data=decode(labels[:count]), units=units, snap=False)


def _checked_positions(count: int, quartiles: bool) -> np.ndarray:
    """The positions of the labels a grid is checked against, ascending."""
    ends = [0, count - 1]
    inner = [count // 4, count // 2, (3 * count) // 4] if quartiles else []
    return np.unique(ends + inner)


def _span_step(first, last, count: int, resolution: float = 0.0):
    """
    The step spreading `count` labels from `first` to `last`.

    Time and integer labels get a whole number of ticks where that keeps the
    last label within the labels' stored `resolution`, as a released reader
    gave them, and otherwise an exact fraction of a tick, its denominator
    bounded so that no grid position overflows int64.
    """
    if not isinstance(first, np.generic) or first.dtype.kind not in "iumM":
        return (float(last) - float(first)) / (count - 1)
    timelike = first.dtype.kind in "mM"
    ticks = int((last - first).astype(np.int64)) if timelike else int(last - first)
    exact = Fraction(ticks, count - 1).limit_denominator(2**62 // max(abs(ticks), 1))
    whole = round(exact)
    if whole and abs(whole - exact) * (count - 1) <= max(resolution, 0.5):
        return np.timedelta64(whole, "ns") if timelike else whole
    return exact / 10**9 if timelike else exact


def _step_ticks(step, kind: str) -> Fraction | float:
    """A step in ticks (nanoseconds for time-like ones), exactly if it is."""
    if isinstance(step, np.timedelta64):
        return Fraction(int(step.astype("timedelta64[ns]").astype(np.int64)))
    if kind in "mM":
        return Fraction(step) * 10**9
    return Fraction(step) if kind in "iu" else float(step)


def _offsets(values, start, ticks, positions) -> np.ndarray:
    """
    How far each label sits from the grid's label at its position, in steps.

    Time and integer grids floor each position to a whole tick, as their
    labels do, so a fractional step is judged at the labels it gives.
    """
    if values.dtype.kind in "mM":
        rel = (values - start).astype("timedelta64[ns]").astype(np.int64)
    elif values.dtype.kind in "iu":
        rel = values - int(start)
    else:
        rel = values.astype(np.float64) - float(start)
        return (rel - positions * ticks) / ticks
    grid = np.array([math.floor(k * ticks) for k in positions.tolist()])
    return (rel - grid) / float(ticks)


def _resolution(stored, decode) -> float:
    """
    The spacing of stored float labels near the largest, as decoded.

    A float stores a grid label only as closely as it can hold it, and a
    grid spanning two such labels inherits their rounding. Integer labels
    are held exactly.
    """
    if stored.dtype.kind != "f":
        return 0.0
    dtype = _stored_float_dtype(stored)
    top = dtype.type(np.max(np.abs(stored)))
    pair = np.array([top, np.nextafter(top, dtype.type(np.inf))], dtype=stored.dtype)
    gap = np.abs(np.diff(decode(pair)))[0]
    if isinstance(gap, np.timedelta64):
        return float(gap.astype("timedelta64[ns]").astype(np.int64))
    return float(gap)


def _warn_snap(message: str, where) -> None:
    """Warn that stored labels are kept rather than snapped, naming the file."""
    msg = (
        f"{_resource_name(where)}: {message}, so they are kept as stored; "
        "Patch.snap_coords puts them on a grid."
    )
    warnings.warn(msg, SnapWarning, skip_file_prefixes=(_DASCORE_DIR,))


def _resource_name(where) -> str:
    """How a file, or an open resource reading one, names itself."""
    if isinstance(where, str | Path):
        return str(where) or "a file"
    for attr in ("_dascore_source_path", "filename", "name"):
        value = getattr(where, attr, None)
        if isinstance(value, str | Path) and str(value):
            return str(value)
    return "a file"


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


def selection_windows(
    coords: CoordManager, indexers: Mapping[str, int | slice | np.ndarray]
) -> tuple[tuple[tuple[int, int], ...], tuple[slice | np.ndarray, ...]]:
    """Return bounding array windows and residual coordinate indexers."""
    windows, residual = [], []
    if not coords.dims:
        return (), ()
    for dim, size in zip(coords.dims, coords.shape, strict=True):
        indexer = indexers.get(dim, slice(None))
        if isinstance(indexer, slice):
            span = range(size)[indexer]
            if not span:
                windows.append((0, 0))
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
                windows.append((0, 0))
                residual.append(slice(None))
                continue
            start, stop = int(indices.min()), int(indices.max()) + 1
            leftover = indices - start
        windows.append((start, stop))
        residual.append(leftover)
    return tuple(windows), tuple(residual)
