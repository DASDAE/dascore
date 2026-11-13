"""
Chunk planning over the flat patch relation.

Implements the "Chunking formalities" spec: the planner consumes the
catalog's flat relation (one row per patch: `{dim}_min/max/step` envelopes,
`_{dim}_def_key` structural identity, attr columns) and produces a
[`ChunkPlan`](`dascore.utils.chunk_plan.ChunkPlan`) — an outputs table (one row
per output patch) plus a members table binding each output to trimmed
slices of source patches. No patch data is touched; assembly happens later.

Portions of the interval/instruction math were ported from the old
`ChunkManager` (now removed) with the spec's adjudicated corrections
applied; `Spool.chunk` runs on these plans.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import numbers
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any, Literal, NamedTuple

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import CONFLICT, attr_conflict_description
from dascore.core.coords import _GRID_RTOL, _GRID_ULPS, _MIN_FRACTION_TICKS, Grid
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidSpoolQueryError,
    MissingPatchError,
    ParameterError,
    UnitError,
)
from dascore.units import (
    Quantity,
    convert_units,
    get_byte_count,
    get_quantity,
    is_data_size,
    is_percent,
)
from dascore.utils.attrs import known_only, validate_conflict
from dascore.utils.chunk import _GRID_SNAP_RTOL, _READ_PAD, get_intervals
from dascore.utils.docs import compose_docstring
from dascore.utils.explicit_ranges import (
    ExplicitRanges,
    explicit_ranges,
    known_coordinates,
)
from dascore.utils.gaps import (
    DEFAULT_TOLERANCE,
    GapTolerance,
    _quantity_to_dim_value,
    contiguous_runs,
)
from dascore.utils.misc import (
    _CanonicalRange,
    express_range_for_coord,
    get_middle_value,
    is_range,
    order_range_tuple,
    stated_units,
)
from dascore.utils.pd import (
    adjust_segments,
    get_dim_names_from_columns,
    get_interval_columns,
)
from dascore.utils.time import (
    _MAX_NS,
    is_datetime64,
    is_timedelta64,
    to_datetime64,
    to_float,
    to_int,
    to_timedelta64,
)
from dascore.warnings import DASCoreWarning

# Columns which never participate in conflict policing and never carry to
# outputs: source bookkeeping (outputs are not file rows) and the two
# lineage ids, which every patch states differently and a merged patch
# folds from its members rather than inheriting from any one of them.
_SOURCE_COLUMNS = (
    "source_path",
    "source_format",
    "source_version",
    "source_patch_key",
    "origin_id",
    "data_id",
    # The ids' former names, which attrs pickled before the rename hold.
    "patch_id",
    "processing_id",
)
_PATCH_LOCAL_EMPTY = "_patch_local_empty"


@dataclass(frozen=True)
class ChunkPlan:
    """
    A materialization-free description of a chunk operation.

    Attributes
    ----------
    outputs
        One row per output patch: `{dim}_min/max/step` for the chunked
        dimension, an `output_id`, and all carried columns (group attrs,
        dims, structural def keys, conflict-policed attrs).
    members
        Instruction rows binding outputs to sources: `output_id`,
        `_patch_row`, the exact `{dim}_min/max` trim for that member, and
        `_modified` (False when the member loads whole). A chunk plan's
        members also say why each starts where it does
        (`reason`), how far its stretch's grid moved its labels (`shift`,
        in steps) and how many samples an earlier patch's overlap dropped
        from it (`dropped`); a member of a stretch whose grid relabels it
        carries its output's slice of that grid (`_{dim}_stretch`).
    dim
        The chunked dimension.
    value
        The requested chunk length (None for merge mode).
    params
        Resolved parameters (group attrs, tolerances, max_shift, overlap,
        keep_partial, conflict, on_missing_dim) — recorded, not referencing
        config.
    trim_dims
        Dimensions besides `dim` a window plan trims (in `{dim}_min/max`).
    """

    outputs: pd.DataFrame
    members: pd.DataFrame
    dim: str
    value: Any
    params: dict = field(default_factory=dict)
    trim_dims: tuple[str, ...] = ()

    @property
    def merge_mode(self) -> bool:
        """Return True when this plan merges (no segmenting length)."""
        return self.value is None


def _stretches(df: pd.DataFrame, dim: str, indexed, read):
    """
    The rows cut at their holes, and each stretch clipped to its row.

    Returns the positions of the rows cut, and ``(position, low, high,
    step)`` for each stretch which survives its row's envelope.
    """
    lo, hi = f"{dim}_min", f"{dim}_max"
    cut, out = [], []
    if not indexed or df.empty or not {lo, hi, "_patch_row"} <= set(df.columns):
        return cut, out
    candidates = df["_patch_row"].isin(list(indexed)) & df[lo].notna()
    for index in np.flatnonzero(candidates.to_numpy()):
        row = df.iloc[index]
        runs = indexed[row["_patch_row"]]
        records = runs.records
        if records is None and (coord := read(row)) is not None:
            records = coord._run_records()
        stretches = contiguous_runs(records or (), runs.step, runs.jittered)
        if len(stretches) < 2:
            continue
        cut.append(index)
        for stretch in stretches:
            low, high = max(stretch.min, row[lo]), min(stretch.max, row[hi])
            if low <= high:
                out.append((index, low, high, stretch.step))
    return cut, out


def with_runs(df: pd.DataFrame, dim: str, indexed, read, modified=False):
    """
    The relation with each row cut into the stretches its coordinate's runs leave.

    ``indexed`` maps a ``_patch_row`` to the runs its index states (an
    `IndexedRuns`), and ``read`` returns a row's coordinate where the index
    stored it without them. Each stretch
    [`contiguous_runs`](`dascore.utils.gaps.contiguous_runs`) joins is the
    row with the stretch's envelope and step, clipped to the row's own
    envelope, so a selection trims it; a row none of whose stretches
    survive the clip lies in a hole and is dropped. ``modified`` marks the
    stretches to load as selections of their patch, and `coalesce_runs`
    reads the stretches landing in one output back as one member.
    """
    cut, stretches = _stretches(df, dim, indexed, read)
    if not cut:
        return df
    split = df.iloc[[x[0] for x in stretches]].copy()
    names = (f"{dim}_min", f"{dim}_max", f"{dim}_step")
    for num, col in enumerate(names, start=1):
        _set_column(split, col, [x[num] for x in stretches], df)
    if modified:
        split["_modified"] = True
    kept = df.drop(index=df.index[cut])
    out = pd.concat([kept, split], ignore_index=True)
    if modified:
        out["_modified"] = out["_modified"].astype("boolean").fillna(False).astype(bool)
    return out


def clip_to_runs(df: pd.DataFrame, dim: str, indexed, read) -> pd.DataFrame:
    """
    The relation with each row's envelope narrowed to the samples it holds.

    A row whose envelope a selection trimmed may start or end in a hole of
    its coordinate; its envelope shrinks to the stretches the trim reaches,
    and a row reaching none lies in a hole and is dropped. Arguments are as
    for `with_runs`.
    """
    cut, stretches = _stretches(df, dim, indexed, read)
    if not cut:
        return df
    reached = pd.DataFrame(stretches, columns=["row", "low", "high", "step"])
    bounds = reached.groupby("row").agg(low=("low", "min"), high=("high", "max"))
    out = df.copy(deep=False)
    rows = bounds.index.to_numpy(dtype=np.int64)
    for col, values in ((f"{dim}_min", bounds["low"]), (f"{dim}_max", bounds["high"])):
        column = out[col].astype(object).to_numpy(copy=True)
        column[rows] = values.to_numpy()
        _set_column(out, col, column, df)
    empty = sorted(set(cut) - set(rows.tolist()))
    return out.drop(index=out.index[empty])


def _set_column(frame: pd.DataFrame, col: str, values, like: pd.DataFrame) -> None:
    """Set a column of python values, in the dtype ``like`` holds it in."""
    column = pd.Series(list(values), index=frame.index, dtype=object)
    dtype = like[col].dtype if col in like.columns else np.dtype(object)
    frame[col] = column if dtype == np.dtype(object) else column.astype(dtype)


def coalesce_runs(plan: ChunkPlan, working: pd.DataFrame) -> ChunkPlan:
    """
    Merge each output's consecutive members cut from one patch's runs.

    A patch cut by `with_runs` plans stretch by stretch (its rows share one
    `_patch_row`), so a hole can end an output; the stretches landing in
    one output are read from their patch once, as one member spanning
    them.
    """
    members = plan.members
    ids = working["_patch_row"]
    split = set(ids[ids.duplicated(keep=False)])
    if not split or members.empty:
        return plan
    lo, hi = f"{plan.dim}_min", f"{plan.dim}_max"
    out, pid = members["output_id"], members["_patch_row"]
    same = (out == out.shift()) & (pid == pid.shift()) & pid.isin(split)
    if not same.any():
        return plan
    piece = (~same).cumsum()
    merged = members[~same].copy()
    merged[lo] = members.groupby(piece, sort=False)[lo].min().to_numpy()
    merged[hi] = members.groupby(piece, sort=False)[hi].max().to_numpy()
    return replace(plan, members=merged.reset_index(drop=True))


def _kind_codes(df: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    """
    Label each row by kind: rows sharing a label hold equal values.

    A plan partitions a whole relation at once, which needs an
    equivalence relation, so kind values are compared for equality: a
    missing value (null or "") is a value like any other, equal to
    another missing one and nothing else. `known_only` is what makes the
    two spellings one value.
    """
    names = list(names)
    if not names or df.empty:
        return pd.Series(0, index=df.index, dtype=np.int64)
    values = known_only(df[names].astype(object))
    return values.groupby(names, dropna=False, sort=False).ngroup()


def _resolve_group_attrs(group, columns) -> tuple[str, ...]:
    """Resolve the group attrs: per-call > config; explicit names must exist."""
    # A name repeated says nothing twice; deduping here keeps it out of the
    # recorded params as well as out of the groupby, which cannot take
    # duplicate labels.
    if group is not None:
        group = (group,) if isinstance(group, str) else tuple(group)
        if missing := [x for x in group if x not in columns]:
            msg = (
                f"group attribute(s) {missing} do not exist on any patch in the spool."
            )
            raise InvalidSpoolQueryError(msg)
        return tuple(dict.fromkeys(group))
    # Config (and default) names are best-effort.
    names = (x for x in dc.get_config().patch_kind_attrs if x in columns)
    return tuple(dict.fromkeys(names))


def _adjust_unit_segments(df, name, canonical):
    """Trim envelopes by a canonical range expressed in each row's units."""
    unit_col = f"_{name}_units"
    if df.empty:
        return df
    if unit_col not in df.columns:
        return adjust_segments(
            df, ignore_bad_kwargs=True, **{name: canonical.magnitudes}
        )
    pieces = []
    for unit, sub in df.groupby(df[unit_col], dropna=False, sort=False):
        if unit is None or pd.isnull(unit) or unit == "":
            rng = canonical.magnitudes
        else:
            rng = canonical.magnitudes_in(str(unit))
        pieces.append(adjust_segments(sub, ignore_bad_kwargs=True, **{name: rng}))
    return pd.concat(pieces).sort_index()


def _adjust_absolute_envelopes(df, coords):
    """Project one absolute coordinate selection onto relation envelopes."""
    for name, value in coords.items():
        # A row lacking the coordinate loads untrimmed (see
        # `apply_exact_residuals`), so its envelopes stand as they are. The
        # stored dtype marks a held coordinate; the identity key does not,
        # being null for one with no data id.
        column = df.get(f"_{name}_coord_dtype")
        missing = np.zeros(len(df), bool) if column is None else column.isna().values
        held, lacking = df[~missing], df[missing]
        if isinstance(value, _CanonicalRange):
            held = _adjust_unit_segments(held, name, value)
        elif is_range(value):
            held = adjust_segments(held, ignore_bad_kwargs=True, **{name: value})
        df = held
        if len(lacking):
            lacking = lacking.assign(_modified=lacking.get("_modified", False))
            df = pd.concat([held, lacking]).sort_index(kind="stable")
    return df


def _relative_bound(mins, maxs, value, open_values, units=None):
    """
    Resolve one patch-local relative bound for every relation row.

    Returns values, whether projection failed, and the fixed origin if known.
    """
    is_quantity = hasattr(value, "units")
    if value is None or value is Ellipsis:
        return open_values, False, None
    if not is_quantity and np.ndim(value) == 0 and pd.isnull(value):
        return open_values, False, None
    try:
        was_percent = is_percent(value)
        sign = value.magnitude if is_quantity else to_float(value)
        if np.ndim(sign):
            return open_values, True, None
        reference = mins if sign >= 0 else maxs
        time_like = is_datetime64(mins) or is_timedelta64(mins)
        if not np.isfinite(sign) and time_like:
            # A non-finite offset becomes NaT, which `Patch.select` ignores,
            # so the patch loads whole and the envelope stays open. A numeric
            # coordinate takes the infinity literally and the arithmetic
            # below already says so: +inf from the start selects nothing.
            return open_values, False, None
        if was_percent:
            offset = (value.magnitude / 100) * (maxs - mins)
        elif is_quantity:
            target_units = "s" if time_like else units
            if target_units is None or pd.isnull(target_units) or target_units == "":
                raise UnitError("Cannot project a quantity without coordinate units.")
            magnitude = convert_units(value.magnitude, target_units, value.units)
            offset = to_timedelta64(magnitude) if time_like else magnitude
        elif time_like:
            offset = to_timedelta64(value)
        else:
            offset = value
        origin = None if was_percent else ("min" if sign >= 0 else "max")
        return reference + offset, False, origin
    except (NotImplementedError, TypeError, UnitError, ValueError):
        # Keep the source envelope as a candidacy superset. The loaded
        # coordinate remains authoritative and applies the exact selection.
        return open_values, True, None


def _relative_endpoints(df, source, previous, names):
    """Recover sampled endpoints before resolving another relative window."""
    # Circular import: patch assembly imports the index and its plan utilities.
    from dascore.utils.patch_assembly import coord_from_row  # noqa: PLC0415

    uncertain = {}
    previous_names = {name for coords, _, _ in previous for name in coords}
    if not previous_names:
        return df, uncertain
    # Relative projections flag every trim or unresolved window. Rows known
    # to load whole still use their source origins, including associated axes.
    whole = set()
    if "_modified" in df and all(r and not s for _, s, r in previous):
        whole = set(df.index[~df["_modified"]])
    for name in names:
        selections = [(c[name], s, r) for c, s, r in previous if name in c]
        cols = [f"{name}_min", f"{name}_max"]
        if not set(cols).issubset(df.columns):
            continue
        unknown = set()
        for index in df.index:
            if index in whole:
                continue
            row = source.loc[index]
            dims = set(str(row.get("dims", "")).split(","))
            if "dims" in row and (name not in dims or previous_names - dims):
                # Selecting an associated coordinate can trim another axis;
                # envelopes do not describe that relationship sample by sample.
                unknown.add(index)
                continue
            if not selections:
                continue
            unit = row.get(f"_{name}_units")
            units = unit if isinstance(unit, str) and unit else None
            coord = coord_from_row(row, name, units=units)
            if coord is None:
                # An irregular envelope does not say which sample an earlier
                # trim reached. Do not reject its patch using guessed endpoints.
                unknown.add(index)
                continue
            for selection, samples, relative in selections:
                if not len(coord):
                    break
                selection = express_range_for_coord(selection, coord)
                coord, _ = coord.select(selection, samples=samples, relative=relative)
            if not len(coord):
                if _PATCH_LOCAL_EMPTY not in df:
                    df[_PATCH_LOCAL_EMPTY] = False
                df.loc[index, _PATCH_LOCAL_EMPTY] = True
            else:
                df.loc[index, cols] = coord.min(), coord.max()
        uncertain[name] = unknown
    return df, uncertain


def _adjust_relative_envelopes(df, coords, uncertain):
    """Project patch-local relative ranges onto relation envelopes."""
    if "_data_size" in df:
        df = df[df["_data_size"].ne(0).fillna(True)]
    for name, value in coords.items():
        cols = [f"{name}_min", f"{name}_max"]
        if not set(cols).issubset(df.columns) or not is_range(value):
            df["_modified"] = True
            continue
        dtype_col = f"_{name}_coord_dtype"
        if dtype_col in df:
            # Index arithmetic cannot prove a no-op at narrower precision.
            # Leave size and identity unknown until the patch is selected.
            df["_modified"] = df.get("_modified", False) | df[dtype_col].isin(
                ("float16", "float32")
            )
        unit_col = f"_{name}_units"
        group_cols = [c for c in (unit_col, dtype_col) if c in df.columns]
        grouped = (
            df.groupby(group_cols, dropna=False, sort=False, observed=True)
            if group_cols
            else [(None, df)]
        )
        pieces = []
        for _, sub in grouped:
            mins, maxs = (sub[c] for c in cols)
            # Relative arithmetic must round as the loaded coordinate does:
            # widening a float32 bound can turn one selected sample into an
            # inverted window and wrongly remove the patch.
            dtype = sub[dtype_col].iloc[0] if dtype_col in sub else None
            if isinstance(dtype, str) and dtype in ("float16", "float32"):
                mins, maxs = mins.astype(dtype), maxs.astype(dtype)
            lo, hi = value
            unit = sub[unit_col].iloc[0] if unit_col in sub else None
            units = None if unit is None or pd.isnull(unit) or unit == "" else str(unit)
            left, left_unresolved, left_origin = _relative_bound(
                mins, maxs, lo, mins, units
            )
            right, right_unresolved, right_origin = _relative_bound(
                mins, maxs, hi, maxs, units
            )
            unresolved = mins.isna() | maxs.isna()
            unresolved |= sub.index.isin(uncertain.get(name, ()))
            if left_unresolved or right_unresolved:
                unresolved |= True
            # Fixed offsets from the same endpoint can cross even when an
            # earlier trim left that endpoint unknown. Shrinking an envelope
            # also cannot rescue a crossed minimum-to-maximum fixed window.
            # Floating-point offsets must remain distinct after rounding.
            same_origin = left_origin is not None and left_origin == right_origin
            min_to_max = left_origin == "min" and right_origin == "max"
            crossed = (left > right) & (same_origin or min_to_max)
            if pd.api.types.is_float_dtype(left):
                extent = np.maximum(left.abs(), right.abs()) + (maxs - mins).abs()
                roundoff = 4 * np.finfo(left.dtype).eps * extent
                crossed &= (left - right) > roundoff
            new_min = left.mask(unresolved, mins)
            new_max = right.mask(unresolved, maxs)
            # Test before clipping so an entirely out-of-range window is not
            # resurrected as a one-sample envelope.
            keep = (new_min <= new_max) & (new_min <= maxs) & (new_max >= mins)
            keep |= unresolved
            keep &= ~crossed
            if _PATCH_LOCAL_EMPTY in sub:
                keep &= ~sub[_PATCH_LOCAL_EMPTY]
            # Preserve whole-source metadata for relative no-ops. Unknown
            # projections still cannot claim that the source loads whole.
            sub["_modified"] = (
                sub.get("_modified", False)
                | unresolved
                | ~keep
                | (new_min > mins)
                | (new_max < maxs)
            )
            sub[cols[0]] = new_min.clip(lower=mins, upper=maxs)
            sub[cols[1]] = new_max.clip(lower=mins, upper=maxs)
            pieces.append(sub[keep])
        df = pd.concat(pieces).sort_index() if pieces else df.iloc[:0]
    return df


# What a row's samples on one dimension are known to be, as a selection
# is replayed: not yet selected, followed position by position, or lost
# to a selection the relation could not follow.
_UNTOUCHED, _FOLLOWED, _LOST = 0, 1, 2


def _set_cells(frame, column, rows, values) -> None:
    """Set the cells of an object column in ``rows`` to values that may be tuples."""
    cells = frame[column].to_numpy(dtype=object, copy=True)
    for position, value in zip(frame.index.get_indexer(rows), values, strict=True):
        cells[position] = value
    frame[column] = cells


def _grid_terms(coord) -> tuple:
    """The exact grid a relation row states for an evenly sampled coordinate."""
    return (*coord.runs[0].canonical()[1:], len(coord))


def _alike(columns) -> tuple[np.ndarray, np.ndarray]:
    """
    Group the rows alike in every column.

    Returns each row's group, numbered in the order the groups first
    appear, and the position of each group's first row.
    """
    codes = np.zeros(len(columns[0]), dtype=np.int64)
    for values in columns:
        part, found = pd.factorize(values, use_na_sentinel=False)
        codes = pd.factorize(codes * len(found) + part)[0]
    return codes, np.unique(codes, return_index=True)[1]


def _held_coord(row: Mapping, name: str):
    """
    The samples of ``name`` a relation row holds, or None if it cannot say.

    A plan member holds the part of its source inside its bounds, selected
    as loading selects it; any other row holds the coordinate its envelope
    and grid state. A row already trimmed keeps its source's grid, which
    its envelope no longer ends on, and so states no coordinate.
    """
    # Circular import: patch assembly imports the index and its plan utilities.
    from dascore.utils.patch_assembly import (  # noqa: PLC0415
        _as_numpy,
        _cut_coord,
        _sized_coord,
    )

    unit = row.get(f"_{name}_units")
    units = unit if isinstance(unit, str) and unit else None
    if (coord := _cut_coord(row, name, units)) is None:
        # a frame hands times back as pandas scalars, which `grid_fits` refuses
        start, stop = (_as_numpy(row.get(f"{name}_{x}")) for x in ("min", "max"))
        grid = row.get(f"_{name}_grid")
        ticks = _value_family(start) not in ("number", "text")
        if isinstance(grid, tuple) and ticks and not grid_fits(grid, start, stop):
            return None
        coord = _sized_coord(row, name, units=units)
    return None if coord is None or coord._partial else coord


class _Held(NamedTuple):
    """The coordinate each row of a relation holds of one name."""

    # each row's group; rows alike in what they hold share one
    codes: np.ndarray
    # each group's coordinate, None where the row cannot say
    coords: list
    # each row's sample count, -1 where the row cannot say
    sizes: np.ndarray
    # how far each row's labels sit past its group's; None when rows are
    # grouped by their labels themselves, and so sit nowhere past them
    shift: np.ndarray | None


def _held_coords(source, name) -> _Held:
    """
    `_held_coord` for every row of ``source``, rebuilt once per group.

    Rows alike in every value the coordinate is rebuilt from share one.
    On a datetime or timedelta coordinate alike means alike counted from
    the row's own start, so an archive of files that differ only in when
    they start rebuilds one coordinate, and each row's labels are its
    group's shifted by the difference in start.
    """
    low, high = f"{name}_min", f"{name}_max"
    private = ("grid", "coord_dtype", "units", "units_source", "src_low")
    private += ("src_high", "src_step")
    columns = [low, high, f"{name}_step", *(f"_{name}_{x}" for x in private)]
    columns = [c for c in columns if c in source]
    keys = {c: source[c].to_numpy() for c in columns}
    labels = [c for c in (low, high, f"_{name}_src_low", f"_{name}_src_high")]
    labels = [c for c in labels if c in keys]
    start, shift = keys[low], None
    if start.dtype.kind in "mM" and all(keys[c].dtype == start.dtype for c in labels):
        keys.update({c: keys[c] - start for c in labels})
        shift = start
    codes, firsts = _alike(list(keys.values()))
    if shift is not None:
        shift = shift - shift[firsts][codes]
    rows = zip(*(source[c].iloc[firsts].tolist() for c in columns), strict=True)
    coords = [_held_coord(dict(zip(columns, row, strict=True)), name) for row in rows]
    sizes = np.array([-1 if x is None else len(x) for x in coords], dtype=np.int64)
    return _Held(codes, coords, sizes[codes], shift)


def _sliced(lo, hi, start, stop):
    """The first and last positions a Python slice keeps of ``lo`` through ``hi``."""
    size = hi - lo + 1
    ends = []
    for index, default in ((start, 0), (stop, size)):
        if index is None:
            ends.append(default)
        elif index >= 0:
            ends.append(np.minimum(index, size))
        else:
            ends.append(np.maximum(size + index, 0))
    return lo + ends[0], lo + ends[1] - 1


def _span(coord, low, high) -> tuple[int, int, int, int]:
    """
    The first and last positions of ``coord`` with labels in a value range.

    Given twice: the widest span rounding allows, then the narrowest. A
    float label is known only to rounding, so a bound within a tenth of a
    step of one may or may not keep it; time and integer labels are exact
    and the two agree. None leaves an end open. An empty span has its
    first position past its last.
    """
    # Circular import: patch assembly imports the index and its plan utilities.
    from dascore.utils.patch_assembly import _READ_PAD, _as_numpy  # noqa: PLC0415

    low, high = (None if pd.isnull(x) else _as_numpy(x) for x in (low, high))
    pads = [0]
    if np.asarray(coord.min()).dtype.kind == "f":
        pad = abs(to_float(coord.step)) * _READ_PAD
        pads = [pad, -pad]
    out = []
    for pad in pads:
        first = low if low is None or not pad else low - pad
        last = high if high is None or not pad else high + pad
        if first is not None and last is not None and first > last:
            out += [1, 0]
            continue
        start, stop, _ = coord.select((first, last))[1].indices(len(coord))
        out += [start, stop - 1]
    return tuple(out * (2 // len(pads)))


def _relative_span(coord, value, lows, highs) -> tuple[int, int, int, int]:
    """
    `_span` of a relative range on a coordinate kept from ``coord``.

    ``lows`` and ``highs`` hold the first and last positions the row may
    have kept. A relative bound counts from the kept coordinate's own
    ends, so each pairing is tried, and the answer spans them all. Any
    pairing that keeps nothing makes the span empty: the row may be.
    """
    found = []
    for lo in sorted(set(lows)):
        for hi in sorted(set(highs)):
            kept = coord[int(lo) : int(hi) + 1]
            range_ = express_range_for_coord(value, kept)
            low, high = kept._get_slice_tuple(range_, relative=True)
            if low is not None and high is not None and low > high:
                return 1, 0, 1, 0
            span = _span(coord, low, high)
            found.append(
                (max(span[0], lo), min(span[1], hi), max(span[2], lo), min(span[3], hi))
            )
    if any(x[0] > x[1] for x in found):
        return 1, 0, 1, 0
    firsts, lasts, in_firsts, in_lasts = zip(*found, strict=True)
    return min(firsts), max(lasts), max(in_firsts), min(in_lasts)


def _outward(values, steps, scales, dtypes, sign) -> np.ndarray:
    """
    Float labels moved toward ``sign`` as far as a stored label may sit.

    A label rebuilt from a start and step is not the one the file holds.
    A float coordinate is evenly sampled only while its labels spread no
    more than twice `_float_allowance` about the grid through the first,
    so a stored label is within that of its grid position; a third
    allowance, four spacings of the largest label or more, covers the
    rounding of the rebuilt label itself. ``scales`` is each row's
    largest label in size, which the allowance is measured at in the
    dtype the file stores.
    """
    out = values.astype(np.float64)
    names = pd.Series(dtypes, dtype=object)
    names = names.where(names.map(lambda x: isinstance(x, str)), "float64").to_numpy()
    for name in pd.unique(names):
        dtype = np.dtype(name) if np.dtype(name).kind == "f" else np.dtype("float64")
        rows = names == name
        ulps = _GRID_ULPS * np.spacing(scales[rows].astype(dtype)).astype(np.float64)
        allowance = np.maximum(
            _GRID_RTOL * np.abs(steps[rows].astype(np.float64)), ulps
        )
        out[rows] += sign * 3 * allowance
    return out


def _as_dtype(values, dtype, sign) -> np.ndarray:
    """Envelope ends cast to a narrower float column, rounded outward."""
    if not (dtype.kind == "f" and values.dtype.kind == "f"):
        return values
    cast = values.astype(dtype)
    inside = (cast.astype(np.float64) - values) * sign < 0
    out = np.nextafter(cast, np.asarray(sign * np.inf, dtype=dtype))
    return np.where(inside, out, cast)


class _Kept:
    """
    Which of the samples each row held on one dimension it still holds.

    Positions count into the coordinate the row held before any
    selection. ``lo`` and ``hi`` are the first and last sample kept at
    the widest, ``in_lo`` and ``in_hi`` at the narrowest; they differ
    where float rounding leaves an end sample in doubt, and the narrowest
    is empty (``in_lo`` past ``in_hi``) where it leaves none certain.
    """

    def __init__(self, count: int):
        self.status = np.full(count, _UNTOUCHED, dtype=np.int8)
        self.size, self.lo, self.hi, self.in_lo, self.in_hi = (
            np.zeros(count, dtype=np.int64) for _ in range(5)
        )
        # rows a selection moved since the envelopes were last trimmed
        self.moved = np.zeros(count, dtype=bool)

    def narrow(self, rows, lo, hi, in_lo, in_hi) -> None:
        """Keep, of ``rows``, the samples inside the spans given."""
        self.lo[rows] = np.maximum(self.lo[rows], lo)
        self.hi[rows] = np.minimum(self.hi[rows], hi)
        self.in_lo[rows] = np.maximum(self.in_lo[rows], in_lo)
        self.in_hi[rows] = np.minimum(self.in_hi[rows], in_hi)
        gone = self.lo[rows] > self.hi[rows]
        # a row that may hold nothing is left to the selected coordinate
        self.status[rows[gone]] = _LOST
        self.moved[rows[~gone]] = True


class _SampleTracker:
    """
    Follow the samples each row keeps as selections are replayed.

    `Patch.select` on a coordinate cuts every coordinate on its dimension
    by the same samples. An envelope cannot carry that from one
    coordinate to another: it may sit on a query bound between samples,
    or be a superset. So the samples themselves are followed, by position,
    wherever the selected coordinate rebuilds from its row (see
    `_held_coord`), and each coordinate on the dimension takes its ends
    from the positions kept. Where a selection cannot be followed the
    dimension is lost for that row: its envelopes stay as they are, which
    is a superset, and nothing is counted from them again.

    Rows are positions in ``source``, the relation before any selection.
    """

    def __init__(self, source: pd.DataFrame):
        self.source = source
        names = {
            str(c)[1 : -len(end)]
            for c in source.columns
            for end in ("_dims", "_grid")
            if str(c).startswith("_") and str(c).endswith(end)
        }
        ends = ("_min", "_max")
        names = {x for x in names if {x + e for e in ends}.issubset(source.columns)}
        self.names = sorted(names)
        # a frame stating no dimensions has each coordinate on its own
        own = {x: np.full(len(source), x, dtype=object) for x in self.names}
        self.dims = {
            x: source[f"_{x}_dims"].to_numpy(dtype=object)
            if f"_{x}_dims" in source
            else own[x]
            for x in self.names
        }
        self.kept: dict[str, _Kept] = {}
        self._held: dict[str, _Held] = {}
        self._watched: dict[str, bool] = {}

    def rows(self, index) -> np.ndarray:
        """The rows of ``source`` the labels of ``index`` name."""
        return self.source.index.get_indexer(index)

    def held(self, name) -> _Held:
        """The coordinate each row holds of ``name``, rebuilt once."""
        if name not in self._held:
            self._held[name] = _held_coords(self.source, name)
        return self._held[name]

    def watched(self, name) -> bool:
        """
        Whether selecting ``name`` moves anything but its own envelope.

        It does where another coordinate shares its dimension, where it
        spans several, or where it is on an exact grid, whose labels a
        sample selection reads from the samples kept. Most selections are
        on a dimension nothing else is on, and are not followed at all.
        """
        if name not in self.dims:
            return False
        if name not in self._watched:
            mine = self.dims[name]
            held = pd.notna(mine)
            others = (self.dims[x] for x in self.names if x != name)
            shared = any((x[held] == mine[held]).any() for x in others)
            several = any("," in str(x) for x in pd.unique(mine[held]))
            grid = self.source.get(f"_{name}_grid", ())
            exact = any(isinstance(x, tuple) for x in grid)
            self._watched[name] = shared or several or exact
        return self._watched[name]

    def _parts(self, name, rows):
        """Yield each dimensions cell ``name`` has in ``rows``, with its rows."""
        cells = self.dims[name][rows]
        for cell in pd.unique(cells):
            if isinstance(cell, str):
                yield cell, rows[cells == cell]

    def _state(self, key) -> _Kept:
        return self.kept.setdefault(key, _Kept(len(self.source)))

    def lose(self, name, rows) -> None:
        """Record a selection on ``name`` that ``rows`` cannot be followed through."""
        if not self.watched(name):
            return
        for cell, part in self._parts(name, rows):
            for key in cell.split(","):
                self._state(key).status[part] = _LOST

    def lost(self, name, rows) -> np.ndarray:
        """Whether an earlier selection on its dimension left each row unfollowed."""
        out = np.zeros(len(rows), dtype=bool)
        if name in self.dims:
            cells = self.dims[name][rows]
            for key, state in self.kept.items():
                out |= (cells == key) & (state.status[rows] == _LOST)
        return out

    def _followed(self, name, rows, sure=False):
        """
        Yield each dimension's state with the rows ``name`` can move on it.

        A row is followed from its first selection, and only while the
        selected coordinate rebuilds and is as long as the dimension; the
        rest of ``rows`` are lost. ``sure`` also loses rows with no sample
        certain, which a count from either end cannot start from.
        """
        if not self.watched(name):
            return
        held = self.held(name)
        for cell, part in self._parts(name, rows):
            if "," in cell:  # a coordinate on several dimensions trims each
                self.lose(name, part)
                continue
            state = self._state(cell)
            new = part[state.status[part] == _UNTOUCHED]
            state.status[new], state.size[new] = _FOLLOWED, held.sizes[new]
            state.lo[new] = state.in_lo[new] = 0
            state.hi[new] = state.in_hi[new] = held.sizes[new] - 1
            told = (state.status[part] == _FOLLOWED) & (held.sizes[part] > 0)
            told &= held.sizes[part] == state.size[part]
            if sure:
                told &= state.in_lo[part] <= state.in_hi[part]
            state.status[part[~told]] = _LOST
            if told.any():
                yield state, part[told]

    def by_samples(self, name, rows, value) -> None:
        """Replay a selection of ``name`` by sample positions onto ``rows``."""
        for state, part in self._followed(name, rows, sure=True):
            firsts, lasts = zip(
                *(
                    _sliced(lo[part], hi[part], *value)
                    for lo in (state.lo, state.in_lo)
                    for hi in (state.hi, state.in_hi)
                ),
                strict=True,
            )
            firsts, lasts = np.array(firsts), np.array(lasts)
            # a count from either end shifts with the end it counts from
            empty = (lasts < firsts).any(axis=0)
            state.lo[part], state.in_lo[part] = firsts.min(axis=0), firsts.max(axis=0)
            state.hi[part], state.in_hi[part] = lasts.max(axis=0), lasts.min(axis=0)
            state.status[part[empty]] = _LOST
            state.moved[part[~empty]] = True

    def by_bounds(self, name, rows, lows, highs) -> None:
        """
        Replay a selection of ``name`` by value onto ``rows``.

        ``lows`` and ``highs`` are the row's new envelope ends, null where
        the selection left that end alone.
        """
        held = self.held(name)
        place = np.full(len(self.source), -1)
        place[rows] = np.arange(len(rows))
        for state, part in self._followed(name, rows):
            ends = [x[place[part]] for x in (lows, highs)]
            if held.shift is not None:
                ends = [x - held.shift[part] for x in ends]
            codes, firsts = _alike([held.codes[part], *ends])
            spans = [
                _span(held.coords[held.codes[part[x]]], ends[0][x], ends[1][x])
                for x in firsts
            ]
            state.narrow(part, *(np.array(x)[codes] for x in zip(*spans, strict=True)))

    def by_relative(self, name, rows, value) -> None:
        """Replay a relative selection of ``name`` onto ``rows``."""
        held = self.held(name)
        for state, part in self._followed(name, rows, sure=True):
            ranges = [x[part] for x in (state.lo, state.in_lo, state.hi, state.in_hi)]
            codes, firsts = _alike([held.codes[part], *ranges])
            spans = [
                _relative_span(
                    held.coords[held.codes[part[x]]],
                    value,
                    (ranges[0][x], ranges[1][x]),
                    (ranges[2][x], ranges[3][x]),
                )
                for x in firsts
            ]
            state.narrow(part, *(np.array(x)[codes] for x in zip(*spans, strict=True)))

    def follow_values(self, before, after, coords, relative, uncertain) -> None:
        """
        Replay a selection by value, from the envelopes it left.

        A row is selected where its envelope moved. ``uncertain`` names,
        by coordinate, the rows a relative selection trimmed without
        knowing where.
        """
        shared = after.index.intersection(before.index)
        was, now = before.index.get_indexer(shared), after.index.get_indexer(shared)
        rows = self.rows(shared)
        for name, value in coords.items():
            if not self.watched(name):
                continue
            cols = [f"{name}_min", f"{name}_max"]
            old = [before[c].to_numpy()[was] for c in cols]
            new = [after[c].to_numpy()[now] for c in cols]
            moved = [x != y for x, y in zip(old, new, strict=True)]
            blind = np.isin(shared, list(uncertain.get(name, ())))
            self.lose(name, rows[blind])
            cut = (moved[0] | moved[1]) & ~blind
            cut &= ~(pd.isnull(new[0]) | pd.isnull(new[1]))
            if not cut.any():
                continue
            if relative:
                self.by_relative(name, rows[cut], value)
                continue
            # an end the selection left alone bounds nothing: the envelope
            # there may be a superset of the samples held
            ends = [
                pd.Series(x[cut]).mask(~y[cut]).to_numpy()
                for x, y in zip(new, moved, strict=True)
            ]
            self.by_bounds(name, rows[cut], *ends)

    def ends(self, name, state, rows):
        """
        The ends of what ``rows`` keep of ``name``.

        Returns each row's least and greatest label, the coordinates kept
        and which of them each row keeps. A float label is moved outward
        to allow for rounding, except at an end of what the row first
        held, where the row's own envelope states it.
        """
        held, source = self.held(name), self.source
        codes, firsts = _alike([held.codes[rows], state.lo[rows], state.hi[rows]])
        kept = [
            held.coords[held.codes[x]][int(state.lo[x]) : int(state.hi[x]) + 1]
            for x in rows[firsts]
        ]
        mins = np.array([x.min() for x in kept])[codes]
        maxs = np.array([x.max() for x in kept])[codes]
        if held.shift is not None:
            mins, maxs = mins + held.shift[rows], maxs + held.shift[rows]
        elif mins.dtype.kind == "f":
            steps = to_float(source[f"{name}_step"].to_numpy()[rows])
            dtypes = source.get(f"_{name}_coord_dtype", pd.Series(index=source.index))
            dtypes = dtypes.to_numpy(dtype=object)[rows]
            first = state.lo[rows] == 0
            last = state.hi[rows] == state.size[rows] - 1
            # the least label is the first sample, or the last if descending
            down = steps < 0
            low = np.minimum(source[f"{name}_min"].to_numpy()[rows], mins)
            high = np.maximum(source[f"{name}_max"].to_numpy()[rows], maxs)
            scales = np.maximum(np.abs(low), np.abs(high))
            out_min = _outward(mins, steps, scales, dtypes, -1)
            out_max = _outward(maxs, steps, scales, dtypes, 1)
            mins = np.where(np.where(down, last, first), low, out_min)
            maxs = np.where(np.where(down, first, last), high, out_max)
        return mins, maxs, kept, codes

    def sample_ends(self, name, rows):
        """
        The ends of what ``rows`` keep of ``name`` after a selection moved them.

        Returns the places in ``rows`` of the rows followed, with their
        least and greatest labels, and their exact grids: None where an
        end sample is in doubt, which a grid cannot state.
        """
        places, mins, maxs, grids = [], [], [], []
        for cell, part in self._parts(name, rows) if self.watched(name) else ():
            state = self.kept.get(cell)
            if state is None:
                continue
            part = part[(state.status[part] == _FOLLOWED) & state.moved[part]]
            if not len(part):
                continue
            low, high, kept, codes = self.ends(name, state, part)
            sure = (state.lo[part] == state.in_lo[part]) & (
                state.hi[part] == state.in_hi[part]
            )
            terms = {x: _grid_terms(kept[x]) for x in set(codes[sure])}
            grids += [terms.get(x) if y else None for x, y in zip(codes, sure)]
            places.append(np.flatnonzero(np.isin(rows, part)))
            mins.append(low)
            maxs.append(high)
        if not places:
            return np.array([], dtype=np.int64), None, None, []
        return np.concatenate(places), np.concatenate(mins), np.concatenate(maxs), grids

    def trim(self, after: pd.DataFrame, names, riders=True) -> pd.DataFrame:
        """
        Give each coordinate on a dimension a selection moved its new ends.

        A coordinate the selection named keeps the envelope its own
        selection drew, unless another on its dimension was named beside
        it: then each is cut to the samples both keep. ``riders`` False
        leaves the coordinates it did not name as they were.
        """
        for key, state in self.kept.items():
            rows = np.flatnonzero(state.moved & (state.status == _FOLLOWED))
            state.moved[:] = False
            if not len(rows):
                continue
            on = [x for x in self.names if (self.dims[x][rows] == key).any()]
            named = [x for x in on if x in names]
            for name in on:
                if (name in named and len(named) < 2) or not (riders or name in named):
                    continue
                sizes = self.held(name).sizes[rows]
                part = rows[
                    (self.dims[name][rows] == key) & (sizes == state.size[rows])
                ]
                place = after.index.get_indexer(self.source.index[part])
                part, place = part[place >= 0], place[place >= 0]
                if not len(part):
                    continue
                mins, maxs, kept, codes = self.ends(name, state, part)
                cols = [f"{name}_min", f"{name}_max"]
                if name in named:
                    mins = np.maximum(mins, after[cols[0]].to_numpy()[place])
                    maxs = np.minimum(maxs, after[cols[1]].to_numpy()[place])
                    some = mins <= maxs
                    part, place, mins, maxs = (
                        x[some] for x in (part, place, mins, maxs)
                    )
                labels = after.index[place]
                after.loc[labels, cols[0]] = _as_dtype(mins, after[cols[0]].dtype, -1)
                after.loc[labels, cols[1]] = _as_dtype(maxs, after[cols[1]].dtype, 1)
                if name in named or (col := f"_{name}_grid") not in after:
                    continue
                # an exact grid restates only samples known to the sample
                sure = (state.lo[part] == state.in_lo[part]) & (
                    state.hi[part] == state.in_hi[part]
                )
                grids = after[col].to_numpy()[place]
                sure &= np.array([isinstance(x, tuple) for x in grids], dtype=bool)
                terms = {x: _grid_terms(kept[x]) for x in set(codes[sure])}
                _set_cells(after, col, labels[sure], [terms[x] for x in codes[sure]])
        return after


def patch_local_adjusted_envelopes(
    df: pd.DataFrame, residuals, drop_empty: bool = True, riders: bool = True
) -> pd.DataFrame:
    """
    Replay coordinate residuals onto relation envelopes in call order.

    Absolute and relative residuals trim membership and envelopes. A
    sample selection preserves row membership, but the planner must
    consume its trimmed envelopes or it publishes outputs
    that lie entirely outside the selected samples (phantom empties).
    Negative indices resolve per patch against the envelope-derived
    sample count (rows whose count is unknown keep their envelope as a
    candidacy superset — exactness is always re-applied at load).
    ``drop_empty`` removes rows emptied by sample selection for planning;
    presentation keeps those rows. Relative selections always remove rows
    whose resolved windows are empty.

    ``riders`` False leaves the coordinates the last selection did not
    name, but which share a dimension with one it did, untrimmed: wider
    than what loads, as an envelope may be, and enough to load from. An
    earlier selection still trims them, since a later one selects rows
    by them.
    """

    def _usable_index(value) -> bool:
        return value is None or isinstance(value, int | np.integer)

    source = df
    df = df.copy(deep=False)
    tracker = _SampleTracker(source)
    for position, (coords, samples, relative) in enumerate(residuals):
        if not samples:
            before, uncertain = df.copy(deep=False), {}
            if relative:
                df, uncertain = _relative_endpoints(
                    df, source, residuals[:position], coords
                )
                df = _adjust_relative_envelopes(df, coords, uncertain)
            else:
                df = _adjust_absolute_envelopes(df, coords)
            # only riders, or two coordinates cut to what both keep, need
            # the samples followed
            if riders or position < len(residuals) - 1 or len(coords) > 1:
                tracker.follow_values(before, df, coords, relative, uncertain)
                df = tracker.trim(df, coords)
            continue
        for name, value in coords.items():
            cols = [f"{name}_min", f"{name}_max", f"{name}_step"]
            rows = tracker.rows(df.index)
            usable = set(cols).issubset(df.columns) and is_range(value)
            usable = usable and all(_usable_index(x) for x in value)
            if not usable or df[cols[2]].isna().all():  # no step to count by
                tracker.lose(name, rows)
                continue
            lo_idx, hi_idx = value
            mins, maxs, steps = (df[c] for c in cols)
            # a dimension an earlier selection lost has no first sample to
            # count from; its envelope stays a superset
            lost = tracker.lost(name, rows)
            tracker.by_samples(name, rows, value)
            # Positions are patch-local sample indices with a stop-exclusive
            # hi, so the last included position is hi - 1. Sample 0 sits at
            # the envelope min for ascending coords and at the max for
            # descending ones.
            abs_steps = steps.abs()
            descending = to_float(steps.values) < 0
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = to_float((maxs - mins).values) / to_float(abs_steps.values)
            counts = pd.Series(np.round(ratio) + 1, index=df.index)

            def _positions(idx, counts=counts, index=df.index):
                """Per-row absolute positions (Python-slice clamping)."""
                if idx is None:
                    return None
                if idx >= 0:
                    return pd.Series(float(idx), index=index)
                return (counts + idx).clip(lower=0)

            lo_pos, hi_pos = _positions(lo_idx), _positions(hi_idx)
            unresolved = steps.isna() | lost
            for pos in (lo_pos, hi_pos):
                if pos is not None:
                    unresolved |= pos.isna()
            lo_off = None if lo_pos is None else lo_pos * abs_steps
            hi_off = None if hi_pos is None else (hi_pos - 1) * abs_steps
            new_min = mins if lo_off is None else mins + lo_off
            new_max = maxs if hi_off is None else mins + hi_off
            desc_min = mins if hi_off is None else maxs - hi_off
            desc_max = maxs if lo_off is None else maxs - lo_off
            new_min = new_min.where(~descending, other=desc_min)
            new_max = new_max.where(~descending, other=desc_max)
            # unresolvable rows keep their envelope (candidacy superset)
            new_min = new_min.mask(unresolved, mins)
            new_max = new_max.mask(unresolved, maxs)
            # A row whose samples are followed takes its ends from the ones
            # it keeps: an envelope end may sit between samples, and on an
            # exact grid samples need not be whole steps apart.
            places, lows, highs, grids = tracker.sample_ends(name, rows)
            if len(places):
                labels = df.index[places]
                new_min.loc[labels] = _as_dtype(lows, new_min.dtype, -1)
                new_max.loc[labels] = _as_dtype(highs, new_max.dtype, 1)
            if (col := f"_{name}_grid") in df and len(places):
                old = df[col].to_numpy()[places]
                exact = [
                    isinstance(x, tuple) and y is not None
                    for x, y in zip(old, grids, strict=True)
                ]
                grids = [x for x, keep in zip(grids, exact, strict=True) if keep]
                _set_cells(df, col, labels[np.array(exact, dtype=bool)], grids)
            # rows whose window is empty or lies entirely outside the
            # patch contribute nothing; test before clipping so such
            # windows are not resurrected as one-sample envelopes
            keep = (new_min <= new_max) & (new_min <= maxs) & (new_max >= mins)
            keep |= unresolved
            df[cols[0]] = new_min.clip(lower=mins, upper=maxs)
            df[cols[1]] = new_max.clip(lower=mins, upper=maxs)
            if drop_empty:
                df = df[keep]
            else:
                empty = ~keep & ~unresolved
                df[cols] = df[cols].mask(empty)
                df[_PATCH_LOCAL_EMPTY] = df.get(_PATCH_LOCAL_EMPTY, False) | empty
        df = tracker.trim(df, coords, riders or position < len(residuals) - 1)
    return df


def _drop_patch_local_empty(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose selected patch is empty from data-bearing plans."""
    empty = df.get(_PATCH_LOCAL_EMPTY)
    return df if empty is None else df[~empty]


def _ensure_patch_row(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the positional identity fallback for plain dataframes."""
    if "_patch_row" in df.columns:
        return df
    return df.assign(_patch_row=np.arange(len(df)))


def _dim_def_key_columns(df: pd.DataFrame, name: str) -> list[str]:
    """Return def-key column names for every non-chunked dimension."""
    dim_names: set[str] = set()
    if "dims" in df.columns:
        for dims_str in df["dims"].dropna().unique():
            dim_names.update(str(dims_str).split(","))
    dim_names.discard(name)
    return [f"_{x}_def_key" for x in sorted(dim_names)]


def _sampling_group(step: pd.Series, tolerance: float) -> pd.Series:
    """
    Label rows whose steps are within relative tolerance (spec 2.3).

    Steps group by orientation (sign) first, then by magnitude against a
    stable group anchor: a group opens at its smallest magnitude and
    admits members up to ``anchor * (1 + tolerance)``, so a chain of
    individually-close steps can never drift a group's endpoints past
    the tolerance. Unknown (NaN) steps share one group.
    """
    col = to_float(step.values)
    sign = np.sign(col)
    mag = np.abs(col)
    # orientation-major, magnitude-minor; NaNs sort to the end of both keys
    order = np.lexsort((mag, sign))
    sorted_sign, sorted_mag = sign[order], mag[order]
    labels = np.zeros(len(col), dtype=np.int64)
    label, i, n = 0, 0, len(col)
    while i < n:
        if np.isnan(sorted_mag[i]):
            # NaN keys sort last, so everything from here on is unknown
            j = n
        else:
            block_end = np.searchsorted(sorted_sign, sorted_sign[i], side="right")
            bound = sorted_mag[i] * (1 + tolerance)
            j = np.searchsorted(sorted_mag[:block_end], bound, side="right")
            j = max(j, i + 1)
        labels[order[i:j]] = label
        label += 1
        i = j
    return pd.Series(labels, index=step.index)


def _cell_tolerance(tolerance: GapTolerance | None, sub, name):
    """Resolve an absolute tolerance into one cell's own units."""
    if tolerance is None:
        return None
    units = sub[f"_{name}_units"].iloc[0] if f"_{name}_units" in sub else ...
    return tolerance.resolve(sub[f"{name}_min"].dtype, units, name)


def _adopt_lone_steps(df: pd.DataFrame, name: str, cells: pd.Series):
    """
    The relation with every lone sample planned at its cell's step.

    `_cell_labels` puts a step-less lone sample in its cell's one
    sampling; planning it at that step has continuity, gap reports and
    sizing all treat it as one sample of that sampling. The median keeps
    the choice independent of row order. `_{name}_adopted` marks those
    rows, whose published step stays unknown.
    """
    step = df[f"{name}_step"]
    if not step.isna().any():
        return df
    stated = step.groupby(cells).transform("median")
    adopted = step.isna() & stated.notna()
    return df.assign(
        **{f"{name}_step": step.fillna(stated), f"_{name}_adopted": adopted}
    )


def _published(steps: np.ndarray, adopted: np.ndarray) -> np.ndarray:
    """Steps as their rows state them: an adopted one is unknown again."""
    if not adopted.any():
        return steps
    unknown = np.array([np.nan]).astype(steps.dtype)
    return np.where(adopted, unknown, steps)


def _normalize_chunk_units(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Re-express the chunk dim's envelopes in one unit per dimensionality.

    Envelope columns are stored in each coordinate's original units, so
    compatible spellings (metres beside feet) are not directly
    comparable. Rows sharing a dimensionality convert to the unit of
    their first row — ordered by (envelope min in base units, patch row),
    the same deterministic order partitions present in — so continuity
    and instruction math stay valid and the plan speaks one unit per
    partition. The rewritten ``_{name}_units`` column tells assembly
    (and quantity-valued sizes) which unit that is. Unitless rows are
    untouched and never grouped with unitful ones.

    Steps convert as deltas (an affine unit's offset must not shift a
    difference), which the min+step round trip provides.
    """
    unit_col = f"_{name}_units"
    if unit_col not in df.columns:
        return df
    # Only numeric envelopes are stored per spelling and so need this;
    # time-like ones are canonical nanoseconds whatever unit the
    # coordinate names, and converting them as floats would raise.
    if not pd.api.types.is_numeric_dtype(df[f"{name}_min"]):
        return df
    units = df[unit_col]
    present = units.notna() & (units != "")
    distinct = set(units[present])
    if len(distinct) < 2:
        return df
    buckets: dict[str, list[str]] = {}
    for unit in distinct:
        quantity = get_quantity(unit)
        # null and empty spellings were filtered out above, so every
        # remaining one names a real unit
        assert quantity is not None
        buckets.setdefault(str(quantity.to_base_units().units), []).append(unit)
    min_name, max_name, step_name = f"{name}_min", f"{name}_max", f"{name}_step"
    df = df.copy()
    for base, spellings in buckets.items():
        if len(spellings) < 2:
            continue
        rows = present & units.isin(spellings)
        sub = df.loc[rows]
        base_min = pd.Series(np.nan, index=sub.index)
        for unit in spellings:
            idx = sub.index[sub[unit_col] == unit]
            values = sub.loc[idx, min_name].to_numpy(dtype=float)
            base_min.loc[idx] = convert_units(values, to_units=base, from_units=unit)
        order = pd.DataFrame({"_min": base_min, "_pid": sub["_patch_row"]}).sort_values(
            ["_min", "_pid"], kind="stable"
        )
        target = str(df.at[order.index[0], unit_col])
        for unit in spellings:
            if unit == target:
                continue
            idx = sub.index[sub[unit_col] == unit]
            mins = df.loc[idx, min_name].to_numpy(dtype=float)
            maxs = df.loc[idx, max_name].to_numpy(dtype=float)
            new_min = convert_units(mins, to_units=target, from_units=unit)
            df.loc[idx, min_name] = new_min
            df.loc[idx, max_name] = convert_units(
                maxs, to_units=target, from_units=unit
            )
            if step_name in df.columns:
                steps = df.loc[idx, step_name].to_numpy(dtype=float)
                stepped = convert_units(mins + steps, to_units=target, from_units=unit)
                df.loc[idx, step_name] = stepped - new_min
            if f"_{name}_coord_dtype" in df.columns:  # converted labels are floats
                df.loc[idx, f"_{name}_coord_dtype"] = "float64"
        df.loc[rows, unit_col] = target
    return df


def _partition_unit(df: pd.DataFrame, name: str, row: int) -> str:
    """The unit a partition's envelope is stated in, or "" if it states none."""
    col = df.get(f"_{name}_units")
    unit = None if col is None else col.iloc[row]
    return "" if pd.isnull(unit) else str(unit)


def _validate_on_missing_dim(on_missing_dim) -> None:
    """Reject an on_missing_dim value that is neither policy."""
    if on_missing_dim not in ("raise", "drop"):
        msg = f"on_missing_dim must be 'raise' or 'drop', got {on_missing_dim!r}"
        raise ParameterError(msg)


def _prepare_relation(
    df: pd.DataFrame,
    name: str,
    on_missing_dim: str,
    dim_label: str = "chunk dimension",
    operation: str = "chunking",
) -> pd.DataFrame:
    """
    Ready a flat relation for planning or reporting along ``name``.

    Attaches patch rows, re-spells compatible units, then applies the
    `on_missing_dim` policy. Missing envelopes, and patches carrying the
    name only as a non-dimensional coordinate (spec 7 / D2), both count
    as missing: envelope presence is not enough, because auxiliary
    coordinates index their envelopes too but their patches cannot be
    trimmed or merged *along* the name.

    `dim_label` and `operation` word the error for the caller; a typo in
    `on_missing_dim` raises here rather than silently taking the drop
    branch, which would truncate exactly the report the user asked for.
    """
    _validate_on_missing_dim(on_missing_dim)
    min_name, max_name = f"{name}_min", f"{name}_max"
    df = _ensure_patch_row(df)
    df = _normalize_chunk_units(df, name)
    null_rows = pd.isnull(df[min_name]) | pd.isnull(df[max_name])
    if "dims" in df.columns:
        dim_lists = df["dims"].fillna("").astype(str).str.split(",")
        not_a_dim = ~dim_lists.map(lambda dims: name in dims)
    else:
        not_a_dim = pd.Series(False, index=df.index)
    unusable = null_rows | not_a_dim
    if not unusable.any():
        return df
    if on_missing_dim == "raise":
        bad = df.loc[unusable, "_patch_row"].tolist()
        rides = int((not_a_dim & ~null_rows).sum())
        detail = (
            f" ({rides} of them carry {name!r} only as a non-dimensional "
            f"coordinate; {operation} is defined on dimensions)"
            if rides
            else ""
        )
        msg = (
            f"{int(unusable.sum())} patch(es) lack the {dim_label} "
            f"{name!r}{detail} (patch rows {bad[:5]}...). Pass "
            "on_missing_dim='drop' to exclude them."
        )
        raise ChunkError(msg)
    return df[~unusable]


def _cell_columns(df, name) -> list[str]:
    """
    Return the structural columns whose values decide a cell.

    The kind attrs decide it too, but through `_kind_codes` rather than
    their own values, so they are not listed here.
    """
    cols = []
    if "dims" in df.columns:
        cols.append("dims")
    # Structural identity: def keys of non-chunked *dimensions* only
    # (spec 2.2). Non-dimensional coordinate conflicts are policed at
    # assembly per the `conflict` argument, never partitioned on.
    cols += [x for x in _dim_def_key_columns(df, name) if x in df.columns]
    # The chunked dim's units partition too. Envelopes are native, but
    # _normalize_chunk_units has already re-spelled every compatible
    # unit family to one unit, so partitioning on the string is exactly
    # partitioning on dimensionality — a metre patch and a seconds
    # patch with contiguous magnitudes stay apart. Unitless (NULL)
    # stays its own group — assembly cannot merge unitless with unitful
    # coordinates either.
    if (unit_col := f"_{name}_units") in df.columns:
        cols.append(unit_col)
    # A name repeated in `group` would repeat the column, and a frame
    # with duplicate labels cannot be grouped or sorted.
    return list(dict.fromkeys(cols))


def _cell_labels(df, name, group_attrs, sampling_tolerance) -> pd.Series:
    """
    Label the cells a partition's continuity runs live inside (spec 2).

    A cell holds the rows which could combine if only their envelopes
    lined up: the same kind, dims signature, structural def keys of the
    non-chunked dimensions, chunk-dim units, and sampling group.
    Continuity is then evaluated *within* a cell, so unrelated patches
    can never bridge (or fabricate) a gap.
    """
    start, stop, step = get_interval_columns(df, name)
    kind = _kind_codes(df, [x for x in group_attrs if x in df.columns])
    keys = [kind, *(df[x] for x in _cell_columns(df, name))]
    base = df.groupby(keys, dropna=False, sort=False).ngroup()
    samp = _sampling_group(step, sampling_tolerance)
    # A lone sample states no step but fits any; one in a gap of its
    # cell's only sampling joins it, so the gap it sits in is seen. One
    # inside a stated envelope stays apart, where no overlap trims it.
    lone = step.isna() & (start == stop)
    if lone.any():
        stated = samp.where(step.notna()).groupby(base)
        lone &= stated.transform("nunique") == 1
        lone &= ~_inside_stated(start, stop, step, base, lone)
        samp = samp.mask(lone, stated.transform("first")).astype(np.int64)
    return base.astype(str) + "_" + samp.astype(str)


def _inside_stated(start, stop, step, base, rows) -> pd.Series:
    """Whether each of ``rows`` starts within a stated row of its cell."""
    out = pd.Series(False, index=start.index)
    has = step.notna()
    # only the cells holding one of the rows are scanned
    scanned = base.isin(base[rows].unique())
    for _, index in start[scanned].groupby(base[scanned]).groups.items():
        stated = has.loc[index]
        lows, highs = start.loc[index][stated], stop.loc[index][stated]
        order = np.argsort(lows.to_numpy(), kind="stable")
        lows = lows.to_numpy()[order]
        reach = np.maximum.accumulate(highs.to_numpy()[order])
        values = start.loc[index][~stated].to_numpy()
        pos = np.searchsorted(lows, values, side="right") - 1
        inside = (pos >= 0) & (reach[np.maximum(pos, 0)] >= values)
        out.loc[stated[~stated].index] = inside
    return out


# --- the seam test and one grid per stretch (coordinate rules 5 and 6)

# The diagnostic columns a plan's members carry and loading never sees.
PIECE_COLUMNS = ("reason", "shift", "dropped")


def resolve_max_shift(max_shift) -> float:
    """
    The seam slack, in steps, that ``max_shift`` names; None is the config's.

    It must be a plain number with 0 <= max_shift < 0.5: half a step would
    let a missing sample pass as a seam.
    """
    if max_shift is None:
        max_shift = dc.get_config().max_shift
    plain = isinstance(max_shift, int | float | np.integer | np.floating)
    if isinstance(max_shift, bool | np.bool_) or not plain or not 0 <= max_shift < 0.5:
        msg = (
            "max_shift must be a plain number of steps with "
            f"0 <= max_shift < 0.5, got {max_shift!r}."
        )
        raise ParameterError(msg)
    return float(max_shift)


def _numbers(values) -> np.ndarray:
    """Values as numbers: time-like ones as integer nanoseconds."""
    values = np.asarray(values)
    if values.dtype.kind in "Mm":
        return values.astype(f"{values.dtype.kind}8[ns]").view(np.int64)
    if values.dtype.kind in "iu":
        return values.astype(np.int64)
    return values.astype(np.float64)


def _counts_ticks(dtype) -> bool:
    """Whether labels of ``dtype`` are whole ticks (integers, or time in ns)."""
    return np.dtype(dtype).kind in "Mmiu"


def _as_values(numbers, like: np.dtype) -> np.ndarray:
    """Numbers back as values of the dtype ``like`` (see `_numbers`)."""
    if like.kind in "Mmiu":
        ticks = np.asarray(numbers).astype(np.int64)  # ticks are whole
        return (
            ticks.view(f"{like.kind}8[ns]") if like.kind in "Mm" else ticks.astype(like)
        )
    return np.asarray(numbers, dtype=like)


def _own_steps(lattice) -> np.ndarray:
    """Each row's exact spacing between its own labels, as a float."""
    _, stride, den, _, width, *_ = lattice
    stride, width = (np.asarray(x, dtype=np.float64) for x in (stride, width))
    exact = np.where(den > 1, stride / den, np.where(stride > 0, stride, width))
    return np.where(np.isfinite(width) & (width > 0), exact, np.nan)


def _precision(ticks: bool, step, narrow, *values) -> np.ndarray:
    """
    What a label is known to: a tick, or float exactness in its stored dtype.

    A tick label is the floor of its grid position, so it is a tick short
    of it at most, and only on a fractional step, which is never under
    `_MIN_FRACTION_TICKS`; labels on a smaller whole step are exact.
    ``narrow`` marks labels stored as float32.
    """
    if ticks:
        step = np.asarray(step, dtype=np.float64)
        return np.where(step >= _MIN_FRACTION_TICKS, 1.0, 0.0)
    biggest = np.max([np.abs(np.asarray(x, dtype=np.float64)) for x in values], 0)
    ulp = np.where(narrow, np.spacing(biggest.astype(np.float32)), np.spacing(biggest))
    return np.maximum(_GRID_RTOL * np.nan_to_num(step), _GRID_ULPS * ulp)


def _narrow(rows: pd.DataFrame, name: str) -> np.ndarray:
    """Which rows store their labels as float32."""
    stored = rows.get(f"_{name}_coord_dtype", pd.Series(None, index=rows.index))
    return stored.astype(str).eq("float32").to_numpy()


def _integer_ticks(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    The relation with integer labels as int64 ticks, as time is counted.

    A frame holds integer labels as floats, exactly only under 2**53; past
    that they stay floats, and `_fit_grids` fits them no grid.
    """
    cols = [f"{name}_min", f"{name}_max"]
    stored = df.get(f"_{name}_coord_dtype", pd.Series(None, index=df.index))
    values = df[cols].to_numpy()
    if values.dtype.kind != "f" or not stored.astype(str).str.match("u?int").all():
        return df
    if not ((np.abs(values) < 2**53) & (values == np.rint(values))).all():
        return df
    return df.astype(dict.fromkeys(cols, np.int64))


def _own_label_at(
    offset, lattice, rows, ticks, forward, first, own, after=True, snap=True
):
    """
    Each row's first own label at or past ``first + offset`` (or the last
    one at or before it, if not ``after``).

    ``first`` and the answer are forward numbers (see `_cell_seams`), and
    ``offset`` is relative to ``first`` so a tick count never passes
    through a float; a descending row asks its value-space lattice for the
    last label at or below the negated value. A row whose lattice is
    unknown counts its own steps from ``first``, or, unless ``snap``, is
    answered with the position asked about, which a read bounds by.
    """
    base, stride, den, phase = (np.asarray(v)[rows] for v in lattice[:4])
    lat = (base, np.where(stride > 0, stride, 1), den, phase)
    offset = np.nan_to_num(np.asarray(offset, dtype=np.float64))
    whole = np.ceil if after else np.floor  # the ticks on the side asked about
    if ticks:
        x = first + whole(offset).astype(np.int64)
        bump = 1
    else:
        x = first + offset
        bump = 2 * _GRID_SNAP_RTOL * np.nan_to_num(own)
    value = x * forward
    # a descending row's first label forward is its last one at or below
    low = after == (np.asarray(forward) > 0)
    found = np.where(
        low,
        _lattice_label(value, lat),
        _lattice_label(value + bump, lat, before=True),
    )
    with np.errstate(invalid="ignore"):
        k = offset / own
        steps = np.nan_to_num(
            (np.ceil(k - 1e-9) if after else np.floor(k + 1e-9)) * own
        )
    steps = steps if snap else offset
    guess = first + (whole(steps).astype(np.int64) if ticks else steps)
    return np.where(stride > 0, found * forward, guess)


def _cell_seams(sub, name, tolerance, max_shift) -> pd.DataFrame:
    """
    Judge every seam in one cell, trimming overlaps (rules 5 and 6).

    Rows are taken in coordinate order. The furthest label reached so far
    is the earlier patch's end, and its step the seam's step. A row
    continues from its first own label at least one step less
    ``max_shift`` past that end, joining if that label is within
    ``max_shift`` of one step past it (give or take label precision) or
    within ``tolerance``; otherwise its labels past the end start a new
    stretch, and a row with none is dropped. Returned frame (indexed like
    ``sub``): the stretch number, the kept envelope, how the row joined,
    the samples it dropped and keeps, its first position on its stretch,
    and the forward numbers the grid fit needs.
    """
    start, stop, step = get_interval_columns(sub, name)
    ticks = _counts_ticks(start.dtype)
    lattice = _sample_lattices(sub, name)
    own = _own_steps(lattice)
    signs = np.sign(to_float(step.to_numpy()))
    forward = -1 if np.nanmin(np.r_[signs, 1]) < 0 else 1
    low, high = _numbers(start.to_numpy()), _numbers(stop.to_numpy())
    first, last = (low, high) if forward > 0 else (-high, -low)
    order = np.lexsort((sub["_patch_row"].to_numpy(), first))
    first, last, own_o = first[order], last[order], own[order]
    reach = np.r_[last[:1], np.maximum.accumulate(last)[:-1]]
    # the row ending furthest so far owns the seam; a row only reaching
    # that end again adds nothing to it
    extends = np.r_[True, last[1:] > reach[1:]]
    owner = np.maximum.accumulate(np.where(extends, np.arange(len(order)), 0))
    owner = np.r_[0, owner[:-1]]
    seam_step = own_o[owner]
    narrow = _narrow(sub, name)[order]
    prec = _precision(ticks, seam_step, narrow | narrow[owner], reach, first)
    stepless = np.isnan(own_o)
    with np.errstate(invalid="ignore"):
        # how far past each row's first label its seam asks it to start
        behind = (reach - first).astype(np.float64)
        threshold = behind + np.nan_to_num((1 - max_shift) * seam_step - prec)
        ahead = threshold <= 0
        eff = np.where(
            ahead,
            first,
            _own_label_at(threshold, lattice, order, ticks, forward, first, own_o),
        )
        delta = (eff - reach).astype(np.float64)
        has = (eff <= last) & ~stepless
        by_shift = has & (np.abs(delta - seam_step) <= max_shift * seam_step + prec)
        by_tolerance = has & _tolerance_joins(delta, seam_step, tolerance, prec)
        # A row with no step has no labels to trim or step to judge: it
        # joins where it overlaps, and across a seam only a distance does.
        spaced = (first - reach).astype(np.float64)
        width_joins = _tolerance_joins(spaced, np.nan * spaced, tolerance, prec)
        by_tolerance = np.where(stepless, (first <= reach) | width_joins, by_tolerance)
        joins = by_shift | by_tolerance
        joins[:1] = False
        past = behind + (1 if ticks else prec)
        beyond = _own_label_at(past, lattice, order, ticks, forward, first, own_o)
        # labels with no step lose what an earlier row already reached
        next_up = reach + 1 if ticks else np.nextafter(reach, np.inf)
        unstepped = np.where(first > reach, first, next_up)
        fresh = np.where(stepless, unstepped, np.where(first > reach, first, beyond))
    lo = np.where(joins & ~stepless, eff, fresh)
    lo[:1] = first[:1]
    keep = lo <= last
    opens = ~joins & keep
    opens[:1] = True
    stretch = np.cumsum(opens) - 1
    with np.errstate(invalid="ignore"):
        dropped = np.where(stepless, 0, np.rint((lo - first) / own_o))
        kept = np.where(stepless, -1, np.rint((last - lo) / own_o) + 1)
        missing = np.rint((lo - reach).astype(np.float64) / seam_step) - 1
    kept_count = np.where(keep, kept, 0)
    position = pd.Series(kept_count).groupby(stretch).cumsum().to_numpy() - kept_count
    how = np.where(by_shift, "seam", "tolerance").astype(object)
    how[opens] = "gap"
    how[:1] = "stretch"
    lo_v, hi_v = (lo, last) if forward > 0 else (-last, -lo)
    gap_edges = (reach, lo) if forward > 0 else (-lo, -reach)
    out = pd.DataFrame(
        {
            "_stretch": stretch,
            "_lo": _as_values(lo_v, start.dtype),
            "_hi": _as_values(hi_v, start.dtype),
            "_keep": keep,
            "_joined": how,
            "_dropped": np.nan_to_num(dropped).astype(np.int64),
            "_count": np.nan_to_num(kept).astype(np.int64),
            "_pos": position.astype(np.int64),
            "_first": lo,
            "_last": last,
            "_own": own_o,
            "_forward": forward,
            "_seam": delta / seam_step,
            "_by_tolerance": by_tolerance & ~by_shift & ~stepless,
            "_gap_min": _as_values(gap_edges[0], start.dtype),
            "_gap_max": _as_values(gap_edges[1], start.dtype),
            "_missing": np.where(
                np.isnan(seam_step), np.nan, np.maximum(np.nan_to_num(missing), 0)
            ),
        },
        index=sub.index[order],
    )
    return out.loc[sub.index]


def _tolerance_joins(delta, step, tolerance, prec) -> np.ndarray:
    """
    Whether ``tolerance`` joins each seam: up to so many steps, a seam
    width, or one step (counting nothing where unknown) plus an excess.
    """
    if tolerance is None:
        return np.zeros(len(delta), dtype=bool)
    with np.errstate(invalid="ignore"):
        if tolerance.count is not None:
            return delta <= tolerance.count * step + prec
        width = _numbers(np.asarray(tolerance.excess))[()]
        return delta <= width + (0 if tolerance.seam else np.nan_to_num(step)) + prec


def _seams(df, name, group, tolerance, max_shift) -> pd.DataFrame:
    """The rows, integers counted in ticks, joined to their seams' judgements."""
    df = _integer_ticks(df, name)
    cells = _cell_labels(df, name, group, dc.get_config().sampling_group_tolerance)
    df = _adopt_lone_steps(df, name, cells)
    return _stretch_frame(df, name, cells, tolerance, max_shift)


def _stretch_frame(df, name, cells, tolerance, max_shift) -> pd.DataFrame:
    """`_cell_seams` over every cell; ``_stretch`` labels rows that may combine."""
    frames = []
    for label, index in df.groupby(cells, sort=False).groups.items():
        sub = df.loc[index]
        seams = _cell_seams(sub, name, _cell_tolerance(tolerance, sub, name), max_shift)
        seams["_stretch"] = f"{label}_" + seams["_stretch"].astype(str)
        frames.append(seams)
    return df.join(pd.concat(frames))


def _roundness(step: float) -> int:
    """Fewest significant digits in a step or in its rate; 99 for neither."""
    if not step or not np.isfinite(step):
        return 99
    digits = (f"{x:.12e}".split("e")[0].rstrip("0.") for x in (step, 1 / step))
    return min(len(x.replace(".", "")) for x in digits)


def _stated_steps(lattice, ticks: bool):
    """
    Each row's stated step as a numerator and denominator of ticks.

    An exact fractional grid states its own; a whole-tick step is over
    one. A float step is its own numerator, over a denominator of zero.
    """
    _, stride, den, _, width, *_ = lattice
    steps = np.where(width > 0, width, np.nan).astype(np.float64)  # NaN unknown
    if not ticks:
        return steps, np.zeros(len(steps), dtype=np.int64)
    exact = (np.asarray(den) > 1) & (np.asarray(stride) > 0)
    whole = np.isfinite(steps) & (np.mod(np.nan_to_num(steps), 1) == 0)
    num = np.where(exact, stride, np.where(whole, np.nan_to_num(steps), 0))
    den = np.where(exact, den, np.where(whole, 1, 0))
    return num.astype(np.int64), den.astype(np.int64)


def _fit_grids(rows, codes, n_parts, name, max_shift, dtype, lattice):
    """
    Fit each stretch's grid (rule 6), from its kept rows.

    The grid starts at the stretch's first label. Its step is the roundest
    any row states (ties to the earliest) for which every row's first and
    last label lies within ``max_shift`` of its numbered position; failing
    that it spreads the stretch evenly between its end labels. A stretch
    holding a row with no step has no grid, nor has one whose step cannot
    be stated on its own ticks: a fraction under `_MIN_FRACTION_TICKS`, or
    integers a float frame holds inexactly. A spread too fine for int64
    to number takes the closest step it can, its move recorded. Returns
    each partition's `Grid` (None without one), whether it relabels its
    rows, and each row's largest move, in steps.
    """
    ticks, time = _counts_ticks(dtype), np.dtype(dtype).kind in "Mm"
    first, last = rows["_first"].to_numpy(), rows["_last"].to_numpy()
    own, pos = rows["_own"].to_numpy(), rows["_pos"].to_numpy()
    end = pos + rows["_count"].to_numpy() - 1
    index = np.arange(len(rows))
    # the head row of each partition holds its first position
    head_of = pd.Series(index).groupby(codes).first()
    frame = pd.DataFrame({"code": codes, "pos": pos, "row": index})
    head = frame.sort_values(["code", "pos"]).drop_duplicates("code")
    head_of[head["code"].to_numpy()] = head["row"].to_numpy()
    head_rows = head_of.reindex(range(n_parts), fill_value=0).to_numpy()
    begin = first[head_rows]
    start = begin[codes]
    near = (first - start).astype(np.float64)
    far = (last - start).astype(np.float64)
    prec = _precision(ticks, own, _narrow(rows, name), first, last)
    with np.errstate(divide="ignore", invalid="ignore"):
        lows = np.fmax(
            np.where(pos > 0, (near - prec) / (pos + max_shift), -np.inf),
            np.where(end > 0, (far - prec) / (end + max_shift), -np.inf),
        )
        highs = np.fmin(
            np.where(pos > 0, (near + prec) / (pos - max_shift), np.inf),
            np.where(end > 0, (far + prec) / (end - max_shift), np.inf),
        )
    grouped = pd.DataFrame({"low": lows, "high": highs, "own": own}).groupby(codes)
    low = grouped["low"].max().reindex(range(n_parts)).to_numpy()
    high = grouped["high"].min().reindex(range(n_parts)).to_numpy()
    stepless = grouped["own"].apply(lambda x: x.isna().any())
    has = ~stepless.reindex(range(n_parts), fill_value=True).to_numpy(dtype=bool)
    has &= not ticks or first.dtype.kind != "f"
    num, den = _stated_steps(lattice, ticks)
    stated = np.where(den > 0, num / np.maximum(den, 1), num) / (1e9 if time else 1)
    kinds, uniques = pd.factorize(stated, use_na_sentinel=False)
    rounds = np.asarray([_roundness(x) for x in uniques], dtype=np.int64)[kinds]
    fits = (own >= low[codes] * (1 - 1e-12)) & (own <= high[codes] * (1 + 1e-12))
    fits &= (den > 0) | (not ticks)
    ranked = pd.DataFrame({"code": codes, "round": rounds, "pos": pos, "row": index})
    ranked = ranked[fits].sort_values(["code", "round", "pos"]).drop_duplicates("code")
    chosen = np.full(n_parts, -1, dtype=np.int64)
    chosen[ranked["code"].to_numpy()] = ranked["row"].to_numpy()
    count = pd.Series(end + 1).groupby(codes).max().reindex(range(n_parts))
    count = count.fillna(1).to_numpy(dtype=np.int64)
    finals = pd.Series(last).groupby(codes).max().reindex(range(n_parts)).to_numpy()
    sign = rows["_forward"].to_numpy()[head_rows]
    base, stride, lat_den, lat_phase = (np.asarray(x) for x in lattice[:4])
    grids = [None] * n_parts
    for part in np.flatnonzero(has):
        pick, top, size = chosen[part], head_rows[part], int(count[part])
        spread = finals[part] - begin[part]
        if not ticks:
            step = num[pick] if pick >= 0 else spread / max(size - 1, 1)
            grids[part] = Grid(sign[part] * begin[part], sign[part] * step, 0, size)
            continue
        terms = (num[pick], den[pick]) if pick >= 0 else (spread, max(size - 1, 1))
        step = Fraction(*(int(x) for x in terms))
        if step.denominator * size >= 2**62:  # the closest step int64 can number
            step = step.limit_denominator(2**62 // size)
        if step.denominator > 1 and step < _MIN_FRACTION_TICKS:
            continue  # no grid states it on its ticks: the labels stay
        origin = int(sign[part] * begin[part])
        grid = Grid(origin, int(sign[part] * step.numerator), step.denominator, size)
        if pick == top and lat_den[top] > 1 and sign[part] > 0:
            # a stretch on its first row's own fractional grid keeps its phase
            args = (int(base[top]), int(stride[top]), int(lat_den[top]), size)
            grid = Grid(*args, phase=int(lat_phase[top]))
            grid = grid.sliced(int(grid.index_of(origin, True)), 1, size)
        grids[part] = grid
    # one whole row on its own grid already has the stretch's labels
    sizes = np.bincount(codes, minlength=n_parts)
    dropped = rows["_dropped"].to_numpy()[head_rows]
    relabel = (sizes > 1) | (chosen != head_rows) | (dropped != 0)
    spacing = [np.nan if x is None else x.step_num / (x.step_den or 1) for x in grids]
    steps = np.abs(np.asarray(spacing, dtype=np.float64))[codes]
    with np.errstate(invalid="ignore", divide="ignore"):
        shift = np.fmax(np.abs(near - pos * steps), np.abs(far - end * steps)) / steps
    return grids, relabel, np.nan_to_num(shift)  # no grid, no move


def _grid_windows(grid, value, overlap, keep_partial):
    """
    The first and last position of each window on a stretch's grid.

    Windows are measured from its first label and hold the samples whose
    labels fall inside (rule 8); an exact grid places them on its ticks.
    """
    count = grid.count
    length = _numbers(np.asarray(value))[()]
    hop = 0 if overlap is None else _numbers(np.asarray(overlap))[()]
    if grid.exact:
        one, last, after = (int(x) for x in grid.labels([0, count - 1, count], "i8"))
        sign = 1 if grid.step_num > 0 else -1
        step = abs(grid.step_num) // grid.step_den
        windows = get_intervals(
            0,
            sign * (last - one),
            length,
            overlap=hop,
            step=step,
            keep_partials=keep_partial,
            tail=sign * (after - last),
        )
        # an integer label is in a window from the first tick at or past its edge
        whole = windows if windows.dtype.kind in "iu" else np.ceil(windows)
        edges = one + sign * whole.astype(np.int64)
        firsts, ends = grid.index_of(edges, True).astype(np.int64).T
    else:
        step = abs(grid.step_num)
        span = (count - 1) * step
        windows = get_intervals(
            0.0, span, length, overlap=hop, step=step, keep_partials=keep_partial
        )
        firsts, ends = np.ceil(windows / step - 1e-9).astype(np.int64).T
    lasts = np.minimum(ends - 1, count - 1)
    keep = lasts >= firsts
    return firsts[keep], lasts[keep]


def _grid_members(windows, kept, kcodes, offsets, lattice, ticks):
    """
    Map every window of positions onto the kept rows feeding it, at once.

    ``windows`` holds each window's partition and first and last position;
    ``offsets`` numbers every partition's positions apart, so one search
    over all rows finds each window's. Returns each pair's window and kept
    row, the row's own first and last label inside the window (value
    space). Pairs go in window order, and within one in the plan's row order,
    which `conflict="keep_first"` and the carried attrs read.
    """
    parts, firsts, lasts = windows
    pos, count = kept["_pos"].to_numpy(), kept["_count"].to_numpy()
    flat = offsets[kcodes] + pos
    order = np.argsort(flat, kind="stable")
    tops = (flat + count - 1)[order]
    lows, highs = offsets[parts] + firsts, offsets[parts] + lasts
    first_src = np.searchsorted(tops, lows)
    last_src = np.searchsorted(flat[order], highs, side="right") - 1
    counts = np.maximum(last_src - first_src + 1, 0)
    rel_out = np.repeat(np.arange(len(firsts)), counts)
    shifts = np.repeat(np.cumsum(counts) - counts, counts)
    rel_src = order[np.arange(counts.sum()) - shifts + np.repeat(first_src, counts)]
    begin, own = kept["_first"].to_numpy()[rel_src], kept["_own"].to_numpy()[rel_src]
    finish = kept["_last"].to_numpy()[rel_src]
    sign = kept["_forward"].to_numpy()[rel_src]
    at, size = flat[rel_src], count[rel_src]
    local0 = np.maximum(lows[rel_out], at) - at
    local1 = np.minimum(highs[rel_out], at + size - 1) - at
    edges = []
    for local, after in ((local0, True), (local1, False)):
        guess = (local + (-0.5 if after else 0.5)) * own
        label = _own_label_at(
            guess, lattice, rel_src, ticks, sign, begin, own, after, snap=False
        )
        whole = begin if after else finish
        edge = local == 0 if after else local == size - 1
        edges.append(np.where(edge, whole, label) * sign)
    low, high = np.minimum(*edges), np.maximum(*edges)
    order = np.lexsort((rel_src, rel_out))
    return rel_out[order], rel_src[order], low[order], high[order]


def _user_stacklevel() -> int:
    """Return the warn stacklevel pointing at the first non-dascore frame.

    Plans are built at several call depths (spool.chunk, spool.chunk_plan,
    build_chunk_plan directly), so a fixed stacklevel would blame library
    frames for some entries.
    """
    # The dascore package directory, resolved from the package itself so
    # this does not depend on this module's location within it.
    package_dir = str(Path(dc.__file__).resolve().parent)
    # Frames after this helper's own align exactly with warn's numbering:
    # level 1 is the frame calling warn.
    for level, frame_info in enumerate(inspect.stack()[1:], start=1):
        filename = str(Path(frame_info.filename).resolve())
        if not filename.startswith(package_dir):
            return level
    return 1


def _coerce_length_overlap(value, overlap, start_dtype):
    """Coerce the chunk length/overlap to the dimension's span dtype."""
    time_like = is_datetime64(start_dtype) or is_timedelta64(start_dtype)
    if time_like:
        value = to_timedelta64(value) if value is not None else None
        overlap = to_timedelta64(overlap) if overlap is not None else None
    return value, overlap


def _validate_quantity(label: str, quant, name: str) -> None:
    """Reject a quantity chunk length or overlap that cannot describe one."""
    if not isinstance(quant, Quantity):
        return
    magnitude = np.asarray(quant.magnitude)
    if magnitude.ndim:
        msg = (
            f"The {label} for {name!r} must be a single quantity, got an "
            f"array of {magnitude.size}."
        )
        raise ParameterError(msg)
    if not np.isfinite(magnitude):
        msg = f"The {label} for {name!r} must be finite, got {quant}."
        raise ParameterError(msg)


def _needs_partition_resolution(value, overlap) -> bool:
    """
    True when a chunk length or overlap can only be resolved per partition.

    Quantities need the partition's units, sampling interval, and (for
    data sizes) its element dtype, none of which are known before the
    partitions exist.
    """
    return isinstance(value, Quantity) or isinstance(overlap, Quantity)


def _combined_dtype(dtypes: pd.Series) -> np.dtype | None:
    """
    Return the dtype an assembled merge of these members would produce.

    Assembly upcasts a mixed-dtype merge with `np.result_type`, so the
    size estimate must use the same rule. Returns None when the relation
    carries no usable dtype; planning never fails on bad metadata here,
    it fails later with an explanatory error.
    """
    values = {str(x) for x in dtypes.dropna().unique() if str(x)}
    if not values:
        return None
    try:
        return np.result_type(*(np.dtype(x) for x in sorted(values)))
    except TypeError:
        return None


def _slab_samples(sub: pd.DataFrame, name: str) -> int | None:
    """
    Samples in one index-slab along `name`.

    This is the product of the *other* dimensions' sample counts, i.e.
    how many elements one step along `name` costs. The widest member of
    the partition wins, so the byte estimate bounds every member rather
    than just the first. Returns None when any count is underivable.
    """
    if "dims" in sub.columns:
        dims = str(sub["dims"].iloc[0]).split(",")
    else:  # bare frames (no dims column) fall back to complete triples
        candidates = (x[: -len("_min")] for x in sub.columns if x.endswith("_min"))
        dims = [
            x for x in candidates if {f"{x}_max", f"{x}_step"}.issubset(sub.columns)
        ]
    total = pd.Series(1.0, index=sub.index)
    for dim in dims:
        if not dim or dim == name:
            continue
        cols = [f"{dim}_min", f"{dim}_max", f"{dim}_step"]
        if not set(cols).issubset(sub.columns):
            return None
        mins, maxs, steps = (sub[c] for c in cols)
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = to_float((maxs - mins).values) / np.abs(to_float(steps.values))
        counts = np.round(ratio) + 1
        if not np.all(np.isfinite(counts)):
            return None
        total = total * counts
    return int(total.max())


def _size_to_length(size_quant, sub, name, size_step):
    """
    Convert a data size into a chunk length along `name`.

    The sample count is floored so an output's data never exceeds the request,
    and clamped to one sample when a single slab is already larger than
    the target (reported back through the returned diagnostics).
    """
    dtype = _combined_dtype(sub["_dtype"]) if "_dtype" in sub.columns else None
    if dtype is None:
        msg = (
            f"Cannot chunk by data size along {name!r}: the patch relation "
            "carries no dtype information. Hand-built dataframes and spools "
            "indexed by an older DASCore lack it; delete and rebuild the "
            "index, or pass an explicit length instead of a size."
        )
        raise ChunkError(msg)
    slab = _slab_samples(sub, name)
    if slab is None:
        msg = (
            f"Cannot chunk by data size along {name!r}: the sample count of "
            "another dimension cannot be determined from its envelope "
            "(missing or non-uniform sampling)."
        )
        raise ChunkError(msg)
    step_float = np.abs(to_float(size_step))
    if not np.isfinite(step_float) or step_float == 0:
        msg = (
            f"Cannot chunk by data size along {name!r}: the sampling "
            "interval is unknown."
        )
        raise ChunkError(msg)
    bytes_per_sample = dtype.itemsize * slab
    requested = get_byte_count(size_quant)
    samples = int(requested // bytes_per_sample)
    clamped = samples < 1
    samples = max(samples, 1)
    # An output holds the sum of its members' sample counts, which only
    # equals span/step + 1 when the partition sits on a single grid.
    # Members separated by less than one sample pack more samples into
    # the same span (each contributes its own trailing sample), so the
    # length is divided by how much denser than the grid the partition
    # actually is. Exactly 1.0 for gridded partitions.
    packing = _packing_factor(sub, name, step_float)
    span_samples = max(int(samples // packing), 1)
    diagnostics = {
        "dtype": str(dtype),
        "itemsize": int(dtype.itemsize),
        "slab_samples": int(slab),
        "bytes_per_sample": int(bytes_per_sample),
        "n_samples": samples,
        "packing": float(packing),
        "clamped": clamped,
    }
    return span_samples * size_step, diagnostics


def _packing_factor(sub: pd.DataFrame, name: str, step_float: float) -> float:
    """
    How densely a partition's members pack samples, relative to its grid.

    1.0 when members tile the grid exactly. Greater when consecutive
    members are separated by less than one sample, which happens with
    near-contiguous files whose boundaries do not land on the grid.
    Overlapping members are excluded: the overlap correction
    deduplicates them at member-build time, so they do not add samples.
    """
    # the caller already read these columns for this partition
    start, stop, step = get_interval_columns(sub, name)
    starts = to_float(start.values)
    order = np.argsort(starts, kind="stable")
    # relative to the first start: absolute datetimes are ~1e9 seconds,
    # where a float64 difference loses the precision a sub-sample gap
    # lives in
    starts = starts[order] - starts[order][0]
    spans = to_float((stop - start).values)[order]
    steps = np.abs(to_float(step.values))[order]
    with np.errstate(invalid="ignore", divide="ignore"):
        counts = np.round(spans / steps) + 1
        # distance to the next member's start; the last member owns its
        # own span plus the one step its trailing sample occupies
        deltas = np.diff(starts, append=starts[-1] + spans[-1] + steps[-1])
        # an overlap is deduplicated, so it cannot exceed grid density
        deltas = np.maximum(deltas, spans)
        density = counts / deltas
    density = density[np.isfinite(density) & (density > 0)]
    # a partition whose steps are all unusable is rejected before this,
    # so at least one member always yields a density
    assert len(density)
    packing = float(density.max() * step_float)
    # residual float noise must not cost a sample on an exactly gridded
    # partition, whose packing is 1.0 by construction
    return 1.0 if packing < 1 + 1e-9 else packing


def _resolve_partition_length(value, overlap, sub, name, size_step, start_dtype):
    """
    Resolve one partition's chunk length and overlap.

    Returns (length, overlap, diagnostics); diagnostics is None unless a
    data size was resolved.
    """
    diagnostics = None

    def _resolve(val, is_overlap):
        nonlocal diagnostics
        if not isinstance(val, Quantity):
            return val
        if is_data_size(val):
            # a negative overlap opens a gap; floor the magnitude so the
            # gap widens monotonically rather than flipping direction
            sign = -1 if val.magnitude < 0 else 1
            length, diag = _size_to_length(abs(val), sub, name, size_step)
            if not is_overlap:
                diagnostics = diag
            return sign * length
        if val.dimensionless:
            msg = (
                f"Cannot chunk {name!r} by {val}: a dimensionless quantity "
                "has no meaning as a length. Use a data size (eg '25 MB') "
                "or the coordinate's units."
            )
            raise UnitError(msg)
        units_col = f"_{name}_units"
        units = sub[units_col].iloc[0] if units_col in sub.columns else ...
        prefix = f"Cannot chunk {name!r} by {val}"
        return _quantity_to_dim_value(val, start_dtype, units, prefix)

    value_out = _resolve(value, False)
    overlap_out = _resolve(overlap, True)
    # a size-derived length is already in the dimension's own units; a
    # plain number against a time-like dim still needs the usual coercion
    if not isinstance(value, Quantity) or not is_data_size(value):
        value_out, _ = _coerce_length_overlap(value_out, None, start_dtype)
    if not isinstance(overlap, Quantity) or not is_data_size(overlap):
        _, overlap_out = _coerce_length_overlap(None, overlap_out, start_dtype)
    return value_out, overlap_out, diagnostics


def _coord_owner(col: str, coord_names: set[str]) -> str | None:
    """
    Return the coordinate owning an envelope column, if any.

    Ownership is decided by matching the full name against known
    coordinates with an interval suffix; splitting on the first
    underscore would mis-assign columns of dims like `event_time`.
    """
    for suffix in ("_min", "_max", "_step", "_units"):
        if col.endswith(suffix):
            base = col[: -len(suffix)]
            if base in coord_names:
                return base
    return None


def _partition_frames(df: pd.DataFrame, labels: pd.Series, name: str):
    """
    Order the relation by (partition, envelope min, patch row).

    Partition order follows spec 8: by (partition min, smallest member
    patch row) — never by anything derived from input row order. Returns
    the sorted frame (fresh RangeIndex), each row's partition ordinal,
    the offsets where partitions begin, and the partition envelopes, of
    the rows' overlap-trimmed envelopes (`_cell_seams`).
    """
    min_name = f"{name}_min"
    grouped = df.groupby(labels, sort=False)
    stats = grouped.agg(
        _min=("_lo", "min"), _max=("_hi", "max"), _pid=("_patch_row", "min")
    ).sort_values(["_min", "_pid"], kind="stable")
    rank = pd.Series(np.arange(len(stats)), index=stats.index)
    codes = labels.map(rank).to_numpy(dtype=np.intp)
    # last lexsort key is primary: partition, then envelope min, then id
    order = np.lexsort((df["_patch_row"].to_numpy(), df[min_name].to_numpy(), codes))
    sorted_df = df.iloc[order].reset_index(drop=True)
    codes = codes[order]
    seg_starts = np.flatnonzero(np.r_[True, np.diff(codes) != 0])
    return (
        sorted_df,
        codes,
        seg_starts,
        stats["_min"].to_numpy(),
        stats["_max"].to_numpy(),
    )


# Prefixes the dims standing in for the envelope of a riding coordinate.
_RIDES = "<rides the merged dimension> "


def _mark_riders(df: pd.DataFrame, name: str) -> tuple[pd.DataFrame, set[str]]:
    """
    Replace the envelopes of coordinates riding dim `name` with a marker.

    Such a coordinate is concatenated along `name` sample for sample, so
    its envelope differs member by member by design; only whether a
    member holds it, and on which dims, is policed. Each row's own
    coordinate dims decide, so the same name on another dim is still
    compared by value.
    A null envelope (a member without the coordinate) stays null.
    Returns the marked frame and the marked columns.
    """
    marked = {}
    for coord in _identified_coords(df) - {name}:
        if (dims := df.get(f"_{coord}_dims")) is None:
            continue
        rides = dims.map(lambda x: isinstance(x, str) and name in x.split(","))
        if not rides.any():
            continue
        for suffix in ("_min", "_max", "_step"):
            if (col := f"{coord}{suffix}") in df.columns:
                marked[col] = df[col].where(~rides | df[col].isna(), _RIDES + dims)
    return df.assign(**marked), set(marked)


def _carried_columns(
    sorted_df: pd.DataFrame,
    codes: np.ndarray,
    seg_starts: np.ndarray,
    name: str,
    conflict: str,
    active: np.ndarray,
) -> dict[str, pd.Series]:
    """
    Resolve every partition's carried columns at once (spec 2.5/6.4).

    Dims and def keys are single-valued by construction. Public attrs
    must hold equal values per partition, policed by `conflict`: a
    missing value (null or "") is a value like any other, equal to
    another missing one and nothing else, so a member which never
    recorded an attr conflicts with one which did. Returns a mapping of
    column -> one carried value per partition (null where a partition
    does not carry the column), in the order columns ride onto outputs.
    A conflict raises for the first active partition (in output order)
    and its first conflicting column, exactly as per-partition policing
    did; partitions which produced no outputs (`active` False) are never
    policed. Coordinates riding `name` are policed only for being held
    (`_mark_riders`) and carry no envelope.
    """
    sorted_df, riders = _mark_riders(sorted_df, name)
    n_parts = len(seg_starts)
    first = sorted_df.iloc[seg_starts].reset_index(drop=True)
    has_dims = "dims" in sorted_df.columns
    # per-partition dims spelling used for column ownership (partition
    # keys make dims single-valued within a partition)
    police_dims = [str(x) for x in first["dims"]] if has_dims else [""] * n_parts
    policed = [
        col
        for col in sorted_df.columns
        if not col.startswith("_")
        and col not in _SOURCE_COLUMNS
        # chunk-dim envelope columns are rebuilt, never carried
        and _coord_owner(col, {name}) != name
    ]
    carried: dict[str, pd.Series] = {}
    if policed:
        # ownership of each policed column under each partition's dims
        owned = np.zeros((n_parts, len(policed)), dtype=bool)
        for dims_str in set(police_dims):
            coord_names = set(dims_str.split(",")) | {name}
            rows = np.asarray([x == dims_str for x in police_dims], dtype=bool)
            for index, col in enumerate(policed):
                if _coord_owner(col, coord_names) is not None:
                    owned[rows, index] = True
        # Aggregate only rows of active partitions: skipped partitions
        # were never policed (their values may not even be hashable),
        # and their carried values are never published.
        rows = active[codes]
        # "" and null are one missing value, which is a value like any other.
        known = known_only(sorted_df.loc[rows, policed])
        grouped = known.groupby(codes[rows], sort=True)
        part_index = grouped.size().index.to_numpy()
        nunique = np.zeros((n_parts, len(policed)), dtype=np.intp)
        # An unhashable attr value in an active partition raises
        # TypeError here. Per-partition policing raised it too, though
        # partition-major rather than at aggregation; such values are
        # outside the relation's contract, so the eager error is fine.
        # dropna=False: a member which recorded nothing is a distinct value.
        nunique[part_index] = grouped.nunique(dropna=False).to_numpy()
        # no conflict: exactly one distinct value
        single = nunique <= 1
        conflicted = ~single & active[:, None]
        # The first *row* of each partition, not `grouped.first()`, which
        # skips nulls and would resurrect the old first-known rule. Only
        # active partitions: `known_only` on the rest could meet an
        # unhashable value, as the comment above says.
        firsts = pd.DataFrame(index=range(n_parts), columns=policed, dtype=object)
        stated = known_only(first.loc[part_index, policed])
        firsts.loc[part_index] = stated.to_numpy()
        # Dims are a partition key and group attrs never conflict within
        # one, so neither reaches the conflict policy here.
        raising = conflicted & (owned | (conflict == "raise"))
        if raising.any():
            part = int(np.argmax(raising.any(axis=1)))
            col = policed[int(np.argmax(raising[part]))]
            msg = (
                f"Cannot merge on dim {name} because {col} holds conflicting "
                "values. Consider using the `conflict` argument to loosen "
                "this restriction."
            )
            raise CoordMergeError(msg)
        keep_first = conflict == "keep_first"
        for index, col in enumerate(policed):
            # keep_first carries the first member's value; drop omits the
            # column for that partition (null after assembly).
            keeps = single[:, index] | (conflicted[:, index] & keep_first)
            if not (keeps & active).any():
                continue  # dropped by every active partition: omit entirely
            values = firsts[col]
            if col in riders:
                # the derived catalog describes riders from the members
                marker = values.map(lambda x: str(x).startswith(_RIDES))
                keeps = keeps & ~marker.to_numpy(dtype=bool)
            carried[col] = values if keeps.all() else values.where(keeps)
    # Structural (dimension) def keys carry — single-valued by
    # partitioning — and canonical units carry for every dimension, the
    # chunked one included (partition-constant: units are a
    # sampling-partition component).
    columns = set(sorted_df.columns)
    extras: dict[str, np.ndarray] = {}
    for part in range(n_parts):
        dims_val = first["dims"].iloc[part] if has_dims else None
        dim_names: set[str] = set()
        if has_dims and not pd.isnull(dims_val):
            dim_names = set(str(dims_val).split(","))
        dim_names.discard(name)
        # a kept identity brings the exact grid it describes, and the
        # dtype that grid's values are stated in: a row is all an output
        # with no members has to rebuild the coordinate from, and an
        # integer coordinate rebuilt from the frame's float envelope
        # would not match the same coordinate on its neighbours
        part_cols = [
            key
            for x in sorted(dim_names)
            for suffix in ("_def_key", "_grid", "_coord_dtype")
            if (key := f"_{x}{suffix}") in columns
        ]
        coord_names = set(police_dims[part].split(",")) | {name}
        part_cols += [key for x in coord_names if (key := f"_{x}_units") in columns]
        for col in part_cols:
            mask = extras.get(col)
            if mask is None:
                mask = extras[col] = np.zeros(n_parts, dtype=bool)
            mask[part] = True
    for col, mask in extras.items():
        if not (mask & active).any():
            continue
        values = first[col]
        carried[col] = values if mask.all() else values.where(mask)
    return carried


def _uniform_dtype_str(value, cache: dict[str, str]) -> str:
    """
    The assembled dtype string for a single-dtype partition.

    Matches `_combined_dtype` exactly (np.result_type normalizes the
    spelling); "" is the same not-known sentinel ingest writes — never
    NaN, which would reach the derived catalog as the string "nan".
    """
    key = "" if value is None or pd.isnull(value) else str(value)
    if key not in cache:
        combined = _combined_dtype(pd.Series([key], dtype=object))
        cache[key] = "" if combined is None else str(combined)
    return cache[key]


_REPORT_COLUMNS = (
    "span",
    "gap_total",
    "gap_size",
    "covered",
    "coverage",
    "missing",
    "group_id",
)


def _report_dim_columns(df: pd.DataFrame, name: str, carried) -> tuple[str, str, str]:
    """
    Return the envelope columns of `name`, checked for a usable report.

    `get_interval_columns` words its error for chunking, and a carried
    attr sharing an emitted column's name would overwrite it rather than
    fail, so both are policed here instead. The envelope columns count
    as emitted: grouping by one would leave the frame two columns of
    that name, which no longer answers to `frame[name]` as a series.
    """
    columns = (f"{name}_min", f"{name}_max", f"{name}_step")
    if missing := set(columns) - set(df.columns):
        dims = get_dim_names_from_columns(df)
        msg = (
            f"Cannot report on {name!r}: no {sorted(missing)} in the "
            f"relation. Valid dimensions or columns are {dims}."
        )
        raise ParameterError(msg)
    emitted = set(_REPORT_COLUMNS) | set(columns)
    if collide := sorted(set(carried) & emitted):
        msg = (
            f"Cannot group a report by {collide}: the name(s) collide with "
            "the columns the report emits."
        )
        raise ParameterError(msg)
    return columns


def _report_preamble(df, name, group, on_missing_dim):
    """
    Resolve what both reports need before they can differ.

    Returns the prepared relation, its carried columns, and the
    envelope column names -- kept in one place so the two reports
    cannot drift apart in how they group, validate, or label.
    """
    group_attrs = _resolve_group_attrs(group, set(df.columns))
    names = _report_dim_columns(df, name, group_attrs)
    df = _prepare_relation(df, name, on_missing_dim, "dimension", "gap reporting")
    return df, group_attrs, _gap_carried_columns(df, name, group_attrs), names


def _gap_carried_columns(df: pd.DataFrame, name: str, group_attrs) -> list[str]:
    """
    Return the cell columns a report shows, in a stable order.

    Def keys are opaque structural identity, so they partition a report
    without appearing in it; `group_id` tells cells apart instead. The
    dimension's units are kept: they name what the reported magnitudes
    are in, and `present_units_columns` makes them public at the spool
    boundary. The kind attrs are shown by value, which is why they are
    listed here rather than reached through `_kind_codes`.
    """
    private = set(_dim_def_key_columns(df, name))
    cols = [x for x in group_attrs if x in df.columns]
    cols += [x for x in _cell_columns(df, name) if x not in private]
    return list(dict.fromkeys(cols))


def _cell_gaps(df: pd.DataFrame, name: str, group_attrs, tolerance, max_shift):
    """
    Yield `(cell rows, gaps in that cell)` for every cell in `df`.

    A gap is each seam `_cell_seams` opens a stretch at, so it is exactly
    where chunk with the same arguments splits. Cells come in envelope-min
    order, as partitions do (spec 8), and ties break on what the cell
    states rather than on a patch row, which is positional — so a report
    never depends on the order the relation happened to arrive in. Each
    cell's ordinal rides on its gaps as `group_id`, which is the only
    thing that always tells two cells apart: sampling group and
    structural def keys partition without being shown.
    """
    min_name, max_name, step_name = _dim_columns(df, name)
    carried = _gap_carried_columns(df, name, group_attrs)
    cells = _cell_labels(
        df, name, group_attrs, dc.get_config().sampling_group_tolerance
    )
    df = _adopt_lone_steps(df, name, cells)
    grouped = df.groupby(cells, sort=False)
    groups = grouped.groups
    mins = grouped[min_name].min()
    stated = {
        label: tuple(df.loc[index, x].iloc[0] for x in carried)
        for label, index in groups.items()
    }
    order = sorted(
        groups, key=lambda label: (mins[label], [str(x) for x in stated[label]])
    )
    for group_id, label in enumerate(order):
        sub = df.loc[groups[label]]
        tol = _cell_tolerance(tolerance, sub, name)
        seams = _cell_seams(_integer_ticks(sub, name), name, tol, max_shift)
        found = (seams["_joined"] == "gap").to_numpy() & seams["_keep"].to_numpy()
        opened = seams[found].sort_values("_gap_min")
        low, high = opened["_gap_min"].to_numpy(), opened["_gap_max"].to_numpy()
        # the row opening each gap states the step, signed as the coordinate is
        gaps = pd.DataFrame(
            {
                min_name: low.astype(sub[min_name].dtype),
                max_name: high.astype(sub[max_name].dtype),
                step_name: sub.loc[opened.index, step_name].to_numpy(),
                "gap_size": high - low,
                "missing": opened["_missing"].to_numpy(),
                "group_id": np.full(len(opened), group_id, dtype=np.int64),
            }
            | {x: sub[x].iloc[:1].repeat(len(opened)).to_numpy() for x in carried}
        )
        yield group_id, sub, gaps


def _empty_report(columns: dict) -> pd.DataFrame:
    """
    Build a report frame with no rows but the right columns and dtypes.

    Callers pass empty slices of relation columns, or expressions over
    them, so a caller which sums or groups an empty report gets the
    dtypes the same relation would give populated. Purely computed
    columns (`coverage`, `group_id`) pass a bare typed Series instead.
    """
    return pd.DataFrame({k: v.iloc[:0] for k, v in columns.items()})


def build_gap_frame(
    df: pd.DataFrame,
    name: str,
    *,
    tolerance: float | Quantity | np.timedelta64 | GapTolerance | None = None,
    max_shift: float | None = None,
    group: str | Sequence[str] | None = None,
    on_missing_dim: Literal["raise", "drop"] = "drop",
) -> pd.DataFrame:
    """
    Report every discontinuity along ``name`` in a flat patch relation.

    One row per gap, over the same cells and the same continuity rule
    [`build_chunk_plan`](`dascore.utils.chunk_plan.build_chunk_plan`)
    uses, so a gap here is exactly a boundary chunk refuses to merge.
    `{name}_min` is the last sample before the gap and `{name}_max` the
    first sample after it, so `gap_size` is their difference.
    """
    df, group_attrs, carried, names = _report_preamble(df, name, group, on_missing_dim)
    min_name, max_name, step_name = names
    tolerance = (
        None
        if tolerance is None
        else GapTolerance.from_user(tolerance, name, seam=True)
    )
    max_shift = resolve_max_shift(max_shift)
    columns = [min_name, max_name, step_name, "gap_size", "missing", "group_id"]
    columns += carried
    # Contiguous cells are dropped rather than concatenated: their empty
    # frames carry object-dtype attr columns, which would widen the
    # dtypes of the gappy cells they are concatenated with.
    cells = _cell_gaps(df, name, group_attrs, tolerance, max_shift)
    frames = [x for _, _, x in cells if len(x)]
    if not frames:
        return _empty_report(
            {
                min_name: df[min_name],
                max_name: df[max_name],
                step_name: df[step_name],
                "gap_size": df[max_name] - df[min_name],
                "missing": pd.Series(dtype=float),
                "group_id": pd.Series(dtype=np.int64),
            }
            | {x: df[x] for x in carried},
        )[columns]
    out = pd.concat(frames, ignore_index=True)
    return out[columns].sort_values(["group_id", min_name]).reset_index(drop=True)


def build_coverage_frame(
    df: pd.DataFrame,
    name: str,
    *,
    tolerance: float | Quantity | np.timedelta64 | GapTolerance | None = None,
    max_shift: float | None = None,
    group: str | Sequence[str] | None = None,
    on_missing_dim: Literal["raise", "drop"] = "drop",
) -> pd.DataFrame:
    """
    Summarize how much of each cell's span along ``name`` holds data.

    One row per cell, over the same cells
    [`build_gap_frame`](`dascore.utils.chunk_plan.build_gap_frame`)
    reports in and sharing its `group_id`: `span` is the cell's full
    extent, `gap_total` the sum of its gaps, `covered` the rest, and
    `coverage` their ratio.
    """
    df, group_attrs, carried, names = _report_preamble(df, name, group, on_missing_dim)
    min_name, max_name, step_name = names
    tolerance = (
        None
        if tolerance is None
        else GapTolerance.from_user(tolerance, name, seam=True)
    )
    max_shift = resolve_max_shift(max_shift)
    columns = [
        min_name,
        max_name,
        step_name,
        "span",
        "gap_total",
        "covered",
        "coverage",
        "missing",
        "group_id",
        *carried,
    ]
    rows = []
    for group_id, sub, gaps in _cell_gaps(df, name, group_attrs, tolerance, max_shift):
        low, high = sub[min_name].min(), sub[max_name].max()
        span = high - low
        # Sum through the span itself when there are no gaps, so a
        # contiguous cell's total is a zero of the span's own dtype.
        gap_total = gaps["gap_size"].sum() if len(gaps) else span - span
        rows.append(
            {
                min_name: low,
                max_name: high,
                # D7: one step everywhere in a cell, as chunk advertises.
                step_name: get_middle_value(sub[step_name].to_numpy()),
                "span": span,
                "gap_total": gap_total,
                # unknown where a gap's step is
                "missing": gaps["missing"].sum(skipna=False),
                "group_id": group_id,
            }
            | {x: sub[x].iloc[0] for x in carried}
        )
    if not rows:
        span = df[max_name] - df[min_name]
        return _empty_report(
            {
                min_name: df[min_name],
                max_name: df[max_name],
                step_name: df[step_name],
                "span": span,
                "gap_total": span,
                "covered": span,
                "coverage": pd.Series(dtype=float),
                "missing": pd.Series(dtype=float),
                "group_id": pd.Series(dtype=np.int64),
            }
            | {x: df[x] for x in carried},
        )[columns]
    out = pd.DataFrame(rows)
    out["covered"] = out["span"] - out["gap_total"]
    # A zero span is a single sample, which is wholly covered; dividing
    # would make it NaN and read as "unknown" rather than "all there".
    covered = np.asarray(to_float(out["covered"]), dtype=float)
    total = np.asarray(to_float(out["span"]), dtype=float)
    out["coverage"] = np.divide(
        covered, total, out=np.ones_like(total), where=total != 0
    )
    return out[columns].reset_index(drop=True)


def build_chunk_plan(
    df: pd.DataFrame,
    *,
    overlap=None,
    keep_partial: bool = False,
    tolerance: float | Quantity | np.timedelta64 | GapTolerance | None = None,
    max_shift: float | None = None,
    conflict: CONFLICT = "raise",
    group=None,
    on_missing_dim: Literal["raise", "drop"] = "raise",
    fill_value=None,
    **kwargs,
) -> ChunkPlan:
    """
    Build a chunk plan from a flat patch relation.

    Parameters mirror `Spool.chunk` (see the chunking formalities spec);
    one keyword names the dimension to chunk and its length (`None`/`...`
    merges). Patches join under the seam test (`_cell_seams`) and each
    stretch gets one grid (`_fit_grids`), which windows are cut from;
    ``fill_value``, which only `Spool.fill_gaps` passes, instead fills the
    seams `tolerance` joins, keeping each source's labels.
    """
    if len(kwargs) != 1:
        msg = (
            f"Chunking only supported along one dimension. You passed kwargs: {kwargs}"
        )
        raise ParameterError(msg)
    (name, value), *_ = kwargs.items()
    if explicit_ranges(value) is not None:
        msg = (
            "chunk takes lengths only. For explicit (n, 2) windows use "
            "Spool.cut, which joins each window, or Spool.select, which "
            "returns each source's matching part unjoined."
        )
        raise ParameterError(msg)
    # Fail here rather than later at assembly, so the error points at the
    # offending chunk call. See #804.
    validate_conflict(conflict)
    if tolerance is not None:
        tolerance = GapTolerance.from_user(tolerance, name, seam=True)
    max_shift = resolve_max_shift(max_shift)
    value = None if value is Ellipsis else value
    # Police quantities before merge_mode is decided: a NaN magnitude is
    # null, so a nan-valued size would silently merge the whole spool
    # when the user asked for a size *cap*.
    for label, quant in (("chunk value", value), ("overlap", overlap)):
        _validate_quantity(label, quant, name)
    merge_mode = bool(pd.isnull(value))
    if merge_mode and (keep_partial or overlap):
        msg = (
            "When chunk value is None (ie chunking is used for merging) "
            "keep_partial and overlap are not supported."
        )
        raise ParameterError(msg)
    if not merge_mode:
        assert value is not None  # pd.isnull(None) is True, so merge_mode covers it
        zero = to_timedelta64(0) if is_timedelta64(value) else 0
        if value <= zero:
            msg = "Chunk value must be greater than 0."
            raise ParameterError(msg)
    _validate_on_missing_dim(on_missing_dim)
    min_name, max_name = f"{name}_min", f"{name}_max"
    if min_name not in df.columns and not df.empty:
        msg = f"No patch in the spool has a {name!r} dimension to chunk."
        raise ChunkError(msg)
    empty_members = pd.DataFrame(
        columns=["output_id", "_patch_row", min_name, max_name, "_modified"]
    )
    params = dict(
        overlap=overlap,
        keep_partial=keep_partial,
        tolerance=tolerance,
        max_shift=max_shift,
        fill_value=fill_value,
        conflict=conflict,
        on_missing_dim=on_missing_dim,
        group=_resolve_group_attrs(group, set(df.columns)),
        sampling_group_tolerance=dc.get_config().sampling_group_tolerance,
    )
    if not df.empty:
        df = _prepare_relation(df, name, on_missing_dim)
    if df.empty:
        outputs = pd.DataFrame(columns=[min_name, max_name, "output_id"])
        return ChunkPlan(outputs, empty_members, name, value, params)
    frame_dtype = df[min_name].dtype
    df = seams = _seams(df, name, params["group"], tolerance, max_shift)
    if fill_value is None:  # a fill spreads no labels
        _warn_tolerance_joins(seams, name, tolerance)
    per_partition = _needs_partition_resolution(value, overlap)
    if not per_partition:
        value_c, overlap_c = _coerce_length_overlap(value, overlap, df[min_name].dtype)
    size_diagnostics: list[dict] = []
    sorted_df, codes, seg_starts, g_starts, g_stops = _partition_frames(
        seams, seams["_stretch"], name
    )
    n_parts = len(seg_starts)
    seg_ends = np.r_[seg_starts[1:], len(sorted_df)]
    step_all = get_interval_columns(sorted_df, name)[2].to_numpy()
    part_steps = np.array(  # D7: one step everywhere per partition
        [get_middle_value(step_all[a:b]) for a, b in zip(seg_starts, seg_ends)]
    )
    adopted_col = f"_{name}_adopted"
    adopted_all = (
        sorted_df[adopted_col].to_numpy(dtype=bool)
        if adopted_col in sorted_df
        else np.zeros(len(sorted_df), dtype=bool)
    )
    start_all = sorted_df[min_name].to_numpy()
    stop_all = sorted_df[max_name].to_numpy()
    ticks = _counts_ticks(start_all.dtype)
    # a grid labels in the dtype every source stores its labels in
    stored = {str(x) for x in sorted_df.get(f"_{name}_coord_dtype", [""])}
    stated = len(stored) == 1 and min(stored).startswith(("int", "uint", "float"))
    label_dtype = np.dtype(min(stored)) if stated else start_all.dtype
    mod_after = (
        sorted_df["_modified"].fillna(False).to_numpy(dtype=bool)
        if "_modified" in sorted_df.columns
        else np.zeros(len(sorted_df), dtype=bool)
    )
    grids: list = []  # each stretch's grid (`_fit_grids`), if fitted
    relabel = np.zeros(n_parts, dtype=bool)
    lo_all = sorted_df["_lo"].to_numpy().astype(start_all.dtype)
    hi_all = sorted_df["_hi"].to_numpy().astype(stop_all.dtype)
    mod_after = mod_after | (lo_all != start_all) | (hi_all != stop_all)
    # a row wholly inside earlier ones adds nothing, though its attrs
    # are still policed with its partition's
    keep_row = sorted_df["_keep"].to_numpy(dtype=bool)
    # kept-row (member candidate) arrays; partitions stay contiguous, so
    # partition p's kept rows sit in [koffsets[p], koffsets[p + 1])
    kept_df = sorted_df[keep_row].reset_index(drop=True)
    kcodes = codes[keep_row]
    src1, src2 = lo_all[keep_row], hi_all[keep_row]
    korig_min, korig_max = start_all[keep_row], stop_all[keep_row]
    kpids = kept_df["_patch_row"].to_numpy()
    ksteps, kmod = step_all[keep_row], mod_after[keep_row]
    kadopted = adopted_all[keep_row]
    koffsets = np.r_[0, np.cumsum(np.bincount(kcodes, minlength=n_parts))]
    shift_all = np.zeros(len(kept_df))
    gridded = np.zeros(n_parts, dtype=bool)
    lattice = _sample_lattices(kept_df, name)
    if fill_value is None:  # a fill keeps source labels
        grids, relabel, shift_all = _fit_grids(
            kept_df, kcodes, n_parts, name, max_shift, label_dtype, lattice
        )
        gridded = np.array([x is not None for x in grids], dtype=bool)
    regular = not merge_mode
    if regular:
        # a window covering a row with its own labels takes it as it is
        _, stride, *_, width, own_min, own_max = lattice
        plain = (stride == 0) | ((src1 == own_min) & (src2 == own_max))
        edge = src2 + 2 * _GRID_SNAP_RTOL * width if src2.dtype.kind == "f" else src2
        # the kept source holding each partition's last sample
        end_rows = pd.Series(src2).groupby(kcodes).idxmax().to_numpy()
    has_dtype = "_dtype" in sorted_df.columns
    if has_dtype:
        # The element dtype resolves per *output*, not per partition: a
        # partition may legitimately mix dtypes, but an output drawing
        # only from its float32 members really is float32, and claiming
        # the partition-wide upcast would both over-size a later chunk
        # and make the plan row disagree with the patch it assembles
        # (which spool equality compares). Uniform partitions (the
        # common case) resolve in one shot; mixed ones resolve per
        # output below. Carried privately (see SPOOL_EARLY_RENAMES)
        # because a size-based chunk needs it.
        dtype_all = sorted_df["_dtype"].to_numpy()
        kdtypes = dtype_all[keep_row]
        dtype_cache: dict[str, str] = {}

    def _dtypes_of(part, m_counts, rel_src, fed):
        """Each fed output's dtype; ``rel_src`` is relative to the partition."""
        part_dtypes = dtype_all[seg_starts[part] : seg_ends[part]]
        if len(pd.unique(part_dtypes)) == 1:
            uniform = _uniform_dtype_str(part_dtypes[0], dtype_cache)
            return np.full(int(fed.sum()), uniform, dtype=object)
        # each output's members are contiguous in rel_src, so the
        # per-output dtype pools are simple slices
        kdt = kdtypes[koffsets[part] : koffsets[part + 1]]
        bounds = np.cumsum(m_counts) - m_counts
        # an all-fill output draws from no member, so it is the
        # partition, not its own pool, which names its dtype
        whole = _combined_dtype(pd.Series(part_dtypes, dtype=object))
        combined = [
            _combined_dtype(
                pd.Series(
                    kdt[rel_src[bounds[out] : bounds[out] + m_counts[out]]],
                    dtype=object,
                )
            )
            if m_counts[out]
            else whole
            for out in np.flatnonzero(fed)
        ]
        return np.array(["" if x is None else str(x) for x in combined], dtype=object)

    active = np.zeros(n_parts, dtype=bool)
    fed_counts = np.zeros(n_parts, dtype=np.intp)
    sub_sample = np.zeros(n_parts, dtype=bool)
    out_starts, out_stops, out_ids = [], [], []
    m_out_ids, m_src, m_lo, m_hi, m_parts, dtype_parts = [], [], [], [], [], []
    out_grids: list = []  # per output: its slice of a grid
    pending = []  # (list slot, partition, ids, first and last positions)
    next_id = 0
    # An error hit while processing partition p is deferred until the
    # partitions before p have been policed for conflicts: the old
    # per-partition loop policed each partition before touching the
    # next, so an earlier partition's CoordMergeError outranks a later
    # partition's resolution error.
    deferred: BaseException | None = None
    for part in range(n_parts):
        part_step = part_steps[part]
        if per_partition and not merge_mode:
            # The bound itself is enforced by the packing factor, which is
            # measured in units of this step and so cancels the choice
            # (length works out to n_samples / max density either way).
            # The smallest step is still the better unit: it makes the
            # flooring granularity the finest the partition allows.
            sub = sorted_df.iloc[seg_starts[part] : seg_ends[part]]
            size_step = np.abs(step_all[seg_starts[part] : seg_ends[part]]).min()
            if gridded[part]:  # sized on the grid's spacing, floored to a tick
                grid = grids[part]
                size_step = abs(grid.step_num)
                size_step = size_step // grid.step_den if grid.exact else size_step
                if start_all.dtype.kind in "Mm":
                    size_step = np.timedelta64(int(size_step), "ns")
            try:
                value_c, overlap_c, diag = _resolve_partition_length(
                    value, overlap, sub, name, size_step, df[min_name].dtype
                )
            except Exception as exc:
                deferred = exc
                break
            if diag is not None:
                size_diagnostics.append({"first_output_id": next_id, **diag})
        if gridded[part]:
            if merge_mode:
                firsts = np.array([0], dtype=np.int64)
                lasts = np.array([grids[part].count - 1], dtype=np.int64)
            else:
                sub_sample[part] = value_c < abs(part_step)
                try:
                    firsts, lasts = _grid_windows(
                        grids[part], value_c, overlap_c, keep_partial
                    )
                except ChunkError:  # partition too short; skip (D8)
                    continue
                except Exception as exc:
                    deferred = exc
                    break
            active[part] = True
            ids_p = np.arange(next_id, next_id + len(firsts))
            next_id += len(firsts)
            # its members are found for every stretch at once, after the loop
            pending.append((len(out_starts), part, ids_p, firsts, lasts))
            held = (out_starts, out_stops, out_ids, m_out_ids, m_src, m_lo, m_hi)
            for slot in (*held, m_parts, out_grids):
                slot.append(None)
            if has_dtype:
                dtype_parts.append(None)
            continue
        if merge_mode:
            starts_p = g_starts[part : part + 1]
            stops_p = g_stops[part : part + 1]
        else:
            sub_sample[part] = not pd.isnull(part_step) and value_c < abs(part_step)
            cell, row = abs(part_step), end_rows[part]
            if lattice[2][row] > 1:  # a fractional grid's gap to its next label
                stop = to_int(g_stops[part : part + 1])
                gap = (_lattice_label(stop + 1, [x[row] for x in lattice]) - stop)[0]
                cell = np.timedelta64(gap, "ns") if g_stops.dtype.kind in "Mm" else gap
            try:
                start_stop = get_intervals(
                    g_starts[part],
                    g_stops[part],
                    value_c,
                    overlap=overlap_c,
                    # interval arithmetic is over (direction-free) envelope
                    # values; a descending coordinate's negative step would
                    # invert the final partial interval
                    step=abs(part_step),
                    keep_partials=keep_partial,
                    tail=cell,
                )
            except ChunkError:  # partition too short; skip (D8)
                continue
            except Exception as exc:
                deferred = exc
                break
            starts_p, stops_p = start_stop[:, 0], start_stop[:, 1]
            ends_p = stops_p
            if fill_value is not None:
                # filled windows hold the first source's grid, else the partition's
                grid = [x[koffsets[part]][None] for x in lattice]
                if not grid[1][0]:
                    grid[0][:] = to_int(g_starts[part])
                    grid[1][:] = to_int(abs(part_step))
                edges = g_starts[part : part + 1], g_stops[part : part + 1]
                starts_p, stops_p = _member_samples(*edges, starts_p, ends_p, grid)
                keep = starts_p <= stops_p
                starts_p, stops_p, ends_p = starts_p[keep], stops_p[keep], ends_p[keep]
            # A filled window can hold no sample at all, nor then a partition.
            if not len(starts_p):
                continue
        active[part] = True
        n_out = len(starts_p)
        ids_p = np.arange(next_id, next_id + n_out)
        next_id += n_out
        # Map each output onto the kept source rows it draws from.
        lo_k, hi_k = koffsets[part], koffsets[part + 1]
        s1, s2 = src1[lo_k:hi_k], src2[lo_k:hi_k]
        reach = starts_p, stops_p
        first_src = np.maximum(np.searchsorted(s1, starts_p, side="right") - 1, 0)
        last_src = np.searchsorted(s2, stops_p, side="left")
        last_src = np.minimum(last_src, len(s1) - 1)
        if regular:  # no member starts at or past its window's end
            last_src = np.minimum(last_src, np.searchsorted(s1, ends_p) - 1)
        m_counts = np.maximum(last_src - first_src + 1, 0)
        total = int(m_counts.sum())
        rel_out = np.repeat(np.arange(n_out), m_counts)
        offsets = np.repeat(np.cumsum(m_counts) - m_counts, m_counts)
        rel_src = np.arange(total) - offsets + np.repeat(first_src, m_counts)
        lo = np.maximum(s1[rel_src], reach[0][rel_out])
        hi = np.minimum(s2[rel_src], reach[1][rel_out])
        if regular:
            rows, windows = rel_src + lo_k, (starts_p[rel_out], ends_p[rel_out])
            lo, hi = s1[rel_src], s2[rel_src]
            whole = plain[rows] & (windows[0] <= lo) & (edge[rows] < windows[1])
            if not whole.all():
                lo, hi = _member_samples(lo, hi, *windows, [x[rows] for x in lattice])
        # A member whose window holds none of its samples drops (#1008).
        overlap_ok = lo <= hi
        if not overlap_ok.all():
            rel_src, rel_out = rel_src[overlap_ok], rel_out[overlap_ok]
            lo, hi = lo[overlap_ok], hi[overlap_ok]
            m_counts = np.bincount(rel_out, minlength=n_out)
            total = int(m_counts.sum())
        # Plan invariant: every published output has at least one member.
        # An advertised row that cannot assemble is never surfaced as a
        # runtime error; it is not surfaced at all. A fill value lifts
        # that: a window lying wholly inside a bridged hole has no source
        # to draw from and is assembled from fill alone, so the outputs
        # cover the partition evenly instead of skipping the hole.
        fed = np.ones(n_out, dtype=bool) if fill_value is not None else m_counts > 0
        fed_counts[part] = int(fed.sum())
        if regular and total:
            # an output's envelope is its members' first and last samples, or
            # with a fill, those beside its window's, which float rounding of
            # the window's edges can leave short of them
            firsts = np.flatnonzero(np.r_[True, rel_out[1:] != rel_out[:-1]])
            starts_p, stops_p = np.array(starts_p), np.array(stops_p)
            held = rel_out[firsts]
            low, high = np.minimum.reduceat(lo, firsts), np.maximum.reduceat(hi, firsts)
            if fill_value is not None:
                low = np.minimum(low, starts_p[held])
                high = np.maximum(high, stops_p[held])
            starts_p[held], stops_p[held] = low, high
        out_grids.append([None] * int(fed.sum()))
        out_starts.append(starts_p[fed])
        out_stops.append(stops_p[fed])
        out_ids.append(ids_p[fed])
        m_out_ids.append(ids_p[rel_out])
        m_src.append(rel_src + lo_k)
        m_lo.append(lo)
        m_hi.append(hi)
        m_parts.append(np.full(total, part, dtype=np.intp))
        if has_dtype:
            dtype_parts.append(_dtypes_of(part, m_counts, rel_src, fed))
    if pending:
        # every stretch's windows at once: one search maps them onto rows
        win_parts = np.concatenate([np.full(len(x[3]), x[1]) for x in pending])
        win_first = np.concatenate([x[3] for x in pending])
        win_last = np.concatenate([x[4] for x in pending])
        counts = np.array([0 if x is None else x.count for x in grids], np.int64)
        # a stretch with no grid numbers no positions any window reaches
        offsets = np.where(gridded, np.r_[0, np.cumsum(counts)[:-1]], -(2**62))
        windows = (win_parts, win_first, win_last)
        rel_out, rel_src, lo, hi = _grid_members(
            windows, kept_df, kcodes, offsets, lattice, ticks
        )
        lo, hi = _as_values(lo, start_all.dtype), _as_values(hi, start_all.dtype)
        # every position belongs to a row, so every window is fed
        opens = np.flatnonzero(np.r_[True, rel_out[1:] != rel_out[:-1]])
        assert np.array_equal(rel_out[opens], np.arange(len(win_first)))
        kind = np.int64 if _counts_ticks(label_dtype) else label_dtype
        ends = [  # a stretch's grid labels its windows' ends
            np.concatenate([grids[x[1]].labels(x[k], kind) for x in pending])
            for k in (3, 4)
        ]
        ends = [
            _as_values(x, start_all.dtype)
            for x in (np.minimum(*ends), np.maximum(*ends))
        ]
        moved = relabel[win_parts]
        starts = np.where(moved, ends[0], np.minimum.reduceat(lo, opens))
        stops = np.where(moved, ends[1], np.maximum.reduceat(hi, opens))
        win_edges = np.r_[0, np.cumsum([len(x[3]) for x in pending])]
        member_edges = np.searchsorted(rel_out, win_edges)
        for num, (slot, part, ids_p, firsts, lasts) in enumerate(pending):
            a, b = win_edges[num], win_edges[num + 1]
            ma, mb = member_edges[num], member_edges[num + 1]
            fed = np.ones(b - a, dtype=bool)
            fed_counts[part] = b - a
            out_starts[slot], out_stops[slot] = starts[a:b], stops[a:b]
            out_ids[slot] = ids_p
            m_out_ids[slot] = ids_p[rel_out[ma:mb] - a]
            m_src[slot], m_lo[slot], m_hi[slot] = rel_src[ma:mb], lo[ma:mb], hi[ma:mb]
            m_parts[slot] = np.full(mb - ma, part, dtype=np.intp)
            out_grids[slot] = list(
                zip([part] * len(firsts), firsts, lasts - firsts + 1)
            )
            if has_dtype:
                m_counts = np.bincount(rel_out[ma:mb] - a, minlength=b - a)
                local = rel_src[ma:mb] - koffsets[part]
                dtype_parts[slot] = _dtypes_of(part, m_counts, local, fed)
    # Police the partitions which produced outputs; too-short (skipped)
    # partitions are exempt, and a conflict in a partition processed
    # before a deferred error outranks that error, exactly as the
    # per-partition loop raised them.
    carried = _carried_columns(sorted_df, codes, seg_starts, name, conflict, active)
    if deferred is not None:
        raise deferred
    if size_diagnostics:
        # Diagnostics are only collected on the size-chunk path, where the
        # chunk value is an information Quantity (is_data_size gated).
        assert isinstance(value, Quantity)
        params["size"] = {
            "requested_bytes": get_byte_count(value),
            "partitions": tuple(size_diagnostics),
        }
        # warn once per call, not once per partition
        if clamped := sum(x["clamped"] for x in size_diagnostics):
            msg = (
                f"A single sample along {name!r} is larger than the requested "
                f"size of {value} in {clamped} partition(s), so those patches "
                "contain one sample each and exceed the request. Chunk another "
                "dimension first to make them smaller."
            )
            warnings.warn(msg, UserWarning, stacklevel=_user_stacklevel())
    if not fed_counts.sum():
        if sub_sample.all():
            msg = (
                f"Could not chunk. The requested chunk length {value} is "
                f"shorter than the sample step along {name!r} "
                f"({np.abs(part_steps).min()}). Use a larger chunk length."
            )
            raise ChunkError(msg)
        msg = "Could not chunk. No segments with sufficient length found."
        # Say how short the data actually is, and name the two knobs which
        # make an over-long request work. Size chunks are excluded; their
        # request is in bytes, not comparable to a coordinate duration.
        if not merge_mode and not (isinstance(value, Quantity) and is_data_size(value)):
            assert value is not None  # a null value is merge_mode
            spans = (g_stops - g_starts) + np.abs(part_steps)
            best = int(np.argmax(spans))
            longest, requested = spans[best], value
            if is_timedelta64(longest):
                # seconds, which is what a time value is given in
                longest = f"{to_float(longest)} s"
                requested = f"{to_float(requested)} s"
            else:
                # the envelope is in the longest partition's own unit, so
                # name it; the request may be stated in another (#1058)
                unit = _partition_unit(sorted_df, name, seg_starts[best])
                longest = f"{longest} {unit}".strip()
            msg = (
                f"Could not chunk. The longest contiguous segment along "
                f"{name!r} is {longest}, shorter than the requested chunk "
                f"value of {requested}. Use a smaller chunk value, a larger "
                f"tolerance to join segments separated by gaps, or "
                f"keep_partial=True to keep the short segments."
            )
        raise ChunkError(msg)
    # Assemble the outputs table: envelopes and step, then the carried
    # columns (each partition's single value repeated over its outputs),
    # then the ids.
    repeats = np.repeat(np.arange(n_parts), fed_counts)
    data: dict[str, Any] = {
        min_name: np.concatenate(out_starts),
        max_name: np.concatenate(out_stops),
        f"{name}_step": part_steps[repeats],
        **({f"_{name}_coord_dtype": min(stored)} if stated else {}),  # as loaded
    }
    for col, values in carried.items():
        data[col] = values.take(repeats).reset_index(drop=True)
    data["output_id"] = np.concatenate(out_ids)
    if has_dtype:
        data["_dtype"] = np.concatenate(dtype_parts)
    outputs = pd.DataFrame(data)
    src_rows = np.concatenate(m_src)
    unchanged = (
        (np.concatenate(m_lo) == korig_min[src_rows])
        & (np.concatenate(m_hi) == korig_max[src_rows])
        & ~kmod[src_rows]
    )
    member_ids = np.concatenate(m_out_ids)
    if kadopted.any():
        # an output only of adopted samples states no step, as they don't
        alone = pd.Series(kadopted[src_rows]).groupby(member_ids).all()
        lone = outputs["output_id"].isin(alone.index[alone.to_numpy()])
        step_col = outputs[f"{name}_step"].to_numpy()
        outputs[f"{name}_step"] = _published(step_col, lone.to_numpy())
    members = pd.DataFrame(
        {
            "output_id": member_ids,
            "_patch_row": kpids[src_rows],
            min_name: np.concatenate(m_lo),
            max_name: np.concatenate(m_hi),
            f"{name}_step": _published(ksteps, kadopted)[src_rows],
            "_modified": ~unchanged,
        }
    )
    # The partition's unit (single-valued after normalization) rides on
    # every member row: trim magnitudes mean THIS unit, not whatever
    # spelling the member's source file used.
    if (unit_col := f"_{name}_units") in sorted_df.columns:
        first_units = sorted_df[unit_col].iloc[seg_starts].reset_index(drop=True)
        member_parts = np.concatenate(m_parts)
        members[unit_col] = first_units.take(member_parts).reset_index(drop=True)
    pieces = (grids, relabel, out_grids, repeats)
    outputs, members = _describe_pieces(
        outputs, members, pieces, kept_df, src_rows, name, label_dtype
    )
    members["shift"] = shift_all[src_rows]
    outputs = _carry_grids(outputs, members, sorted_df, name, fill_value is not None)
    # integers counted in ticks go back to the frame's floats
    back = dict.fromkeys([min_name, max_name], frame_dtype)
    return ChunkPlan(outputs.astype(back), members.astype(back), name, value, params)


@compose_docstring(conflict_desc=attr_conflict_description)
def build_concat_plan(
    df: pd.DataFrame,
    *,
    conflict: CONFLICT = "raise",
    group=None,
    **kwargs,
) -> ChunkPlan:
    """
    Build a plan concatenating patches in order (`Spool.concatenate`).

    Partitions as a chunk plan does — kind (the config's attrs, compared
    for equality, so a missing value is a value), dimensions, the identity
    of every other dimension, and how the concatenated dimension is
    spelled (its units and value kind; a patch with no values along it
    states neither, so it follows the one spelling the others share and
    forms its own group when they state two) — and then groups each
    partition's rows by the requested count in the order of the dimension
    (ascending or descending as the data run; given order when the rows
    have no envelope), with no sampling tolerance, no gap test, and no
    overlap removal. Patches which cannot be concatenated together land in
    separate outputs, never in an error. Remaining attributes must hold
    equal values within an output, policed by `conflict` exactly as a
    chunk plan polices them; non-dimensional coordinates are checked when
    the output is assembled.

    Parameters
    ----------
    df
        The flat patch relation to plan over.
    conflict
        {conflict_desc}
    group
        Attributes to partition on instead of the config's `patch_kind_attrs`.
    **kwargs
        One keyword naming the dimension and the number of patches per
        output; None (or ...) puts every patch of a partition in one
        output. A dimension no patch has concatenates along a new one.
    """
    if len(kwargs) != 1:
        msg = (
            f"concatenate requires exactly one dimension keyword, got {sorted(kwargs)}"
        )
        raise ParameterError(msg)
    validate_conflict(conflict)
    ((name, value),) = kwargs.items()
    value = None if value is Ellipsis else value
    if value is not None and not isinstance(value, (int, np.integer)):
        msg = (
            "The number of patches per concatenated output is a whole number; "
            f"got {value!r}. Pass None to put a whole partition in one output."
        )
        raise ParameterError(msg)
    count = None if value is None else int(value)
    if count is not None and count < 1:
        msg = "The number of patches per concatenated output must be at least 1."
        raise ParameterError(msg)
    min_name, max_name, step_name = f"{name}_min", f"{name}_max", f"{name}_step"
    unit_col = f"_{name}_units"
    names = _resolve_group_attrs(group, set(df.columns))
    params: dict[str, Any] = dict(mode="concat", count=count, conflict=conflict)
    params["group"] = names
    if df.empty:
        outputs = pd.DataFrame({"output_id": pd.Series(dtype=np.int64)})
        members = pd.DataFrame(
            {
                "output_id": pd.Series(dtype=np.int64),
                "_patch_row": pd.Series(dtype=object),
                "_modified": pd.Series(dtype=bool),
            }
        )
        return ChunkPlan(outputs, members, name, value, params)
    df = _ensure_patch_row(df).reset_index(drop=True)
    # rows which carry the name as a dimension; the others (a non-dimensional
    # coordinate of that name, or none) gain a new dimension in its place
    along = _structural(df, name)
    key_col = f"_{name}_def_key"
    has_coord = along | (df[key_col].notna().to_numpy() if key_col in df else False)
    # the public spelling names the coordinate just as the private one does
    envelope_cols = [
        x for x in (min_name, max_name, step_name, unit_col, f"{name}_units") if x in df
    ]
    if envelope_cols and (df[envelope_cols].notna().any(axis=1) & ~has_coord).any():
        # the envelope names belong to the dimension about to be created
        msg = (
            f"Cannot concatenate along the new dimension {name!r}: the "
            f"attributes {envelope_cols} would describe it. Rename them first."
        )
        raise ParameterError(msg)
    has_envelope = min_name in df.columns and max_name in df.columns
    if has_envelope and along.all() and df[[min_name, max_name]].isna().all(axis=None):
        # every member carries the dimension and none states a value for
        # it, as after reducing it with the default dim_reduce: the join
        # is an all-missing coordinate, which is rarely what was meant
        msg = (
            f"Concatenating along {name!r}, for which no patch holds values: "
            "the result's coordinate is all missing. This follows reducing "
            "the dimension with the default dim_reduce='empty'; reduce with "
            "dim_reduce='min' (or another reducer) to keep a label per patch."
        )
        warnings.warn(msg, DASCoreWarning, stacklevel=_user_stacklevel())
    if has_envelope:
        # one spelling per dimensionality, as a chunk plan: metres and
        # centimetres plan together and members convert on loading; only
        # numbers are stored per spelling, so only they convert
        df = _normalize_numeric_units(df, name)
    # The concatenated dimension's spelling partitions too: values with no
    # units cannot join values with units (they would be mixed), and a
    # datetime cannot join a number sharing its name and unit. A row with
    # no values along the dimension states no spelling; it follows the one
    # spelling the stated rows share (assembly fills its placeholders from
    # them, see `_joinable`) and forms its own group when they state two.
    spelling = pd.Series("", index=df.index, dtype=object)
    if unit_col in df.columns:
        # `known_only` first: a coordinate stated with no units spells that
        # null or "", and the two are one statement here as everywhere else.
        units = known_only(df[unit_col]).fillna("unitless").astype(object)
        if has_envelope:
            units = units.where(df[min_name].notna(), "")
        spelling = spelling.str.cat(units.where(along, ""))
    if has_envelope:
        family = df[min_name].map(_value_family)
        spelling = spelling.str.cat(family.where(along, ""), sep="|")
    keys: list = [_kind_codes(df, list(names))]
    if "dims" in df.columns:
        keys.append(df["dims"])
    for dim_key in _dim_def_key_columns(df, name):
        dim = dim_key[1 : -len("_def_key")]
        key = df[dim_key] if dim_key in df.columns else pd.Series(None, index=df.index)
        # a coordinate's identity partitions only the rows it is a dimension
        # of; elsewhere it is non-dimensional and `conflict` polices it
        key = key.astype(object).where(_structural(df, dim), "")
        env_cols = [f"{dim}_{x}" for x in ("min", "max", "step")]
        if all(x in df.columns for x in env_cols):
            # a re-planned output states no identity key; its envelope says
            # what it is, which is enough to tell two of them apart
            spelled = df[env_cols].itertuples(index=False, name=None)
            env = pd.Series(["env:" + "|".join(map(str, x)) for x in spelled])
            key = key.where(key.notna(), env)
        keys.append(key)
    # Everything a row states about itself; the dimension's spelling then
    # splits these further, the value-less rows following the stated ones.
    base = df.groupby(keys, dropna=False, sort=False).ngroup()
    stated = spelling.where(spelling.str.strip("|") != "")
    by_base = stated.groupby(base, sort=False)
    lone = by_base.transform("nunique") == 1
    spelling = spelling.mask(stated.isna() & lone, by_base.transform("first"))
    labels = df.groupby([base, spelling], dropna=False, sort=False).ngroup().to_numpy()
    # Order: partitions in order of first appearance, rows within one the
    # way the data run along the dimension (ascending by start, descending
    # by stop), then cut into runs of `count`. A partition without a known
    # step (irregular values, labels) or along a new dimension cannot tell
    # its orientation from envelopes, so it keeps its order; rows without
    # an envelope sort last.
    within = np.zeros(len(df))
    if has_envelope:
        starts = _order_key(df[min_name])
        stops = _order_key(df[max_name])
        no_steps = np.full(len(df), np.nan)
        steps = _directions(df[step_name]) if step_name in df else no_steps
        # the first row's step, null or not: a partition whose first row
        # has no step has no orientation, whatever a later row knows
        rows = pd.Series(np.arange(len(df)))
        first_step = steps[rows.groupby(labels).transform("first").to_numpy()]
        descending = np.nan_to_num(first_step, nan=0.0) < 0
        within = np.where(descending, -stops, starts)
        within = np.where(np.isnan(first_step) | ~along, 0.0, within)
        within = np.where(np.isnan(within), np.inf, within)
    perm = np.lexsort((np.arange(len(df)), within, labels))
    sorted_df = df.iloc[perm].reset_index(drop=True)
    part = labels[perm]
    position = pd.Series(part).groupby(part).cumcount().to_numpy()
    run = np.zeros_like(position) if count is None else position // count
    codes = pd.DataFrame({"p": part, "r": run}).groupby(["p", "r"], sort=False).ngroup()
    codes = codes.to_numpy()
    n_out = int(codes.max()) + 1
    sizes = np.bincount(codes, minlength=n_out)
    seg_starts = np.r_[0, np.cumsum(sizes)[:-1]]
    active = np.ones(n_out, dtype=bool)
    # non-dimensional coordinates are described from the members when the
    # catalog is derived (riders of the dimension follow it member by
    # member, so their envelopes differ by design) and policed by their
    # identity keys below; their envelopes are not attrs to carry
    blanked = {}
    for coord in _identified_coords(sorted_df) - {name}:
        structural = _structural(sorted_df, coord)
        if structural.all():
            continue
        for suffix in ("min", "max", "step"):
            if (col := f"{coord}_{suffix}") in sorted_df.columns:
                # kept where the coordinate is a dimension of the row
                blanked[col] = sorted_df[col].where(structural)
    policed_df = sorted_df.assign(**blanked)
    carried = _carried_columns(policed_df, codes, seg_starts, name, conflict, active)
    data: dict[str, Any] = {k: v.reset_index(drop=True) for k, v in carried.items()}
    if "data_units" in sorted_df.columns:
        # Data units scale the data, so assembly converts every member to
        # the first units any member states rather than letting a loosened
        # policy splice differently scaled samples (`concatenate_planned`).
        # The row has to say the same, or it would describe kilometre-scaled
        # samples by a first member's silence -- or, under `drop`, not at all.
        first_stated = known_only(sorted_df[["data_units"]])["data_units"]
        first_stated = first_stated.groupby(codes, sort=True).first()
        if first_stated.notna().any():
            data["data_units"] = first_stated.reset_index(drop=True)
    if has_envelope and not along.all():
        # rows gaining the dimension have no values along it; what their
        # column holds belongs to a coordinate the new dimension replaces
        kept = along[perm]
        blank = {c: sorted_df[c].where(kept) for c in envelope_cols if c in sorted_df}
        sorted_df = sorted_df.assign(**blank)
    by_output = sorted_df.groupby(codes, sort=True)
    if has_envelope:
        # A member with no values along the dimension (an aggregated
        # coordinate) has a null envelope, which the output's skips.
        # Labels are the awkward pair: they have no missing value, so a
        # member which states none cannot join them, and such an output
        # claims nothing until assembly refuses it.
        families = sorted_df[min_name].map(_value_family)
        by_family = families.replace("", np.nan).groupby(codes, sort=True)
        text = (by_family.first() == "text").fillna(False).to_numpy()
        blank = (families == "").groupby(codes, sort=True).any().to_numpy()
        undecided = pd.Series(text & blank)
        data[min_name] = by_output[min_name].min().where(~undecided).to_numpy()
        data[max_name] = by_output[max_name].max().where(~undecided).to_numpy()
        if unit_col in sorted_df.columns:
            # a member with no values along the dimension states no unit;
            # the output speaks the unit its stated members share, which is
            # what assembly joins in
            known = known_only(sorted_df[[unit_col]])[unit_col]
            data[unit_col] = known.groupby(codes, sort=True).first().to_numpy()
        if step_name in sorted_df.columns:
            data[step_name] = _concatenated_steps(sorted_df, codes, name)
    if "dims" in sorted_df.columns:
        dims = sorted_df["dims"].to_numpy()[seg_starts].astype(str)
        new_dim = np.array([name not in d.split(",") for d in dims])
        # a dimension the members carry without values is resized the same
        # way, and is identified the same way: by how long it comes out
        stated = np.zeros(n_out, dtype=bool)
        if min_name in data:
            stated = ~pd.isnull(pd.Series(data[min_name])).to_numpy()
        value_less = new_dim | ~stated
        if new_dim.any():
            # an output whose members lack the dimension gains it, one
            # sample per member, as a dimension without values (the name
            # may have been a non-dimensional coordinate, which the new
            # dimension replaces, so no envelope is claimed for it)
            data["dims"] = [
                ",".join([*[x for x in d.split(",") if x], name]) if n else d
                for d, n in zip(dims, new_dim)
            ]
            for x in (min_name, max_name, step_name):
                if x in data:
                    data[x] = [None if n else v for v, n in zip(data[x], new_dim)]
            # its identity is that of a value-less coordinate of its length
            # (what ingesting the assembled patch would record), so outputs
            # of different counts stay apart when a later plan partitions
            # on it
            keys = list(data.get(key_col, pd.Series([None] * n_out, dtype=object)))
            keys = [
                f"fp:{dc.core.coords.get_coord(shape=(int(s),)).data_id}" if n else k
                for k, n, s in zip(keys, new_dim, sizes)
            ]
            data[key_col] = pd.Series(keys, dtype=object)
        if (existing := value_less & ~new_dim).any():
            # a dimension the members carry without values is resized too,
            # and the relation says nothing about how long it comes out;
            # what the members are is the best identity available, and it
            # is not a claim about values
            keys = list(data.get(key_col, pd.Series([None] * n_out, dtype=object)))
            member_keys = _member_key_digests(sorted_df, codes, name)
            keys = [
                digest if e and digest is not None else k
                for k, e, digest in zip(keys, existing, member_keys)
            ]
            data[key_col] = pd.Series(keys, dtype=object)
    if "_dtype" in sorted_df.columns:
        all_dtypes = sorted_df["_dtype"]
        dtypes = by_output["_dtype"]
        uniform = dtypes.nunique(dropna=False).to_numpy() == 1
        first = dtypes.first().to_numpy(dtype=object)
        combined = [
            first[i] if uniform[i] else _combined_dtype(all_dtypes.iloc[a : a + n])
            for i, (a, n) in enumerate(zip(seg_starts, sizes))
        ]
        if conflict != "raise" and "data_units" in sorted_df.columns:
            # settling on one spelling of the data units converts the members
            # stated otherwise, which floats an integer array; `drop` reaches
            # the same conversion, since units are reconciled either way
            stated = known_only(sorted_df[["data_units"]])["data_units"]
            converts = stated.groupby(codes, sort=True).nunique(dropna=True) > 1
            combined = [
                str(np.result_type(x, np.float64)) if c and not pd.isnull(x) else x
                for x, c in zip(combined, converts.to_numpy())
            ]
        data["_dtype"] = ["" if pd.isnull(x) else str(x) for x in combined]
    data["output_id"] = np.arange(n_out)
    outputs = pd.DataFrame(data)
    # members load whole unless the rows are themselves trims (a re-plan
    # over a chunked view), whose ranges they then keep
    members = pd.DataFrame(
        {"output_id": codes, "_patch_row": sorted_df["_patch_row"].to_numpy()}
    )
    if has_envelope:
        for col in (min_name, max_name, step_name, unit_col):
            if col in sorted_df.columns:
                members[col] = sorted_df[col].to_numpy()
    modified = sorted_df["_modified"] if "_modified" in sorted_df.columns else False
    members["_modified"] = np.asarray(modified, dtype=bool)
    return ChunkPlan(outputs, members, name, value, params)


def _normalize_numeric_units(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    `_normalize_chunk_units` for the numeric rows of an envelope of mixed kinds.

    A dimension name shared by numeric and, say, datetime coordinates holds
    an object column, which the chunk normalizer leaves alone; the numeric
    rows still deserve one spelling per dimensionality.
    """
    min_name, max_name, step_name = f"{name}_min", f"{name}_max", f"{name}_step"
    if min_name not in df.columns:
        return df
    numeric = df[min_name].map(_value_family).to_numpy() == "number"
    if not numeric.any():
        return df
    cols = [x for x in (min_name, max_name, step_name, f"_{name}_units") if x in df]
    # the numeric rows are coerced from the column's kind-agnostic dtype;
    # otherwise the chunk normalizer, which reads the dtype, would pass
    # over rows which are numbers in an object column
    part = df.loc[numeric, :].copy()
    for col in (min_name, max_name, step_name):
        if col in part:
            part[col] = pd.to_numeric(part[col])
    part = _normalize_chunk_units(part, name)
    if numeric.all():
        return df.assign(**{c: part[c] for c in cols})
    out = df.copy()
    for col in cols:
        out[col] = out[col].astype(object)
        out.loc[numeric, col] = part[col].astype(object)
    return out


def _directions(steps: pd.Series) -> np.ndarray:
    """The sign of each step (±1.0), NaN where missing or without a sign."""

    def direction(step):
        if pd.isnull(step) or isinstance(step, str):
            return np.nan
        zero = step * 0
        if step == zero:  # a constant coordinate runs neither way
            return np.nan
        return -1.0 if step < zero else 1.0

    return np.array([direction(x) for x in steps], dtype=float)


def _value_family(value) -> str:
    """The kind of an envelope value: datetime, timedelta, number, or text."""
    if pd.isnull(value):
        return ""
    if isinstance(value, np.datetime64 | pd.Timestamp):
        return "datetime"
    if isinstance(value, np.timedelta64 | pd.Timedelta):
        return "timedelta"
    if isinstance(value, str):
        return "text"
    return "number"


def _identified_coords(df: pd.DataFrame) -> set[str]:
    """The coordinates a relation identifies (those with a def-key column)."""
    return {
        col[1 : -len("_def_key")]
        for col in df.columns
        if col.startswith("_") and not col.startswith("__") and col.endswith("_def_key")
    }


def _structural(df: pd.DataFrame, coord: str) -> np.ndarray:
    """Per row, whether `coord` is one of the row's dimensions."""
    if "dims" not in df.columns:
        return np.zeros(len(df), dtype=bool)
    dims = df["dims"].astype(str).str.split(",")
    return dims.apply(lambda d: coord in d).to_numpy(dtype=bool)


def _member_key_digests(sorted_df: pd.DataFrame, codes: np.ndarray, name: str):
    """
    A "cat:" identity per output, digesting the identities it joins.

    Outputs joining the same members in the same order come out the same;
    the digest is deliberately not an "fp:" key, since it claims nothing
    about values the relation never saw.
    """
    col = f"_{name}_def_key"
    assert col in sorted_df.columns, "a value-less target states an identity"
    out: list[str | None] = []
    for _, group in sorted_df[col].groupby(codes, sort=True):
        if group.isna().any():
            out.append(None)
            continue
        if len(group) == 1:
            # nothing was joined, so the member's own identity stands
            out.append(str(group.iloc[0]))
            continue
        spelled = ",".join(map(str, group))
        out.append(f"cat:{hashlib.sha256(spelled.encode()).hexdigest()[:32]}")
    return out


def _order_key(values: pd.Series) -> np.ndarray:
    """
    Sortable floats for an envelope column: dense ranks of the native values.

    Ranks rather than float conversions, which would fold nanosecond
    timestamps a few hundred apart into one key. Missing values rank NaN.
    """
    out = np.full(len(values), np.nan)
    families = values.map(_value_family).to_numpy()
    for family in {x for x in families if x}:
        # each kind ranks among its own (kinds never share a partition)
        mask = families == family
        out[mask] = _coerce(values[mask], family).rank(method="dense").to_numpy()
    return out


def _coerce(values: pd.Series, family: str) -> pd.Series:
    """Give an object column of one value family its native dtype."""
    if family == "datetime":
        return pd.to_datetime(values)
    if family == "timedelta":
        return pd.to_timedelta(values)
    if family == "text":
        return values.astype(str)
    return pd.to_numeric(values)


def _concatenated_steps(sorted_df: pd.DataFrame, codes: np.ndarray, name: str):
    """
    The step of each output's concatenated coordinate, where it stays even.

    One shared step and each member starting one step after the previous
    member stops keep the coordinate a range; anything else (a gap, an
    overlap, mixed steps, no step) leaves it without a step, as the
    assembled coordinate will be.
    """
    steps = sorted_df[f"{name}_step"]
    by_output = steps.groupby(codes, sort=True)
    one_step = (by_output.nunique(dropna=True) == 1) & (
        by_output.count() == by_output.size()
    )
    starts, stops = sorted_df[f"{name}_min"], sorted_df[f"{name}_max"]
    same_output = pd.Series(codes).shift(1).to_numpy() == codes
    # descending data run from their max to their min, so the next member
    # starts (at its max) one step below the previous member's min. Each
    # value family does its own arithmetic (an output holds one family);
    # labels have none, so a text output has no step.
    families = starts.map(_value_family).to_numpy()
    follows = np.zeros(len(sorted_df), dtype=bool)
    for family in {x for x in families if x and x != "text"}:
        mask = families == family
        lo, hi = _coerce(starts[mask], family), _coerce(stops[mask], family)
        step = steps[mask]
        if family == "number":
            step = pd.to_numeric(step)
        ascending = _directions(step) >= 0
        forward = (lo == hi.shift(1) + step).to_numpy()
        backward = (hi == lo.shift(1) + step).to_numpy()
        follows[mask] = np.where(ascending, forward, backward)
    follows |= ~same_output
    contiguous = pd.Series(follows).groupby(codes).all()
    first = by_output.first()
    return first.where(one_step & contiguous).to_numpy()


def _sample_lattices(rows: pd.DataFrame, name: str):
    """
    Each row's source sample grid: (base, stride, den, phase, width, min, max).

    Tick labels are ``base + (phase + k * stride) // den``, float ones
    ``base + k * stride``, from the source's first label ``base``. A row with
    no known origin, or a source counted in other units, has a zero stride.
    ``width`` is the step's size; min/max are a whole row's own labels.
    """
    low, high = rows[f"{name}_min"], rows[f"{name}_max"]
    blank = pd.Series(None, index=rows.index, dtype=object)
    whole = ~rows.get("_modified", blank).fillna(False).astype(bool)
    stated = rows.get(f"_{name}_units_source", blank)
    converted = stated.notna() & (stated != rows.get(f"_{name}_units"))
    first, known = low, whole
    # a plan member keeps its source's range, a selected view its envelope
    sources = [rows.get(f"_{name}_src_low")]
    if (envelope := rows.get(f"_{name}_source_envelope")) is not None:
        pick = lambda x: x.get(f"{name}_min") if isinstance(x, dict) else None  # noqa
        sources.append(envelope.map(pick).astype(low.dtype))
    for source in (x for x in sources if x is not None):
        usable = source.notna() & ~converted
        first, known = first.where(known | ~usable, source), known | usable
    steps = rows[f"{name}_step"]
    base, steps = first.to_numpy(), steps.fillna(steps.iloc[:0].sum() * 0)
    if base.dtype.kind in "Mm":  # nanosecond ticks, whatever the frame's unit
        base, steps = to_int(base), to_int(pd.to_timedelta(steps).to_numpy())
    width = np.abs(np.asarray(steps))
    stride = np.where(known, width, 0 * width)
    if base.dtype.kind in "iu":  # integer labels sit on whole ticks
        stride = np.where(stride % 1 == 0, stride, 0).astype(np.int64)
    den, phase = np.ones(len(rows), np.int64), np.zeros(len(rows), np.int64)
    codes, grids = pd.factorize(rows.get(f"_{name}_grid", blank))
    for code, grid in enumerate(grids):
        if base.dtype.kind in "iu" and grid[1] > 1:  # an exact fractional grid
            num, step_den, offset, count = grid
            on = (codes == code) & (known & ~converted).to_numpy()
            # a descending grid's first label is its last position's
            stride[on], den[on] = abs(num), step_den
            phase[on] = (min(num, 0) * (count - 1) + offset) % step_den
    own = (x.where(whole).to_numpy() for x in (low, high))
    return base, stride, den, phase, width, *own


def _lattice_label(x, lattice, before=False):
    """The first label at or after each value (ticks), or the last one before."""
    base, stride, den, phase, *_ = lattice
    if x.dtype.kind == "f":
        return base + (np.ceil((x - base) / stride - _GRID_SNAP_RTOL) - before) * stride
    # in Python ints: a long fractional grid's terms overflow int64 products
    x, base, stride, den, phase = (
        np.asarray(v, dtype=object) for v in (x, *lattice[:4])
    )
    found = (
        base + (phase - ((phase - (x - base) * den) // stride + before) * stride) // den
    )
    return found.astype(np.int64)


def _member_samples(firsts, lasts, starts, ends, lattice):
    """
    Each member's first and last label in its window ``[start, end)``.

    Edges not on the row's own labels move onto its lattice (every tick if
    unknown); a float row without one keeps the window's edges, moved in by
    a read's pad (see `_row_bounds`). A high below its low holds nothing.
    """
    base, stride, den, phase, width, own_min, own_max = lattice
    kind, unit = firsts.dtype.kind, f"{firsts.dtype.kind}8[ns]"
    ticks = (lambda x: x.astype(unit).view(np.int64)) if kind in "Mm" else np.asarray
    if (
        kind in "iu"
    ):  # an integer label is in a window from the tick at or past its edge
        ticks = lambda x: np.ceil(x).astype(np.int64)  # noqa
    inside = np.maximum(firsts, starts)
    low, end, last = ticks(inside), ticks(ends), ticks(lasts)
    grid = (base, np.where(stride > 0, stride, 1), den, phase)
    bump = 2 * _GRID_SNAP_RTOL * width if kind == "f" else 1
    on_grid = _lattice_label(low, grid)
    at_last, at_end = (_lattice_label(x, grid, True) for x in (last + bump, end))
    top = np.minimum(at_last, at_end)
    lo = np.where(inside == own_min, low, on_grid)
    hi = np.where((at_last < at_end) & (lasts == own_max), last, top)
    # a row's own label and a lattice label naming one sample differ by rounding
    lo, hi = np.where(on_grid <= top, np.sort([lo, hi], 0), [lo, hi])
    if kind == "f":  # without a lattice, move window edges in by a read's pad
        free, pad = stride == 0, _READ_PAD * width
        inner = np.minimum(low + pad - bump, np.maximum(last, low - bump))
        short = np.nextafter(end - pad - bump, -np.inf)
        short = np.where(low < end - bump, np.maximum(short, low), short)
        lo = np.where(free, np.where(firsts < starts, inner, low), lo)
        hi = np.where(free, np.where(lasts < ends, last, short), hi)
    if kind in "Mm":
        lo, hi = (x.astype(np.int64).view(unit).astype(firsts.dtype) for x in (lo, hi))
    return lo, hi


def _snapped_cuts(cuts, start, step) -> list:
    """
    Return each cut moved up to the first sample at or after it.

    A cut is an arbitrary value — an epoch boundary owes the sample grid
    nothing — while the pieces are described by inclusive envelopes,
    which can only name samples. Snapping *up* keeps the split faithful
    to a half-open ``[start, end)`` interval: the sample a cut lands
    exactly on opens the piece that cut opens, and a cut falling between
    two samples leaves the earlier one behind.
    """
    out = []
    for cut in cuts:
        # Dividing the native types rounds once, where converting each to
        # float seconds first would round three times — enough to lift an
        # on-grid cut above its own index, putting the boundary sample in
        # the piece before the boundary. One rounding is still a rounding,
        # so the ratio only starts the search and comparing values on the
        # grid settles it; each loop steps at most once.
        index = math.ceil((cut - start) / step)
        while start + (index - 1) * step >= cut:
            index -= 1
        while start + index * step < cut:
            index += 1
        # Cuts sit above the row's minimum (see `build_subdivision_plan`'s
        # contract), so each one really does open a piece.
        assert index >= 1
        # Two cuts inside one sample interval name one split: the epoch
        # between them covers no sample of this row.
        if (value := start + index * step) not in out:
            out.append(value)
    # Which cuts a row has is a set, not a sequence — but the pieces are
    # read off consecutive pairs, so an unordered one would describe
    # envelopes running backwards rather than an error.
    return sorted(out)


def _dim_columns(df: pd.DataFrame, name: str) -> tuple[str, str, str]:
    """Return the envelope columns of one dimension, checked present."""
    columns = (f"{name}_min", f"{name}_max", f"{name}_step")
    assert set(columns).issubset(df.columns)
    return columns


def subdivision_pieces(df: pd.DataFrame, cuts, name: str) -> list[list[tuple]]:
    """
    Return the inclusive pieces each row's cuts divide it into.

    The pieces of a row partition its samples: none is dropped and none
    is duplicated, whatever instant a cut falls on, because each piece
    ends one step short of where the next begins.

    Parameters
    ----------
    df
        The relation being subdivided; one row per patch.
    cuts
        One sequence of cut values per row, in the row's own units, each
        above that row's minimum and no greater than its maximum — a cut
        on the maximum yields a one-sample final piece. Order does not
        matter. A cut opens a new piece at the first sample at or after
        it, so a row with `n` distinct cuts becomes at most `n + 1`
        pieces.
    name
        The dimension being subdivided.

    Notes
    -----
    Every cut row needs a usable step to find its sample grid with, so
    callers must reject a null or zero step themselves — they are the
    ones which can name the file it came from.
    """
    min_name, max_name, step_name = _dim_columns(df, name)
    assert len(cuts) == len(df)
    df = df.reset_index(drop=True)
    out = []
    for position, row_cuts in enumerate(cuts):
        start, stop = df.at[position, min_name], df.at[position, max_name]
        # Envelopes are value-ordered whatever the coordinate's
        # orientation, so the grid is walked by the step's magnitude.
        step = abs(df.at[position, step_name])
        assert not len(row_cuts) or (not pd.isnull(step) and step)
        bounds = [start, *_snapped_cuts(row_cuts, start, step)]
        out.append(
            [
                (low, bounds[index + 1] - step if index + 1 < len(bounds) else stop)
                for index, low in enumerate(bounds)
            ]
        )
    return out


def build_subdivision_plan(df: pd.DataFrame, pieces, name: str) -> ChunkPlan:
    """
    Build a plan cutting each row of a relation into its own pieces.

    Unlike a chunk plan, no row ever meets another: each output is one
    contiguous piece of exactly one source row. A row handed the whole of
    its own envelope passes through as a single unmodified output, and
    one handed no pieces at all leaves the relation — which is how a
    selection along the dimension drops a patch none of whose samples it
    keeps. This is what operations which re-describe patches rather than
    restructure them (`Spool.conform_to_inventory`, inventory-backed
    channel selection) need.

    Parameters
    ----------
    df
        The relation to subdivide; one row per patch.
    pieces
        One sequence of inclusive `(low, high)` envelopes per row, in the
        row's own units and already on its sample grid. They must be
        disjoint within a row but need not be in envelope order —
        `Spool.expand_by` emits them grouped by value. `subdivision_pieces`
        builds them from cut values, and a mask over the row's samples
        gives them directly.
    name
        The dimension being subdivided.
    """
    min_name, max_name, _ = _dim_columns(df, name)
    # One entry per row, even where it is empty: a short sequence would
    # drop the rows past its end from the plan, and so from the spool.
    assert len(pieces) == len(df)
    df = _ensure_patch_row(df).reset_index(drop=True)
    positions = [n for n, x in enumerate(pieces) for _ in x]
    lows, highs = ([x[k] for row in pieces for x in row] for k in (0, 1))
    # A piece covering its whole row *is* the row, and saying so is what
    # lets it load without a trim.
    whole = zip(df[min_name].to_numpy()[positions], df[max_name].to_numpy()[positions])
    modified = [tuple(x) != y for x, y in zip(zip(lows, highs), whole)]
    return _pieces_plan(df, positions, {name: (lows, highs)}, modified)


def grid_fits(grid, low, high) -> bool:
    """Whether an exact grid's first and last labels are the tick bounds given."""
    if not isinstance(grid, tuple) or _value_family(low) in ("number", "text"):
        return False
    num, den, phase, count = grid
    start, end = (int(to_int(x)) for x in ((high, low) if num < 0 else (low, high)))
    return start + (phase + (count - 1) * num) // den == end


def _lattice_span(lattice, low, high):
    """The first and last sample k a lattice labels inside [low, high] ticks."""
    num, den, _, ideal = lattice
    top, bottom = (int(high) + 1) * den - ideal, int(low) * den - ideal
    if num > 0:
        return -(-bottom // num), -(-top // num) - 1
    return top // num + 1, bottom // num


def _warn_tolerance_joins(seams: pd.DataFrame, name: str, tolerance) -> None:
    """Warn, as v0.1.24 did, where only `tolerance` joins a seam over 1.5 steps."""
    if tolerance is None:
        return
    joined = seams["_by_tolerance"].to_numpy(dtype=bool) & seams["_keep"].to_numpy()
    widths = seams["_seam"].to_numpy(dtype=np.float64)[joined]
    if not (widths > DEFAULT_TOLERANCE + 1e-9).any():
        return
    shown = f"{tolerance.count:g}" if tolerance.count is not None else tolerance.excess
    msg = (
        f"There is a gap in the patch along dimension {name} but a merge "
        f"tolerance of {shown} was used to force merging the patches. As a "
        "result, some chunked patches have their labels spread evenly across "
        "the gap, lengthening their sampling step."
    )
    warnings.warn(msg, UserWarning, stacklevel=_user_stacklevel())


def _describe_pieces(outputs, members, pieces, rows, src, name, dtype):
    """
    State each output's slice of its stretch's grid, and why each piece starts.

    An output whose stretch has a grid it relabels states that grid's step
    and slice (``_{name}_stretch``, which its members carry to assembly);
    one whose stretch has none says "labels", so assembly keeps its
    labels as recorded. Each member says why it starts where it does
    (`reason`): one starting where its source's kept labels do opens a
    stretch ("stretch" or "gap") or joins the piece before it across a
    seam the seam test passes ("seam") or only `tolerance` does
    ("tolerance"); any other starts a window ("window").
    """
    grids, relabel, out_grids, parts = pieces
    slices = [x if x is not None else (-1, 0, 0) for part in out_grids for x in part]
    step_col, grid_col = f"{name}_step", f"_{name}_grid"
    steps = outputs[step_col].to_numpy().copy()
    if steps.dtype.kind == "m":  # a frame may count steps more coarsely
        steps = steps.astype("m8[ns]")
    exact = list(outputs[grid_col]) if grid_col in outputs else [None] * len(outputs)
    specs = [None] * len(outputs)
    for out, (part, first, size) in enumerate(slices):
        first, size = int(first), int(size)  # Python ints: grid terms are big
        if not grids or (part >= 0 and not relabel[part]):
            continue
        grid = grids[parts[out]]
        if grid is None:
            specs[out], exact[out] = "labels", None
            continue
        piece = grid.sliced(first, 1, size)
        steps[out] = piece.step(dtype)
        fraction = piece.step_den > 1
        exact[out] = (
            (piece.step_num, piece.step_den, piece.phase, size) if fraction else None
        )
        terms = (grid.origin, grid.step_num, grid.step_den, grid.count, grid.phase)
        specs[out] = (*terms, first, size, str(dtype))
    stretch_col = f"_{name}_stretch"
    steps = _published(steps, np.array([x == "labels" for x in specs], dtype=bool))
    outputs = outputs.assign(**{step_col: steps, grid_col: exact, stretch_col: specs})
    ids = outputs["output_id"].to_numpy()
    by_id = dict(zip(ids, range(len(ids))))
    position = members["output_id"].map(by_id).to_numpy()
    # a descending source starts, in coordinate order, at its top
    down = rows["_forward"].to_numpy()[src] < 0
    edge = np.where(down, members[f"{name}_max"], members[f"{name}_min"])
    kept = np.where(down, rows["_hi"].to_numpy()[src], rows["_lo"].to_numpy()[src])
    reason = np.where(edge == kept, rows["_joined"].to_numpy()[src], "window")
    members = members.assign(
        **{
            stretch_col: [specs[x] for x in position],
            "reason": reason,
            "dropped": rows["_dropped"].to_numpy()[src],
        }
    )
    return outputs, members


def stretch_coordinate(spec, units=None):
    """
    The coordinate an output's slice of its stretch's grid labels it with.

    ``spec`` is what a plan's ``_{dim}_stretch`` column holds: the stretch
    grid's terms, then the output's first position, its count, and the
    coordinate's dtype. A float slice keeps its parent grid, so its labels
    are the stretch's bit for bit.
    """
    from dascore.core.coords import get_coord  # noqa: PLC0415

    origin, num, den, count, phase, first, length, dtype = spec
    # Python ints: a platform's numpy int times a big grid term overflows
    run = Grid(origin, num, den, count, phase=phase).sliced(int(first), 1, int(length))
    return get_coord(runs=(run,), dtype=dtype, units=units)


def _carry_grids(outputs, members, sources, name, fill=False):
    """
    Give outputs on one exact grid the grid of their samples.

    A whole-tick envelope cannot restate a fractional step, so a later plan
    over the output would cut on a drifting lattice. A grid's labels are
    ``(ideal + k * num) // den`` ticks; members on one lattice share num,
    den and ideal origin modulo num. A grid counts only where its labels
    end on the row's bounds (a trimmed row's do not) and its members hold
    every sample, or a fill writes the rest on its first member's grid.
    """
    col, (low, high, _) = f"_{name}_grid", _dim_columns(outputs, name)
    fitted = outputs.get(f"_{name}_stretch", pd.Series(None, index=outputs.index))
    fitted = fitted.notna().to_numpy()
    if col not in sources or fitted.all():
        return outputs
    lattice = {}  # source -> (num, den, ideal % |num|, ideal)
    for pid, grid, lo, hi in zip(*(sources[x] for x in ("_patch_row", col, low, high))):
        if grid_fits(grid, lo, hi):  # a trimmed row still states its source's
            num, den, phase, _ = grid  # a descending grid starts at its max
            ideal = int(to_int(hi if num < 0 else lo)) * den + phase
            lattice[pid] = (num, den, ideal % abs(num), ideal)
    first, kinds, held = {}, {}, {}
    rows = zip(*(members[x] for x in ("output_id", "_patch_row", low, high)))
    for out, pid, lo, hi in rows:
        first.setdefault(out, pid)
        kinds.setdefault(out, set()).add(lattice.get(pid, (None,))[:3])
        if pid in lattice:
            k0, k1 = _lattice_span(lattice[pid], to_int(lo), to_int(hi))
            held[out] = held.get(out, 0) + k1 - k0 + 1
    grids = []
    for out, lo, hi in zip(outputs["output_id"], outputs[low], outputs[high]):
        if first.get(out) not in lattice or (len(kinds[out]) != 1 and not fill):
            grids.append(None)
            continue
        num, den, _, ideal = lattice[first[out]]
        k0, k1 = _lattice_span(lattice[first[out]], to_int(lo), to_int(hi))
        # slicing keeps a reduced grid reduced: num and den set what divides phase
        grid = (num, den, (ideal + k0 * num) % den, k1 - k0 + 1)
        grids.append(grid if fill or grid[-1] == held[out] else None)
    if fitted.any():  # a stretch's grid states its own (see `_describe_pieces`)
        grids = [x if not y else z for x, y, z in zip(grids, fitted, outputs[col])]
    return outputs.assign(**{col: grids})


def _absolute_bounds(bounds, start, unit, stop):
    """Bounds in a row's units, swapped if reversed (rule 8); open: the edges."""
    out = []
    time, duration = is_datetime64(start), is_timedelta64(start)
    for bound, edge in zip(bounds, (start, stop)):
        if bound is None:
            out.append(edge)
        elif time and isinstance(bound, Quantity | np.timedelta64 | timedelta):
            msg = f"Datetime bounds must be absolute instants, got {bound}; "
            raise ParameterError(msg + "add an offset to a start time instead.")
        elif isinstance(bound, Quantity):
            target = "s" if duration else (unit or None)
            if target is None:
                msg = (
                    f"Cannot use quantity bounds on a unitless coordinate: {bounds!r}."
                )
                raise UnitError(msg)
            try:
                point = convert_units(bound.magnitude, target, bound.units)
            except UnitError as exc:
                msg = f"Explicit range {bounds!r} has bounds in incompatible units."
                raise ParameterError(msg) from exc
            out.append(to_timedelta64(point) if duration else point)
        elif not (time or duration or isinstance(bound, numbers.Real)):
            msg = f"Explicit range {bounds!r} bounds numbers; strings and datetimes"
            raise ParameterError(msg + " bound only datetime coordinates.")
        elif time or duration:
            try:
                out.append((to_datetime64 if time else to_timedelta64)(bound))
            except (TypeError, ValueError) as exc:
                msg = f"Explicit range {bounds!r} has bounds which are not datetimes."
                raise ParameterError(msg) from exc
        else:
            out.append(bound)
    return order_range_tuple(tuple(out))


_TICKS = np.array([-_MAX_NS, _MAX_NS], np.int64)
_OPEN = {  # the ends of an open window, by value family
    "number": (-np.inf, np.inf),
    "datetime": tuple(_TICKS.astype("M8[ns]")),
    "timedelta": tuple(_TICKS.astype("m8[ns]")),
}


def _window_edges(df: pd.DataFrame, dim: str, ranges: ExplicitRanges):
    """Rows holding `dim`, their envelopes, spanning windows, and window bounds."""
    low, high = f"{dim}_min", f"{dim}_max"
    blank = pd.Series(None, index=df.index, dtype=object)
    has = df.get(low, blank).notna().to_numpy()
    families = set(df.get(low, blank)[has].map(_value_family))
    if len(families) > 1 or not families <= set(_OPEN):
        msg = f"Windows on {dim!r} need it to hold numbers, datetimes or timedeltas."
        raise ParameterError(msg)
    family = min(families, default="number")
    edge = _OPEN[family]
    mins, maxs = (
        _coerce(df.get(x, blank).where(has, edge[0]), family).to_numpy()
        for x in (low, high)
    )
    units = [stated_units(x) or "" for x in df.get(f"_{dim}_units", blank)]
    # rows lacking the dim (NaN here) are never read in any unit
    codes, names = pd.factorize(pd.Series(units, dtype=object).where(has))
    rows = [
        [_absolute_bounds(x, edge[0], u, edge[1]) for x in ranges.rows] for u in names
    ]
    ends = np.array(rows or [[edge] * len(ranges.rows)])
    ends = ends.reshape(len(rows or [0]), len(ranges.rows), 2)
    spans = np.array([x == (None, None) for x in ranges.rows], dtype=bool)
    return has, mins, maxs, spans, lambda r, i: ends[codes[i], r].T


def _sample_bounds(coord, lows, highs, unit):
    """The first and last of `coord`'s labels inside each [low, high], and if any."""
    if not len(coord):  # a residual selection left none
        return lows, highs, np.zeros(len(lows), dtype=bool)
    native = stated_units(getattr(coord, "units", None))
    if back := unit and native and native != unit:  # bounds in the labels' units
        lows, highs = (
            np.asarray(convert_units(x, native, unit)) for x in (lows, highs)
        )
    step, runs = getattr(coord, "step", None), getattr(coord, "runs", ())
    if lows.dtype.kind == "f" and step is not None and not pd.isnull(step) and step:
        margin = abs(step) * _GRID_SNAP_RTOL  # labels known to rounding
        lows, highs = lows - margin, highs + margin
    if (
        len(runs) == 1
        and isinstance(runs[0], Grid)
        and coord.sorted
        and len(coord) > 1e4
    ):
        # a long grid's samples by index arithmetic, never expanding its labels
        label = lambda x: runs[0].labels(x, coord.dtype)  # noqa
        nexts = (
            coord.get_next_index(x, allow_out_of_bounds=True) for x in (lows, highs)
        )
        first, last = (np.clip(x, 0, len(coord) - 1) for x in nexts)
        last = np.where(label(last) > highs, last - 1, last)
        held = (last >= first) & (label(first) >= lows)
        ends = [label(np.where(held, x, 0)) for x in (first, last)]
    else:
        values = np.sort(np.asarray(coord.values).ravel())
        first = np.searchsorted(values, lows, side="left")
        last = np.searchsorted(values, highs, side="right") - 1
        held = last >= first
        ends = [values[np.where(held, x, 0)] for x in (first, last)]
    if back:  # each row's labels in its own unit
        ends = [np.asarray(convert_units(x, unit, native)) for x in ends]
    if ends[0].dtype.kind in "Mm":  # the frame's unit
        ends = [x.astype(lows.dtype) for x in ends]
    return ends[0], ends[1], held


def _owned(df, same, name, req, pos, join, spans):
    """Pairs' parts no earlier row in their window holds (rule 6; ``same``: 1 unit)."""
    lo, hi = (df[f"{name}_{x}"].to_numpy()[pos] for x in ("min", "max"))
    keep, sets, unit = np.ones(len(pos), dtype=bool), {}, f"_{name}_units"
    for window, rows in pd.Series(pos).groupby(req).agg(tuple).items():
        if len(rows) > 1:  # windows reaching the same rows share their judgement
            sets.setdefault((rows, spans[window]), []).append(window)
    # rows overlap only where alike on every other dim, as in chunk's cells
    alike = [f"{x}_{y}" for x in join["others"] for y in ("min", "max")]
    group = [*_resolve_group_attrs(None, df.columns), *alike]
    for (rows, spanned), windows in sets.items():  # max_shift if bounding `name`
        shift = None if spanned else join["max_shift"]
        sub = same.iloc[list(rows)].sort_values(f"{name}_min")
        if not (sub[f"{name}_min"] <= sub[f"{name}_max"].cummax().shift()).any():
            continue  # no row overlaps another
        seams = _seams(sub, name, group, None, resolve_max_shift(shift))
        part, codes, starts, *_ = _partition_frames(seams, seams["_stretch"], name)
        _carried_columns(part, codes, starts, name, join["conflict"], starts >= 0)
        seams, at = seams.loc[list(rows)], np.isin(req, windows)
        own = df[unit].iloc[list(rows)] if unit in df else []
        for num, (a, b) in enumerate(zip(own, seams.get(unit, []))):
            if stated_units(a) != stated_units(b):  # back in the row's own unit
                for x in ("_lo", "_hi"):
                    got = convert_units(seams[x].iloc[num], a, b)
                    seams.iloc[num, seams.columns.get_loc(x)] = got
        index = pd.Index(rows).get_indexer(pos[at])
        lo[at], hi[at] = (seams[x].to_numpy()[index] for x in ("_lo", "_hi"))
        keep[at] &= seams["_keep"].to_numpy(dtype=bool)[index]
    return lo, hi, keep


def window_plan(df, windows: Mapping, catalog, sources=None, join=None) -> ChunkPlan:
    """
    Plan each window's piece of each row it reaches, clipped to its samples.

    ``windows`` maps dims to `ExplicitRanges` of one row count, row ``i`` of
    each making window ``i`` (``_request_row`` on the outputs, in window then
    source order); bounds are inclusive, and a window open at both ends spans
    its dim, keeping rows lacking it. The rows' coordinates in `catalog`,
    found by ``sources`` (else the rows), name the samples kept. With
    ``join`` (cut's `max_shift`, `conflict` and other dims), what a window's
    earlier rows hold is theirs (`_owned`). Windows and rows pair in array
    operations, never a window at a time; the first dim is the plan's.
    """
    df = _drop_patch_local_empty(_ensure_patch_row(df)).reset_index(drop=True)
    for x in list(windows) if len(windows) > 1 else ():  # beside another window
        if (
            df.get(f"{x}_min", pd.Series(dtype=float)).notna() & ~_structural(df, x)
        ).any():
            msg = f"{x!r} is non-dimensional on some patches; window it alone."
            raise ParameterError(msg)
    sources = df if sources is None else sources
    # a row's source, to tell which dims cutting at holes narrowed
    own = sources.drop_duplicates("_patch_row").set_index("_patch_row")
    own = own.reindex(df["_patch_row"]).reset_index(drop=True)
    edges = {dim: _window_edges(df, dim, x) for dim, x in windows.items()}
    name, count = next(iter(windows)), len(next(iter(windows.values())).rows)
    block = max(1, 4_000_000 // max(len(df), 1))  # pairs held at once
    found = [(np.empty(0, np.int64),) * 2]
    for begin in range(0, count, block):
        req = np.repeat(np.arange(begin, min(begin + block, count)), len(df))
        pos = np.tile(np.arange(len(df)), len(req) // max(len(df), 1))
        keep = np.ones(len(req), dtype=bool)
        for has, mins, maxs, spans, bounds in edges.values():
            low, high = bounds(req, pos)  # windows meet rows on every dim
            keep &= np.where(
                has[pos], (low <= maxs[pos]) & (high >= mins[pos]), spans[req]
            )
        found.append((req[keep], pos[keep]))
    req, pos = (np.concatenate(x) for x in zip(*found))
    keep, clipped, narrowed = np.ones(len(req), dtype=bool), {}, {name}
    if join is not None:  # a window's pieces in their sources' order, so an
        # overlap's earlier source comes first, and keeps what it holds (rule 6)
        same = _normalize_numeric_units(df, name)  # one unit per dimensionality
        ends = _order_key(pd.concat([same[f"{name}_min"], same[f"{name}_max"]]))
        down = _directions(df[f"{name}_step"]) < 0
        first = np.where(down, -ends[len(df) :], ends[: len(df)])
        order = np.lexsort((df["_patch_row"].to_numpy()[pos], first[pos], req))
        req, pos = req[order], pos[order]
        owned_lo, owned_hi, owned = _owned(
            df, same, name, req, pos, join, edges[name][3]
        )
        keep &= owned
    modified = df.get("_modified", pd.Series(False, index=df.index))
    modified = modified.fillna(False).to_numpy(dtype=bool)[pos]
    for dim, (has, mins, maxs, spans, bounds) in edges.items():
        lo, hi = mins[pos], maxs[pos]
        if dim == name and join is not None:
            lo, hi = owned_lo.astype(lo.dtype), owned_hi.astype(hi.dtype)
        trim = np.flatnonzero(has[pos] & ~spans[req])
        low, high = bounds(req[trim], pos[trim])
        low, high = np.maximum(low, lo[trim]), np.minimum(high, hi[trim])
        held = sources["_patch_row"].isin(df["_patch_row"].iloc[np.unique(pos[trim])])
        coords = known_coordinates(catalog, sources[held], dim)
        units = df.get(f"_{dim}_units", pd.Series(None, index=df.index))
        order = np.argsort(pos[trim], kind="stable")
        for at in np.split(order, np.flatnonzero(np.diff(pos[trim][order])) + 1):
            row = df.iloc[pos[trim][at[0]]] if len(at) else None
            if row is not None and (coord := coords.get(row["_patch_row"])) is None:
                msg = f"No {dim!r} labels in {row.get('source_path')!r}: "
                raise MissingPatchError(msg + "coordinate metadata is unavailable.")
            if row is not None:
                unit = stated_units(units.iloc[pos[trim][at[0]]])
                out = _sample_bounds(coord, low[at], high[at], unit)
                lo[trim[at]], hi[trim[at]], held = out
                keep[trim[at]] &= held
        changed = (lo != mins[pos]) | (hi != maxs[pos])
        modified |= changed
        holes = own.get(f"{dim}_min", df[f"{dim}_min"]) != df[f"{dim}_min"]
        if ((changed | holes.to_numpy()[pos]) & keep & has[pos]).any():
            narrowed.add(dim)  # a trim; a dim left whole keeps its identity
        clipped[dim] = tuple(pd.Series(x).where(has[pos]).to_numpy() for x in (lo, hi))
    picked = {x: (y[keep], z[keep]) for x, (y, z) in clipped.items() if x in narrowed}
    plan = _pieces_plan(df, pos[keep], picked, modified[keep])
    # a piece states its own slice of an exact grid, found over the range of
    # the source a trimmed row was cut from
    spans = {f"{name}_min": f"_{name}_src_low", f"{name}_max": f"_{name}_src_high"}
    whole = df.assign(**{x: df[y].fillna(df[x]) for x, y in spans.items() if y in df})
    outputs = _carry_grids(plan.outputs, plan.members, whole, name)
    return replace(plan, outputs=outputs.assign(_request_row=req[keep]))


def _pieces_plan(df, positions, bounds: Mapping, modified) -> ChunkPlan:
    """
    Plan one output per piece of a row at ``positions`` of `df`; ``bounds``
    maps each dim (the plan's first, then trims) to the pieces' (lows, highs).
    """
    name, *trims = bounds
    ids = np.arange(len(positions), dtype=np.int64)
    # Outputs are not file rows: source bookkeeping stays on the members,
    # and the dimension's structural identity described the whole row.
    outputs = df.iloc[positions].drop(
        columns=["_patch_row", f"_{name}_def_key", "_data_size", *_SOURCE_COLUMNS],
        errors="ignore",
    )
    # a slice of an earlier plan's grid describes that plan's output
    outputs = outputs.drop(columns=[f"_{name}_stretch"], errors="ignore")
    ends = {
        f"{x}_{end}": y
        for x, pair in bounds.items()
        for end, y in zip(*[("min", "max"), pair])
    }
    outputs = outputs.assign(**ends, output_id=ids).reset_index(drop=True)
    members = pd.DataFrame(
        {
            "output_id": ids,
            "_patch_row": df["_patch_row"].to_numpy()[positions],
            **ends,
            f"{name}_step": df[f"{name}_step"].to_numpy()[positions],
            "_modified": np.asarray(modified, dtype=bool),
        }
    )
    if (unit := f"_{name}_units") in df:  # the unit trims are in, as in chunk's
        members[unit] = df[unit].to_numpy()[positions]
    return ChunkPlan(outputs, members, name, None, {}, tuple(trims))
