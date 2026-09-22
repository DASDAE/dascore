"""
Execute spool views: turn member instructions into loaded patches.

This is the consumer of the members (instruction) table that
`dascore.utils.chunk_plan` produces and every spool view carries: it
joins member rows to their source rows, loads each source patch through
a caller-supplied loader, applies exact trims, and merges multi-member
outputs. When the output size is known, a merge whose members the rows
fully describe becomes a
[`LazyArray`](`dascore.core.lazy_array.LazyArray`) recipe -- the window
of a stored array each member is, read into one output -- and any other
such merge streams loaded patches into a pre-allocated buffer; the rest
concatenate. The spool owns *what* rows exist; this module owns *how* a
row becomes a Patch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import _EXACT_GRID_FIELDS, CoordRange, get_coord
from dascore.core.lazy_array import LazyArray
from dascore.core.source import ArraySource
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidFiberIOError,
    ParameterError,
    UnitError,
)
from dascore.io.index.ingest import _is_missing
from dascore.io.index.schema import RESERVED_ATTR_COLUMNS
from dascore.units import get_quantity
from dascore.utils.array_api import to_numpy
from dascore.utils.attrs import combine_patch_attrs, warn_if_histories_differ
from dascore.utils.chunk_plan import _SOURCE_COLUMNS
from dascore.utils.identity import (
    ids_enabled,
    operation_context,
    stamp,
    try_operation_id,
)
from dascore.utils.misc import broadcast_for_index, is_range
from dascore.utils.patch import (
    _force_patch_merge,
    _get_merge_dim,
    _get_merged_coord,
    _split_coord_merge_kwargs,
    drop_associated_coords,
)
from dascore.utils.pd import (
    _convert_min_max_in_kwargs,
    get_dim_names_from_columns,
)

# Slack when counting whole steps between a window edge and a sample, so a
# ratio a float rounding short of a whole number still counts it.
_STEP_SNAP_RTOL = 1e-9


def _get_varying_dim(df) -> str | None:
    """
    Get the single dimension whose range varies across rows of df.

    Returns None when no dimension varies, several do, or the dataframe
    doesn't carry range columns for the varying dimension; those cases
    need the fully materialized merge to sort out.
    """
    dims = get_dim_names_from_columns(df)
    varying = []
    for dim in dims:
        mins, maxs = df.get(f"{dim}_min"), df.get(f"{dim}_max")
        if mins.nunique(dropna=False) > 1 or maxs.nunique(dropna=False) > 1:
            varying.append(dim)
    return varying[0] if len(varying) == 1 else None


def _estimate_merge_samples(df, dim) -> int | None:
    """
    Estimate the total number of samples along dim of the merged rows.

    Returns None if the estimate cannot be made (eg unknown steps), in
    which case streaming the merge isn't possible.
    """
    if dim is None:
        return None
    cols = [f"{dim}_min", f"{dim}_max", f"{dim}_step"]
    if not set(cols).issubset(df.columns):
        return None
    mins, maxs, steps = (df[x] for x in cols)
    if mins.isnull().any() or maxs.isnull().any() or steps.isnull().any():
        return None
    ratios = (maxs - mins) / steps
    # Degenerate steps (eg 0) make the sample counts meaningless.
    if not np.isfinite(ratios.astype(np.float64)).all():
        return None
    counts = np.round(ratios).astype(np.int64) + 1
    if (counts < 0).any():
        return None
    return int(counts.sum())


def _match_merge_units(patch, merge_dim, target_units):
    """
    Convert a member's merge-dim units to the first member's.

    The planner buckets compatible unit spellings by dimensionality and
    normalizes their envelopes to one unit, so one output may mix
    unit spellings of one dimensionality (metres with feet); merging
    requires a single spelling, and the first member's wins. Returns
    (patch, target_units); incompatible or missing units pass through
    for the merge itself to police.
    """
    if merge_dim is None or merge_dim not in getattr(patch.coords, "coord_map", {}):
        return patch, target_units
    units = patch.coords.coord_map[merge_dim].units
    if target_units is None:
        return patch, units
    if units is None or units == target_units:
        return patch, target_units
    try:
        patch = patch.convert_units(**{merge_dim: target_units})
    except UnitError:  # incompatible dimensionality: merge will raise
        return patch, target_units
    return patch, target_units


def _drop_associated_ranges(row, kwargs, plan_dim) -> dict:
    """
    Drop the ranges of coordinates which merely ride a dimension.

    `_convert_min_max_in_kwargs` collapses the row's `<name>_min`/
    `<name>_max` pairs into `<name>: [min, max]` ranges, which travel on
    as read hints. Only the planned dimension's range is a trim; an
    associated (non-dimensional) coordinate's is its whole extent, and
    asking a reader to select on it drops the channels a string
    coordinate labels with neither endpoint, or a numeric one leaves NaN.
    """
    raw_dims = row["dims"]
    assert isinstance(raw_dims, str), "a member row always names its dimensions"
    dims = {x for x in raw_dims.split(",") if x}
    ranged = {x.rsplit("_", 1)[0] for x in row if x.endswith(("_min", "_max"))}
    drop = ranged - dims - {plan_dim}
    return {k: v for k, v in kwargs.items() if k not in drop}


def _plan_trim_kwargs(patch, kwargs, plan_dim) -> dict:
    """
    Keep only the trim the plan actually narrows.

    A member row is its source row with the *planned* dimension's
    envelope replaced by the member's trim range; every other range
    column still describes the whole source. Selecting on those would
    re-select a coordinate to its own extent, which is a no-op for a
    sorted numeric range but not for a string coordinate (a range of
    labels), one holding NaN (missing values fall outside every range),
    or one which cannot be range-selected at all.
    """
    if plan_dim not in kwargs:  # an unmodified member states no range
        return {}
    coord_map = patch.coords.coord_map
    assert plan_dim in coord_map, "the plan's dimension is on every member"
    return {plan_dim: kwargs[plan_dim]}


def _as_plan_units(patch, kwargs, row) -> dict:
    """
    Re-express plan trims in the units each patch coordinate needs.

    A plan hands down magnitudes in the partition's unit (the row's
    ``_{name}_units``, one unit per partition after normalization).
    The loaded member may keep a different compatible spelling — a feet
    patch in a metre-normalized partition — so unit-bearing trims become
    quantities and `Patch.select` converts them to the coordinate's own
    units. Unitless rows pass bare magnitudes, which the coordinate
    reads natively.
    """
    coord_map = patch.coords.coord_map
    out = {}
    for name, value in kwargs.items():
        plan_units = row.get(f"_{name}_units")
        no_units = plan_units is None or pd.isnull(plan_units) or plan_units == ""
        # plan trims arrive as a raw 2-list; only a tuple reads as a range
        ranged = tuple(value) if isinstance(value, list) else value
        numeric = is_range(ranged) and all(
            v is None
            or v is Ellipsis
            or isinstance(v, int | float | np.integer | np.floating)
            for v in ranged
        )
        # only numeric magnitudes carry the plan's unit; time trims are
        # absolute datetimes and pass through untouched
        if name not in coord_map or no_units or not numeric:
            out[name] = value
            continue
        quantity = get_quantity(str(plan_units))
        out[name] = tuple(
            None if v is None or v is Ellipsis else v * quantity for v in ranged
        )
    return out


@dataclass
class _MemberMeta:
    """What an index row states about a member, before its array is read."""

    dims: tuple[str, ...]
    coords: CoordManager
    attrs: dc.PatchAttrs
    # The shape of the whole stored array and the part of it this member
    # is; the coords' shape and all of it when the plan trims nothing.
    extent: tuple[int, ...]
    window: tuple[slice, ...]


@dataclass
class _Member:
    """What the streaming merge takes from one member, however it was loaded."""

    dims: tuple[str, ...]
    data: np.ndarray
    coords: CoordManager
    attrs: dc.PatchAttrs

    def transpose(self, dims: tuple[str, ...]) -> _Member:
        """The same member with its axes in ``dims`` order."""
        order = [self.dims.index(d) for d in dims]
        return _Member(
            dims,
            np.transpose(self.data, order),
            self.coords.transpose(*dims),
            self.attrs,
        )


def _attrs_from_row(
    row: Mapping, dims: tuple[str, ...], coord_names: Iterable[str] | None = None
) -> dc.PatchAttrs:
    """
    The attrs an index row states for its member.

    Every attr the index could hold is a column; what is not a column is
    the plan's own bookkeeping, the coordinate envelopes, and the storage
    provenance (the lineage ids of which are added back below). A field
    the file left unset, or one ingest could not type -- a list, an
    array, a name colliding with a structural column -- is null here and
    takes its default, as it would on the patch `read` builds.
    """
    # every coordinate's envelope, not only every dimension's: a patch
    # carrying latitude(distance) has latitude_min columns which describe
    # that coordinate and are not attrs of its own
    named = dims if coord_names is None else coord_names
    envelope = {f"{d}_{x}" for d in named for x in ("min", "max", "step", "units")}
    # RESERVED_ATTR_COLUMNS names what an index row spends on structure
    # rather than on attrs, the time and distance envelopes among them --
    # those columns exist on every row, whether or not the member has
    # those dimensions.
    skip = envelope | set(_SOURCE_COLUMNS) | set(RESERVED_ATTR_COLUMNS)
    out = {
        k: v
        for k, v in row.items()
        if not str(k).startswith("_") and k not in skip and not _is_null(v)
    }
    for name, value in out.items():
        if isinstance(value, pd.Timestamp):
            out[name] = value.to_datetime64()
        elif isinstance(value, pd.Timedelta):
            out[name] = value.to_timedelta64()
    stored = row.get("_attr_dtypes")
    if isinstance(stored, str) and stored:
        for name, dtype in json.loads(stored).items():
            # Directory overrides are strings, even for numeric source attrs.
            if name in out and isinstance(out[name], int | float | np.number):
                out[name] = np.dtype(dtype).type(out[name])
    out["dims"] = dims
    # the lineage ids are source columns, so the comprehension drops
    # them; a merged patch folds them, and folding nothing is not the
    # same as folding what the members carried. A moved source has its
    # patch id cleared until it is read again.
    for name in ("origin_id", "data_id"):
        if not _is_null(value := row.get(name)):
            out[name] = value
    return dc.PatchAttrs.from_dict(out)


def _is_null(value) -> bool:
    """True for a missing scalar; an array is a value."""
    return np.ndim(value) == 0 and pd.isnull(value)


def _row_range(row: Mapping, dim: str) -> tuple[Any, Any, Any] | None:
    """
    A dimension's evenly sampled range as the row states it, or None.

    The envelope orders values, not samples: a descending coordinate's
    start is its maximum, which the row does not say, so only an
    ascending range is stated.
    """
    values = _row_values(row, dim)
    if values is None or values[2] < np.zeros((), dtype=np.asarray(values[2]).dtype):
        return None
    return values


def _row_values(row: Mapping, dim: str) -> tuple[Any, Any, Any] | None:
    """
    A dimension's (min, max, step) envelope in the file's dtype, or None.

    The frame hands datetimes back as pandas scalars, which `get_coord`
    would keep as an object array; numpy scalars make the coordinate a
    datetime64 one, as the patch path builds it.
    """
    values = []
    for name in ("min", "max", "step"):
        value = row.get(f"{dim}_{name}")
        if value is None or _is_null(value):
            return None
        if isinstance(value, pd.Timestamp):
            value = value.to_datetime64()
        elif isinstance(value, pd.Timedelta):
            value = value.to_timedelta64()
        values.append(value)
    lo, hi, step = values
    if step == np.zeros((), dtype=np.asarray(step).dtype):
        return None  # a zero step is not a range
    stored = row.get(f"_{dim}_coord_dtype")
    if not (isinstance(stored, str) and np.issubdtype(np.dtype(stored), np.number)):
        return lo, hi, step
    # The frame holds every numeric envelope as float, so an integer
    # coordinate must be cast back. That is only right when the values
    # are the file's own: a unit conversion made them float in truth,
    # and past 2**53 a float cannot have held the integer exactly.
    if _units_converted(row, dim):
        return None
    dtype = np.dtype(stored)
    kind = dtype.type
    # SQLite float envelopes cannot retain extended coordinate precision.
    if dtype.kind == "f" and dtype.itemsize > 8:
        return None
    if np.issubdtype(kind, np.integer) and max(abs(lo), abs(hi)) > 2**53:
        return None
    return kind(lo), kind(hi), kind(step)


def _row_bounds(row: Mapping, dim: str) -> tuple[Any, Any] | None:
    """
    A dimension's (min, max) bounds exactly as the row states them, or None.

    Never cast to the coordinate's dtype: these are the window the plan
    drew, which falls where it likes between samples, and rounding one
    onto the stored grid would take in a sample the plan left out. What
    the bounds mean is decided by the coordinate they select.
    """
    values = []
    for name in ("min", "max"):
        value = row.get(f"{dim}_{name}")
        if value is None or _is_null(value):
            return None
        if isinstance(value, pd.Timestamp):
            value = value.to_datetime64()
        elif isinstance(value, pd.Timedelta):
            value = value.to_timedelta64()
        values.append(value)
    return values[0], values[1]


def _units_converted(row: Mapping, dim: str) -> bool:
    """Whether the row's envelope was converted from the file's own units."""
    source_units = row.get(f"_{dim}_units_source")
    return not _is_null(source_units) and source_units != row.get(f"_{dim}_units")


# What a source's own range is called beside a member's trim of it. Not
# min/max/step: code scanning a frame for `<x>_min`/`<x>_max`/`<x>_step`
# triples would read these as a dimension named `<dim>_src`.
SOURCE_RANGE_ENDS = ("low", "high", "step")


def source_range_column(dim: str, end: str) -> str:
    """The private column holding a member source's own envelope value."""
    return f"_{dim}_src_{end}"


def source_coord_from_row(row: Mapping, dim: str, units=None):
    """
    The evenly sampled coordinate a member's *source* states, or None.

    A member row's envelope is the window the plan trimmed it to; the
    source's own stays beside it under `source_range_column`, and with
    the grid, dtype and units — which describe the file either way —
    rebuilds the coordinate the file holds.
    """
    envelope = {}
    for end, name in zip(SOURCE_RANGE_ENDS, ("min", "max", "step"), strict=True):
        value = row.get(source_range_column(dim, end))
        if value is None or _is_null(value):
            return None
        envelope[f"{dim}_{name}"] = value
    for suffix in ("_grid", "_coord_dtype", "_units", "_units_source"):
        envelope[f"_{dim}{suffix}"] = row.get(f"_{dim}{suffix}")
    return coord_from_row(envelope, dim, units=units)


def coord_from_row(row: Mapping, dim: str, units=None):
    """
    The evenly sampled coordinate a row states for ``dim``, or None.

    The exact grid a row carries rebuilds the coordinate the file holds,
    either way it runs; the whole-tick envelope only approximates a
    fractional step and states no direction. The grid counts ticks in the
    file's units and dtype, so a unit-converted row, or one stating only
    the float dtype the frame holds every envelope as, is rebuilt from
    its envelope alone.
    """
    values = _row_values(row, dim)
    if values is None:
        return None
    lo, hi, step = values
    grid = row.get(f"_{dim}_grid")
    ticks = np.asarray(lo).dtype.kind in "iuMm"
    if isinstance(grid, tuple) and ticks and not _units_converted(row, dim):
        *terms, length = grid
        start = hi if terms[0] < 0 else lo
        return CoordRange(
            start=start,
            shape=(length,),
            units=units,
            **dict(zip(_EXACT_GRID_FIELDS, terms)),
        )
    if step < np.zeros((), dtype=np.asarray(step).dtype):
        return None
    return get_coord(start=lo, stop=hi + step, step=step, units=units)


def _at_unit(coord, unit: str):
    """The same evenly sampled coordinate counted in ``unit``, or None."""
    name = "datetime64" if np.asarray(coord.start).dtype.kind == "M" else "timedelta64"
    start = np.asarray(coord.start).astype(f"{name}[{unit}]")
    step = np.asarray(coord.step).astype(f"timedelta64[{unit}]")
    if step == np.zeros((), dtype=step.dtype):
        return None  # a unit this coarse cannot count this step
    return get_coord(start=start, step=step, shape=coord.shape, units=coord.units)


def coord_at_stored_unit(coord, row: Mapping, dim: str):
    """
    ``coord`` counted as the file counts it, or None when nothing says.

    A row holds datetimes in nanoseconds however its file stores them,
    and the dtype it records names the unit they are counted in, so the
    coordinate is cast to it. A row whose dtype names no unit is left to
    the patch path, which reads the file's own.
    """
    if coord is None or np.asarray(coord.min()).dtype.kind not in "mM":
        return coord
    stored = row.get(f"_{dim}_coord_dtype")
    if not isinstance(stored, str) or not stored:
        return None
    if np.dtype(stored).kind not in "mM":
        return None
    unit, count = np.datetime_data(stored)
    if unit == "generic" or count != 1:
        return None
    return coord if unit == "ns" else _at_unit(coord, unit)


def patch_from_fill(
    coords: CoordManager, row: Mapping, plan_dim: str, fill_value
) -> dc.Patch:
    """
    The all-fill patch an output row with no members describes.

    A window lying wholly inside a bridged hole has no source to read
    from, so its samples are all the fill value. It must still look like
    the outputs beside it, though: `coords` is a sibling's coordinates,
    which is what states the values, dtypes and units of every dimension
    this plan did not chunk. Only the chunked dimension comes from the
    row, which is the window the plan advertised.
    """
    # the cast check lives with fill_gaps, which proc imports from utils
    from dascore.proc.coords import _fill_scalar  # noqa: PLC0415

    sibling = coords.coord_map[plan_dim]
    bounds = _row_values(row, plan_dim)
    step = sibling.step
    if bounds is None or _is_null(step):
        msg = (
            f"Cannot fill a hole along {plan_dim!r}: a window no source feeds "
            "takes the span its plan row states and the step its neighbours "
            "are sampled at, and one of those is missing. Chunk without a "
            "pending sample or relative selection, or without fill_value."
        )
        raise ChunkError(msg)
    # The row's envelope orders values and so states no direction; the
    # step beside it does, which is why the window is rebuilt from both.
    low, high, _ = bounds
    length = int(np.round(abs(high - low) / abs(step))) + 1
    descending = step < step * 0
    coord = get_coord(
        start=high if descending else low,
        step=step,
        shape=(length,),
        units=sibling.units,
    )
    # A coordinate riding the chunked dimension describes samples which
    # are not there, so it cannot come along; one riding any other
    # dimension is the same for every patch here and does.
    coords = drop_associated_coords(coords, plan_dim, "Filling the gaps along")
    coords = coords._update_grid(plan_dim, **{plan_dim: coord})
    dtype = np.dtype(row.get("_dtype") or np.float64)
    data = np.full(coords.shape, _fill_scalar(fill_value, dtype), dtype=dtype)
    attrs = _attrs_from_row(row, coords.dims, coord_names=set(coords.coord_map))
    return dc.Patch(data=data, coords=coords, dims=coords.dims, attrs=attrs)


def _whole_steps(ratio) -> int:
    """How many whole steps fit in a ratio, forgiving float error at the edge."""
    return int(np.floor(float(ratio) + _STEP_SNAP_RTOL))


def _fill_limit(tolerance) -> tuple:
    """
    The widest hole a fill may close, as `Patch.fill_gaps` states limits.

    A tolerance admits a spacing of so many steps; a hole of that spacing
    is missing one fewer sample than it spans. An infinite count sets no
    limit, which is the one way to ask for every hole to be filled.
    """
    if (count := tolerance.count) is not None:
        if not np.isfinite(count):
            return None, False
        return max(int(np.floor(count)) - 1, 0), True
    # an absolute excess is already stated per hole, not per spacing
    return tolerance, False


def fill_to_row(
    patch: dc.Patch, dim: str, row: Mapping, fill_value, tolerance
) -> dc.Patch:
    """
    Write the fill value into every sample an assembled output is missing.

    Two kinds are missing, and only the whole output knows both. Inside
    it are the holes a bridged gap left, whether the members were merged
    across one or a single member carried one of its own. Outside them
    are the edge positions a window laid over a hole advertises but no
    source feeds. A row stating no evenly sampled window (a descending
    coordinate, whose direction an envelope cannot express) keeps the
    interior fill and is not padded.
    """
    # the placement and the cast check live with fill_gaps, which proc
    # imports from utils
    from dascore.proc.coords import _fill_scalar, _place_blocks  # noqa: PLC0415

    # Bounded by the same tolerance which decided the merge: planning
    # normally splits a patch at its holes, but a pending sample or
    # relative selection keeps it from describing them, and the whole
    # patch arrives here with its holes intact.
    limit, samples = _fill_limit(tolerance)
    bound = {dim: limit} if limit is not None else {}
    args = () if bound else (dim,)
    patch = patch.fill_gaps(*args, value=fill_value, samples=samples, **bound)
    coord = patch.get_coord(dim)
    bounds = _row_values(row, dim)
    step = coord.step
    # A row whose envelope the plan could not state -- a pending relative
    # selection resolves against the patch, not the plan -- says nothing
    # about where the window ends, so there is no span to pad out to.
    if bounds is None or _is_null(step):
        return patch
    # Counted on the samples' own grid and anchored on their own first
    # label, so padding can only ever add positions around them: a target
    # built from the window's edges instead would move every label
    # whenever those edges did not land on the samples' grid.
    low, high, _ = bounds
    size = abs(step)
    below = _whole_steps((coord.min() - low) / size)
    above = _whole_steps((high - coord.max()) / size)
    # the envelope orders values; the array may run the other way
    before, after = (above, below) if step < step * 0 else (below, above)
    if before <= 0 and after <= 0:
        return patch
    before, after = max(before, 0), max(after, 0)
    target = get_coord(
        start=coord.values[0] - before * step,
        step=step,
        shape=(before + len(coord) + after,),
        units=coord.units,
    )
    data = to_numpy(patch.data)
    axis = patch.get_axis(dim)
    fill = _fill_scalar(fill_value, data.dtype)
    blocks = ((0, data.shape[axis], before),)
    data = _place_blocks(data, axis, len(target), blocks, fill)
    coords = drop_associated_coords(patch.coords, dim, "Filling the gaps along")
    # What the padding is and where it went, so the same chunk loaded twice
    # is the same array rather than two random ids.
    params = {"dim": dim, "value": fill_value, "before": before, "after": after}
    attrs = stamp(patch.attrs, [patch.attrs], try_operation_id("Fill", params))
    with operation_context():
        return patch.new(
            data=data, coords=coords._update_grid(dim, **{dim: target}), attrs=attrs
        )


@dataclass
class PatchAssembler:
    """
    Assemble output patches from joined member rows.

    ``load_patch`` resolves one member row to its source patch (residual
    selections included); ``merge_kwargs`` carries the merge behavior;
    ``plan_dim`` names the one dimension whose range the plan narrowed,
    and so the only one a member needs trimming on. The plan resolver
    hands this the joined member frame for one output at a time.
    """

    load_patch: Callable[[Mapping], dc.Patch]
    merge_kwargs: Mapping
    plan_dim: str
    # Names the whole stored array a member is part of, reading nothing;
    # the index then stands in for its coordinates and attrs. None sends
    # the member down the patch path.
    array_source: Callable[[Mapping, tuple[int, ...]], ArraySource | None] | None = None
    # Whether the rows alone can describe this member; nothing is read.
    can_use_index: Callable[[Mapping], bool] | None = None
    # Whether every member source still is what the index recorded. A
    # recipe reads the window the index promised rather than the whole
    # array, so a file rewritten since would otherwise pass unnoticed.
    # Takes the joined member frame; the caller keeps what was recorded.
    sources_unchanged: Callable[[pd.DataFrame], bool] | None = None

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        df_dict_list = self._df_to_dict_list(joined)
        expected_len = len(joined["current_index"].unique())
        merging = len(df_dict_list) > expected_len
        merge_dim = _get_varying_dim(joined) if merging else None
        if merging:
            # Several sources merge into one patch. When the output size can
            # be determined from the instructions, stream the sources into a
            # pre-allocated array so they don't all need to be in memory with
            # the merged output at once.
            samples = _estimate_merge_samples(joined, merge_dim)
            if samples is not None:
                patch = self._merge_patches_streaming(
                    joined, df_dict_list, merge_dim, samples
                )
                return [patch]
        out = []
        target_units = None
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined)
            patch, target_units = _match_merge_units(patch, merge_dim, target_units)
            # The index doesn't carry all the dimensional info, so get what
            # merging needs from the patch coords (cheaper than attr dumps).
            info = patch.coords._get_dim_summary()
            info["patch"] = patch
            out.append(info)
        if len(out) > expected_len:
            out = _force_patch_merge(out, merge_kwargs=self.merge_kwargs)
        return [x["patch"] for x in out]

    def _load_trimmed_patch(self, patch_kwargs, joined) -> dc.Patch:
        """Load a single patch and trim it to its instruction range."""
        # convert kwargs to format understood by parser/patch.select
        kwargs = _convert_min_max_in_kwargs(patch_kwargs, joined)
        kwargs = _drop_associated_ranges(patch_kwargs, kwargs, self.plan_dim)
        patch = self.load_patch(kwargs)
        # If the limits of the source patch were not modified, we can just
        # skip selection. This is important for missing coordinates
        # (NaN values) to not get trimmed out.
        source_kwargs = kwargs if kwargs.get("_modified") else {}
        # attr-style entries filter rows above, and the plan only ever
        # narrows its own dimension; everything else loads untouched.
        if select_kwargs := _plan_trim_kwargs(patch, source_kwargs, self.plan_dim):
            patch = patch.select(**_as_plan_units(patch, select_kwargs, patch_kwargs))
        return patch

    def _merge_patches_streaming(self, joined, df_dict_list, merge_dim, samples):
        """
        Merge the patches described by the instructions along merge_dim.

        All members come from the index or none do: a loaded patch can
        carry what the index cannot hold (an array attr, a coordinate it
        could not represent), and the merge would then see it on some
        members and not others, refusing what it accepts whole. The rows
        decide that before anything is read, and a source the index no
        longer matches is refused there too; a shape the row did not
        predict is only found once it is loaded, and abandons the
        attempt. Returning from that attempt releases its output before
        the retry reloads the source coordinates, attrs, and arrays.
        """
        metas = self._member_meta_from_index(df_dict_list)
        if metas is not None:
            out = self._merge_from_index(joined, df_dict_list, merge_dim, metas)
            if out is not None:
                return out
        return self._stream(joined, df_dict_list, merge_dim, samples)

    def _merge_from_index(self, joined, df_dict_list, merge_dim, metas):
        """
        Merge members the rows describe, as one recipe read in one pass.

        Every member is a window of its source's stored array, so the
        output's shape and dtype are known before a file is opened and
        the members need never be patches. Returns None when a member
        cannot be named without reading it, when a source is no longer
        what the index recorded, or when a file did not deliver the array
        its row predicted, so the caller can start over on the patch path.
        """
        dims = metas[0].dims
        axis = dims.index(merge_dim)
        recipe = self._recipe(df_dict_list, metas, dims, axis)
        if recipe is None:
            return None
        if self.sources_unchanged is not None and not self.sources_unchanged(joined):
            return None
        try:
            data = recipe.load()
        except (InvalidFiberIOError, ParameterError):
            # the resource did not hold the array its row described: a
            # different shape, or fewer axes than the windows name
            return None
        coords = [meta.coords for meta in metas]
        attrs = [meta.attrs for meta in metas]
        summaries = [x._get_dim_summary() for x in coords]
        return self._assemble(data, dims, merge_dim, coords, attrs, summaries)

    def _recipe(self, rows, metas, dims, axis) -> LazyArray | None:
        """
        The lazy array the members make, or None when one cannot be named.

        Nothing is read here: every member is a window of a stored array
        whose shape its row states, so the members are laid end to end
        along the merged axis by arithmetic alone. A member holding its
        axes in another order than the first, or disagreeing off the
        merged axis, takes the patch path instead, which transposes each
        one as it is loaded and says what cannot be merged.

        The dtype is promoted a member at a time, in placement order, as
        the streaming merge promotes its buffer: promotion is not
        associative, and the two routes must give one answer.
        """
        assert self.array_source is not None, "the caller checks for a source"
        sources, rest, dtype = [], None, None
        for row, meta in zip(rows, metas, strict=True):
            if meta.dims != dims:
                return None
            source = self.array_source(row, meta.extent)
            if source is None:
                return None
            source = source[meta.window]
            placed = meta.coords.shape
            assert source.shape == placed, "the window is the member's own samples"
            if source.shape != source.extent:
                # a trim is a window of the array its row names, and holds
                # that window's id rather than the whole array's
                meta.attrs = meta.attrs.update(data_id=source.data_id)
            others = placed[:axis] + placed[axis + 1 :]
            if rest is None:
                rest = others
            elif others != rest:
                return None
            sources.append(source)
            dtype = (
                source.dtype if dtype is None else np.result_type(dtype, source.dtype)
            )
        return LazyArray.from_sources(sources, axis=axis, dtype=dtype)

    def _stream(self, joined, df_dict_list, merge_dim, samples):
        """
        Copy each member into the output buffer as it is loaded.

        A member is released once copied; this avoids holding all source
        patches and the merged output in memory at the same time, as
        concatenating would.
        """
        buffer, offset, axis, dims = None, 0, None, None
        coords, attrs, summaries = [], [], []
        target_units = None
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined)
            patch, target_units = _match_merge_units(patch, merge_dim, target_units)
            member = _Member(patch.dims, patch.data, patch.coords, patch.attrs)
            if dims is None:
                dims = member.dims
                axis = dims.index(merge_dim)
            elif member.dims != dims:
                member = member.transpose(dims)
            assert axis is not None  # set on the first pass through the loop
            data = member.data
            if buffer is None:
                shape = list(data.shape)
                shape[axis] = samples
                buffer = np.empty(shape, dtype=data.dtype)
            # Mixed dtypes upcast, a member at a time and so in order.
            dtype = np.result_type(buffer.dtype, data.dtype)
            if dtype != buffer.dtype:
                buffer = buffer.astype(dtype)
            end = offset + data.shape[axis]
            if end > buffer.shape[axis]:
                # The estimate came up short (eg from slightly uneven
                # sampling); grow the buffer to fit.
                shape = list(buffer.shape)
                shape[axis] = end
                new_buffer = np.empty(shape, dtype=buffer.dtype)
                head = broadcast_for_index(buffer.ndim, axis, slice(0, offset))
                new_buffer[head] = buffer[head]
                buffer = new_buffer
            try:
                index = broadcast_for_index(buffer.ndim, axis, slice(offset, end))
                buffer[index] = data
            except ValueError as e:
                msg = (
                    f"Cannot merge patches; their shapes are incompatible "
                    f"along the dimensions not being merged ({merge_dim})."
                )
                raise CoordMergeError(msg) from e
            offset = end
            coords.append(member.coords)
            attrs.append(member.attrs)
            summaries.append(member.coords._get_dim_summary())
        # All set on the first pass of the loop, which always runs.
        assert buffer is not None
        assert axis is not None
        assert dims is not None
        if offset != buffer.shape[axis]:  # over-estimated; trim excess.
            buffer = buffer[broadcast_for_index(buffer.ndim, axis, slice(0, offset))]
        return self._assemble(buffer, dims, merge_dim, coords, attrs, summaries)

    def _assemble(self, data, dims, merge_dim, coords, attrs, summaries):
        """Build the merged patch from the members' data, coords and attrs."""
        # Ensure the loaded patches only vary along the expected dimension,
        # the same requirement _force_patch_merge enforces.
        summary_df = pd.DataFrame(summaries)
        found_dim = _get_merge_dim(summary_df)
        if found_dim != merge_dim:
            msg = (
                f"Cannot merge patches; expected them to vary along "
                f"{merge_dim} but found {found_dim}."
            )
            raise CoordMergeError(msg)
        attr_kwargs, coord_kwargs = _split_coord_merge_kwargs(self.merge_kwargs)
        conf = attr_kwargs.get("conflict", None)
        drop_conflicting = conf in {"drop", "keep_first"}
        new_coord = _get_merged_coord(
            summary_df, merge_dim, coords, drop_conflicting, **coord_kwargs
        )
        warn_if_histories_differ(attrs, "Merging")
        new_attrs = combine_patch_attrs(
            attrs, **attr_kwargs, merge_params=self.merge_kwargs
        )
        # The fold named the result; building it is not another array.
        with operation_context():
            return dc.Patch(
                data=data, coords=new_coord, attrs=new_attrs, dims=list(dims)
            )

    def _member_meta_from_index(self, rows) -> list[_MemberMeta] | None:
        """What the rows state about every member, or None if any is silent.

        Metadata only: nothing is read here, so a merge the index cannot
        describe costs no array reads before it falls back.
        """
        if self.array_source is None:
            return None
        if self.can_use_index is not None and not all(
            self.can_use_index(row) for row in rows
        ):
            return None
        metas = []
        for row in rows:
            meta = self._meta_from_index(row)
            if meta is None:
                return None
            metas.append(meta)
        return metas

    def _meta_from_index(self, row: Mapping) -> _MemberMeta | None:
        """
        What an index row states about one member, without reading it.

        The row states each dimension's evenly sampled range in the
        plan's units and every attr the file defined, which is what the
        merge needs from a member; the patch, its coordinate parsing and
        its attr decoding are skipped. Anything the row cannot state --
        a dimension without a range, a trim whose source grid is not
        beside it -- sends the whole merge down the patch path instead.

        The index holds no history, a list rather than a column, so a
        member built here states none and the merged patch carries none.
        Only a format which stores a history to begin with (DASDAE) has
        one to lose, and only until it is read as a patch again.
        """
        # A moved source has its id cleared until it is read again, and
        # folding no id is not folding the one the patch carries; an attr
        # the index could not hold is on the patch and would be lost here.
        # A column no row carries states no id either: the patch path
        # would give the loaded patches' own, which the rows cannot.
        if ids_enabled() and _is_missing(row.get("origin_id")):
            return None
        if not _is_null(complete := row.get("_attrs_complete")) and not complete:
            return None
        dims = tuple(str(row["dims"]).split(","))
        trimmed = bool(row.get("_modified"))
        coord_map, extent, window = {}, [], []
        for dim in dims:
            # a coordinate with no units is NaN in a frame, not None,
            # and NaN would build a dimensionless quantity the patch
            # path does not have.
            units = None if _is_null(u := row.get(f"_{dim}_units")) else u
            if trimmed and dim == self.plan_dim:
                # a window's id builds on the whole array's, so a row
                # which does not name the array cannot name the window
                if ids_enabled() and _is_missing(row.get("data_id")):
                    return None
                placed = self._trim_window(row, dim, units)
                if placed is None:
                    return None
                coord, span, length = placed
            else:
                # The plan narrows its own dimension and no other, so
                # every envelope here is its source's own.
                coord = coord_at_stored_unit(
                    coord_from_row(row, dim, units=units), row, dim
                )
                if coord is None:
                    return None
                span, length = slice(0, len(coord)), len(coord)
            coord_map[dim] = coord
            extent.append(length)
            window.append(span)
        coords = get_coord_manager(coord_map, dims=dims)
        return _MemberMeta(
            dims, coords, _attrs_from_row(row, dims), tuple(extent), tuple(window)
        )

    def _trim_window(self, row: Mapping, dim: str, units):
        """
        The source samples a trimmed member is, or None if it cannot say.

        The window comes from selecting the source's own coordinate to
        the member's range, so the samples a recipe reads are the ones
        `Patch.select` would have kept, and the coordinate it presents is
        that selection's own. An evenly sampled coordinate can only
        answer with a contiguous forward run, so the one refusal here is
        a trim naming no sample of the source.
        """
        # A converted envelope is not on the grid the source's own units
        # count, so the two cannot be compared sample for sample.
        if _units_converted(row, dim):
            return None
        source = coord_at_stored_unit(
            source_coord_from_row(row, dim, units=units), row, dim
        )
        bounds = _row_bounds(row, dim)
        if source is None or bounds is None:
            return None
        coord, indexer = source.select(bounds)
        # an evenly sampled coordinate answers a range with a forward run
        assert isinstance(indexer, slice), indexer
        start, stop, step = indexer.indices(len(source))
        assert step == 1 and stop - start == len(coord), indexer
        if start == stop:  # the trim names no sample of this source
            return None
        return coord, slice(start, stop), len(source)

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        This is significantly faster than iterating rows. Empty strings
        (missing format fields on file rows) normalize to None; stored
        relative paths pass through unchanged — the catalog's resolver
        owns resolving them against the spool root.
        """
        return df.replace("", None).to_dict("records")
