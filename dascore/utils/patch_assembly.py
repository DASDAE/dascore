"""
Execute spool views: turn member instructions into loaded patches.

This is the consumer of the members (instruction) table that
`dascore.utils.chunk_plan` produces and every spool view carries: it
joins member rows to their source rows, loads each source patch through
a caller-supplied loader, applies exact trims, and merges multi-member
outputs. When the output size is known, a merge whose members the rows
fully describe becomes a
[`Recipe`](`dascore.lazy.recipe.Recipe`) -- the window
of a stored array each member is, read into one output -- and any other
such merge streams loaded patches into a pre-allocated buffer; the rest
concatenate. The spool owns *what* rows exist; this module owns *how* a
row becomes a Patch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Any

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import _grid_coord, get_coord
from dascore.core.source import ArraySource
from dascore.exceptions import (
    CoordMergeError,
    InvalidFiberIOError,
    ParameterError,
    UnitError,
)
from dascore.io.index.schema import RESERVED_ATTR_COLUMNS
from dascore.lazy.recipe import Recipe
from dascore.units import conversion_factors, get_quantity, units_match
from dascore.utils.array_api import to_numpy
from dascore.utils.attrs import (
    _is_missing,
    combine_patch_attrs,
    warn_if_histories_differ,
)
from dascore.utils.chunk import _READ_PAD
from dascore.utils.chunk_plan import _SOURCE_COLUMNS, stretch_coordinate
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
)
from dascore.utils.pd import (
    _convert_min_max_in_kwargs,
    get_dim_names_from_columns,
)
from dascore.utils.time import to_float


def _get_varying_dim(df) -> str | None:
    """
    Get the single dimension whose range varies across rows of df.

    Returns None when no dimension varies, several do, or the dataframe
    doesn't carry range columns for the varying dimension; those cases
    need the fully materialized merge to sort out.
    """
    dims = get_dim_names_from_columns(df)
    if "dims" in df:
        dims = set(dims) & set(df["dims"].iloc[0].split(","))
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


def plan_data_units(units: Iterable):
    """The data units members merge into: the first stated, in plan order."""
    stated = [x for x in units if not _is_missing(x)]
    return get_quantity(stated[0]) if stated else None


def _converted_dtype(dtype, units, target) -> np.dtype:
    """The dtype `convert_units` gives ``dtype`` data going ``units`` to ``target``."""
    factors = conversion_factors(units, target)
    if factors is None:
        return np.dtype(dtype)
    plan = dict(zip(("mult1", "add", "mult2"), factors, strict=True))
    return dc.proc.units.ConvertUnits().kernel(np.empty(0, dtype), **plan).dtype


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


def _plan_trim_kwargs(patch, kwargs, plan_dim, trim_dims=()) -> dict:
    """
    Keep only the trims the plan actually narrows.

    A member row is its source row with the *planned* dimensions'
    envelopes replaced by the member's trim ranges; every other range
    column still describes the whole source. Selecting on those would
    re-select a coordinate to its own extent, which is a no-op for a
    sorted numeric range but not for a string coordinate (a range of
    labels), one holding NaN (missing values fall outside every range),
    or one which cannot be range-selected at all. A trim spanning the patch
    (or on a dim it lacks) is dropped too, so the patch keeps its id; only
    an open select window admits a patch lacking the plan's own dim.
    """
    coord_map = patch.coords.coord_map
    # an unmodified member states no range
    has_dim = plan_dim in kwargs and plan_dim in coord_map
    out = {plan_dim: kwargs[plan_dim]} if has_dim else {}
    for dim in trim_dims:
        (low, high), coord = kwargs.get(dim, (None, None)), coord_map.get(dim)
        if coord is not None and low is not None:
            if low > coord.min() or high < coord.max():
                out[dim] = kwargs[dim]
    return out


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
    # Empty when the row only sizes the member, which then loads as a patch.
    extent: tuple[int, ...]
    window: tuple[slice, ...]
    # The stored window this member reads as, if any (see `describe_output`).
    source: ArraySource | None = None
    # Where the merge's dtype promotion first rounds this member; `_cast_via`.
    cast_via: np.dtype | None = None


@dataclass
class OutputMeta:
    """What one plan output is, stated by its member rows alone."""

    dims: tuple[str, ...]
    coords: CoordManager
    attrs: dc.PatchAttrs
    dtype: np.dtype
    members: list[_MemberMeta]


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
    out = {name: _as_numpy(value) for name, value in out.items()}
    stored = row.get("_attr_dtypes")
    if isinstance(stored, str) and stored:
        for name, dtype in json.loads(stored).items():
            # Directory overrides are strings, even for numeric source attrs.
            if name in out and isinstance(out[name], int | float | np.number):
                out[name] = np.dtype(dtype).type(out[name])
    out = _row_defaults(row) | out
    out["dims"] = dims
    # the lineage ids are source columns, so the comprehension drops
    # them; a merged patch folds them, and folding nothing is not the
    # same as folding what the members carried. A moved source has its
    # patch id cleared until it is read again.
    for name in ("origin_id", "data_id"):
        if not _is_null(value := row.get(name)):
            out[name] = value
    return dc.PatchAttrs.from_dict(out)


def _row_defaults(row: Mapping) -> dict[str, Any]:
    """The declared defaults of the vendor attrs model a row was read with."""
    stored = row.get("_attr_defaults")
    return json.loads(stored) if isinstance(stored, str) and stored else {}


def _is_null(value) -> bool:
    """True for a missing scalar; an array is a value."""
    return np.ndim(value) == 0 and pd.isnull(value)


def _as_numpy(value) -> Any:
    """A frame's pandas time scalar as numpy's; anything else as it is."""
    if isinstance(value, pd.Timestamp):
        return value.to_datetime64()
    if isinstance(value, pd.Timedelta):
        return value.to_timedelta64()
    return value


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
        values.append(_as_numpy(value))
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
    A dimension's (min, max) bounds as a read takes them, or None.

    Never cast to the coordinate's dtype: these are the window the plan
    drew, which falls where it likes between samples, and rounding one
    onto the stored grid would take in a sample the plan left out. What
    the bounds mean is decided by the coordinate they select. A float
    bound is known only to rounding, so it is widened by a tenth of a step.
    """
    values = []
    for name in ("min", "max"):
        value = row.get(f"{dim}_{name}")
        if value is None or _is_null(value):
            return None
        values.append(_as_numpy(value))
    step = row.get(f"{dim}_step", np.nan)
    if np.asarray(values[0]).dtype.kind == "f" and not _is_null(step):
        pad = abs(step) * _READ_PAD
        return values[0] - pad, values[1] + pad
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
        return _grid_coord(terms, units, start=start, shape=(length,))
    if step < np.zeros((), dtype=np.asarray(step).dtype):
        return None
    return get_coord(start=lo, stop=hi + step, step=step, units=units)


def _sized_coord(row: Mapping, dim: str, units=None):
    """A coordinate sizing the row's envelope, even one `coord_from_row` refuses."""
    if (coord := coord_from_row(row, dim, units=units)) is not None:
        return coord
    low, high, step = (_as_numpy(row.get(f"{dim}_{x}")) for x in ("min", "max", "step"))
    if _is_null(step) or to_float(step) == 0:
        return get_coord(data=[low], units=units) if low == high else None
    start, stop = (high, low + step) if to_float(step) < 0 else (low, high + step)
    return get_coord(start=start, stop=stop, step=step, units=units)


def _cut_coord(row: Mapping, dim: str, units=None):
    """
    The samples of a cut member's source inside the cut, or None.

    As loading selects them: the source's own grid, in its own units,
    selected to the cut and then converted to ``units``. A coordinate
    counted from the cut's bound instead would put every label off the
    grid by however far the bound fell between two samples.
    """
    stored = row.get(f"_{dim}_units_source")
    stored = units if _is_null(stored) else stored
    # the source's range is in its own units, as its row states them
    source = source_coord_from_row({**row, f"_{dim}_units": stored}, dim, stored)
    if source is None or (bounds := _row_bounds(row, dim)) is None:
        return None
    if units is not None and stored != units:
        bounds = tuple(x * get_quantity(units) for x in bounds)
    coord = source.select(bounds)[0]
    return coord if units is None else coord.convert_units(units)


def _at_unit(coord, unit: str):
    """The same evenly sampled coordinate counted in ``unit``, or None."""
    first = coord.max() if coord.reverse_sorted else coord.min()
    name = "datetime64" if np.asarray(first).dtype.kind == "M" else "timedelta64"
    start = np.asarray(first).astype(f"{name}[{unit}]")
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


def on_stretch_grid(coords: CoordManager, dim: str, row: Mapping) -> CoordManager:
    """
    The coordinates with ``dim`` labelled by the stretch grid ``row`` names.

    A chunk plan fits one grid to each stretch of joined patches and gives
    each output's members its slice (``_{dim}_stretch``); an output whose
    stretch keeps its one patch's own labels names none and is returned
    as it is. The grid is in the plan's dtype; a time counted in another
    unit keeps it where every label is whole in that unit, and otherwise
    takes nanoseconds, as the plan does. A coordinate which did not load
    the samples the plan numbered keeps the labels it was assembled with.
    """
    spec = row.get(f"_{dim}_stretch")
    if dim not in coords.coord_map:
        return coords
    current = coords.coord_map[dim]
    if spec == "labels" and len(getattr(current, "runs", ())) > 1:  # one list
        labels = get_coord(data=current.values, units=current.units)
        return coords._update_grid(dim, **{dim: labels})
    if not isinstance(spec, tuple):
        return coords
    dtype, (origin, num, den, *rest, stated) = np.dtype(current.dtype), spec
    spec = (origin, num, den, *rest, str(dtype))
    if dtype.kind in "Mm" and dtype != stated:  # the plan's ticks are nanoseconds
        per = int(np.array(1, dtype).astype(stated).view(np.int64))
        start, step = np.int64(origin).view(stated), np.int64(num).view("m8[ns]")
        steps = np.dtype(dtype.str.replace("M", "m"))  # a timedelta in the unit
        own = (start.astype(dtype), step.astype(steps), 0, *rest, str(dtype))
        whole = den == 1 and not origin % per and not num % per
        spec = own if whole else (origin, num, den, *rest, stated)
    coord = stretch_coordinate(spec, current.units)
    if len(coord) != len(current):
        # a source changed since it was planned, so its samples are not
        # the ones the plan numbered; they keep the labels they came with
        return coords
    return coords._update_grid(dim, **{dim: coord})


def _fuses(rows: Sequence[Mapping], dim: str) -> bool:
    """
    Whether a merge may fuse its members' labels: as the plan judged them,
    every member states one step and the plan keeps no stretch's labels.
    """
    keeps = (x.get(f"_{dim}_stretch") == "labels" for x in rows)
    steps = {x.get(f"{dim}_step") for x in rows}
    return not any(keeps) and len(steps) == 1 and not pd.isnull(steps.pop())


def _on_stretch_grid(patch: dc.Patch, dim: str, row: Mapping) -> dc.Patch:
    """The patch with its ``dim`` labels on its stretch's grid (see above)."""
    coords = on_stretch_grid(patch.coords, dim, row)
    if coords is patch.coords:
        return patch
    # relabelling is part of the merge the plan named, not another operation
    with operation_context():
        return dc.Patch(
            data=patch.data, coords=coords, attrs=patch.attrs, dims=patch.dims
        )


def fill_on_grid(patch, dim: str, grid, merge, dtype=None):
    """
    Fill a fill plan's assembled members, as `Patch.fill_gaps` does, onto the
    nearest positions of ``grid``; the fill, in ``dtype``, pads the rest.
    """
    fill_value = merge["fill_value"]
    # proc imports from utils, so this is imported here
    from dascore.proc.coords import (  # noqa: PLC0415
        _fill_riders,
        _fill_scalar,
        _place_blocks,
    )

    if dtype and patch.data.dtype != np.dtype(dtype):  # as its row states it
        with operation_context():
            patch = patch.new(data=to_numpy(patch.data).astype(dtype))
    # the size `Spool.fill_gaps` gave: samples and one, every gap, or a width
    tol = merge["tolerance"]
    samples = tol.count is not None and bool(np.isfinite(tol.count))
    limit = int(tol.count) - 1 if samples else None if tol.count else tol
    patch = patch.fill_gaps(**{dim: limit}, fill_value=fill_value, samples=samples)
    coord: Any = patch.get_coord(dim)  # numeric, as only those fill
    if grid is None or coord == grid:
        return patch
    sign = 1 if grid.sorted or len(grid) < 2 else -1
    first = round(grid._samples_in((coord[0] - grid[0]) * sign)) if len(grid) else 0
    start, stop = max(-first, 0), min(len(coord), len(grid) - first)
    blocks = ((start, stop, first + start),) if stop > start else ()
    data = to_numpy(patch.data)
    fill = _fill_scalar(fill_value, data.dtype)
    data = _place_blocks(data, patch.get_axis(dim), len(grid), blocks, fill)
    coords, riders = _fill_riders(
        patch.coords, dim, len(grid), blocks, "Filling the gaps along"
    )
    # where the members went in the window, so loading twice gives one id
    params = {"dim": dim, "value": fill_value, "first": first, "grid": grid}
    attrs = stamp(patch.attrs, [patch.attrs], try_operation_id("Fill", params))
    with operation_context():
        return patch.new(
            data=data,
            coords=coords._update_grid(dim, **{dim: grid}).update(**riders),
            attrs=attrs,
        )


def _cast_via(dtype: np.dtype, chain: Sequence[np.dtype]) -> np.dtype | None:
    """
    The dtype a member's samples first round at along a promotion chain.

    The streaming buffer casts everything it holds at every promotion, so
    a member placed early passes through each buffer dtype after it. Only
    an integer reaching a float rounds, and a promotion chain leaves the
    integers once, so one dtype states what the whole chain does. None
    means nothing rounds before the dtype the array casts to anyway.
    """
    if dtype.kind not in "bui":
        return None
    via = next((x for x in chain if x.kind not in "bui"), None)
    return None if via is None or via == chain[-1] else via


def _cast_chain(dtypes) -> tuple[list[np.dtype], list[np.dtype | None]]:
    """The promotion chain of members placed in order, and each one's cast."""
    dtypes = [np.dtype(x) for x in dtypes]
    chain = list(accumulate(dtypes, np.result_type))
    return chain, [_cast_via(x, chain[num:]) for num, x in enumerate(dtypes)]


@dataclass
class PatchAssembler:
    """
    Assemble output patches from joined member rows.

    ``load_patch`` resolves one member row to its source patch (residual
    selections included); ``merge_kwargs`` carries the merge behavior;
    ``plan_dim`` names the dimension whose range the plan narrowed, and
    ``trim_dims`` any others, which are the only ones a member needs
    trimming on. The plan resolver hands this the joined member frame for
    one output at a time.
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
    # Only a source measured now and found unchanged keeps the recipe;
    # with strict=False, only one measured and found changed fails.
    sources_unchanged: Callable[..., bool] | None = None
    trim_dims: tuple[str, ...] = ()
    # Whether members carry coordinates the rows cannot rebuild, so the
    # index cannot stand for their patches (it can still size them).
    aux_coords: bool = False
    # Attrs the plan restates on its output, so members' values never clash.
    stamped: tuple[str, ...] = ()

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        expected_len = len(joined["current_index"].unique())
        merging = len(joined) > expected_len
        merge_dim = _get_varying_dim(joined) if merging else None
        units = plan_data_units(joined.get("data_units", []))
        if merge_dim is not None:
            # members go in the output coordinate's order
            descending = bool((to_float(joined[f"{merge_dim}_step"].values) < 0).all())
            joined = joined.sort_values(
                f"{merge_dim}_min", ascending=not descending, kind="stable"
            )
        df_dict_list = self._df_to_dict_list(joined)
        if merging:
            # Several sources merge into one patch. When the output size can
            # be determined from the instructions, stream the sources into a
            # pre-allocated array so they don't all need to be in memory with
            # the merged output at once.
            samples = _estimate_merge_samples(joined, merge_dim)
            if samples is not None:
                patch = self._merge_patches_streaming(
                    joined, df_dict_list, merge_dim, samples, units
                )
                return [patch]
        out = []
        target_units = None
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined, units)
            patch, target_units = _match_merge_units(patch, merge_dim, target_units)
            # The index doesn't carry all the dimensional info, so get what
            # merging needs from the patch coords (cheaper than attr dumps).
            info = patch.coords._get_dim_summary()
            info["patch"] = patch
            out.append(info)
        if len(out) > expected_len:
            fuse = _fuses(df_dict_list, self.plan_dim)
            out = _force_patch_merge(out, merge_kwargs=self.merge_kwargs, fuse=fuse)
        patches = [x["patch"] for x in out]
        if len(patches) == 1 and df_dict_list:
            patches = [self._on_grid(patches[0], joined, df_dict_list[0])]
        return patches

    def _on_grid(self, patch, joined, row) -> dc.Patch:
        """The patch on its stretch's grid, unless a source changed since planning."""
        unchanged, spec = self.sources_unchanged, row.get(f"_{self.plan_dim}_stretch")
        if (
            isinstance(spec, tuple)
            and unchanged
            and not unchanged(joined, strict=False)
        ):
            return patch  # its samples are not the ones the plan numbered
        return _on_stretch_grid(patch, self.plan_dim, row)

    def _load_trimmed_patch(self, patch_kwargs, joined, units=None) -> dc.Patch:
        """Load a single patch and trim it to its instruction range."""
        # convert kwargs to format understood by parser/patch.select
        kwargs = _convert_min_max_in_kwargs(patch_kwargs, joined)
        kwargs = _drop_associated_ranges(patch_kwargs, kwargs, self.plan_dim)
        if kwargs.get("_modified"):  # read hints and trims take the same bounds
            for dim in {self.plan_dim, *self.trim_dims} & set(kwargs):
                kwargs[dim] = _row_bounds(patch_kwargs, dim) or kwargs[dim]
        patch = self.load_patch(kwargs)
        # If the limits of the source patch were not modified, we can just
        # skip selection. This is important for missing coordinates
        # (NaN values) to not get trimmed out.
        source_kwargs = kwargs if kwargs.get("_modified") else {}
        # attr-style entries filter rows above; only the plan's dimension
        # and its trim dims are narrowed, everything else loads untouched.
        trims = (self.plan_dim, self.trim_dims)
        if select_kwargs := _plan_trim_kwargs(patch, source_kwargs, *trims):
            patch = patch.select(**_as_plan_units(patch, select_kwargs, patch_kwargs))
        return patch if units is None else patch.convert_units(units)

    def _merge_patches_streaming(
        self, joined, df_dict_list, merge_dim, samples, units=None
    ):
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
            if out is not None:  # its sources were found unchanged
                return _on_stretch_grid(out, self.plan_dim, df_dict_list[0])
        out = self._stream(joined, df_dict_list, merge_dim, samples, units)
        return self._on_grid(out, joined, df_dict_list[0])

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
        if len({meta.attrs.data_units for meta in metas}) > 1:
            return None
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
        defaults = _row_defaults(df_dict_list[0])
        return self._assemble(
            data, dims, merge_dim, coords, attrs, df_dict_list, defaults
        )

    def _recipe(self, rows, metas, dims, axis) -> Recipe | None:
        """
        The recipe the members make, or None when one cannot be named.

        Nothing is read here: every member is a window of a stored array
        whose shape its row states, so the members are laid end to end
        along the merged axis by arithmetic alone. A member holding its
        axes in another order than the first, or disagreeing off the
        merged axis, takes the patch path instead, which transposes each
        one as it is loaded and says what cannot be merged.

        The dtype is promoted a member at a time, in placement order, as
        the streaming merge promotes its buffer: promotion is not
        associative, and the two routes must give one answer. The buffer
        casts what it already holds at every promotion, so each member is
        also told where that chain first rounded it.
        """
        assert self.array_source is not None, "the caller checks for a source"
        sources, rest = [], None
        for row, meta in zip(rows, metas, strict=True):
            source = self._member_source(row, meta, dims)
            if source is None:
                return None
            placed = meta.coords.shape
            others = placed[:axis] + placed[axis + 1 :]
            if rest is None:
                rest = others
            elif others != rest:
                return None
            sources.append(source)
        chain, casts = _cast_chain([x.dtype for x in sources])
        return Recipe.from_sources(sources, axis=axis, dtype=chain[-1], cast_via=casts)

    def _member_source(self, row, meta, dims) -> ArraySource | None:
        """The stored window a member is, or None if it is not one."""
        if meta.dims != dims or not meta.window or self.array_source is None:
            return None
        source = self.array_source(row, meta.extent)
        if source is None:
            return None
        source = source[meta.window]
        assert source.shape == meta.coords.shape, "the window is the member's samples"
        if source.shape != source.extent:
            # a trim is a window of the array its row names, and holds
            # that window's id rather than the whole array's
            meta.attrs = meta.attrs.update(data_id=source.data_id)
        return source

    def describe_output(self, rows, units=None) -> OutputMeta | None:
        """
        What one output is, from its member rows in order; nothing is read.

        Members are sized as `_meta_from_index` sizes them, without its
        identity checks, or else from their envelopes (see `_sized_meta`).
        Only a member read as a window of its stored array has a source;
        the rest load as patches, as does one stating other data units
        than ``units``, which loading converts (its dtype and attrs are
        the converted ones). The data id is empty unless every member's is
        known without loading it. None when a row states no evenly sampled range
        or no dtype.
        """
        metas = [self._sized_meta(row, loose=True) for row in rows]
        dtypes = [x.get("_dtype") for x in rows]
        if None in metas or not all(isinstance(x, str) and x for x in dtypes):
            return None
        dims, known = metas[0].dims, []
        for num, (row, meta) in enumerate(zip(rows, metas, strict=True)):
            stated = meta.attrs.data_units
            converted = units is not None and not units_match(stated, units)
            if converted:
                dtypes[num] = _converted_dtype(dtypes[num], stated, units)
                meta.attrs = meta.attrs.update(data_units=units)
            elif self.can_use_index is None or self.can_use_index(row):
                meta.source = self._member_source(row, meta, dims)
            # a loaded member keeps its row's id only if loading changes
            # nothing, and a coordinate in other units changes it either way
            whole = meta.source is not None or not (converted or row.get("_modified"))
            whole = whole and not _units_converted(row, self.plan_dim)
            known.append(whole and not _is_missing(row.get("data_id")))
        chain, casts = _cast_chain(dtypes)
        for meta, cast in zip(metas, casts, strict=True):
            meta.cast_via = cast
        coords, attrs = metas[0].coords, metas[0].attrs
        if len(metas) > 1:
            coords, attrs = self._merged_meta(
                self.plan_dim,
                [x.coords for x in metas],
                [x.attrs for x in metas],
                rows,
            )
        coords = on_stretch_grid(coords, self.plan_dim, rows[0])
        if not all(known):
            attrs = attrs.update(data_id="")
        return OutputMeta(dims, coords, attrs, chain[-1], metas)

    def _stream(self, joined, df_dict_list, merge_dim, samples, units=None):
        """
        Copy each member into the output buffer as it is loaded.

        A member is released once copied; this avoids holding all source
        patches and the merged output in memory at the same time, as
        concatenating would.
        """
        buffer, offset, axis, dims = None, 0, None, None
        coords, attrs = [], []
        target_units = None
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined, units)
            patch, target_units = _match_merge_units(patch, merge_dim, target_units)
            data, coord = patch.data, patch.coords
            if dims is None:
                dims = patch.dims
                axis = dims.index(merge_dim)
            elif patch.dims != dims:
                # the same member with its axes in the merge's order
                data = np.transpose(data, [patch.dims.index(x) for x in dims])
                coord = coord.transpose(*dims)
            assert axis is not None  # set on the first pass through the loop
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
            coords.append(coord)
            attrs.append(patch.attrs)
        # All set on the first pass of the loop, which always runs.
        assert buffer is not None
        assert axis is not None
        assert dims is not None
        if offset != buffer.shape[axis]:  # over-estimated; trim excess.
            buffer = buffer[broadcast_for_index(buffer.ndim, axis, slice(0, offset))]
        return self._assemble(buffer, dims, merge_dim, coords, attrs, df_dict_list)

    def _assemble(self, data, dims, merge_dim, coords, attrs, rows, defaults=None):
        """Build the merged patch from the members' data, coords and attrs."""
        coords, attrs = self._merged_meta(merge_dim, coords, attrs, rows)
        # The merge drops empty and conflicting attrs, which the first
        # member's vendor model fills with its defaults; rows rebuild base
        # attrs, so restate them.
        if defaults:
            present = attrs.model_dump()
            attrs = attrs.update(
                **{k: v for k, v in defaults.items() if k not in present}
            )
        # The fold named the result; building it is not another array.
        with operation_context():
            return dc.Patch(data=data, coords=coords, attrs=attrs, dims=list(dims))

    def _merged_meta(self, merge_dim, coords, attrs, rows):
        """The coordinates and attrs members merged along ``merge_dim`` have."""
        # Ensure the loaded patches only vary along the expected dimension,
        # the same requirement _force_patch_merge enforces.
        summary_df = pd.DataFrame([coord._get_dim_summary() for coord in coords])
        found_dim = _get_merge_dim(summary_df)
        if found_dim != merge_dim:
            msg = (
                f"Cannot merge patches; expected them to vary along "
                f"{merge_dim} but found {found_dim}."
            )
            raise CoordMergeError(msg)
        attr_kwargs, coord_kwargs = _split_coord_merge_kwargs(self.merge_kwargs)
        fuse = _fuses(rows, merge_dim)
        new_coord = _get_merged_coord(
            summary_df, merge_dim, coords, fuse=fuse, **coord_kwargs
        )
        warn_if_histories_differ(attrs, "Merging")
        new_attrs = combine_patch_attrs(
            attrs,
            **attr_kwargs,
            drop_attrs=self.stamped,
            merge_params=self.merge_kwargs,
        )
        return new_coord, new_attrs

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
        # a window's id builds on the whole array's, so a row which does
        # not name the array cannot name the window
        cut = row.get("_modified") and self.plan_dim in str(row["dims"]).split(",")
        if cut and ids_enabled() and _is_missing(row.get("data_id")):
            return None
        return None if self.aux_coords else self._sized_meta(row)

    def _sized_meta(self, row: Mapping, loose: bool = False):
        """
        The coordinates an index row states for one member, or None.

        Only a member which is a window of its stored array is sized,
        unless ``loose``: then any other is sized from its envelope (a
        plan cuts members on their own grids), and states no window.
        """
        dims = tuple(str(row["dims"]).split(","))
        coord_map, extent, window = {}, [], []
        for dim in dims:
            # a coordinate with no units is NaN in a frame, not None,
            # and NaN would build a dimensionless quantity the patch
            # path does not have.
            units = None if _is_null(u := row.get(f"_{dim}_units")) else u
            cut = bool(row.get("_modified")) and dim == self.plan_dim
            if cut:
                placed = self._trim_window(row, dim, units)
            else:
                # Only the plan's dimension is trimmed here (a trim-dim plan
                # withholds the source range, so never takes this path).
                coord = coord_at_stored_unit(
                    coord_from_row(row, dim, units=units), row, dim
                )
                placed = (
                    None if coord is None else (coord, slice(0, len(coord)), len(coord))
                )
            if placed is None and loose:
                coord = _cut_coord(row, dim, units) if cut else None
                if coord is None:
                    coord = _sized_coord(row, dim, units)
                    if coord is not None and cut:  # a cut's end need not be a sample
                        coord = coord.select(_row_bounds(row, dim))[0]
                placed = None if coord is None else (coord, None, None)
            if placed is None:
                return None
            coord_map[dim], span, length = placed
            window.append(span)
            extent.append(length)
        coords = get_coord_manager(coord_map, dims=dims)
        # every coordinate's envelope is a coordinate's, not an attr
        named = {
            x.removeprefix("_").removesuffix("_coord_dtype")
            for x in map(str, row)
            if x.endswith("_coord_dtype")
        }
        attrs = _attrs_from_row(row, dims, coord_names={*dims, *named})
        if None in window:  # sized, but not a window of its stored array
            return _MemberMeta(dims, coords, attrs, (), ())
        return _MemberMeta(dims, coords, attrs, tuple(extent), tuple(window))

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
        if not len(coord):
            return None
        return coord, indexer, len(source)

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        This is significantly faster than iterating rows. Empty strings
        (missing format fields on file rows) normalize to None; stored
        relative paths pass through unchanged — the catalog's resolver
        owns resolving them against the spool root.
        """
        return [
            {k: None if isinstance(v, str) and not v else v for k, v in row.items()}
            for row in df.to_dict("records")
        ]
