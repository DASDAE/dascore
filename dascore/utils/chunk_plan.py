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

import inspect
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

import dascore as dc
from dascore.exceptions import (
    ChunkError,
    CoordMergeError,
    InvalidSpoolQueryError,
    ParameterError,
    UnitError,
)
from dascore.units import (
    DimensionalityError,
    Quantity,
    convert_units,
    get_byte_count,
    get_quantity,
    is_data_size,
)
from dascore.utils.attrs import validate_conflict
from dascore.utils.chunk import get_intervals
from dascore.utils.misc import get_middle_value, is_range
from dascore.utils.pd import _remove_overlaps, get_interval_columns
from dascore.utils.time import is_datetime64, is_timedelta64, to_float, to_timedelta64

# Columns which never participate in conflict policing and never carry to
# outputs: source bookkeeping (outputs are not file rows).
_SOURCE_COLUMNS = ("source_path", "source_format", "source_version", "source_patch_id")
# The default continuity tolerance; looser values warn when they force
# merges (#662).
_DEFAULT_TOLERANCE = 1.5


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
        `_patch_id`, the exact `{dim}_min/max` trim for that member, and
        `_modified` (False when the member loads whole).
    dim
        The chunked dimension.
    value
        The requested chunk length (None for merge mode).
    params
        Resolved parameters (group attrs, tolerances, overlap,
        keep_partial, conflict, snap_coords, missing_dim) — recorded, not
        referencing config.
    """

    outputs: pd.DataFrame
    members: pd.DataFrame
    dim: str
    value: Any
    params: dict = field(default_factory=dict)

    @property
    def merge_mode(self) -> bool:
        """Return True when this plan merges (no segmenting length)."""
        return self.value is None


def _resolve_group_attrs(group, columns) -> tuple[str, ...]:
    """Resolve the group attrs: per-call > config; explicit names must exist."""
    if group is not None:
        group = (group,) if isinstance(group, str) else tuple(group)
        if missing := [x for x in group if x not in columns]:
            msg = (
                f"group attribute(s) {missing} do not exist on any patch in the spool."
            )
            raise InvalidSpoolQueryError(msg)
        return group
    # Config (and default) names are best-effort.
    return tuple(x for x in dc.get_config().groupby_attrs if x in columns)


def samples_adjusted_envelopes(
    df: pd.DataFrame, residuals, drop_empty: bool = True
) -> pd.DataFrame:
    """
    Adjust envelope columns for patch-local samples residuals.

    A ``samples=True`` index window trims each patch at load, so the
    planner must consume the trimmed envelopes or it publishes outputs
    that lie entirely outside the selected samples (phantom empties).
    Negative indices resolve per patch against the envelope-derived
    sample count (rows whose count is unknown keep their envelope as a
    candidacy superset — exactness is always re-applied at load).
    ``drop_empty`` removes rows whose window selects nothing (planning
    truth); equality comparison keeps them, since a presented-but-empty
    row is still a presented row.
    """

    def _usable_index(value) -> bool:
        return value is None or isinstance(value, int | np.integer)

    df = df.copy(deep=False)
    for coords, samples in residuals:
        if not samples:
            continue
        for name, value in coords.items():
            cols = [f"{name}_min", f"{name}_max", f"{name}_step"]
            if not set(cols).issubset(df.columns) or not is_range(value):
                continue
            lo_idx, hi_idx = value
            if not (_usable_index(lo_idx) and _usable_index(hi_idx)):
                continue
            mins, maxs, steps = (df[c] for c in cols)
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
            unresolved = pd.Series(False, index=df.index)
            for pos in (lo_pos, hi_pos):
                if pos is not None:
                    unresolved |= pos.isna()
            lo_off = None if lo_pos is None else lo_pos * abs_steps
            hi_off = None if hi_pos is None else (hi_pos - 1) * abs_steps
            new_min = mins if lo_off is None else mins + lo_off
            new_max = maxs if hi_off is None else mins + hi_off
            desc_min = maxs if hi_off is None else maxs - hi_off
            desc_max = maxs if lo_off is None else maxs - lo_off
            new_min = new_min.where(~descending, other=desc_min)
            new_max = new_max.where(~descending, other=desc_max)
            # unresolvable rows keep their envelope (candidacy superset)
            new_min = new_min.mask(unresolved, mins)
            new_max = new_max.mask(unresolved, maxs)
            # rows whose window is empty or lies entirely outside the
            # patch contribute nothing; test before clipping so such
            # windows are not resurrected as one-sample envelopes
            keep = (new_min <= new_max) & (new_min <= maxs) & (new_max >= mins)
            keep |= unresolved
            df[cols[0]] = new_min.clip(lower=mins, upper=maxs)
            df[cols[1]] = new_max.clip(lower=mins, upper=maxs)
            if drop_empty:
                df = df[keep]
    return df


def _ensure_patch_id(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the positional identity fallback for plain dataframes."""
    if "_patch_id" in df.columns:
        return df
    return df.assign(_patch_id=np.arange(len(df)))


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


def _continuity_group(start, stop, step, tolerance) -> pd.Series:
    """Label maximal near-contiguous runs (spec 2.4)."""
    args = np.argsort(start.to_numpy())
    start_sorted = start.iloc[args]
    stop_sorted = stop.iloc[args]
    # envelopes are value-ordered regardless of coordinate orientation,
    # so the continuity margin uses the step magnitude
    step_sorted = step.iloc[args].abs()
    stop_cum_max = stop_sorted.cummax()
    end_markers = stop_cum_max.shift() + step_sorted * tolerance
    has_gap = start_sorted > end_markers
    group = has_gap.astype(np.int64).cumsum()
    return group[start.index]


def _normalize_chunk_units(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Re-express the chunk dim's envelopes in one unit per dimensionality.

    Envelope columns are stored in each coordinate's original units, so
    compatible spellings (metres beside feet) are not directly
    comparable. Rows sharing a dimensionality convert to the unit of
    their first row — ordered by (envelope min in base units, patch id),
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
        order = pd.DataFrame({"_min": base_min, "_pid": sub["_patch_id"]}).sort_values(
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
        df.loc[rows, unit_col] = target
    return df


def _partition(
    df, name, group_attrs, tolerance, sampling_tolerance
) -> tuple[pd.Series, bool]:
    """
    Return (partition labels, forced_merge): rows sharing a label may
    combine (spec 2).

    Components: group attrs, dims signature, structural def keys of
    non-chunked coords, sampling group, and continuity group. Continuity
    is evaluated *within* each other-component cell so unrelated patches
    can never bridge a gap. `forced_merge` is True when a loosened
    tolerance merged patches the default would have kept apart (#662);
    the caller owns warning about it.
    """
    _start, _stop, step = get_interval_columns(df, name)
    cols = [x for x in group_attrs if x in df.columns]
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
    base = (
        df.groupby(cols, dropna=False, sort=False).ngroup()
        if cols
        else pd.Series(0, index=df.index)
    )
    samp = _sampling_group(step, sampling_tolerance)
    cell = base.astype(str) + "_" + samp.astype(str)
    cont = pd.Series(0, index=df.index, dtype=np.int64)
    forced_merge = False
    for _, index in df.groupby(cell, sort=False).groups.items():
        sub = df.loc[index]
        s, e, st = get_interval_columns(sub, name)
        labels = _continuity_group(s, e, st, tolerance).astype(np.int64)
        cont.loc[index] = labels
        if tolerance > _DEFAULT_TOLERANCE and not forced_merge:
            default = _continuity_group(s, e, st, _DEFAULT_TOLERANCE)
            forced_merge = default.nunique() > labels.nunique()
    return cell + "_" + cont.astype(str), forced_merge


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
    """Reject quantity chunk lengths that cannot describe a length."""
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
    Overlapping members are excluded: `_remove_overlaps` deduplicates
    them at member-build time, so they do not add samples.
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


def _quantity_to_dim_value(quant, sub, name, start_dtype):
    """Convert a (non-size) quantity into the chunked dimension's units."""
    if is_datetime64(start_dtype) or is_timedelta64(start_dtype):
        try:
            seconds = quant.to("s").magnitude
        except DimensionalityError:
            msg = (
                f"Cannot chunk {name!r} by {quant}: the coordinate is "
                "time-like, so the value must have units of time."
            )
            raise UnitError(msg) from None
        return to_timedelta64(seconds)
    units_col = f"_{name}_units"
    if units_col in sub.columns:
        units = sub[units_col].iloc[0]
        if units is None or pd.isnull(units) or units == "":
            msg = (
                f"Cannot chunk {name!r} by {quant}: the coordinate has no "
                "units, so a unit-bearing length is ambiguous."
            )
            raise UnitError(msg)
        try:
            # A length is a DELTA: converting through two anchor points
            # cancels an affine unit's offset (20 degC of extent is 36
            # degF, never 68), and convert_units also accepts scaled
            # unit spellings that pint's .to() rejects.
            magnitude = float(quant.magnitude)
            from_units = str(quant.units)
            anchor = convert_units(0.0, to_units=units, from_units=from_units)
            end = convert_units(magnitude, to_units=units, from_units=from_units)
            return end - anchor
        except (DimensionalityError, UnitError):
            msg = (
                f"Cannot chunk {name!r} by {quant}: incompatible with the "
                f"coordinate's units of {units}."
            )
            raise UnitError(msg) from None
    # A frame with no units column states no units at all; envelopes are
    # native magnitudes, so there is nothing to convert the quantity to.
    msg = (
        f"Cannot chunk {name!r} by {quant}: the frame records no units "
        "for the coordinate, so a unit-bearing length is ambiguous."
    )
    raise UnitError(msg)


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
        return _quantity_to_dim_value(val, sub, name, start_dtype)

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


def _police_columns(sub: pd.DataFrame, name, conflict) -> dict:
    """
    Return the carried column values for one partition (spec 2.5/6.4).

    Group attrs, dims, and def keys are single-valued by construction.
    Remaining public attrs must be single-valued, policed by `conflict`.
    """
    dims = set(str(sub.iloc[0].get("dims", "")).split(","))
    coord_names = dims | {name}
    carried: dict[str, Any] = {}
    for col in sub.columns:
        if col.startswith("_") or col in _SOURCE_COLUMNS:
            continue
        owner = _coord_owner(col, coord_names)
        if owner == name:  # chunk-dim envelope columns are rebuilt
            continue
        values = sub[col].unique()
        single = len(values) == 1 or (len(values) and pd.isnull(values).all())
        if single:
            carried[col] = values[0]
            continue
        # Group attrs and dims are partition keys, so they are always
        # single-valued above and never reach the conflict policy here.
        if owner is not None or conflict == "raise":
            msg = (
                f"Cannot merge on dim {name} because all values for "
                f"{col} are not equal. Consider using the `conflict` "
                "argument to loosen this restriction."
            )
            raise CoordMergeError(msg)
        if conflict == "keep_first":
            carried[col] = sub[col].iloc[0]
        # conflict == "drop": omit the column entirely.
    # Structural (dimension) def keys carry — single-valued by partitioning.
    for col in _dim_def_key_columns(sub, name):
        if col in sub.columns:
            carried[col] = sub[col].iloc[0]
    # Canonical units carry for every dimension, the chunked one included
    # (partition-constant: units are a sampling-partition component).
    for coord in coord_names:
        col = f"_{coord}_units"
        if col in sub.columns:
            carried[col] = sub[col].iloc[0]
    return carried


def _output_dtypes(sub: pd.DataFrame, members: pd.DataFrame) -> pd.Series | None:
    """
    The element dtype each output assembles to, indexed by output_id.

    Carried privately (see SPOOL_PRIVATE_RENAMES) because a size-based
    chunk needs it. It is resolved per *output* rather than per
    partition: a partition may legitimately mix dtypes, but an output
    drawing only from its float32 members really is float32, and
    claiming the partition-wide upcast would both over-size a later
    chunk and make the plan row disagree with the patch it assembles
    (which spool equality compares).
    """
    if "_dtype" not in sub.columns or members.empty:
        return None
    by_patch = sub.drop_duplicates("_patch_id").set_index("_patch_id")["_dtype"]
    grouped = members.groupby("output_id")["_patch_id"]
    # A scalar-returning apply on a SeriesGroupBy is always a Series; the
    # stubs cannot see through the lambda and claim DataFrame | Series.
    dtypes: pd.Series = grouped.apply(  # ty: ignore[invalid-assignment]
        lambda pids: _combined_dtype(by_patch.reindex(pids))
    )
    # "" is the same not-known sentinel ingest writes; never leave NaN,
    # which would reach the derived catalog as the string "nan".
    return dtypes.map(lambda x: "" if x is None else str(x))


def _build_members(sub: pd.DataFrame, outputs: pd.DataFrame, name) -> pd.DataFrame:
    """
    Bind one partition's outputs to trimmed source slices.

    Sources are ordered by (start, _patch_id); overlapping coverage is
    deduplicated so the earlier source owns the overlap (D3: complete
    overlaps keep the first member, deterministically).
    """
    min_name, max_name = f"{name}_min", f"{name}_max"
    step_name = f"{name}_step"
    # The partition's unit (single-valued after normalization) rides on
    # every member row: trim magnitudes mean THIS unit, not whatever
    # spelling the member's source file used.
    unit_col = f"_{name}_units"
    plan_units = sub[unit_col].iloc[0] if unit_col in sub.columns else None
    sub = sub.sort_values([min_name, "_patch_id"], kind="stable")
    original = sub[[min_name, max_name]].reset_index(drop=True)
    sub = _remove_overlaps(sub, name)
    # Fully-covered sources become degenerate after start correction; they
    # contribute nothing (deterministic keep-first dedup).
    keep = sub[min_name].values <= sub[max_name].values
    sub = sub[keep]
    original = original[keep].reset_index(drop=True)
    # sub and outputs are always non-empty here: a partition too short to
    # yield an interval raises ChunkError in the caller (and the earliest
    # source is never fully covered), so both keep at least one row.
    steps = sub[step_name].values
    src1 = sub[min_name].values
    src2 = sub[max_name].values
    chu1 = outputs[min_name].values
    chu2 = outputs[max_name].values
    out_ids = outputs["output_id"].to_numpy()
    patch_ids = sub["_patch_id"].to_numpy()
    orig_min = original[min_name].to_numpy()
    orig_max = original[max_name].to_numpy()
    modified_src = (
        sub["_modified"].to_numpy()
        if "_modified" in sub
        else np.zeros(len(sub), dtype=bool)
    )
    # Map each output onto the source rows it draws from.
    starts_ind = np.searchsorted(src1, chu1, side="right") - 1
    ends_ind = np.searchsorted(src2, chu2, side="left")
    rows = []
    for out_num, (a, b) in enumerate(zip(starts_ind, ends_ind)):
        a = max(int(a), 0)
        for src_num in range(a, int(b) + 1):
            if src_num >= len(sub):
                continue
            lo = max(src1[src_num], chu1[out_num])
            hi = min(src2[src_num], chu2[out_num])
            # Sources within a partition are continuous (partitioning
            # splits on gaps) and start-corrected, so searchsorted never
            # offers a source which does not overlap the output. Assert it
            # rather than skipping: silently dropping a source would lose
            # data, and the state cannot be reached from the public API.
            assert lo <= hi, f"source {src_num} does not overlap output {out_num}"
            unchanged = (
                lo == orig_min[src_num]
                and hi == orig_max[src_num]
                and not modified_src[src_num]
            )
            row = {
                "output_id": out_ids[out_num],
                "_patch_id": patch_ids[src_num],
                min_name: lo,
                max_name: hi,
                step_name: steps[src_num],
                "_modified": not unchanged,
            }
            if plan_units is not None:
                row[unit_col] = plan_units
            rows.append(row)
    return pd.DataFrame(rows)


def build_chunk_plan(
    df: pd.DataFrame,
    *,
    overlap=None,
    keep_partial: bool = False,
    snap_coords: bool = True,
    tolerance: float = 1.5,
    conflict: Literal["drop", "raise", "keep_first"] = "raise",
    group=None,
    missing_dim: Literal["raise", "drop"] = "raise",
    **kwargs,
) -> ChunkPlan:
    """
    Build a chunk plan from a flat patch relation.

    Parameters mirror `Spool.chunk` (see the chunking formalities spec);
    exactly one keyword names the dimension to chunk and its length
    (`None`/`...` merges).
    """
    if len(kwargs) != 1:
        msg = (
            f"Chunking only supported along one dimension. You passed kwargs: {kwargs}"
        )
        raise ParameterError(msg)
    # Fail here rather than later at assembly, so the error points at the
    # offending chunk call. See #804.
    validate_conflict(conflict)
    ((name, value),) = kwargs.items()
    value = None if value is Ellipsis else value
    # Police quantities before merge_mode is decided: a NaN magnitude is
    # null, so a nan-valued size would silently merge the whole spool
    # when the user asked for a size *cap*.
    for label, quant in (("chunk value", value), ("overlap", overlap)):
        _validate_quantity(label, quant, name)
    merge_mode = pd.isnull(value)
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
    if missing_dim not in ("raise", "drop"):
        msg = f"missing_dim must be 'raise' or 'drop', got {missing_dim!r}"
        raise ParameterError(msg)

    min_name, max_name = f"{name}_min", f"{name}_max"
    if min_name not in df.columns and not df.empty:
        msg = f"No patch in the spool has a {name!r} dimension to chunk."
        raise ChunkError(msg)
    empty_members = pd.DataFrame(
        columns=["output_id", "_patch_id", min_name, max_name, "_modified"]
    )
    params = dict(
        overlap=overlap,
        keep_partial=keep_partial,
        snap_coords=snap_coords,
        tolerance=tolerance,
        conflict=conflict,
        missing_dim=missing_dim,
        group=_resolve_group_attrs(group, set(df.columns)),
        sampling_group_tolerance=dc.get_config().sampling_group_tolerance,
    )
    if df.empty:
        outputs = pd.DataFrame(columns=[min_name, max_name, "output_id"])
        return ChunkPlan(outputs, empty_members, name, value, params)
    df = _ensure_patch_id(df)
    df = _normalize_chunk_units(df, name)
    # Missing chunk-dim envelopes, and patches carrying the name only as
    # a non-dimensional coordinate (spec 7 / D2): chunking is defined on
    # dimensions, so both fall under missing_dim. Envelope presence is
    # not enough — auxiliary coordinates index their envelopes too, but
    # their patches cannot be trimmed or merged *along* the name.
    null_rows = pd.isnull(df[min_name]) | pd.isnull(df[max_name])
    if "dims" in df.columns:
        dim_lists = df["dims"].fillna("").astype(str).str.split(",")
        not_a_dim = ~dim_lists.map(lambda dims: name in dims)
    else:
        not_a_dim = pd.Series(False, index=df.index)
    unusable = null_rows | not_a_dim
    if unusable.any():
        if missing_dim == "raise":
            bad = df.loc[unusable, "_patch_id"].tolist()
            rides = int((not_a_dim & ~null_rows).sum())
            detail = (
                f" ({rides} of them carry {name!r} only as a non-dimensional "
                "coordinate; chunking is defined on dimensions)"
                if rides
                else ""
            )
            msg = (
                f"{int(unusable.sum())} patch(es) lack the chunk dimension "
                f"{name!r}{detail} (patch ids {bad[:5]}...). Pass "
                "missing_dim='drop' to exclude them."
            )
            raise ChunkError(msg)
        df = df[~unusable]
    if df.empty:
        outputs = pd.DataFrame(columns=[min_name, max_name, "output_id"])
        return ChunkPlan(outputs, empty_members, name, value, params)

    labels, forced_merge = _partition(
        df, name, params["group"], tolerance, params["sampling_group_tolerance"]
    )
    if forced_merge:
        msg = (
            f"There is a gap in the patch along dimension {name} but a "
            f"merge tolerance of {tolerance} was used to force merging "
            "the patches. As a result, some patches in the chunked spool "
            "may be unevenly sampled, or have their sampling rate increased."
        )
        warnings.warn(msg, UserWarning, stacklevel=_user_stacklevel())
    per_partition = _needs_partition_resolution(value, overlap)
    if not per_partition:
        value_c, overlap_c = _coerce_length_overlap(value, overlap, df[min_name].dtype)
    size_diagnostics: list[dict] = []
    out_frames, member_frames = [], []
    next_id = 0
    # Deterministic partition order (spec 8): by (partition min, smallest
    # member patch id) — never by anything derived from input row order.
    grouped = df.groupby(labels, sort=False)
    stats = grouped.agg(_min=(min_name, "min"), _pid=("_patch_id", "min"))
    part_order = stats.sort_values(["_min", "_pid"], kind="stable").index
    groups = grouped.groups
    for label in part_order:
        sub = df.loc[groups[label]]
        start, stop, step = get_interval_columns(sub, name)
        part_step = get_middle_value(step.values)  # D7: one step everywhere
        g_start, g_stop = start.min(), stop.max()
        if per_partition and not merge_mode:
            # The bound itself is enforced by the packing factor, which is
            # measured in units of this step and so cancels the choice
            # (length works out to n_samples / max density either way).
            # The smallest step is still the better unit: it makes the
            # flooring granularity the finest the partition allows.
            size_step = step.abs().min()
            value_c, overlap_c, diag = _resolve_partition_length(
                value, overlap, sub, name, size_step, df[min_name].dtype
            )
            if diag is not None:
                size_diagnostics.append({"first_output_id": next_id, **diag})
        if merge_mode:
            start_stop = np.atleast_2d(np.asarray([g_start, g_stop]))
        else:
            try:
                start_stop = get_intervals(
                    g_start,
                    g_stop,
                    value_c,
                    overlap=overlap_c,
                    # interval arithmetic is over (direction-free) envelope
                    # values; a descending coordinate's negative step would
                    # invert the final partial interval
                    step=abs(part_step),
                    keep_partials=keep_partial,
                )
            except ChunkError:  # partition too short; skip (D8)
                continue
        sub_sorted = sub.sort_values([min_name, "_patch_id"], kind="stable")
        carried = _police_columns(sub_sorted, name, conflict)
        outputs = pd.DataFrame(start_stop, columns=[min_name, max_name])
        outputs[f"{name}_step"] = part_step
        for col, val in carried.items():
            outputs[col] = val
        outputs["output_id"] = np.arange(next_id, next_id + len(outputs))
        next_id += len(outputs)
        members = _build_members(sub, outputs, name)
        # Plan invariant: every published output has at least one member.
        # An advertised row that cannot assemble is never surfaced as a
        # runtime error; it is not surfaced at all.
        fed = set(members["output_id"]) if not members.empty else set()
        outputs = outputs[outputs["output_id"].isin(fed)]
        if (dtypes := _output_dtypes(sub_sorted, members)) is not None:
            outputs["_dtype"] = outputs["output_id"].map(dtypes).fillna("")
        out_frames.append(outputs)
        member_frames.append(members)
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
    if not out_frames or all(x.empty for x in out_frames):
        msg = "Could not chunk. No segments with sufficient length found."
        raise ChunkError(msg)
    outputs = pd.concat(out_frames, ignore_index=True)
    members = pd.concat(
        [x for x in member_frames if not x.empty] or [empty_members],
        ignore_index=True,
    )
    return ChunkPlan(outputs, members, name, value, params)


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
    span = to_float(step)
    out = []
    for cut in cuts:
        # The ratio is inexact — `to_float` rounds both operands and then
        # the quotient — so it only starts the search, and the grid it
        # describes settles it. `CoordRange._get_index`'s fudge factor is
        # not enough here: at a million samples the error outgrows any
        # fixed tolerance, and an index one too high puts the boundary
        # sample in the epoch *before* the boundary, which is the one
        # place it must not go. Comparing the values themselves is exact,
        # and each loop steps at most once for any plausible error.
        index = math.ceil(to_float(cut - start) / span)
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
    return out


def build_subdivision_plan(df: pd.DataFrame, cuts, name: str) -> ChunkPlan:
    """
    Build a plan splitting each row of a relation at its own cut values.

    Unlike a chunk plan, no row ever meets another: each output is one
    contiguous piece of exactly one source row, so the pieces of a row
    partition its samples — none is dropped, none is duplicated, and a
    row with no cuts passes through as a single unmodified output. This
    is what an operation which must re-describe patches rather than
    restructure them (`Spool.conform_to_inventory`) needs.

    Parameters
    ----------
    df
        The relation to subdivide; one row per patch.
    cuts
        One sequence of cut values per row, in the row's own units, each
        above that row's minimum and no greater than its maximum — a cut
        on the maximum yields a one-sample final piece. A cut opens a new
        piece at the first sample at or after it, so a row with `n`
        distinct cuts becomes at most `n + 1` outputs.
    name
        The dimension being subdivided.

    Notes
    -----
    Every cut row needs a usable step to find its sample grid with, so
    callers must reject a null or zero step themselves — they are the
    ones which can name the file it came from.
    """
    min_name, max_name = f"{name}_min", f"{name}_max"
    step_name = f"{name}_step"
    assert {min_name, max_name, step_name}.issubset(df.columns)
    # One entry per row, even where it is empty: a short sequence would
    # drop the rows past its end from the plan, and so from the spool.
    assert len(cuts) == len(df)
    df = _ensure_patch_id(df).reset_index(drop=True)
    positions, lows, highs, modified = [], [], [], []
    for position, row_cuts in enumerate(cuts):
        start, stop = df.at[position, min_name], df.at[position, max_name]
        # Envelopes are value-ordered whatever the coordinate's
        # orientation, so the grid is walked by the step's magnitude.
        step = abs(df.at[position, step_name])
        assert not len(row_cuts) or (not pd.isnull(step) and step)
        grid = _snapped_cuts(row_cuts, start, step)
        bounds = [start, *grid]
        for index, low in enumerate(bounds):
            high = bounds[index + 1] - step if index + 1 < len(bounds) else stop
            positions.append(position)
            lows.append(low)
            highs.append(high)
            modified.append(bool(grid))
    ids = np.arange(len(positions), dtype=np.int64)
    # Outputs are not file rows: source bookkeeping stays on the members,
    # and the dimension's structural identity described the whole row.
    outputs = df.iloc[positions].drop(
        columns=["_patch_id", f"_{name}_def_key", *_SOURCE_COLUMNS], errors="ignore"
    )
    outputs = outputs.assign(**{min_name: lows, max_name: highs, "output_id": ids})
    members = pd.DataFrame(
        {
            "output_id": ids,
            "_patch_id": df["_patch_id"].to_numpy()[positions],
            min_name: lows,
            max_name: highs,
            step_name: df[step_name].to_numpy()[positions],
            "_modified": modified,
        }
    )
    return ChunkPlan(outputs.reset_index(drop=True), members, name, None, {})
