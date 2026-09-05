"""
Execute spool views: turn member instructions into loaded patches.

This is the consumer of the members (instruction) table that
`dascore.utils.chunk_plan` produces and every spool view carries: it
joins member rows to their source rows, loads each source patch through
a caller-supplied loader, applies exact trims, and merges multi-member
outputs (streaming into a pre-allocated buffer when the output size is
known). The spool owns *what* rows exist; this module owns *how* a row
becomes a Patch.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

import dascore as dc
from dascore.core.coordmanager import CoordManager, get_coord_manager
from dascore.core.coords import get_coord
from dascore.exceptions import CoordMergeError, UnitError
from dascore.io.index.ingest import _is_missing
from dascore.io.index.schema import RESERVED_ATTR_COLUMNS
from dascore.units import get_quantity
from dascore.utils.attrs import combine_patch_attrs, warn_if_histories_differ
from dascore.utils.chunk_plan import _SOURCE_COLUMNS
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
from dascore.workflow.identity import ids_enabled


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


def _attrs_from_row(row: Mapping, dims: tuple[str, ...]) -> dc.PatchAttrs:
    """
    The attrs an index row states for its member.

    Every attr the index could hold is a column; what is not a column is
    the plan's own bookkeeping, the coordinate envelopes, and the storage
    provenance (the lineage ids of which are added back below). A field
    the file left unset, or one ingest could not type -- a list, an
    array, a name colliding with a structural column -- is null here and
    takes its default, as it would on the patch `read` builds.
    """
    envelope = {f"{d}_{x}" for d in dims for x in ("min", "max", "step", "units")}
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
            if name in out:
                out[name] = np.dtype(dtype).type(out[name])
    out["dims"] = dims
    # the lineage ids are source columns, so the comprehension drops
    # them; a merged patch folds them, and folding nothing is not the
    # same as folding what the members carried. A moved source has its
    # patch id cleared until it is read again.
    for name in ("patch_id", "processing_id"):
        if not _is_null(value := row.get(name)):
            out[name] = value
    if not _is_null(key := row.get("source_patch_key")):
        out["_source_patch_key"] = key
    return dc.PatchAttrs.from_dict(out)


def _is_null(value) -> bool:
    """True for a missing scalar; an array is a value."""
    return np.ndim(value) == 0 and pd.isnull(value)


def _row_range(row: Mapping, dim: str) -> tuple[Any, Any, Any] | None:
    """
    A dimension's evenly sampled range as the row states it, or None.

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
    if step < np.zeros((), dtype=np.asarray(step).dtype):
        # the envelope orders values, not samples; a descending
        # coordinate's start is its maximum, which the row does not say
        return None
    stored = row.get(f"_{dim}_coord_dtype")
    if not (isinstance(stored, str) and np.issubdtype(np.dtype(stored), np.number)):
        return lo, hi, step
    # The frame holds every numeric envelope as float, so an integer
    # coordinate must be cast back. That is only right when the values
    # are the file's own: a unit conversion made them float in truth,
    # and past 2**53 a float cannot have held the integer exactly.
    source_units = row.get(f"_{dim}_units_source")
    if not _is_null(source_units) and source_units != row.get(f"_{dim}_units"):
        return None
    kind = np.dtype(stored).type
    if np.issubdtype(kind, np.integer) and max(abs(lo), abs(hi)) > 2**53:
        return None
    return kind(lo), kind(hi), kind(step)


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
    # Hands back an untrimmed member's whole array, or None when the
    # member must be loaded as a patch; the index then stands in for
    # the member's coordinates and attrs.
    load_array: Callable[[Mapping], np.ndarray | None] | None = None
    can_load_array: Callable[[Mapping], bool] | None = None

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
        decide that before anything is read; an array whose shape the row
        did not predict is only found once it is loaded, and abandons the
        attempt. Returning from that attempt releases its buffer before
        the retry reloads the source coordinates, attrs, and arrays.
        """
        metas = self._member_meta_from_index(df_dict_list)
        if metas is not None:
            out = self._stream(joined, df_dict_list, merge_dim, samples, metas)
            if out is not None:
                return out
        return self._stream(joined, df_dict_list, merge_dim, samples, None)

    def _stream(self, joined, df_dict_list, merge_dim, samples, metas):
        """
        Copy each member into the output buffer as it is loaded.

        A member is released once copied; this avoids holding all source
        patches and the merged output in memory at the same time, as
        concatenating would. Returns None when ``metas`` promised an
        array shape the file did not deliver, so the caller can start
        over on the patch path.
        """
        buffer, offset, axis, dims = None, 0, None, None
        coords, attrs, summaries = [], [], []
        target_units = None
        for num, patch_kwargs in enumerate(df_dict_list):
            member = None
            if metas is not None:
                member = self._member_from_meta(patch_kwargs, metas[num])
                if member is None:
                    return None
            if member is None:
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
            # Mixed dtypes upcast, mirroring np.concatenate behavior.
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
        new_attrs = combine_patch_attrs(attrs, **attr_kwargs)
        return dc.Patch(data=buffer, coords=new_coord, attrs=new_attrs, dims=list(dims))

    def _member_meta_from_index(self, rows) -> list[_MemberMeta] | None:
        """What the rows state about every member, or None if any is silent.

        Metadata only: nothing is read here, so a merge the index cannot
        describe costs no array reads before it falls back.
        """
        if self.load_array is None:
            return None
        if self.can_load_array is not None and not all(
            self.can_load_array(row) for row in rows
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
        merge needs from a member whose plan trims nothing; the patch,
        its coordinate parsing and its attr decoding are skipped.
        Anything the row cannot state -- a dimension without a range --
        sends the whole merge down the patch path instead.

        The index holds no history, a list rather than a column, so a
        member built here states none and the merged patch carries none.
        Only a format which stores a history to begin with (DASDAE) has
        one to lose, and only until it is read as a patch again.
        """
        # A moved source has its id cleared until it is read again, and
        # folding no id is not folding the one the patch carries; an attr
        # the index could not hold is on the patch and would be lost here.
        if ids_enabled() and "patch_id" in row and _is_missing(row["patch_id"]):
            return None
        if not _is_null(complete := row.get("_attrs_complete")) and not complete:
            return None
        dims = tuple(str(row["dims"]).split(","))
        coord_map = {}
        for dim in dims:
            envelope = _row_range(row, dim)
            if envelope is None:
                return None
            lo, hi, step = envelope
            # a coordinate with no units is NaN in a frame, not None,
            # and NaN would build a dimensionless quantity the patch
            # path does not have.
            units = None if _is_null(u := row.get(f"_{dim}_units")) else u
            coord_map[dim] = get_coord(start=lo, stop=hi + step, step=step, units=units)
        coords = get_coord_manager(coord_map, dims=dims)
        return _MemberMeta(dims, coords, _attrs_from_row(row, dims))

    def _member_from_meta(self, row: Mapping, meta: _MemberMeta) -> _Member | None:
        """
        The member a row describes, with its array read.

        Returns None when the file did not deliver the shape the row
        predicted, which the caller can only discover here.
        """
        assert self.load_array is not None, "the caller checks for a loader"
        data = self.load_array(row)
        if data is None or data.shape != meta.coords.shape:
            return None
        return _Member(meta.dims, data, meta.coords, meta.attrs)

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        This is significantly faster than iterating rows. Empty strings
        (missing format fields on file rows) normalize to None; stored
        relative paths pass through unchanged — the catalog's resolver
        owns resolving them against the spool root.
        """
        df = df.copy(deep=False).replace("", None)
        return df.to_dict("records")
