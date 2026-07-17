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

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import dascore as dc
from dascore.exceptions import CoordMergeError
from dascore.utils.attrs import combine_patch_attrs
from dascore.utils.misc import broadcast_for_index
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


def _coord_only_kwargs(patch, kwargs) -> dict:
    """Keep only the kwargs naming a dim or coordinate of patch."""
    return {
        k: v
        for k, v in kwargs.items()
        if k in patch.dims or k in patch.coords.coord_map
    }


@dataclass
class PatchAssembler:
    """
    Assemble patches for one spool view.

    The frames define the view (presented rows, member instructions,
    source rows); ``load_patch`` resolves one joined row to its source
    patch; policy fields carry the merge behavior and patch-local
    post-selections. Instances cache the instruction-row index and are
    themselves cached per spool view.
    """

    df: pd.DataFrame
    source_df: pd.DataFrame
    instruction_df: pd.DataFrame
    load_patch: Callable[[Mapping], dc.Patch]
    merge_kwargs: Mapping
    post_selects: tuple = ()
    drop_columns: tuple = ()
    _indices: dict | None = field(default=None, repr=False)

    def get_patch(self, df_ind: int) -> dc.Patch:
        """Assemble the single patch presented at a row index."""
        patches = self.get_patches_from_index(df_ind)
        assert len(patches) == 1
        return patches[0]

    def get_patches_from_index(self, df_ind):
        """Given an index (from current df), return the corresponding patch."""
        source = self.source_df
        instruction = self.instruction_df
        # handle negative index.
        df_ind = df_ind if df_ind >= 0 else len(self.df) + df_ind
        try:
            inds = self.df.index[df_ind]
        except IndexError:
            msg = f"index of [{df_ind}] is out of bounds for spool."
            raise IndexError(msg) from None
        # Group positional instruction rows by current index (and cache) to
        # avoid a full instruction df scan for each requested patch.
        if self._indices is None:
            self._indices = instruction.groupby("current_index").indices
        positions = self._indices.get(inds)
        assert positions is not None and len(positions), "no instructions found"
        df1 = instruction.iloc[positions]
        joined = df1.join(source.drop(columns=df1.columns, errors="ignore"))
        # Occasionally, duplicates can creep into the source_df,
        # but it costs a bit to check for duplicates, so only check and drop
        # duplicates on large joined dataframes where performance might be
        # affected.
        if len(joined) > 10:
            cols = set(joined.columns) - set(self.drop_columns)
            joined = joined.drop_duplicates(subset=list(cols), keep="first")
        return self._patch_from_instruction_df(joined)

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        df_dict_list = self._df_to_dict_list(joined)
        expected_len = len(joined["current_index"].unique())
        if len(df_dict_list) > expected_len:
            # Several sources merge into one patch. When the output size can
            # be determined from the instructions, stream the sources into a
            # pre-allocated array so they don't all need to be in memory with
            # the merged output at once.
            merge_dim = _get_varying_dim(joined)
            samples = _estimate_merge_samples(joined, merge_dim)
            if samples is not None:
                patch = self._merge_patches_streaming(
                    joined, df_dict_list, merge_dim, samples
                )
                return [patch]
        out = []
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined)
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
        patch = self.load_patch(kwargs)
        # If the limits of the source patch were not modified, we can just
        # skip selection. This is important for missing coordinates
        # (NaN values) to not get trimmed out.
        source_kwargs = kwargs if kwargs.get("_modified") else {}
        # attr-style entries filter rows above; only coordinate entries
        # are valid patch selections.
        if select_kwargs := _coord_only_kwargs(patch, source_kwargs):
            patch = patch.select(**select_kwargs)
        # patch-local selections (samples=True) recorded by spool.select
        for post_kwargs, samples in self.post_selects:
            if usable := _coord_only_kwargs(patch, post_kwargs):
                patch = patch.select(**usable, samples=samples)
        return patch

    def _merge_patches_streaming(self, joined, df_dict_list, merge_dim, samples):
        """
        Merge the patches described by the instructions along merge_dim.

        Each patch is copied into a pre-allocated output array as it is
        loaded, then released; this avoids holding all source patches and
        the merged output in memory at the same time, as concatenating
        would.
        """
        buffer, offset, axis, dims = None, 0, None, None
        coords, attrs, summaries = [], [], []
        for patch_kwargs in df_dict_list:
            patch = self._load_trimmed_patch(patch_kwargs, joined)
            if dims is None:
                dims = patch.dims
                axis = patch.get_axis(merge_dim)
            elif patch.dims != dims:
                patch = patch.transpose(*dims)
            data = patch.data
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
            coords.append(patch.coords)
            attrs.append(patch.attrs)
            summaries.append(patch.coords._get_dim_summary())
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
        conf = attr_kwargs.get("conflicts", None)
        drop_conflicting = conf in {"drop", "keep_first"}
        new_coord = _get_merged_coord(
            summary_df, merge_dim, coords, drop_conflicting, **coord_kwargs
        )
        new_attrs = combine_patch_attrs(attrs, **attr_kwargs)
        return dc.Patch(data=buffer, coords=new_coord, attrs=new_attrs, dims=list(dims))

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
