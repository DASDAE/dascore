"""Module for spools, containers of patches."""
from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from functools import singledispatch
from pathlib import Path

import pandas as pd
from rich.text import Text
from typing_extensions import Self

import dascore as dc
import dascore.io
from dascore.constants import PatchType, numeric_types, timeable_types
from dascore.exceptions import InvalidSpoolError
from dascore.utils.chunk import ChunkManager
from dascore.utils.display import get_dascore_text, get_nice_text
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import CacheDescriptor
from dascore.utils.patch import _force_patch_merge, patches_to_df
from dascore.utils.pd import (
    _convert_min_max_in_kwargs,
    adjust_segments,
    filter_df,
    get_column_names_from_dim,
    get_dim_names_from_columns,
)


class BaseSpool(abc.ABC):
    """Spool Abstract Base Class (ABC) for defining Spool interface."""

    _rich_style = "bold"

    @abc.abstractmethod
    def __getitem__(self, item: int) -> PatchType:
        """Returns a patch from the spool."""

    @abc.abstractmethod
    def __iter__(self) -> PatchType:
        """Iterate through the Patches in the spool."""

    def update(self) -> Self:
        """Updates the contents of the spool and returns a spool."""
        return self

    @abc.abstractmethod
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        **kwargs,
    ) -> Self:
        """
        Chunk the data in the spool along specified dimensions.

        Parameters
        ----------
        overlap
            The amount of overlap between each segment, starting with the end of
            first patch. Negative values can be used to induce gaps.
        keep_partial
            If True, keep the segments which are smaller than chunk size.
            This often occurs because of data gaps or at end of chunks.
        snap_coords
            If True, snap the coords on joined patches such that the spacing
            remains constant.
        tolerance
            The number of samples a block of data can be spaced and still be
            considered contiguous.
        kwargs
            kwargs are used to specify the dimension along which to chunk, eg:
            `time=10` chunks along the time axis in 10 second increments.
        """

    @abc.abstractmethod
    def select(self, **kwargs) -> Self:
        """Select only part of the data."""

    @abc.abstractmethod
    def get_contents(self) -> pd.DataFrame:
        """Get a dataframe of the patches that will be returned by the spool."""

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return len of spool."""

    def __eq__(self, other) -> bool:
        """Simple equality checks on spools."""

        def _vals_equal(dict1, dict2):
            if set(dict1) != set(dict2):
                return False
            for key in set(dict1):
                val1, val2 = dict1[key], dict2[key]
                if isinstance(val1, dict):
                    if not _vals_equal(val1, val2):
                        return False
                elif hasattr(val1, "equals"):
                    if not val1.equals(val2):
                        return False
                elif val1 != val2:
                    return False
            return True

        my_dict = self.__dict__
        other_dict = getattr(other, "__dict__", {})
        return _vals_equal(my_dict, other_dict)

    def __rich__(self):
        """Rich rep. of spool."""
        text = get_dascore_text() + Text(" ")
        text += Text(self.__class__.__name__, style=self._rich_style)
        text += Text(" 🧵 ")
        patch_len = len(self)
        text += Text(f"({patch_len:d}")
        text += Text(" Patches)") if patch_len != 1 else Text(" Patch)")
        return text

    def __str__(self):
        return str(self.__rich__())

    __repr__ = __str__


class DataFrameSpool(BaseSpool):
    """An abstract class for spools whose contents are managed by a dataframe."""

    # A dataframe which represents contents as they will be output
    _df: pd.DataFrame = CacheDescriptor("_cache", "_get_df")
    # A dataframe which shows patches in the source
    _source_df: pd.DataFrame = CacheDescriptor("_cache", "_get_source_df")
    # A dataframe of instructions for going from source_df to df
    _instruction_df: pd.DataFrame = CacheDescriptor("_cache", "_get_instruction_df")
    # kwargs for filtering contents
    _select_kwargs: Mapping | None = FrozenDict()
    # attributes which effect merge groups for internal patches
    _group_columns = ("network", "station", "dims", "data_type", "history", "tag")
    _drop_columns = ("patch",)

    def _get_df(self):
        """Function to get the current df."""

    def _get_source_df(self):
        """Function to get the current df."""

    def _get_instruction_df(self):
        """Function to get the current df."""

    def __init__(self):
        self._cache = {}

    def __getitem__(self, item):
        out = self._get_patches_from_index(item)
        # a single index was used, should return a single patch
        if not isinstance(item, slice):
            out = self._unbox_patch(out)
        # a slice was used, return a sub-spool
        else:
            out = self.__class__(out)
        return out

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        for ind in range(len(self._df)):
            yield self._unbox_patch(self._get_patches_from_index(ind))

    def _unbox_patch(self, patch_list):
        """Unbox a single patch from a patch list, check len."""
        assert len(patch_list) == 1
        return patch_list[0]

    def _get_patches_from_index(self, df_ind):
        """Given an index (from current df), return the corresponding patch."""
        source = self._source_df
        instruction = self._instruction_df
        if isinstance(df_ind, slice):  # handle slicing
            df1 = instruction.loc[instruction["current_index"].values[df_ind]]
        else:  # Filter instruction df to only include current index.
            # handle negative index.
            df_ind = df_ind if df_ind >= 0 else len(self._df) + df_ind
            inds = self._df.index[df_ind]
            df1 = instruction[instruction["current_index"] == inds]
        if df1.empty:
            msg = f"index of [{df_ind}] is out of bounds for spool."
            raise IndexError(msg)
        joined = df1.join(source.drop(columns=df1.columns, errors="ignore"))
        return self._patch_from_instruction_df(joined)

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        out = []
        df_dict_list = self._df_to_dict_list(joined)
        expected_len = len(joined["current_index"].unique())
        for patch_kwargs in df_dict_list:
            # convert kwargs to format understood by parser/patch.select
            kwargs = _convert_min_max_in_kwargs(patch_kwargs, joined)
            patch = self._load_patch(kwargs)
            # apply any trimming needed on patch
            select_kwargs = {
                i: v
                for i, v in kwargs.items()
                if i in patch.dims or i in patch.coords.coord_map
            }
            trimmed_patch: dc.Patch = patch.select(**select_kwargs)
            # its unfortunate, but currently we need to regenerate the patch
            # dict because the index doesn't carry all the dimensional info
            info = trimmed_patch.attrs.flat_dump(exclude=["history"])
            info["patch"] = trimmed_patch
            out.append(info)
        if len(out) > expected_len:
            out = _force_patch_merge(out)
        return [x["patch"] for x in out]

    @staticmethod
    def _get_dummy_dataframes(input_df):
        """
        Return dummy current, source, and instruction dataframes.

        Dummy because the source and current df are the same, so the
        instruction df is a straight mapping between the two.
        """
        output = input_df.copy(deep=False)
        dims = get_dim_names_from_columns(output)
        cols2keep = get_column_names_from_dim(dims)
        instruction = (
            input_df.copy()[cols2keep]
            .assign(source_index=output.index, current_index=output.index)
            .set_index("source_index")
            .sort_values("current_index")
        )
        return input_df, output, instruction

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        This is significantly faster than iterating rows.
        """
        df_dict_list = list(df.T.to_dict().values())
        return df_dict_list

    @abc.abstractmethod
    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""

    @compose_docstring(doc=BaseSpool.chunk.__doc__)
    def chunk(
        self,
        overlap: numeric_types | timeable_types | None = None,
        keep_partial: bool = False,
        snap_coords: bool = True,
        tolerance: float = 1.5,
        **kwargs,
    ) -> Self:
        """{doc}."""
        df = self._df.drop(columns=list(self._drop_columns), errors="ignore")
        chunker = ChunkManager(
            overlap=overlap,
            keep_partial=keep_partial,
            group_columns=self._group_columns,
            tolerance=tolerance,
            **kwargs,
        )
        in_df, out_df = chunker.chunk(df)
        if df.empty:
            instructions = None
        else:
            instructions = chunker.get_instruction_df(in_df, out_df)
        return self.new_from_df(out_df, source_df=self._df, instruction_df=instructions)

    def new_from_df(self, df, source_df=None, instruction_df=None):
        """Create a new instance from dataframes."""
        new = self.__class__(self)
        df_, source_, inst_ = self._get_dummy_dataframes(df)
        new._df = df
        new._source_df = source_df if source_df is not None else source_
        new._instruction_df = instruction_df if instruction_df is not None else inst_
        return new

    def select(self, **kwargs) -> Self:
        """Sub-select certain dimensions for Spool."""
        filtered_df = adjust_segments(self._df, ignore_bad_kwargs=True, **kwargs)
        inst = adjust_segments(
            self._instruction_df,
            ignore_bad_kwargs=True,
            **kwargs,
        ).loc[lambda x: x["current_index"].isin(filtered_df.index)]
        out = self.new_from_df(
            filtered_df,
            source_df=self._source_df,
            instruction_df=inst,
        )
        return out

    @compose_docstring(doc=BaseSpool.get_contents.__doc__)
    def get_contents(self) -> pd.DataFrame:
        """{doc}."""
        return self._df[filter_df(self._df, **self._select_kwargs)]


class MemorySpool(DataFrameSpool):
    """A Spool for storing patches in memory."""

    # tuple of attributes to remove from table

    def __init__(self, data: PatchType | Sequence[PatchType] | None = None):
        super().__init__()
        if data is not None:
            dfs = self._get_dummy_dataframes(patches_to_df(data))
            self._df, self._source_df, self._instruction_df = dfs

    def __rich__(self):
        base = super().__rich__()
        df = self._df
        t1, t2 = df["time_min"].min(), df["time_max"].max()
        tmin = get_nice_text(t1)
        tmax = get_nice_text(t2)
        duration = get_nice_text(t2 - t1)
        base += Text(f"\n    Time Span: <{duration}> {tmin} to {tmax}")
        return base

    def _load_patch(self, kwargs) -> Self:
        """Load the patch into memory."""
        return kwargs["patch"]


@singledispatch
def spool(obj: Path | str | BaseSpool | Sequence[PatchType], **kwargs) -> BaseSpool:
    """
    Load a spool from some data source.

    Parameters
    ----------
    obj
        An object from which a spool can be derived.
    """
    msg = f"Could not get spool from: {obj}"
    raise ValueError(msg)


@spool.register(str)
@spool.register(Path)
def spool_from_str(path, **kwargs):
    """Get a spool from a path."""
    path = Path(path)
    # A directory was passed, create Directory Spool
    if path.is_dir():
        from dascore.clients.dirspool import DirectorySpool

        return DirectorySpool(path, **kwargs)
    # A single file was passed. If the file format supports quick scanning
    # Return a FileSpool (lazy file reader), else return DirectorySpool.
    elif path.exists():  # a single file path was passed.
        _format, _version = dc.get_format(path)
        formatter = dascore.io.FiberIO.manager.get_fiberio(_format, _version)
        if formatter.implements_scan:
            from dascore.clients.filespool import FileSpool

            return FileSpool(path, _format, _version)

        else:
            return MemorySpool(dc.read(path, _format, _version))
    else:
        msg = (
            f"could not get spool from argument: {path}. "
            f"If it is a path, it may not exist."
        )
        raise InvalidSpoolError(msg)


@spool.register(BaseSpool)
def spool_from_spool(spool, **kwargs):
    """Return a spool from a spool."""
    return spool


@spool.register(list)
@spool.register(tuple)
def spool_from_patch_list(patch_list, **kwargs):
    """Return a spool from a sequence of patches."""
    return MemorySpool(patch_list)


@spool.register(dc.Patch)
def spool_from_patch(patch):
    """Get a spool from a single patch."""
    return MemorySpool([patch])
