"""
A module for storing streams of fiber data.
"""
import abc
from typing import Mapping, Optional, Sequence, Union

import pandas as pd
from typing_extensions import Self

from dascore.constants import PatchType, SpoolType, numeric_types, timeable_types
from dascore.utils.chunk import ChunkManager
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.patch import merge_patches, patches_to_df
from dascore.utils.pd import (
    _convert_min_max_in_kwargs,
    adjust_segments,
    filter_df,
    get_column_names_from_dim,
    get_dim_names_from_columns,
)


class BaseSpool(abc.ABC):
    """
    Spool Abstract Base Class (ABC) for defining Spool interface.
    """

    @abc.abstractmethod
    def __getitem__(self, item: int) -> PatchType:
        """Returns a patch from the spool."""

    @abc.abstractmethod
    def __iter__(self) -> PatchType:
        """Iterate through the Patches in the spool."""

    def update(self: SpoolType) -> Self:
        """
        Updates the contents of the spool and returns a spool.
        """
        return self

    @abc.abstractmethod
    def chunk(
        self: SpoolType,
        overlap: Optional[Union[numeric_types, timeable_types]] = None,
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
            The amount of overlap for each chunk
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
    def select(self: SpoolType, **kwargs) -> Self:
        """
        Select only part of the data.
        """

    @abc.abstractmethod
    def get_contents(self: SpoolType, **kwargs) -> pd.DataFrame:
        """
        Get a dataframe of the patches in spool.

        Can be filtered using kwargs. For example:
            get_contents(time=('2012-01-01', None))
        Will only return a dataframe of patches which contain data after
        the specified minimum value for time.
        """


class DataFrameSpool(BaseSpool):
    """A spool whose contents are managed by a dataframe."""

    # A dataframe which represents contents as they will be output
    _df: Optional[pd.DataFrame] = None
    # A dataframe which shows patches in the source
    _source_df: Optional[pd.DataFrame] = None
    # A dataframe of instructions for going from source_df to df
    _instruction_df: Optional[pd.DataFrame] = None
    # kwargs for filtering contents
    _select_kwargs: Optional[Mapping] = FrozenDict()
    # attributes which effect merge groups for internal patches
    _group_columns = ("network", "station", "dims", "data_type", "history")
    _drop_columns = ("patch",)

    def __getitem__(self, item):
        return self._get_patches_from_index(item)

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        for ind in self._df.index:
            yield self._get_patches_from_index(ind)

    def _get_patches_from_index(self, df_ind):
        """
        Given an index (from current df), return the corresponding patch.
        """
        source = self._source_df
        instruction = self._instruction_df
        df1 = instruction[instruction["current_index"] == df_ind]
        if df1.empty:
            msg = f"index of [{df_ind}] is out of bounds for spool."
            raise IndexError(msg)
        joined = df1.join(source.drop(columns=df1.columns, errors="ignore"))
        return self._patch_from_instruction_df(joined)

    def _patch_from_instruction_df(self, joined):
        """Get the patches joined columns of instruction df."""
        df_dict_list = self._df_to_dict_list(joined)
        out_list = []
        for patch_kwargs in df_dict_list:
            # convert kwargs to format understood by parser/patch.select
            kwargs = _convert_min_max_in_kwargs(patch_kwargs, joined)
            patch = self._load_patch(kwargs)
            select_kwargs = {
                i: v for i, v in kwargs.items() if i in patch.dims or i in patch.coords
            }
            out_list.append(patch.select(**select_kwargs))
        if len(out_list) > 1:
            merged = merge_patches(out_list)
            assert len(merged) == 1, "failed to merge patches"
            out = merged[0]
        else:
            out = out_list[0]
        return out

    def _get_dataframes(self, input_df):
        """Return dummy current, source, and instruction dataframes."""
        output = input_df.copy(deep=False)
        dims = get_dim_names_from_columns(output)
        cols2keep = get_column_names_from_dim(dims)
        instruction = (
            input_df.copy()[cols2keep]
            .assign(source_index=output.index)
            .assign(current_index=output.index)
            .set_index("source_index")
            .sort_values("current_index")
        )
        return input_df, output, instruction

    def _get_source_patches(self, row):
        if self._instruction_df is not None:
            pass

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
        self: SpoolType,
        overlap: Optional[Union[numeric_types, timeable_types]] = None,
        keep_partial: bool = False,
        **kwargs,
    ) -> Self:
        """
        {doc}
        """
        df = self._df.drop(columns=list(self._drop_columns), errors="ignore")
        chunker = ChunkManager(
            overlap=overlap,
            keep_partial=keep_partial,
            group_columns=self._group_columns,
            **kwargs,
        )
        out = chunker.chunk(df)
        instructions = chunker.get_instruction_df(df, out)
        return self.new_from_df(out, source_df=self._df, instruction_df=instructions)

    @classmethod
    def new_from_df(cls, df, source_df=None, instruction_df=None, select_kwargs=None):
        """Create a new instance from dataframes."""
        new = cls()
        new._df = df
        new._source_df = source_df if source_df is not None else df
        new._instruction_df = instruction_df
        if select_kwargs is not None:
            new._select_kwargs = select_kwargs
        return new

    def select(self, **kwargs) -> Self:
        """Sub-select certain dimensions for Spool"""
        out = self.new_from_df(
            adjust_segments(self._df, **kwargs),
            source_df=self._source_df,
            instruction_df=adjust_segments(self._instruction_df, **kwargs),
            select_kwargs=kwargs,
        )
        return out

    @compose_docstring(doc=BaseSpool.get_contents.__doc__)
    def get_contents(self) -> pd.DataFrame:
        """
        {doc}
        """
        return self._df[filter_df(self._df, **self._select_kwargs)]


class MemorySpool(DataFrameSpool):
    """
    A Spool for storing patches in memory.
    """

    # tuple of attributes to remove from table

    def __init__(self, data: Optional[Union[PatchType, Sequence[PatchType]]] = None):
        if data is not None:
            dfs = self._get_dataframes(patches_to_df(data))
            self._df, self._source_df, self._instruction_df = dfs

    def merge(self, dim="time"):
        """
        Merge all compatible patches in stream together.

        Parameters
        ----------
        dim
            dimension along which to try to merge.

        See also :func:`dascore.utils.patch.merge_patches`
        """
        new_patches = merge_patches(self._df, dim=dim)
        return self.__class__(new_patches)

    def _load_patch(self, kwargs) -> Self:
        """Load the patch into memory"""
        return kwargs["patch"]

    @classmethod
    def new_from_df(cls, df, source_df=None, instruction_df=None, select_kwargs=None):
        """Create a new instance from dataframes."""
        # iterating the patches forces the trim/select/merging to occur
        patches = [x for x in super().new_from_df(df, source_df, instruction_df)]
        return cls(patches)
