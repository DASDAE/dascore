"""
A module for storing streams of fiber data.
"""
import abc
from typing import Optional, Sequence, Union

import pandas as pd

import dascore
from dascore.constants import PatchType, SpoolType
from dascore.utils.docs import compose_docstring
from dascore.utils.patch import merge_patches, scan_patches
from dascore.utils.pd import filter_df


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

    def update(self: SpoolType) -> SpoolType:
        """
        Updates the contents of the spool and returns a spool.
        """
        return self

    @abc.abstractmethod
    def chunk(self: SpoolType, **kwargs) -> SpoolType:
        """
        Chunk the data in the spool along specified dimensions.
        """

    @abc.abstractmethod
    def select(self: SpoolType, **kwargs) -> SpoolType:
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

    _df: Optional[pd.DataFrame] = None

    @abc.abstractmethod
    def _extract_patch_from_row(self, row) -> PatchType:
        """Given a row from the managed dataframe, return a patch."""

    def load_patch(self, row):
        """
        Load a patch from a row of the dataframe.
        """
        patch = self._extract_patch_from_row(row)
        funcs = row["patch_funcs"]
        for func in funcs:
            patch = func(patch)
        return patch


class MemorySpool(DataFrameSpool):
    """
    A Spool for storing patches in memory.
    """

    # a tuple of attrs that must be compatible for patches to be merged
    _merge_attrs = ("network", "station", "dims", "data_type", "category")

    # tuple of attributes to remove from table

    def __init__(self, data: Union[PatchType, Sequence[PatchType]]):
        self._df = self._get_patch_table(data)

    def __getitem__(self, item):
        return self.load_patch(self._df.iloc[0])

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        for ind in range(len(self._df)):
            yield self.load_patch(self._df.iloc[ind])

    def chunk(self, **kwargs):
        """chunk the contents of the spool."""

    def _get_patch_table(self, patch_iterable: Sequence[PatchType]) -> pd.DataFrame:
        """
        Create a table with metadata about patches.
        """
        if isinstance(patch_iterable, dascore.Patch):
            patch_iterable = [patch_iterable]
        df = pd.DataFrame(scan_patches(patch_iterable))
        df["patch"] = patch_iterable
        df["patch_funcs"] = [[] for _ in range(len(df))]
        return df

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

    def _extract_patch_from_row(self, row):
        """Load the patch into memory"""
        return row["patch"]

    def select(self, **kwargs):
        """Sub-select certain dimensions for Spool"""

    @compose_docstring(doc=BaseSpool.get_contents.__doc__)
    def get_contents(self, **kwargs) -> pd.DataFrame:
        """
        {doc}
        """
        return self._df[filter_df(self._df, **kwargs)]
