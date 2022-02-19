"""
A module for storing streams of fiber data.
"""
import abc
from typing import Sequence, Union

import pandas as pd

import dascore
from dascore.constants import PatchType, SpoolType
from dascore.utils.patch import merge_patches, scan_patches


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

    def get_contents(self: SpoolType) -> pd.DataFrame:
        """Get the contents from the data"""


class MemorySpool(BaseSpool):
    """
    A Spool for storing patches in memory.
    """

    # a tuple of attrs that must be compatible for patches to be merged
    _merge_attrs = ("network", "station", "dims", "data_type", "category")

    # tuple of attributes to remove from table

    def __init__(self, data: Union[PatchType, Sequence[PatchType]]):
        self._df = self._get_patch_table(data)

    def __getitem__(self, item):
        return self._df.iloc[0]["patch"]

    def __len__(self):
        return len(self._df)

    def __iter__(self):
        patches = self._df["patch"].values
        return iter(patches)

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
        return df

    def merge(self, dim="time"):
        """
        Merge all compatible patches in stream together.

        Parameters
        ----------
        dimension along which to try to merge.

        """
        new_patches = merge_patches(self._df, dim=dim)
        return self.__class__(new_patches)

    def select(self):
        """Sub-select certain dimensions for Spool"""
