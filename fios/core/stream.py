"""
A module for storing streams of fiber data.
"""
from collections import UserList
from typing import Sequence, Union

import pandas as pd

import fios
from fios.constants import PatchType
from fios.io.base import scan_patches


class Stream:
    """
    A stream of fiber data.
    """
    # a tuple of attrs that must be compatible for patches to be merged
    _merge_attrs = ('network', 'station', 'dims', 'data_type', 'category')
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

    def _get_patch_table(self, patch_iterable: Sequence[PatchType]) -> pd.DataFrame:
        """
        Create a table with metadata about patches.
        """
        if isinstance(patch_iterable, fios.Patch):
            patch_iterable = [patch_iterable]
        df = pd.DataFrame(scan_patches(patch_iterable))
        df["patch"] = patch_iterable
        return df

    def merge(self, dim='time'):
        """
        Merge all compatible patches in stream together.

        Parameters
        ----------
        dimension along which to try to merge.

        """
        min_name, max_name = f"{dim}_min", f"{dim}_max"
        sort_names = list(self._merge_attrs) + [min_name, max_name]
        # make a shallow copy for shorting, merging
        df = self._df.sort_values(sort_names)
        # first get groups of compatible traces
        breakpoint()

