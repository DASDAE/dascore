"""
Class for interacting with data sources.

The data source is typically only used indirectly by the
:class:`dascore.Spool` object.
"""
import abc
from typing import Sequence, Union

import pandas as pd

import dascore
from dascore.constants import PatchType, SpoolType
from dascore.utils.patch import scan_patches


class BaseDataSource(abc.ABC):
    """Abstract datasource."""

    def update(self) -> "BaseDataSource":
        """Update the data in the data source."""
        return self

    @abc.abstractmethod
    def get_df(self, **kwargs) -> pd.DataFrame:
        """
        Return a dataframe of contents, which can be filtered based on kwargs.

        The standard dim=(start, stop) is used for filtering.
        """

    def yield_patches(self, df):
        """
        Yield patches from sources for specs in dataframe.
        """


class InMemoryDataSource(BaseDataSource):
    """
    A datasource of in-memory patches.
    """

    def __init__(self, patch_or_traces: Union[PatchType, SpoolType]):
        self._source = self._get_patch_table(patch_or_traces)

    def get_df(self, **kwargs) -> pd.DataFrame:
        """
        Simply return pre-computed df
        """
        return self._source

    def _get_patch_table(self, patch_iterable: Sequence[PatchType]) -> pd.DataFrame:
        """
        Create a table with metadata about patches.
        """
        if isinstance(patch_iterable, dascore.Patch):
            patch_iterable = [patch_iterable]
        df = pd.DataFrame(scan_patches(patch_iterable))
        df["patch"] = patch_iterable
        return df
