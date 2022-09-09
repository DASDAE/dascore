"""
A spool for working with file systems.

The spool uses a simple hdf5 index for keeping track of files.
"""
import copy
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from typing_extensions import Self

import dascore as dc
from dascore.core.spool import DataFrameSpool
from dascore.io.indexer import AbstractIndexer, DirectoryIndexer
from dascore.utils.pd import adjust_segments


class DirectorySpool(DataFrameSpool):
    """
    A spool for interacting with DAS files on disk.

    FileSpool creates and index of all files then allows for simple querying
    and bulk processing of the files.
    """

    _drop_columns = ("file_format", "file_version", "path")

    def __init__(
        self,
        base_path: Union[str, Path, Self, AbstractIndexer] = ".",
        preferred_format: Optional[str] = None,
        select_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        # Init file spool from another file spool
        if isinstance(base_path, self.__class__):
            self.__dict__.update(copy.deepcopy(base_path.__dict__))
            return
        # Init file spool from indexer
        elif isinstance(base_path, AbstractIndexer):
            self.indexer = base_path
        elif isinstance(base_path, (Path, str)):
            self.indexer = DirectoryIndexer(base_path)
        self._preferred_format = preferred_format
        self._select_kwargs = {} if select_kwargs is None else select_kwargs

    def __str__(self):
        out = (
            f"FileSpool object managing: {self.spool_path}"
            f" with select kwargs: {self._select_kwargs}"
        )
        return out

    __repr__ = __str__

    def _get_df(self):
        """Get the dataframe of current contents."""
        out = adjust_segments(self._source_df, **self._select_kwargs)
        return out

    def _get_instruction_df(self):
        """Return instruction df on how to get from source_df to df."""
        _, _, instruction = self._get_dummy_dataframes(self._df)
        return instruction

    def _get_source_df(self):
        """Return a dataframe of sources in spool."""
        return self.indexer(**self._select_kwargs).reset_index(drop=True)

    @property
    def spool_path(self):
        """Return the path in which the spool contents are found."""
        return self.indexer.path

    def get_contents(self) -> pd.DataFrame:
        """
        Return a dataframe of the contents of the data files.

        Parameters
        ----------
        time
            If not None, a tuple of start/end time where either can be None
            indicating an open interval.
        """
        return self._df

    def select(self, **kwargs) -> Self:
        """Sub-select certain dimensions for Spool"""
        new_kwargs = dict(self._select_kwargs)
        new_kwargs.update(kwargs)
        out = self.__class__(
            base_path=self.indexer,
            preferred_format=self._preferred_format,
            select_kwargs=new_kwargs,
        )
        return out

    def update(self) -> Self:
        """
        Updates the contents of the spool and returns a spool.

        Resets any previous selection.
        """
        out = self.__class__(
            base_path=self.indexer.update(),
            preferred_format=self._preferred_format,
            select_kwargs=self._select_kwargs,
        )
        return out

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        This is significantly faster than iterating rows.
        """
        df = df.copy(deep=False).replace("", None)
        df["path"] = str(self.spool_path) + df["path"]
        return super()._df_to_dict_list(df)

    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""
        patch = dc.read(**kwargs)[0]
        return patch
