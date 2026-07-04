"""
A spool for working with file systems.

The spool uses a simple hdf5 index for keeping track of files.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.compat import UPath
from dascore.constants import PROGRESS_LEVELS
from dascore.core.spool import BaseSpool, DataFrameSpool, MemorySpool
from dascore.exceptions import MissingPatchError
from dascore.io.indexer import AbstractIndexer, DirectoryIndexer
from dascore.utils.docs import compose_docstring
from dascore.utils.pd import adjust_segments


class DirectorySpool(DataFrameSpool):
    """
    A spool for interacting with DAS files on disk.

    FileSpool creates and index of all files then allows for simple querying
    and bulk processing of the files.

    Parameters
    ----------
    base_path
        The path to the directory to index.
    index_path
        The path to the index file containing the contents of the directory.
        By default it will be created in the top-level of the data directory.
    preferred_format
        A string to specify the format of the data. Specifying this parameter
        will save time in indexing.
    select_kwargs
        Dict of keyword arguments to restrict output contents.
    """

    _drop_columns = ("file_format", "file_version", "path", "source_patch_id")

    def __init__(
        self,
        base_path: str | Path | UPath | Self | AbstractIndexer = ".",
        *,
        index_path: Path | None = None,
        preferred_format: str | None = None,
        select_kwargs: dict | None = None,
        merge_kwargs: dict | None = None,
    ):
        super().__init__(select_kwargs=select_kwargs, merge_kwargs=merge_kwargs)
        # Init file spool from another file spool
        if isinstance(base_path, self.__class__):
            self.__dict__.update(copy.deepcopy(base_path.__dict__))
            return
        # Init file spool from indexer
        elif isinstance(base_path, AbstractIndexer):
            self.indexer = base_path
        elif isinstance(base_path, Path | str | UPath):
            self.indexer = DirectoryIndexer(base_path, index_path=index_path)
        assert hasattr(self, "indexer"), "indexer not set."
        self._preferred_format = preferred_format

    def __rich__(self):
        """Augment rich string directory spool stuff."""
        base = super().__rich__()
        path = self.indexer.path
        kwargs = self._select_kwargs
        out = base + Text(f"\n    Path: {path}")
        out += Text(f"\n    Select kwargs: {kwargs}") if kwargs else Text("")
        return out

    def _get_df(self):
        """Get the dataframe of current contents."""
        out = adjust_segments(
            self._source_df, ignore_bad_kwargs=True, **self._select_kwargs
        )
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

    @compose_docstring(doc=BaseSpool.get_contents.__doc__)
    def get_contents(self) -> pd.DataFrame:
        """{doc}."""
        return self._df

    @compose_docstring(doc=BaseSpool.update.__doc__)
    def update(self, progress: PROGRESS_LEVELS = "standard") -> Self:
        """{doc}."""
        out = self.__class__(
            base_path=self.indexer.update(progress=progress),
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
        # note: need to add extra / here since we no longer store it in db.
        df["path"] = (str(self.spool_path) + "/") + df["path"]
        return super()._df_to_dict_list(df)

    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""
        final_kwargs = dict(kwargs)
        final_kwargs.update(self._select_kwargs)
        patches = self._read_patches(final_kwargs)
        if patches is None:  # fast path doesn't apply, use generic read.
            return self._read_and_resolve_patch(final_kwargs)
        if not patches:
            # Iteration skips these with a warning, see #583.
            msg = (
                f"No patch in {final_kwargs.get('path')} matches the "
                f"requested range; it may have been trimmed to nothing."
            )
            raise MissingPatchError(msg)
        return patches[0]

    def _read_patches(self, kwargs) -> list[dc.Patch] | None:
        """
        Read patches directly through the file's FiberIO.

        This skips the format detection of dc.read and the spool indexing
        machinery applied to its output, which add up when loading many
        files. Returns None when the fast path can't be safely used.
        """
        fmt, version = kwargs.get("file_format"), kwargs.get("file_version")
        if not fmt or not version:
            # Without a concrete version get_fiberio would return the newest
            # reader; let dc.read detect the file's actual version instead.
            return None
        fiber_io = dc.io.FiberIO.manager.get_fiberio(format=fmt, version=version)
        # Only apply select kwargs when the source patch is trimmed by the
        # instruction df or the spool itself; otherwise the whole file is
        # wanted and selection is wasted work.
        if kwargs.get("_modified") or self._select_kwargs:
            select = {
                k: v
                for k, v in kwargs.items()
                if k not in self._drop_columns and not k.startswith("_")
            }
        else:
            select = {}
        spool = fiber_io.read(kwargs["path"], **select)
        if not isinstance(spool, MemorySpool):
            return None
        patches = list(spool)
        # Without selection, a multi-patch file is ambiguous: the row refers
        # to one specific patch. Let the generic path pick the right one.
        if len(patches) > 1 and not select:
            return None
        return patches
