"""
A spool for working with file systems.

The spool uses a database index (sqlite by default) to track files.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
from rich.text import Text
from typing_extensions import Self

from dascore.compat import UPath
from dascore.constants import PROGRESS_LEVELS
from dascore.core.spool import BaseSpool, DataFrameSpool
from dascore.io.index.catalog import FileResolver, PatchCatalog
from dascore.io.indexer import AbstractIndexer
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
            self._catalog = PatchCatalog(
                backend=self.indexer._backend,
                resolver=FileResolver(root=self.indexer.path),
                syncer=self.indexer,
            )
        elif isinstance(base_path, Path | str | UPath):
            self._catalog = PatchCatalog.from_directory(
                base_path, index_path=index_path
            )
            self.indexer = self._catalog._syncer
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
        if not self._select_kwargs:
            return self._source_df
        # constructor select_kwargs restrict contents (docstring contract)
        return adjust_segments(
            self._source_df, ignore_bad_kwargs=True, **self._select_kwargs
        )

    def _get_instruction_df(self):
        """Return instruction df on how to get from source_df to df."""
        _, _, instruction = self._get_dummy_dataframes(self._df)
        return instruction

    def _get_source_df(self):
        """Return a dataframe of sources in spool."""
        return self._catalog.to_df().reset_index(drop=True)

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
        self._catalog.update(progress=progress)
        return self._new_from_catalog(self._catalog)

    def _df_to_dict_list(self, df):
        """
        Convert the dataframe to a list of dicts for iteration.

        Stored (relative) paths pass through unchanged; the catalog's
        FileResolver owns resolving them against the spool root, so path
        resolution lives in exactly one place.
        """
        df = df.copy(deep=False).replace("", None)
        return super()._df_to_dict_list(df)

    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""
        # Push trims into the reader only when the instruction row narrows
        # the source (chunk/select) or constructor select_kwargs restrict
        # it; otherwise the whole file is wanted and selection is wasted.
        trim = {}
        if kwargs.get("_modified") or self._select_kwargs:
            merged = {**kwargs, **self._select_kwargs}
            trim = {
                k: v
                for k, v in merged.items()
                if k not in self._drop_columns and not k.startswith("_")
            }
        return self._catalog.resolve_row(kwargs, extra_trim=trim)
