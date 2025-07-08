"""A spool for working with a single file."""

from __future__ import annotations

import copy
from pathlib import Path

from rich.text import Text
from typing_extensions import Self

import dascore as dc
from dascore.constants import PROGRESS_LEVELS, SpoolType
from dascore.core.spool import BaseSpool, DataFrameSpool
from dascore.io.core import FiberIO
from dascore.utils.docs import compose_docstring


class FileSpool(DataFrameSpool):
    """
    A spool for a single file.

    Parameters
    ----------
    path
        The path to the file.
    file_format
        The format name, optional.
    file_version
        The version string of the format, optional.

    Notes
    -----
    Some file formats support storing multiple patches, this is most useful
    for those formats, but should work on all dascore supported formats.
    """

    def __init__(
        self,
        path: str | Path,
        file_format: str | None = None,
        file_version: str | None = None,
    ):
        super().__init__()
        # Init file spool from another file spool
        if isinstance(path, self.__class__):
            self.__dict__.update(copy.deepcopy(path.__dict__))
            return
        self._path = Path(path)
        if not self._path.exists() or self._path.is_dir():
            msg = f"{path} does not exist or is a directory"
            raise FileNotFoundError(msg)

        _format, _version = dc.get_format(path, file_format, file_version)
        source_df = dc.scan_to_df(path, file_format=_format, file_version=_version)
        dfs = self._get_dummy_dataframes(source_df)
        self._df, self._source_df, self._instruction_df = dfs
        self._file_format = _format
        self._file_version = _version

    def __rich__(self):
        """Augment rich string with path."""
        base = super().__rich__()
        out = base + Text(f" Path: {self._path}")
        return out

    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""
        return dc.read(**kwargs)[0]

    @compose_docstring(doc=BaseSpool.update.__doc__)
    def update(self: SpoolType, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        {doc}.

        Note: If the file format supports indexing (e.g. DASDAE) this will
        trigger an indexing of the file.
        """
        formater = FiberIO.manager.get_fiberio(
            format=self._file_format, version=self._file_version
        )
        getattr(formater, "index", lambda x: None)(self._path)
        return self
