"""
A spool for working with a single file.
"""

from pathlib import Path
from typing import Union

from typing_extensions import Self

import dascore as dc
from dascore.core.spool import DataFrameSpool


class FileSpool(DataFrameSpool):
    """
    A spool for a single file.

    Some file formats support storing multiple patches, this is most useful
    for those formats, but should work on all dascore supported formats.
    """

    def __init__(self, path: Union[str, Path], file_format=None, file_version=None):
        """"""
        super().__init__()
        self._path = Path(path)
        if not self._path.exists() or self._path.is_dir():
            msg = f"{path} does not exist or is a directory"
            raise FileNotFoundError(msg)
        source_df = dc.scan_to_df(
            path, file_format=file_format, file_version=file_version
        )
        dfs = self._get_dummy_dataframes(source_df)
        self._df, self._source_df, self._instruction_df = dfs

    def __str__(self):
        """Returns a (hopefully) useful string rep of spool."""
        out = f"FileSpool object managing {self._path}"
        return out

    __repr__ = __str__

    def _load_patch(self, kwargs) -> Self:
        """Given a row from the managed dataframe, return a patch."""
        return dc.read(**kwargs)[0]
