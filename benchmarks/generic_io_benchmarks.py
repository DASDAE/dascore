"""Benchmark for generic memory spool operations."""

from __future__ import annotations

from functools import cache

import dascore as dc
from dascore.utils.downloader import fetch, get_registry_df


@cache
def test_file_paths():
    """Get a dict of name: path for all files in data registry."""
    df = get_registry_df().loc[lambda x: ~x["name"].str.endswith(".csv")]
    out = {row["name"]: fetch(row["name"]) for _, row in df.iterrows()}
    return out


class IoSuite:
    """Basic io functions."""

    def setup(self):
        """Get paths of test files."""
        self.path_dict = test_file_paths()

    def time_scan(self):
        """Time for basic scanning of all datafiles."""
        for path in self.path_dict.values():
            dc.scan(path)

    def time_scan_df(self):
        """Time for basic scanning of all datafiles."""
        for path in self.path_dict.values():
            dc.scan_to_df(path)

    def time_get_format(self):
        """Time for basic scanning of all datafiles."""
        for path in self.path_dict.values():
            dc.get_format(path)

    def time_read(self):
        """Time for basic scanning of all datafiles."""
        for path in self.path_dict.values():
            dc.read(path)[0]

    def time_spool(self):
        """Time for basic scanning of all datafiles."""
        for path in self.path_dict.values():
            dc.spool(path)[0]
