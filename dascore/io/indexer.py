"""An HDF5-based indexer for local file systems."""

from __future__ import annotations

import abc
import json
import os
import time
import warnings
from contextlib import suppress
from functools import cache
from pathlib import Path

import pandas as pd
import pooch
from typing_extensions import Self

import dascore as dc
from dascore.constants import ONE_SECOND_IN_NS, PROGRESS_LEVELS
from dascore.exceptions import InvalidIndexVersionError
from dascore.utils.hdf5 import HDFPatchIndexManager
from dascore.utils.misc import iterate
from dascore.utils.pd import filter_df
from dascore.utils.time import get_max_min_times, to_timedelta64

# supported read_hdf5 kwargs
READ_HDF5_KWARGS = frozenset(
    {"columns", "where", "mode", "errors", "start", "stop", "key", "chunksize"}
)


@cache
def _get_index_map(cache_path) -> dict:
    """
    Get a dict of index locations.

    Note: this is purposefully mutable; handle with care.
    """
    path = Path(cache_path)
    out = {}
    successful_read = True
    if path.exists():
        try:
            with path.open("r") as fi:
                out = json.load(fi)
        # On rare occasions, the file can become corrupt. See #508.
        except (OSError, json.JSONDecodeError):
            successful_read = False
    if not isinstance(out, dict) or not successful_read:
        out = {}
        with suppress(FileNotFoundError, PermissionError):
            path.unlink(missing_ok=True)
    return out


def _update_index_map(updates, cache_path) -> dict:
    """Update index map to track new index."""
    data = _get_index_map(cache_path=cache_path)
    data.update(updates)
    Path(cache_path).parent.mkdir(exist_ok=True, parents=True)
    with open(cache_path, "w") as fi:
        json.dump(data, fi)
    return data


def _directory_writable(path):
    """Return True if the directory is writable else False."""
    name = "._dascore_write_test_delete_me"
    path = Path(path) / name
    path.parent.mkdir(exist_ok=True, parents=True)
    try:
        open(path, "w").close()
    except (PermissionError, IsADirectoryError):
        return False
    else:
        os.remove(path)
    return True


class AbstractIndexer:
    """
    A base class for indexers.

    This is primarily here for a place-holder.
    """

    path: Path

    @abc.abstractmethod
    def update(self) -> Self:
        """
        Updates the contents of the Indexer.

        Resets any previous selection.
        """


class DirectoryIndexer(AbstractIndexer):
    """
    A class for indexing a directory of dascore-readable files.

    This works by crawling the directory, getting a summary about the data it
    contains, then creating a small HDF index file which can be queried later
    on.

    Parameters
    ----------
    path
        The path to a directory containing DAS files.
    index_path
        The path to the index. By default, the index will be created on the
        top level of the data directory. If another index is
    """

    ext = ""
    _namespace = ""
    _index_name = ".dascore_index.h5"  # name of index file

    # A user-specific file which tracks where indices are stored if not on
    # the same directory as the path to index.
    index_map_path = pooch.os_cache("dascore") / "caches" / "cache_paths.json"

    def __init__(self, path: str | Path, cache_size: int = 5, index_path=None):
        self.max_size = cache_size
        self.path = Path(path).absolute()
        self.index_path = Path(self._find_index_file(self.path, index_path))
        self._current_index = 0
        self._index_table = HDFPatchIndexManager(
            self.index_path,
            self._namespace,
        )
        self.cache = pd.DataFrame(
            index=range(cache_size), columns="t1 t2 kwargs cindex".split()
        )

    def _find_index_file(self, data_path, index_path=None):
        """Find the path to the index file."""
        data_path = Path(data_path).absolute()
        # user specified index path
        if index_path:
            update = {str(data_path): str(Path(index_path).absolute())}
            _update_index_map(update, cache_path=str(self.index_map_path))
            return index_path
        # see if expected path is in data path
        expected_path = data_path / self._index_name
        with suppress(PermissionError):
            if expected_path.exists():
                return expected_path
        # else load path map and see if it knows where the index is.
        path_map = _get_index_map(cache_path=str(self.index_map_path))
        if out := path_map.get(str(data_path)):
            return out
        # if not, set the path to either the data path, if writable,
        # else the dascore cache
        if not _directory_writable(data_path):
            new_path = "_dascore_index_" + str(abs(hash(data_path))) + ".h5"
            index_path = self.index_map_path.parent / new_path
            update = {str(data_path): str(index_path.absolute())}
            _update_index_map(update, cache_path=str(self.index_map_path))
        else:
            index_path = data_path / self._index_name
        return index_path

    def get_contents(self, buffer=ONE_SECOND_IN_NS, **kwargs) -> pd.DataFrame:
        """
        Get contents of directory with specific query params.

        Parameters
        ----------
        buffer
            A buffer to ensure enough info is returned from hdf index.
        kwargs
            Used to query contents.
        """
        # create index if it doesn't exist
        if not self.index_path.exists():
            self.update()
        # if the index still doesn't exist there are no readable files, return
        # empty df.
        if not self.index_path.exists():
            return pd.DataFrame(columns=self._index_table.index_columns)
        time_min, time_max = get_max_min_times(kwargs.pop("time", None))
        hdf5_kwargs, kwargs = self._separate_hdf5_kwargs(kwargs)
        buffer = to_timedelta64(buffer)
        # find out if the query falls within one cached times
        con1 = self.cache.t1 <= time_min
        con2 = self.cache.t2 >= time_max
        con3 = self.cache.kwargs == self._kwargs_to_str(kwargs)
        cached_index = self.cache[con1 & con2 & con3]
        if not len(cached_index):  # query is not cached get it from hdf5 file
            index = self._index_table.get_index(
                time_min=time_min,
                time_max=time_max,
                **hdf5_kwargs,
            )
            self._set_cache(index, time_min, time_max, hdf5_kwargs)
        else:
            index = cached_index.iloc[0]["cindex"]
        # trim down index
        con1 = index["time_min"] >= (time_max + buffer)
        con2 = index["time_max"] <= (time_min - buffer)
        pre_filter_df = index[~(con1 | con2)]
        out = pre_filter_df[
            filter_df(
                pre_filter_df,
                time=(time_min, time_max),
                ignore_bad_kwargs=True,
                **kwargs,
            )
        ]
        return out

    def __str__(self):
        """Rep. indexer as a string."""
        msg = f"{self.__class__.__name__} managing: {self.path}"
        return msg

    __repr__ = __str__

    __call__ = get_contents

    def _separate_hdf5_kwargs(self, kwargs):
        """Ensure kwargs are supported."""
        kdf_kwargs = {i: v for i, v in kwargs.items() if i in READ_HDF5_KWARGS}
        kwargs = {i: v for i, v in kwargs.items() if i not in READ_HDF5_KWARGS}
        return kdf_kwargs, kwargs

    def _set_cache(self, index, starttime, endtime, kwargs):
        """Cache the current index."""
        ser = pd.Series(
            {
                "t1": starttime,
                "t2": endtime,
                "cindex": index,
                "kwargs": self._kwargs_to_str(kwargs),
            }
        )
        self.cache.loc[self._get_next_index()] = ser

    def clear_cache(self):
        """Removes all cached dataframes."""
        self.cache = pd.DataFrame(
            index=range(self.max_size), columns="t1 t2 kwargs cindex".split()
        )

    def _get_next_index(self):
        """
        Get the next index value on cache.
        Note we can't use itertools.cycle here because it cant be pickled.
        """
        if self._current_index == len(self.cache.index) - 1:
            self._current_index = 0
        else:
            self._current_index += 1
        return self.cache.index[self._current_index]

    def _kwargs_to_str(self, kwargs):
        """Convert kwargs to a string."""
        keys = sorted(list(kwargs.keys()))
        out = str([(item, kwargs[item]) for item in keys])
        return out

    def _get_mtime(self, only_new=True):
        """Return an iterator of potential un-indexed files."""
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        # getting last updated might need the db so only call once.
        last_updated = self._index_table.last_updated_timestamp if only_new else None
        if last_updated is not None and only_new:
            mtime = last_updated - 0.001
        # get paths to iterate
        return mtime

    def _get_paths(self, paths):
        path = self.path
        if paths is None:
            paths = path
        else:
            paths = [
                f"{path}/{x}" if str(path) not in str(x) else str(x)
                for x in iterate(paths)
            ]
        return paths

    def _enforce_min_version(self):
        """Ensure the minimum version is met, else delete index file."""
        try:
            self._index_table.validate_version()
        except InvalidIndexVersionError:
            msg = (
                f"The index file at {self.path} is not compatible with this"
                f" version of DASCore ({dc.__last_version__}). "
                f"Recreating the index now."
            )
            warnings.warn(msg, UserWarning)
            os.remove(self.index_path)
            self.update()

    def get_index_metadata(self):
        """Return a dict of metadata about the index."""
        self.update()
        up_time = dc.to_datetime64(self._index_table.last_updated_timestamp)
        out = {
            "index_version": self._index_table._index_version,
            "last_update": up_time,
        }
        return out

    def update(self, paths=None, progress: PROGRESS_LEVELS = "standard") -> Self:
        """
        Updates the contents of the Indexer.

        Also resets any previous selection.

        Parameters
        ----------
        paths
            A sequence of paths to limit the updates, if None, index all
            the contents of directory.
        progress
            The type of progress bar to use. None disables progress bar and
            "basic" is best for low latency scenarios.
        """
        self._enforce_min_version()  # delete index if schema has changed
        update_time = time.time()
        timestamp = self._get_mtime(only_new=True)
        paths = self._get_paths(paths)
        df = dc.scan_to_df(
            path=paths,
            timestamp=timestamp,
            progress=progress,
            ext=self.ext,
        )
        # Put contents found into database.
        if not df.empty:
            # Some users were surprised the spool wasn't sorted. We still cant
            # guarantee all spools will be sorted but we can make sure most are
            # by sorting the contents before dumping to index.
            if "time_min" in df.columns:
                df = df.sort_values("time_min").reset_index(drop=True)
            # ensure the base path is not in the path column
            assert "path" in set(df.columns), f"{df} has no path column"
            self._index_table.write_update(df, update_time, base_path=self.path)
            # clear cache out when new traces are added
            self.clear_cache()
        return self
