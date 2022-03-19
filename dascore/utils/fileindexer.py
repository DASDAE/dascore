"""
Utilities for creating an index of directories of files.
"""
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional

import numpy as np
import pandas as pd
from tables.exceptions import ClosedNodeError

from dascore.constants import LARGEDT64, SMALLDT64, path_types
from dascore.exceptions import UnsupportedKeyword
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import iter_files
from dascore.utils.time import to_datetime64

# supported read_hdf5 kwargs
READ_HDF5_KWARGS = frozenset(
    {"columns", "where", "mode", "errors", "start", "stop", "key", "chunksize"}
)


def iterate(obj):
    """
    Return an iterable from any object.

    If string, do not iterate characters, return str in tuple .
    """
    if obj is None:
        return ()
    if isinstance(obj, str):
        return (obj,)
    return obj if isinstance(obj, Iterable) else (obj,)


def _get_kernel_query(starttime: int, endtime: int, buffer: int):
    """
    Create a HDF5 kernel query based on start and end times.

    This is necessary because hdf5 doesnt accept inverted conditions.
    A slight buffer is applied to the ranges to make sure no edge files
    are excluded.
    """
    t1 = starttime - buffer
    t2 = endtime + buffer
    con = (
        f"(starttime>{t1:d} & starttime<{t2:d}) | "
        f"((endtime>{t1:d} & endtime<{t2:d}) | "
        f"(starttime<{t1:d} & endtime>{t2:d}))"
    )
    return con


class _IndexCache:
    """A simple class for caching indices."""

    def __init__(self, bank, cache_size=5):
        self.max_size = cache_size
        self.bank = bank
        self.cache = pd.DataFrame(
            index=range(cache_size), columns="t1 t2 kwargs cindex".split()
        )
        self._current_index = 0
        # self.next_index = itertools.cycle(self.cache.index)

    def __call__(self, starttime, endtime, buffer, **kwargs):
        """get start and end times, perform in kernel lookup"""
        starttime, endtime = self._get_times(starttime, endtime)
        self._validate_kwargs(kwargs)
        # find out if the query falls within one cached times
        con1 = self.cache.t1 <= starttime
        con2 = self.cache.t2 >= endtime
        con3 = self.cache.kwargs == self._kwargs_to_str(kwargs)
        cached_index = self.cache[con1 & con2 & con3]
        if not len(cached_index):  # query is not cached get it from hdf5 file
            where = _get_kernel_query(
                starttime.astype(np.int64), endtime.astype(np.int64), int(buffer)
            )
            raw_index = self._get_index(where, **kwargs)
            # replace "None" with None
            ic = self.bank.index_str
            raw_index.loc[:, ic] = raw_index.loc[:, ic].replace(["None"], [None])
            # convert data types used by bank back to those seen by user
            index = raw_index.astype(dict(self.bank._dtypes_output))
            self._set_cache(index, starttime, endtime, kwargs)
        else:
            index = cached_index.iloc[0]["cindex"]
        # trim down index
        con1 = index["starttime"] >= (endtime + buffer)
        con2 = index["endtime"] <= (starttime - buffer)
        return index[~(con1 | con2)]

    @staticmethod
    def _get_times(starttime, endtime):
        """Return starttimes and endtimes."""
        # get defaults if starttime or endtime is none
        starttime = None if pd.isnull(starttime) else starttime
        endtime = None if pd.isnull(endtime) else endtime
        starttime = to_datetime64(starttime or SMALLDT64)
        endtime = to_datetime64(endtime or LARGEDT64)
        if starttime is not None and endtime is not None:
            if starttime > endtime:
                msg = "starttime cannot be greater than endtime."
                raise ValueError(msg)
        return starttime, endtime

    def _validate_kwargs(self, kwargs):
        """Ensure kwargs are supported."""
        kwarg_set = set(kwargs)
        if not kwarg_set.issubset(READ_HDF5_KWARGS):
            bad_kwargs = kwarg_set - set(READ_HDF5_KWARGS)
            msg = f"The following kwargs are not supported: {bad_kwargs}. "
            raise UnsupportedKeyword(msg)

    def _set_cache(self, index, starttime, endtime, kwargs):
        """Cache the current index"""
        ser = pd.Series(
            {
                "t1": starttime,
                "t2": endtime,
                "cindex": index,
                "kwargs": self._kwargs_to_str(kwargs),
            }
        )
        self.cache.loc[self._get_next_index()] = ser

    def _kwargs_to_str(self, kwargs):
        """convert kwargs to a string"""
        keys = sorted(list(kwargs.keys()))
        ou = str([(item, kwargs[item]) for item in keys])
        return ou

    def _get_index(self, where, fail_counts=0, **kwargs):
        """read the hdf5 file"""
        try:
            return pd.read_hdf(
                self.bank.index_path, self.bank._index_node, where=where, **kwargs
            )

        except (ClosedNodeError, Exception) as e:
            # Sometimes in concurrent updates the nodes need time to open/close
            if fail_counts > 10:
                raise e
            # Wait a bit and try again (up to 10 times)
            time.sleep(0.1)
            return self._get_index(where, fail_counts=fail_counts + 1, **kwargs)

    def clear_cache(self):
        """removes all cached dataframes."""
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


class DASIndexer:
    """
    A class for interacting with an HDF5 index of DAS files.
    """

    # hdf5 compression defaults
    _complib = "blosc"
    _complevel = 9
    # attributes subclasses need to define
    ext = ""
    bank_path: Path = ""
    namespace = ""
    index_name = ".index.h5"  # name of index file
    executor = None  # an executor for using parallelism
    # optional str defining the directory structure and file name schemes
    path_structure = None
    name_structure = None
    # the minimum obsplus version. If not met delete index and re-index
    # bump when database schema change.
    _min_version = "0.0.3"
    # status bar attributes
    _bar_update_interval = 50  # number of files before updating bar
    _min_files_for_bar = 100  # min number of files before using bar enabled
    _read_func: callable  # function for reading datatype
    # required dtypes for input to storage layer
    _dtypes_input: Mapping = FrozenDict()
    # required dtypes for output from bank
    _dtypes_output: Mapping = FrozenDict()
    # the index cache (can greatly reduce IO efforts)
    _index_cache: Optional[_IndexCache] = None

    def _unindexed_iterator(self, paths: Optional[path_types] = None):
        """Return an iterator of potential unindexed files."""
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        last_updated = self.last_updated_timestamp  # this needs db so only call once
        if last_updated is not None:
            mtime = last_updated - 0.001
        # get paths to iterate
        bank_path = self.bank_path
        if paths is None:
            paths = self.bank_path
        else:
            paths = [
                f"{self.bank_path}/{x}" if str(bank_path) not in str(x) else str(x)
                for x in iterate(paths)
            ]
        # return file iterator
        return iter_files(paths, ext=self.ext, mtime=mtime)
