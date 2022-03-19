"""
Utilities for creating an index of directories of files.
"""
import os
import time
import warnings
from concurrent.futures import Executor
from contextlib import suppress
from itertools import chain
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
from packaging.version import parse as get_version
from tables.exceptions import ClosedNodeError

import dascore
from dascore.constants import LARGEDT64, SMALLDT64, path_types
from dascore.exceptions import UnsupportedKeyword
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import iter_files, iterate
from dascore.utils.time import to_datetime64

# supported read_hdf5 kwargs
READ_HDF5_KWARGS = frozenset(
    {"columns", "where", "mode", "errors", "start", "stop", "key", "chunksize"}
)


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


class DASBank:
    """
    A class for interacting with a directory of DAS files.
    """

    # hdf5 compression defaults
    _complib = "blosc"
    _complevel = 9
    # attributes subclasses need to define
    ext = ""
    bank_path: Path = ""
    namespace = ""
    index_name = ".dascore_index.h5"  # name of index file
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

    def __init__(
        self,
        base_path: Union[str, Path, "DASBank"] = ".",
        cache_size: int = 5,
        executor: Optional[Executor] = None,
    ):
        if isinstance(base_path, self.__class__):
            self.__dict__.update(base_path.__dict__)
            return
        self.format = format
        self.bank_path = Path(base_path).absolute()
        self.executor = executor
        # initialize cache
        self._index_cache = _IndexCache(self, cache_size=cache_size)
        # enforce min version or warn on newer version
        self._enforce_min_version()
        self._warn_on_newer_version()

    def contents_to_df(self, only_new: bool = True) -> pd.DataFrame:
        """
        Return a dataframe of the contents of the data files.

        Parameters
        ----------
        only_new
            If True, only return contents of files created since the index
            was last updated.
        """
        smooth_iterator = self._get_file_iterator(only_new=only_new)
        data = list(chain.from_iterable(dascore.scan(x) for x in smooth_iterator))
        df = pd.DataFrame(data)
        return df

    def _get_file_iterator(self, paths: Optional[path_types] = None, only_new=True):
        """Return an iterator of potential unindexed files."""
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        # getting last updated might need the db so only call once.
        last_updated = self.last_updated_timestamp if only_new else None
        if last_updated is not None and only_new:
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

    @property
    def last_updated_timestamp(self) -> Optional[float]:
        """
        Return the last modified time stored in the index, else None.
        """
        self.ensure_bank_path_exists()
        node = self._time_node
        try:
            out = pd.read_hdf(self.index_path, node)[0]
        except (IOError, IndexError, ValueError, KeyError, AttributeError):
            out = None
        return out

    def ensure_bank_path_exists(self, create=False):
        """
        Ensure the bank_path exists else raise an BankDoesNotExistError.

        If create is True, simply create the bank.
        """
        path = Path(self.bank_path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            msg = f"{path} is not a directory, cant read spool"
            raise FileExistsError(msg)

    def _enforce_min_version(self):
        """
        Check version of obsplus used to create index and delete index if the
        minimum version requirement is not met.
        """
        version = self._version_or_none
        if version is not None:
            min_version_tuple = get_version(self._min_version)
            version_tuple = get_version(version)
            if min_version_tuple > version_tuple:
                msg = (
                    f"The indexing schema has changed since {self._min_version} "
                    f"the index will be recreated."
                )
                warnings.warn(msg)
                os.remove(self.index_path)

    def _warn_on_newer_version(self):
        """
        Issue a warning if the bank was created by a newer version of obsplus.

        If this is the case, there is no guarantee it will work.
        """
        version = self._version_or_none
        if version is not None:
            obsplus_version = get_version(dascore.__last_version__)
            bank_version = get_version(version)
            if bank_version > obsplus_version:
                msg = (
                    f"The bank was created with a newer version of DASCore ("
                    f"{version}), you are running ({dascore.__last_version__}),"
                    f"You may encounter problems, consider updating DASCore."
                )
                warnings.warn(msg)

    @property
    def _time_node(self):
        """The node/table where the update time information is stored."""
        return "/".join([self.namespace, "last_updated"])

    @property
    def index_path(self):
        """Return the expected path to the index file."""
        return Path(self.bank_path) / self.index_name

    @property
    def _index_node(self):
        """Return the node/table where the index information is stored."""
        return "/".join([self.namespace, "index"])

    @property
    def _index_version(self) -> str:
        """Get the version of obsplus used to create the index."""
        return self._read_metadata()["dascore_version"].iloc[0]

    @property
    def _meta_node(self):
        """The node/table where the update metadata is stored."""
        return "/".join([self.namespace, "metadata"])

    @property
    def _version_or_none(self) -> Optional[str]:
        """Return the version string or None if it doesn't yet exist."""
        try:
            version = self._index_version
        except (FileNotFoundError):
            return
        return version

    def _read_metadata(self):
        """
        Read the metadata table.
        """
        try:
            with pd.HDFStore(self.index_path, "r") as store:
                out = store.get(self._meta_node)
            store.close()
            return out
        except (FileNotFoundError, ValueError, KeyError, OSError):
            with suppress(UnboundLocalError):
                store.close()
            self._ensure_meta_table_exists()
            return pd.read_hdf(self.index_path, self._meta_node)

    def _ensure_meta_table_exists(self):
        """
        If the bank path exists ensure it has a meta table, if not create it.
        """
        if not Path(self.index_path).exists():
            return
        with pd.HDFStore(self.index_path) as store:
            # add metadata if not in store
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    def _make_meta_table(self):
        """get a dataframe of meta info"""
        meta = dict(
            path_structure=self.path_structure,
            name_structure=self.name_structure,
            obsplus_version=dascore.__last_version__,
        )
        return pd.DataFrame(meta, index=[0])
