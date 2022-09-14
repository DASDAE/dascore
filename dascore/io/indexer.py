"""
An HDF5-based indexer for local file systems.
"""
import abc
import os
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from typing_extensions import Self

import dascore as dc
from dascore.constants import ONE_SECOND_IN_NS, path_types
from dascore.exceptions import InvalidIndexVersionError
from dascore.utils.hdf5 import HDFPatchIndexManager
from dascore.utils.misc import iter_files, iterate
from dascore.utils.pd import filter_df
from dascore.utils.progress import track
from dascore.utils.time import get_max_min_times, to_timedelta64

# supported read_hdf5 kwargs
READ_HDF5_KWARGS = frozenset(
    {"columns", "where", "mode", "errors", "start", "stop", "key", "chunksize"}
)


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
    cache_size
        The number of queries to store in memory to avoid frequent reads
        of the index file.
    """

    # hdf5 compression defaults
    ext = ""
    namespace = ""
    index_name = ".dascore_index.h5"  # name of index file
    executor = None  # an executor for using parallelism

    def __init__(self, path: Union[str, Path], cache_size: int = 5):
        self.max_size = cache_size
        self.path = Path(path).absolute()
        self.index_path = self.path / self.index_name
        self.cache = pd.DataFrame(
            index=range(cache_size), columns="t1 t2 kwargs cindex".split()
        )
        self._current_index = 0
        self._index_table = HDFPatchIndexManager(
            self.index_path,
            self.namespace,
        )

    def get_contents(self, buffer=ONE_SECOND_IN_NS, **kwargs):
        """get start and end times, perform in kernel lookup"""
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
        return pre_filter_df[filter_df(pre_filter_df, **kwargs)]

    def __str__(self):
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
        out = str([(item, kwargs[item]) for item in keys])
        return out

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

    def _get_file_iterator(self, paths: Optional[path_types] = None, only_new=True):
        """Return an iterator of potential un-indexed files."""
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        # getting last updated might need the db so only call once.
        last_updated = self._index_table.last_updated_timestamp if only_new else None
        if last_updated is not None and only_new:
            mtime = last_updated - 0.001
        # get paths to iterate
        path = self.path
        if paths is None:
            paths = path
        else:
            paths = [
                f"{path}/{x}" if str(path) not in str(x) else str(x)
                for x in iterate(paths)
            ]
        # return file iterator
        return iter_files(paths, ext=self.ext, mtime=mtime)

    def ensure_path_exists(self, create=False):
        """
        Ensure the base path exists else raise.

        If create is True, simply create an empty directory.
        """
        path = Path(self.path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            msg = f"{path} is not a directory, cant read spool"
            raise FileExistsError(msg)

    def _enforce_min_version(self):
        """
        Ensure the minimum version is met, else delete index file.
        """
        try:
            self._index_table.validate_version()
        except InvalidIndexVersionError:
            os.remove(self.index_path)
            self.update()

    def update(self, paths=None) -> Self:
        """
        Updates the contents of the Indexer.

        Resets any previous selection.
        """
        self._enforce_min_version()  # delete index if schema has changed
        update_time = time.time()
        new_files = list(self._get_file_iterator(paths=paths, only_new=True))
        smooth_iterator = track(new_files, f"Indexing {self.path.name}")
        data_list = [y.dict() for x in smooth_iterator for y in dc.scan(x, ignore=True)]
        df = pd.DataFrame(data_list)
        if not df.empty:
            # ensure the base path is not in the path column
            assert "path" in set(df.columns), f"{df} has no path column"
            self._index_table.write_update(df, update_time, base_path=self.path)
            # clear cache out when new traces are added
            self.clear_cache()
        return self
