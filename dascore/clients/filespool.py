"""
A spool for working with file systems.

The spool uses a simple hdf5 index for keeping track of files.
"""
import os
import time
import warnings
from contextlib import suppress
from functools import partial
from pathlib import Path
from typing import Mapping, Optional, Union

import numpy as np
import pandas as pd
from packaging.version import parse as get_version
from tables.exceptions import ClosedNodeError
from typing_extensions import Self

import dascore as dc
from dascore.constants import LARGEDT64, SMALLDT64, path_types
from dascore.core.schema import PatchFileSummary
from dascore.core.spool import DataFrameSpool
from dascore.exceptions import UnsupportedKeyword
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import iter_files, iterate
from dascore.utils.pd import _remove_base_path, filter_df, update_ranges_with_kwargs
from dascore.utils.progress import track_index_update
from dascore.utils.time import to_datetime64, to_number, to_timedelta64

# supported read_hdf5 kwargs
READ_HDF5_KWARGS = frozenset(
    {"columns", "where", "mode", "errors", "start", "stop", "key", "chunksize"}
)


ns_to_datetime = partial(pd.to_datetime, unit="ns")
ns_to_timedelta = partial(pd.to_timedelta, unit="ns")

one_nanosecond = np.timedelta64(1, "ns")
one_second = np.timedelta64(1_000_000_000, "ns")


def _get_kernel_query(starttime: int, endtime: int, buffer: int):
    """
    Create a HDF5 kernel query based on start and end times.

    This is necessary because hdf5 doesn't accept inverted conditions.
    A slight buffer is applied to the ranges to make sure no edge files
    are excluded.
    """
    t1 = starttime - buffer
    t2 = endtime + buffer
    con = (
        f"(time_min>{t1:d} & time_min<{t2:d}) | "
        f"((time_max>{t1:d} & time_max<{t2:d}) | "
        f"(time_min<{t1:d} & time_max>{t2:d}))"
    )
    return con


class _TimeIndexCache:
    """A Cache for indices based on time values"""

    def __init__(self, spool: "FileSpool", cache_size=5):
        self.max_size = cache_size
        self.spool = spool
        self.cache = pd.DataFrame(
            index=range(cache_size), columns="t1 t2 kwargs cindex".split()
        )
        self._current_index = 0
        # self.next_index = itertools.cycle(self.cache.index)

    def __call__(self, time_min=None, time_max=None, buffer=one_second, **kwargs):
        """get start and end times, perform in kernel lookup"""
        time_min, time_max = self._get_times(time_min, time_max)
        self._validate_kwargs(kwargs)
        buffer = to_timedelta64(buffer)
        # find out if the query falls within one cached times
        con1 = self.cache.t1 <= time_min
        con2 = self.cache.t2 >= time_max
        con3 = self.cache.kwargs == self._kwargs_to_str(kwargs)
        cached_index = self.cache[con1 & con2 & con3]
        if not len(cached_index):  # query is not cached get it from hdf5 file
            where = _get_kernel_query(
                time_min.astype(np.int64), time_max.astype(np.int64), int(buffer)
            )
            raw_index = self._get_index(where, **kwargs)
            index = self.spool._decode_df_from_hdf(raw_index)
            self._set_cache(index, time_min, time_max, kwargs)
        else:
            index = cached_index.iloc[0]["cindex"]
        # trim down index
        con1 = index["time_min"] >= (time_max + buffer)
        con2 = index["time_max"] <= (time_min - buffer)
        return index[~(con1 | con2)]

    @staticmethod
    def _get_times(time_min, time_max):
        """Return starttimes and endtimes."""
        # get defaults if starttime or endtime is none
        time_min = None if pd.isnull(time_min) else time_min
        time_max = None if pd.isnull(time_max) else time_max
        time_min = to_datetime64(time_min or SMALLDT64)
        time_max = to_datetime64(time_max or LARGEDT64)
        if time_min is not None and time_max is not None:
            if time_min > time_max:
                msg = "starttime cannot be greater than endtime."
                raise ValueError(msg)
        return time_min, time_max

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
                self.spool.index_path, self.spool._index_node, where=where, **kwargs
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


class FileSpool(DataFrameSpool):
    """
    A spool for interacting with DAS files on disk.

    FileSpool creates and index of all files then allows for simple querying
    and bulk processing of the files.
    """

    # hdf5 compression defaults
    _complib = "blosc"
    _complevel = 9
    # attributes subclasses need to define
    ext = ""
    spool_path: Path = ""
    namespace = ""
    index_name = ".dascore_index.h5"  # name of index file
    executor = None  # an executor for using parallelism
    # buffer for queries
    buffer = one_second
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
    _index_cache: Optional[_TimeIndexCache] = None
    # string column sizes in hdf5 table
    _min_itemsize = {
        "path": 79,
        "file_format": 15,
        "tag": 8,
        "network": 8,
        "station": 8,
        "dims": 40,
    }
    # columns which should be indexed for fast querying
    _query_columns = ("time_min", "time_max", "distance_min", "distance_max")
    # functions applied to encode dataframe before saving to hdf5
    _column_encoders = {
        "time_min": to_number,
        "time_max": to_number,
        "d_time": to_number,
    }
    # functions to apply to decode dataframe after loading from hdf file
    _column_decorders = {
        "time_min": ns_to_datetime,
        "time_max": ns_to_datetime,
        "d_time": ns_to_timedelta,
    }
    _index_columns = tuple([x for x in PatchFileSummary.__annotations__])

    def __init__(
        self,
        base_path: Union[str, Path, Self] = ".",
        file_format: Optional[str] = None,
        select_kwargs: Optional[dict] = None,
    ):
        if isinstance(base_path, self.__class__):
            self.__dict__.update(base_path.__dict__)
            return
        self.file_format = file_format
        self.spool_path = Path(base_path).absolute()
        # initialize cache
        self._index_cache = _TimeIndexCache(self)
        # enforce min version or warn on newer version
        self._enforce_min_version()
        self._warn_on_newer_version()
        self._select_kwargs = select_kwargs if select_kwargs is not None else {}

    def get_contents(self) -> pd.DataFrame:
        """
        Return a dataframe of the contents of the data files.

        Parameters
        ----------
        time
            If not None, a tuple of start/end time where either can be None
            indicating an open interval.
        """
        self.ensure_bank_path_exists()
        if not self.index_path.exists():
            self = self.update()
        # if no file was created (dealing with empty bank) return empty index
        if not self.index_path.exists():
            return pd.DataFrame(columns=self._index_columns)
        # grab index from cache
        index = self._index_cache(buffer=self.buffer)
        # filter and return
        filt = filter_df(index, **self._select_kwargs)
        out = index[filt]
        return update_ranges_with_kwargs(out, **self._select_kwargs)

    def select(self, **kwargs) -> Self:
        """Sub-select certain dimensions for Spool"""
        out = self.__class__(
            base_path=self.spool_path,
            file_format=self.file_format,
            select_kwargs=kwargs,
        )
        return out

    @property
    def _df(self):
        """Return get_contents."""
        return self.get_contents()

    def update(self) -> Self:
        """
        Updates the contents of the spool and returns a spool.

        Resets any previous selection.
        """
        self._enforce_min_version()  # delete index if schema has changed
        update_time = time.time()
        new_files = list(self._get_file_iterator(only_new=True))
        smooth_iterator = track_index_update(
            new_files, f"Indexing {self.spool_path.name}"
        )

        data_list = [y.dict() for x in smooth_iterator for y in dc.scan(x)]
        df = pd.DataFrame(data_list)
        if not df.empty:
            self._write_update(df, update_time)
            # clear cache out when new traces are added
            self.clear_cache()
        return self

    def _get_file_iterator(self, paths: Optional[path_types] = None, only_new=True):
        """Return an iterator of potential un-indexed files."""
        # get mtime, subtract a bit to avoid odd bugs
        mtime = None
        # getting last updated might need the db so only call once.
        last_updated = self.last_updated_timestamp if only_new else None
        if last_updated is not None and only_new:
            mtime = last_updated - 0.001
        # get paths to iterate
        spool_path = self.spool_path
        if paths is None:
            paths = self.spool_path
        else:
            paths = [
                f"{self.spool_path}/{x}" if str(spool_path) not in str(x) else str(x)
                for x in iterate(paths)
            ]
        # return file iterator
        return iter_files(paths, ext=self.ext, mtime=mtime)

    def __iter__(self):
        # get dataframe, add absolute path and iterate
        df = self._df.copy(deep=False)
        df["path"] = str(self.spool_path) + df["path"]
        for ind in range(len(df)):
            yield self.load_patch(df.iloc[ind])

    def _extract_patch_from_row(self, row) -> Self:
        """Given a row from the managed dataframe, return a patch."""
        patch = dc.read(**dict(row))[0]
        return patch

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
        path = Path(self.spool_path)
        if create:
            path.mkdir(parents=True, exist_ok=True)
        if not path.is_dir():
            msg = f"{path} is not a directory, cant read spool"
            raise FileExistsError(msg)

    def _enforce_min_version(self):
        """
        Check version of dascore used to create index and delete index if the
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
        Issue a warning if the bank was created by a newer version of dascore.

        If this is the case, there is no guarantee it will work.
        """
        version = self._version_or_none
        if version is not None:
            dascore_version = get_version(dc.__last_version__)
            bank_version = get_version(version)
            if bank_version > dascore_version:
                msg = (
                    f"The bank was created with a newer version of dc ("
                    f"{version}), you are running ({dc.__last_version__}),"
                    f"You may encounter problems, consider updating DASCore."
                )
                warnings.warn(msg)

    @property
    def hdf_kwargs(self) -> dict:
        """A dict of hdf_kwargs to pass to PyTables"""
        return dict(
            complib=self._complib,
            complevel=self._complevel,
            format="table",
            data_columns=list(self._query_columns),
        )

    @property
    def _time_node(self):
        """The node/table where the update time information is stored."""
        return "/".join([self.namespace, "last_updated"])

    @property
    def index_path(self):
        """Return the expected path to the index file."""
        return Path(self.spool_path) / self.index_name

    @property
    def _index_node(self):
        """Return the node/table where the index information is stored."""
        return "/".join([self.namespace, "index"])

    @property
    def _index_version(self) -> str:
        """Get the version of dascore used to create the index."""
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
            dascore_version=dc.__last_version__,
        )
        return pd.DataFrame(meta, index=[0])

    def _write_update(self, update_df, update_time):
        """convert updates to dataframe, then append to index table"""
        # read in dataframe and prepare for input into hdf5 index
        df = self._encode_df_for_hdf(update_df)
        with pd.HDFStore(str(self.index_path)) as store:
            node = self._index_node
            try:
                nrows = store.get_storer(node).nrows
            except (AttributeError, KeyError):
                store.append(
                    node, df, min_itemsize=self._min_itemsize, **self.hdf_kwargs
                )
            else:
                df.index += nrows
                store.append(node, df, append=True, **self.hdf_kwargs)
            # update timestamp
            update_time = time.time() if update_time is None else update_time
            store.put(self._time_node, pd.Series(update_time))
            # make sure meta table also exists.
            # Note this is hear to avoid opening the store again.
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    def _encode_df_for_hdf(self, df):
        """Prepare the dataframe to put it into the HDF5 store."""
        # ensure the bank path is not in the path column
        assert "path" in set(df.columns), f"{df} has no path column"
        df["path"] = _remove_base_path(df["path"], self.spool_path)
        for col, func in self._column_encoders.items():
            df[col] = func(df[col])
        # populate index store and update metadata
        assert not df.isnull().any().any(), "null values found in index"
        return df

    def _decode_df_from_hdf(self, df):
        """Decode the dataframe from the hdf5 file."""
        # ensure the bank path is not in the path column
        for col, func in self._column_decorders.items():
            df[col] = func(df[col])
        # populate index store and update metadata
        assert not df.isnull().any().any(), "null values found in index"
        return df

    def clear_cache(self):
        """
        Clear the index cache if the bank is using one.
        """
        if self._index_cache is not None:
            self._index_cache.clear_cache()
