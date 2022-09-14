"""
Utilities for working with HDF5 files.

Pytables should only be imported in this module in case we need to switch
out the hdf5 backend in the future.
"""
import time
import warnings
from contextlib import contextmanager, suppress
from functools import cache, partial
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import tables
from packaging.version import parse as get_version
from tables import ClosedNodeError

import dascore as dc
from dascore.constants import ONE_SECOND_IN_NS
from dascore.core.schema import PatchFileSummary
from dascore.exceptions import InvalidFileHandler, InvalidIndexVersionError
from dascore.utils.misc import suppress_warnings
from dascore.utils.pd import (
    _remove_base_path,
    fill_defaults_from_pydantic,
    list_ser_to_str,
)
from dascore.utils.time import get_max_min_times, to_number

HDF5ExtError = tables.HDF5ExtError
NoSuchNodeError = tables.NoSuchNodeError
NodeError = tables.NodeError

ns_to_datetime = partial(pd.to_datetime, unit="ns")
ns_to_timedelta = partial(pd.to_timedelta, unit="ns")


@contextmanager
def open_hdf5_file(
    path_or_handler: Union[Path, str, tables.File],
    mode: Literal["r", "w", "a"] = "r",
) -> tables.File:
    """
    A helper function for getting a `tables.file.File` object.

    If a file reference (str or Path) is passed this context manager will
    close the file, if a File

    Parameters
    ----------
    path_or_handler
        The input
    mode
        The mode in which to open the file.

    Raises
    ------
    InvalidBuffer if a writable mode is requested from a read only handler.
    """

    def _validate_mode(current_mode, desired_mode):
        """Ensure modes are compatible else raise."""
        if desired_mode == "r":
            return
        # if a or w is desired the current mode should be w
        if not current_mode == "w":
            msg = (
                f"A HDF5 file handler with mode 'r' was provided but "
                f"mode: {desired_mode} was requested."
            )
            raise InvalidFileHandler(msg)

    if isinstance(path_or_handler, (str, Path)):
        # Note: We suppress DataTypeWarnings because pytables fails to read
        # 8 bit enum indicating true or false written by h5py. See:
        # https://github.com/PyTables/PyTables/issues/647
        with suppress_warnings(tables.DataTypeWarning):
            with tables.open_file(path_or_handler, mode) as fi:
                yield fi
    elif isinstance(path_or_handler, tables.File):
        _validate_mode(path_or_handler.mode, mode)
        yield path_or_handler


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


class HDFPatchIndexManager:
    """
    A class for writing/querying an index table of summary patch info to hdf5.

    It creates a table of patch summary info, a table of metadata and a time
    stamp of the last time it was updated.
    """

    _complib = "blosc"
    _complevel = 9
    # attributes subclasses need to define
    buffer = ONE_SECOND_IN_NS
    # string column sizes in hdf5 table
    _min_itemsize = {
        "path": 120,
        "file_format": 15,
        "tag": 8,
        "network": 8,
        "station": 8,
        "dims": 40,
        "file_version": 9,
        "cable_id": 40,
        "instrument_id": 40,
    }
    # columns which should be indexed for fast querying
    _query_columns = ("time_min", "time_max", "distance_min", "distance_max")
    # functions applied to encode dataframe before saving to hdf5
    _column_encoders = {
        "time_min": to_number,
        "time_max": to_number,
        "d_time": to_number,
        "dims": list_ser_to_str,
    }
    # functions to apply to decode dataframe after loading from hdf file
    _column_decorders = {
        "time_min": ns_to_datetime,
        "time_max": ns_to_datetime,
        "d_time": ns_to_timedelta,
    }
    _base_model = PatchFileSummary
    index_columns = tuple(_base_model.__fields__)
    # The minimum version of dascore required to read this index. If an older
    # version is used an error will be raised.
    _min_version = "0.0.1"

    def __init__(self, path, namespace=""):
        super().__init__()
        self.namespace = namespace
        self.path = path
        # self.node = node

    # columns which should be indexed for fast querying
    @property
    def _time_node(self):
        """The node/table where the update time information is stored."""
        return "/".join([self.namespace, "last_updated"])

    @property
    def _index_node(self):
        """Return the node/table where the index information is stored."""
        return "/".join([self.namespace, "index"])

    @property
    def _meta_node(self):
        """The node/table where the update metadata is stored."""
        return "/".join([self.namespace, "metadata"])

    def encode_table(self, df, path=None):
        """Encode the table for writing to hdf5."""
        # apply column encoders, make paths relative to reference path
        # and drop any non-index columns.
        for col, func in self._column_encoders.items():
            df[col] = func(df[col])
        out = (
            df.pipe(fill_defaults_from_pydantic, self._base_model)
            .loc[:, list(self.index_columns)]
            .assign(path=lambda x: _remove_base_path(x["path"], path))
        )
        # there shouldn't be any null values in index now
        assert not out.isnull().any().any(), "null values found in index"
        return out

    def decode_table(self, df):
        """Decode the table from hdf5."""
        # ensure the base path is not in the path column
        for col, func in self._column_decorders.items():
            df[col] = func(df[col])
        # populate index store and update metadata
        assert not df.isnull().any().any(), "null values found in index"
        return df

    def get_index(self, time_min=None, time_max=None, **kwargs):
        """
        Read part of the hdf5 index from path meeting time min/max reqs.

        Parameters
        ----------
        time_min
            The start time of the entries to read.
        time_max
            The end time of the entries to read.
        """

        def _get_index(where, fail_counts=0, **kwargs):
            try:
                df = pd.read_hdf(self.path, self._index_node, where=where, **kwargs)
            except (ClosedNodeError, Exception) as e:
                # Sometimes in concurrent updates the nodes need time to open/close
                # so we implement a simply "wait and retry" strategy.
                # This is a bit wonky but we have found it to work well in practice.
                if fail_counts > 10:
                    raise e
                time.sleep(0.1)
                return _get_index(where, fail_counts=fail_counts + 1, **kwargs)
            else:
                return df

        time_min, time_max = get_max_min_times((time_min, time_max))
        where = _get_kernel_query(
            time_min.view(np.int64),
            time_max.view(np.int64),
            self.buffer.view(np.int64),
        )
        df = _get_index(where, **kwargs)
        return self.decode_table(df)

    def write_update(
        self,
        update_df,
        update_time=None,
        base_path: Union[str, Path] = "",
    ):
        """convert updates to dataframe, then append to index table"""
        # read in dataframe and prepare for input into hdf5 index
        update_time = update_time or time.time()
        df = self.encode_table(update_df, path=base_path)
        with pd.HDFStore(str(self.path)) as store:
            try:
                nrows = store.get_storer(self._index_node).nrows
            except (AttributeError, KeyError):
                store.append(
                    self._index_node,
                    df,
                    min_itemsize=self._min_itemsize,
                    **self.hdf_kwargs,
                )
            else:
                df.index += nrows
                store.append(self._index_node, df, append=True, **self.hdf_kwargs)
            self._update_metadata(store, update_time)

    def _update_metadata(self, store, update_time):
        # update timestamp
        update_time = time.time() if update_time is None else update_time
        store.put(self._time_node, pd.Series(update_time))
        # make sure meta table also exists.
        # Note this is here to avoid opening the store again.
        if self._meta_node not in store:
            meta = self._make_meta_table()
            store.put(self._meta_node, meta, format="table")

    def _read_metadata(self):
        """
        Read the metadata table.
        """
        try:
            with pd.HDFStore(self.path, "r") as store:
                out = store.get(self._meta_node)
            store.close()
            return out
        except (FileNotFoundError, ValueError, KeyError, OSError):
            with suppress(UnboundLocalError):
                store.close()
            self._ensure_meta_table_exists()
            return pd.read_hdf(self.path, self._meta_node)

    def _ensure_meta_table_exists(self):
        """
        If the base path exists ensure it has a meta table, if not create it.
        """
        if not Path(self.path).exists():
            return
        with pd.HDFStore(str(self.path)) as store:
            # add metadata if not in store
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    def _make_meta_table(self):
        """get a dataframe of meta info"""
        meta = dict(
            dascore_version=dc.__last_version__,
        )
        return pd.DataFrame(meta, index=[0])

    @property
    def hdf_kwargs(self) -> dict:
        """A dict of hdf_kwargs to pass to PyTables"""
        return dict(
            complib=self._complib,
            complevel=self._complevel,
            format="table",
            data_columns=list(self._query_columns),
        )

    @cache
    def validate_version(self):
        """
        This method handles issues with version mismatches.

        If this is the case, there is no guarantee it will work, but no knowing
        if it won't...
        """
        # get the version from file, if the file doesnt exist then None
        version = self._version_or_none
        if version is not None:
            # check if index is too old to be read by this version of the parser.
            # If this is the case, users of this class should handle its
            # re-creation.
            min_version_tuple = get_version(self._min_version)
            index_version = get_version(version)
            if min_version_tuple > index_version:
                msg = (
                    f"The indexing schema has changed since {self._min_version} "
                    f"and must be regenerated."
                )
                raise InvalidIndexVersionError(msg)
            # check if index was created with newer version of dascore
            dascore_version = get_version(dc.__last_version__)
            if index_version > dascore_version:
                msg = (
                    f"The index was created with a newer version of dascore ("
                    f"{version}), you are running ({dc.__last_version__}), "
                    f"You may encounter problems, consider updating DASCore."
                )
                warnings.warn(msg)

    @property
    def _index_version(self) -> str:
        """Get the version of dascore used to create the index."""
        return self._read_metadata()["dascore_version"].iloc[0]

    @property
    def has_index(self) -> bool:
        """Return True if an index table has been writen."""
        expected_node = "/".join([self.namespace, "metadata"])
        with open_hdf5_file(self.path) as h5:
            try:
                h5.get_node(expected_node)
            except NoSuchNodeError:
                return False
            else:
                return True

    @property
    def _version_or_none(self) -> Optional[str]:
        """Return the version string or None if it doesn't yet exist."""
        try:
            version = self._index_version
        except (FileNotFoundError):
            return
        return version

    @property
    def last_updated_timestamp(self) -> Optional[float]:
        """
        Return the last modified time stored in the index, else None.
        """
        try:
            out = pd.read_hdf(self.path, self._time_node)[0]
        except (IOError, IndexError, ValueError, KeyError, AttributeError):
            out = None
        return out
