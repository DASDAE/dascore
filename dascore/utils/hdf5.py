"""
Utilities for working with HDF5 files.

Pytables should only be imported in this module in case we need to switch
out the hdf5 backend in the future.
"""

from __future__ import annotations

import io
import os
import shutil
import tempfile
import time
import warnings
from collections.abc import Sequence
from contextlib import contextmanager, suppress
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import tables
from h5py import File as H5pyFile
from packaging.version import parse as get_version
from pandas.io.common import stringify_path
from tables import ClosedNodeError
from tables import File as PyTablesFile

import dascore as dc
from dascore.compat import UPath
from dascore.config import config_attr, get_config
from dascore.constants import max_lens, remote_hdf5_tuned_protocols
from dascore.exceptions import InvalidFileHandlerError, InvalidIndexVersionError
from dascore.io.core import PatchFileSummary
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _maybe_make_parent_directory,
    _maybe_unpack,
    cached_method,
    suppress_warnings,
    unbyte,
)
from dascore.utils.pd import (
    _remove_base_path,
    fill_defaults_from_pydantic,
    list_ser_to_str,
)
from dascore.utils.remote_io import (
    FallbackFileObj,
    ensure_local_file,
    get_cached_local_file,
    get_local_handle,
    is_no_range_http_error,
)
from dascore.utils.time import get_max_min_times, to_datetime64, to_int, to_timedelta64

HDF5ExtError = tables.HDF5ExtError
NoSuchNodeError = tables.NoSuchNodeError
NodeError = tables.NodeError

ns_to_datetime = partial(pd.to_datetime, unit="ns")
ns_to_timedelta = partial(pd.to_timedelta, unit="ns")


class _ManagedH5pyFile:
    """
    Proxy an h5py file while owning the wrapped file object's lifecycle.

    This is used for ``h5py.File(..., driver="fileobj")`` paths where DASCore
    constructs the Python file-like object on behalf of the caller. Closing the
    returned handle should close both layers:
    1. the h5py/HDF5 handle
    2. the underlying Python file object (for example a remote fallback handle)
    """

    def __init__(self, handle: H5pyFile, owned_fileobj):
        self._handle = handle
        self._owned_fileobj = owned_fileobj
        self._closed = False

    def close(self):
        """Close both the h5py file and the owned file object."""
        if self._closed:
            return
        try:
            self._handle.close()
        finally:
            with suppress(Exception):
                self._owned_fileobj.close()
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def __getitem__(self, item):
        return self._handle[item]

    def __contains__(self, item):
        return item in self._handle

    def __iter__(self):
        return iter(self._handle)

    @property
    def closed(self):
        """Return True when close has been called on the proxy."""
        return self._closed

    def __getattr__(self, item):
        return getattr(self._handle, item)


class _HDF5Store(pd.HDFStore):
    """
    A work-around for pandas HDF5 store not accepting
    pytables.File objects.
    """

    def __init__(  # pragma: no cover
        self,
        path,
        mode: str = "a",
        complevel: int | None = None,
        complib=None,
        fletcher32: bool = False,
        **kwargs,
    ) -> None:
        if isinstance(path, str | Path):
            self._path = stringify_path(path)
        elif isinstance(path, tables.File):
            self._path = stringify_path(path.filename)
        self._mode = "a" if mode is None else mode
        self._handle = None
        self._complevel = complevel if complevel else 0
        self._complib = complib
        self._fletcher32 = fletcher32
        self._filters = None
        if isinstance(path, tables.File):
            self._handle = path
        else:
            self.open(mode)


@contextmanager
def open_hdf5_file(
    path_or_handler: Path | str | tables.File,
    mode: Literal["r", "w", "a"] = "r",
) -> tables.File:
    """
    A helper function for getting a `tables.file.File` object.

    If a file reference (str or Path) is passed this context manager will
    close the file when it exists.

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
            raise InvalidFileHandlerError(msg)

    if isinstance(path_or_handler, str | Path):
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

    # string column sizes in hdf5 table
    _min_itemsize = max_lens
    # columns which should be indexed for fast querying
    _query_columns = ("time_min", "time_max")
    # functions applied to encode dataframe before saving to hdf5
    _column_encoders = FrozenDict(
        {
            "time_min": lambda x: to_int(to_datetime64(x)),
            "time_max": lambda x: to_int(to_datetime64(x)),
            "time_step": lambda x: to_int(to_timedelta64(x)),
            "dims": list_ser_to_str,
            "path": lambda x: x.astype(str),
        }
    )
    # functions to apply to decode dataframe after loading from hdf file
    _column_decoders = FrozenDict(
        {
            "time_min": ns_to_datetime,
            "time_max": ns_to_datetime,
            "time_step": ns_to_timedelta,
        }
    )
    # base model which determines fields
    _base_model = PatchFileSummary
    # any fields to skip
    _skip_fields = ()
    # The minimum version of dascore required to read this index. If an older
    # version is used an error will be raised.
    _min_version = "0.0.13"

    def __init__(self, path, namespace=""):
        super().__init__()
        self.namespace = namespace
        self.path = path

    @property
    def index_columns(self):
        """Get the columns used for indexing."""
        out = set(self._base_model.model_fields) - set(self._skip_fields)
        return tuple(out)

    buffer: np.timedelta64 = config_attr("index_query_buffer")
    complib: str = config_attr("hdf_index_complib")
    complevel: int = config_attr("hdf_index_complevel")
    max_retries: int = config_attr("hdf_index_max_retries")

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
        cols = set(df.columns)
        for col, func in self._column_encoders.items():
            if col not in cols:
                continue
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
        for col, func in self._column_decoders.items():
            df[col] = func(df[col])
        # populate index store and update metadata
        # assert not df.isnull().any().any(), "null values found in index"
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
                if fail_counts >= self.max_retries:
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
        base_path: str | Path = "",
    ):
        """Convert updates to dataframe, then append to index table."""
        # read in dataframe and prepare for input into hdf5 index
        update_time = update_time or time.time()
        df = self.encode_table(update_df.copy(), path=base_path)
        with _HDF5Store(self.path) as store:
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
        """Read the metadata table."""
        try:
            with _HDF5Store(self.path, "r") as store:
                out = store.get(self._meta_node)
            store.close()
            return out
        except (FileNotFoundError, ValueError, KeyError, OSError):
            with suppress(UnboundLocalError):
                store.close()
            self._ensure_meta_table_exists()
            return pd.read_hdf(self.path, self._meta_node)

    def _ensure_meta_table_exists(self):
        """If the base path exists ensure it has a meta table, if not create it."""
        if not Path(self.path).exists():
            return
        with _HDF5Store(self.path) as store:
            # add metadata if not in store
            if self._meta_node not in store:
                meta = self._make_meta_table()
                store.put(self._meta_node, meta, format="table")

    def _make_meta_table(self):
        """Get a dataframe of meta info."""
        meta = dict(
            dascore_version=dc.__last_version__,
        )
        return pd.DataFrame(meta, index=[0])

    @property
    def hdf_kwargs(self) -> dict:
        """A dict of hdf_kwargs to pass to PyTables."""
        return dict(
            complib=self.complib,
            complevel=self.complevel,
            format="table",
            data_columns=list(self._query_columns),
        )

    @cached_method
    def validate_version(self):
        """Handles issues with version mismatches."""
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
        """Return True if an index table has been written."""
        expected_node = "/".join([self.namespace, "metadata"])
        with open_hdf5_file(self.path) as h5:
            try:
                h5.get_node(expected_node)
            except NoSuchNodeError:
                return False
            else:
                return True

    @property
    def _version_or_none(self) -> str | None:
        """Return the version string or None if it doesn't yet exist."""
        try:
            version = self._index_version
        except FileNotFoundError:
            return
        return version

    @property
    def last_updated_timestamp(self) -> float | None:
        """Return the last modified time stored in the index, else None."""
        try:
            out = pd.read_hdf(self.path, self._time_node)[0]
        except (OSError, IndexError, ValueError, KeyError, AttributeError):
            out = None
        return out


class PyTablesReader(PyTablesFile):
    """A thin wrapper around pytables File object for reading."""

    mode = "r"
    constructor = PyTablesFile

    @classmethod
    def get_handle(cls, resource):
        """Get the File object from various sources."""
        if isinstance(resource, cls | PyTablesFile):
            return resource
        try:
            _maybe_make_parent_directory(resource)
            return cls.constructor(resource, mode=cls.mode)
        except TypeError:
            msg = f"Couldn't get handle from {resource} using {cls}"
            raise NotImplementedError(msg)


class LocalPyTablesReader(PyTablesReader):
    """A PyTables reader which first materializes remote resources locally."""

    @classmethod
    def get_handle(cls, resource):
        """Get a local-file-backed PyTables handle."""
        return get_local_handle(resource, super().get_handle)


class PyTablesWriter(PyTablesReader):
    """A thin wrapper around pytables File object for writing."""

    mode = "a"


class H5Reader(PyTablesReader):
    """A thin wrapper around h5py for reading files.

    Remote UPath resources stay remote-first and transparently retry against
    a cached local file when no-range HTTP access prevents later random reads.
    """

    mode = "r"
    constructor = H5pyFile

    @classmethod
    def _open_fileobj_handle(cls, fileobj):
        """
        Open an h5py file and retain ownership of the wrapped file object.

        Returning the managed proxy keeps cleanup deterministic for all
        fileobj-backed HDF5 reads, including remote UPath resources.
        """
        handle = cls.constructor(fileobj, mode=cls.mode, driver="fileobj")
        return _ManagedH5pyFile(handle, fileobj)

    @staticmethod
    def _get_open_kwargs(resource: UPath) -> dict[str, object]:
        """Return backend-specific kwargs for remote HDF5 file objects."""
        protocol = getattr(resource, "protocol", None)
        if protocol in remote_hdf5_tuned_protocols:
            # h5py performs many small seeks while opening HDF5 metadata.
            # s3fs defaults to 50 MB readahead blocks, which can pull most of
            # a large remote file just to satisfy metadata probes.
            return {
                "block_size": get_config().remote_hdf5_block_size,
                "cache_type": "readahead",
            }
        return {}

    @classmethod
    def get_handle(cls, resource):
        """
        Get the HDF5 handle from local paths, remote paths, or open handles.

        Unlike PyTablesReader, h5py can consume a binary file object via the
        ``fileobj`` driver, so remote UPath inputs stay streaming-based here.
        """
        if isinstance(resource, cls | H5pyFile | _ManagedH5pyFile):
            return resource
        if isinstance(resource, io.IOBase):
            return cls._open_fileobj_handle(resource)
        if isinstance(resource, UPath):
            # If a previous metadata/read path already materialized a local copy,
            # prefer reopening that file directly instead of going remote-first
            # again. This avoids re-entering the remote fallback path once a
            # stable cached artifact already exists.
            if cached_path := get_cached_local_file(resource):
                return super().get_handle(cached_path)
            mode = "rb" if cls.mode == "r" else "r+b"
            open_kwargs = cls._get_open_kwargs(resource)
            handle = FallbackFileObj(
                remote_opener=lambda: resource.open(mode, **open_kwargs),
                local_opener=lambda: ensure_local_file(resource).open(mode),
                error_predicate=is_no_range_http_error,
            )
            try:
                return cls._open_fileobj_handle(handle)
            except Exception:
                handle.close()
                raise
        return super().get_handle(resource)


class LocalH5Reader(H5Reader):
    """An h5py reader which first materializes remote resources locally."""

    @classmethod
    def get_handle(cls, resource):
        """Get a local-file-backed h5py handle."""
        return get_local_handle(resource, super().get_handle)


class H5Writer(H5Reader):
    """A thin wrapper around h5py for writing files."""

    mode = "a"

    class _RemoteH5Writer:
        """Wrap a local h5py file and upload it back to the remote resource."""

        def __init__(self, resource: UPath, mode: str):  # pragma: no cover
            self._resource = resource
            suffix = resource.suffix or ".h5"
            fd, temp_name = tempfile.mkstemp(suffix=suffix)
            os.close(fd)
            self._temp_path = Path(temp_name)
            self._closed = False
            try:
                if mode != "w" and resource.exists():
                    with resource.open("rb") as src, self._temp_path.open("wb") as dst:
                        shutil.copyfileobj(src, dst)
                local_mode = (
                    "a"
                    if self._temp_path.exists() and self._temp_path.stat().st_size
                    else "w"
                )
                self._handle = H5pyFile(self._temp_path, mode=local_mode)
            except Exception:
                self._temp_path.unlink(missing_ok=True)
                raise

        def __getitem__(self, item):
            return self._handle[item]

        def __setitem__(self, key, value):  # pragma: no cover
            self._handle[key] = value

        def __contains__(self, item):  # pragma: no cover
            return item in self._handle

        def commit(self):
            """Finalize local writes, then upload the temp file to the remote path."""
            if self._closed:
                return
            self._handle.close()
            # The upload happens only after closing the local h5py handle because
            # h5py persists metadata and final file structure on close. Remote
            # backends are written back from the completed temp file as one blob.
            with self._temp_path.open("rb") as src, self._resource.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            self._temp_path.unlink(missing_ok=True)
            self._closed = True

        def close(self):
            """Commit remote writes on close to preserve normal file-like semantics."""
            self.commit()

        def abort(self):
            """Close and discard the local temp file without uploading it."""
            if self._closed:
                return
            self._handle.close()
            self._temp_path.unlink(missing_ok=True)
            self._closed = True

        def _abort(self):
            """Backward-compatible alias for abort()."""
            self.abort()

        def __enter__(self):  # pragma: no cover
            return self

        def __exit__(self, exc_type, exc, tb):  # pragma: no cover
            if exc_type is None:
                self.commit()
            else:
                self.abort()
            return False

        def __getattr__(self, item):
            return getattr(self._handle, item)

    @classmethod
    def get_handle(cls, resource):
        """Return an HDF5 writer handle for local or remote resources."""
        if isinstance(resource, UPath):
            return cls._RemoteH5Writer(resource, cls.mode)
        return super().get_handle(resource)


# These are left here for backward compatibility, but should not be
# used in new code.
HDF5Writer = PyTablesWriter
HDF5Reader = PyTablesReader


def unpack_scalar_h5_dataset(dataset):
    """
    Unpack a scalar H5Py dataset.
    """
    assert dataset.size == 1
    # This gets weird because datasets can be of shape () or (1,).
    value = dataset[()]
    if isinstance(value, np.ndarray):
        value = value[0]
    return value


def h5_matches_structure(h5file: H5pyFile, structure: Sequence[str]):
    """
    Check if an H5 file matches a spec given by a structure.

    Parameters
    ----------
    h5file
        A an open h5file as returned by h5py.File.
    structure
        A sequence of strings which indicates required groups/datasets/attrs.
        For example ("data", "data/raw", "data/raw.sampling") would require
        the 'data' group to exist, the data/raw group/dataset to exist and
        that raw has an attributed called 'sampling'.
    """
    for address in structure:
        split = address.split(".")
        assert len(split) in {1, 2}, "address can have at most one '.'"
        if len(split) == 2:
            base, attr = split
        else:
            base, attr = split[0], None
        try:
            obj = h5file[base]
        except KeyError:
            return False
        if attr is not None and attr not in set(obj.attrs):
            return False
    return True


def extract_h5_attrs(
    h5file: H5pyFile,
    name_map: dict[str, str],
    fill_values=None,
):
    """
    Extract attributes from h5 file based on structure.

    Parameters
    ----------
    h5file
        A an open h5file as returned by h5py.File.
    name_map
        A mapping from {old_name: new_name}. Old name must include one
        dot which separates the path from the attribute name.
        eg {"DasData.SamplingRate": "sampling_rate"}.

    Raises
    ------
    KeyError if any datasets/attributes are missing.
    """
    fill_values = fill_values or {}
    out = {}
    for address, out_name in name_map.items():
        split = address.split(".")
        assert len(split) == 2, "Struct must have exactly one '.'"
        base, attr = split
        obj = h5file[base]
        value = _maybe_unpack(unbyte(obj.attrs[attr]))
        out[out_name] = fill_values.get(value, value)
    return out
