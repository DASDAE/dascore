"""Utilities for working with HDF5 files (h5py-based)."""

from __future__ import annotations

import io
import os
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import suppress
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from h5py import File as H5pyFile

from dascore.compat import UPath
from dascore.config import get_config
from dascore.constants import remote_hdf5_tuned_protocols
from dascore.utils.misc import (
    _maybe_make_parent_directory,
    _maybe_unpack,
    iterate,
    unbyte,
)
from dascore.utils.remote_io import (
    _FallbackFileObj,
    _get_cached_local_file,
    ensure_local_file,
    get_local_handle,
    is_no_range_http_error,
)

ns_to_datetime = partial(pd.to_datetime, unit="ns")
ns_to_timedelta = partial(pd.to_timedelta, unit="ns")


def encode_h5_strings(values: str | Sequence[str]) -> np.ndarray:
    """Encode strings as a fixed-length UTF-8 HDF5 byte array."""
    return np.asarray([value.encode() for value in iterate(values)], dtype="S")


class _ManagedH5pyFile:
    """
    DASCore's internal h5py handle wrapper with deterministic close behavior.

    All h5py-backed DASCore reads return this wrapper so callers see one handle
    type regardless of whether the underlying resource came from:
    - a local path
    - an existing h5py handle
    - a Python file object
    - a remote ``UPath`` opened through the fallback fileobj path

    For path-backed opens, this wrapper owns only the h5py handle. For
    ``h5py.File(..., driver="fileobj")`` paths, it also owns the Python
    file-like object DASCore created on behalf of the caller. ``close()`` is
    therefore the point where DASCore tears down the entire HDF5 access stack.
    """

    def __init__(self, handle: H5pyFile, owned_fileobj=None):
        self._handle = handle
        self._owned_fileobj = owned_fileobj
        self._closed = False

    def close(self):
        """Close the h5py file and, when present, the owned file object."""
        if self._closed:
            return
        try:
            self._handle.close()
        finally:
            if self._owned_fileobj is not None:
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


def get_h5py_file(handle) -> H5pyFile:
    """
    Return the underlying ``h5py.File`` for a DASCore h5 handle.

    Consumers such as the ``h5netcdf`` xarray engine require a real
    ``h5py.File``/group and do not accept DASCore's ``_ManagedH5pyFile`` proxy.
    Unwrapping the proxy preserves DASCore's ownership: closing the returned
    ``h5py.File`` remains the responsibility of the managing handle (or the
    ``IOResourceManager``), not the consumer.
    """
    if isinstance(handle, _ManagedH5pyFile):
        return handle._handle
    return handle


def open_h5_resource(
    resource,
    *,
    mode: str,
    constructor,
    open_kwargs_getter,
) -> _ManagedH5pyFile:
    """
    Open an HDF5 resource and return DASCore's managed h5py handle wrapper.

    This is the central constructor for h5py-backed reads in DASCore. It keeps
    the branching needed for local paths, already-open handles, remote
    fileobj-backed reads, cached-local reuse, and no-range HTTP fallback in one
    place so ``H5Reader.get_handle()`` stays thin.

    Parameters
    ----------
    resource
        A local path, remote ``UPath``, open file object, or existing h5py
        handle.
    mode
        The mode to pass to the h5py constructor.
    constructor
        The callable used to construct an h5py handle.
    open_kwargs_getter
        Callback which returns backend-specific kwargs for remote file opens.
    """
    if isinstance(resource, _ManagedH5pyFile):
        return resource
    if isinstance(resource, H5pyFile):
        return _ManagedH5pyFile(resource)
    if isinstance(resource, io.IOBase):
        handle = constructor(resource, mode=mode, driver="fileobj")
        return _ManagedH5pyFile(handle, resource)
    if isinstance(resource, UPath):
        # Reuse an already-materialized local artifact when present so later
        # HDF5 reads do not re-enter the remote fallback path unnecessarily.
        if cached_path := _get_cached_local_file(resource):
            return open_h5_resource(
                cached_path,
                mode=mode,
                constructor=constructor,
                open_kwargs_getter=open_kwargs_getter,
            )
        file_mode = "rb" if mode == "r" else "r+b"
        open_kwargs = open_kwargs_getter(resource)
        handle = _FallbackFileObj(
            remote_opener=lambda: resource.open(file_mode, **open_kwargs),
            local_opener=lambda: ensure_local_file(resource).open(file_mode),
            error_predicate=is_no_range_http_error,
        )
        try:
            h5_handle = constructor(handle, mode=mode, driver="fileobj")
            return _ManagedH5pyFile(h5_handle, handle)
        except Exception:
            handle.close()
            raise
    try:
        if mode != "r":
            _maybe_make_parent_directory(resource)
        return _ManagedH5pyFile(constructor(resource, mode=mode))
    except TypeError:
        msg = f"Couldn't get handle from {resource} using h5py"
        raise NotImplementedError(msg)


class H5Reader:
    """A thin wrapper around h5py for reading files.

    Remote UPath resources stay remote-first and transparently retry against
    a cached local file when no-range HTTP access prevents later random reads.
    """

    mode = "r"
    constructor = H5pyFile

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

        h5py can consume a binary file object via the
        ``fileobj`` driver, so remote UPath inputs stay streaming-based here.
        """
        if isinstance(resource, cls | _ManagedH5pyFile):
            return resource
        return open_h5_resource(
            resource,
            mode=cls.mode,
            constructor=cls.constructor,
            open_kwargs_getter=cls._get_open_kwargs,
        )


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
