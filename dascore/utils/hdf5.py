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
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from h5py import File as H5pyFile

from dascore.compat import UPath
from dascore.config import get_config
from dascore.constants import http_protocols, remote_hdf5_tuned_protocols
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
    pause_gc,
    resume_gc,
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
    file-like object underneath, whether DASCore created it or the caller
    supplied it. ``close()`` is therefore the point where DASCore tears down
    the entire HDF5 access stack.
    """

    # Class defaults, so a half-built instance is still closeable rather than
    # falling through __getattr__ to a handle that may not be set yet.
    _closed = False
    _gc_paused_pid = None

    def __init__(self, handle: H5pyFile, owned_fileobj=None, gc_paused=False):
        self._handle = handle
        self._owned_fileobj = owned_fileobj
        if gc_paused:
            # The pid that paused, so a handle inherited through a fork
            # cannot resume a pause the child never took.
            self._gc_paused_pid = os.getpid()

    def close(self):
        """Close the h5py file and, when present, the owned file object."""
        if self._closed:
            return
        self._closed = True
        try:
            self._handle.close()
        finally:
            # Nested so nothing raised by the teardown, including a
            # BaseException, can skip the resume and strand the pause.
            try:
                if self._owned_fileobj is not None:
                    with suppress(Exception):
                        self._owned_fileobj.close()
            finally:
                # dict.pop is atomic, so racing closes resume exactly once.
                pid = self.__dict__.pop("_gc_paused_pid", None)
                if pid == os.getpid():
                    resume_gc()

    def __del__(self):
        """
        Release a leaked handle's pause so it cannot stop collection forever.

        Only the pause: closing here would also close a caller-supplied h5py
        file or stream that this wrapper never had permission to close.
        Reference counting still tears the underlying handles down.
        """
        pid = self.__dict__.pop("_gc_paused_pid", None)
        if pid == os.getpid():
            with suppress(Exception):
                resume_gc()

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


def _is_loop_backed(resource) -> bool:
    """
    Return True when reads on a resource are served by an event-loop thread.

    fsspec async filesystems (http, s3, ...) bridge each read onto a shared
    event-loop thread and mark themselves with ``async_impl``; duck-typing it
    avoids importing fsspec here. Local, memory, and other synchronous
    backends need no GC pause. A buffered reader hides the filesystem behind
    ``raw``, so unwrap it: missing it would leave the deadlock window open.
    """
    while resource is not None:
        try:
            if getattr(getattr(resource, "fs", None), "async_impl", False):
                return True
            wrapped = getattr(resource, "raw", None)
        except Exception:
            # Probing can fail rather than return nothing: a UPath whose
            # backend is not installed raises from ``fs``, and a wrapper
            # can refuse an attribute with something other than
            # AttributeError. Either way the open below reports it.
            return False
        resource = None if wrapped is resource else wrapped
    return False


def _open_h5_fileobj(
    fileobj, constructor, mode, *, pause: bool, close_on_error: bool
) -> _ManagedH5pyFile:
    """
    Open a file object with h5py through the fileobj driver.

    Loop-backed resources pause automatic collection first; the returned
    wrapper owns that pause and releases it on close, and a failed open
    releases it here.

    ``close_on_error`` is set only for file objects DASCore created. A
    caller-supplied one must survive a failed open: ``get_format`` offers the
    same object to every FiberIO in turn, and an HDF5 miss is the expected
    outcome for most of them.
    """
    try:
        if pause:
            # Inside the try, and paired unconditionally below, because
            # pause_gc always leaves the depth consistent with the pauses it
            # took -- even when interrupted partway.
            pause_gc()
        handle = constructor(fileobj, mode=mode, driver="fileobj")
        return _ManagedH5pyFile(handle, fileobj, gc_paused=pause)
    except BaseException:
        # Everything the pause covers runs in here, so an interrupt at any
        # point still rebalances. Nested so nothing raised while closing,
        # including a BaseException, can skip the resume.
        try:
            if close_on_error:
                with suppress(Exception):
                    fileobj.close()
        finally:
            if pause:
                resume_gc()
        raise


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
        # A user-supplied fsspec file object delegates reads to the same
        # event-loop thread as the UPath branch below and needs the same
        # GC pause; plain local/in-memory streams do not.
        return _open_h5_fileobj(
            resource,
            constructor,
            mode,
            pause=_is_loop_backed(resource),
            close_on_error=False,
        )
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
        # Note: only mode == "r" is a supported remote path here; H5Writer
        # intercepts UPath targets with its temp-file write-back handle.
        file_mode = "rb" if mode == "r" else "r+b"
        open_kwargs = open_kwargs_getter(resource)
        handle = _FallbackFileObj(
            remote_opener=lambda: resource.open(file_mode, **open_kwargs),
            local_opener=lambda: ensure_local_file(resource).open(file_mode),
            error_predicate=is_no_range_http_error,
        )
        # h5py holds its global lock while blocking on fsspec's event-loop
        # thread for remote fetches; an automatic garbage collection on that
        # thread deallocating h5py objects then deadlocks on the same lock.
        # Pause collection for the handle's lifetime (resumed in close()).
        return _open_h5_fileobj(
            handle,
            constructor,
            mode,
            pause=_is_loop_backed(resource),
            close_on_error=True,
        )
    try:
        if mode != "r":
            _maybe_make_parent_directory(resource)
        return _ManagedH5pyFile(constructor(resource, mode=mode))
    except TypeError:
        msg = f"Couldn't get handle from {resource} using h5py"
        raise NotImplementedError(msg)


# FiberIO read/scan/get_format annotate the *caster* class (H5Reader and
# friends): the io machinery swaps the annotated resource for whatever
# `get_handle` returns, so what those methods actually receive is the
# managed handle. Inheriting it while type checking makes the annotation
# describe the value the method really gets; nothing is instantiated at
# runtime, where the casters stay plain classes.
_H5CasterBase = _ManagedH5pyFile if TYPE_CHECKING else object


class H5Reader(_H5CasterBase):
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
        if protocol not in remote_hdf5_tuned_protocols:
            return {}
        # One snapshot: config is swappable, and reading the size and the
        # block count separately could pair one setting with the other's
        # replacement, for a cap neither configuration asked for.
        config = get_config()
        # h5py performs many small seeks while opening HDF5 metadata, and
        # remote backends default to large readahead blocks (s3fs uses 50 MB)
        # which can pull most of a file just to satisfy those probes.
        out = {"block_size": config.remote_hdf5_block_size}
        if protocol not in http_protocols:
            return out | {"cache_type": "readahead"}
        # HTTP needs a block LRU instead: the probe alternates between the
        # file header and footer, and fsspec's default single-window cache
        # refetches a full block (or the whole file on range-less servers) on
        # every jump. A few blocks keep both ends resident; the cap bounds
        # what one open handle retains.
        max_blocks = config.remote_hdf5_max_blocks
        return out | {
            "cache_type": "blockcache",
            "cache_options": {"maxblocks": max_blocks},
        }

    @classmethod
    def get_handle(cls, resource):
        """
        Get the HDF5 handle from local paths, remote paths, or open handles.

        h5py can consume a binary file object via the
        ``fileobj`` driver, so remote UPath inputs stay streaming-based here.
        """
        if isinstance(resource, (cls, _ManagedH5pyFile)):
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

        def __init__(self, resource: UPath, mode: str):
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

        def __setitem__(self, key, value):
            self._handle[key] = value

        def __contains__(self, item):
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

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
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
