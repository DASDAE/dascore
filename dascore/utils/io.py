"""Utilities for basic IO tasks."""
from __future__ import annotations

import abc
import typing
from contextlib import suppress
from functools import cache, singledispatch
from inspect import isfunction, ismethod
from pathlib import Path
from typing import Any, Protocol, get_type_hints

import tables

from dascore.utils.misc import cached_method, register_func

HANDLE_FUNCTIONS = {}


RequiredType = typing.TypeVar("RequiredType")


@typing.runtime_checkable
class BinaryWriter(Protocol):
    """Dummy class for streams which write binary."""

    closed: bool

    @abc.abstractmethod
    def read(self, *args, **kwargs):
        """Read resource contents."""

    @abc.abstractmethod
    def seek(self, *args, **kwargs):
        """Seek to specific place in file."""

    @abc.abstractmethod
    def close(self, *args, **kwargs):
        """Close resource."""


@typing.runtime_checkable
class BinaryReader(Protocol):
    """Dummy class for streams which read binary."""

    closed: bool

    @abc.abstractmethod
    def write(self, *args, **kwargs):
        """Write resource contents."""

    @abc.abstractmethod
    def seek(self, *args, **kwargs):
        """Seek to specific place in file."""

    @abc.abstractmethod
    def close(self, *args, **kwargs):
        """Close resource."""


@typing.runtime_checkable
class HDF5Writer(Protocol):
    """An HDF 5 file open in write mode."""

    root: Any
    params: Any
    title: str
    filters: Any
    isopen: bool
    filename: str


@typing.runtime_checkable
class HDF5Reader(Protocol):
    """An HDF 5 file open in read mode."""

    root: Any
    params: Any
    title: str
    filters: Any
    isopen: bool
    filename: str


@cache
def _get_required_type(required_type, arg_name=None):
    """Get the type hint for the first argument."""
    if required_type not in HANDLE_FUNCTIONS:
        # here we try to get the type from the function type hints
        # but we need to skip things that aren't functions
        is_func_y = isfunction(required_type) or ismethod(required_type)
        if not is_func_y or not (hints := get_type_hints(required_type)):
            return required_type
        arg_name = arg_name if arg_name is not None else next(iter(hints))
        return hints.get(arg_name)
    return required_type


def get_handle_from_resource(uri, required_type):
    """
    Get a handle for a file of preferred type.

    return uri if required type is not specified.
    """
    with suppress(TypeError):
        if isinstance(uri, required_type):
            return uri
    if (func := HANDLE_FUNCTIONS.get(required_type)) is None:
        return uri
    return func(uri)


@register_func(HANDLE_FUNCTIONS, key=Path)
def _get_path(uri):
    """Get a path from uri."""
    return Path(uri)


@register_func(HANDLE_FUNCTIONS, key=str)
def _get_str(uri):
    """Get a string from uri."""
    return str(uri)


@register_func(HANDLE_FUNCTIONS, key=BinaryReader)
@singledispatch
def _get_buffered_reader(uri):
    """Get a buffered reader from unique resource."""
    raise NotImplementedError(f"not implemented for type {type(uri)}")


@_get_buffered_reader.register(str)
@_get_buffered_reader.register(Path)
def _uri_to_buffered_reader(path_like):
    """Get a read buffer from as string/path."""
    fi = open(path_like, "rb")
    return fi


@register_func(HANDLE_FUNCTIONS, key=BinaryWriter)
@singledispatch
def _get_buffered_writer(uri):
    """Get a buffered reader from unique resource."""
    raise NotImplementedError(f"not implemented for type {type(uri)}")


@_get_buffered_writer.register(str)
@_get_buffered_writer.register(Path)
def _uri_to_buffered_writer(path_like):
    """Get a write buffer from as string/path."""
    fi = open(path_like, "wb")
    return fi


@register_func(HANDLE_FUNCTIONS, key=HDF5Reader)
@singledispatch
def _get_hdf5_reader(uri):
    """Get a buffered reader from unique resource."""
    raise NotImplementedError(f"not implemented for type {type(uri)}")


@_get_hdf5_reader.register(str)
@_get_hdf5_reader.register(Path)
def _uri_to_hdf_reader(path_like):
    """Get a read buffer from as string/path."""
    fi = tables.open_file(path_like, "r")
    return fi


@register_func(HANDLE_FUNCTIONS, key=HDF5Writer)
@singledispatch
def _get_hdf5_writer(uri):
    """Get a buffered reader from unique resource."""
    raise NotImplementedError(f"not implemented for type {type(uri)}")


@_get_hdf5_writer.register(str)
@_get_hdf5_writer.register(Path)
def _uri_to_hdf5_writer(path_like):
    """Get a write buffer from as string/path."""
    # ensure the file structure exists
    Path(path_like).parent.mkdir(exist_ok=True, parents=True)
    fi = tables.open_file(path_like, "a")
    return fi


class IOResourceManager:
    """A class for managing opening/closing files."""

    def __init__(self, source: Any):
        self._source = source
        self._cache = {}

    @property
    @cached_method
    def source(self):
        """Get the source of the IO manager."""
        source = self._source
        # this handles IO managers derived from other IO managers;
        # effectively, we need to go back to the original, non-io manager source
        while isinstance(source, self.__class__):
            source = source.source
        return source

    def get_resource(self, required_type: RequiredType) -> RequiredType:
        """Get the requested resource from."""
        # no required type, just return source of manager.
        if required_type is None:
            return self.source
        # this is so the context managers can be nested and the child
        # context manager only calls to the parent. Then, the resources
        # get closed only after the original exists its context.
        if isinstance(self._source, self.__class__):
            return self._source.get_resource(required_type)
        required_type = _get_required_type(required_type)
        if required_type not in self._cache:
            out = get_handle_from_resource(self._source, required_type)
            self._cache[required_type] = out
        return self._cache[required_type]

    def __enter__(self):
        """Entering context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Simply ensure all file handles are closed."""
        for handle in self._cache.values():
            getattr(handle, "close", lambda: None)()
