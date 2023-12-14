"""Utilities for basic IO tasks."""
from __future__ import annotations

import io
import typing
from functools import cache
from inspect import isfunction, ismethod
from pathlib import Path
from typing import Any, get_type_hints

from dascore.utils.misc import _maybe_make_parent_directory, cached_method

HANDLE_FUNCTIONS = {
    str: lambda x: str(x),
    Path: lambda x: Path(x),
}


RequiredType = typing.TypeVar("RequiredType")


class BinaryReader(io.BytesIO):
    """Base file for binary things."""

    mode = "rb"

    @classmethod
    def get_handle(cls, resource):
        """Get the handle object from various sources."""
        if isinstance(resource, cls | io.BufferedWriter):
            return resource
        try:
            _maybe_make_parent_directory(resource)
            return open(resource, mode=cls.mode)
        except TypeError:
            msg = f"Couldn't get handle from {resource} using {cls}"
            raise NotImplementedError(msg)


class BinaryWriter(BinaryReader):
    """Dummy class for streams which write binary."""

    mode = "ab"


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

    return uri if required type is not specified or supported in either
    handle function or has a `get_handle` method.
    """
    if hasattr(required_type, "get_handle"):
        uri = required_type.get_handle(uri)
    elif required_type in HANDLE_FUNCTIONS:
        uri = HANDLE_FUNCTIONS[required_type](uri)
    return uri


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

    def close_all(self):
        """Close any open file handles."""
        for handle in self._cache.values():
            getattr(handle, "close", lambda: None)()

    def __enter__(self):
        """Entering context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Simply ensure all file handles are closed."""
        self.close_all()

    def __del__(self):
        self.close_all()
