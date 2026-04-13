"""
Tests for compatibility module.
"""

from __future__ import annotations

import numpy as np

from dascore.compat import array, is_array_like


class ArrayWithFlags:
    """Array-like object exposing array protocol and writable flags."""

    def __init__(self, data):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype
        self.flags = type("Flags", (), {"writeable": True})()

    def __array__(self, dtype=None, copy=None):
        out = self._data
        if dtype is not None:
            out = out.astype(dtype)
        if copy:
            out = out.copy()
        return out


class ArrayNoFlags:
    """Array-like object exposing array protocol without writable flags."""

    def __init__(self, data):
        self._data = data
        self.shape = data.shape
        self.dtype = data.dtype

    def __array__(self, dtype=None, copy=None):
        out = self._data
        if dtype is not None:
            out = out.astype(dtype)
        if copy:
            out = out.copy()
        return out


class ArrayNamespaceOnly:
    """Array-like object identified through the array API namespace hook."""

    def __init__(self, data):
        self.shape = data.shape
        self.dtype = data.dtype

    def __array_namespace__(self):
        return np


class ReadOnlyFlags:
    """Simple flags object exposing a writable attribute."""

    def __init__(self, writeable=False):
        self.writeable = writeable


class RaisingFlags:
    """Flags object whose writable setter always raises."""

    @property
    def writeable(self):
        return True

    @writeable.setter
    def writeable(self, value):
        raise ValueError("cannot change writeable")


class ArraySubclass(np.ndarray):
    """Simple ndarray subclass for compatibility tests."""


class TestIsArrayLike:
    """Tests for identifying array-like objects."""

    def test_array_protocol_object(self):
        """Objects with array protocol should be considered array-like."""
        wrapped = ArrayNoFlags(np.arange(6).reshape(2, 3))
        assert is_array_like(wrapped)

    def test_array_namespace_object(self):
        """Objects with array API namespace hook should be array-like."""
        wrapped = ArrayNamespaceOnly(np.arange(6).reshape(2, 3))
        assert is_array_like(wrapped)

    def test_non_array_like_object(self):
        """Objects without required attrs/protocols should not pass."""
        assert not is_array_like(object())


class TestArray:
    """Tests for creating immutable arrays."""

    def test_numpy_array_is_marked_read_only(self):
        """Ensure plain numpy arrays remain immutable."""
        data = np.arange(5.0)
        out = array(data)
        assert out is data
        assert not out.flags.writeable

    def test_numpy_subclass_is_preserved_and_marked_read_only(self):
        """Ensure numpy subclasses retain their type and become read-only."""
        data = np.arange(5.0).view(ArraySubclass)
        out = array(data)
        assert type(out) is ArraySubclass
        assert out is data
        assert not out.flags.writeable

    def test_array_like_with_flags_has_writeable_disabled(self):
        """Ensure array-like objects with flags get writeable set to false."""
        wrapped = ArrayWithFlags(np.arange(6).reshape(2, 3))
        out = array(wrapped)
        assert out is wrapped
        assert not out.flags.writeable

    def test_array_like_with_read_only_flags_is_unchanged(self):
        """Ensure already read-only flags stay read-only without error."""
        wrapped = ArrayWithFlags(np.arange(6).reshape(2, 3))
        wrapped.flags = ReadOnlyFlags(writeable=False)
        out = array(wrapped)
        assert out is wrapped
        assert not out.flags.writeable

    def test_array_like_with_raising_flag_setter_is_preserved(self):
        """Ensure failing writable setters do not force coercion or raise."""
        wrapped = ArrayWithFlags(np.arange(6).reshape(2, 3))
        wrapped.flags = RaisingFlags()
        out = array(wrapped)
        assert out is wrapped
        assert out.flags.writeable

    def test_array_like_without_flags_is_preserved(self):
        """Ensure array-like objects without flags are preserved as-is."""
        wrapped = ArrayNoFlags(np.arange(6).reshape(2, 3))
        out = array(wrapped)
        assert out is wrapped

    def test_non_array_like_is_converted_to_numpy_array(self):
        """Ensure non-array-like inputs still get converted to numpy arrays."""
        out = array([1, 2, 3])
        assert isinstance(out, np.ndarray)
        assert not out.flags.writeable
