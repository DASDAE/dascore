"""
Tests for compatibility module.
"""

import builtins
import importlib
from functools import cache

import numpy as np
import pytest

from dascore.compat import maybe_jit
from dascore.utils.misc import optional_import


class TestMaybeJit:
    """Tests for optional jit'ing."""

    @pytest.fixture(scope="function")
    def hide_numba(self, monkeypatch):
        """Makes the function under test think numba is missing."""
        import_orig = builtins.__import__

        def mocked_import(name, *args, **kwargs):
            if name == "pkg":
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mocked_import)
        yield
        importlib.reload(builtins)

    def test_jit(self):
        """Ensure the jit works. Only test if numba installed."""
        pytest.importorskip("numba")

        @maybe_jit
        def my_jit(ar):
            return ar

        ar = np.array([1, 2, 3])
        new_ar = my_jit(ar)
        assert np.all(new_ar, ar)

    def test_warning(self, hide_numba):
        """When numba is not installed ensure a warning is issued."""

        @maybe_jit()
        def _jit_test_func(ar):
            return ar

        match = "can be compiled to improve performance"
        with pytest.warns(UserWarning, match=match):
            _jit_test_func(np.array([1, 2, 3]))

    def test_raises(self, hide_numba):
        """Ensure an error is raised when specified by the decorator."""

        @maybe_jit(required=True)
        def _jit_test_func(ar):
            return ar

        match = "requires python module"
        with pytest.raises(ImportError, match=match):
            _jit_test_func(np.array([1, 2, 3]))

    def test_compiler_name(self):
        """Ensure the compiler name raises if not supported."""
        with pytest.raises(NotImplementedError):

            @maybe_jit(compiler="fancy_compiler")
            def _dummy_func(ar):
                return ar

    def test_example(self):
        """Test docstring examples."""
        pytest.importorskip("numba")

        @maybe_jit(nopython=True, nogil=True)
        def _jit_func(array):
            return array

        @cache
        def jit_wrapper():
            numba = optional_import("numba")

            @maybe_jit
            def jit_func(array):
                for a in numba.prange(len(array)):
                    pass
                return array

            return jit_func

        out = jit_wrapper()(np.array([1, 2, 3]))
        assert isinstance(out, np.ndarray)
