"""
Tests for applying just in time compilations.
"""

from functools import cache

import numpy as np
import pytest

from dascore.utils.jit import maybe_numba_jit
from dascore.utils.misc import optional_import, suppress_warnings


class TestMaybeNumbaJit:
    """Tests for optional jit'ing."""

    def test_jit(self):
        """Ensure the jit works. Only test if numba installed."""
        pytest.importorskip("numba")

        @maybe_numba_jit
        def my_jit(ar):
            return ar

        ar = np.array([1, 2, 3])
        new_ar = my_jit(ar)
        assert np.all(new_ar == ar)

    def test_warning(self):
        """When numba is not installed ensure a warning is issued."""

        @maybe_numba_jit(_missing_numba=True)
        def _jit_test_func(ar):
            return ar

        match = "can be compiled to improve performance"
        with pytest.warns(UserWarning, match=match):
            _jit_test_func(np.array([1, 2, 3]))

    def test_raises(self):
        """Ensure an error is raised when specified by the decorator."""

        @maybe_numba_jit(required=True, _missing_numba=True)
        def _jit_test_func(ar):
            return ar

        match = "requires python module"
        with pytest.raises(ImportError, match=match):
            _jit_test_func(np.array([1, 2, 3]))

    def test_example(self):
        """Test docstring examples."""
        pytest.importorskip("numba")

        @maybe_numba_jit(nopython=True, nogil=True)
        def _jit_func(array):
            return array

        @cache
        def jit_wrapper():
            numba = optional_import("numba")

            @maybe_numba_jit
            def jit_func(array):
                for a in numba.prange(len(array)):
                    pass
                return array

            return jit_func

        out = jit_wrapper()(np.array([1, 2, 3]))
        assert isinstance(out, np.ndarray)

    def test_numba_used_in_function(self):
        """Tests for numba used in function without being imported."""
        pytest.importorskip("numba")

        @maybe_numba_jit(nopython=True, nogil=True)
        def _my_jit(array):
            for sub in numba.prange(len(array)):  # noqa
                pass
            return array

        array = np.array([1, 2, 3])
        out = _my_jit(array)
        assert np.all(out == array)

    def test_prange_no_numba(self):
        """
        In order to make the doctests work, we had to implement a dummy
        numba module. It doesn't support everything, (barely anything) but
        should be enough for now.
        """

        @maybe_numba_jit(_missing_numba=True)
        def _my_jit(array):
            for sub in numba.prange(len(array)):  # noqa
                pass
            return array

        array = np.array([1, 2, 3])

        with suppress_warnings(UserWarning):
            out = _my_jit(array)
        assert np.all(out == array)
