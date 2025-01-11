"""Misc. tests for misfit utilities."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.misc import (
    MethodNameSpace,
    cached_method,
    get_stencil_coefs,
    iterate,
    maybe_get_items,
    optional_import,
    to_object_array,
    warn_or_raise,
)


class ParentClass:
    """A test parent class."""

    @property
    def namespace(self):
        """Your run-o-the-mill namespace."""
        return MNS(self)


class MNS(MethodNameSpace):
    """method name space subclass."""

    def func1(self, expected_type):
        """First func."""
        return isinstance(self, expected_type)


class TestNamespaceClass:
    """Tests for namespace class."""

    def test_parent_self_passed_to_namespace(self):
        """Ensure the parent of namespace gets passed to self."""
        pc = ParentClass()
        assert pc.namespace.func1(ParentClass)

    def test_assign_adhoc_method(self):
        """Ensure methods added after class definition still work."""

        def new_method(self, expected_type):
            return isinstance(self, expected_type)

        MNS.new_method = new_method
        pc = ParentClass()
        assert pc.namespace.new_method(ParentClass)


class TestIterate:
    """Test case for iterate."""

    def test_none(self):
        """None should return an empty tuple."""
        assert iterate(None) == tuple()

    def test_object(self):
        """A single object should be returned in a tuple."""
        assert iterate(1) == (1,)

    def test_str(self):
        """A single string object should be returned as a tuple."""
        assert iterate("hey") == ("hey",)


class TestOptionalImport:
    """Ensure the optional import works."""

    def test_import_installed_module(self):
        """Test to ensure an installed module imports."""
        import dascore as dc

        mod = optional_import("dascore")
        assert mod is dc
        sub_mod = optional_import("dascore.core")
        assert sub_mod is dc.core

    def test_missing_module_raises(self):
        """Ensure a module which is missing raises the appropriate Error."""
        with pytest.raises(MissingOptionalDependencyError, match="boblib4"):
            optional_import("boblib4")


class TestGetStencilCoefficients:
    """Tests for stencil coefficients."""

    def test_3_point_1st_derivative(self):
        """3 point 1st derivative."""
        out = get_stencil_coefs(1, 1)
        expected = np.array([-1 / 2, 0, 1 / 2])
        assert np.allclose(out, expected)

    def test_5_point_1st_derivative(self):
        """5 point 1st derivative."""
        out = get_stencil_coefs(2, 1)
        expected = np.array([1, -8, 0, 8, -1]) / 12.0
        assert np.allclose(out, expected)

    def test_3_point_2nd_derivative(self):
        """3 point 2nd derivative."""
        out = get_stencil_coefs(1, 2)
        expected = np.array([1, -2, 1])
        assert np.allclose(out, expected)


class TestCachedMethod:
    """Ensure cached methods caches method calls (duh)."""

    class _JohnnyCached:
        """Tests class for caching."""

        @cached_method
        def no_args(self):
            """No argument method."""
            return {"output", "defined"}

        @cached_method
        def multiargs(self, a, b):
            """Multiple arguments for cache testing."""
            return a + b

    def test_no_args_kwargs(self):
        """Ensure objects cache without args or kwargs."""
        john = self._JohnnyCached()
        first = john.no_args()
        assert first is john.no_args()

    def test_positional(self):
        """Ensure positional arguments work."""
        john = self._JohnnyCached()
        out = john.multiargs(1, 2)
        assert john.multiargs(1.0, 2.0) == 3.0
        assert out == 3

    def test_kwargs(self):
        """Ensure keywords also work."""
        john = self._JohnnyCached()
        assert john.multiargs(1, b=1) == 2
        assert john.multiargs(a=2, b=3) == 5


class TestMaybeGetItems:
    """Tests for maybe_get_attrs."""

    def test_missed_itme(self):
        """Ensure it still works when a key is missing."""
        data = {"bob": 1, "bill": 2}
        expected = {"bob": "sue", "lary": "who"}
        out = maybe_get_items(data, attr_map=expected)
        assert "sue" in out
        assert "who" not in out


class TestWarnOrRaise:
    """Ensure warn or raise works."""

    def test_warn(self):
        """Ensure a warning is emitted."""
        msg = "Warning: this is a warning"
        with pytest.warns(UserWarning, match=msg):
            warn_or_raise(msg, warning=UserWarning, behavior="warn")

    def test_raise(self):
        """Ensure an exception can be raised."""
        msg = "Something went wrong."
        with pytest.raises(ValueError, match=msg):
            warn_or_raise(msg, exception=ValueError, behavior="raise")

    def test_nothing(self):
        """Ensure when  None does nothing."""
        msg = "Big nothing burger"
        # Now exceptions or warnings will crash the program.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            warn_or_raise(msg, behavior=None)


class TestToObjectArray:
    """Tests for converting a sequence of objects to an object array."""

    def test_patches_to_array(self, random_patch):
        """Ensure a list of patches can be converted to an object array."""
        patches = [random_patch] * 3
        out = to_object_array(patches)
        assert isinstance(out, np.ndarray)
