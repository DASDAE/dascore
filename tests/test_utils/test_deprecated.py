"""
Tests for the deprecate decorator.
"""

import pytest

from dascore.utils.deprecate import deprecate


class TestDeprecated:
    """Tests for the deprecated decorator."""

    @pytest.fixture(scope="class")
    def deprecated_func(self):
        """Create a deprecated function."""

        @deprecate("Too old, use new()", since="0.1.0", removed_in="0.2.0")
        def old_func(*args, **kwargs):
            """Just a dummy function."""
            return 42

        return old_func

    def test_warning_issued(self, deprecated_func):
        """Ensure a warning is issued."""
        msg = "Too old"
        with pytest.warns(DeprecationWarning, match=msg):
            deprecated_func()

    def test_docstring(self, deprecated_func):
        """Ensure the docstring was updated."""
        assert "deprecated" in deprecated_func.__doc__
