"""Tests for rendering api stuff"""
import pytest

# These tests only work if doc deps are installed.
pytest.importorskip("jinja2")  # noqa

from _render_api import to_quarto_code  # noqa


class TestToQuartoCode:
    """Tests for code parsing to quarto-style code strings."""

    def test_basic(self):
        """Ensure a simple example works."""
        code = """
        print("hey")
        """
        out = to_quarto_code(code)
        assert "```{python}" in out

    def test_docstring(self):
        """Ensure docstring works."""
        code = """
        >>> print("bob")
        >>> for a in range(10):
        ...     print(a)
        """
        out = to_quarto_code(code)
        assert "    print(a)" in out.splitlines()

    def test_titles(self):
        """Ensure titles are carried forward."""
        code1 = """
        >>> ### Simple example
        >>> print("a")
        >>>
        >>> ### More complex example
        >>> print(1 + 2)
        """
        out1 = to_quarto_code(code1)
        code2 = """

        ### Simple example
        print("a")
        ### More complex example
        print(1 + 2)

        """
        out2 = to_quarto_code(code2)
        assert out1 == out2

    def test_options(self):
        """Ensure quarto options cary forward."""
        code1 = """
        >>> #| fold: true
        >>> print("bob")
        >>>
        >>> ### Another example
        >>> print("bill")
        """
        out = to_quarto_code(code1)
        expected_str = "#| fold: true"
        assert expected_str in out
        assert out.count(expected_str) == 2

    def test_combination(self):
        """A combination of stuff."""
        code1 = """
        >>> #| code-fold: true
        >>> # This is a base example
        >>> print(1 + 2)
        >>> ### This is a sub-section
        >>> print("cool beans")
        """
        out = to_quarto_code(code1)
        assert out
