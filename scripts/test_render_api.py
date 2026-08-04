"""Tests for rendering api stuff."""

from __future__ import annotations

import inspect
import typing

import pytest

# These tests only work if doc deps are installed.
pytest.importorskip("jinja2")

from _render_api import build_signature, get_type_hints, to_quarto_code  # noqa


class TestGetTypeHints:
    """Tests for resolving the type hints of documented objects."""

    def test_resolvable_hints(self):
        """Annotations which resolve should still return their objects."""

        def func(a: int) -> str:
            """A documented function."""

        hints = get_type_hints(func)
        assert hints["a"] is int
        assert hints["return"] is str

    def test_type_checking_only_annotation(self):
        """
        Annotations imported only under TYPE_CHECKING can't be resolved when
        the docs are built, but they shouldn't break the build.
        """

        class Klass:
            """A class annotated with a name missing at runtime."""

            attr: OnlyImportedWhileTypeChecking  # noqa: F821

        # The un-guarded call is what used to kill the doc build.
        with pytest.raises(NameError):
            typing.get_type_hints(Klass)
        assert get_type_hints(Klass) == {"attr": "OnlyImportedWhileTypeChecking"}

    def test_signature_of_type_checking_annotated_class(self):
        """Spool annotates a TYPE_CHECKING-only import; it must still render."""
        from dascore.core.spool import Spool

        data = {
            "signature": inspect.signature(Spool),
            "object": Spool,
            "name": "Spool",
        }
        out = build_signature(data, {}, {})
        assert "<b>Spool</b>" in out
        assert "data" in out


class TestToQuartoCode:
    """Tests for code parsing to quarto-style code strings."""

    def test_basic(self):
        """Ensure a simple example works."""
        code = """
        print("hey")
        """
        out = to_quarto_code(code)
        assert '```{python}\nprint("hey")\n```' == out

    def test_docstring(self):
        """Ensure docstring works."""
        code = """
        >>> print("bob")
        >>> for a in range(10):
        ...     print(a)
        """
        out = to_quarto_code(code)
        assert "    print(a)" in out.splitlines()

    def test_output_handled(self):
        """Docstrings can have outputs in them, we need to strip them out."""
        code = """
        >>> print("bob")
        bob
        """
        out = to_quarto_code(code)
        assert '```{python}\nprint("bob")\n```' == out

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
        """Ensure quarto options carry forward."""
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
