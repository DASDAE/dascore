"""Tests for docstring utils."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

import dascore.utils.namespace as ns_module
from dascore.core.attrs import PatchAttrs
from dascore.examples import EXAMPLE_PATCHES
from dascore.utils.docs import (
    compose_docstring,
    format_dtypes,
    get_docstring,
    get_plugin_table,
    objs_to_doc_df,
)


class TestFormatDtypes:
    """Tests for formatting datatypes to display in docstrings."""

    def test_formatting(self):
        """Test for formatting StationDtypes."""
        out = format_dtypes(PatchAttrs.__annotations__)
        assert isinstance(out, str)


class TestDocstring:
    """tests for DASCore's simple docstring substitution function."""

    def count_white_space(self, some_str):
        """Count the number of whitespace chars in a str."""
        return len(some_str) - len(some_str.lstrip(" "))

    def test_docstring(self):
        """Ensure docstrings can be composed with the docstring decorator."""
        params = textwrap.dedent(
            """
        Parameters
        ----------
        a
            a
        b
            b
        """
        )

        @compose_docstring(params=params)
        def testfun1():
            """
            A simple test function.

            {params}
            """

        assert "Parameters" in testfun1.__doc__
        line = next(x for x in testfun1.__doc__.split("\n") if "Parameters" in x)
        base_spaces = line.split("Parameters")[0]
        # py3.13+ automatically strips white space from docstrings so 12
        # and 0 are valid lengths.
        assert len(base_spaces) in {12, 0}

    def test_list_indent(self):
        """Ensure lists are indented equally."""
        str_list = ["Hey", "who", "moved", "my", "cheese!?"]

        @compose_docstring(params=str_list)
        def dummy_func():
            """
            Some useful information indeed:
                {params}.
            """

        doc_str_list = dummy_func.__doc__.split("\n")
        # the number of spaces between each list element should be the same.
        list_lines = doc_str_list[2:-1]
        white_space_counts = [self.count_white_space(x) for x in list_lines]
        # all whitespace counts should be the same for the list lines.
        assert len(set(white_space_counts)) == 1

    def test_raises_when_no_placeholders_are_replaced(self):
        """Unused substitutions should fail if nothing in the docstring matches."""
        with pytest.raises(ValueError, match="did not replace any placeholders"):

            @compose_docstring(params="value")
            def dummy_func():
                """A docstring with no replacement slots."""

    def test_raises_when_some_placeholders_are_unused(self):
        """Providing extra substitution keys should fail loudly."""
        with pytest.raises(ValueError, match=r"unused keys.*extra"):

            @compose_docstring(params="value", extra="not-used")
            def dummy_func():
                """
                A docstring with one replacement slot.

                {params}
                """


class TestGetDocstring:
    """Tests for pulling a docstring out of an object."""

    def test_returns_docstring(self):
        """The docstring of a documented object is returned unchanged."""

        def documented():
            """Some words."""

        assert get_docstring(documented) == documented.__doc__

    def test_raises_when_undocumented(self):
        """An object with no docstring is a source error, not a None value."""

        def undocumented(): ...

        with pytest.raises(AssertionError, match="has no docstring"):
            get_docstring(undocumented)


class TestGetPluginTable:
    """Tests for get_plugin_table."""

    def test_contains_registered_namespace(self):
        """Registered namespaces should appear in the returned DataFrame."""
        df = get_plugin_table()
        assert "zug" in df["namespace"].values
        assert "derzug" in df["package_name"].values

    def test_one_row_per_host(self):
        """A namespace registered on two hosts keeps a row for each."""
        df = get_plugin_table()
        rows = df[df["namespace"] == "zug"]
        assert set(rows["host"]) == {"patch", "spool"}
        assert len(rows) == 2

    def test_empty_registry_returns_empty_dataframe(self, monkeypatch, tmp_path):
        """An empty registry directory returns a DataFrame with the correct columns."""
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        df = get_plugin_table()
        assert list(df.columns) == ["host", "namespace", "package_name", "package_url"]
        assert df.empty


class TestObjToDocDF:
    """Tests for generating documentation dataframes."""

    def test_examples_cross_ref(self):
        """Tests for documenting examples with cross references."""
        df = objs_to_doc_df(EXAMPLE_PATCHES, cross_reference=True)
        assert "(`dascore.examples" in df["Name"].iloc[0]
        assert isinstance(df, pd.DataFrame)

    def test_example_no_cross_ref(self):
        """Tests for documenting examples without cross references."""
        df = objs_to_doc_df(EXAMPLE_PATCHES, cross_reference=False)
        assert "(`dascore.examples" not in df["Name"].iloc[0]
        assert isinstance(df, pd.DataFrame)
