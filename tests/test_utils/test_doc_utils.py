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
    get_doc_anchor,
    get_docstring,
    get_plugin_table,
    iter_public,
    objs_to_doc_df,
    render_module_api,
    render_package_api,
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

    def test_empty_registry_returns_empty_dataframe(self, monkeypatch, tmp_path):
        """An empty registry directory returns a DataFrame with the correct columns."""
        monkeypatch.setattr(ns_module, "_PLUGIN_REGISTRY_DIR", tmp_path)
        df = get_plugin_table()
        assert list(df.columns) == ["namespace", "package_name", "package_url"]
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


class TestRenderApi:
    """Tests for rendering a module's API onto one documentation page."""

    @pytest.fixture(scope="class")
    def misc_markdown(self):
        """Rendered markdown for a module with a mix of documented objects."""
        return render_module_api("dascore.utils.misc")

    def test_anchor_is_stable_and_html_safe(self):
        """Anchors must be reproducible so the inventory can point at them."""
        anchor = get_doc_anchor("dascore.utils.misc.iterate")
        assert anchor == "dascore-utils-misc-iterate"
        assert anchor == get_doc_anchor("dascore.utils.misc.iterate")

    def test_entry_has_anchor_signature_and_summary(self, misc_markdown):
        """Each entry needs the pieces a reader and the inventory rely on."""
        assert "#### iterate {#dascore-utils-misc-iterate}" in misc_markdown
        assert "iterate(" in misc_markdown

    def test_details_are_collapsed(self, misc_markdown):
        """Parameters and examples sit in a collapsed callout, not inline."""
        assert 'collapse="true"' in misc_markdown
        assert "| Parameter | Type | Description |" in misc_markdown

    def test_only_objects_defined_in_the_module(self, misc_markdown):
        """Imported names belong to the module which defines them."""
        # misc imports numpy as np; it should not document numpy.
        assert "#### np " not in misc_markdown

    def test_package_render_covers_every_module(self):
        """Every non-private module with public objects gets a section."""
        markdown = render_package_api("dascore.utils")
        assert "### dascore.utils.misc {#dascore-utils-misc}" in markdown
        assert "### dascore.utils.patch {#dascore-utils-patch}" in markdown

    def test_decorated_helpers_are_documented(self):
        """Cached helpers are callables, not functions, and must not vanish."""
        names = {name for name, _ in iter_public("dascore.utils.downloader")}
        assert "get_registry_df" in names

    def test_griffe_models_never_reach_the_page(self):
        """Unhandled docstring sections must not render object reprs."""
        # ChunkPlan documents an Attributes section, which has its own model.
        markdown = render_module_api("dascore.utils.chunk_plan")
        assert "object at 0x" not in markdown

    def test_an_example_which_is_a_code_block_stays_one_block(self):
        """A fenced example must not close the fence the page wraps it in."""
        # compose_docstring's example is itself a fenced block. Wrapping it in
        # a fence of the same length ends the block at the example's own
        # opening fence, so the example body lands outside any block and the
        # rest of the page is swallowed by the next fence it meets.
        markdown = render_module_api("dascore.utils.docs")
        inside, fence = set(), ""
        for line in markdown.splitlines():
            if line.startswith("```") and not fence:
                fence = line[: len(line) - len(line.lstrip("`"))]
            elif fence and line.strip() == fence:
                fence = ""
            elif fence:
                inside.add(line)
        assert not fence, "a code block was left open"
        assert "from dascore.utils.docs import compose_docstring" in inside

    def test_examples_are_shown_rather_than_executed(self):
        """A `{python}` cell would run when the page is rendered."""
        markdown = render_package_api("dascore.utils")
        assert "```{python}" not in markdown

    def test_every_callout_is_closed(self):
        """An unclosed callout swallows the entries which follow it."""
        markdown = render_package_api("dascore.utils")
        lines = markdown.splitlines()
        opened = sum(1 for line in lines if line.startswith(":::") and "{" in line)
        closed = sum(1 for line in lines if set(line.strip()) == {":"})
        assert opened == closed
        assert "| Attribute | Type | Description |" in markdown
