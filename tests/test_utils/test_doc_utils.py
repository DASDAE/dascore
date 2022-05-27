"""Tests for docstring utils."""

import textwrap

from dascore.core.schema import PatchSummary
from dascore.utils.docs import compose_docstring, format_dtypes


class TestFormatDtypes:
    """Tests for formatting datatypes to display in docstrings."""

    def test_formatting(self):
        """Test for formatting StationDtypes."""
        out = format_dtypes(PatchSummary.__annotations__)
        assert isinstance(out, str)


class TestDocsting:
    """tests for obsplus' simple docstring substitution function."""

    def count_white_space(self, some_str):
        """count the number of whitespace chars in a str"""
        return len(some_str) - len(some_str.lstrip(" "))

    def test_docstring(self):
        """Ensure docstrings can be composed with the docstring decorator."""
        params = textwrap.dedent(
            """
        Parameters
        ----------
        a: int
            a
        b int
            b
        """
        )

        @compose_docstring(params=params)
        def testfun1():
            """
            {params}
            """

        assert "Parameters" in testfun1.__doc__
        line = [x for x in testfun1.__doc__.split("\n") if "Parameters" in x][0]
        base_spaces = line.split("Parameters")[0]
        assert len(base_spaces) == 12

    def test_list_indent(self):
        """Ensure lists are indented equally."""
        str_list = ["Hey", "who", "moved", "my", "cheese!?"]

        @compose_docstring(params=str_list)
        def dummy_func():
            """
            Some useful information indeed:
                {params}
            """

        doc_str_list = dummy_func.__doc__.split("\n")
        # the number of spaces between each list element should be the same.
        list_lines = doc_str_list[2:-1]
        white_space_counts = [self.count_white_space(x) for x in list_lines]
        # all whitespace counts should be the same for the list lines.
        assert len(set(white_space_counts)) == 1
