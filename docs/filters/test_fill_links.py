"""Tests for the cross-reference filter."""

from __future__ import annotations

import json

import pytest

fill_links = pytest.importorskip("fill_links")


class TestFindDocsPath:
    """Tests for finding the cross-reference file."""

    @pytest.fixture
    def docs_tree(self, tmp_path, monkeypatch):
        """A docs directory nested in a path which says "docs" twice."""
        docs = tmp_path / "worktrees" / "repr-docs" / "docs"
        filters = docs / "filters"
        filters.mkdir(parents=True)
        (docs / ".cross_ref.json").write_text(json.dumps({"a": "/api/a.qmd"}))
        monkeypatch.setattr(fill_links, "__file__", str(filters / "fill_links.py"))
        return docs

    def test_found_by_its_file(self, docs_tree):
        """The docs directory is the one holding the cross references."""
        assert fill_links._find_docs_path() == docs_tree

    def test_no_cross_ref_file(self, tmp_path, monkeypatch):
        """A build which never generated one is told so, not left looping."""
        monkeypatch.setattr(fill_links, "__file__", str(tmp_path / "fill_links.py"))

        with pytest.raises(ValueError, match="failed to find cross-ref file"):
            fill_links._find_docs_path()
