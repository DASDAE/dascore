"""Tests for the cross-reference filter."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pytest

fill_links = pytest.importorskip("fill_links")


class TestFindDocsPath:
    """Tests for finding the cross-reference file."""

    @pytest.fixture
    def docs_tree(self, tmp_path, monkeypatch):
        """A docs directory nested in a path which says "docs" twice."""
        docs = tmp_path / "worktrees" / "repr-docs" / "dascore" / "docs"
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


class TestFilterCommand:
    """The filter transforms UTF-8 Pandoc input through its command entry point."""

    def test_stdio(self, tmp_path, monkeypatch, capsys):
        """Repeated anchored links are rewritten without losing Unicode labels."""
        source = Path(fill_links.__file__)
        docs = tmp_path / "docs"
        filters = docs / "filters"
        filters.mkdir(parents=True)
        (docs / ".cross_ref.json").write_text(
            json.dumps({"target": "/tutorial/spool.qmd"}), encoding="utf-8"
        )
        link = {
            "t": "Link",
            "c": [
                ["", [], []],
                [{"t": "Str", "c": "✅"}],
                ["%60target%60#concatenate", ""],
            ],
        }
        data = {"blocks": [link, link]}
        raw = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        stdin = io.TextIOWrapper(io.BytesIO(raw.encode("utf-8")), encoding="utf-8")
        monkeypatch.setattr(sys, "stdin", stdin)
        namespace = {"__name__": "__main__", "__file__": str(filters / source.name)}
        exec(
            compile(source.read_text(encoding="utf-8"), str(source), "exec"), namespace
        )
        result = json.loads(capsys.readouterr().out)
        for item in result["blocks"]:
            assert item["c"][-1][0] == "/tutorial/spool.qmd#concatenate"
            assert item["c"][1][0]["c"] == "✅"
