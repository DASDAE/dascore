"""Search finds installed APIs and procedures without a second documentation source."""

from __future__ import annotations

import json
import subprocess
import sys
from importlib.metadata import version

import pytest

import dascore as dc
from dascore.cli import main
from dascore.utils import doc_corpus, doc_search
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import DocumentationError


@pytest.fixture(scope="module")
def cache_directory(tmp_path_factory):
    """Share one real corpus and index across relevance tests."""
    return tmp_path_factory.mktemp("search-cache")


@pytest.fixture
def search_cache(cache_directory):
    """Keep search artifacts out of the user's documentation cache."""
    with dc.config_context(docs_cache_dir=cache_directory):
        yield cache_directory / dc.__version__


@pytest.fixture
def engine():
    """The native search engine is optional, including in WebAssembly tests."""
    return pytest.importorskip("tantivy")


@pytest.mark.usefixtures("engine")
class TestSearch:
    """Exercise relevance, exact tags, aliases, and human-readable results."""

    @pytest.mark.parametrize(
        "query, expected",
        [
            ("low pass", "dascore.proc.filter.pass_filter"),
            ("bandpass", "dascore.proc.filter.pass_filter"),
            ("median filtering", "dascore.proc.filter.median_filter"),
            ("overlapping chunks", "dascore.core.spool.Spool.chunk"),
            ("inventory geometry", "recipes/tunnel_inventory"),
            ("Patch.select", "dascore.proc.coords.select"),
        ],
    )
    def test_relevance(self, search_cache, query, expected):
        """Real task phrases find the relevant API or authored procedure."""
        results = doc_search.search_documents(query)
        identifiers = [x["id"] for x in results]
        assert expected in identifiers
        assert len(identifiers) == len(set(identifiers))
        with documentation_cache() as (root, manifest):
            for result in results:
                assert read_document(root, manifest, result["id"])
                assert result["excerpt"] and "\n" not in result["excerpt"]

    def test_tags(self, search_cache):
        """Tags match whole keywords, ignore case, and combine with text."""
        results = doc_search.search_documents(tag=" FiLtErInG ", limit=100)
        assert results
        assert all("filtering" in result["keywords"] for result in results)
        assert not doc_search.search_documents(tag="filter")
        matches = doc_search.search_documents("median", tag="filtering")
        assert matches and all("filtering" in x["keywords"] for x in matches)
        assert not doc_search.search_documents("zzznomatchzzz", tag="filtering")

    def test_limits(self, search_cache):
        """The requested limit bounds a broad result set."""
        assert len(doc_search.search_documents("patch", limit=2)) == 2

    def test_invalid_query(self, search_cache):
        """Malformed engine syntax becomes an actionable documentation error."""
        with pytest.raises(DocumentationError, match="Invalid search query"):
            doc_search.search_documents('"unterminated')

    def test_cli(self, search_cache, capsys):
        """CLI results include the exact read command and useful subject keywords."""
        assert (
            main(["doc-search", "bandpass", "--tag", "filtering", "--limit", "1"]) == 0
        )
        text = capsys.readouterr().out
        assert "Read: dascore doc dascore.proc.filter.pass_filter" in text
        assert "Keywords:" in text and "filtering" in text
        assert main(["doc-search", "zzznomatchzzz"]) == 0
        assert "No matching documentation" in capsys.readouterr().out
        assert main(["doc-search", '"unterminated']) == 1
        captured = capsys.readouterr()
        assert "Invalid search query" in captured.err and not captured.out

    def test_tag_only_excerpt(self, search_cache):
        """Keyword-only results still have useful text when no body terms match."""
        results = doc_search.search_documents(tag="inventory")
        assert results and all(x["excerpt"] for x in results)
        assert all(not x["excerpt"].startswith("# " + x["title"]) for x in results)


@pytest.mark.usefixtures("engine")
class TestIndex:
    """A disposable index follows the existing corpus generation and lock."""

    def test_reuse(self, search_cache):
        """A second search leaves the completed index untouched."""
        before = doc_search.search_documents("bandpass")
        marker = search_cache / "search" / "identity.json"
        stamp = marker.stat().st_mtime_ns
        assert doc_search.search_documents("bandpass") == before
        assert marker.stat().st_mtime_ns == stamp

    @pytest.mark.parametrize("damage", ["marker", "metadata", "version"])
    def test_recovery(self, search_cache, damage):
        """Interrupted, corrupt, and incompatible indexes rebuild from Markdown."""
        expected = doc_search.search_documents("bandpass")
        directory = search_cache / "search"
        if damage == "marker":
            (directory / "identity.json").unlink()
        elif damage == "metadata":
            (directory / "meta.json").write_text("{", encoding="utf-8")
        else:
            marker = directory / "identity.json"
            identity = json.loads(marker.read_text(encoding="utf-8"))
            identity["engine"] = "old-engine"
            marker.write_text(json.dumps(identity), encoding="utf-8")
        assert doc_search.search_documents("bandpass") == expected
        identity = json.loads((directory / "identity.json").read_text(encoding="utf-8"))
        assert identity["engine"] == version("tantivy")

    def test_native_failure(self, tmp_path, monkeypatch, engine, capsys):
        """Native index errors produce a diagnostic and leave a recoverable cache."""
        blocked = tmp_path / "not-a-directory"
        blocked.write_text("occupied")
        original = engine.Index

        def blocked_index(schema, **kwargs):
            return original(schema, path=str(blocked), reuse=False)

        with dc.config_context(docs_cache_dir=tmp_path / "cache"):
            with monkeypatch.context() as patcher:
                patcher.setattr(engine, "Index", blocked_index)
                assert main(["doc-search", "bandpass"]) == 1
            captured = capsys.readouterr()
            assert "Documentation search failed" in captured.err
            assert "not-a-directory" in captured.err
            assert not captured.out and "Traceback" not in captured.err
            assert doc_search.search_documents("bandpass")

    def test_keyword_spelling(self, tmp_path, monkeypatch):
        """Results retain authored keyword case while filtering ignores it."""
        source = tmp_path / "docs"
        source.mkdir()
        (source / "example.qmd").write_text(
            "---\nkeywords: [MiXeD, Optical Distance]\n---\nExample topic."
        )
        monkeypatch.setattr(doc_corpus, "DOC_PATH", source)
        with dc.config_context(docs_cache_dir=tmp_path / "cache"):
            results = doc_search.search_documents(tag="mixed")
            assert len(results) == 1
            assert results[0]["id"] == "example"
            assert results[0]["keywords"] == ["MiXeD", "Optical Distance"]

    def test_corpus_rebuild(self, search_cache):
        """Replacing the Markdown corpus also invalidates its search index."""
        expected = doc_search.search_documents("bandpass")
        with documentation_cache(rebuild=True):
            assert not (search_cache / "search").exists()
        assert doc_search.search_documents("bandpass") == expected

    @pytest.mark.concurrency
    def test_concurrent(self, tmp_path):
        """Concurrent first searches see one complete corpus and index."""
        code = (
            "import dascore as dc; from dascore.cli import main; "
            f"dc.set_config(docs_cache_dir={str(tmp_path)!r}); "
            "raise SystemExit(main(['doc-search', 'bandpass']))"
        )
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        for process in processes:
            stdout, stderr = process.communicate(timeout=90)
            assert process.returncode == 0, stderr
            assert "pass_filter" in stdout


class TestOptionalSearch:
    """Basic documentation and validation do not depend on the native engine."""

    def test_missing_engine(self, hide_module, capsys, search_cache):
        """A missing optional engine is reported without breaking doc lookup."""
        hide_module("tantivy")
        assert main(["doc-search", "bandpass"]) == 1
        captured = capsys.readouterr()
        assert "tantivy" in captured.err and "install" in captured.err.lower()
        assert not captured.out and "Traceback" not in captured.err
        assert main(["doc", "Patch.select"]) == 0
        assert "select(" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "query, tag, limit",
        [("", None, 5), (" ", " ", 5), ("x", None, 0), ("x", None, 101)],
    )
    def test_invalid_request(self, query, tag, limit):
        """Empty requests and invalid limits fail before touching the cache."""
        with pytest.raises(DocumentationError):
            doc_search.search_documents(query, tag=tag, limit=limit)

    def test_cli_usage(self, capsys):
        """A bare search and out-of-range limits return useful errors."""
        assert main(["doc-search"]) == 1
        assert "Provide a search query or --tag" in capsys.readouterr().err
        assert main(["doc-search", "patch", "--limit", "0"]) == 1
        assert "limit" in capsys.readouterr().err


class TestKeywordMetadata:
    """Validate authored keyword metadata before building an optional index."""

    @pytest.mark.parametrize("value", ["[2024]", "false", "null", "{filtering: true}"])
    def test_invalid(self, tmp_path, monkeypatch, value):
        """Malformed keyword types fail with a document-specific diagnostic."""
        source = tmp_path / "docs"
        source.mkdir()
        (source / "example.qmd").write_text(f"---\nkeywords: {value}\n---\nExample.")
        monkeypatch.setattr(doc_corpus, "DOC_PATH", source)
        with pytest.raises(
            DocumentationError, match="Keywords for 'example' must be strings"
        ):
            doc_corpus.write_corpus(tmp_path / "output")

    @pytest.mark.parametrize(
        "value, expected",
        [
            ('" Filtering "', ["Filtering"]),
            ('[" filtering ", "", "Median"]', ["filtering", "Median"]),
        ],
    )
    def test_normalized(self, tmp_path, monkeypatch, value, expected):
        """String and list forms share normalization without losing authored case."""
        source = tmp_path / "docs"
        source.mkdir()
        (source / "example.qmd").write_text(f"---\nkeywords: {value}\n---\nExample.")
        monkeypatch.setattr(doc_corpus, "DOC_PATH", source)
        manifest = doc_corpus.write_corpus(tmp_path / "output")
        record = next(x for x in manifest["documents"] if x["id"] == "example")
        assert record["keywords"] == expected
