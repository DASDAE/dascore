"""Tests for freezing the URLs the docs publish."""

from __future__ import annotations

import json

import pytest

_api_urls = pytest.importorskip("_api_urls")


@pytest.fixture
def cross_ref(tmp_path):
    """Write a small cross reference with an API key and a page key."""
    path = tmp_path / ".cross_ref.json"
    mapping = {
        "dascore.io.core.read": "/api/dascore/io/core/read.qmd",
        "dascore.read": "/api/dascore/io/core/read.qmd",
        "docs/tutorial/patch.qmd": "/tutorial/patch.qmd",
    }
    path.write_text(json.dumps(mapping))
    return path


class TestCurrentUrls:
    """Tests for reading the published URLs."""

    def test_api_keys_only(self, cross_ref):
        """The narrative pages are not part of the API surface."""
        urls = _api_urls.current_urls(cross_ref)

        assert set(urls) == {"dascore.io.core.read", "dascore.read"}

    def test_aliases_share_a_page(self, cross_ref):
        """An alias resolves to the page of the object it names."""
        urls = _api_urls.current_urls(cross_ref)

        assert urls["dascore.read"] == urls["dascore.io.core.read"]


class TestBaselineFile:
    """Tests for reading and writing the baseline."""

    def test_round_trip(self, tmp_path):
        """What is frozen is what is read back."""
        path = tmp_path / "api_urls.tsv"
        urls = {"b": "/api/b.qmd", "a": "/api/a.qmd"}

        _api_urls.write_baseline(urls, path)

        assert _api_urls.load_baseline(path) == urls

    def test_sorted_and_commented(self, tmp_path):
        """The file is sorted, so a diff shows only what changed."""
        path = tmp_path / "api_urls.tsv"
        _api_urls.write_baseline({"b": "/api/b.qmd", "a": "/api/a.qmd"}, path)

        lines = [x for x in path.read_text().splitlines() if not x.startswith("#")]
        assert lines == ["a\t/api/a.qmd", "b\t/api/b.qmd"]

    def test_missing_baseline(self, tmp_path):
        """Nothing frozen yet reads as an empty mapping, not an error."""
        assert _api_urls.load_baseline(tmp_path / "nope.tsv") == {}


class TestCompare:
    """Tests for comparing against the baseline."""

    def test_unchanged(self):
        """An unchanged mapping reports nothing."""
        urls = {"a": "/api/a.qmd"}

        difference = _api_urls.compare(urls, urls)

        assert not any(difference.values())

    def test_dropped_alias_publishes_the_same_page(self):
        """An alias can go while the page it named is still served."""
        baseline = {"dascore.read": "/api/read.qmd", "dascore.io.read": "/api/read.qmd"}
        current = {"dascore.io.read": "/api/read.qmd"}

        difference = _api_urls.compare(current, baseline)

        assert difference["removed"] == ["dascore.read"]
        assert difference["unpublished"] == []

    def test_added_removed_and_moved(self):
        """Each kind of change is reported apart from the others."""
        baseline = {"gone": "/api/gone.qmd", "moved": "/api/old.qmd"}
        current = {"new": "/api/new.qmd", "moved": "/api/new_home.qmd"}

        difference = _api_urls.compare(current, baseline)

        assert difference["added"] == ["new"]
        assert difference["removed"] == ["gone"]
        assert difference["moved"] == {"moved": ["/api/old.qmd", "/api/new_home.qmd"]}


class TestCheck:
    """Tests for the check sub command."""

    def test_no_baseline(self, tmp_path):
        """Without a baseline the check says so instead of passing quietly."""
        assert _api_urls.check(path=tmp_path / "nope.tsv") == 0
        assert _api_urls.check(strict=True, path=tmp_path / "nope.tsv") == 1

    def test_strict_fails_on_broken_url(self, tmp_path, monkeypatch):
        """A URL which vanished fails the strict check."""
        path = tmp_path / "api_urls.tsv"
        _api_urls.write_baseline({"gone": "/api/gone.qmd"}, path)
        monkeypatch.setattr(_api_urls, "current_urls", lambda: {})

        assert _api_urls.check(path=path) == 0
        assert _api_urls.check(strict=True, path=path) == 1

    def test_strict_allows_a_dropped_alias(self, tmp_path, monkeypatch):
        """Dropping an alias breaks a cross reference, not a saved link."""
        path = tmp_path / "api_urls.tsv"
        baseline = {"alias": "/api/read.qmd", "canonical": "/api/read.qmd"}
        _api_urls.write_baseline(baseline, path)
        monkeypatch.setattr(
            _api_urls, "current_urls", lambda: {"canonical": "/api/read.qmd"}
        )

        assert _api_urls.check(strict=True, path=path) == 0

    def test_strict_fails_on_a_move(self, tmp_path, monkeypatch):
        """A key which moved fails even though its old page is still served."""
        path = tmp_path / "api_urls.tsv"
        baseline = {"method": "/api/function.qmd", "function": "/api/function.qmd"}
        _api_urls.write_baseline(baseline, path)
        moved = {"method": "/api/method.qmd", "function": "/api/function.qmd"}
        monkeypatch.setattr(_api_urls, "current_urls", lambda: moved)

        assert _api_urls.check(path=path) == 0
        assert _api_urls.check(strict=True, path=path) == 1

    def test_strict_allows_additions(self, tmp_path, monkeypatch):
        """A new public object breaks no link, so it passes."""
        path = tmp_path / "api_urls.tsv"
        _api_urls.write_baseline({"kept": "/api/kept.qmd"}, path)
        urls = {"kept": "/api/kept.qmd", "new": "/api/new.qmd"}
        monkeypatch.setattr(_api_urls, "current_urls", lambda: urls)

        assert _api_urls.check(strict=True, path=path) == 0


class TestFrozenBaseline:
    """Tests for the baseline committed to the repository."""

    def test_baseline_is_frozen(self):
        """The repository holds a baseline to compare later builds against."""
        baseline = _api_urls.load_baseline()

        assert len(baseline) > 1000
        assert all(x.startswith("/api/") for x in baseline.values())
