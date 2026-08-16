"""Tests for dascore's downloader."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import pandas as pd
import pytest

from dascore.config import config_context
from dascore.utils.downloader import (
    REGISTRY_PATH,
    _fetch_cached,
    fetch,
    fetcher,
    get_fetcher,
    get_registry_df,
)


@pytest.fixture()
def registry_df():
    """Load the registry df."""
    df = get_registry_df()
    return df


class TestRegistryDF:
    """Tests for getting the data registry."""

    def test_dataframe(self, registry_df):
        """Ensure a non-empty df was returned."""
        assert len(registry_df)
        assert isinstance(registry_df, pd.DataFrame)

    def test_contains_all_registry_entries(self, registry_df):
        """The dataframe should include every non-comment registry line."""
        expected = [
            line.split(maxsplit=1)[0]
            for line in REGISTRY_PATH.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        ]
        assert registry_df["name"].tolist() == expected


class TestRegistryURLs:
    """The registry urls must be fetchable from a browser (eg Pyodide)."""

    # github.com/<owner>/<repo>/raw/... answers with a 302 whose
    # Access-Control-Allow-Origin header is empty. Every hop of a redirect
    # chain must pass CORS, so browsers abort before reaching the raw host.
    redirect_pattern = re.compile(r"https?://(www\.)?github\.com/[^/]+/[^/]+/raw/")

    def test_no_redirecting_github_urls(self, registry_df):
        """Registry urls must not use the redirecting github.com/../raw/ form."""
        bad = [url for url in registry_df["url"] if self.redirect_pattern.match(url)]
        assert not bad, (
            "These registry urls redirect and fail CORS in the browser; use "
            f"raw.githubusercontent.com instead: {bad}"
        )

    def test_urls_are_absolute_https(self, registry_df):
        """Every registry url should be an absolute https url."""
        split = [(url, urlsplit(url)) for url in registry_df["url"]]
        # netloc guards against values like "https:///path", which have a
        # valid scheme but no host and so are not absolute urls.
        bad = [u for u, parts in split if parts.scheme != "https" or not parts.netloc]
        assert not bad, f"Registry urls must be absolute https urls: {bad}"


class TestFetch:
    """Tests for fetching filepaths of test files."""

    def test_multiple_fetch(self, registry_df):
        """Ensure multiple fetch calls return same path."""
        path = registry_df["name"].iloc[0]
        assert fetch(path) == fetch(path)

    def test_existing_file(self, registry_df):
        """Ensure an existing file just returns."""
        path = fetch(registry_df["name"].iloc[0])
        assert fetch(path) == path

    def test_fetcher_path_comes_from_config(self, tmp_path, monkeypatch):
        """Downloader fetchers should honor the configured cache directory."""
        # DFS_DATA_DIR (set in CI) overrides the path at the pooch level, so
        # clear it to observe the config-supplied path.
        monkeypatch.delenv("DFS_DATA_DIR", raising=False)
        cache_dir = tmp_path / "downloads"
        with config_context(downloader_cache_dir=cache_dir):
            active_fetcher = get_fetcher()
            assert fetcher.path == active_fetcher.path
            assert active_fetcher.path.parent == cache_dir

    def test_fetch_cached_fetches_by_name_and_cache_dir(self, monkeypatch, tmp_path):
        """The cached fetch wrapper should call pooch with the requested name."""

        class _Fetcher:
            def fetch(self, name):
                assert name == "example.dat"
                return tmp_path / name

        monkeypatch.setattr(
            "dascore.utils.downloader._get_fetcher",
            lambda _cache_dir: _Fetcher(),
        )

        out = _fetch_cached(name="example.dat", cache_dir=str(tmp_path))

        assert out == tmp_path / "example.dat"
