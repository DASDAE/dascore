"""Tests for dascore's downloader."""

from __future__ import annotations

import pandas as pd
import pytest

from dascore.config import set_config
from dascore.constants import DATA_VERSION
from dascore.utils.downloader import (
    LARGE_REGISTRY_FILES,
    REGISTRY_PATH,
    _fetch_cached,
    fetch,
    fetcher,
    get_fetcher,
    get_registry_df,
    get_test_data_cache_info,
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

    def test_exclude_large_filters_large_entries(self):
        """Large registry files should be excluded only when requested."""
        all_df = get_registry_df()
        filtered_df = get_registry_df(exclude_large=True)

        assert LARGE_REGISTRY_FILES <= set(all_df["name"])
        assert not (LARGE_REGISTRY_FILES & set(filtered_df["name"]))


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

    def test_fetcher_path_comes_from_config(self, tmp_path):
        """Downloader fetchers should honor the configured cache directory."""
        cache_dir = tmp_path / "downloads"
        with set_config(downloader_cache_dir=cache_dir):
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


class TestTestDataCacheInfo:
    """Tests for CI cache metadata derived from downloader state."""

    def test_cache_info_matches_downloader_configuration(self):
        """Ensure cache metadata stays aligned with downloader config."""
        info = get_test_data_cache_info()

        assert info.registry_path == REGISTRY_PATH
        assert info.cache_path == fetcher.path.parent
        assert info.data_version == DATA_VERSION
        assert len(info.registry_hash) == 64

    def test_cache_key_includes_expected_parts(self):
        """Ensure the generated cache key matches the CI convention."""
        info = get_test_data_cache_info()

        out = info.get_key(runner_os="Linux", cache_number=7)

        assert out == f"data-Linux-{DATA_VERSION}-{info.registry_hash}-7"

    def test_cache_info_respects_configured_cache_dir(self, tmp_path):
        """Cache info should reflect the configured downloader cache root."""
        cache_dir = tmp_path / "downloads"
        with set_config(downloader_cache_dir=cache_dir):
            info = get_test_data_cache_info()
        assert info.cache_path == cache_dir
