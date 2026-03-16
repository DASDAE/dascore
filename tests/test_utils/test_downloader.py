"""Tests for dascore's downloader."""

from __future__ import annotations

import pandas as pd
import pytest

from dascore.constants import DATA_VERSION
from dascore.utils.downloader import (
    REGISTRY_PATH,
    fetch,
    fetcher,
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
