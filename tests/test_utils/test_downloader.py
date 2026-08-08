"""Tests for dascore's downloader."""

from __future__ import annotations

import pandas as pd
import pytest

from dascore.utils.downloader import fetch, get_registry_df


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

    def test_urls_hosted_in_test_data_repo(self, registry_df):
        """
        All test-suite data files must be hosted in the DASDAE test_data repo;
        CI primes its cache from a single checkout of it (see
        .github/scripts/prime_test_data.py). Files hosted elsewhere would be
        re-downloaded per job and bypass the shared cache.
        """
        pattern = r"(?i)^https?://github\.com/dasdae/test_data/raw/master/"
        bad = registry_df[~registry_df["url"].str.match(pattern)]
        assert bad.empty, f"URLs not hosted in DASDAE/test_data: {bad['name'].tolist()}"


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
