"""Tests for dascore's downloader."""
from __future__ import annotations

import pandas as pd
import pytest

from dascore.utils.downloader import get_registry_df


class TestRegistryDF:
    """Tests for getting the data registry."""

    @pytest.fixture()
    def registry_df(self):
        """Load the registry df."""
        df = get_registry_df()
        return df

    def test_dataframe(self, registry_df):
        """Ensure a non-empty df was returned."""
        assert len(registry_df)
        assert isinstance(registry_df, pd.DataFrame)
