"""
Tests for duckdb.
"""

import pandas as pd
import pytest

from dascore.utils.duck import DuckIndexer


@pytest.fixture(scope="class")
def duck_indexer():
    """The default duck indexer fixture."""
    return DuckIndexer()


class TestDuckIndexer:
    """Basic tests for DuckIndexer."""

    def test_schema_created(self, duck_indexer):
        """Iterate the expected tables and ensure they exist."""
        for table_name, schema in duck_indexer._schema.items():
            cols = [x[0] for x in schema]
            df = duck_indexer.get_table(table_name)
            assert isinstance(df, pd.DataFrame)
            assert set(df.columns) == set(cols)

    def test_insert_summary(self, duck_indexer, random_patch):
        """Ensure we can insert a summary"""
        duck_indexer.insert_summaries(random_patch)
        breakpoint()
