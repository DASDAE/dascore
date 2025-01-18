"""
DuckDB utils.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager

import duckdb
import pandas as pd

import dascore as dc
from dascore.utils.pd import _patch_summary_to_dataframes


def make_schema_str(schema_list):
    """Make the string of schema."""
    commas = (" ".join(x) for x in schema_list)
    return f"({', '.join(commas)})"


class DuckIndexer:
    """
    A class to encapsulate DuckDB interactions for spool indexing.
    """

    # If the tables have been tried to be created.
    _tried_create_tables = False

    # The schema for most data values; optimally flexible to avoid making
    # multiple tables per dtype.
    # Note: as of duckdb v1.1.3 intervals with ns precision are not supported.
    _flexible_vals = "UNION(str VARCHAR, int LONG, float DOUBLE, dt TIMESTAMP_NS)"

    # Primary keys for patch table, secondary for others.
    _patch_keys = ("patch_key", "spool_key")

    # The names of the source tables and their keys/schema.
    _schema = {
        "patch_source": (
            ("patch_key", "INTEGER"),
            ("spool_key", "INTEGER"),
            ("data_units", "VARCHAR"),
            ("ndims", "INTEGER"),
            ("data_shape", "INTEGER[]"),
            ("data_dtype", "VARCHAR"),
            ("dims", "VARCHAR[]"),
            ("coords", "VARCHAR[]"),
            ("format_name", "VARCHAR"),
            ("format_version", "VARCHAR"),
            ("path", "VARCHAR"),
            ("acquisition_id", "VARCHAR"),
            ("tag", "VARCHAR"),
        ),
        "coord_source": (
            ("patch_key", "INTEGER"),
            ("spool_key", "INTEGER"),
            ("name", "VARCHAR"),
            ("shape", "INTEGER[]"),
            ("dtype", "VARCHAR"),
            ("ndims", "INTEGER"),
            ("units", "VARCHAR"),
            ("dims", "VARCHAR[]"),
            ("start", _flexible_vals),
            ("stop", _flexible_vals),
            ("step", _flexible_vals),
        ),
        "attr_source": (
            ("patch_key", "INTEGER"),
            ("spool_key", "INTEGER"),
            ("name", "VARCHAR"),
            ("value", _flexible_vals),
        ),
    }

    # SQL type Schema for patch table.
    _patch_schema = ()

    # SQL type Schema for attribute table.
    _attr_schema = ()

    # SQL type Schema for coordinate table.
    _coord_schema = ()

    def __init__(self, connection="", **kwargs):
        self._connection = connection
        self._kwargs = kwargs

    def __repr__(self):
        out = f"DuckDB indexer ({self._connection}, {self._kwargs})) "
        return out

    @contextmanager
    def connection(self):
        """A context manager to create (and close) the connection."""
        with duckdb.connect(self._connection, **self._kwargs) as conn:
            # Ensure tables have been created.
            if not self._tried_create_tables:
                self._add_spool_tables(conn)
            yield conn
        conn.close()

    def _add_spool_tables(self, conn):
        """Add the tables (schema) to database if they aren't defined."""
        for table_name, schema in self._schema.items():
            schema_str = make_schema_str(schema)
            conn.sql(f"CREATE TABLE IF NOT EXISTS {table_name} {schema_str};")

    def get_table(self, name) -> pd.DataFrame:
        """Retrieve a table name as a dataframe."""
        with self.connection() as conn:
            out = conn.sql(f"SELECT * FROM {name}").df()
        return out

    def upsert_table(self, df, table):
        """Insert or update a dataframe."""
        with self.connection() as conn:
            cmd_str = (
                f"INSERT INTO {table} " "SELECT * FROM ? " "ON CONFLICT DO UPDATE SET "
            )
            conn.sql(cmd_str, [df])

    def insert_summaries(self, summaries: Sequence[dc.PatchSummary]):
        """Insert the Patch Summaries into the duck index."""
        patch, coord, attr = _patch_summary_to_dataframes(summaries)
        breakpoint()
