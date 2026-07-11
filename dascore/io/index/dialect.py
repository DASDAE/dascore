"""
SQL dialect translation for index backends.

Everything engine-specific lives here: type names, identifier quoting,
glob matching, and table DDL generation. The rest of the package emits
logical types and parametrized SQL with `?` placeholders.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar


class BaseDialect:
    """Shared SQL generation for engines close to the standard."""

    # logical type -> engine type; concrete dialects define the values.
    type_map: ClassVar[Mapping[str, str]]
    strict_suffix: ClassVar[str]

    def quote(self, identifier: str) -> str:
        """Quote an identifier."""
        return '"' + identifier.replace('"', '""') + '"'

    def create_table(
        self, name: str, columns: Mapping[str, str], constraints: tuple[str, ...] = ()
    ) -> str:
        """Return DDL for one table from logical column types."""
        definitions = [
            f"{self.quote(col)} {self.type_map[typ]}" for col, typ in columns.items()
        ]
        definitions.extend(constraints)
        cols = ", ".join(definitions)
        quoted = self.quote(name)
        return f"CREATE TABLE IF NOT EXISTS {quoted} ({cols}){self.strict_suffix}"

    def add_column(self, table: str, column: str, logical_type: str) -> str:
        """Return DDL to add one nullable column."""
        return (
            f"ALTER TABLE {self.quote(table)} "
            f"ADD COLUMN {self.quote(column)} {self.type_map[logical_type]}"
        )

    def glob(self, column_sql: str) -> str:
        """Return a parametrized unix-glob match expression."""
        return f"{column_sql} GLOB ?"


class SQLiteDialect(BaseDialect):
    """Dialect for SQLite; STRICT tables enforce the type contract."""

    # SQLite STRICT tables accept INTEGER/REAL/TEXT (and INT for bool).
    type_map: ClassVar[Mapping[str, str]] = {
        "int64": "INTEGER",
        "float64": "REAL",
        "str": "TEXT",
        "bool": "INTEGER",
    }
    strict_suffix: ClassVar[str] = " STRICT"
