"""Persisted coordinate variants and patch counts."""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import InvalidIndexError, InvalidIndexVersionError
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import (
    SourceRecord,
    coord_dtype_is_stateable,
    patch_record,
    summaries_to_records,
)
from dascore.io.index.query import Query
from dascore.io.index.schema import INDEX_VERSION
from tests.test_io.test_index.test_heterogeneity_stress import make_random_summaries

# The coordinate discovery this replaces: a scan of every whole-coordinate link.
_LINK_VARIANTS = (
    "SELECT json_array(pc.coord_name, pc.dtype, cd.value_kind, cd.units, "
    "cd.is_relative) AS variant_key, count(*) AS patch_count "
    "FROM patch_coords pc JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
    "WHERE pc.run_index = 0 GROUP BY 1"
)
_LINK_COORD_META = (
    "SELECT DISTINCT pc.coord_name, cd.value_kind, cd.units, cd.is_relative "
    "FROM patch_coords pc JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
    "WHERE pc.run_index = 0"
)


def _rows(backend, sql) -> set[tuple]:
    """The rows a query returns, as a set."""
    return set(backend._con.execute(sql).fetchall())


def _assert_consistent(backend):
    """The maintained summaries equal a recount of the authoritative tables."""
    stored = _rows(backend, "SELECT variant_key, patch_count FROM coord_variants")
    assert stored == _rows(backend, _LINK_VARIANTS)
    (count,) = backend._con.execute("SELECT count(*) FROM patches").fetchone()
    assert backend.get_metadata()["patch_count"] == count
    expected = backend._fetch_df(_LINK_COORD_META)
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(
        backend.coord_meta().sort_values(columns).reset_index(drop=True),
        expected.sort_values(columns).reset_index(drop=True),
    )


def _set_version(path, version):
    """Stamp an index version on a closed index file."""
    con = sqlite3.connect(path)
    try:
        con.execute("UPDATE meta_data SET index_version = ?", (version,))
        con.commit()
    finally:
        con.close()


def _vm_steps(backend, func) -> int:
    """Count the SQLite VM instructions func runs on the backend's connection."""
    steps = 0

    def tick():
        nonlocal steps
        steps += 1
        return 0

    backend._con.set_progress_handler(tick, 1)
    try:
        func()
    finally:
        backend._con.set_progress_handler(None, 1)
    return steps


@pytest.fixture(scope="module")
def records():
    """Heterogeneous source records: many coordinate dtypes, kinds and units."""
    return summaries_to_records(make_random_summaries(60, seed=1))


@pytest.fixture()
def backend(tmp_path, records):
    """A file index holding the first 40 records."""
    back = get_backend(tmp_path / "index.sqlite3")
    back.write_sources(records[:40])
    yield back
    back.close()


class TestMaintainedCounts:
    """Every write keeps the variant and patch counts equal to a recount."""

    def test_insert(self, backend, records):
        """Appending sources adds their links."""
        _assert_consistent(backend)
        backend.write_sources(records[40:])
        _assert_consistent(backend)

    def test_replace(self, backend):
        """Rewriting a source swaps its old links for its new ones."""
        others = summaries_to_records(make_random_summaries(10, seed=2))
        # the same source paths with different contents
        assert {x.source_path for x in others} <= {
            x.source_path for x in backend.export_records()
        }
        backend.write_sources(others)
        _assert_consistent(backend)

    def test_delete(self, backend, records):
        """Removing sources removes their links, and emptied variants."""
        backend.delete_sources([x.source_path for x in records[:25]])
        _assert_consistent(backend)
        backend.delete_sources([x.source_path for x in records[25:40]])
        _assert_consistent(backend)
        assert not _rows(backend, "SELECT * FROM coord_variants")
        assert backend.get_metadata()["patch_count"] == 0

    def test_move(self, backend, records):
        """Renaming sources leaves every count unchanged."""
        before = _rows(backend, "SELECT * FROM coord_variants")
        old = records[0].source_path
        backend.move_sources({old: f"moved/{old}"})
        assert _rows(backend, "SELECT * FROM coord_variants") == before
        _assert_consistent(backend)

    def test_rollback(self, backend, records, monkeypatch):
        """A write which fails after replacing and adding rows changes no count."""
        before = (
            _rows(backend, "SELECT * FROM coord_variants"),
            backend.get_metadata()["patch_count"],
        )
        replacing = summaries_to_records(make_random_summaries(10, seed=2))
        insert = backend._bulk_insert

        def failing(table, columns, rows):
            insert(table, columns, rows)
            if table == "patch_coords":
                raise RuntimeError("interrupted")

        monkeypatch.setattr(backend, "_bulk_insert", failing)
        with pytest.raises(RuntimeError, match="interrupted"):
            backend.write_sources([*replacing, *records[40:]])
        after = (
            _rows(backend, "SELECT * FROM coord_variants"),
            backend.get_metadata()["patch_count"],
        )
        assert after == before
        _assert_consistent(backend)

    def test_reopen(self, backend, tmp_path):
        """The counts persist with the file."""
        before = _rows(backend, "SELECT * FROM coord_variants")
        backend.close()
        reopened = get_backend(tmp_path / "index.sqlite3")
        assert _rows(reopened, "SELECT * FROM coord_variants") == before
        _assert_consistent(reopened)
        reopened.close()

    def test_coord_names(self, backend):
        """Selectable names are the linked names whose dtype states an envelope."""
        rows = _rows(
            backend,
            "SELECT DISTINCT coord_name, dtype FROM patch_coords WHERE run_index = 0",
        )
        expected = {name for name, dtype in rows if coord_dtype_is_stateable(dtype)}
        assert backend.coord_names() == expected

    def test_root_count(self, backend):
        """A root's length is the maintained count."""
        (count,) = backend._con.execute("SELECT count(*) FROM patches").fetchone()
        assert backend.count() == count == 40

    def test_filtered_count_from_iterator(self, backend):
        """Queries passed as a one-shot iterable still filter the count."""
        query = Query(attrs={"station": "v1"})
        expected = backend.count([query])
        assert expected < backend.count()
        assert backend.count(iter([query])) == expected

    def test_writes_which_bypass_the_backend_are_counted(self, backend):
        """Rows written straight through SQL are counted like any other."""
        con = backend._con
        (last,) = con.execute("SELECT max(patch_id) FROM patches").fetchone()
        # a copy of the last patch and its links, then a whole source removed
        con.execute(
            "INSERT INTO patches (patch_id, source_id, source_patch_key, dims) "
            "SELECT ?, source_id, 'copy', dims FROM patches WHERE patch_id = ?",
            (last + 1, last),
        )
        con.execute(
            "INSERT INTO patch_coords SELECT ?, coord_name, run_index, "
            "coord_dims, coord_def_id, dtype FROM patch_coords WHERE patch_id = ?",
            (last + 1, last),
        )
        _assert_consistent(backend)
        assert backend.count() == 41
        con.execute("DELETE FROM sources WHERE source_id = 1")
        _assert_consistent(backend)

    def test_mixed_kind_sorts_within_each_kind(self):
        """Each kind of a mixed-kind coordinate sorts among itself, in any view."""

        def patch(values, tag):
            coords = {"depth": np.array(values)}
            data = np.zeros(len(values))
            return dc.Patch(
                data=data, coords=coords, dims=("depth",), attrs={"tag": tag}
            )

        values = (["c", "d"], [8.0, 9.0], ["a", "e"], [2.0, 3.0], ["b", "f"])
        spool = dc.spool(
            [patch(x, "num" if isinstance(x[0], float) else "text") for x in values]
        )

        def firsts(view):
            return [p.get_coord("depth").values[0] for p in view.sort("depth")]

        assert firsts(spool) == [2.0, 8.0, "a", "b", "c"]
        assert firsts(spool.select(tag="num")) == [2.0, 8.0]
        assert firsts(spool.select(tag="text")) == ["a", "b", "c"]


def _uniform_backend(count: int):
    """An in-memory index of count identical one-patch sources."""
    record = patch_record(dc.get_example_patch().summary)
    back = get_backend(":memory:")
    back.write_sources(
        [
            SourceRecord(
                source_path=f"file_{i:05d}.h5",
                source_format="DASDAE",
                format_version="1",
                patches=(record,),
            )
            for i in range(count)
        ]
    )
    return back


@pytest.fixture(scope="module")
def sized_backends():
    """The same archive at two sizes, eight times apart."""
    backends = {count: _uniform_backend(count) for count in (50, 400)}
    yield backends
    for back in backends.values():
        back.close()


_SMALL_READS = {
    "coord_meta": lambda b: b.coord_meta(),
    "coord_meta_of_one": lambda b: b.coord_meta({"time"}),
    "coord_names": lambda b: b.coord_names(),
    "root_count": lambda b: b.count(),
}


class TestWorkIsBounded:
    """Coordinate discovery and root counts do not grow with the archive."""

    @pytest.mark.parametrize("name", list(_SMALL_READS))
    def test_flat_across_sizes(self, sized_backends, name):
        """Eight times the patches costs well under twice the VM work."""
        read = _SMALL_READS[name]
        small, large = (
            _vm_steps(b, lambda b=b: read(b)) for b in sized_backends.values()
        )
        assert large < 2 * small

    @pytest.mark.parametrize(
        "read", [lambda b: b.count(), lambda b: b.query(patch_ids=[1])]
    )
    def test_no_whole_table_count(self, sized_backends, read):
        """
        A root count or a one-row projection counts no table.

        SQLite counts a table in one VM instruction, so VM steps cannot see
        this; the statements can.
        """
        backend = sized_backends[400]
        statements = []
        backend._con.set_trace_callback(statements.append)
        try:
            read(backend)
        finally:
            backend._con.set_trace_callback(None)
        assert statements
        assert not [x for x in statements if "count(" in x.lower()]


class TestNewerIndex:
    """An index from a newer DASCore is refused and left exactly as it was."""

    def test_backend_refuses_and_preserves(self, backend, tmp_path):
        """Opening it raises an error which does not ask for a rebuild."""
        path = tmp_path / "index.sqlite3"
        backend.close()
        _set_version(path, INDEX_VERSION + 1)
        before = path.read_bytes()
        with pytest.raises(InvalidIndexError, match="newer") as info:
            get_backend(path)
        assert not isinstance(info.value, InvalidIndexVersionError)
        assert path.read_bytes() == before

    def test_directory_spool_keeps_file(self, tmp_path):
        """The directory indexer does not delete it to rebuild."""
        dc.write(dc.get_example_patch(), tmp_path / "a.h5", "dasdae")
        spool = dc.spool(tmp_path).update(progress=None)
        index_path = spool.indexer.index_path
        spool.indexer.close()
        _set_version(index_path, INDEX_VERSION + 1)
        before = index_path.read_bytes()
        with pytest.raises(InvalidIndexError, match="newer"):
            dc.spool(tmp_path)
        assert index_path.read_bytes() == before
