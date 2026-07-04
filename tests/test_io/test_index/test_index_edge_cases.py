"""
Edge-case and error-path tests for the index package.

Complements the contract suite: exercises failure branches, kind
mismatches, rollbacks, and the directory-format walk so the package has
full line coverage.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest
from test_index_contract import make_summaries

import dascore as dc
from dascore.core.summary import PatchSummary
from dascore.io.index import Query, get_backend, summaries_to_records
from dascore.io.index.backend import adapt_params, resolve_query
from dascore.io.index.indexer import DBDirectoryIndexer
from dascore.io.index.ingest import (
    _coord_record,
    typed_value,
)
from dascore.io.index.ingest import (
    summaries_to_records as s2r,
)
from dascore.io.index.query import InvalidSpoolQueryError, glob_match
from dascore.units import get_quantity

BACKENDS = ("duckdb", "sqlite", "parquet")


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    """One duckdb backend with the contract summaries plus extras."""
    extra = PatchSummary(
        attrs={
            "tag": "extra",
            "trigger_time": np.datetime64("2024-06-01T00:00:00", "ns"),
            "window": np.timedelta64(10, "s"),
        },
        coords={
            "time": {
                "dtype": "datetime64",
                "min": np.datetime64("2024-06-01T00:00:00", "ns"),
                "max": np.datetime64("2024-06-01T00:01:00", "ns"),
                "dims": ("time",),
                "len": 100,
            },
        },
        dims=("time",),
        shape=(100,),
        dtype="float32",
        source_path="extras/trigger.h5",
        source_format="DASDAE",
        source_version="1",
    )
    path = tmp_path_factory.mktemp("edge") / "idx.duckdb"
    back = get_backend(path, kind="duckdb")
    back.write_sources(summaries_to_records([*make_summaries(), extra]))
    yield back
    back.close()


class TestAdaptAndBackendBasics:
    """Small helpers and backend plumbing."""

    def test_adapt_params_nan_becomes_none(self):
        """NaN floats bind as NULL."""
        assert adapt_params([float("nan"), 1])[0] is None

    def test_unknown_backend_kind_raises(self, tmp_path):
        """Asking for a nonexistent engine errors clearly."""
        with pytest.raises(ValueError, match="Unknown index backend"):
            get_backend(tmp_path / "x", kind="mongodb")

    @pytest.mark.parametrize("kind", BACKENDS)
    def test_bulk_insert_empty_rows_noop(self, tmp_path, kind):
        """Empty bulk inserts are no-ops on every backend."""
        back = get_backend(tmp_path / f"i_{kind}", kind=kind)
        back._bulk_insert("attr_meta", ("attr_name",), [])
        back._executemany(
            "INSERT INTO attr_meta VALUES (?, ?, ?, ?)",
            [("a", "num", "a__num", None)],
        )
        assert len(back._attr_meta()) == 1
        back.close()

    @pytest.mark.parametrize("kind", BACKENDS)
    def test_write_failure_rolls_back(self, tmp_path, kind):
        """A failing write leaves the index unchanged."""
        back = get_backend(tmp_path / f"r_{kind}", kind=kind)
        records = summaries_to_records(make_summaries())
        back.write_sources(records[:1])
        before = len(back.query())

        def boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        back._bulk_insert = boom
        with pytest.raises(RuntimeError, match="simulated"):
            back.write_sources(records[1:])
        del back.__dict__["_bulk_insert"]
        assert len(back.query()) == before
        back.close()

    @pytest.mark.parametrize("kind", BACKENDS)
    def test_delete_failure_rolls_back(self, tmp_path, kind):
        """A failing delete leaves the index unchanged."""
        back = get_backend(tmp_path / f"d_{kind}", kind=kind)
        back.write_sources(summaries_to_records(make_summaries()))
        before = len(back.query())

        def boom(paths):
            raise RuntimeError("simulated failure")

        back._delete_by_paths = boom
        with pytest.raises(RuntimeError, match="simulated"):
            back.delete_sources(["das/file_1.h5"])
        del back.__dict__["_delete_by_paths"]
        assert len(back.query()) == before
        back.close()

    def test_delete_no_paths_noop(self, backend):
        """Deleting an empty path list does nothing."""
        before = len(backend.query())
        backend.delete_sources([])
        assert len(backend.query()) == before

    def test_flatten_skips_absent_columns(self, backend):
        """attr_meta rows without a matching result column are skipped."""
        df = backend._fetch_df("SELECT patch_id FROM patches LIMIT 2")
        out = backend._flatten(df, backend._attr_meta())
        assert len(out) == 2

    def test_duration_attr_roundtrip(self, backend):
        """dur-kind attrs come back as timedeltas."""
        df = backend.query(Query(attrs={"window": np.timedelta64(10, "s")}))
        assert len(df) == 1
        assert pd.api.types.is_timedelta64_dtype(df["window"])


class TestResolveQueryErrors:
    """Explicit-namespace validation."""

    def test_unknown_attr_in_explicit_namespace(self, backend):
        """Unknown key in _attrs raises."""
        with pytest.raises(InvalidSpoolQueryError, match="Unknown attribute"):
            resolve_query(backend, _attrs={"nope": 1})

    def test_unknown_coord_in_explicit_namespace(self, backend):
        """Unknown key in _coords raises."""
        with pytest.raises(InvalidSpoolQueryError, match="Unknown coordinate"):
            resolve_query(backend, _coords={"nope": (1, 2)})


class TestQueryValueEdges:
    """Coercion, kind-mismatch, and malformed-value behavior."""

    def test_none_value_raises(self, backend):
        """None is not a valid predicate value."""
        with pytest.raises(InvalidSpoolQueryError, match="Cannot use"):
            backend.query(Query(attrs={"station": None}))

    def test_datetime_string_matches_time_attr(self, backend):
        """A datetime-like string queries a time-kind attr."""
        df = backend.query(Query(attrs={"trigger_time": "2024-06-01T00:00:00"}))
        assert list(df["tag"]) == ["extra"]

    def test_mixed_kind_range_raises(self, backend):
        """Range bounds of different kinds raise."""
        with pytest.raises(InvalidSpoolQueryError, match="mixed kinds"):
            backend.query(Query(attrs={"gauge_length": ("a", 5)}))

    def test_fully_open_range_raises(self, backend):
        """A range with no usable bounds raises."""
        with pytest.raises(InvalidSpoolQueryError, match="no usable bounds"):
            backend.query(Query(attrs={"gauge_length": (None, ...)}))

    def test_inverted_range_raises(self, backend):
        """Lo > hi raises."""
        with pytest.raises(InvalidSpoolQueryError, match="lo > hi"):
            backend.query(Query(attrs={"gauge_length": (5, 1)}))

    def test_unknown_attr_in_query_raises(self, backend):
        """A Query naming an unknown attr raises at SQL build."""
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            backend.query(Query(attrs={"nope": 1}))

    def test_regex_on_non_str_attr_empty(self, backend):
        """Regex against a numeric-only attr matches nothing."""
        df = backend.query(Query(attrs={"gauge_length": re.compile("x")}))
        assert df.empty

    def test_range_kind_mismatch_empty(self, backend):
        """A numeric range on a str-only attr matches nothing."""
        df = backend.query(Query(attrs={"station": (1, 2)}))
        assert df.empty

    def test_membership_mixed_kinds(self, backend):
        """Wrong-kind members are ignored; right-kind ones match."""
        df = backend.query(Query(attrs={"station": ["STA1", 5]}))
        assert list(df["station"]) == ["STA1"]

    def test_membership_all_wrong_kind_empty(self, backend):
        """All-wrong-kind membership matches nothing."""
        df = backend.query(Query(attrs={"station": [1, 2]}))
        assert df.empty

    def test_glob_on_non_str_attr_empty(self, backend):
        """Glob against a numeric-only attr matches nothing."""
        df = backend.query(Query(attrs={"gauge_length": "1*"}))
        assert df.empty

    def test_boolean_array_coord_requires_presence_only(self, backend):
        """Boolean masks are patch-local; index only checks coord presence."""
        mask = np.array([True, False, True])
        df = backend.query(Query(coords={"distance": mask}))
        assert len(df) == 4  # every patch with a distance coord

    def test_glob_match_helper(self):
        """Reference glob semantics."""
        assert glob_match("STA1", "STA*")
        assert not glob_match(5, "STA*")


class TestIngestEdges:
    """typed_value and record-building edge cases."""

    def test_plain_array_skipped(self):
        """Arrays are complex attrs; skipped."""
        assert typed_value(np.array([1, 2])) is None

    def test_array_quantity_skipped(self):
        """Array-valued quantities are skipped."""
        assert typed_value(np.array([1.0, 2.0]) * get_quantity("m")) is None

    def test_reserved_attr_name_warns(self):
        """An attr named patch_id is skipped with a warning."""
        summary = PatchSummary(
            attrs={"patch_id": 5, "tag": "x"},
            coords={
                "distance": {
                    "dtype": "float64",
                    "min": 0.0,
                    "max": 1.0,
                    "dims": ("distance",),
                    "len": 2,
                }
            },
            dims=("distance",),
            shape=(2,),
            dtype="float32",
            source_path="a.h5",
            source_format="DASDAE",
            source_version="1",
        )
        with pytest.warns(UserWarning, match="reserved attr name"):
            records = s2r([summary])
        assert "patch_id" not in records[0].patches[0].attrs

    def test_unsupported_coord_dtype_skipped(self):
        """A coord with no usable dtype produces no record."""

        class _Stub:
            dtype = ""
            dims = ("x",)
            len = 2
            units = None
            fingerprint = None
            min = 0
            max = 1
            step = None

        assert _coord_record("x", _Stub()) is None

    def test_multipatch_source_gets_positional_ids(self):
        """Multi-patch sources get positional source_patch_ids."""
        base = make_summaries()[0].dump_structured()
        one = PatchSummary(**base)
        two = PatchSummary(**{**base, "attrs": {"station": "STA9"}})
        records = s2r([one, two])
        assert len(records) == 1
        ids = [p.source_patch_id for p in records[0].patches]
        assert ids == ["0", "1"]


class TestIndexerEdges:
    """DBDirectoryIndexer edge behavior."""

    def test_auto_update_on_first_query(self, tmp_path, random_patch):
        """A brand-new index triggers one update on first query."""
        random_patch.io.write(tmp_path / "one.hdf5", "dasdae")
        indexer = DBDirectoryIndexer(tmp_path)
        assert len(indexer()) == 1  # no explicit update() call

    def test_directory_format_unit(self, tmp_path):
        """Directory-format sources (xml binary) group as one scan unit."""
        import sys

        sys.path.insert(0, "tests/test_io/test_xml_binary")
        from test_xml_binary import metadata

        sub = tmp_path / "unit"
        sub.mkdir()
        (sub / "metadata.xml").write_text(metadata)
        rand = np.random.default_rng(0).random((5000, 10)).astype("float32")
        for name in (
            "DAS_20240530T011500_000000Z.raw",
            "DAS_20240530T011530_000000Z.raw",
        ):
            with (sub / name).open("wb") as fi:
                rand.tofile(fi)
        indexer = DBDirectoryIndexer(tmp_path).update(progress=None)
        df = indexer()
        assert len(df) == 2
        # unchanged: second update rescans nothing
        before = indexer._backend.get_sources()["last_indexed_ns"].max()
        indexer.update(progress=None)
        after = indexer._backend.get_sources()["last_indexed_ns"].max()
        assert before == after
        indexer.close()


class TestDirSpoolPassthrough:
    """DirectorySpool accepts a prebuilt indexer."""

    def test_spool_from_indexer(self, tmp_path, random_patch):
        """Passing an indexer instance to DirectorySpool works."""
        from dascore.clients.dirspool import DirectorySpool

        random_patch.io.write(tmp_path / "one.hdf5", "dasdae")
        indexer = DBDirectoryIndexer(tmp_path, engine="duckdb")
        spool = DirectorySpool(indexer).update(progress=None)
        assert len(spool) == 1


class TestFinalCoverage:
    """Remaining edge branches."""

    def test_datetime_object_becomes_time(self):
        """A python datetime routes through the datetime fallback."""
        import datetime

        out = typed_value(datetime.datetime(2024, 1, 1))
        assert out is not None and out.kind == "time"

    def test_arbitrary_object_skipped(self):
        """Unclassifiable objects are skipped."""

        class _Odd:
            """Not datetime-convertible, not a scalar."""

        assert typed_value(_Odd()) is None

    def test_parquet_cleanup_failure_tolerated(self, tmp_path, monkeypatch):
        """A failed unlink of superseded parquet files is not an error."""
        import pathlib

        back = get_backend(tmp_path / "pq", kind="parquet")
        back.write_sources(summaries_to_records(make_summaries()[:1]))

        def bad_unlink(self, missing_ok=False):
            raise OSError("simulated busy file")

        monkeypatch.setattr(pathlib.Path, "unlink", bad_unlink)
        back.write_sources(summaries_to_records(make_summaries()[1:2]))
        monkeypatch.undo()
        assert len(back.query()) == 2
        back.close()


class TestCoordDeduplication:
    """Coord summaries are stored once per unique definition."""

    def test_shared_coord_stored_once(self, tmp_path):
        """Identical distance coords across patches share one def row."""
        back = get_backend(tmp_path / "dedup", kind="duckdb")
        back.write_sources(summaries_to_records(make_summaries()))
        links = back._fetch_df("SELECT * FROM patch_coords")
        defs = back._fetch_df("SELECT * FROM coord_defs")
        assert len(defs) < len(links)
        # das1 and das2 share an identical distance coord: one def, two links
        dist_links = links[links["coord_name"] == "distance"]
        das_defs = dist_links["coord_def_id"].value_counts()
        assert (das_defs >= 2).any()
        back.close()

    def test_defs_reused_across_writes(self, tmp_path):
        """A second write with known coords creates no new defs."""
        back = get_backend(tmp_path / "reuse", kind="duckdb")
        summaries = make_summaries()
        back.write_sources(summaries_to_records(summaries[:1]))
        n_defs = len(back._fetch_df("SELECT * FROM coord_defs"))
        # das2 shares the distance def with das1; only time is new
        back.write_sources(summaries_to_records(summaries[1:2]))
        n_defs_after = len(back._fetch_df("SELECT * FROM coord_defs"))
        assert n_defs_after == n_defs + 1
        back.close()

    def test_fingerprint_backed_defs(self, tmp_path):
        """Summaries from real patches carry fingerprints into defs."""
        summary = PatchSummary.from_patch(dc.get_example_patch())
        structured = summary.dump_structured()
        structured.update(
            {
                "source_path": "fp/one.h5",
                "source_format": "DASDAE",
                "source_version": "1",
            }
        )
        back = get_backend(tmp_path / "fp", kind="duckdb")
        back.write_sources(summaries_to_records([PatchSummary(**structured)]))
        defs = back._fetch_df("SELECT def_key, fingerprint FROM coord_defs")
        assert defs["fingerprint"].notna().all()
        assert defs["def_key"].str.startswith("fp:").all()
        back.close()

    def test_orphan_defs_tolerated(self, tmp_path):
        """Deleting sources leaves defs behind without breaking queries."""
        back = get_backend(tmp_path / "orphan", kind="duckdb")
        back.write_sources(summaries_to_records(make_summaries()))
        n_defs = len(back._fetch_df("SELECT * FROM coord_defs"))
        back.delete_sources(["das/file_1.h5", "das/file_2.h5"])
        assert len(back._fetch_df("SELECT * FROM coord_defs")) == n_defs
        assert len(back.query()) == 2
        back.close()


class TestPivotEdge:
    """Pivot with coord-less patches."""

    def test_no_coords_patch(self, tmp_path):
        """A patch with no coords pivots to nothing, without error."""
        summary = PatchSummary(
            attrs={"tag": "bare"},
            coords={},
            dims=(),
            shape=(),
            dtype="float32",
            source_path="bare.h5",
            source_format="DASDAE",
            source_version="1",
        )
        back = get_backend(tmp_path / "bare", kind="duckdb")
        back.write_sources(summaries_to_records([summary]))
        df = back.query()
        assert len(df) == 1
        assert not [c for c in df.columns if c.endswith("_def_key")]
        back.close()
