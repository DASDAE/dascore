"""
Edge-case and error-path tests for the index package.

Complements the contract suite: exercises failure branches, kind
mismatches, rollbacks, and the directory-format walk so the package has
full line coverage.
"""

from __future__ import annotations

import re
import sqlite3

import numpy as np
import pandas as pd
import pytest
from test_index_contract import make_summaries

import dascore as dc
from dascore.core.summary import PatchSummary
from dascore.exceptions import UnitError
from dascore.io.index import Query, get_backend, summaries_to_records
from dascore.io.index.backend import _ns_to_time, adapt_params, resolve_query
from dascore.io.index.indexer import DBDirectoryIndexer
from dascore.io.index.ingest import (
    SourceRecord,
    _coord_record,
    typed_value,
)
from dascore.io.index.ingest import (
    summaries_to_records as s2r,
)
from dascore.io.index.query import InvalidSpoolQueryError, glob_match
from dascore.units import get_quantity


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    """One SQLite backend with the contract summaries plus extras."""
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
    path = tmp_path_factory.mktemp("edge") / "index.sqlite3"
    back = get_backend(path)
    back.write_sources(summaries_to_records([*make_summaries(), extra]))
    yield back
    back.close()


class TestAdaptAndBackendBasics:
    """Small helpers and backend plumbing."""

    def test_adapt_params_nan_becomes_none(self):
        """NaN floats bind as NULL."""
        assert adapt_params([float("nan"), 1])[0] is None

    def test_bulk_insert_empty_rows_noop(self, tmp_path):
        """Empty bulk inserts are no-ops."""
        back = get_backend(tmp_path / "insert.sqlite3")
        back._bulk_insert("attr_meta", ("attr_name",), [])
        back._executemany(
            "INSERT INTO attr_meta VALUES (?, ?, ?, ?)",
            [("a", "num", "a__num", None)],
        )
        assert len(back._attr_meta()) == 1
        back.close()

    def test_write_failure_rolls_back(self, tmp_path):
        """A failing write leaves the index unchanged."""
        back = get_backend(tmp_path / "rollback.sqlite3")
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

    def test_delete_failure_rolls_back(self, tmp_path):
        """A failing delete leaves the index unchanged."""
        back = get_backend(tmp_path / "delete.sqlite3")
        back.write_sources(summaries_to_records(make_summaries()))
        before = len(back.query())

        def boom(paths, base_uri=""):
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
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            resolve_query(backend, _attrs={"nope": 1})

    def test_unknown_coord_in_explicit_namespace(self, backend):
        """Unknown key in _coords raises."""
        with pytest.raises(InvalidSpoolQueryError, match="not a coordinate"):
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

    def test_slice_range_form(self, backend):
        """Slices resolve to the same range tuples patch selects accept."""
        lo = np.datetime64("2024-06-01T00:00:00", "ns")
        query = resolve_query(backend, time=slice(lo, None))
        assert query.coords["time"] == (lo, None)
        df = backend.query(query)
        assert list(df["tag"]) == ["extra"]


class TestUnitHeterogeneity:
    """Mixed unit populations across sources."""

    @staticmethod
    def _summary(path, attrs=None, coord_units=None):
        """One summary with a 100-200 distance coord."""
        coord = {
            "dtype": "float64",
            "min": 100.0,
            "max": 200.0,
            "dims": ("distance",),
            "len": 10,
        }
        if coord_units is not None:
            coord["units"] = coord_units
        return PatchSummary(
            attrs=attrs or {"tag": "units"},
            coords={"distance": coord},
            dims=("distance",),
            shape=(10,),
            dtype="float32",
            source_path=path,
            source_format="DASDAE",
            source_version="1",
        )

    def test_null_unit_coord_defs_stay_candidates(self, tmp_path):
        """Quantity queries must not drop unitless coord defs (candidacy)."""
        back = get_backend(tmp_path / "units.sqlite3")
        back.write_sources(
            summaries_to_records(
                [
                    self._summary("with_units.h5", coord_units="m"),
                    self._summary("no_units.h5"),
                ]
            )
        )
        meters = get_quantity("m")
        df = back.query(Query(coords={"distance": (150 * meters, 300 * meters)}))
        assert set(df["path"]) == {"with_units.h5", "no_units.h5"}
        back.close()

    def test_all_unitless_quantity_query_keeps_candidates(self, tmp_path):
        """Unitless defs cannot be proven incompatible; they stay candidates."""
        back = get_backend(tmp_path / "unitless.sqlite3")
        back.write_sources(summaries_to_records([self._summary("no_units.h5")]))
        meters = get_quantity("m")
        df = back.query(Query(coords={"distance": (150 * meters, 300 * meters)}))
        assert list(df["path"]) == ["no_units.h5"]
        back.close()

    def test_incompatible_units_only_raises(self, tmp_path):
        """When every def carries units and none are compatible, raise."""
        back = get_backend(tmp_path / "incompat.sqlite3")
        back.write_sources(
            summaries_to_records([self._summary("s.h5", coord_units="s")])
        )
        meters = get_quantity("m")
        with pytest.raises(UnitError, match="no units compatible"):
            back.query(Query(coords={"distance": (150 * meters, 300 * meters)}))
        back.close()

    def test_incompatible_attr_units_warn_not_fail(self, tmp_path):
        """One rogue attr dimension must not abort the whole index update."""
        summaries = [
            self._summary("a.h5", attrs={"resolution": 1.0 * get_quantity("m")}),
            self._summary("b.h5", attrs={"resolution": 1.0 * get_quantity("s")}),
        ]
        back = get_backend(tmp_path / "attr_units.sqlite3")
        with pytest.warns(UserWarning, match="incompatible"):
            back.write_sources(summaries_to_records(summaries))
        # both patches indexed; only the incompatible value is skipped
        df = back.query()
        assert set(df["path"]) == {"a.h5", "b.h5"}
        got = back.query(Query(attrs={"resolution": (0.5, 2.0)}))
        assert list(got["path"]) == ["a.h5"]
        back.close()


class TestExactNsFetch:
    """Nullable ns-integer columns must never round through float64."""

    # an epoch-ns value float64 rounds to ...768: exactness is observable
    NS = 1_752_244_251_123_456_789

    def _summary(self, path, coords, dims):
        return PatchSummary(
            attrs={"tag": "ns"},
            coords=coords,
            dims=dims,
            shape=tuple(10 for _ in dims),
            dtype="float32",
            source_path=path,
            source_format="DASDAE",
            source_version="1",
        )

    def test_time_envelopes_exact_when_column_nullable(self, tmp_path):
        """NULLs from other kinds/patches must not degrade ns columns."""
        t0 = np.datetime64(self.NS, "ns")
        with_time = self._summary(
            "abs.h5",
            {
                "event_time": {
                    "dtype": "datetime64",
                    "min": t0,
                    "max": t0 + np.timedelta64(60, "s"),
                    "dims": ("event_time",),
                    "len": 10,
                }
            },
            ("event_time",),
        )
        # a numeric coord puts NULL min_ns rows in the same link fetch,
        # and no time coord leaves patches.time_min NULL for this patch.
        numeric_only = self._summary(
            "num.h5",
            {
                "distance": {
                    "dtype": "float64",
                    "min": 0.0,
                    "max": 10.0,
                    "dims": ("distance",),
                    "len": 10,
                }
            },
            ("distance",),
        )
        back = get_backend(tmp_path / "exact.sqlite3")
        back.write_sources(summaries_to_records([with_time, numeric_only]))
        df = back.query().set_index("path")
        got = df.loc["abs.h5", "event_time_min"]
        assert pd.Timestamp(got).value == self.NS
        back.close()

    def test_mtime_ns_exact_with_null_row(self, tmp_path):
        """A single NULL mtime row must not corrupt the others (rescans)."""
        back = get_backend(tmp_path / "mtime.sqlite3")
        records = [
            SourceRecord(
                source_path="a.h5",
                source_format="",
                format_version="",
                mtime_ns=self.NS,
                size_bytes=1,
            ),
            SourceRecord(source_path="b.h5", source_format="", format_version=""),
        ]
        back.write_sources(records)
        sources = back.get_sources().set_index("source_path")
        assert int(sources.loc["a.h5", "mtime_ns"]) == self.NS
        back.close()

    def test_float_ns_column_rejected(self):
        """The conversion helper refuses already-corrupted float input."""
        series = pd.Series([1.5e18, np.nan], name="min_ns")
        with pytest.raises(TypeError, match="already corrupted"):
            _ns_to_time(series, "datetime")


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

    def test_empty_index_file_updates_on_first_query(self, tmp_path, random_patch):
        """A pre-created empty SQLite path is still a new index."""
        random_patch.io.write(tmp_path / "one.hdf5", "dasdae")
        index_path = tmp_path / "empty.sqlite3"
        index_path.touch()
        indexer = DBDirectoryIndexer(tmp_path, index_path=index_path)
        assert len(indexer()) == 1

    def test_directory_format_unit(self, tmp_path):
        """Directory-format sources (xml binary) group as one scan unit."""
        import sys

        sys.path.insert(0, "tests/test_io/test_xml_binary")
        from test_xml_binary import metadata

        sub = tmp_path / "unit"
        sub.mkdir()
        (sub / "metadata.xml").write_text(metadata)
        # hidden files and subdirectories inside a unit are ignored
        (sub / ".hidden_state").write_text("x")
        (sub / "logs").mkdir()
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
        indexer = DBDirectoryIndexer(tmp_path)
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


class TestCoordDeduplication:
    """Coord summaries are stored once per unique definition."""

    def test_shared_coord_stored_once(self, tmp_path):
        """Identical distance coords across patches share one def row."""
        back = get_backend(tmp_path / "dedup.sqlite3")
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
        back = get_backend(tmp_path / "reuse.sqlite3")
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
        back = get_backend(tmp_path / "fp.sqlite3")
        back.write_sources(summaries_to_records([PatchSummary(**structured)]))
        defs = back._fetch_df("SELECT def_key, fingerprint FROM coord_defs")
        assert defs["fingerprint"].notna().all()
        assert defs["def_key"].str.startswith("fp:").all()
        back.close()

    def test_irregular_coord_hashes_values(self):
        """A non-range coordinate carries the hash of its complete array."""
        patch = dc.get_example_patch()
        old = patch.get_coord("distance")
        values = np.arange(len(old), dtype=float)
        values[2:] += 0.5
        patch = patch.update_coords(distance=values)
        summary = PatchSummary.from_patch(patch)
        record = _coord_record("distance", summary.coords["distance"])
        assert record.coord_hash == patch.get_coord("distance").fingerprint()
        assert record.def_key.startswith("fp:")

    def test_orphan_defs_tolerated(self, tmp_path):
        """Deleting sources leaves defs behind without breaking queries."""
        back = get_backend(tmp_path / "orphan.sqlite3")
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
        back = get_backend(tmp_path / "bare.sqlite3")
        back.write_sources(summaries_to_records([summary]))
        df = back.query()
        assert len(df) == 1
        assert not [c for c in df.columns if c.endswith("_def_key")]
        back.close()


class TestCompositeSourceIdentity:
    """Sources are identified by (base_uri, source_path)."""

    def test_same_path_different_base_coexist(self, tmp_path):
        """Identical relative paths under different bases don't collide."""
        base = make_summaries()[0].dump_structured()
        one = PatchSummary(**base)
        records_a = summaries_to_records([one], base_uri="s3://bucket-a")
        # base_uri strip only applies when paths share the base; set directly
        records_a = [
            type(r)(**{**r.__dict__, "base_uri": "s3://bucket-a"}) for r in records_a
        ]
        records_b = [
            type(r)(**{**r.__dict__, "base_uri": "s3://bucket-b"}) for r in records_a
        ]
        back = get_backend(tmp_path / "multi.sqlite3")
        back.write_sources(records_a)
        back.write_sources(records_b)
        df = back.query()
        assert len(df) == 2
        prefixes = {p.split("/das/")[0] for p in df["path"]}
        assert prefixes == {"s3://bucket-a", "s3://bucket-b"}
        # deletion is base-scoped
        back.delete_sources([records_a[0].source_path], base_uri="s3://bucket-a")
        df = back.query()
        assert len(df) == 1
        assert df["path"].iloc[0].startswith("s3://bucket-b")
        back.close()

    def test_replacement_is_base_scoped(self, tmp_path):
        """Rewriting a source under one base leaves the other base alone."""
        base = make_summaries()[0].dump_structured()
        one = PatchSummary(**base)
        rec = summaries_to_records([one])[0]
        rec_a = type(rec)(**{**rec.__dict__, "base_uri": "s3://a"})
        rec_b = type(rec)(**{**rec.__dict__, "base_uri": "s3://b"})
        back = get_backend(tmp_path / "scoped.sqlite3")
        back.write_sources([rec_a, rec_b])
        assert len(back.query()) == 2
        back.write_sources([rec_a])  # replace only the s3://a copy
        assert len(back.query()) == 2
        back.close()


class TestResourceCleanup:
    """Backends must not leak SQLite connections (review P2)."""

    def test_gc_closes_connection(self):
        """Garbage collection closes the backend connection silently."""
        import gc
        import warnings as warnings_mod

        import dascore as dc

        spool = dc.spool([dc.get_example_patch()])
        spool.get_contents()  # realize the backend
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always", ResourceWarning)
            del spool
            gc.collect()
        resource = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert not resource

    def test_explicit_close_idempotent(self):
        """Explicit close works and GC afterwards stays quiet."""
        import dascore as dc
        from dascore.io.index.catalog import PatchCatalog

        catalog = PatchCatalog.from_patches([dc.get_example_patch()])
        catalog.to_df()
        catalog.close()

    def test_finalization_from_worker_thread(self):
        """Dropping the last backend reference off-thread does not raise."""
        import gc
        import sys
        import threading

        from dascore.io.index.lite import SQLiteBackend

        holder = [SQLiteBackend(":memory:")]
        unraisable = []
        original = sys.unraisablehook

        def _drop():
            holder.clear()
            gc.collect()

        sys.unraisablehook = unraisable.append
        try:
            worker = threading.Thread(target=_drop)
            worker.start()
            worker.join()
            gc.collect()
        finally:
            sys.unraisablehook = original
        errors = [u for u in unraisable if u.exc_type is sqlite3.ProgrammingError]
        assert not errors


class TestLegacyIndexMap:
    """Index-map entries pointing at retired .h5 indexes are bypassed."""

    def test_legacy_entry_ignored(self, tmp_path):
        """A mapped legacy HDF5 index does not break index creation."""
        import h5py

        import dascore as dc
        from dascore.io.index.indexer import (
            _update_index_map,
        )

        # data directory with one file, plus a fake legacy index mapping.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dc.write(dc.get_example_patch(), data_dir / "a.h5", "dasdae")
        legacy = tmp_path / "legacy_index.h5"
        with h5py.File(legacy, "w") as fh:
            fh.create_dataset("x", data=[1, 2, 3])
        map_path = tmp_path / "cache_paths.json"
        with dc.set_config(directory_index_map_path=map_path):
            _update_index_map({str(data_dir): str(legacy)}, cache_path=str(map_path))
            spool = dc.spool(data_dir).update()
            assert len(spool) == 1


class TestReservedAttrNames:
    """Attrs colliding with structural columns are skipped with a warning."""

    @pytest.mark.parametrize(
        "name,value",
        [("path", "user-path"), ("file_format", "attr-format"), ("source_id", 42)],
    )
    def test_reserved_attr_warns_and_skips(self, name, value):
        """Reserved names warn at ingest and never corrupt the relation."""
        import dascore as dc

        patch = dc.get_example_patch().update_attrs(**{name: value})
        with pytest.warns(UserWarning, match="reserved attr"):
            df = dc.spool([patch]).get_contents()
        assert not df.columns.duplicated().any()
        # structural values win; the attr stays on the patch itself.
        assert patch.attrs[name] == value

    def test_non_reserved_attr_round_trips(self):
        """Ordinary arbitrary attrs still index and select normally."""
        import dascore as dc

        patch = dc.get_example_patch().update_attrs(experiment="exp42")
        spool = dc.spool([patch])
        assert spool.get_contents()["experiment"].iloc[0] == "exp42"
        assert len(spool.select(experiment="exp42")) == 1

    def test_cross_patch_envelope_attr_reserved(self):
        """An envelope-shaped attr is reserved even against another patch's coord."""
        import dascore as dc

        base = dc.get_example_patch()
        # construction-time attrs bypass the update_attrs coord-name guard.
        p1 = dc.Patch(
            data=base.data,
            coords=base.coords,
            dims=base.dims,
            attrs=dict(base.attrs) | {"event_time_min": 123.0},
        )
        p2 = dc.get_example_patch().rename_coords(time="event_time")
        with pytest.warns(UserWarning, match="reserved attr name 'event_time_min'"):
            spool = dc.spool([p1, p2])
            df = spool.get_contents()
        backend = spool._get_catalog().backend
        assert "event_time_min" not in backend.attr_names()
        # the column is the coordinate envelope, not the stray attr value.
        assert "event_time_min" in df.columns
        assert 123.0 not in set(df["event_time_min"].dropna())

    def test_data_units_attr_still_indexed(self):
        """A real ``*_units`` attr is not an envelope column; stays queryable."""
        import dascore as dc

        patch = dc.get_example_patch().update_attrs(data_units="strain")
        spool = dc.spool([patch])
        assert "data_units" in spool._get_catalog().backend.attr_names()
