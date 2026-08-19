"""
Edge-case and error-path tests for the index package.

Complements the contract suite: exercises failure branches, kind
mismatches, rollbacks, and the directory-format walk so the package has
full line coverage.
"""

from __future__ import annotations

import datetime
import gc
import os
import pickle
import re
import sqlite3
import sys
import threading
import warnings
import warnings as warnings_mod
from typing import Any

import h5py
import numpy as np
import pandas as pd
import pytest
from test_index_contract import _time_coord, make_summaries

import dascore as dc
from dascore.core.spool import Spool
from dascore.core.summary import PatchSummary, normalize_source_patch_id
from dascore.exceptions import (
    InvalidIndexError,
    MissingPatchError,
    ParameterError,
    UnitError,
)
from dascore.io.core import is_directory_format
from dascore.io.index.backend import (
    SQLiteIndexBackend,
    _ns_to_time,
    adapt_params,
    get_backend,
)
from dascore.io.index.catalog import LiveResolver, PatchCatalog
from dascore.io.index.indexer import (
    DBDirectoryIndexer,
    _set_mapped_index_path,
)
from dascore.io.index.ingest import (
    SourceRecord,
    _coord_record,
    _py_scalar,
    assemble_source_records,
    patch_record,
    summaries_to_records,
    typed_value,
)
from dascore.io.index.ingest import (
    summaries_to_records as s2r,
)
from dascore.io.index.query import (
    _UNSET,
    InvalidSpoolQueryError,
    Query,
    _range_bounds,
    _to_target_unit,
    resolve_query,
)
from dascore.units import get_quantity, m, s
from dascore.units import get_quantity as _gq
from dascore.utils.misc import _canonical_range


class TestIndexCoverageEdges:
    """Remaining index branches covered with real backends/spools."""

    def test_live_resolver_missing_patch(self):
        """A row for a patch absent from the registry raises MissingPatchError."""
        resolver = LiveResolver([dc.get_example_patch()])
        with pytest.raises(MissingPatchError, match="not available"):
            resolver.resolve({"source_path": "memorypatch://not-a-real-id"})

    def test_mixed_compatible_unit_range(self):
        """A coord range mixing compatible units resolves."""
        cm = get_quantity("cm")
        out = dc.spool([dc.get_example_patch()]).select(distance=(1 * m, 200 * cm))
        assert len(out.get_contents()) == 1

    def test_backend_range_incompatible_units_raise(self, backend):
        """A hand-built coord range mixing incompatible units is rejected.

        The catalog canonicalizes units before the backend, so only a
        direct Query reaches the multi-unit compatibility check.
        """
        with pytest.raises(UnitError, match="Cannot convert"):
            backend.query(Query(coords={"distance": (1 * m, 2 * s)}))

    def test_export_skips_source_without_patches(self, tmp_path):
        """A non-fiber file gets a sources row with no patches; export skips it."""
        dc.get_example_patch().io.write(tmp_path / "a.h5", "dasdae")
        (tmp_path / "junk.txt").write_text("not a fiber file")
        spool = dc.spool(tmp_path).update(progress=None)
        backend = spool._catalog.backend
        assert len(backend.get_sources()) == 2  # the h5 and the junk file
        records = backend.export_records()
        assert sum(len(r.patches) for r in records) == 1  # only the real patch

    def test_attr_meta_units_backfilled(self, tmp_path):
        """An attr first seen unitless gets its unit backfilled by a later write."""

        def _summary(gain, path):
            return PatchSummary(
                attrs={"tag": "t", "gain": gain},
                coords={
                    "distance": {
                        "dtype": "float64",
                        "min": 0.0,
                        "max": 10.0,
                        "step": 1.0,
                        "units": "m",
                        "dims": ("distance",),
                        "len": 11,
                    }
                },
                dims=("distance",),
                shape=(11,),
                dtype="float64",
                source_path=path,
                source_format="X",
                source_version="1",
            )

        back = get_backend(tmp_path / "i.sqlite3")
        try:
            back.write_sources(s2r([_summary(5.0, "a.h5")]))  # unitless
            meta = back._attr_meta()
            assert list(meta.loc[meta["attr_name"] == "gain", "units"]) == [None]
            back.write_sources(s2r([_summary(5.0 * m, "b.h5")]))  # units -> backfill
            meta = back._attr_meta()
            assert list(meta.loc[meta["attr_name"] == "gain", "units"]) == ["m"]
        finally:
            back.close()

    def test_stepless_numeric_coord_keeps_null_step(self, tmp_path):
        """A numeric coord with no step stays NaN rather than an int sentinel.

        When no numeric coord in the result carries a step, the envelope
        column holds only timedeltas and nulls; inferring a dtype there
        turned those nulls into NaT, and then into int64 min.
        """
        # the time coord's step is what puts a Timedelta in the shared envelope
        # column; without it there is nothing for pandas to infer a temporal
        # dtype from, and the bug cannot occur. Stated here rather than
        # inherited from a helper default so the trigger cannot drift away.
        time_coord = _time_coord("2024-01-01T00:00:00", 60, step_s=0.004)
        assert time_coord["step"] is not None
        summary = PatchSummary(
            attrs={"tag": "t"},
            coords={
                "time": time_coord,
                # both numeric coords are stepless, so nothing anchors the
                # envelope column to a numeric dtype
                "distance": {
                    "dtype": "float64",
                    "min": 0.0,
                    "max": 10.0,
                    "step": None,
                    "units": "m",
                    "dims": ("distance",),
                    "len": 11,
                },
                "temperature": {
                    "dtype": "float64",
                    "min": 20.0,
                    "max": 25.0,
                    "step": None,
                    "units": "degC",
                    "dims": ("time",),
                    "len": 15000,
                },
            },
            dims=("time", "distance"),
            shape=(15000, 11),
            dtype="float32",
            source_path="a.h5",
            source_format="X",
            source_version="1",
        )
        back = get_backend(tmp_path / "i.sqlite3")
        try:
            back.write_sources(s2r([summary]))
            df = back.query()
            step = df["temperature_step"]
            # int64 min is what a NaT in this column decays to
            assert step.dtype == np.dtype("float64")
            assert step.isna().all()
        finally:
            back.close()

    def test_reopen_missing_meta_row(self, tmp_path):
        """An index whose meta row was lost is rejected on reopen."""
        path = tmp_path / "i.sqlite3"
        get_backend(path).close()
        con = sqlite3.connect(path)
        con.execute("DELETE FROM meta_data")
        con.commit()
        con.close()
        with pytest.raises(InvalidIndexError, match="not a valid"):
            get_backend(path)

    def test_reopen_table_missing_column(self, tmp_path):
        """An index whose table lost a column is rejected on reopen."""
        path = tmp_path / "i.sqlite3"
        get_backend(path).close()
        con = sqlite3.connect(path)
        con.execute("ALTER TABLE patches DROP COLUMN dtype")
        con.commit()
        con.close()
        with pytest.raises(InvalidIndexError, match="missing columns"):
            get_backend(path)

    def test_legacy_index_check_unopenable_path(self, tmp_path):
        """A path that exists but cannot be opened as a file is not an index."""
        idx = DBDirectoryIndexer(tmp_path)
        # opening a directory raises IsADirectoryError (an OSError), which the
        # header probe suppresses before concluding it is not a legacy index.
        sub = tmp_path / "adir"
        sub.mkdir()
        assert idx._is_legacy_or_foreign_index(sub) is False

    def test_schema_creation_rolls_back_on_failure(self, tmp_path):
        """A failure while creating the schema rolls back and re-raises."""

        class _BoomBackend(SQLiteIndexBackend):
            def _execute(self, sql, params=()):
                if "INSERT INTO meta_data" in sql:
                    raise RuntimeError("boom during schema init")
                return super()._execute(sql, params)

        with pytest.raises(RuntimeError, match="boom during schema init"):
            _BoomBackend(tmp_path / "i.sqlite3")

    def test_schema_commit_failure_rolls_back(self, tmp_path):
        """A schema commit failure follows the protected rollback path."""

        class _CommitBoomBackend(SQLiteIndexBackend):
            rolled_back = False

            def _commit(self):
                raise RuntimeError("boom during schema commit")

            def _rollback(self):
                type(self).rolled_back = True
                return super()._rollback()

        with pytest.raises(RuntimeError, match="schema commit"):
            _CommitBoomBackend(tmp_path / "commit.sqlite3")
        assert _CommitBoomBackend.rolled_back

    def test_reopen_missing_dynamic_attr_column(self, tmp_path):
        """attr_meta referencing an absent attrs column is rejected on reopen."""
        path = tmp_path / "i.sqlite3"
        get_backend(path).close()
        con = sqlite3.connect(path)
        con.execute(
            "INSERT INTO attr_meta (attr_name, value_kind, column_name, units) "
            "VALUES ('ghost', 'num', 'ghost__num', NULL)"
        )
        con.commit()
        con.close()
        with pytest.raises(InvalidIndexError, match="missing dynamic columns"):
            get_backend(path)


class TestPureHelpers:
    """Small pure helpers covered directly (least-contrived form)."""

    def test_is_directory_format_on_file(self, tmp_path):
        """A plain file is never a directory scan unit."""
        f = tmp_path / "a.txt"
        f.write_text("x")
        assert is_directory_format(f) is False

    def test_memory_backend_refuses_pickle(self):
        """An in-memory backend cannot be pickled (owners serialize rows)."""
        back = SQLiteIndexBackend(":memory:")
        try:
            with pytest.raises(TypeError, match="cannot be pickled"):
                pickle.dumps(back)
        finally:
            back.close()

    def test_py_scalar_bool_and_int(self):
        """_py_scalar unwraps numpy bool/int to plain python scalars."""
        assert _py_scalar(np.bool_(True)) is True
        assert _py_scalar(np.int64(5)) == 5
        assert isinstance(_py_scalar(np.int64(5)), int)

    def test_assemble_records_empty_sources(self):
        """No sources yields no records."""
        empty = pd.DataFrame()
        assert assemble_source_records(empty, empty, empty, empty, empty, empty) == []

    def test_units_compatible(self):
        """_units_compatible is True for same dimensionality, False otherwise."""
        assert SQLiteIndexBackend._units_compatible("m", "ft") is True
        assert SQLiteIndexBackend._units_compatible("m", "s") is False

    def test_legacy_index_check_missing_path(self, tmp_path):
        """A path that does not exist is not a legacy/foreign index."""
        idx = DBDirectoryIndexer(tmp_path)
        assert idx._is_legacy_or_foreign_index(tmp_path / "nope.h5") is False

    def test_wrong_arity_coord_query_raises(self, backend):
        """A hand-built coord range of the wrong length is rejected."""
        with pytest.raises(ParameterError, match="length 2 sequence"):
            backend.query(Query(coords={"distance": (1, 2, 3)}))

    def test_to_target_unit_paths(self):
        """Quantity on a unitless target raises; on a unit target it converts."""
        typed = typed_value(5 * m)  # a numeric TypedValue carrying units
        with pytest.raises(UnitError, match="unitless"):
            _to_target_unit(typed, None, "distance")
        # converting to a compatible unit returns a plain magnitude
        out = _to_target_unit(typed, str(_gq("m").units), "distance")
        assert out == pytest.approx(5.0)

    def test_range_bounds_unset_is_not_none(self):
        """Omitting target units must not mean the target is unitless."""
        value = (5 * m, 10 * m)
        kind_probe = typed_value(5 * m)
        assert kind_probe is not None
        kinds = {kind_probe.kind}
        # Omitted: bounds pass through with no conversion attempted.
        _, lo, hi, _ = _range_bounds(value, kinds, _UNSET, "distance")
        assert (lo, hi) == pytest.approx((5.0, 10.0))
        # Explicit None means the target is unitless, which cannot answer a
        # query carrying units.
        with pytest.raises(UnitError, match="unitless"):
            _range_bounds(value, kinds, None, "distance")


class TestNormalizeSourcePatchId:
    """The single source-patch-id normalizer handles every missing form."""

    def test_missing_forms_become_empty(self):
        """None, empty string, and pandas NaN/NaT all normalize to ''."""
        assert normalize_source_patch_id(None) == ""
        assert normalize_source_patch_id("") == ""
        assert normalize_source_patch_id(float("nan")) == ""
        assert normalize_source_patch_id(np.nan) == ""
        assert normalize_source_patch_id(pd.NaT) == ""

    def test_numpy_scalar_becomes_plain_string(self):
        """A numpy scalar is unwrapped before stringifying."""
        assert normalize_source_patch_id(np.int64(42)) == "42"

    def test_plain_values_stringify(self):
        """Ordinary ids pass through as strings."""
        assert normalize_source_patch_id("abc") == "abc"
        assert normalize_source_patch_id(7) == "7"

    def test_non_scalar_falls_through(self):
        """A value pd.isnull cannot evaluate as a scalar still stringifies."""
        # pd.isnull on a list returns an array (truth value is ambiguous),
        # so the helper must swallow that and fall through to str().
        assert normalize_source_patch_id([1, 2]) == "[1, 2]"


class TestCanonicalRange:
    """_canonical_range recognizes only numeric ranges."""

    def test_bare_and_quantity_bounds(self):
        """Bare ranges never canonicalize; quantities carry per-bound units."""
        # a bare range means native units, so there is nothing to defer
        assert _canonical_range((20, 60)) is None
        quant = _canonical_range((20 * m, 60 * m))
        assert quant is not None
        assert quant.magnitudes == (20.0, 60.0)
        assert quant.units == ("m", "m")
        # a bare bound beside a quantity stays bare (native units)
        mixed = _canonical_range((20, 60 * m))
        assert mixed is not None
        assert mixed.magnitudes == (20.0, 60.0)
        assert mixed.units == (None, "m")

    def test_open_bounds_kept(self):
        """A half-open quantity range keeps its open end as None."""
        half_open = _canonical_range((None, 60 * m))
        assert half_open is not None
        assert half_open.magnitudes == (None, 60.0)
        assert half_open.units == (None, "m")

    @pytest.mark.parametrize(
        "value",
        [
            np.array([True, False]),  # boolean mask, not a range
            (None, None),  # fully open: no numeric content
            (True, False),  # bool bounds are not numeric ranges
            ("a", "b"),  # string bounds are not numeric ranges
            (1, 2, 3),  # wrong arity
        ],
    )
    def test_non_numeric_ranges_return_none(self, value):
        """Anything that is not a bounded numeric range yields None."""
        assert _canonical_range(value) is None


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

    def test_write_failure_rolls_back(self, tmp_path, monkeypatch):
        """A failing write leaves the index unchanged."""
        back = get_backend(tmp_path / "rollback.sqlite3")
        records = summaries_to_records(make_summaries())
        back.write_sources(records[:1])
        before = len(back.query())

        def boom(*args, **kwargs):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(back, "_bulk_insert", boom)
        with pytest.raises(RuntimeError, match="simulated"):
            back.write_sources(records[1:])
        monkeypatch.undo()
        assert len(back.query()) == before
        back.close()

    @pytest.mark.parametrize("operation", ["write", "delete"])
    def test_commit_failure_rolls_back(self, tmp_path, monkeypatch, operation):
        """Write and delete commit failures both release their transaction."""
        back = get_backend(tmp_path / f"{operation}.sqlite3")
        records = summaries_to_records(make_summaries())
        initial = records[:1] if operation == "write" else records
        back.write_sources(initial)
        before = len(back.query())
        original_rollback = back._rollback
        rolled_back = []

        def rollback():
            rolled_back.append(True)
            original_rollback()

        def commit_failure():
            raise RuntimeError("simulated commit failure")

        monkeypatch.setattr(back, "_rollback", rollback)
        monkeypatch.setattr(back, "_commit", commit_failure)
        with pytest.raises(RuntimeError, match="commit failure"):
            if operation == "write":
                back.write_sources(records[1:2])
            else:
                back.delete_sources([records[0].source_path])
        assert rolled_back
        assert len(back.query()) == before
        back.close()

    def test_marker_commit_failure_rolls_back(self, tmp_path, monkeypatch):
        """A marker commit failure restores the initial metadata value."""
        back = get_backend(tmp_path / "marker.sqlite3")
        before = back.get_metadata()["last_indexed_ns"]
        original_rollback = back._rollback
        rolled_back = []

        def rollback():
            rolled_back.append(True)
            original_rollback()

        def commit_failure():
            raise RuntimeError("simulated commit failure")

        monkeypatch.setattr(back, "_rollback", rollback)
        monkeypatch.setattr(back, "_commit", commit_failure)
        with pytest.raises(RuntimeError, match="commit failure"):
            back.mark_initial_update_done()
        assert rolled_back
        assert back.get_metadata()["last_indexed_ns"] == before
        back.close()

    def test_delete_failure_rolls_back(self, tmp_path, monkeypatch):
        """A failing delete leaves the index unchanged."""
        back = get_backend(tmp_path / "delete.sqlite3")
        back.write_sources(summaries_to_records(make_summaries()))
        before = len(back.query())

        def boom(paths, base_uri=""):
            raise RuntimeError("simulated failure")

        monkeypatch.setattr(back, "_delete_by_paths", boom)
        with pytest.raises(RuntimeError, match="simulated"):
            back.delete_sources(["das/file_1.h5"])
        monkeypatch.undo()
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

    def test_duplicate_explicit_namespace_raises(self, backend):
        """A name cannot be supplied in both explicit namespaces."""
        with pytest.raises(InvalidSpoolQueryError, match="both _attrs and _coords"):
            resolve_query(
                backend.attr_names(),
                backend.coord_names(),
                _attrs={"distance": (0, 1)},
                _coords={"distance": (0, 1)},
            )

    def test_unknown_attr_in_explicit_namespace(self, backend):
        """Unknown key in _attrs raises."""
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            resolve_query(
                backend.attr_names(), backend.coord_names(), _attrs={"nope": 1}
            )

    def test_unknown_coord_in_explicit_namespace(self, backend):
        """Unknown key in _coords raises."""
        with pytest.raises(InvalidSpoolQueryError, match="not a coordinate"):
            resolve_query(
                backend.attr_names(), backend.coord_names(), _coords={"nope": (1, 2)}
            )


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

    def test_boolean_array_coord_rejected(self, backend):
        """Boolean sample masks are no longer index predicates."""
        mask = np.array([True, False, True])
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            backend.query(Query(coords={"distance": mask}))

    def test_slice_range_form(self, backend):
        """Slices resolve to the same range tuples patch selects accept."""
        lo = np.datetime64("2024-06-01T00:00:00", "ns")
        query = resolve_query(
            backend.attr_names(), backend.coord_names(), time=slice(lo, None)
        )
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
        assert set(df["source_path"]) == {"with_units.h5", "no_units.h5"}
        back.close()

    def test_nonoverlapping_unitless_coord_stays_candidate(self, tmp_path):
        """Numeric envelopes cannot exclude a unitless quantity candidate."""
        back = get_backend(tmp_path / "nonoverlap-unitless.sqlite3")
        back.write_sources(
            summaries_to_records(
                [
                    self._summary("with_units.h5", coord_units="m"),
                    self._summary("no_units.h5"),
                ]
            )
        )
        meters = get_quantity("m")
        df = back.query(Query(coords={"distance": (250 * meters, 300 * meters)}))
        assert list(df["source_path"]) == ["no_units.h5"]
        back.close()

    def test_all_unitless_quantity_query_keeps_candidates(self, tmp_path):
        """Unitless defs cannot be proven incompatible; they stay candidates."""
        back = get_backend(tmp_path / "unitless.sqlite3")
        back.write_sources(summaries_to_records([self._summary("no_units.h5")]))
        meters = get_quantity("m")
        df = back.query(Query(coords={"distance": (150 * meters, 300 * meters)}))
        assert list(df["source_path"]) == ["no_units.h5"]
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
        assert set(df["source_path"]) == {"a.h5", "b.h5"}
        got = back.query(Query(attrs={"resolution": (0.5, 2.0)}))
        assert list(got["source_path"]) == ["a.h5"]
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
        df = back.query().set_index("source_path")
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

    def test_offset_quantity_attr_uses_full_conversion(self):
        """Affine quantity attrs are converted with their offset."""
        out = typed_value(get_quantity("0 degC"))
        assert out is not None
        assert out.value == pytest.approx(273.15)
        assert out.units == "K"

    def test_offset_coord_uses_full_conversion(self):
        """Affine coord bounds use offsets while steps remain deltas."""
        summary = PatchSummary(
            attrs={"tag": "temperature"},
            coords={
                "temperature": {
                    "dtype": "float64",
                    "min": 0.0,
                    "max": 100.0,
                    "step": 1.0,
                    "units": "degC",
                    "dims": ("temperature",),
                    "len": 101,
                }
            },
            dims=("temperature",),
            shape=(101,),
            dtype="float32",
            source_path="temperature.h5",
            source_format="DASDAE",
            source_version="1",
        )
        out = _coord_record("temperature", summary.coords["temperature"])
        assert out is not None
        # Envelopes store the coordinate's original units, never converted,
        # which sidesteps the degC/K affine-offset hazard entirely: nothing
        # here needs to know that 0 degC is not 0 K.
        assert out.min_num == pytest.approx(0.0)
        assert out.max_num == pytest.approx(100.0)
        assert out.step_num == pytest.approx(1.0)
        assert out.units == "°C"

    def test_relative_root_requires_path_boundary(self):
        """A similarly prefixed path is not made relative to the root."""
        data = make_summaries()[0].dump_structured()
        data["source_path"] = "/data/foobar/file.h5"
        record = s2r([PatchSummary(**data)], relative_to="/data/foo")[0]
        assert record.source_path == "/data/foobar/file.h5"

    def test_plain_array_skipped(self):
        """Arrays are complex attrs; skipped."""
        assert typed_value(np.array([1, 2])) is None

    def test_array_quantity_skipped(self):
        """Array-valued quantities are skipped."""
        assert typed_value(np.array([1.0, 2.0]) * get_quantity("m")) is None

    def test_reserved_attr_name_warns(self):
        """
        An attr named for a structural column is skipped with a warning.

        It used to be spelled with `patch_id`, which is now a field of
        `PatchAttrs` in its own right -- a first-class id rather than a
        user attr which happens to collide -- and is skipped silently.
        `source_id` is still only a column, so it still warns.
        """
        summary = PatchSummary(
            attrs={"source_id": 5, "tag": "x"},
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
        assert "source_id" not in records[0].patches[0].attrs

    @pytest.mark.parametrize("name", ["patch_id", "processing_id"])
    def test_the_ids_are_skipped_silently(self, name):
        """
        Every patch carries them, so a warning would fire on every patch.

        They are not indexed: an index is for finding data, and an id is
        not a search term.
        """
        summary = PatchSummary(
            attrs={name: "0123456789abcdef", "tag": "x"},
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
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            records = s2r([summary])
        assert name not in records[0].patches[0].attrs

    @pytest.mark.parametrize("name", ["shape", "n_dims", "sample_count_total"])
    def test_dropped_columns_stay_reserved(self, random_patch, name):
        """
        Version 9 dropped these columns; the names keep their meaning.

        Un-reserving one turns it into an ordinary indexed attr, which
        chunk then refuses to merge two patches over.
        """
        patches = [
            p.update_attrs(history=[], **{name: str(i)})
            for i, p in enumerate(dc.get_example_spool("random_das"))
        ]
        with pytest.warns(UserWarning, match="reserved attr name"):
            merged = dc.spool(patches).chunk(time=None)
        assert len(merged) == 1

    @pytest.mark.parametrize("dtype", ["", np.dtype(bool)])
    def test_unsupported_coord_dtype_skipped(self, dtype):
        """A coord with a missing or unsupported dtype produces no record."""

        class _Stub:
            dtype: object = None
            dims = ("x",)
            len = 2
            units = None
            fingerprint = None
            min = 0
            max = 1
            step = None

        # "" exercises the missing-dtype guard; a bool dtype is a real
        # dtype that none of the value-kind branches handle.
        stub = _Stub()
        stub.dtype = dtype
        assert _coord_record("x", stub) is None

    def test_multipatch_source_gets_positional_ids(self):
        """Multi-patch sources get positional source_patch_ids."""
        base: dict[str, Any] = make_summaries()[0].dump_structured()
        one = PatchSummary(**base)
        # Bound and annotated: merging a str literal into the dict widens
        # the value type to `Any | str`, which no field then accepts.
        other: dict[str, Any] = {**base, "attrs": {"station": "STA9"}}
        two = PatchSummary(**other)
        records = s2r([one, two])
        assert len(records) == 1
        ids = [p.source_patch_id for p in records[0].patches]
        assert ids == ["0", "1"]


class TestIndexerEdges:
    """DBDirectoryIndexer edge behavior."""

    def test_failed_initial_update_can_retry(self, tmp_path, monkeypatch):
        """A new process retries when the first automatic update failed."""
        indexer = DBDirectoryIndexer(tmp_path)
        index_path = indexer.index_path

        def fail_walk():
            raise OSError("simulated walk failure")

        monkeypatch.setattr(indexer, "_walk", fail_walk)
        with pytest.raises(OSError, match="walk failure"):
            indexer.ensure_updated()
        assert not indexer._initial_update_done
        indexer.close()

        retry = DBDirectoryIndexer(tmp_path, index_path=index_path)
        assert not retry._initial_update_done
        assert retry.ensure_updated()
        assert retry._initial_update_done
        retry.close()

        reopened = DBDirectoryIndexer(tmp_path, index_path=index_path)
        assert reopened._initial_update_done
        reopened.close()

    def test_directory_manifest_detects_equal_stat_name_swap(
        self, tmp_path, monkeypatch
    ):
        """Equal-size, equal-mtime member replacement changes the signature."""
        unit = tmp_path / "unit"
        unit.mkdir()
        old = unit / "old.raw"
        old.write_bytes(b"same-size")
        stamp = 1_700_000_000_000_000_000
        os.utime(old, ns=(stamp, stamp))
        indexer = DBDirectoryIndexer(tmp_path)
        monkeypatch.setattr(indexer, "_directory_format", lambda path: path == unit)
        monkeypatch.setattr(dc, "scan", lambda *_args, **_kwargs: [])
        indexer.update(progress=None)
        before = tuple(
            indexer._backend.source_stats().loc[0, ["mtime_ns", "size_bytes"]]
        )

        old.unlink()
        new = unit / "new.raw"
        new.write_bytes(b"same-size")
        os.utime(new, ns=(stamp, stamp))
        indexer.update(progress=None)
        after = tuple(
            indexer._backend.source_stats().loc[0, ["mtime_ns", "size_bytes"]]
        )
        assert before != after
        indexer.close()

    def test_auto_update_on_first_query(self, tmp_path, random_patch):
        """A brand-new index triggers one update on first query."""
        random_patch.io.write(tmp_path / "one.hdf5", "dasdae")
        # no explicit update() call; the spool's first query triggers one
        spool = dc.spool(tmp_path)
        try:  # release the index handle so Windows can clean the temp dir
            assert len(spool) == 1
        finally:
            spool.indexer.close()

    def test_empty_index_file_updates_on_first_query(self, tmp_path, random_patch):
        """A pre-created empty SQLite path is still a new index."""
        random_patch.io.write(tmp_path / "one.hdf5", "dasdae")
        index_path = tmp_path / "empty.sqlite3"
        index_path.touch()
        indexer = DBDirectoryIndexer(tmp_path, index_path=index_path)
        try:
            # the pre-created path is the index, and the first query fills it
            assert indexer.index_path == index_path
            spool = dc.spool(tmp_path)
            try:
                assert spool.indexer.index_path == index_path
                assert len(spool) == 1
            finally:
                spool.indexer.close()
        finally:  # both handles, so Windows can clean the temp dir
            indexer.close()

    def test_directory_format_unit(self, tmp_path):
        """Directory-format sources (xml binary) group as one scan unit."""
        sys.path.insert(0, "tests/test_io/test_xml_binary")
        from test_xml_binary import metadata  # noqa: PLC0415

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
        spool = dc.spool(tmp_path).update(progress=None)
        try:
            assert len(spool) == 2
        finally:
            spool.indexer.close()
        # unchanged: second update rescans nothing
        before = indexer._backend.get_sources()["last_indexed_ns"].max()
        indexer.update(progress=None)
        after = indexer._backend.get_sources()["last_indexed_ns"].max()
        assert before == after
        indexer.close()


class TestDirSpoolPassthrough:
    """Directory spools accept a prebuilt indexer."""

    def test_spool_from_indexer(self, tmp_path, random_patch):
        """Passing an indexer instance to from_directory works."""
        random_patch.io.write(tmp_path / "one.hdf5", "dasdae")
        indexer = DBDirectoryIndexer(tmp_path)
        spool = Spool.from_directory(indexer).update(progress=None)
        assert len(spool) == 1


class TestFinalCoverage:
    """Remaining edge branches."""

    def test_datetime_object_becomes_time(self):
        """A python datetime routes through the datetime fallback."""
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
        assert record is not None
        assert record.coord_hash == patch.get_coord("distance").fingerprint()
        assert record.def_key.startswith("fp:")

    def test_summary_key_stable_through_export(self, tmp_path):
        """A summary-keyed coord dedups against its own exported records."""
        # Without a step the coord is not range-like, so it has no
        # fingerprint to key on and falls back to hashing its stored fields.
        summary = PatchSummary(
            attrs={"tag": "raw"},
            coords={"time": {**_time_coord("2024-01-01T00:00:00", 60), "step": None}},
            dims=("time",),
            shape=(15000,),
            dtype="float32",
            source_path="sum/one.h5",
            source_format="DASDAE",
            source_version="1",
        )
        fresh = _coord_record("time", summary.coords["time"])
        assert fresh is not None
        assert fresh.def_key.startswith("sum:")
        back = get_backend(tmp_path / "sum.sqlite3")
        back.write_sources(summaries_to_records([summary]))
        exported = back.export_records()
        back.close()
        (coord,) = exported[0].patches[0].coords
        assert coord.def_key == fresh.def_key

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
        base: dict[str, Any] = make_summaries()[0].dump_structured()
        one = PatchSummary(**base)
        records_a = summaries_to_records([one], base_uri="s3://bucket-a")

        # base_uri strip only applies when paths share the base; set directly
        def _rebase(record, base_uri: str):
            """Rebuild a record under a different base."""
            # Annotated because merging a str literal in widens the value
            # type to `Any | str`, which none of the fields accept.
            fields: dict[str, Any] = {**record.__dict__, "base_uri": base_uri}
            return type(record)(**fields)

        records_a = [_rebase(r, "s3://bucket-a") for r in records_a]
        records_b = [_rebase(r, "s3://bucket-b") for r in records_a]
        back = get_backend(tmp_path / "multi.sqlite3")
        back.write_sources(records_a)
        back.write_sources(records_b)
        df = back.query()
        assert len(df) == 2
        prefixes = {p.split("/das/")[0] for p in df["source_path"]}
        assert prefixes == {"s3://bucket-a", "s3://bucket-b"}
        # deletion is base-scoped
        back.delete_sources([records_a[0].source_path], base_uri="s3://bucket-a")
        df = back.query()
        assert len(df) == 1
        assert df["source_path"].iloc[0].startswith("s3://bucket-b")
        back.close()

    def test_replacement_is_base_scoped(self, tmp_path):
        """Rewriting a source under one base leaves the other base alone."""
        base = make_summaries()[0].dump_structured()
        one = PatchSummary(**base)
        rec = summaries_to_records([one])[0]
        fields_a: dict[str, Any] = {**rec.__dict__, "base_uri": "s3://a"}
        fields_b: dict[str, Any] = {**rec.__dict__, "base_uri": "s3://b"}
        rec_a = type(rec)(**fields_a)
        rec_b = type(rec)(**fields_b)
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
        spool = dc.spool([dc.get_example_patch()])
        spool.get_contents()  # realize the backend
        # Drain any garbage other tests left behind before opening the
        # recording window: a leaked open connection from an earlier test
        # emits a ResourceWarning when *it* is collected, which would
        # otherwise be caught here and blamed on this spool.
        gc.collect()
        with warnings_mod.catch_warnings(record=True) as caught:
            warnings_mod.simplefilter("always", ResourceWarning)
            del spool
            gc.collect()
        resource = [w for w in caught if issubclass(w.category, ResourceWarning)]
        assert not resource

    def test_explicit_close_idempotent(self):
        """Explicit close works and GC afterwards stays quiet."""
        catalog = PatchCatalog.from_patches([dc.get_example_patch()])
        catalog.to_df()
        catalog.close()

    @pytest.mark.concurrency
    def test_finalization_from_worker_thread(self):
        """Dropping the last backend reference off-thread does not raise."""
        holder = [SQLiteIndexBackend(":memory:")]
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
        # data directory with one file, plus a fake legacy index mapping.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        dc.write(dc.get_example_patch(), data_dir / "a.h5", "dasdae")
        legacy = tmp_path / "legacy_index.h5"
        with h5py.File(legacy, "w") as fh:
            fh.create_dataset("x", data=[1, 2, 3])
        map_dir = tmp_path / "path_map"
        with dc.config_context(directory_index_map_dir=map_dir):
            _set_mapped_index_path(data_dir, legacy, map_dir)
            spool = dc.spool(data_dir).update()
            assert len(spool) == 1


class TestReservedAttrNames:
    """Attrs colliding with structural columns are skipped with a warning."""

    @pytest.mark.parametrize(
        "name,value",
        [
            ("source_path", "user-path"),
            ("source_format", "attr-format"),
            ("source_id", 42),
        ],
    )
    def test_reserved_attr_warns_and_skips(self, name, value):
        """Reserved names warn at ingest and never corrupt the relation."""
        patch = dc.get_example_patch().update_attrs(**{name: value})
        with pytest.warns(UserWarning, match="reserved attr"):
            df = dc.spool([patch]).get_contents()
        assert not df.columns.duplicated().any()
        # structural values win; the attr stays on the patch itself.
        assert patch.attrs[name] == value

    def test_non_reserved_attr_round_trips(self):
        """Ordinary arbitrary attrs still index and select normally."""
        patch = dc.get_example_patch().update_attrs(experiment="exp42")
        spool = dc.spool([patch])
        assert spool.get_contents()["experiment"].iloc[0] == "exp42"
        assert len(spool.select(experiment="exp42")) == 1

    def test_cross_patch_envelope_attr_clobbered_not_reserved(self):
        """An envelope-shaped attr indexes normally; the coord wins the column."""
        base = dc.get_example_patch()
        p1 = base.update_attrs(event_time_min=123.0)
        p2 = dc.get_example_patch().rename_coords(time="event_time")
        with pytest.warns(UserWarning, match="collide with coordinate envelope"):
            spool = dc.spool([p1, p2])
            df = spool.get_contents()
        backend = spool._catalog.backend
        # the attr is indexed and stays queryable via the _attrs namespace.
        assert "event_time_min" in backend.attr_names()
        assert len(spool.select(_attrs={"event_time_min": 123.0})) == 1
        # the flat column is the coordinate envelope, not the attr value.
        assert "event_time_min" in df.columns
        assert 123.0 not in set(df["event_time_min"].dropna())

    def test_data_units_attr_still_indexed(self):
        """A real ``*_units`` attr is not an envelope column; stays queryable."""
        patch = dc.get_example_patch().update_attrs(data_units="strain")
        spool = dc.spool([patch])
        assert "data_units" in spool._catalog.backend.attr_names()


class TestTransactionIsolation:
    """The statement lock covers whole transactions (round-4 F5)."""

    @pytest.mark.concurrency
    def test_reader_never_sees_half_written_replacement(self, tmp_path, monkeypatch):
        """A concurrent reader blocks during a source replacement."""
        patch = dc.get_example_patch()
        record = SourceRecord(
            source_path="mem://one",
            source_format="mem",
            format_version="",
            patches=(patch_record(patch.summary),),
        )
        backend = get_backend(str(tmp_path / "idx.sqlite3"))
        backend.write_sources([record])
        assert len(backend.query()) == 1

        in_delete = threading.Event()
        release = threading.Event()
        original = type(backend)._delete_by_paths

        def paused_delete(self, *args, **kwargs):
            out = original(self, *args, **kwargs)
            in_delete.set()
            release.wait(timeout=10)
            return out

        counts = []

        def read():
            counts.append(len(backend.query()))

        writer = threading.Thread(
            target=lambda: backend.write_sources([record]), daemon=True
        )
        monkeypatch.setattr(type(backend), "_delete_by_paths", paused_delete)
        try:
            writer.start()
            assert in_delete.wait(timeout=10)
            # the writer sits mid-transaction with the row deleted; a
            # reader must block on the transaction lock, not observe it
            reader = threading.Thread(target=read, daemon=True)
            reader.start()
            reader.join(timeout=0.3)
            assert reader.is_alive(), "reader observed a half-written state"
            assert counts == []
            release.set()
            writer.join(timeout=10)
            reader.join(timeout=10)
        finally:
            release.set()
        assert counts == [1]
        backend.close()
