"""
Contract tests for spool index backends.

Every backend must pass this suite unchanged; it encodes the selector
semantics spec and the summary-only/no-false-negatives contract from the
index design doc (see discussion #648).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import pytest

from dascore.core.summary import PatchSummary
from dascore.io.index import Query, get_backend, summaries_to_records
from dascore.io.index.backend import resolve_query
from dascore.io.index.query import InvalidSpoolQueryError

BACKENDS = ("duckdb", "sqlite", "parquet")


def _time_coord(t0: str, seconds: float, step_s: float = 0.004):
    """Make an absolute time coord summary dict."""
    start = np.datetime64(t0, "ns")
    return {
        "dtype": "datetime64",
        "min": start,
        "max": start + np.timedelta64(int(seconds * 1e9), "ns"),
        "step": np.timedelta64(int(step_s * 1e9), "ns"),
        "units": "s",
        "dims": ("time",),
        "len": int(seconds / step_s),
    }


def _distance_coord(d0: float, d1: float, step: float, units="m"):
    """Make a numeric distance coord summary dict."""
    return {
        "dtype": "float64",
        "min": d0,
        "max": d1,
        "step": step,
        "units": units,
        "dims": ("distance",),
        "len": int((d1 - d0) / step) + 1,
    }


def make_summaries() -> list[PatchSummary]:
    """A deliberately heterogeneous set of patch summaries."""
    das1 = PatchSummary(
        attrs={
            "station": "STA1",
            "network": "NW",
            "tag": "raw",
            "data_type": "strain_rate",
            "gauge_length": 10,
        },
        coords={
            "time": _time_coord("2024-01-01T00:00:00", 60),
            "distance": _distance_coord(0, 1000, 1),
        },
        dims=("time", "distance"),
        shape=(15000, 1001),
        dtype="float32",
        source_path="das/file_1.h5",
        source_format="PRODML",
        source_version="2.1",
    )
    das2 = PatchSummary(
        attrs={
            "station": "STA2",
            "network": "NW",
            "tag": "raw",
            "data_type": "strain_rate",
            "gauge_length": 10.0,
        },
        coords={
            "time": _time_coord("2024-01-01T00:01:00", 60),
            "distance": _distance_coord(0, 1000, 1),
        },
        dims=("time", "distance"),
        shape=(15000, 1001),
        dtype="float32",
        source_path="das/file_2.h5",
        source_format="PRODML",
        source_version="2.1",
    )
    # correlogram: relative (timedelta) lag_time coord, shot_number attr
    correlogram = PatchSummary(
        attrs={"tag": "corr", "shot_number": 42, "data_type": ""},
        coords={
            "lag_time": {
                "dtype": "timedelta64",
                "min": np.timedelta64(-5_000_000_000, "ns"),
                "max": np.timedelta64(5_000_000_000, "ns"),
                "step": np.timedelta64(10_000_000, "ns"),
                "dims": ("lag_time",),
                "len": 1001,
            },
            "distance": _distance_coord(0, 500, 5),
        },
        dims=("lag_time", "distance"),
        shape=(1001, 101),
        dtype="float64",
        source_path="products/corr_1.h5",
        source_format="DASDAE",
        source_version="1",
    )
    # PSD-like product with distance in feet (tests SI normalization)
    psd = PatchSummary(
        attrs={"tag": "psd", "shot_number": "unknown"},
        coords={
            "frequency": {
                "dtype": "float64",
                "min": 0.0,
                "max": 500.0,
                "step": 0.5,
                "units": "Hz",
                "dims": ("frequency",),
                "len": 1001,
            },
            "distance": _distance_coord(0.0, 3280.0, 3.28, units="ft"),
        },
        dims=("frequency", "distance"),
        shape=(1001, 1001),
        dtype="float64",
        source_path="products/psd_1.h5",
        source_format="DASDAE",
        source_version="1",
    )
    return [das1, das2, correlogram, psd]


@pytest.fixture(scope="function", params=BACKENDS)
def backend(request, tmp_path):
    """An index backend of each kind, freshly ingested."""
    path = tmp_path / f"index_{request.param}"
    back = get_backend(path, kind=request.param)
    back.write_sources(summaries_to_records(make_summaries()))
    yield back
    back.close()


class TestFlatRelation:
    """The flat patch-row relation contract."""

    def test_row_per_patch(self, backend):
        """Row per patch."""
        df = backend.query()
        assert len(df) == 4

    def test_structural_columns(self, backend):
        """Structural columns."""
        df = backend.query()
        for col in ("path", "file_format", "file_version", "dims", "shape"):
            assert col in df.columns
        assert pd.api.types.is_datetime64_dtype(df["time_min"])
        assert pd.api.types.is_timedelta64_dtype(df["time_step"])

    def test_attr_columns_use_original_names(self, backend):
        """Attr columns use original names."""
        df = backend.query()
        assert "station" in df.columns
        assert "gauge_length" in df.columns
        assert set(df["station"].replace("", None).dropna()) == {"STA1", "STA2"}

    def test_missing_str_attrs_are_empty_string(self, backend):
        """Missing str attrs are empty string."""
        df = backend.query()
        corr = df[df["tag"] == "corr"]
        assert (corr["station"] == "").all()

    def test_relative_time_patches_have_null_time_min(self, backend):
        """Relative time patches have null time min."""
        df = backend.query()
        corr = df[df["tag"] == "corr"]
        assert corr["time_min"].isnull().all()

    def test_ordering_deterministic(self, backend):
        """Ordering deterministic."""
        df1, df2 = backend.query(), backend.query()
        pd.testing.assert_frame_equal(df1, df2)
        # NULLS LAST: relative-time patches sort after absolute ones.
        nulls = df1["time_min"].isnull().to_numpy()
        assert not nulls[: (~nulls).sum()].any()


class TestAttrPredicates:
    """Attr predicates are exact at the index."""

    def test_equality(self, backend):
        """Equality."""
        df = backend.query(Query(attrs={"station": "STA1"}))
        assert len(df) == 1
        assert df["station"].iloc[0] == "STA1"

    def test_glob(self, backend):
        """Glob."""
        df = backend.query(Query(attrs={"station": "STA*"}))
        assert len(df) == 2

    def test_regex(self, backend):
        """Regex."""
        df = backend.query(Query(attrs={"station": re.compile(r"STA\d")}))
        assert len(df) == 2

    def test_membership(self, backend):
        """Membership."""
        df = backend.query(Query(attrs={"station": ["STA1", "STA2", "NOPE"]}))
        assert len(df) == 2

    def test_int_matches_float_storage(self, backend):
        """Int matches float storage."""
        # gauge_length stored from int 10 and float 10.0; int query hits both
        df = backend.query(Query(attrs={"gauge_length": 10}))
        assert len(df) == 2

    def test_range(self, backend):
        """Range."""
        df = backend.query(Query(attrs={"gauge_length": (5, 15)}))
        assert len(df) == 2

    def test_open_range(self, backend):
        """Open range."""
        df = backend.query(Query(attrs={"gauge_length": (5, None)}))
        assert len(df) == 2

    def test_kind_mismatch_matches_nothing(self, backend):
        """Kind mismatch matches nothing."""
        # station is a str attr; numeric query is valid but matches nothing
        df = backend.query(Query(attrs={"station": 5}))
        assert df.empty

    def test_mixed_kind_attr(self, backend):
        """Mixed kind attr."""
        # shot_number exists as num (42) and str ("unknown")
        num = backend.query(Query(attrs={"shot_number": 42}))
        assert list(num["tag"]) == ["corr"]
        txt = backend.query(Query(attrs={"shot_number": "unknown"}))
        assert list(txt["tag"]) == ["psd"]


class TestCoordPredicates:
    """Coord predicates: envelope candidacy, never false negatives."""

    def test_time_range(self, backend):
        """Time range."""
        t = (np.datetime64("2024-01-01T00:00:30"), np.datetime64("2024-01-01T00:00:40"))
        df = backend.query(Query(coords={"time": t}))
        assert list(df["station"]) == ["STA1"]

    def test_time_range_overlap_both(self, backend):
        """Time range overlap both."""
        t = (np.datetime64("2024-01-01T00:00:30"), np.datetime64("2024-01-01T00:01:30"))
        df = backend.query(Query(coords={"time": t}))
        assert set(df["station"]) == {"STA1", "STA2"}

    def test_absolute_time_excludes_relative(self, backend):
        """Absolute time excludes relative."""
        t = (np.datetime64("1990-01-01"), np.datetime64("2100-01-01"))
        df = backend.query(Query(coords={"time": t}))
        assert "corr" not in set(df["tag"])

    def test_relative_time_coord(self, backend):
        """Relative time coord."""
        lag = (np.timedelta64(0, "s"), np.timedelta64(2, "s"))
        df = backend.query(Query(coords={"lag_time": lag}))
        assert list(df["tag"]) == ["corr"]

    def test_numeric_coord_si_normalized(self, backend):
        """Numeric coord si normalized."""
        # psd distance is 0-3280 ft = 0-999.7 m; a 900-950 m query hits it
        df = backend.query(Query(coords={"distance": (900, 950)}))
        assert "psd" in set(df["tag"])

    def test_scalar_coord(self, backend):
        """Scalar coord."""
        df = backend.query(Query(coords={"frequency": 100}))
        assert list(df["tag"]) == ["psd"]

    def test_array_membership_envelope(self, backend):
        """Array membership envelope."""
        values = np.array([10.0, 20.0, 480.0])
        df = backend.query(Query(coords={"distance": values}))
        assert len(df) == 4  # candidacy: all patches overlap the envelope

    def test_coord_missing_excludes_patch(self, backend):
        """Coord missing excludes patch."""
        df = backend.query(Query(coords={"frequency": (0, 1000)}))
        assert list(df["tag"]) == ["psd"]


class TestNameResolution:
    """Bare kwargs resolve attrs first, then coords; unknown raises."""

    def test_attr_wins(self, backend):
        """Attr wins."""
        query = resolve_query(backend, station="STA1")
        assert "station" in query.attrs

    def test_coord_fallback(self, backend):
        """Coord fallback."""
        query = resolve_query(backend, lag_time=(0, 1))
        assert "lag_time" in query.coords

    def test_unknown_raises(self, backend):
        """Unknown raises."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            resolve_query(backend, wavelength=(1, 2))

    def test_double_specification_raises(self, backend):
        """Double specification raises."""
        with pytest.raises(InvalidSpoolQueryError, match="both"):
            resolve_query(backend, station="STA1", _attrs={"station": "STA2"})

    def test_explicit_namespaces(self, backend):
        """Explicit namespaces."""
        query = resolve_query(
            backend, _attrs={"tag": "raw"}, _coords={"distance": (0, 10)}
        )
        assert query.attrs == {"tag": "raw"} and "distance" in query.coords


class TestNoFalseNegatives:
    """Property: reference-matching patches always appear in results."""

    def test_random_time_ranges(self, backend):
        """Random time ranges."""
        rng = np.random.default_rng(42)
        summaries = make_summaries()
        base = np.datetime64("2024-01-01T00:00:00").astype("datetime64[ns]")
        for _ in range(25):
            lo = base + np.timedelta64(int(rng.integers(-60, 180)), "s")
            hi = lo + np.timedelta64(int(rng.integers(1, 120)), "s")
            result_paths = set(backend.query(Query(coords={"time": (lo, hi)}))["path"])
            for summary in summaries:
                tcoord = summary.coords.get("time")
                if tcoord is None or "datetime" not in str(tcoord.dtype):
                    continue
                overlaps = tcoord.min <= hi and tcoord.max >= lo
                if overlaps:
                    assert str(summary.source_path) in result_paths

    def test_random_numeric_ranges(self, backend):
        """Random numeric ranges."""
        rng = np.random.default_rng(7)
        factor = {"m": 1.0, "ft": 0.3048}
        summaries = make_summaries()
        for _ in range(25):
            lo = float(rng.uniform(-100, 1000))
            hi = lo + float(rng.uniform(1, 500))
            result_paths = set(
                backend.query(Query(coords={"distance": (lo, hi)}))["path"]
            )
            for summary in summaries:
                dcoord = summary.coords.get("distance")
                if dcoord is None:
                    continue
                scale = factor[str(dcoord.units.units)] if dcoord.units else 1.0
                if dcoord.min * scale <= hi and dcoord.max * scale >= lo:
                    assert str(summary.source_path) in result_paths


class TestSourceLifecycle:
    """Source-scoped transactional replacement and deletion."""

    def test_replace_source_drops_stale_rows(self, backend):
        """Replace source drops stale rows."""
        summaries = [s for s in make_summaries() if "file_1" in str(s.source_path)]
        structured = summaries[0].dump_structured()
        structured["attrs"] = {"station": "NEW1"}
        modified = PatchSummary(**structured)
        backend.write_sources(summaries_to_records([modified]))
        df = backend.query()
        assert len(df) == 4  # still one row for that source
        assert "STA1" not in set(df["station"])
        assert "NEW1" in set(df["station"])

    def test_delete_cascades(self, backend):
        """Delete cascades."""
        backend.delete_sources(["das/file_1.h5"])
        df = backend.query()
        assert len(df) == 3
        assert "das/file_1.h5" not in set(df["path"])

    def test_reopen_persists(self, backend, tmp_path):
        """Reopen persists."""
        kind = type(backend).__name__.replace("Backend", "").lower()
        path_map = {
            "duckdb": tmp_path / "index_duckdb",
            "sqlite": tmp_path / "index_sqlite",
            "parquet": tmp_path / "index_parquet",
        }
        backend.close()
        reopened = get_backend(path_map[kind], kind=kind)
        try:
            assert len(reopened.query()) == 4
        finally:
            reopened.close()
        # reopen once more so fixture teardown close() has a live handle
        reopened_again = get_backend(path_map[kind], kind=kind)
        backend.__dict__.update(reopened_again.__dict__)


class TestMetadata:
    """Index metadata and introspection."""

    def test_metadata(self, backend):
        """Metadata."""
        meta = backend.get_metadata()
        assert meta["what_is_this"] == "dascore_spool_index"
        assert meta["index_version"] == 1

    def test_names(self, backend):
        """Names."""
        assert {"station", "tag", "shot_number"} <= backend.attr_names()
        assert {"time", "distance", "lag_time", "frequency"} <= backend.coord_names()

    def test_sources(self, backend):
        """Sources."""
        sources = backend.get_sources()
        assert len(sources) == 4
        assert set(sources["source_format"]) == {"PRODML", "DASDAE"}


class TestCoordPivot:
    """Per-coord envelope columns in the flat relation."""

    def test_generic_coord_envelopes_present(self, backend):
        """Non-conventional dims get {name}_min/max/step columns."""
        df = backend.query()
        for col in ("lag_time_min", "lag_time_max", "frequency_min"):
            assert col in df.columns
        corr = df[df["tag"] == "corr"]
        assert pd.api.types.is_timedelta64_dtype(corr["lag_time_min"].dtype) or (
            corr["lag_time_min"].map(lambda x: hasattr(x, "total_seconds")).all()
        )

    def test_time_distance_envelopes_not_duplicated(self, backend):
        """patches-level envelopes are authoritative; pivot skips them."""
        df = backend.query()
        assert pd.api.types.is_datetime64_dtype(df["time_min"])
        assert df.columns.tolist().count("distance_min") == 1

    def test_def_key_columns_private_and_shared(self, backend):
        """_{name}_def_key exists for every coord; shared coords share keys."""
        df = backend.query()
        assert "_distance_def_key" in df.columns
        das = df[df["station"].isin(["STA1", "STA2"])]
        assert das["_distance_def_key"].nunique() == 1

    def test_pivot_respects_query(self, backend):
        """A filtered result only pivots the rows it contains."""
        df = backend.query(Query(attrs={"tag": "corr"}))
        assert df["lag_time_min"].notna().all()
        assert "frequency_min" not in df.columns or df["frequency_min"].isna().all()
