"""
Contract tests for the spool index backend.

The SQLite backend must pass this suite; it encodes the selector
semantics spec and the summary-only/no-false-negatives contract from the
index design doc (see discussion #648).
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.core.coords import get_coord
from dascore.core.summary import PatchSummary
from dascore.exceptions import UnitError
from dascore.io.index.backend import get_backend
from dascore.io.index.ingest import summaries_to_records
from dascore.io.index.query import InvalidSpoolQueryError, Query, resolve_query
from dascore.io.index.schema import INDEX_VERSION
from dascore.units import get_quantity


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


@pytest.fixture(scope="function")
def backend(tmp_path):
    """A freshly ingested SQLite index backend."""
    path = tmp_path / "index.sqlite3"
    back = get_backend(path)
    back._test_path = path
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
        for col in ("source_path", "source_format", "source_version", "dims"):
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


class TestElementDtype:
    """The data array's dtype is recorded per patch (size-based chunking)."""

    def test_dtype_round_trips(self, backend):
        """Each patch keeps the dtype its summary reported."""
        df = backend.query()

        def _key(path):
            """Compare path-independently; Windows round-trips backslashes."""
            return str(path).replace("\\", "/")

        expected = {_key(x.source_path): x.dtype for x in make_summaries()}
        got = {_key(k): v for k, v in zip(df["source_path"], df["dtype"], strict=True)}
        assert got == expected

    def test_dtype_is_private_in_flat_relation(self):
        """The spool sees `_dtype`, never a public `dtype` column."""
        spool = dc.get_example_spool("random_das")
        df = spool._df
        assert "dtype" not in df.columns
        assert set(df["_dtype"]) == {str(spool[0].data.dtype)}
        assert "_dtype" not in spool.get_contents().columns

    def test_dtype_attr_does_not_shadow_column(self, tmp_path):
        """A patch attr named `dtype` is skipped, not written to the column."""
        summary = make_summaries()[0]
        shadowed = PatchSummary(
            attrs=dict(summary.attrs.model_dump(), dtype="not a dtype"),
            coords={k: v.model_dump() for k, v in summary.coords.items()},
            dims=summary.dims,
            shape=summary.shape,
            dtype=summary.dtype,
            source_path=summary.source_path,
            source_format=summary.source_format,
            source_version=summary.source_version,
        )
        path = tmp_path / "shadow.sqlite3"
        back = get_backend(path)
        try:
            with pytest.warns(UserWarning, match="dtype"):
                back.write_sources(summaries_to_records([shadowed]))
            df = back.query()
            # the structural column keeps the element dtype, not the attr
            assert df["dtype"].iloc[0] == summary.dtype
        finally:
            back.close()


class TestDataSize:
    """The samples a patch holds are recorded per patch."""

    def test_data_size_round_trips(self, backend):
        """Each patch keeps the sample count its summary's shape implied."""
        df = backend.query()

        def _key(path):
            """Compare path-independently; Windows round-trips backslashes."""
            return str(path).replace("\\", "/")

        expected = {_key(x.source_path): math.prod(x.shape) for x in make_summaries()}
        got = {
            _key(k): int(v)
            for k, v in zip(df["source_path"], df["data_size"], strict=True)
        }
        assert got == expected

    def test_shapeless_summary_states_no_size(self, tmp_path):
        """A summary with no shape states no size; none is inferred."""
        summary = make_summaries()[0]
        shapeless = summary.new(shape=())
        path = tmp_path / "shapeless.sqlite3"
        back = get_backend(path)
        try:
            back.write_sources(summaries_to_records([shapeless]))
            assert pd.isnull(back.query()["data_size"].iloc[0])
        finally:
            back.close()

    def test_data_size_attr_does_not_shadow_column(self, tmp_path):
        """A patch attr named `data_size` is skipped, not written."""
        summary = make_summaries()[0]
        shadowed = PatchSummary(
            attrs=dict(summary.attrs.model_dump(), data_size="not a size"),
            coords={k: v.model_dump() for k, v in summary.coords.items()},
            dims=summary.dims,
            shape=summary.shape,
            dtype=summary.dtype,
            source_path=summary.source_path,
            source_format=summary.source_format,
            source_version=summary.source_version,
        )
        path = tmp_path / "shadow.sqlite3"
        back = get_backend(path)
        try:
            with pytest.warns(UserWarning, match="data_size"):
                back.write_sources(summaries_to_records([shadowed]))
            df = back.query()
            assert df["data_size"].iloc[0] == math.prod(summary.shape)
        finally:
            back.close()


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

    @pytest.mark.parametrize(
        "value",
        [
            get_quantity("1 m"),
            [get_quantity("1 m"), get_quantity("2 m")],
            (get_quantity("900 m"), get_quantity("1 km")),
        ],
    )
    def test_quantity_kind_mismatch_matches_nothing(self, backend, value):
        """Quantity forms do not convert against a string-only attribute."""
        df = backend.query(Query(attrs={"station": value}))
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

    def test_quantity_coord_converts_units(self, backend):
        """Quantity selectors convert to the coordinate's canonical units."""
        meter = get_quantity("m")
        df = backend.query(Query(coords={"distance": (900 * meter, 950 * meter)}))
        assert "psd" in set(df["tag"])

    def test_incompatible_quantity_coord_raises(self, backend):
        """A time quantity cannot query a length coordinate."""
        second = get_quantity("s")
        with pytest.raises(UnitError):
            backend.query(Query(coords={"distance": (1 * second, 2 * second)}))

    def test_scalar_coord_rejected(self, backend):
        """Scalar coord predicates have no exact patch meaning; rejected."""
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            backend.query(Query(coords={"frequency": 100}))

    def test_array_membership_rejected(self, backend):
        """Numeric value membership on a coord is rejected, not candidacy."""
        values = np.array([10.0, 20.0, 480.0])
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            backend.query(Query(coords={"distance": values}))

    def test_coord_missing_excludes_patch(self, backend):
        """Coord missing excludes patch."""
        df = backend.query(Query(coords={"frequency": (0, 1000)}))
        assert list(df["tag"]) == ["psd"]


class TestNameResolution:
    """Bare kwargs resolve attrs first, then coords; unknown raises."""

    def test_attr_wins(self, backend):
        """Attr wins."""
        query = resolve_query(
            backend.attr_names(), backend.coord_names(), station="STA1"
        )
        assert "station" in query.attrs

    def test_coord_fallback(self, backend):
        """Coord fallback."""
        query = resolve_query(
            backend.attr_names(), backend.coord_names(), lag_time=(0, 1)
        )
        assert "lag_time" in query.coords

    def test_unknown_raises(self, backend):
        """Unknown raises."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            resolve_query(
                backend.attr_names(), backend.coord_names(), wavelength=(1, 2)
            )

    def test_double_specification_raises(self, backend):
        """Double specification raises."""
        with pytest.raises(InvalidSpoolQueryError, match="both"):
            resolve_query(
                backend.attr_names(),
                backend.coord_names(),
                station="STA1",
                _attrs={"station": "STA2"},
            )

    def test_explicit_namespaces(self, backend):
        """Explicit namespaces."""
        query = resolve_query(
            backend.attr_names(),
            backend.coord_names(),
            _attrs={"tag": "raw"},
            _coords={"distance": (0, 10)},
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
            result_paths = set(
                backend.query(Query(coords={"time": (lo, hi)}))["source_path"]
            )
            for summary in summaries:
                tcoord = summary.coords.get("time")
                if tcoord is None or "datetime" not in str(tcoord.dtype):
                    continue
                overlaps = tcoord.min <= hi and tcoord.max >= lo
                if overlaps:
                    assert str(summary.source_path).replace("\\", "/") in result_paths

    def test_random_numeric_ranges(self, backend):
        """Random numeric ranges."""
        rng = np.random.default_rng(7)
        summaries = make_summaries()
        for _ in range(25):
            lo = float(rng.uniform(-100, 1000))
            hi = lo + float(rng.uniform(1, 500))
            result_paths = set(
                backend.query(Query(coords={"distance": (lo, hi)}))["source_path"]
            )
            for summary in summaries:
                dcoord = summary.coords.get("distance")
                if dcoord is None:
                    continue
                # bare bounds mean native units: the oracle compares raw
                # stored values, no conversion
                if dcoord.min <= hi and dcoord.max >= lo:
                    assert str(summary.source_path).replace("\\", "/") in result_paths


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
        """Delete cascades to patches, attrs, and coord links via the FK."""
        before = backend._fetch_df("SELECT patch_id FROM patches")
        gone = backend._fetch_df(
            "SELECT p.patch_id FROM patches p JOIN sources s "
            "ON s.source_id = p.source_id WHERE s.source_path = 'das/file_1.h5'"
        )["patch_id"].tolist()
        assert gone  # the source had patches to cascade-delete
        backend.delete_sources(["das/file_1.h5"])
        df = backend.query()
        assert len(df) == 3
        assert "das/file_1.h5" not in set(df["source_path"])
        # the deleted source's patches (and their dependents) are gone,
        # not merely filtered out of the query.
        remaining = set(backend._fetch_df("SELECT patch_id FROM patches")["patch_id"])
        assert remaining == set(before["patch_id"]) - set(gone)
        for table in ("attrs", "patch_coords"):
            ids = set(backend._fetch_df(f"SELECT patch_id FROM {table}")["patch_id"])
            assert not (ids & set(gone))

    def test_reopen_persists(self, backend, tmp_path):
        """Reopen persists."""
        path = backend._test_path
        backend.close()
        reopened = get_backend(path)
        try:
            assert len(reopened.query()) == 4
        finally:
            reopened.close()
        # reopen once more so fixture teardown close() has a live handle
        reopened_again = get_backend(path)
        backend.__dict__.update(reopened_again.__dict__)


class TestMetadata:
    """Index metadata and introspection."""

    def test_metadata(self, backend):
        """Metadata."""
        meta = backend.get_metadata()
        assert meta["what_is_this"] == "dascore_spool_index"
        assert meta["index_version"] == INDEX_VERSION

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


class TestLineageIds:
    """A patch can be found by the id it carries, without loading it."""

    @pytest.fixture(scope="class")
    def written_spool(self, tmp_path_factory):
        """Three patches on disk, each its own datum."""
        path = tmp_path_factory.mktemp("lineage_ids")
        with config_context(patch_provenance="disabled"):
            for index, patch in enumerate(dc.get_example_spool("random_das")):
                patch.io.write(path / f"{index}.h5", "dasdae")
        return dc.spool(path).update()

    def test_the_two_ids_are_different_columns(self, written_spool):
        """The row's id is private; the patch's owns the public name."""
        df = written_spool._df
        assert {"_patch_id", "patch_id"}.issubset(df.columns)
        assert df["_patch_id"].tolist() != df["patch_id"].tolist()
        assert "_patch_id" not in written_spool.get_contents().columns

    def test_scanning_and_reading_agree(self, tmp_path):
        """Or an id found in the index would not name the patch it loads."""
        path = tmp_path / "one.h5"
        with config_context(patch_provenance="disabled"):
            dc.get_example_patch().io.write(path, "dasdae")
        assert dc.scan(path)[0].attrs.patch_id == dc.read(path)[0].attrs.patch_id

    def test_selecting_by_id_finds_that_patch(self, written_spool):
        """Which is what indexing the id is for."""
        wanted = written_spool.get_contents()["patch_id"].iloc[1]
        selected = written_spool.select(patch_id=wanted)
        assert len(selected) == 1
        assert selected[0].attrs.patch_id == wanted

    def test_what_was_done_is_indexed(self, tmp_path):
        """
        `processing_id` says what was done, which provenance asks about,
        so it is recorded: the column states what the stored patch had.
        It is lineage rather than a describing attr, so merging never
        compares it (see `_SOURCE_COLUMNS`); it is folded from the
        members instead.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        indexed = spool.get_contents()["processing_id"].iloc[0]
        assert indexed == spool[0].attrs.processing_id
        assert indexed

    @pytest.mark.parametrize("samples", [False, True])
    @pytest.mark.parametrize("bounds_type", [tuple, list])
    def test_a_trim_records_what_a_trim_records(self, tmp_path, samples, bounds_type):
        """
        Native and sample ranges record identically through a spool or patch.

        Lists and tuples already serialize identically. Open ellipses and
        quantity spellings have separate normalization semantics documented
        in the spool selection note.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        whole = spool[0]
        time = whole.get_coord("time")
        window = bounds_type(
            (0, 10) if samples else (None, time.min() + 10 * time.step)
        )
        through_spool = spool.select(time=window, samples=samples)[0]
        on_patch = whole.select(time=window, samples=samples)
        assert through_spool.shape == on_patch.shape != whole.shape
        assert through_spool.attrs.processing_id == on_patch.attrs.processing_id
        assert through_spool.attrs.processing_id != whole.attrs.processing_id
        assert through_spool.attrs.history == on_patch.attrs.history

    @pytest.mark.parametrize("upper", [False, True])
    @pytest.mark.parametrize("start,step", [(0.0, 1.0), (0.1, 0.1), (0.1, 0.001)])
    def test_single_range_keeps_grid_tolerance(self, tmp_path, upper, start, step):
        """A bound within CoordRange's edge tolerance does not advance lineage."""
        patch = dc.get_example_patch().abs()
        values = start + np.arange(patch.shape[0]) * step
        patch.update_coords(distance=values).io.write(tmp_path / "source.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        source = spool[0]
        coord = source.get_coord("distance")
        bounds = (
            (None, coord.max() - coord.step * 1e-12)
            if upper
            else (coord.min() + coord.step * 1e-12, None)
        )
        expected = source.select(distance=bounds)
        selected = spool.select(distance=bounds)[0]
        assert expected.shape == source.shape
        assert np.array_equal(selected.data, source.data)
        assert selected.attrs.processing_id == expected.attrs.processing_id

    def test_a_trim_which_cuts_nothing_records_nothing(self, tmp_path):
        """A selection which leaves the patch whole is not an operation.

        The same selection on the patch hands it straight back, so
        neither route records anything.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        whole = spool[0]
        time = whole.get_coord("time")
        through_spool = spool.select(time=(time.min(), time.max()))[0]
        assert through_spool.shape == whole.shape
        assert through_spool.attrs.processing_id == whole.attrs.processing_id
        assert through_spool.attrs.history == whole.attrs.history

    def test_composed_trims_record_once_each(self, tmp_path):
        """Two trims through a spool record what two trims on the patch do.

        Only the narrower bound reaches the reader, so the wider
        `select` also finds nothing to do; both still happened.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        whole = spool[0]
        time = whole.get_coord("time")
        duration = time.max() - time.min()
        wide = (time.min(), time.min() + duration / 2)
        narrow = (time.min(), time.min() + duration / 4)
        through_spool = spool.select(time=wide).select(time=narrow)[0]
        on_patch = whole.select(time=wide).select(time=narrow)
        assert through_spool.shape == on_patch.shape != whole.shape
        assert through_spool.attrs.processing_id == on_patch.attrs.processing_id
        assert through_spool.attrs.history == on_patch.attrs.history

    def test_a_trim_beside_one_which_cuts_nothing_records_only_the_cut(self, tmp_path):
        """One selection cutting must not make another one look like a trim.

        The two are on different dimensions, so the patch route records
        the distance trim alone; a spool which knows only that the row
        was narrowed somewhere would record the time one as well.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        whole = spool[0]
        time, distance = whole.get_coord("time"), whole.get_coord("distance")
        # wider than the patch on time, so that selection cuts nothing
        span = time.max() - time.min()
        wide = (time.min() - span, time.max() + span)
        narrow = (distance.min(), distance.max() - 2 * distance.step)
        view = spool.select(distance=narrow).select(time=wide)
        on_patch = whole.select(distance=narrow).select(time=wide)
        assert view[0].shape == on_patch.shape != whole.shape
        assert view[0].attrs.processing_id == on_patch.attrs.processing_id
        assert view[0].attrs.history == on_patch.attrs.history

    def test_composed_trims_on_one_dim_record_only_the_ones_which_cut(self, tmp_path):
        """A widening selection composed over a narrowing one is no trim.

        Both bounds are on the same coordinate and the last one reaches
        the reader, so both `select` calls find nothing to do; only the
        one which would have cut the loaded patch is an operation.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        whole = spool[0]
        time = whole.get_coord("time")
        span = time.max() - time.min()
        wide = (time.min() - span, time.max() + span)
        narrow = (time.min(), time.min() + span / 2)
        view = spool.select(time=wide).select(time=narrow)
        on_patch = whole.select(time=wide).select(time=narrow)
        assert view[0].shape == on_patch.shape != whole.shape
        assert view[0].attrs.processing_id == on_patch.attrs.processing_id
        assert view[0].attrs.history == on_patch.attrs.history

    @pytest.mark.parametrize(
        "dim,scale", [("distance", 1), ("distance", 0.1), ("time", 1)]
    )
    @pytest.mark.parametrize("upper", [False, True])
    def test_off_grid_noop_keeps_lineage(self, tmp_path, dim, scale, upper):
        """Two bounds within one sample interval record only the effective trim."""
        patch = dc.get_example_patch()
        if scale != 1:
            patch = patch.update_coords(distance=patch.get_array("distance") * scale)
        patch.io.write(tmp_path / "source.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        whole = spool[0]
        coord = whole.get_coord(dim)
        base = coord.min() + 10 * coord.step
        first = base + coord.step / 2
        second = base + 7 * coord.step / 10
        bounds = (
            ((None, second), (None, first))
            if upper
            else ((first, None), (second, None))
        )
        view = spool.select(**{dim: bounds[0]}).select(**{dim: bounds[1]})
        direct = whole.select(**{dim: bounds[0]}).select(**{dim: bounds[1]})
        assert np.array_equal(view[0].data, direct.data)
        assert view[0].attrs.processing_id == direct.attrs.processing_id
        assert view[0].attrs.history == direct.attrs.history

    @pytest.mark.parametrize("repeats", [2, 64])
    @pytest.mark.parametrize("regular", [False, True])
    def test_untracked_trims_replay_exactly(self, tmp_path, repeats, regular):
        """Uneven grids and chains exceeding the mask still omit no-op calls."""
        values = (
            np.arange(5, dtype=float)
            if regular
            else np.array([0.0, 1.0, 3.0, 6.0, 10.0])
        )
        patch = dc.Patch(
            data=np.arange(5),
            coords={"distance": values},
            dims=("distance",),
        )
        patch.io.write(tmp_path / "source.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        view, direct = spool, spool[0]
        for _ in range(repeats):
            view = view.select(distance=(1.5, None))
            direct = direct.select(distance=(1.5, None))
        assert np.array_equal(view[0].data, direct.data)
        assert view[0].attrs.processing_id == direct.attrs.processing_id
        assert view[0].attrs.history == direct.attrs.history

    @pytest.mark.parametrize("dim", ["distance", "time"])
    @pytest.mark.parametrize("bounds", [(0, None), (None, None), (None, 1e9)])
    def test_relative_noop_keeps_metadata(self, tmp_path, dim, bounds):
        """A relative selection that leaves samples alone keeps source metadata."""
        dc.get_example_patch().abs().io.write(tmp_path / "source.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        view = spool.select(**{dim: bounds}, relative=True)
        columns = ["_data_size", "processing_id"]
        assert view._df[columns].equals(spool._df[columns])
        assert view[0].attrs.processing_id == spool[0].attrs.processing_id
        assert np.array_equal(view[0].data, spool[0].data)

    @pytest.mark.parametrize("percent_bound", [False, True])
    def test_relative_float32_keeps_precision(self, tmp_path, percent_bound):
        """Relative endpoints use the loaded coordinate's arithmetic precision."""
        origin, step = np.float32(538.14794921875), np.float32(3.3)
        values = origin + np.arange(300, dtype=np.float32) * step
        base = dc.get_example_patch().abs()
        for dtype in (np.float32, np.float64):
            base.update_coords(distance=values.astype(dtype)).io.write(
                tmp_path / f"{np.dtype(dtype).name}.h5", "dasdae"
            )
        spool = dc.spool(tmp_path).update()
        bound = (
            100 * get_quantity("percent")
            if percent_bound
            else float(values.max()) - float(values.min())
        )
        view = spool.select(distance=(0, bound), relative=True)
        for index, (source, selected) in enumerate(zip(spool, view, strict=True)):
            direct = source.select(distance=(0, bound), relative=True)
            assert np.array_equal(selected.data, direct.data)
            assert selected.attrs.processing_id == direct.attrs.processing_id
            row = view._df.iloc[index]
            if selected.shape != source.shape:
                assert pd.isna(row["_data_size"])
                assert pd.isna(row["processing_id"])
            else:
                assert row["_data_size"] == selected.data.size
                assert row["processing_id"] == selected.attrs.processing_id
        if percent_bound:
            assert view[0].shape != spool[0].shape
        assert view[1].shape == spool[1].shape
        noop = spool.select(distance=(0, None), relative=True)
        columns = ["_data_size", "processing_id"]
        assert noop._df[columns].equals(spool._df[columns])

    def test_relative_metadata_tracks_each_row(self, tmp_path):
        """A shared relative window trims only the longer source patch."""
        base = dc.get_example_patch().abs()
        for size in (5, 20):
            patch = base.select(distance=(0, size), samples=True)
            patch.io.write(tmp_path / f"{size}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        view = spool.select(distance=(0, 10), relative=True)
        whole = spool._df["distance_max"] <= 10
        assert whole.any() and not whole.all()
        for column in ("_data_size", "processing_id"):
            assert view._df.loc[~whole, column].isna().all()
            assert (
                view._df.loc[whole, column].to_numpy()
                == spool._df.loc[whole, column].to_numpy()
            ).all()
        for original, selected in zip(spool, view, strict=True):
            direct = original.select(distance=(0, 10), relative=True)
            assert selected.attrs.processing_id == direct.attrs.processing_id
            assert np.array_equal(selected.data, direct.data)

    def test_processing_id_queries_source_metadata(self, tmp_path):
        """Attribute predicates identify the source, before residual processing."""
        patch = dc.get_example_patch().abs()
        patch.io.write(tmp_path / "source.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        source_id = spool[0].attrs.processing_id
        trimmed = spool.select(distance=(10, 20)).select(processing_id=source_id)
        assert len(trimmed) == 1
        assert trimmed.get_contents()["processing_id"].isnull().all()
        assert trimmed[0].attrs.processing_id != source_id
        assert np.array_equal(trimmed[0].data, spool[0].select(distance=(10, 20)).data)

    def test_a_trimmed_row_states_no_processing_id(self, tmp_path):
        """What the source had was undone by the trim, so the row states none.

        Stating it would let a provenance query compare the row against
        an id the patch it resolves to does not carry. `patch_id` is
        untouched: a trim does not change which data this is.
        """
        patch = dc.get_example_patch().pass_filter(time=(1, 10))
        patch.io.write(tmp_path / "filtered.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        time = spool[0].get_coord("time")
        view = spool.select(time=(time.min(), time.min() + 10 * time.step))
        contents = view.get_contents()
        assert contents["processing_id"].isnull().all()
        assert contents["patch_id"].iloc[0] == view[0].attrs.patch_id

    def test_a_row_a_view_leaves_whole_still_states_its_id(self, tmp_path):
        """A view which trims one patch has not trimmed the others."""
        patches = list(dc.get_example_spool("random_das"))
        for index, each in enumerate(patches):
            each.pass_filter(time=(1, 10)).io.write(tmp_path / f"{index}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        contents = spool.get_contents()
        span = contents["time_max"].max() - contents["time_min"].min()
        view = spool.select(
            time=(contents["time_min"].min() + span / 6, contents["time_max"].max())
        )
        stated = view.get_contents()["processing_id"]
        assert stated.notna().any() and stated.isnull().any()
        for index, value in stated.items():
            if pd.notna(value):
                assert view[index].attrs.processing_id == value

    def test_a_trim_keeps_the_id_it_says_it_keeps(self, written_spool):
        """The index and the patch which loads must not disagree."""
        trimmed = written_spool.select(time=(10, 20), samples=True)
        indexed = trimmed.get_contents()["patch_id"].iloc[0]
        assert trimmed[0].attrs.patch_id == indexed

    def test_a_memory_spool_too(self):
        """A summary carries the ids, so a patch never written is findable."""
        patches = list(dc.get_example_spool("random_das"))
        spool = dc.spool(patches)
        wanted = patches[1].attrs.patch_id
        assert spool.select(patch_id=wanted)[0].attrs.patch_id == wanted

    def test_an_id_no_patch_carries(self, written_spool):
        """An id which names nothing selects nothing, rather than raising."""
        assert len(written_spool.select(patch_id="0" * 16)) == 0

    def test_a_renamed_source_forgets_its_id(self, tmp_path):
        """
        A derived id names the path it came from.

        Renaming a file is how metadata is attached to an archive, and
        the index rewrites the path without re-reading the file. The id
        the row held is the id of where it used to be, so it is cleared
        rather than left to select a patch which no longer carries it.
        """
        with config_context(patch_provenance="disabled"):
            dc.get_example_patch().io.write(tmp_path / "x.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        stale = spool.get_contents()["patch_id"].iloc[0]
        assert stale
        (tmp_path / "x.h5").rename(tmp_path / "tag=renamed.h5")
        moved = dc.spool(tmp_path).update()
        assert moved.get_contents()["patch_id"].iloc[0] == ""
        assert len(moved.select(patch_id=stale)) == 0
        # The patch itself still says which data it is; only the index
        # stopped claiming to know without looking.
        assert moved[0].attrs.patch_id

    def test_a_path_may_not_claim_the_lineage(self, tmp_path):
        """
        A directory name says where data is kept, not which data it is.

        Hive-style path keys become ordinary attrs and override what a
        file states, which is how a rename corrects metadata. Letting one
        claim `patch_id` would rewrite the lineage of everything beneath
        it -- on the loaded patch, not merely in the index.
        """
        directory = tmp_path / "patch_id=bogus"
        directory.mkdir()
        with config_context(patch_provenance="disabled"):
            dc.get_example_patch().io.write(directory / "x.h5", "dasdae")
        with pytest.warns(UserWarning, match="not which data it is"):
            spool = dc.spool(tmp_path).update()
        assert spool.get_contents()["patch_id"].iloc[0] != "bogus"
        assert spool[0].attrs.patch_id != "bogus"

    def test_a_rename_when_no_id_was_indexed(self, tmp_path):
        """An archive indexed with the ids off has none to forget."""
        with config_context(patch_provenance="disabled"):
            dc.get_example_patch().io.write(tmp_path / "x.h5", "dasdae")
            spool = dc.spool(tmp_path).update()
            assert "patch_id" not in spool.get_contents().columns
            (tmp_path / "x.h5").rename(tmp_path / "tag=renamed.h5")
            moved = dc.spool(tmp_path).update()
        assert len(moved) == 1

    def test_chunk_still_merges_across_ids(self, written_spool):
        """Every patch states a different id; none of them blocks a merge."""
        assert len(set(written_spool.get_contents()["patch_id"])) == 3
        assert len(written_spool.chunk(time=None)) == 1


class TestCoordDefinitionDedup:
    """One coordinate is stored once, however its values were written."""

    def _definitions(self, spool, name):
        """The def keys stored for one coordinate name."""
        sql = (
            "SELECT cd.def_key FROM patch_coords pc "
            "JOIN coord_defs cd ON cd.coord_def_id = pc.coord_def_id "
            "WHERE pc.coord_name = ?"
        )
        frame = spool._catalog.backend._fetch_df(sql, [name])
        return frame["def_key"].tolist()

    def test_scalar_spelling_shares_one_definition(self):
        """A range written with an int start is the one written with a float."""
        base = dc.get_example_patch()
        n = base.shape[base.get_axis("distance")]
        as_ints = base.update_coords(distance=get_coord(start=0, stop=n, step=1.0))
        as_floats = base.update_coords(
            distance=get_coord(start=0.0, stop=float(n), step=1.0)
        )
        keys = self._definitions(dc.spool([as_ints, as_floats]), "distance")
        assert len(keys) == 2
        assert len(set(keys)) == 1

    def test_time_precision_shares_one_definition(self):
        """A step in milliseconds is the same step in nanoseconds."""
        t0 = np.datetime64("2020-01-01", "ns")
        data = np.random.default_rng(0).random((2, 100))
        coarse = get_coord(
            start=t0, stop=t0 + np.timedelta64(400, "ms"), step=np.timedelta64(4, "ms")
        )
        fine = get_coord(
            start=t0,
            stop=t0 + np.timedelta64(400_000_000, "ns"),
            step=np.timedelta64(4_000_000, "ns"),
        )
        distance = get_coord(values=np.array([0.0, 1.0]))
        patches = [
            dc.Patch(
                data=data,
                coords={"distance": distance, "time": x},
                dims=("distance", "time"),
            )
            for x in (coarse, fine)
        ]
        keys = self._definitions(dc.spool(patches), "time")
        assert len(keys) == 2
        assert len(set(keys)) == 1
