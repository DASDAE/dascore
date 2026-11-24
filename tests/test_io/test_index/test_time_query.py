"""Time envelope queries agree with the general coordinate query engine."""

from __future__ import annotations

import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.exceptions import InvalidSpoolQueryError, TimeOverflowError
from dascore.io.index.backend import get_backend
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.query import Query, build_sql


@pytest.fixture()
def time_backend(tmp_path):
    """Identical time/reference coordinates across heterogeneous patch types."""
    start = np.datetime64("2024-01-01", "ns")
    offsets = np.arange(4) * np.timedelta64(1, "s")
    coordinates = [
        start + offsets,
        start + np.timedelta64(10, "s") + offsets,
        offsets,
        np.arange(4.0),
        np.array(["a", "b", "c", "d"]),
    ]
    patches = [
        dc.Patch(
            data=np.arange(4),
            coords={"time": values, "reference_time": ("time", values)},
            dims=("time",),
            attrs={"tag": f"patch-{index}"},
        )
        for index, values in enumerate(coordinates)
    ]
    patches.append(
        dc.Patch(
            data=np.arange(4), coords={"distance": np.arange(4)}, dims=("distance",)
        )
    )
    catalog = PatchCatalog.from_patches(patches)
    path = tmp_path / "index.sqlite3"
    backend = get_backend(path)
    backend.write_sources(catalog.backend.export_records())
    catalog.close()
    yield backend
    backend.close()


class TestTimeQuery:
    """Only eligible absolute-time bounds use the existing hot columns."""

    @pytest.mark.parametrize(
        "bounds",
        [
            (np.datetime64("2024-01-01"), np.datetime64("2024-01-01")),
            (
                np.datetime64("2024-01-01T00:00:02"),
                np.datetime64("2024-01-01T00:00:12"),
            ),
            (None, np.datetime64("2024-01-01T00:00:12")),
            (np.datetime64("2024-01-01T00:00:12"), None),
            (Ellipsis, np.datetime64("2023-01-01")),
            (np.datetime64("2025-01-01"), Ellipsis),
            (np.timedelta64(1, "s"), np.timedelta64(2, "s")),
            (1, 2),
            ("b", "c"),
        ],
    )
    def test_matches_general_query(self, time_backend, bounds):
        """Projection, membership, and counts agree for all coordinate kinds."""
        direct = Query(coords={"time": bounds})
        general = Query(coords={"reference_time": bounds})
        pd.testing.assert_frame_equal(
            time_backend.query(direct), time_backend.query(general)
        )
        assert time_backend.query_rows(direct) == time_backend.query_rows(general)
        assert time_backend.count(direct) == time_backend.count(general)

    def test_sorted_membership(self, time_backend):
        """Fast time predicates compose with attribute filters and fixed membership."""
        bounds = (np.datetime64("2024-01-01"), None)
        direct = [Query(coords={"time": bounds}), Query(attrs={"tag": "patch-*"})]
        general = [
            Query(coords={"reference_time": bounds}),
            Query(attrs={"tag": "patch-*"}),
        ]
        options = {"order_by": ("coord", "time", False), "patch_rows": (1, 2, 3)}
        assert time_backend.query_rows(direct, **options) == time_backend.query_rows(
            general, **options
        )

    def test_uses_stored_time_bounds(self, time_backend):
        """An absolute-time count avoids visiting coordinate attachment tables."""
        query = Query(coords={"time": (np.datetime64("2024-01-01T00:00:10"), None)})
        queries, attrs, coords = time_backend._query_context(query)
        sql, params, residuals = build_sql(queries, attrs, coords, count=True)
        assert not residuals
        assert "patch_coords" not in sql
        plan = time_backend._con.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()
        assert not any(
            "patch_coords" in row[-1] or "coord_defs" in row[-1] for row in plan
        )
        assert time_backend.count(query) == 1

    def test_existing_catalog(self, time_backend):
        """Old catalogs remain valid without rebuilding sources or adding schema."""
        query = Query(coords={"time": (np.datetime64("2024-01-01"), None)})
        expected = time_backend.query_rows(query)
        reopened = get_backend(time_backend._path)
        try:
            assert reopened.query_rows(query) == expected
        finally:
            reopened.close()

    def test_mixed_time_sort(self):
        """Relative times retain their position after absolute-time patches."""
        origin = np.datetime64("2024-01-01", "ns")
        offsets = np.arange(4) * np.timedelta64(1, "s")
        coordinates = [
            offsets,
            origin + offsets + np.timedelta64(10, "s"),
            origin + offsets,
        ]
        patches = [
            dc.Patch(
                data=np.arange(4),
                coords={"time": values},
                dims=("time",),
                attrs={"tag": tag},
            )
            for values, tag in zip(coordinates, ["relative", "later", "earlier"])
        ]
        spool = dc.spool(patches)
        assert list(spool.sort("time").get_contents()["tag"]) == [
            "earlier",
            "later",
            "relative",
        ]
        spool._catalog.close()

    def test_date_strings(self):
        """Date strings on ordinary datetime catalogs retain coercion and trimming."""
        start = np.datetime64("2024-01-01", "ns")
        patch = dc.Patch(
            data=np.arange(4),
            coords={"time": start + np.arange(4) * np.timedelta64(1, "s")},
            dims=("time",),
        )
        spool = dc.spool([patch])
        selected = spool.select(time=("2024-01-01T00:00:01", "2024-01-01T00:00:02"))
        assert len(selected) == 1
        assert np.array_equal(selected[0].data, [1, 2])
        spool._catalog.close()

    def test_unrepresentable_date_string_raises(self):
        """A bound no nanosecond time can hold raises rather than matching nothing."""
        spool = dc.get_example_spool("random_das")
        with pytest.raises(TimeOverflowError):
            len(spool.select(time=(None, "3000-01-01")))


class TestNumbersOnTimes:
    """A bare number on a time coordinate reads as `Patch.select` reads it."""

    @pytest.fixture(params=["memory", "directory"])
    def make_spool(self, request, tmp_path):
        """Build a memory spool, or a directory spool of the patches' files."""

        def _make(patches):
            if request.param == "memory":
                return dc.spool(patches)
            for index, patch in enumerate(patches):
                patch.io.write(tmp_path / f"patch_{index}.h5", "DASDAE")
            return dc.spool(tmp_path).update()

        return _make

    @staticmethod
    def _assert_selects_as_patches(spool, patches, bounds):
        """Each patch loads, and its row states, what its own select keeps."""
        selected = spool.select(time=bounds)
        want = {x.get_coord("time").dtype.kind: x.select(time=bounds) for x in patches}
        contents = selected.get_contents()
        assert len(selected) == len(want)
        for patch, low, high in zip(
            selected, contents["time_min"], contents["time_max"], strict=True
        ):
            coord = patch.get_coord("time")
            assert coord == want[coord.dtype.kind].get_coord("time")
            # a float end is stated within its rounding
            stated, loaded = (low, high), (coord.min(), coord.max())
            if coord.dtype.kind == "f":
                assert np.allclose(stated, loaded, rtol=0, atol=1e-12)
            else:
                assert stated == loaded

    def test_seconds_since_1970(self, make_spool):
        """A number on absolute times is seconds since 1970."""
        patch = dc.get_example_patch()
        start = patch.get_coord("time").min()
        seconds = (start - np.datetime64(0, "s")) / np.timedelta64(1, "s")
        spool = make_spool([patch])
        for bounds in [(seconds + 1, seconds + 2), (seconds + 1, None)]:
            self._assert_selects_as_patches(spool, [patch], bounds)

    def test_reversed_seconds_refused(self, make_spool):
        """Reversed numbers are refused on absolute times, as on numbers."""
        patch = dc.get_example_patch()
        start = patch.get_coord("time").min()
        seconds = (start - np.datetime64(0, "s")) / np.timedelta64(1, "s")
        with pytest.raises(InvalidSpoolQueryError, match="lo > hi"):
            len(make_spool([patch]).select(time=(seconds + 2, seconds + 1)))

    def test_numbers_past_nanoseconds(self, make_spool):
        """A number no nanosecond time holds still selects among numbers."""
        patch = dc.get_example_patch()
        steps = np.arange(patch.get_coord("time").shape[0])
        durations = patch.update_coords(time=steps * np.timedelta64(4, "ms"))
        numbers = patch.update_coords(time=steps * 0.004)
        tagged = [durations.update_attrs(tag="d"), numbers.update_attrs(tag="n")]
        spool = make_spool(tagged)
        for big in (1e12, 1e308):
            assert len(spool.select(tag="n", time=(None, big))) == 1
            assert len(spool.select(tag="n", time=(-big, 1))) == 1
            assert len(spool.select(time=(big, None))) == 0

    def test_durations_beside_numbers(self, make_spool):
        """Each patch reads the number in its own kind: seconds on durations."""
        patch = dc.get_example_patch()
        steps = np.arange(patch.get_coord("time").shape[0])
        durations = patch.update_coords(time=steps * np.timedelta64(4, "ms"))
        numbers = patch.update_coords(time=steps * 0.004)
        spool = make_spool([durations, numbers])
        for bounds in [(1.0, 2.0), (None, 0.5), (-np.inf, 1)]:
            self._assert_selects_as_patches(spool, [durations, numbers], bounds)


class TestPlannedTimeQueries:
    """Planned and rebuilt records keep the same absolute-time cache contract."""

    @pytest.mark.parametrize("operation", ["chunk", "concatenate"])
    @pytest.mark.parametrize("serialized", [False, True])
    def test_auxiliary_time(self, operation, serialized):
        """An auxiliary time envelope survives planning on its distance dimension."""
        start = np.datetime64("2024-01-01", "ns")
        patches = [
            dc.Patch(
                data=np.arange(offset, offset + 4),
                coords={
                    "distance": np.arange(offset, offset + 4),
                    "time": (
                        "distance",
                        start + np.arange(offset, offset + 4) * np.timedelta64(1, "s"),
                    ),
                },
                dims=("distance",),
            )
            for offset in (0, 4)
        ]
        spool = getattr(dc.spool(patches), operation)(distance=None, conflict="drop")
        if serialized:
            spool = pickle.loads(pickle.dumps(spool))
        selected = spool.select(time=(start + np.timedelta64(6, "s"), None))
        assert len(selected) == 1
        assert np.array_equal(selected[0].data, [6, 7])

    @pytest.mark.parametrize("duration", [False, True])
    @pytest.mark.parametrize("serialized", [False, True])
    def test_non_absolute_plan(self, duration, serialized):
        """Numeric and duration plans never become absolute-time candidates."""
        values = np.arange(4)
        if duration:
            values = values * np.timedelta64(1, "s")
        patch = dc.Patch(data=np.arange(4), coords={"time": values}, dims=("time",))
        spool = dc.spool([patch]).chunk(time=None)
        if serialized:
            spool = pickle.loads(pickle.dumps(spool))
        assert len(spool.select(time=(None, np.datetime64("2024-01-01")))) == 0

    @pytest.mark.parametrize("duration", [False, True])
    @pytest.mark.parametrize("serialized", [False, True])
    def test_non_absolute_sort(self, duration, serialized):
        """Clearing absolute-time caches preserves numeric and duration ordering."""
        patches = []
        for offset in (10, 0):
            values = np.arange(offset, offset + 4)
            if duration:
                values = values * np.timedelta64(1, "s")
            patches.append(
                dc.Patch(data=np.arange(4), coords={"time": values}, dims=("time",))
            )
        spool = dc.spool(patches).concatenate(time=1)
        if serialized:
            spool = pickle.loads(pickle.dumps(spool))
        ordered = spool.sort("time")
        expected = [patch.coords.get_coord("time").min() for patch in patches]
        assert list(ordered.get_contents()["time_min"]) == sorted(expected)

    def test_rebuilt_record_bounds(self, time_backend):
        """Re-ingesting legacy records derives cached bounds from their coordinates."""
        records = time_backend.export_records()
        records = [
            replace(
                record,
                patches=tuple(
                    replace(patch, time_min=0, time_max=3, time_step=1)
                    for patch in record.patches
                ),
            )
            for record in records
        ]
        time_backend.write_sources(records)
        bounds = (None, np.datetime64("2025-01-01"))
        assert (
            time_backend.count(Query(coords={"time": bounds}))
            == time_backend.count(Query(coords={"reference_time": bounds}))
            == 2
        )
