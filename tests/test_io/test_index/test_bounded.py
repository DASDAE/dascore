"""Bounded metadata access preserves spool selection and presentation."""

from __future__ import annotations

import re
from unittest import mock

import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import get_coord
from dascore.units import m


@pytest.fixture()
def indexed_spool():
    """A small catalog in SQL, its frame unread, whose patches stay in memory."""
    start = np.datetime64("2024-01-01", "ns")
    patches = [
        dc.Patch(
            data=np.arange(4),
            coords={"time": start + (10 * i + np.arange(4)) * np.timedelta64(1, "s")},
            dims=("time",),
            attrs={"tag": f"patch-{i:02d}"},
        )
        for i in range(12)
    ]
    spool = dc.spool(patches)
    spool.select(tag="*")  # a selection puts the catalog in SQL, caching nothing
    return spool


@pytest.fixture()
def unitless_selection():
    """A quantity range whose first SQL candidate is removed by exact filtering."""
    patches = [
        dc.Patch(
            data=np.arange(4),
            coords={"distance": np.arange(start, start + 4)},
            dims=("distance",),
            attrs={"tag": str(start)},
        ).set_units(distance=None)
        for start in (0, 10, 11)
    ]
    return dc.spool(patches).select(distance=(10 * m, 12 * m))


class TestBoundedMetadata:
    """Small requests do not project or return the whole catalog."""

    @pytest.mark.parametrize("index", [0, 5, -1, -12])
    @pytest.mark.parametrize("order", ["original", "sorted", "reordered"])
    def test_item_projects_one_row(self, indexed_spool, index, order):
        """
        Integer reads preserve ordering while decoding only their metadata row.

        A performance guard that keeps its spy: projecting every row gives the
        same patch, only slowly, so the rows projected are its only witness.
        """
        spool = indexed_spool
        if order == "sorted":
            spool = spool.sort("time")
        elif order == "reordered":
            spool = spool[list(reversed(range(12)))]
        catalog = spool._catalog
        expected = catalog.to_df().iloc[index]
        fresh = catalog._view(catalog._queries, catalog._residuals)
        backend = fresh.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            patch = fresh.get_patch(index)
        assert patch.attrs.tag == expected["tag"]
        assert [len(call.args[0]) for call in pivot.call_args_list] == [1]
        assert fresh._df_cache.get(fresh._revision.value) is None

    @pytest.mark.parametrize("index", [12, -13])
    def test_item_out_of_bounds(self, indexed_spool, index):
        """Uncached indexed spools reject either end of the valid positions."""
        with pytest.raises(IndexError, match="out of bounds"):
            indexed_spool[index]

    def test_cached_item_reuses_frame(self, indexed_spool):
        """
        An already realized relation still serves repeated access without SQL.

        A performance guard that keeps its spy: querying again gives the same
        patch, only slowly, so the call count is its only witness.
        """
        catalog = indexed_spool._catalog
        catalog.to_df()
        with mock.patch.object(
            catalog.backend, "query", wraps=catalog.backend.query
        ) as query:
            assert indexed_spool[0].attrs.tag == "patch-00"
        query.assert_not_called()

    @pytest.mark.parametrize("sorted_view", [False, True])
    @pytest.mark.parametrize("stride", [1, 3])
    def test_repeated_items_reuse_frame(self, indexed_spool, sorted_view, stride):
        """
        Repeated positional reads pay for at most two metadata projections.

        A performance guard that keeps its spy: projecting again for each read
        gives the same patches, only slowly, so the rows projected are its only
        witness.
        """
        spool = indexed_spool.sort("time") if sorted_view else indexed_spool
        backend = spool._catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            for index in range(0, 12, stride):
                assert spool[index].attrs.tag == f"patch-{index:02d}"
        assert [len(call.args[0]) for call in pivot.call_args_list] == [1, 12]

    def test_items_after_revision(self, indexed_spool):
        """
        A new catalog revision starts with a bounded read again.

        A performance guard that keeps its spy: projecting every row gives the
        same patches, only slowly, so the rows projected are its only witness.
        """
        catalog = indexed_spool._catalog
        backend = catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            assert indexed_spool[0].attrs.tag == "patch-00"
            assert indexed_spool[1].attrs.tag == "patch-01"
            catalog._invalidate()
            assert indexed_spool[2].attrs.tag == "patch-02"
        assert [len(call.args[0]) for call in pivot.call_args_list] == [1, 12, 1]

    def test_slice_returns_only_requested_ids(self, indexed_spool):
        """
        A forward slice returns bounded membership without projecting metadata.

        A performance guard that keeps its spies: fetching every row and slicing
        in Python gives the same membership, only slowly, so the SQL sent is its
        only witness.
        """
        backend = indexed_spool._catalog.backend
        with mock.patch.object(backend, "_fetch_df", wraps=backend._fetch_df) as fetch:
            with mock.patch.object(
                backend, "_pivot_coords", wraps=backend._pivot_coords
            ) as pivot:
                selected = indexed_spool[7:9]
        queries = [
            call.args
            for call in fetch.call_args_list
            if call.args[0].startswith("SELECT p.patch_row ")
        ]
        assert len(queries) == 1
        assert "LIMIT ? OFFSET ?" in queries[0][0]
        assert queries[0][1][-2:] == [2, 7]
        pivot.assert_not_called()
        assert selected.get_contents()["tag"].tolist() == ["patch-07", "patch-08"]

    @pytest.mark.parametrize(
        "item",
        [
            slice(None, 2),
            slice(7, None),
            slice(9, 2),
            slice(-3, None),
            slice(None, None, -1),
            slice(1, 10, 2),
            slice(None, 10**30),
            slice(10**30, None),
            slice(10**30, 10**31),
        ],
    )
    def test_slice_semantics(self, indexed_spool, item):
        """Open, negative, reversed, strided, and empty slices retain membership."""
        tags = [f"patch-{i:02d}" for i in range(12)]
        assert [x.attrs.tag for x in indexed_spool[item]] == tags[item]

    @pytest.mark.parametrize(
        "item", [slice(1.5, 3), slice(None, 2.5), slice(None, None, 1.5)]
    )
    def test_non_integer_slice(self, indexed_spool, item):
        """SQL slicing retains Python's rejection of non-integer bounds."""
        with pytest.raises(TypeError):
            indexed_spool[item]

    def test_zero_step(self, indexed_spool):
        """A zero stride retains Python's slice error."""
        with pytest.raises(ValueError, match="slice step cannot be zero"):
            indexed_spool[::0]

    def test_regex_limit_after_filter(self, indexed_spool):
        """Offset and limit count exact regex matches, not SQL candidates."""
        spool = indexed_spool.select(tag=re.compile("patch-0[579]"))
        assert spool[1:2].get_contents()["tag"].tolist() == ["patch-07"]
        assert spool[1].attrs.tag == "patch-07"

    @pytest.mark.parametrize("item", [slice(20, 30), slice(0, 0), []])
    def test_empty_regex_window(self, indexed_spool, item):
        """Empty regex-filtered views retain the columns needed for presentation."""
        spool = indexed_spool.select(tag=re.compile("patch-0[579]"))
        selected = spool[item]
        assert selected.get_contents().empty
        assert list(selected) == []

    def test_backend_regex_count(self, indexed_spool):
        """The count still applies regex residuals exactly."""
        assert len(indexed_spool.select(tag=re.compile("patch-0[579]"))) == 3

    def test_regex_item_filters_once(self, indexed_spool):
        """
        Regex extraction must not realize separate count and membership frames.

        A performance guard that keeps its spy: projecting twice gives the same
        patch, only slowly, so the call count is its only witness.
        """
        spool = indexed_spool.select(tag=re.compile("patch-0[579]"))
        backend = spool._catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            assert spool[1].attrs.tag == "patch-07"
        assert pivot.call_count == 1

    def test_selection_after_reordering(self, indexed_spool):
        """Later predicates filter a fixed membership without restoring SQL order."""
        spool = indexed_spool[[9, 3, 7, 1]].select(tag="patch-0[379]")
        assert spool[1:2][0].attrs.tag == "patch-03"
        assert spool[1].attrs.tag == "patch-03"

    def test_unitless_quantity_filter(self, unitless_selection):
        """Exact positional filtering can refine the existing SQL candidate count."""
        assert len(unitless_selection) == 3
        assert unitless_selection[0].attrs.tag == "10"
        assert unitless_selection[1].attrs.tag == "11"
        assert len(unitless_selection) == 2

    def test_unitful_count_stays_in_sql(self):
        """
        A count over known compatible units does not project matching rows.

        A performance guard that keeps its spy: projecting the matching rows
        gives the same count, only slowly, so the call count is its only
        witness.
        """
        patches = [
            dc.Patch(
                data=np.arange(4), coords={"distance": np.arange(4)}, dims=("distance",)
            ).set_units(distance="m")
            for _ in range(12)
        ]
        spool = dc.spool(patches).select(distance=(1 * m, 2 * m))
        backend = spool._catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            assert len(spool) == 12
        pivot.assert_not_called()
        spool._catalog.close()

    def test_mixed_unit_count_stays_in_sql(self):
        """
        Steps in other units do not make a wide range look narrow (#1703).

        A performance guard that keeps its spy: projecting the matching rows
        gives the same count, only slowly, so the call count is its only
        witness.
        """
        coords = [
            get_coord(start=0, step=1, shape=(400,), units="m"),
            get_coord(start=0, step=1000, shape=(400,), units="mm"),
        ]
        patches = [
            dc.Patch(data=np.zeros(400), coords={"distance": x}, dims=("distance",))
            for x in coords
        ]
        spool = dc.spool(patches).select(distance=(100 * m, 200 * m))
        backend = spool._catalog.backend
        with mock.patch.object(backend, "_pivot_coords") as pivot:
            assert len(spool) == 2
        pivot.assert_not_called()
        spool._catalog.close()

    def test_unitless_mask(self, unitless_selection):
        """A mask sized from a fresh view uses the same candidate membership."""
        selected = unitless_selection[np.ones(len(unitless_selection), dtype=bool)]
        assert [patch.attrs.tag for patch in selected] == ["10", "11"]

    def test_positive_index_needs_no_count(self, indexed_spool):
        """
        Nonnegative extraction detects bounds from its limited result.

        A performance guard that keeps its spy: counting first gives the same
        patch, only slowly, so the call count is its only witness.
        """
        backend = indexed_spool._catalog.backend
        with mock.patch.object(backend, "count", wraps=backend.count) as count:
            assert indexed_spool[0].attrs.tag == "patch-00"
            assert indexed_spool[0].attrs.tag == "patch-00"
        count.assert_not_called()

    def test_trimmed_item(self, indexed_spool):
        """Bounded extraction still applies exact coordinate trimming."""
        start = np.datetime64("2024-01-01T00:00:11", "ns")
        patch = indexed_spool.select(time=(start, start + np.timedelta64(1, "s")))[0]
        assert patch.get_coord("time").min() == start
        assert patch.shape == (2,)

    def test_chained_absolute_bounds(self, indexed_spool):
        """
        Composing ordinary absolute ranges keeps the first read bounded.

        A performance guard that keeps its spy: projecting every row gives the
        same patch, only slowly, so the rows projected are its only witness.
        """
        start = np.datetime64("2024-01-01T00:00:11", "ns")
        spool = indexed_spool.select(time=(start, None)).select(
            time=(None, start + np.timedelta64(40, "s"))
        )
        backend = spool._catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            assert spool[0].attrs.tag == "patch-01"
        assert [len(call.args[0]) for call in pivot.call_args_list] == [1]

    def test_filter_after_local_trim(self, indexed_spool):
        """Rows removed by residual composition cannot shift integer extraction."""
        start = np.datetime64("2024-01-01T00:00:02", "ns")
        spool = indexed_spool.select(time=(0, 2), samples=True).select(
            time=(start, None)
        )
        assert spool[0].attrs.tag == "patch-01"
        assert len(spool) == 11


class TestSplitMembership:
    """Splitting fetches membership once and preserves each view's state."""

    @pytest.mark.parametrize("options", [{"size": 2}, {"count": 6}])
    def test_one_membership_fetch(self, indexed_spool, options):
        """
        The number of full membership queries does not grow with batches.

        A performance guard that keeps its spy: a query per batch gives the same
        parts, only slowly, so the call count is its only witness.
        """
        backend = indexed_spool._catalog.backend
        with mock.patch.object(
            backend, "query_rows", wraps=backend.query_rows
        ) as query:
            parts = list(indexed_spool.split(**options))
        assert query.call_count == 1
        assert [len(part) for part in parts] == [2] * 6
        assert [patch.attrs.tag for part in parts for patch in part] == [
            f"patch-{i:02d}" for i in range(12)
        ]

    def test_unitless_candidates(self, unitless_selection):
        """Splitting a fresh quantity selection cannot discard a later match."""
        parts = list(unitless_selection.split(size=1))
        assert [patch.attrs.tag for part in parts for patch in part] == ["10", "11"]

    @pytest.mark.parametrize("options", [{"size": 1}, {"count": 12}])
    def test_filtered_candidates(self, indexed_spool, options):
        """A refined count cannot truncate the candidate membership being split."""
        start = np.datetime64("2024-01-01T00:00:02", "ns")
        spool = indexed_spool.select(time=(0, 2), samples=True).select(
            time=(start, None)
        )
        parts = list(spool.split(**options))
        assert [patch.attrs.tag for part in parts for patch in part] == [
            f"patch-{index:02d}" for index in range(1, 12)
        ]

    def test_empty_cold_catalog(self):
        """
        Splitting an empty in-memory spool does not allocate an index.

        A performance guard (rule C): an index built for nothing changes no
        batch, so whether the catalog built one is asked of it.
        """
        spool = dc.spool([])
        assert list(spool.split(size=1)) == []
        assert spool._catalog._backend is None

    def test_reordered_selection(self, indexed_spool):
        """Batching preserves explicit order, predicates, and coordinate residuals."""
        spool = (
            indexed_spool[[9, 3, 7, 1]]
            .select(tag="patch-0[379]")
            .select(time=(0, 2), samples=True)
        )
        parts = list(spool.split(size=2))
        patches = [patch for part in parts for patch in part]
        assert [patch.attrs.tag for patch in patches] == [
            "patch-09",
            "patch-03",
            "patch-07",
        ]
        assert all(patch.shape == (2,) for patch in patches)

    @pytest.mark.parametrize("options", [{"size": 2}, {"count": 6}])
    def test_empty(self, indexed_spool, options):
        """An empty selection yields no batches for either splitting mode."""
        assert list(indexed_spool.select(tag="absent").split(**options)) == []
