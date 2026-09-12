"""Bounded metadata access preserves spool selection and presentation."""

from __future__ import annotations

import re
from unittest import mock

import numpy as np
import pytest

import dascore as dc
from dascore.io.index import catalog as catalog_module
from dascore.units import m


@pytest.fixture()
def indexed_spool():
    """A small realized catalog whose patch data remains independently readable."""
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
    assert spool._catalog.backend is not None
    yield spool
    spool._catalog.close()


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
    spool = dc.spool(patches).select(distance=(10 * m, 12 * m))
    yield spool
    spool._catalog.close()


class TestBoundedMetadata:
    """Small requests do not project or return the whole catalog."""

    @pytest.mark.parametrize("index", [0, 5, -1, -12])
    @pytest.mark.parametrize("order", ["original", "sorted", "reordered"])
    def test_item_projects_one_row(self, indexed_spool, index, order):
        """Integer reads preserve ordering while decoding only their metadata row."""
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
        """An already realized relation still serves repeated access without SQL."""
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
        """Repeated positional reads pay for at most two metadata projections."""
        spool = indexed_spool.sort("time") if sorted_view else indexed_spool
        backend = spool._catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            for index in range(0, 12, stride):
                assert spool[index].attrs.tag == f"patch-{index:02d}"
        assert [len(call.args[0]) for call in pivot.call_args_list] == [1, 12]

    def test_items_after_revision(self, indexed_spool):
        """A new catalog revision starts with a bounded read again."""
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
        """A forward slice returns bounded membership without projecting metadata."""
        backend = indexed_spool._catalog.backend
        with mock.patch.object(backend, "_fetch_df", wraps=backend._fetch_df) as fetch:
            with mock.patch.object(
                backend, "_pivot_coords", wraps=backend._pivot_coords
            ) as pivot:
                selected = indexed_spool[7:9]
        queries = [
            call.args
            for call in fetch.call_args_list
            if call.args[0].startswith("SELECT p.patch_id ")
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
        catalog = indexed_spool._catalog
        expected = catalog.ordered_ids()[item]
        assert catalog.window(item).ordered_ids() == expected

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
        """The backend count API still applies regex residuals exactly."""
        catalog = indexed_spool.select(tag=re.compile("patch-0[579]"))._catalog
        assert catalog.backend.count(catalog._queries) == 3

    def test_regex_item_filters_once(self, indexed_spool):
        """Regex extraction must not realize separate count and membership frames."""
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
        """A count over known compatible units does not project matching rows."""
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

    def test_unitless_mask(self, unitless_selection):
        """A mask sized from a fresh view uses the same candidate membership."""
        selected = unitless_selection[np.ones(len(unitless_selection), dtype=bool)]
        assert [patch.attrs.tag for patch in selected] == ["10", "11"]

    def test_positive_index_needs_no_count(self, indexed_spool):
        """Nonnegative extraction detects bounds from its limited result."""
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
        """Composing ordinary absolute ranges keeps the first read bounded."""
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


class _Index:
    """A bound which is only usable through __index__, as Python allows."""

    def __init__(self, value: int):
        self._value = value

    def __index__(self) -> int:
        return self._value


def _counting_ids(backend):
    """Patch query_ids to record how many ids each call returned."""
    fetched = []
    original = backend.query_ids

    def counting(*args, **kwargs):
        out = original(*args, **kwargs)
        fetched.append(len(out))
        return out

    return fetched, counting


@pytest.fixture()
def big_spool():
    """A spool whose metadata lives in SQL, large enough to bound work against."""
    start = np.datetime64("2024-01-01", "ns")
    patches = [
        dc.Patch(
            data=np.arange(4),
            coords={"time": start + (10 * i + np.arange(4)) * np.timedelta64(1, "s")},
            dims=("time",),
            attrs={"tag": f"patch-{i:03d}"},
        )
        for i in range(200)
    ]
    spool = dc.spool(patches)
    assert spool._catalog.backend is not None
    yield spool
    spool._catalog.close()


# Every window shape, as (slice, how many ids a bounded path may fetch).
_WINDOWS = {
    "head": (slice(None, 10), 10),
    "mid": (slice(100, 110), 10),
    "tail": (slice(-10, None), 10),
    "tail_stop": (slice(-10, -3), 10),
    "stepped": (slice(None, 20, 2), 20),
    "stepped_late": (slice(150, 190, 3), 40),
    "reverse_tail": (slice(-1, -11, -1), 10),
    "mixed_signs": (slice(150, -5), 45),
}


class TestWindowMembership:
    """Every window keeps the membership Python's own slicing would give."""

    @pytest.mark.parametrize("name", list(_WINDOWS))
    def test_matches_python_slicing(self, big_spool, name):
        """A window presents exactly the ids the same slice of the ids gives."""
        item = _WINDOWS[name][0]
        catalog = big_spool._catalog
        expected = catalog.ordered_ids()[item]
        assert catalog.window(item).ordered_ids() == expected

    @pytest.mark.parametrize(
        "positions", [[3, 1, 2], [150, 151, 3], [0], [199, 0], [5, 5, 6]]
    )
    def test_array_matches_python(self, big_spool, positions):
        """An integer array keeps caller order and drops duplicates."""
        catalog = big_spool._catalog
        ids = catalog.ordered_ids()
        expected = tuple(dict.fromkeys(ids[x] for x in positions))
        assert catalog.restrict(np.array(positions)).ordered_ids() == expected


class TestWindowWorkIsBounded:
    """A small window fetches a bounded membership, not the whole relation."""

    @pytest.mark.parametrize("name", list(_WINDOWS))
    def test_ids_fetched_are_bounded(self, big_spool, name):
        """No window of ten rows reads two hundred ids to find them."""
        item, allowed = _WINDOWS[name]
        catalog = big_spool._catalog
        backend = catalog.backend
        fetched, counting = _counting_ids(backend)
        with mock.patch.object(backend, "query_ids", counting):
            catalog.window(item).ordered_ids()
        assert fetched and max(fetched) <= allowed

    @pytest.mark.parametrize("positions", [[3, 1, 2], [150, 151, 3]])
    def test_small_array_is_bounded(self, big_spool, positions):
        """A short array of positions reads only the span it covers."""
        catalog = big_spool._catalog
        backend = catalog.backend
        fetched, counting = _counting_ids(backend)
        with mock.patch.object(backend, "query_ids", counting):
            catalog.restrict(np.array(positions)).ordered_ids()
        span = max(positions) - min(positions) + 1
        assert fetched and max(fetched) <= span

    def test_last_item_does_not_walk_the_root(self, big_spool):
        """Reading the final patch seeks from the end, not past every row."""
        catalog = big_spool._catalog
        backend = catalog.backend
        calls = []
        original = backend.query_ids

        def recording(*args, **kwargs):
            calls.append(kwargs.get("offset", 0))
            return original(*args, **kwargs)

        with mock.patch.object(backend, "query_ids", recording):
            assert catalog.get_patch(-1).attrs.tag == "patch-199"
        assert calls and max(calls) <= 10

    def test_repeated_reads_stay_bounded(self, big_spool):
        """Ten reads project pages, never the whole relation."""
        catalog = big_spool._catalog
        backend = catalog.backend
        with mock.patch.object(
            backend, "_pivot_coords", wraps=backend._pivot_coords
        ) as pivot:
            for index in range(10):
                assert catalog.get_patch(index).attrs.tag == f"patch-{index:03d}"
        projected = [len(call.args[0]) for call in pivot.call_args_list]
        assert sum(projected) < 200


def _tied_patches(prefix):
    """Patches which all start at one time, so the tiebreaks decide order."""
    start = np.datetime64("2024-01-01", "ns")
    return [
        dc.Patch(
            data=np.arange(4),
            coords={"time": start + np.arange(4) * np.timedelta64(1, "s")},
            dims=("time",),
            attrs={"tag": f"{prefix}-{i:03d}"},
        )
        for i in range(200)
    ]


@pytest.fixture()
def tied_spool():
    """Tied patches under the catalog's own order."""
    spool = dc.spool(_tied_patches("tie"))
    assert spool._catalog.backend is not None
    yield spool
    spool._catalog.close()


@pytest.fixture()
def sorted_tied_spool():
    """Tied patches under an explicit sort, where the sort breaks the ties."""
    spool = dc.spool(_tied_patches("sorted")).sort("time")
    assert spool._catalog.backend is not None
    yield spool
    spool._catalog.close()


class TestTiedOrder:
    """A window of tied rows presents them in the same order either way."""

    @pytest.mark.parametrize("name", list(_WINDOWS))
    def test_window_matches_python(self, tied_spool, name):
        """Seeking from the end returns the order the forward read would."""
        item = _WINDOWS[name][0]
        catalog = tied_spool._catalog
        assert catalog.window(item).ordered_ids() == catalog.ordered_ids()[item]

    # A reversed window is left out: a view carrying an explicit order
    # re-sorts a fixed membership, so its rows come back ascending. That
    # predates this work (`dev` does the same) and is not changed here.
    @pytest.mark.parametrize(
        "name", [x for x in _WINDOWS if not x.startswith("reverse")]
    )
    def test_sorted_window_matches_python(self, sorted_tied_spool, name):
        """An explicit sort over tied rows windows the same either way."""
        item = _WINDOWS[name][0]
        catalog = sorted_tied_spool._catalog
        assert catalog.window(item).ordered_ids() == catalog.ordered_ids()[item]

    def test_last_tied_patch(self, tied_spool):
        """The final row of a tied order is the one a forward read ends on."""
        assert tied_spool[-1].attrs.tag == tied_spool.get_contents()["tag"].iloc[-1]

    def test_last_sorted_tied_patch(self, sorted_tied_spool):
        """The same holds when an explicit sort breaks the ties."""
        catalog = sorted_tied_spool._catalog
        expected = catalog.to_df()["tag"].iloc[-1]
        # a view of its own, so the read pages rather than reusing that frame
        fresh = catalog._view(catalog._queries, catalog._residuals)
        assert fresh.get_patch(-1).attrs.tag == expected


class TestWindowFallbacks:
    """What a bounded seek cannot serve still comes back correct."""

    def test_wide_span_reads_membership(self, big_spool, monkeypatch):
        """A window spanning more rows than the seek limit still matches."""
        monkeypatch.setattr(catalog_module, "_WINDOW_SPAN", 4)
        catalog = big_spool._catalog
        # a negative bound, so the span is known and found too wide to seek
        item = slice(-190, None, 3)
        assert catalog.window(item).ordered_ids() == catalog.ordered_ids()[item]

    def test_wide_array_span_reads_membership(self, big_spool, monkeypatch):
        """Positions too far apart to seek between keep their caller order."""
        monkeypatch.setattr(catalog_module, "_WINDOW_SPAN", 4)
        catalog = big_spool._catalog
        ids = catalog.ordered_ids()
        assert catalog.restrict(np.array([150, 0])).ordered_ids() == (
            ids[150],
            ids[0],
        )

    def test_out_of_range_position_raises(self, big_spool):
        """A position past the end raises, as indexing the ids would."""
        with pytest.raises(IndexError):
            big_spool._catalog.restrict(np.array([500]))

    def test_array_over_fixed_membership(self, big_spool):
        """An array over an already-fixed membership picks from what it has."""
        view = big_spool._catalog.window(slice(0, 10))
        ids = view.ordered_ids()
        assert view.restrict(np.array([2, 0])).ordered_ids() == (ids[2], ids[0])

    def test_unsigned_bounds_match_python(self, big_spool):
        """Bounds which are unsigned integers cannot underflow into a window."""
        catalog = big_spool._catalog
        ids = catalog.ordered_ids()
        item = slice(np.uint64(5), np.uint64(2))
        assert catalog.window(item).ordered_ids() == ids[5:2] == ()

    @pytest.mark.parametrize(
        "bounds", [(np.int64(2), np.int64(5)), (_Index(2), _Index(5))]
    )
    def test_index_like_bounds(self, big_spool, bounds):
        """Anything Python accepts as an index works as a bound."""
        catalog = big_spool._catalog
        start, stop = bounds
        assert (
            catalog.window(slice(start, stop)).ordered_ids()
            == (catalog.ordered_ids()[2:5])
        )

    def test_wide_contiguous_window_stays_bounded(self, big_spool, monkeypatch):
        """A contiguous window costs what it returns, however wide it is."""
        monkeypatch.setattr(catalog_module, "_WINDOW_SPAN", 4)
        catalog = big_spool._catalog
        backend = catalog.backend
        fetched, counting = _counting_ids(backend)
        item = slice(-60, None)
        with mock.patch.object(backend, "query_ids", counting):
            chosen = catalog.window(item).ordered_ids()
        assert chosen == catalog.ordered_ids()[item]
        assert fetched and max(fetched) <= 60

    def test_repeated_last_reads_count_once(self, big_spool):
        """Reading the final patch again does not count the view again."""
        catalog = big_spool.select(tag="patch-1[0-9][0-9]")._catalog
        backend = catalog.backend
        with mock.patch.object(backend, "count", wraps=backend.count) as count:
            for _ in range(10):
                assert catalog.get_patch(-1).attrs.tag == "patch-199"
        assert count.call_count <= 1

    def test_pages_are_evicted(self, big_spool, monkeypatch):
        """Reads which walk away from their page do not accumulate pages."""
        monkeypatch.setattr(catalog_module, "_MAX_PAGES", 1)
        catalog = big_spool._catalog
        assert catalog.get_patch(0).attrs.tag == "patch-000"
        for index in (10, 70, 140):
            assert catalog.get_patch(index).attrs.tag == f"patch-{index:03d}"
        assert len(catalog._pages) <= 1


class TestSplitMembership:
    """Splitting fetches membership once and preserves each view's state."""

    @pytest.mark.parametrize("options", [{"size": 2}, {"count": 6}])
    def test_one_membership_fetch(self, indexed_spool, options):
        """The number of full membership queries does not grow with batches."""
        backend = indexed_spool._catalog.backend
        with mock.patch.object(backend, "query_ids", wraps=backend.query_ids) as query:
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
        """Splitting an empty in-memory spool does not allocate an index."""
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
