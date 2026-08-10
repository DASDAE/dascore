"""Tests for PatchCatalog: the unified spool metadata engine."""

from __future__ import annotations

import pickle
import re

import numpy as np
import pytest

import dascore as dc
from dascore.io.index import PatchCatalog
from dascore.io.index.query import InvalidSpoolQueryError
from dascore.utils.misc import _canonical_range, _CanonicalRange


@pytest.fixture(scope="class")
def patches():
    """Patches from the random example spool."""
    return tuple(dc.get_example_spool("random_das"))


@pytest.fixture()
def live_catalog(patches):
    """A catalog over live patches."""
    return PatchCatalog.from_patches(patches)


class TestLaziness:
    """Catalog construction does no metadata work."""

    def test_no_backend_until_needed(self, patches):
        """from_patches must not bootstrap a backend."""
        catalog = PatchCatalog.from_patches(patches)
        assert catalog._backend is None

    def test_len_serves_from_registry(self, live_catalog, patches):
        """Len (and patch access) never bootstrap a backend."""
        assert len(live_catalog) == len(patches)
        assert live_catalog.get_patch(0) is patches[0]
        assert live_catalog._backend is None

    def test_first_relation_op_bootstraps(self, live_catalog, patches):
        """Realizing the flat relation creates the backend and ingests."""
        df = live_catalog.to_df()
        assert len(df) == len(patches)
        assert live_catalog._backend is not None


class TestLiveRoundtrip:
    """Live patches come back identical."""

    def test_iteration_returns_same_patches(self, live_catalog, patches):
        """Iterated patches are the registered objects (construction order)."""
        out = list(live_catalog)
        assert len(out) == len(patches)
        starts = [p.get_coord("time").min() for p in out]
        assert starts == sorted(starts)
        assert {id(p) for p in out} == {id(p) for p in patches}

    def test_get_patch_by_index(self, live_catalog):
        """Integer access works."""
        patch = live_catalog.get_patch(0)
        assert isinstance(patch, dc.Patch)

    def test_add_more_patches(self, patches):
        """add() ingests additional live patches."""
        catalog = PatchCatalog.from_patches(patches[:1])
        assert len(catalog) == 1
        catalog.add(patches[1])
        assert len(catalog) == 2


class TestSelectComposition:
    """select composes lazily with eager validation."""

    def test_select_narrows(self, live_catalog, patches):
        """A time range select excludes non-overlapping patches."""
        t0 = patches[0].get_coord("time").min()
        t1 = patches[0].get_coord("time").max()
        view = live_catalog.select(time=(t0, t1))
        assert len(view) == 1

    def test_chained_selects_and(self, live_catalog, patches):
        """Chained selects AND together."""
        t0 = patches[0].get_coord("time").min()
        view = live_catalog.select(time=(t0, None)).select(
            time=(None, t0 + np.timedelta64(1, "s"))
        )
        assert len(view) == 1

    def test_two_stage_exact_trim(self, live_catalog, patches):
        """Coord range selects trim the loaded patch exactly."""
        t0 = patches[0].get_coord("time").min() + np.timedelta64(2, "s")
        view = live_catalog.select(time=(t0, None))
        patch = view.get_patch(0)
        assert patch.get_coord("time").min() >= t0

    def test_unknown_name_raises_at_select(self, live_catalog):
        """Validation is eager (#435)."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            live_catalog.select(bad_dim=(1, 2))

    def test_no_sql_at_select(self, live_catalog):
        """Selection composes without realizing the dataframe."""
        view = live_catalog.select(distance=(0, 10))
        assert view._df_cache.get(view._revision.value) is None

    def test_views_cannot_mutate(self, live_catalog, patches):
        """Mutation only on the root."""
        view = live_catalog.select(distance=(0, 10))
        with pytest.raises(InvalidSpoolQueryError, match="root catalog"):
            view.add(patches[0])


class TestResidualSelects:
    """samples/relative are patch-local (two-stage)."""

    def test_samples_never_excludes(self, live_catalog, patches):
        """samples=True keeps every patch, trims on load (#447)."""
        view = live_catalog.select(distance=(0, 10), samples=True)
        assert len(view) == len(patches)
        patch = view.get_patch(0)
        # patch-level samples semantics are authoritative (0..9)
        assert len(patch.get_coord("distance")) == 10

    def test_samples_unknown_coord_raises(self, live_catalog):
        """Samples selections validate coord names."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            live_catalog.select(wavelength=(0, 10), samples=True)

    def test_samples_attr_raises(self, live_catalog):
        """Samples selections reject attribute names."""
        with pytest.raises(InvalidSpoolQueryError, match="coordinate-only"):
            live_catalog.select(tag="test", samples=True)

    def test_relative_select(self, live_catalog):
        """Relative bounds resolve against the global envelope (#362)."""
        full = live_catalog.to_df()
        span = (full["time_max"].max() - full["time_min"].min()).total_seconds()
        view = live_catalog.select(time=(1, -1), relative=True)
        patch = view.get_patch(0)
        got_span = (
            patch.get_coord("time").max() - patch.get_coord("time").min()
        ) / np.timedelta64(1, "s")
        assert got_span <= span - 1


class TestRelativeTimeCoords:
    """Patches whose time axis is an offset rather than a date (#798)."""

    @pytest.fixture()
    def relative_time_catalog(self, patches):
        """A catalog over patches with a timedelta64 time coordinate."""
        out = []
        for patch in patches:
            coord = patch.get_coord("time")
            out.append(patch.update_coords(time=coord.values - coord.min()))
        return PatchCatalog.from_patches(tuple(out))

    def test_select_then_load(self, relative_time_catalog):
        """A selected relative range loads, trimmed to the requested bounds."""
        start, stop = np.timedelta64(1, "s"), np.timedelta64(3, "s")
        view = relative_time_catalog.select(time=(start, stop))
        assert len(view)
        coord = view.get_patch(0).get_coord("time")
        assert coord.min() >= start
        assert coord.max() <= stop

    def test_select_open_ended(self, relative_time_catalog):
        """One-sided relative ranges load as well."""
        stop = np.timedelta64(2, "s")
        coord = relative_time_catalog.select(time=(None, stop)).get_patch(0)
        assert coord.get_coord("time").max() <= stop

    @pytest.mark.parametrize(
        "bounds",
        [
            (np.timedelta64(1, "s"), np.timedelta64(3, "s")),
            (np.datetime64("2020-01-01"), np.datetime64("2020-01-02")),
            (None, np.timedelta64(3, "s")),
        ],
    )
    def test_time_ranges_are_not_canonical(self, bounds):
        """Time bounds keep their native form for the residual select."""
        assert _canonical_range(bounds) is None

    def test_numeric_ranges_still_canonical(self):
        """Numeric bounds are still resolved to SI magnitudes."""
        assert _canonical_range((1, 10)).magnitudes == (1.0, 10.0)


class TestDirectoryCatalog:
    """Directory-backed catalogs share machinery with live ones."""

    @pytest.fixture(scope="class")
    def spool_dir(self, tmp_path_factory):
        """A directory of example files."""
        spool = dc.get_example_spool("random_das")
        return dc.examples.spool_to_directory(
            spool, path=tmp_path_factory.mktemp("catalog_dir")
        )

    def test_roundtrip(self, spool_dir):
        """Directory catalog serves the same patches."""
        catalog = PatchCatalog.from_directory(spool_dir).update(progress=None)
        patches = list(catalog)
        assert len(patches) == 3
        assert all(isinstance(p, dc.Patch) for p in patches)
        catalog.close()

    def test_select_and_trim(self, spool_dir):
        """Two-stage select works through files too."""
        catalog = PatchCatalog.from_directory(spool_dir).update(progress=None)
        df = catalog.to_df()
        t0 = df["time_min"].min().to_datetime64() + np.timedelta64(2, "s")
        view = catalog.select(time=(t0, None))
        patch = view.get_patch(0)
        assert patch.get_coord("time").min() >= t0
        catalog.close()


class TestCatalogEdges:
    """Remaining branches: errors, passthroughs, offsets."""

    def test_relative_on_unknown_coord_raises(self, live_catalog):
        """Relative select against an absent coord errors clearly."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            live_catalog.select(wavelength=(1, -1), relative=True)

    def test_relative_requires_range(self, live_catalog):
        """Relative selects take (start, stop) tuples only."""
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            live_catalog.select(time=5, relative=True)

    def test_add_on_file_catalog_not_implemented(self, tmp_path, patches):
        """add() is memory-only for now."""
        path = dc.examples.spool_to_directory(
            dc.spool(list(patches)), path=tmp_path / "d"
        )
        catalog = PatchCatalog.from_directory(path).update(progress=None)
        with pytest.raises(NotImplementedError, match="in-memory"):
            catalog.add(patches[0])
        catalog.close()

    def test_remove(self, patches):
        """remove() drops sources by identity."""
        catalog = PatchCatalog.from_patches(patches)
        target = catalog.sources()["source_path"].iloc[0]
        catalog.remove([target])
        assert len(catalog) == len(patches) - 1


class TestCount:
    """len(catalog) counts in SQL and agrees with the realized relation."""

    @pytest.fixture()
    def diverse_catalog(self):
        """A catalog with heterogeneous attrs and coords."""
        return PatchCatalog.from_patches(list(dc.get_example_spool("diverse_das")))

    def _selections(self, catalog):
        """Views spanning attr, coord range, regex, and chained forms."""
        df = catalog.to_df()
        t0 = df["time_min"].min()
        window = (t0, t0 + dc.to_timedelta64(1))
        return [
            catalog,
            catalog.select(tag="random"),
            catalog.select(time=window),
            catalog.select(distance=(0, 50)),
            catalog.select(tag=re.compile("rand.*")),  # regex residual path
            catalog.select(tag="random").select(time=window),
        ]

    def test_count_matches_realization(self, diverse_catalog):
        """Every view's len equals len(to_df()) (fresh, uncached)."""
        for view in self._selections(diverse_catalog):
            # a fresh view has no cached relation, so len() counts in SQL
            expected = len(view.to_df())
            fresh = view._view(view._queries, view._residuals)
            assert len(fresh) == expected

    def test_len_does_not_realize(self, diverse_catalog, monkeypatch):
        """A cold len() must not pivot coordinates or fetch the relation."""
        catalog = diverse_catalog.select(network="das2")
        fresh = catalog._view(catalog._queries, catalog._residuals)

        def _boom(self):
            raise AssertionError("flat relation realized during len()")

        monkeypatch.setattr(type(fresh), "to_df", _boom)
        assert isinstance(len(fresh), int)

    def test_introspection(self, live_catalog):
        """Names, sources, and metadata pass through."""
        assert "time" in live_catalog.coord_names()
        assert "tag" in live_catalog.attr_names()
        assert len(live_catalog.sources()) == 3
        assert live_catalog.get_metadata()["what_is_this"] == "dascore_spool_index"
        live_catalog.close()

    def test_open_relative_bound(self, live_catalog):
        """Ellipsis/None bounds stay open through relative resolution."""
        view = live_catalog.select(time=(1, None), relative=True)
        assert len(view) >= 1

    def test_numeric_relative_offset(self, live_catalog):
        """Relative selects work on numeric coords too."""
        view = live_catalog.select(distance=(5, -5), relative=True)
        patch = view.get_patch(0)
        assert patch.get_coord("distance").min() >= 5


class TestCanonicalRange:
    """Value semantics of the deferred canonical-SI range."""

    def test_eq_and_hash(self):
        """Equal magnitudes compare and hash equal; other types don't."""
        r1, r2 = _CanonicalRange((1.0, 2.0)), _CanonicalRange((1.0, 2.0))
        assert r1 == r2
        assert hash(r1) == hash(r2)
        assert r1 != _CanonicalRange((1.0, 3.0))
        assert r1 != (1.0, 2.0)  # non-CanonicalRange comparand


class TestViewSerialization:
    """Views serialize only the live entries their rows reference."""

    def test_view_pickles_membership_only(self):
        """A one-patch view of an N-patch live spool ships one patch."""
        base = dc.get_example_patch()
        patches = [
            base.update_attrs(tag=str(i)).new(
                data=np.random.default_rng(i).random(base.shape)
            )
            for i in range(5)
        ]
        spool = dc.spool(patches)
        view = spool.select(tag="0")
        assert len(view) == 1
        payload = pickle.dumps(view)
        baseline = pickle.dumps(dc.spool([patches[0]]))
        assert len(payload) < 2 * len(baseline)
        # the round trip serves the right patch from a one-entry registry
        loaded = pickle.loads(payload)
        assert len(loaded._catalog.resolver.live_entries()) == 1
        assert loaded[0].attrs["tag"] == "0"
        # and the root spool's registry is untouched
        assert len(spool._catalog.resolver.live_entries()) == 5


class TestCatalogConcurrency:
    """Catalog caches must stay coherent under concurrent readers."""

    @pytest.mark.concurrency
    def test_concurrent_realization(self, live_catalog, patches, run_in_threads):
        """Racing first realizations build one backend and one relation."""
        results = run_in_threads(lambda _: len(live_catalog.to_df()))
        assert set(results) == {len(patches)}
        # one cached relation, not one per thread
        assert live_catalog.to_df() is live_catalog.to_df()

    @pytest.mark.concurrency
    def test_concurrent_reads(self, live_catalog, patches, run_in_threads):
        """Mixed len/patch access from several threads agrees."""

        def read(index):
            """Read the catalog a few different ways."""
            return (len(live_catalog), live_catalog.get_patch(index).shape)

        results = run_in_threads(read, len(patches))
        assert {x[0] for x in results} == {len(patches)}
        assert [x[1] for x in results] == [x.shape for x in patches]

    def test_pickled_revision_gets_new_lock(self, live_catalog):
        """Unpickling a catalog installs a fresh revision lock."""
        rebuilt = pickle.loads(pickle.dumps(live_catalog))
        assert rebuilt._revision.lock is not live_catalog._revision.lock
        assert len(rebuilt) == len(live_catalog)
