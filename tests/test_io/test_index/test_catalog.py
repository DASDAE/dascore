"""Tests for PatchCatalog: the unified spool metadata engine."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.io.index import PatchCatalog
from dascore.io.index.query import InvalidSpoolQueryError


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

    def test_first_len_bootstraps(self, live_catalog, patches):
        """First metadata op creates the backend and ingests."""
        assert len(live_catalog) == len(patches)
        assert live_catalog._backend is not None


class TestLiveRoundtrip:
    """Live patches come back identical."""

    def test_iteration_returns_same_patches(self, live_catalog, patches):
        """Iterated patches are the registered objects (order: time)."""
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
        assert view._df_cache is None

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
