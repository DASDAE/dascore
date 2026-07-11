"""
Selector-spec behavior of Spool.select (all spool types).

These encode the hard-break semantics adopted for 0.2: unknown names
raise (#435), samples selections are patch-local (#447), relative ranges
work at spool level (#362), and _attrs/_coords disambiguate explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidSpoolQueryError


@pytest.fixture(scope="module", params=("memory", "directory"))
def spool(request, tmp_path_factory):
    """The same patches served by each spool type."""
    base = dc.get_example_spool("random_das")
    if request.param == "memory":
        return dc.spool(list(base))
    path = dc.examples.spool_to_directory(
        base, path=tmp_path_factory.mktemp("select_spec")
    )
    return dc.spool(path).update(progress=None)


class TestUnknownNames:
    """Unknown selector names raise eagerly (#435)."""

    def test_unknown_kwarg_raises(self, spool):
        """A name that is neither attr nor coord errors."""
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            spool.select(bad_dimension=(1, 2))

    def test_unknown_attr_namespace_raises(self, spool):
        """_attrs validates against attributes only."""
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            spool.select(_attrs={"time": (None, None)})

    def test_unknown_coord_namespace_raises(self, spool):
        """_coords validates against coordinates only."""
        with pytest.raises(InvalidSpoolQueryError, match="not a coordinate"):
            spool.select(_coords={"tag": "random"})

    def test_double_specification_raises(self, spool):
        """A name can't be bare and namespaced at once."""
        with pytest.raises(InvalidSpoolQueryError, match="both"):
            spool.select(tag="random", _attrs={"tag": "random"})


class TestNamespaces:
    """Explicit namespaces select as their bare equivalents."""

    def test_attr_namespace(self, spool):
        """_attrs behaves like the bare attr kwarg."""
        assert len(spool.select(_attrs={"tag": "random"})) == len(spool)

    def test_coord_namespace(self, spool):
        """_coords behaves like the bare coord kwarg."""
        df = spool.get_contents()
        t0 = df["time_min"].min()
        out = spool.select(_coords={"time": (t0, t0 + np.timedelta64(2, "s"))})
        assert len(out) == 1


class TestCatalogPushdown:
    """Public spool selection composes a lazy SQLite query."""

    def test_coord_predicate_reaches_backend(self, spool, monkeypatch):
        """Selection does not query all rows before applying its predicate."""
        catalog = spool._catalog or spool._get_catalog()
        backend = catalog.backend
        calls = []
        original = backend.query

        def wrapped(query=None):
            calls.append(query)
            return original(query)

        monkeypatch.setattr(backend, "query", wrapped)
        selected = spool.select(time=("2020-01-03", "2020-01-04"))
        assert calls == []
        assert len(selected)
        queries = calls[0]
        assert isinstance(queries, list)
        assert queries[0].coords["time"] == ("2020-01-03", "2020-01-04")


class TestLazySelection:
    """Selection construction never realizes the flat relation (review P1).

    Cold spools must compose the selected view without an unfiltered
    backend query; realization happens on first content access. The
    module-scoped ``spool`` fixture is warm by then, so these tests
    build their own fresh spools.
    """

    @pytest.fixture()
    def forbid_realization(self, monkeypatch):
        """Return a callable that makes flat realization fail loudly."""
        from dascore.io.index.catalog import PatchCatalog

        def _boom(self):
            msg = "flat relation realized during selection construction"
            raise AssertionError(msg)

        def _arm():
            monkeypatch.setattr(PatchCatalog, "to_df", _boom)

        return _arm

    def test_cold_directory_select(self, tmp_path_factory, forbid_realization):
        """A cold directory spool selects without touching the relation."""
        path = dc.examples.spool_to_directory(
            dc.get_example_spool("random_das"),
            path=tmp_path_factory.mktemp("lazy_select_dir"),
        )
        dc.spool(path).update(progress=None)  # build the index
        fresh = dc.spool(path)
        forbid_realization()
        selected = fresh.select(time=("2020-01-03", "2020-01-04"))
        assert selected._catalog_native

    def test_cold_memory_select(self, forbid_realization):
        """A fresh patch-list spool selects via the catalog, lazily."""
        patches = list(dc.get_example_spool("random_das"))
        forbid_realization()
        fresh = dc.spool(patches)
        selected = fresh.select(tag="random")
        assert selected._catalog_native


class TestSamples:
    """samples=True never excludes patches; trims on load (#447)."""

    def test_length_preserved(self, spool):
        """The spool keeps every patch."""
        out = spool.select(distance=(0, 10), samples=True)
        assert len(out) == len(spool)

    def test_patch_trimmed_on_load(self, spool):
        """Loaded patches carry the sample trim."""
        out = spool.select(distance=(0, 10), samples=True)
        patch = out[0]
        assert len(patch.get_coord("distance")) == 10

    def test_survives_chunk(self, spool):
        """Post-selects propagate through derived spools."""
        out = spool.select(distance=(0, 10), samples=True).chunk(time=None)
        patch = out[0]
        assert len(patch.get_coord("distance")) == 10

    def test_non_coord_raises(self, spool):
        """Samples selections must name coordinates."""
        with pytest.raises(InvalidSpoolQueryError, match="coordinate-only"):
            spool.select(tag="random", samples=True)


class TestRelative:
    """relative=True resolves against the spool envelope (#362)."""

    def test_trims_both_ends(self, spool):
        """One second off each end of the spool."""
        df = spool.get_contents()
        gmin = df["time_min"].min()
        gmax = df["time_max"].max()
        out = spool.select(time=(1, -1), relative=True)
        merged = out.chunk(time=None)[0]
        time = merged.get_coord("time")
        assert time.min() >= np.datetime64(gmin) + np.timedelta64(1, "s")
        assert time.max() <= np.datetime64(gmax) - np.timedelta64(1, "s")

    def test_requires_range(self, spool):
        """Scalars are rejected with a clear message."""
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            spool.select(time=5, relative=True)

    def test_namespaced_coord_with_attr(self, spool):
        """Only the coordinate range is converted to relative offsets."""
        out = spool.select(_coords={"time": (1, -1)}, tag="random", relative=True)
        assert len(out)
        assert set(out.get_contents()["tag"]) == {"random"}


class TestExistingBehaviorKept:
    """The conventional selections still work."""

    def test_attr_glob(self, spool):
        """Unix-style attr matching."""
        assert len(spool.select(tag="rand*")) == len(spool)

    def test_time_range_narrows(self, spool):
        """Plain time range selection."""
        df = spool.get_contents()
        t0 = df["time_min"].min()
        out = spool.select(time=(t0, t0 + np.timedelta64(2, "s")))
        assert len(out) == 1


class TestUnitCanonicalSelection:
    """Coordinate selection on non-SI patches (review P1).

    The index stores numeric coordinate summaries in canonical SI, so
    range bounds are interpreted as SI end to end: bare numbers mean
    canonical SI, quantities convert, and the exact per-patch residual
    converts back to each patch's native units.
    """

    @pytest.fixture(scope="class")
    def ft_patch(self):
        """An example patch with distance in feet (0..~984 ft)."""
        return dc.get_example_patch().convert_units(distance="ft")

    def test_bare_numbers_are_canonical_si(self, ft_patch):
        """(20, 60) means 20-60 m even on a feet-coordinate patch."""
        coord = dc.spool([ft_patch]).select(distance=(20, 60))[0].get_coord("distance")
        assert float(coord.min()) >= 65  # 20 m == 65.6 ft
        assert float(coord.max()) <= 197  # 60 m == 196.9 ft

    def test_quantity_selector(self, ft_patch):
        """Quantity bounds select the same physical interval."""
        from dascore.units import m

        selected = dc.spool([ft_patch]).select(distance=(20 * m, 60 * m))
        assert len(selected.get_contents()) == 1  # no DimensionalityError
        coord = selected[0].get_coord("distance")
        assert float(coord.min()) >= 65
        assert float(coord.max()) <= 197

    def test_quantity_in_native_units(self, ft_patch):
        """Quantities in the coordinate's own units also work."""
        from dascore.units import get_quantity

        ft = get_quantity("ft")
        coord = (
            dc.spool([ft_patch])
            .select(distance=(100 * ft, 200 * ft))[0]
            .get_coord("distance")
        )
        assert float(coord.min()) >= 99
        assert float(coord.max()) <= 201

    def test_unitless_coords_unchanged(self):
        """Coordinates without units keep plain numeric semantics."""
        patch = dc.get_example_patch()
        coord = dc.spool([patch]).select(distance=(20, 60))[0].get_coord("distance")
        assert 20 <= float(coord.min()) and float(coord.max()) <= 60

    def test_directory_spool(self, ft_patch, tmp_path):
        """The same semantics hold for file-backed spools."""
        from dascore.units import m

        dc.write(ft_patch, tmp_path / "ft.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        coord = spool.select(distance=(20 * m, 60 * m))[0].get_coord("distance")
        assert float(coord.min()) >= 65
        assert float(coord.max()) <= 197

    def test_mixed_unitless_and_feet_bare(self, ft_patch):
        """A bare range trims each patch in its own units (SI meaning)."""
        plain = dc.get_example_patch()  # unitless distance 0..~300
        plain = plain.update_coords(
            distance=plain.get_coord("distance").set_units(None)
        )
        got = dc.spool([plain, ft_patch]).select(distance=(20, 60))
        materialized = [p.get_coord("distance") for p in got]
        assert len(materialized) == 2  # both overlap 20..60 m
        by_units = {str(c.units): c for c in materialized}
        # unitless patch: bare magnitudes applied directly
        assert float(by_units["None"].min()) >= 20
        assert float(by_units["None"].max()) <= 60
        # feet patch: 20..60 m == 65.6..196.9 ft
        assert float(by_units["1 ft"].min()) >= 65
        assert float(by_units["1 ft"].max()) <= 197

    def test_mixed_unitless_and_feet_quantity(self, ft_patch):
        """A metre quantity range works across a mixed population."""
        from dascore.units import m

        plain = dc.get_example_patch()
        plain = plain.update_coords(
            distance=plain.get_coord("distance").set_units(None)
        )
        got = dc.spool([plain, ft_patch]).select(distance=(20 * m, 60 * m))
        assert len(got.get_contents()) == 2  # no UnitError on the unitless row

    def test_scalar_coord_rejected(self, ft_patch):
        """A scalar coordinate selector is rejected eagerly, clearly."""
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            dc.spool([ft_patch]).select(distance=100)

    def test_value_membership_rejected(self, ft_patch):
        """A wrong-arity list is reported as a malformed range."""
        from dascore.exceptions import ParameterError

        with pytest.raises(ParameterError, match="length 2 sequence"):
            dc.spool([ft_patch]).select(distance=[10, 20, 50])

    def test_chained_views(self, ft_patch):
        """Canonicalization holds across chained selections."""
        from dascore.units import m

        coord = (
            dc.spool([ft_patch])
            .select(distance=(0 * m, 90 * m))
            .select(distance=(20, 60))[0]
            .get_coord("distance")
        )
        assert float(coord.min()) >= 65
        assert float(coord.max()) <= 197
