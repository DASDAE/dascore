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


@pytest.fixture(
    scope="module",
    params=("memory", "directory", "memory_derived", "directory_derived"),
)
def spool(request, tmp_path_factory):
    """
    The same patches served by each spool type and catalog state.

    The ``*_derived`` params run every spec test over a derived
    (plan-backed) catalog via a content-preserving concatenate — the
    parity net proving one selector engine serves identity and
    restructured spools alike.
    """
    base = dc.get_example_spool("random_das")
    if request.param.startswith("memory"):
        out = dc.spool(list(base))
    else:
        path = dc.examples.spool_to_directory(
            base, path=tmp_path_factory.mktemp("select_spec")
        )
        out = dc.spool(path).update(progress=None)
    if request.param.endswith("_derived"):
        from dascore.io.index.planned import PlanResolver

        out = out.concatenate(time=1)
        assert isinstance(out._catalog.resolver, PlanResolver)
    return out


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
        catalog = spool._catalog
        backend = catalog.backend
        calls = []
        original = backend.query

        def wrapped(query=None, **kwargs):
            calls.append(query)
            return original(query, **kwargs)

        monkeypatch.setattr(backend, "query", wrapped)
        selected = spool.select(time=("2020-01-03", "2020-01-04"))
        assert calls == []
        # realizing the relation (get_contents) runs the composed query;
        # len() alone counts in SQL and never fetches rows.
        selected.get_contents()
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
        assert selected._catalog.is_view

    def test_cold_memory_select(self, forbid_realization):
        """A fresh patch-list spool selects via the catalog, lazily."""
        patches = list(dc.get_example_spool("random_das"))
        forbid_realization()
        fresh = dc.spool(patches)
        selected = fresh.select(tag="random")
        assert selected._catalog.is_view


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

    def test_samples_on_materialized_spool(self, spool):
        """Samples select works after chunk (the dataframe select path)."""
        materialized = spool.chunk(time=None)
        out = materialized.select(distance=(0, 10), samples=True)
        assert len(out) == len(materialized)
        assert len(out[0].get_coord("distance")) == 10

    def test_non_coord_on_materialized_raises(self, spool):
        """The coordinate-only rule also holds on the dataframe path."""
        materialized = spool.chunk(time=None)
        with pytest.raises(InvalidSpoolQueryError, match="coordinate-only"):
            materialized.select(tag="random", samples=True)


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

    def test_relative_on_materialized_spool(self, spool):
        """Relative select works after chunk (the dataframe select path)."""
        materialized = spool.chunk(time=None)
        gmin = materialized.get_contents()["time_min"].min()
        gmax = materialized.get_contents()["time_max"].max()
        out = materialized.select(time=(1, -1), relative=True)
        merged = out.chunk(time=None)[0]
        time = merged.get_coord("time")
        assert time.min() >= np.datetime64(gmin) + np.timedelta64(1, "s")
        assert time.max() <= np.datetime64(gmax) - np.timedelta64(1, "s")


class TestMaterializedNamespaces:
    """_attrs/_coords validation on the dataframe (materialized) path."""

    def test_namespaces_and_unknown_names(self, spool):
        """Namespaced selects and unknown-name errors on a chunked spool."""
        materialized = spool.chunk(time=None)
        assert len(materialized.select(_attrs={"tag": "random"})) == len(materialized)
        # a valid _coords range narrows the materialized spool
        df = materialized.get_contents()
        t0 = df["time_min"].min()
        narrowed = materialized.select(
            _coords={"time": (t0, t0 + np.timedelta64(2, "s"))}
        )
        assert len(narrowed) <= len(materialized)
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            materialized.select(_attrs={"distance": (0, 10)})
        with pytest.raises(InvalidSpoolQueryError, match="not a coordinate"):
            materialized.select(_coords={"tag": "random"})
        with pytest.raises(InvalidSpoolQueryError, match="both"):
            materialized.select(tag="random", _attrs={"tag": "random"})
        with pytest.raises(InvalidSpoolQueryError, match="neither an attribute"):
            materialized.select(nope=1)

    def test_duplicate_namespace_raises(self, spool):
        """A name in both explicit namespaces raises on either path."""
        materialized = spool.chunk(time=None)
        for target in (spool, materialized):
            with pytest.raises(InvalidSpoolQueryError, match="both _attrs and _coords"):
                target.select(_attrs={"time": (None, None)}, _coords={"time": (1, 2)})

    def test_slice_range_form(self, spool):
        """Slice selectors resolve the same on either path (#435 spec)."""
        materialized = spool.chunk(time=None)
        t0 = spool.get_contents()["time_min"].min()
        window = slice(t0, t0 + np.timedelta64(2, "s"))
        for target in (spool, materialized):
            sliced = target.select(time=window)
            tupled = target.select(time=(window.start, window.stop))
            assert len(sliced) == len(tupled)
            assert len(sliced) >= 1


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

    def test_boolean_mask_selectors_rejected(self, ft_patch):
        """Sample masks are patch-level only; the spool points at map()."""
        coord = ft_patch.get_coord("distance")
        mask = np.zeros(len(coord), dtype=bool)
        mask[:5] = True
        with pytest.raises(InvalidSpoolQueryError, match="boolean sample"):
            dc.spool([ft_patch]).select(distance=mask)
        with pytest.raises(InvalidSpoolQueryError, match="boolean sample"):
            dc.spool([ft_patch]).select(distance=list(mask))
        # the per-patch escape hatch still works
        got = ft_patch.select(distance=mask)
        assert len(got.get_coord("distance")) == 5


class TestQuantityDimensionality:
    """Quantity queries keep their dimensionality end to end (review P1)."""

    @pytest.fixture()
    def mixed_unit_spool(self):
        """Two patches whose distance coords are metres and seconds."""
        p_m = dc.get_example_patch()
        p_s = p_m.update_coords(
            distance=p_m.get_coord("distance").set_units("s")
        ).update_attrs(history=[])
        return dc.spool([p_m, p_s])

    def test_incompatible_coord_excluded(self, mixed_unit_spool):
        """A metre query never returns (or trims) a seconds coordinate."""
        from dascore.units import get_quantity, m

        out = mixed_unit_spool.select(_coords={"distance": (1 * m, 2 * m)})
        patches = list(out)
        assert len(patches) == 1
        units = get_quantity(str(patches[0].get_coord("distance").units))
        assert units.dimensionality == m.dimensionality

    def test_all_incompatible_raises(self):
        """A query incompatible with every stored unit raises UnitError."""
        from dascore.exceptions import UnitError
        from dascore.units import m

        p_s = dc.get_example_patch().update_coords(
            distance=dc.get_example_patch().get_coord("distance").set_units("s")
        )
        spool = dc.spool([p_s])
        with pytest.raises(UnitError, match="no units compatible"):
            spool.select(_coords={"distance": (1 * m, 2 * m)}).get_contents()


class TestLazyOrderAndWindow:
    """sort/slice/array selection are lazy Selection specs (D2)."""

    def test_sort_is_lazy_and_ordered(self, tmp_path_factory):
        """Sorting composes a spec; realization returns ordered rows."""
        patches = list(dc.get_example_spool("random_das"))
        spool = dc.spool(list(reversed(patches)))
        out = spool.sort("time")
        assert not out._catalog.is_view or out._catalog._order is not None
        df = out.get_contents()
        assert df["time_min"].is_monotonic_increasing
        assert list(out) == patches

    def test_slice_is_lazy_window(self):
        """Slicing keeps the catalog state and correct membership."""
        patches = list(dc.get_example_spool("random_das"))
        spool = dc.spool(patches)
        part = spool[1:]
        assert part._catalog._ids is not None  # lazy id membership
        assert len(part) == len(patches) - 1
        assert list(part) == patches[1:]

    def test_select_after_slice_filters_within_window(self):
        """D2 composition: predicates apply inside the window."""
        patches = list(dc.get_example_spool("random_das"))
        spool = dc.spool(patches)
        t0 = patches[0].get_coord("time").min()
        # the first patch is outside the window, so selecting its time
        # range inside the window matches nothing
        windowed = spool[1:]
        assert len(windowed.select(time=(None, t0 + np.timedelta64(1, "s")))) == 0

    def test_slice_of_slice_composes(self):
        """Windows compose arithmetically."""
        patches = list(dc.get_example_spool("random_das"))
        part = dc.spool(patches)[1:][1:]
        assert list(part) == patches[2:]

    def test_sorted_spool_slice(self):
        """A slice of a sorted view respects the sort order."""
        patches = list(dc.get_example_spool("random_das"))
        spool = dc.spool(list(reversed(patches)))
        first = spool.sort("time")[0:1]
        assert list(first) == patches[0:1]

    def test_split_parts_pickle_small(self):
        """split() windows keep map() payloads at member size."""
        import pickle

        base = dc.get_example_patch()
        rng = np.random.default_rng(0)
        patches = [base.new(data=rng.random(base.shape)) for _ in range(5)]
        spool = dc.spool(patches)
        parts = list(spool.split(size=1))
        assert len(parts) == 5
        payload = len(pickle.dumps(parts[0]))
        baseline = len(pickle.dumps(dc.spool([patches[0]])))
        assert payload < 2 * baseline


class TestNamespaceTagForm:
    """_attrs/_coords accept names of bare kwargs (tag form)."""

    @pytest.fixture()
    def sensor_spool(self):
        """One patch with an aux coord whose name is not an attr."""
        patch = dc.get_example_patch()
        n = patch.shape[patch.get_axis("distance")]
        return dc.spool(
            [patch.update_coords(sensor=("distance", np.arange(n, dtype=float)))]
        )

    def test_coords_tag_string(self, sensor_spool):
        """A single name tags one bare kwarg as a coordinate."""
        out = sensor_spool.select(sensor=(10, 20), _coords="sensor")
        coord = out[0].get_coord("sensor")
        assert coord.min() == 10.0
        assert coord.max() == 20.0

    def test_coords_tag_collection(self, sensor_spool):
        """A collection of names tags several bare kwargs."""
        out = sensor_spool.select(sensor=(10, 20), _coords=["sensor"])
        assert len(out) == 1

    def test_attrs_tag_string(self, sensor_spool):
        """The attr side accepts the same tag form."""
        out = sensor_spool.select(tag="random", _attrs="tag")
        assert len(out) == 1

    def test_dict_form_unchanged(self, sensor_spool):
        """The general mapping form keeps working."""
        out = sensor_spool.select(_coords={"sensor": (10, 20)})
        assert len(out) == 1

    def test_tag_without_kwarg_raises(self, sensor_spool):
        """Tagging a name with no matching bare kwarg is an error."""
        with pytest.raises(InvalidSpoolQueryError, match="names no bare keyword"):
            sensor_spool.select(_coords="sensor")

    def test_non_string_tag_raises(self, sensor_spool):
        """Tag collections must contain strings."""
        with pytest.raises(InvalidSpoolQueryError, match="mapping of name"):
            sensor_spool.select(sensor=(1, 2), _coords=[3])

    def test_tagged_name_validates_namespace(self, sensor_spool):
        """A tagged name must belong to the claimed namespace."""
        with pytest.raises(InvalidSpoolQueryError, match="not an attribute"):
            sensor_spool.select(sensor=(1, 2), _attrs="sensor")
