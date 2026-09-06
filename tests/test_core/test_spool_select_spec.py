"""
Selector-spec behavior of Spool.select (all spool types).

These encode the hard-break semantics adopted for 0.2: unknown names
raise (#435), samples selections are patch-local (#447), relative ranges
resolve against each patch (#362), and _attrs/_coords disambiguate
explicitly.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import (
    ChunkError,
    InvalidSpoolQueryError,
    ParameterError,
    UnitError,
)
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.planned import PlanResolver
from dascore.units import get_quantity, m


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
    # 8,000 samples per patch rather than 600,000, at a step which keeps
    # each of them 8 seconds long: the specs select windows in seconds, and
    # a window narrower than one patch is what several of them are about.
    # Distance stays wider than the 10-sample window TestSamples asks for,
    # or that trim would be a no-op.
    base = dc.get_example_spool(
        "random_das", shape=(40, 200), time_step=dc.to_timedelta64(0.04)
    )
    if request.param.startswith("memory"):
        out = dc.spool(list(base))
    else:
        path = dc.examples.spool_to_directory(
            base, path=tmp_path_factory.mktemp("select_spec")
        )
        out = dc.spool(path).update(progress=None)
    if request.param.endswith("_derived"):
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

    def test_cold_relative_select(self, forbid_realization):
        """Relative selection defers each patch-local coordinate envelope."""
        patches = list(dc.get_example_spool("random_das"))
        fresh = dc.spool(patches)
        forbid_realization()
        selected = fresh.select(time=(0, 2), relative=True)
        assert selected._catalog.is_view


class TestSamples:
    """samples=True never excludes patches; trims on load (#447)."""

    def test_length_preserved(self, spool):
        """The spool keeps every patch, and the window is a real trim."""
        out = spool.select(distance=(0, 10), samples=True)
        assert len(out) == len(spool)
        assert len(spool[0].get_coord("distance")) > 10

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

    def test_contents_match_loaded_window(self, spool):
        """Presented envelopes describe the trimmed patch, not its source."""
        out = spool.select(time=(0, 100), samples=True)
        for row, patch in zip(out.get_contents().itertuples(), out, strict=True):
            coord = patch.get_coord("time")
            assert row.time_min == coord.min()
            assert row.time_max == coord.max()

    def test_chunk_keeps_the_whole_selected_window(self, spool):
        """The sample trim also belongs to the source, not to each chunk."""
        chunked = spool.select(time=(25, 175), samples=True).chunk(time=2)
        assert len(chunked)
        rows = chunked.get_contents().itertuples()
        for row, patch in zip(rows, chunked, strict=True):
            coord = patch.get_coord("time")
            assert coord.min() == row.time_min
            assert coord.max() == row.time_max

    def test_union_keeps_emptied_members(self, spool):
        """An emptied sample window keeps members, as a relative one does."""
        out = spool.select(time=(5000, 6000), samples=True)
        assert len(out + dc.spool([])) == len(out) == len(spool)


class TestRelative:
    """relative=True resolves independently against each patch."""

    @staticmethod
    def _assert_patch_local(source, selected, selection):
        """The spool result is the patch-level relative selection."""
        expected = source.select(time=selection, relative=True)
        assert selected.equals(expected)

    def test_first_two_seconds_of_each_patch(self, spool):
        """A positive range starts at every patch's own beginning."""
        selection = (0, 2)
        out = spool.select(time=selection, relative=True)
        assert len(out) == len(spool)
        for source, selected in zip(spool, out, strict=True):
            self._assert_patch_local(source, selected, selection)

    def test_trims_both_ends_of_each_patch(self, spool):
        """Positive and negative bounds use each patch's endpoints."""
        selection = (1, -1)
        out = spool.select(time=selection, relative=True)
        assert len(out) == len(spool)
        for source, selected in zip(spool, out, strict=True):
            self._assert_patch_local(source, selected, selection)

    def test_contents_show_each_patch_local_window(self, spool):
        """Presented envelopes match every loaded relative selection."""
        out = spool.select(time=(0, 2), relative=True)
        contents = out.get_contents()
        for row, patch in zip(contents.itertuples(), out, strict=True):
            coord = patch.get_coord("time")
            assert row.time_min == coord.min()
            assert row.time_max == coord.max()

    def test_relative_then_absolute_preserves_order(self, spool):
        """A later absolute window intersects the relative result in order."""
        source = spool[0]
        start = source.get_coord("time").min()
        out = dc.spool([source]).select(time=(0, 2), relative=True)
        out = out.select(time=(start + np.timedelta64(1, "s"), None))
        expected = source.select(time=(0, 2), relative=True)
        expected = expected.select(time=(start + np.timedelta64(1, "s"), None))
        assert out[0].equals(expected)
        assert (
            out.get_contents()["time_max"].iloc[0] == expected.get_coord("time").max()
        )

    def test_requires_range(self, spool):
        """Scalars are rejected with a clear message."""
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            spool.select(time=5, relative=True)

    def test_namespaced_coord_with_attr(self, spool):
        """Attribute predicates still filter a patch-local relative view."""
        selection = (1, -1)
        out = spool.select(_coords={"time": selection}, tag="random", relative=True)
        assert len(out) == len(spool)
        assert set(out.get_contents()["tag"]) == {"random"}
        for source, selected in zip(spool, out, strict=True):
            self._assert_patch_local(source, selected, selection)

    def test_relative_on_materialized_spool(self, spool):
        """Patch-local relative selection survives a derived catalog."""
        materialized = spool.chunk(time=None)
        selection = (0, 2)
        out = materialized.select(time=selection, relative=True)
        assert len(out) == len(materialized)
        for source, selected in zip(materialized, out, strict=True):
            self._assert_patch_local(source, selected, selection)

    @pytest.mark.parametrize("selection", [(1000, None), (None, -1000)])
    def test_open_bound_past_the_patch_selects_nothing(self, spool, selection):
        """A one-sided window off the end keeps nothing, and reports that."""
        out = spool.select(time=selection, relative=True)
        assert len(out) == len(spool)
        for source, selected in zip(spool, out, strict=True):
            self._assert_patch_local(source, selected, selection)
        # The open side is the envelope extreme already. If it stood in for
        # the closed bound the row would present a one-instant window, and
        # the plans below would describe data no patch actually has.
        contents = out.get_contents()
        assert contents["time_min"].isna().all()
        assert contents["time_max"].isna().all()
        assert not len(out.get_gaps())
        assert not len(out.chunk(time=None))

    def test_emptied_view_survives_union(self, spool):
        """Union keeps emptied members and says why they cannot be chunked."""
        empt = spool.select(time=(1000, 2000), relative=True)
        other = dc.get_example_spool("random_das", length=2)
        union = empt + other
        assert len(union) == len(empt) + len(other)
        # Baking the view writes through the index schema, which cannot
        # carry the emptied marker, so these arrive with a null envelope
        # and the error reports them as missing the dimension.
        with pytest.raises(ChunkError, match="lack the"):
            union.chunk(time=None)
        assert len(union.chunk(time=None, missing_dim="drop")) >= 1

    def test_chunk_keeps_the_whole_selected_window(self, spool):
        """Chunking a relative view windows the selection, not each chunk."""
        chunked = spool.select(time=(1, -1), relative=True).chunk(time=2)
        assert len(chunked)
        rows = chunked.get_contents().itertuples()
        for row, patch in zip(rows, chunked, strict=True):
            coord = patch.get_coord("time")
            assert coord.min() == row.time_min
            assert coord.max() == row.time_max

    def test_absolute_after_relative_keeps_both_windows(self, spool):
        """A later absolute window narrows the relative one, not the source."""
        relative = spool.select(time=(1, -1), relative=True)
        start = relative.get_contents()["time_min"].min()
        window = (start + dc.to_timedelta64(1), start + dc.to_timedelta64(3))
        out = relative.select(time=window)
        assert len(out)
        for row, patch in zip(out.get_contents().itertuples(), out, strict=True):
            coord = patch.get_coord("time")
            assert coord.min() == row.time_min
            assert coord.max() == row.time_max

    def test_string_coordinate_rejected(self, spool):
        """String coordinates have no offset arithmetic; say so at select."""
        labelled = dc.spool(
            [
                patch.update_coords(
                    label=(
                        "distance",
                        np.array([f"s{i}" for i in range(patch.shape[0])]),
                    )
                )
                for patch in spool
            ]
        )
        with pytest.raises(InvalidSpoolQueryError, match="String coordinate"):
            labelled.select(label=("s1", "s3"), relative=True)

    def test_string_definitions_are_not_candidates(self):
        """Where a name is a string on some patches, only the rest answer."""
        patch = dc.get_example_patch("random_das")
        size = patch.shape[0]
        strings = patch.update_coords(
            label=("distance", np.array([f"s{i}" for i in range(size)]))
        )
        numbers = patch.update_coords(label=("distance", np.arange(size, dtype=float)))
        out = dc.spool([strings, numbers]).select(label=(0, 1), relative=True)
        assert len(out) == len(out.get_contents()) == len(list(out)) == 1

    def test_bounds_resolving_out_of_order_are_ordered(self, spool):
        """A window whose ends resolve reversed is presented in order."""
        selection = (-2, 2)
        out = spool.select(time=selection, relative=True)
        contents = out.get_contents()
        assert (contents["time_min"] <= contents["time_max"]).all()
        for source, selected in zip(spool, out, strict=True):
            self._assert_patch_local(source, selected, selection)

    def test_window_wider_than_the_patch_is_clipped(self, spool):
        """A window past the end reports the patch, not the window."""
        out = spool.select(time=(0, 100), relative=True)
        for row, patch in zip(out.get_contents().itertuples(), out, strict=True):
            coord = patch.get_coord("time")
            assert row.time_min == coord.min()
            assert row.time_max == coord.max()

    def test_trimmed_size_is_forgotten(self, spool):
        """A relative trim invalidates the source's recorded data size."""
        out = spool.select(time=(0, 2), relative=True)
        assert out.get_contents()["data_size"].isna().all()

    def test_equals_its_own_materialization(self, spool):
        """The emptied marker is internal and never splits equality."""
        out = spool.select(time=(0, 2), relative=True)
        assert out == (out + dc.spool([]))

    def test_len_agrees_with_iteration_after_narrowing(self, spool):
        """An emptied view narrowed again keeps len, contents and iteration together."""
        out = spool.select(distance=(500, 600), relative=True)
        out = out.select(distance=(0, 10))
        assert len(out) == len(out.get_contents()) == len(list(out))


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


class TestOriginalUnitsPresentation:
    """The index stores and presents envelopes in original units (#863)."""

    def test_get_contents_shows_units_beside_native_values(self):
        """Envelope columns are native and carry their unit column."""
        spool = dc.spool([dc.get_example_patch().convert_units(distance="ft")])
        df = spool.get_contents()
        assert str(df["distance_units"].iloc[0]) == "ft"
        coord = spool[0].get_coord("distance")
        assert float(df["distance_min"].iloc[0]) == pytest.approx(float(coord.min()))
        assert float(df["distance_max"].iloc[0]) == pytest.approx(float(coord.max()))

    def test_quantity_select_presents_converted_envelopes(self):
        """A metre range shows as the feet interval it selects."""
        spool = dc.spool([dc.get_example_patch().convert_units(distance="ft")])
        df = spool.select(distance=(20 * m, 60 * m)).get_contents()
        assert float(df["distance_min"].iloc[0]) == pytest.approx(65.6, abs=0.1)
        assert float(df["distance_max"].iloc[0]) == pytest.approx(196.9, abs=0.1)

    def test_degree_coordinate_stays_degrees(self):
        """The motivating case: geographic degrees never show as radians."""
        patch = dc.get_example_patch()
        values = np.linspace(-117.05, -116.9, patch.coord_shapes["distance"][0])
        patch = patch.update_coords(distance=values).set_units(distance="degrees")
        spool = dc.spool([patch])
        df = spool.get_contents()
        assert str(df["distance_units"].iloc[0]) == "deg"
        assert float(df["distance_min"].iloc[0]) == pytest.approx(-117.05)
        # a bare geographic range selects, matching Patch.select
        got = spool.select(distance=(-117.0, -116.95))
        assert len(got) == 1
        direct = patch.select(distance=(-117.0, -116.95))
        assert got[0].get_coord("distance") == direct.get_coord("distance")


class TestUnitCanonicalSelection:
    """Coordinate selection on non-SI patches (#863).

    The index stores numeric coordinate envelopes in each coordinate's
    original units, so a bare range means native units — exactly what
    `Patch.select` means by it — while quantities convert themselves to
    whatever units each stored definition uses.
    """

    @pytest.fixture(scope="class")
    def ft_patch(self):
        """An example patch with distance in feet (0..~984 ft)."""
        return dc.get_example_patch().convert_units(distance="ft")

    def test_bare_numbers_are_native_units(self, ft_patch):
        """(20, 60) means 20-60 ft on a feet patch, as Patch.select does."""
        spooled = dc.spool([ft_patch]).select(distance=(20, 60))[0]
        direct = ft_patch.select(distance=(20, 60))
        assert spooled.get_coord("distance") == direct.get_coord("distance")
        coord = spooled.get_coord("distance")
        assert float(coord.min()) >= 20
        assert float(coord.max()) <= 60

    def test_quantity_selector(self, ft_patch):
        """Quantity bounds select the same physical interval."""
        selected = dc.spool([ft_patch]).select(distance=(20 * m, 60 * m))
        assert len(selected.get_contents()) == 1  # no DimensionalityError
        coord = selected[0].get_coord("distance")
        assert float(coord.min()) >= 65
        assert float(coord.max()) <= 197

    def test_quantity_in_native_units(self, ft_patch):
        """Quantities in the coordinate's own units also work."""
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
        dc.write(ft_patch, tmp_path / "ft.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        coord = spool.select(distance=(20 * m, 60 * m))[0].get_coord("distance")
        assert float(coord.min()) >= 65
        assert float(coord.max()) <= 197

    def test_mixed_unitless_and_feet_bare(self, ft_patch):
        """A bare range trims each patch in its own native units."""
        plain = dc.get_example_patch()  # unitless distance 0..~300
        plain = plain.update_coords(
            distance=plain.get_coord("distance").set_units(None)
        )
        got = dc.spool([plain, ft_patch]).select(distance=(20, 60))
        materialized = [p.get_coord("distance") for p in got]
        assert len(materialized) == 2  # both overlap their own 20..60
        by_units = {str(c.units): c for c in materialized}
        # unitless patch: bare magnitudes applied directly
        assert float(by_units["None"].min()) >= 20
        assert float(by_units["None"].max()) <= 60
        # feet patch: 20..60 means 20..60 ft
        assert float(by_units["1 ft"].min()) >= 20
        assert float(by_units["1 ft"].max()) <= 60

    def test_mixed_unitless_and_feet_quantity(self, ft_patch):
        """A metre quantity range works across a mixed population."""
        plain = dc.get_example_patch()
        plain = plain.update_coords(
            distance=plain.get_coord("distance").set_units(None)
        )
        got = dc.spool([plain, ft_patch]).select(distance=(20 * m, 60 * m))
        assert len(got.get_contents()) == 2  # no UnitError on the unitless row
        # materializing applies the residual: the unitless coordinate reads
        # the canonical SI magnitudes bare (documented policy), the feet
        # coordinate converts.
        by_units = {str(p.get_coord("distance").units): p for p in got}
        plain_coord = by_units["None"].get_coord("distance")
        assert float(plain_coord.min()) >= 20
        assert float(plain_coord.max()) <= 60
        ft_coord = by_units["1 ft"].get_coord("distance")
        assert float(ft_coord.min()) >= 65  # 20 m == 65.6 ft
        assert float(ft_coord.max()) <= 197

    def test_scalar_coord_rejected(self, ft_patch):
        """A scalar coordinate selector is rejected eagerly, clearly."""
        with pytest.raises(InvalidSpoolQueryError, match="range selectors"):
            dc.spool([ft_patch]).select(distance=100)

    def test_value_membership_rejected(self, ft_patch):
        """A wrong-arity list is reported as a malformed range."""
        with pytest.raises(ParameterError, match="length 2 sequence"):
            dc.spool([ft_patch]).select(distance=[10, 20, 50])

    def test_chained_views(self, ft_patch):
        """Native reading holds across chained selections."""
        coord = (
            dc.spool([ft_patch])
            .select(distance=(0 * m, 90 * m))
            .select(distance=(20, 60))[0]
            .get_coord("distance")
        )
        # the quantity view keeps 0..295 ft; the bare view means feet
        assert float(coord.min()) >= 20
        assert float(coord.max()) <= 60

    def test_mixed_bare_and_quantity_bounds(self, ft_patch):
        """A bare bound stays feet beside a metre bound, as on the patch."""
        mixed = (20, 60 * m)
        direct = ft_patch.select(distance=mixed)
        via = dc.spool([ft_patch]).select(distance=mixed)[0]
        assert via.get_coord("distance") == direct.get_coord("distance")

    def test_mixed_bound_ordering_is_checked_after_conversion(self, ft_patch):
        """(100, 50*m) is a valid interval once both bounds speak feet.

        The bare bound means the coordinate's units and the quantity
        means its own, so comparing their raw magnitudes rejects a range
        the patch accepts.
        """
        value = (100, 50 * m)
        direct = ft_patch.select(distance=value)
        via = dc.spool([ft_patch]).select(distance=value)[0]
        assert via.get_coord("distance") == direct.get_coord("distance")

    def test_reversed_ranges_still_raise(self, ft_patch):
        """Bounds in one frame of reference keep their ordering check."""
        for value in ((60, 20), (60 * m, 20 * m)):
            with pytest.raises(InvalidSpoolQueryError, match="lo > hi"):
                len(dc.spool([ft_patch]).select(distance=value))

    def test_relative_excludes_dimensionally_incompatible(self, ft_patch):
        """A relative metre window drops a seconds coordinate, as absolute does."""
        seconds = ft_patch.set_units(distance="s")
        spool = dc.spool([ft_patch, seconds])
        out = spool.select(distance=(1 * m, 2 * m), relative=True)
        assert len(out) == len(out.get_contents()) == len(list(out)) == 1
        assert str(out[0].get_coord("distance").units) == "1 ft"

    def test_relative_only_unitless_survive_incompatible_units(self, ft_patch):
        """When no stored unit fits, only unitless definitions stay candidates."""
        seconds = ft_patch.set_units(distance="s")
        plain = ft_patch.update_coords(
            distance=ft_patch.get_coord("distance").set_units(None)
        )
        out = dc.spool([seconds, plain]).select(distance=(1 * m, 2 * m), relative=True)
        contents = out.get_contents()
        assert len(out) == len(contents) == 1
        assert contents["distance_units"].isna().all()
        # A unitless coordinate cannot answer a metre offset. The patch-level
        # selection says so, and the spool passes that through rather than
        # silently treating the magnitude as native.
        with pytest.raises(UnitError):
            list(out)

    def test_chunk_of_selected_view_converts_plan(self, ft_patch):
        """Chunking after a quantity select trims the converted interval."""
        sel = dc.spool([ft_patch]).select(distance=(20 * m, 60 * m))
        out = sel.chunk(distance=None)
        coord = out[0].get_coord("distance")
        assert float(coord.min()) >= 65.5  # 20 m in ft
        assert float(coord.max()) <= 197.0  # 60 m in ft

    def test_chunked_spool_keeps_units(self, ft_patch):
        """A derived (chunked) view still reports original units."""
        df = dc.spool([ft_patch]).chunk(distance=None).get_contents()
        assert str(df["distance_units"].iloc[0]) == "ft"

    def test_directory_spool_units_column(self, ft_patch, tmp_path):
        """The persisted directory index presents units too."""
        dc.write(ft_patch, tmp_path / "ft.h5", "dasdae")
        df = dc.spool(tmp_path).update().get_contents()
        assert str(df["distance_units"].iloc[0]) == "ft"

    def test_union_of_mixed_unit_spools(self):
        """A union keeps each member's native units and semantics."""
        pm = dc.get_example_patch().set_units(distance="m")
        pf = pm.convert_units(distance="ft")
        union = dc.spool([pm]) + dc.spool([pf])
        units = set(union.get_contents()["distance_units"])
        assert units == {"m", "ft"}
        for patch in union.select(distance=(20, 60)):
            coord = patch.get_coord("distance")
            assert float(coord.min()) >= 20
            assert float(coord.max()) <= 60

    def test_mixed_unit_archive(self):
        """Bare selects per-file native intervals; a quantity selects one."""
        pm = dc.get_example_patch().set_units(distance="m")
        pf = pm.convert_units(distance="ft")
        sp = dc.spool([pm, pf])
        for patch in sp.select(distance=(20, 60)):
            coord = patch.get_coord("distance")
            assert float(coord.min()) >= 20
            assert float(coord.max()) <= 60
        for patch in sp.select(distance=(20 * m, 60 * m)):
            coord = patch.get_coord("distance").convert_units("m")
            assert float(coord.min()) >= 19.9
            assert float(coord.max()) <= 60.1

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
        out = mixed_unit_spool.select(_coords={"distance": (1 * m, 2 * m)})
        patches = list(out)
        assert len(patches) == 1
        units = get_quantity(str(patches[0].get_coord("distance").units))
        assert units.dimensionality == m.dimensionality

    def test_all_incompatible_raises(self):
        """A query incompatible with every stored unit raises UnitError."""
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
