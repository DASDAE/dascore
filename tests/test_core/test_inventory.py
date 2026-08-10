"""Tests for the DASDAE inventory model."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

import dascore as dc
from dascore.core import Inventory
from dascore.core import inventory as inv
from dascore.exceptions import InvalidInventoryError
from dascore.utils.models import _values_equal


def build_inventory() -> inv.Inventory:
    """Build a small, valid inventory used across tests."""
    acquisition = inv.Acquisition(
        code="RAW",
        location_code="00",
        start_time="2026-06-01",
        data_category="DAS",
        data_type="strain_rate",
        sample_rate=500.0,
        gauge_length=10.0,
        spatial_interval=1.0,
        start_distance=0.0,
    )
    geometry = inv.Geometry(
        name="survey",
        distance=(0.0, 100.0, 200.0),
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 1.0)),
    )
    path = inv.OpticalPath(
        name="main",
        location_code="00",
        start_time="2026-06-01",
        optical_components=(inv.FiberSegment(name="fiber", optical_length=250.0),),
        geometry=(geometry,),
        coupling=(
            inv.CouplingCondition(
                start_distance=0.0, end_distance=200.0, coupling_type="trench"
            ),
        ),
        annotations=(
            inv.OpticalPathAnnotation(
                start_distance=0.0, end_distance=100.0, label="east"
            ),
        ),
    )
    array = inv.FiberArray(
        code="L001",
        start_time="2026-06-01",
        acquisitions=(acquisition,),
        optical_paths=(path,),
    )
    network = inv.Network(code="DAS", fiber_arrays=(array,))
    return inv.Inventory(networks=(network,))


class TestGeometry:
    """Geometry segment rules."""

    def test_requires_two_points(self):
        """Requires two points."""
        with pytest.raises(ValidationError, match="at least two"):
            inv.Geometry(distance=(1.0,), coordinates=((0.0, 0.0),))

    def test_strictly_increasing(self):
        """Strictly increasing."""
        with pytest.raises(ValidationError, match="strictly increasing"):
            inv.Geometry(distance=(1.0, 1.0), coordinates=((0.0, 0.0), (1.0, 1.0)))

    def test_paired_lengths(self):
        """Paired lengths."""
        with pytest.raises(ValidationError, match="same length"):
            inv.Geometry(distance=(0.0, 1.0), coordinates=((0.0, 0.0),))

    def test_coil_repeated_coordinates(self):
        """A coil interpolates to a constant coordinate."""
        coil = inv.Geometry(
            distance=(1200.0, 1300.0),
            coordinates=((500.0, 120.0), (500.0, 120.0)),
        )
        out = coil.interpolate([1200.0, 1250.0, 1299.0])
        assert np.allclose(out, [[500.0, 120.0]] * 3)

    def test_uncovered_is_nan(self):
        """Uncovered is nan."""
        geo = inv.Geometry(distance=(10.0, 20.0), coordinates=((0.0, 0.0), (1.0, 1.0)))
        out = geo.interpolate([5.0, 25.0])
        assert np.all(np.isnan(out))


class TestPathTracks:
    """Track-kind rules: tiling, function tracks, set track."""

    def test_partial_coverage_is_legal(self):
        """Partial coverage is legal."""
        path = build_inventory().networks[0].fiber_arrays[0].optical_paths[0]
        assert path.check() is path

    def test_coupling_overlap_raises(self):
        """Coupling overlap raises."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    start_distance=0.0, end_distance=60.0, coupling_type="trench"
                ),
                inv.CouplingCondition(
                    start_distance=50.0, end_distance=70.0, coupling_type="conduit"
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Overlapping coupling"):
            path.check()

    def test_geometry_overlap_raises(self):
        """Geometry overlap raises."""
        seg = dict(coordinates=((0.0, 0.0), (1.0, 1.0)))
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(distance=(0.0, 60.0), **seg),
                inv.Geometry(distance=(50.0, 80.0), **seg),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Overlapping geometry"):
            path.check()

    def test_annotations_overlap_freely(self):
        """Annotations overlap freely."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            annotations=(
                inv.OpticalPathAnnotation(
                    start_distance=0.0, end_distance=60.0, label="a"
                ),
                inv.OpticalPathAnnotation(
                    start_distance=50.0, end_distance=70.0, label="b"
                ),
            ),
        )
        assert path.check() is path

    def test_out_of_bounds_raises(self):
        """Out of bounds raises."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    start_distance=90.0, end_distance=150.0, coupling_type="trench"
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="extends past"):
            path.check()

    def test_outer_endpoint_included(self):
        """The outermost covered endpoint of the geometry track resolves."""
        path = build_inventory().networks[0].fiber_arrays[0].optical_paths[0]
        coords = path.coordinates_at([200.0])
        assert np.allclose(coords, [[2.0, 0.0, 1.0]])

    def test_uncovered_distance_is_nan(self):
        """Uncovered distance is nan."""
        path = build_inventory().networks[0].fiber_arrays[0].optical_paths[0]
        assert np.all(np.isnan(path.coordinates_at([225.0])))


class TestDistanceMap:
    """DistanceMap rules and mapping behavior."""

    def test_exactly_one_input_axis(self):
        """Exactly one input axis."""
        with pytest.raises(ValidationError, match="exactly one input axis"):
            inv.DistanceMap(channel=(1.0,), instrument_distance=(1.0,), distance=(5.0,))
        with pytest.raises(ValidationError, match="exactly one input axis"):
            inv.DistanceMap(distance=(5.0,))

    def test_two_point_map(self):
        """Two point map."""
        dmap = inv.DistanceMap(channel=(512.0, 1710.0), distance=(500.0, 1698.0))
        out = dmap.map_to_distance([512, 1111, 1710])
        assert np.isclose(out[0], 500.0)
        assert np.isclose(out[-1], 1698.0)

    def test_outside_coverage_is_nan(self):
        """Outside coverage is nan."""
        dmap = inv.DistanceMap(channel=(512.0, 1710.0), distance=(500.0, 1698.0))
        assert np.isnan(dmap.map_to_distance([2000.0])[0])

    def test_single_point_needs_slope(self):
        """Single point needs slope."""
        dmap = inv.DistanceMap(channel=(512.0,), distance=(500.0,))
        with pytest.raises(InvalidInventoryError, match="slope"):
            dmap.map_to_distance([600.0])
        out = dmap.map_to_distance([612.0], slope=1.0)
        assert np.isclose(out[0], 600.0)

    def test_instrument_distance_axis(self):
        """Instrument distance axis."""
        dmap = inv.DistanceMap(
            instrument_distance=(0.0, 2000.0), distance=(-12.4, 1985.2)
        )
        out = dmap.map_to_distance([0.0])
        assert np.isclose(out[0], -12.4)


class TestAcquisition:
    """Acquisition resolution mechanisms and code rules."""

    def test_mechanisms_mutually_exclusive(self):
        """Mechanisms mutually exclusive."""
        dmap = inv.DistanceMap(channel=(1.0, 2.0), distance=(5.0, 6.0))
        with pytest.raises(ValidationError, match="mutually exclusive"):
            inv.Acquisition(code="RAW", start_distance=0.0, distance_map=dmap)

    def test_affine_mapping(self):
        """Affine mapping."""
        acq = inv.Acquisition(code="RAW", start_distance=10.0, spatial_interval=2.0)
        assert np.allclose(acq.channel_to_distance([0, 5]), [10.0, 20.0])

    def test_map_mapping_used_when_present(self):
        """Map mapping used when present."""
        dmap = inv.DistanceMap(channel=(0.0, 10.0), distance=(100.0, 120.0))
        acq = inv.Acquisition(code="RAW", distance_map=dmap)
        assert np.isclose(acq.channel_to_distance([5])[0], 110.0)

    def test_no_mechanism_raises(self):
        """No mechanism raises."""
        acq = inv.Acquisition(code="RAW")
        with pytest.raises(InvalidInventoryError, match="no channel-resolution"):
            acq.channel_to_distance([0])

    def test_code_charset(self):
        """Code charset."""
        with pytest.raises(ValidationError, match="Invalid"):
            inv.Acquisition(code="MY_RAW")
        with pytest.raises(ValidationError, match="Invalid"):
            inv.Acquisition(code="")

    def test_blank_location_legal(self):
        """Blank location legal."""
        acq = inv.Acquisition(code="RAW", location_code="")
        assert acq.location_code == ""


class TestEpochs:
    """Half-open epoch rules across the tree."""

    def test_concurrent_paths_same_location_raise(self):
        """Concurrent paths same location raise."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(name="a", start_time="2020-01-01"),
                inv.OpticalPath(name="b", start_time="2021-01-01"),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="overlap in time"):
            array.check()

    def test_concurrent_paths_different_locations_legal(self):
        """Location-scoped paths: similar co-located fibers share an array."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(name="a", location_code="00", start_time="2020-01-01"),
                inv.OpticalPath(name="b", location_code="01", start_time="2020-01-01"),
            ),
        )
        assert array.check() is array

    def test_abutting_epochs_legal(self):
        """Half-open handoff: end == next start is legal."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(
                    name="a", start_time="2020-01-01", end_time="2021-01-01"
                ),
                inv.OpticalPath(name="b", start_time="2021-01-01"),
            ),
        )
        assert array.check() is array

    def test_acquisition_epoch_overlap_raises(self):
        """Acquisition epoch overlap raises."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(
                inv.Acquisition(code="RAW", start_time="2020-01-01"),
                inv.Acquisition(code="RAW", start_time="2020-06-01"),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="overlap in time"):
            array.check()

    def test_acquisition_gap_legal(self):
        """Acquisition gap legal."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(
                inv.Acquisition(
                    code="RAW", start_time="2020-01-01", end_time="2020-06-01"
                ),
                inv.Acquisition(code="RAW", start_time="2020-07-15"),
            ),
        )
        assert array.check() is array

    def test_station_fiber_code_collision_raises(self):
        """Station fiber code collision raises."""
        network = inv.Network(
            code="DAS",
            fiber_arrays=(inv.FiberArray(code="L001", start_time="2020-01-01"),),
            stations=(inv.Station(code="L001", start_time="2020-06-01"),),
        )
        with pytest.raises(InvalidInventoryError, match="share code"):
            network.check()


class TestResolution:
    """data_source_id + time resolution."""

    def test_happy_path(self):
        """Happy path."""
        inventory = build_inventory()
        ctx = inventory.resolve("DAS.L001.00.RAW", time="2026-07-01")
        assert ctx.acquisition.code == "RAW"
        assert ctx.optical_path.name == "main"
        assert ctx.fiber_array.code == "L001"

    def test_time_selects_epoch(self):
        """Time selects epoch."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(
                inv.Acquisition(
                    code="RAW",
                    start_time="2020-01-01",
                    end_time="2021-01-01",
                    gauge_length=10.0,
                ),
                inv.Acquisition(code="RAW", start_time="2021-01-01", gauge_length=20.0),
            ),
        )
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        early = inventory.resolve("DAS.L001..RAW", time="2020-06-01")
        late = inventory.resolve("DAS.L001..RAW", time="2021-01-01")
        assert early.acquisition.gauge_length == 10.0
        # The boundary instant belongs to the newer epoch (half-open).
        assert late.acquisition.gauge_length == 20.0

    def test_blank_location_id(self):
        """DAS.L001..RAW is a legal identifier with a blank location."""
        array = inv.FiberArray(code="L001", acquisitions=(inv.Acquisition(code="RAW"),))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        ctx = inventory.resolve("DAS.L001..RAW")
        assert ctx.acquisition.location_code == ""

    def test_missing_raises(self):
        """Missing raises."""
        inventory = build_inventory()
        with pytest.raises(InvalidInventoryError, match="0 acquisitions"):
            inventory.resolve("DAS.L001.00.NOPE")

    def test_wrong_shape_raises(self):
        """Wrong shape raises."""
        with pytest.raises(InvalidInventoryError, match="four"):
            build_inventory().resolve("DAS.L001.RAW")


class TestPathOperations:
    """select / split_at / reverse / concatenation."""

    @pytest.fixture()
    def path(self):
        """Return the example path."""
        return build_inventory().networks[0].fiber_arrays[0].optical_paths[0]

    def test_component_intervals(self, path):
        """Components tile the absolute axis cumulatively."""
        two = inv.OpticalPath(
            start_distance=100.0,
            optical_components=(
                inv.FiberSegment(optical_length=50.0),
                inv.Splice(optical_length=0.5),
            ),
        )
        assert two.component_intervals() == ((100.0, 150.0), (150.0, 150.5))

    def test_select_preserves_absolute_distances(self, path):
        """Select preserves absolute distances."""
        piece = path.select(distance=(50.0, 150.0))
        assert piece.start_distance == 50.0
        assert np.isclose(piece.optical_length, 100.0)
        assert piece.geometry[0].distance[0] == 50.0
        piece.check()

    def test_split_and_rejoin(self, path):
        """Split and rejoin."""
        left, right = path.split_at(100.0)
        assert left.end_distance == right.start_distance == 100.0
        joined = left + right
        assert np.isclose(joined.optical_length, path.optical_length)
        joined.check()

    def test_reverse_involution(self, path):
        """Reverse involution."""
        twice = path.reverse().reverse()
        assert twice.model_dump(mode="json") == path.model_dump(mode="json")

    def test_reverse_rewrites_all_tracks(self, path):
        """Reverse rewrites all tracks."""
        rev = path.reverse()
        # 0-100 annotation on a 0-250 path becomes 150-250.
        assert rev.annotations[0].interval == (150.0, 250.0)
        # 0-200 coupling becomes 50-250.
        assert rev.coupling[0].interval == (50.0, 250.0)
        rev.check()

    def test_empty_selection_raises(self, path):
        """Empty selection raises."""
        with pytest.raises(Exception, match="Empty distance selection"):
            path.select(distance=(300.0, 400.0))


class TestInventory:
    """Whole-inventory behaviors."""

    def test_validate(self):
        """Validate."""
        assert build_inventory().check() is not None

    def test_duplicate_network_codes_raise(self):
        """Duplicate network codes raise."""
        net = inv.Network(code="DAS")
        with pytest.raises(InvalidInventoryError, match="unique"):
            inv.Inventory(networks=(net, net)).check()

    def test_yaml_roundtrip(self, tmp_path):
        """Yaml roundtrip."""
        pytest.importorskip("yaml")
        inventory = build_inventory()
        path = tmp_path / "inventory.yaml"
        inventory.to_yaml(path)
        loaded = inv.Inventory.from_yaml(path)
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_resources_accept_iterable(self):
        """Resources accept iterable."""
        cable = inv.Cable(resource_id="cable_01")
        inventory = inv.Inventory(resources=[cable])
        assert inventory.resources["cable_01"] == cable

    def test_replace_is_a_correction(self):
        """Replace is a correction."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        new = old.new(gauge_length=12.0)
        updated = inventory.replace(old, new)
        acq = updated.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.gauge_length == 12.0
        # Original is untouched (immutability).
        assert inventory.networks[0].fiber_arrays[0].acquisitions[0] == old

    def test_replace_missing_raises(self):
        """Replace missing raises."""
        inventory = build_inventory()
        with pytest.raises(InvalidInventoryError, match="not found"):
            inventory.replace(inv.Network(code="XX"), inv.Network(code="YY"))

    def test_dc_namespace(self, tmp_path):
        """Dc namespace."""
        pytest.importorskip("yaml")
        assert isinstance(dc.inventory(), inv.Inventory)
        path = tmp_path / "inv.yaml"
        build_inventory().to_yaml(path)
        assert isinstance(dc.inventory(path), inv.Inventory)

    def test_crs_vocabulary_enforced(self):
        """Structural column names are outside the coordinate vocabulary."""
        with pytest.raises(ValidationError, match="vocabulary"):
            inv.CoordinateReferenceSystem(coordinate_labels=("x", "distance"))


class TestReviewRegressions:
    """Regression tests for counterpart-review findings."""

    def test_far_future_open_epochs_overlap(self):
        """Ongoing epochs overlap finite epochs beyond any sentinel date."""
        a = inv.OpticalPath(name="a", start_time="2259-01-01")
        b = inv.OpticalPath(name="b", start_time="2261-01-01", end_time="2262-01-01")
        assert a.overlaps(b) and b.overlaps(a)

    def test_reversed_epoch_raises(self):
        """End before start fails at construction."""
        with pytest.raises(ValidationError, match="must be after"):
            inv.Acquisition(code="RAW", start_time="2022-01-01", end_time="2021-01-01")

    def test_replace_type_mismatch_raises(self):
        """Replace type mismatch raises."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        with pytest.raises(InvalidInventoryError, match="does not match"):
            inventory.replace(old, "not-an-acquisition")

    def test_full_select_keeps_terminal_zero_length_component(self):
        """A terminator at the path end survives full selection and splits."""
        path = inv.OpticalPath(
            optical_components=(
                inv.FiberSegment(optical_length=100.0),
                inv.Terminator(optical_length=0.0),
            ),
        )
        full = path.select(distance=(None, None))
        assert len(full.optical_components) == 2
        _, right = path.split_at(50.0)
        assert isinstance(right.optical_components[-1], inv.Terminator)

    def test_resource_key_mismatch_raises(self):
        """Resource key mismatch raises."""
        cable = inv.Cable(resource_id="right")
        with pytest.raises(ValidationError, match="disagrees"):
            inv.Inventory(resources={"wrong": cable})

    def test_duplicate_resource_ids_raise(self):
        """Duplicate resource ids raise."""
        with pytest.raises(ValidationError, match="Duplicate resource_id"):
            inv.Inventory(
                resources=[inv.Cable(resource_id="x"), inv.Cable(resource_id="x")]
            )

    def test_zero_dim_geometry_raises(self):
        """Zero dim geometry raises."""
        with pytest.raises(ValidationError, match="nonzero"):
            inv.Geometry(distance=(0.0, 1.0), coordinates=((), ()))

    def test_station_extra_fields_forbidden(self):
        """Coordinates are canonical (x, y, z); label fields are not stored."""
        with pytest.raises(ValidationError):
            inv.Station(code="VA01", latitude=48.8)


class TestCanonicalAxes:
    """Canonical (x, y, z) storage with CRS-resolved aliases."""

    def test_canonical_axes_always_resolve(self):
        """Canonical axes always resolve."""
        crs = inv.CoordinateReferenceSystem()
        assert [crs.axis_index(x) for x in ("x", "y", "z")] == [0, 1, 2]

    def test_alias_resolves_when_crs_defines_it(self):
        """Alias resolves when crs defines it."""
        crs = inv.CoordinateReferenceSystem()
        assert crs.axis_index("latitude") == 1
        assert crs.axis_index("longitude") == 0

    def test_alias_not_defined_raises(self):
        """An alias absent from the CRS labels raises."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("easting", "northing", "elevation"),
            units=("meter", "meter", "meter"),
        )
        assert crs.axis_index("easting") == 0
        with pytest.raises(InvalidInventoryError, match="not defined"):
            crs.axis_index("latitude")

    def test_labels_outside_vocabulary_raise(self):
        """Labels outside vocabulary raise."""
        with pytest.raises(ValidationError, match="vocabulary"):
            inv.CoordinateReferenceSystem(coordinate_labels=("depth", "along"))

    def test_two_axis_crs_has_no_z(self):
        """Two axis crs has no z."""
        crs = inv.CoordinateReferenceSystem(
            coordinate_labels=("x", "y"), units=("meter", "meter")
        )
        with pytest.raises(InvalidInventoryError, match="no 'z' axis"):
            crs.axis_index("z")


class TestResourcePool:
    """Shareable resources normalize into the flat pool with id references."""

    @staticmethod
    def _inventory_with(component, **acq_kwargs):
        """Wrap a component (and optional acquisition) into an inventory."""
        acq = (inv.Acquisition(code="RAW", **acq_kwargs),) if acq_kwargs else ()
        path = inv.OpticalPath(optical_components=(component,))
        array = inv.FiberArray(code="L001", acquisitions=acq, optical_paths=(path,))
        return inv.Inventory(networks=(inv.Network(code="DAS", fiber_arrays=(array,)),))

    def test_inline_resource_normalizes_to_pool(self):
        """An inline cable moves to the pool; the field keeps its id."""
        cable = inv.Cable(resource_id="cable-01", name="c")
        seg = inv.FiberSegment(optical_length=100.0, container=cable)
        inventory = self._inventory_with(seg)
        stored = (
            inventory.networks[0].fiber_arrays[0].optical_paths[0].optical_components[0]
        )
        assert stored.container == "cable-01"
        assert inventory.get_resource("cable-01") == cable

    def test_shared_resource_registered_once(self):
        """Two components sharing one enclosure yield one pool entry."""
        coupler = inv.Enclosure(resource_id="coupler-01")
        path = inv.OpticalPath(
            optical_components=(
                inv.Connector(container=coupler),
                inv.Connector(container=coupler),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        assert list(inventory.resources) == ["coupler-01"]

    def test_conflicting_inline_definitions_raise(self):
        """One id, two different contents: agree-or-raise."""
        a = inv.Cable(resource_id="cable-01", fiber_count=1)
        b = inv.Cable(resource_id="cable-01", fiber_count=4)
        path = inv.OpticalPath(
            optical_components=(
                inv.FiberSegment(optical_length=1.0, container=a),
                inv.FiberSegment(optical_length=1.0, container=b),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        with pytest.raises(ValidationError, match="defined twice"):
            inv.Inventory(networks=(inv.Network(code="DAS", fiber_arrays=(array,)),))

    def test_dangling_reference_raises(self):
        """Dangling reference raises."""
        seg = inv.FiberSegment(optical_length=1.0, container="no-such-cable")
        with pytest.raises(ValidationError, match="Dangling"):
            self._inventory_with(seg)

    def test_nested_resources_normalize(self):
        """A cable inside a pipe: both land in the pool, linked by id."""
        pipe = inv.Enclosure(resource_id="pipe-01", enclosure_type="pipe")
        cable = inv.Cable(resource_id="cable-01", container=pipe)
        seg = inv.FiberSegment(optical_length=1.0, container=cable)
        inventory = self._inventory_with(seg)
        assert inventory.get_resource("cable-01").container == "pipe-01"
        assert inventory.get_resource("pipe-01") == pipe

    def test_interrogator_normalizes(self):
        """Interrogator normalizes."""
        unit = inv.Interrogator(resource_id="int-01", model="DAS-1000")
        seg = inv.FiberSegment(optical_length=1.0)
        inventory = self._inventory_with(seg, interrogator=unit)
        acq = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.interrogator == "int-01"
        assert inventory.get_resource("int-01") == unit

    def test_resource_correction_is_single_site(self):
        """Replacing a pooled resource touches only the pool."""
        cable = inv.Cable(resource_id="cable-01", fiber_count=1)
        seg = inv.FiberSegment(optical_length=1.0, container=cable)
        inventory = self._inventory_with(seg)
        fixed = cable.new(fiber_count=4)
        updated = inventory.replace(cable, fixed)
        assert updated.get_resource("cable-01").fiber_count == 4
        stored = (
            updated.networks[0].fiber_arrays[0].optical_paths[0].optical_components[0]
        )
        assert stored.container == "cable-01"

    def test_resource_correction_must_keep_id(self):
        """Resource correction must keep id."""
        cable = inv.Cable(resource_id="cable-01")
        seg = inv.FiberSegment(optical_length=1.0, container=cable)
        inventory = self._inventory_with(seg)
        renamed = cable.new(resource_id="cable-02")
        with pytest.raises(InvalidInventoryError, match="same resource_id"):
            inventory.replace(cable, renamed)

    def test_yaml_roundtrip_stays_flat(self, tmp_path):
        """Serialized form holds ids, not inline copies, and round-trips."""
        pytest.importorskip("yaml")
        cable = inv.Cable(resource_id="cable-01", name="c")
        seg = inv.FiberSegment(optical_length=100.0, container=cable)
        inventory = self._inventory_with(seg)
        text = inventory.to_yaml()
        assert text.count("cable-01") >= 2
        loaded = inv.Inventory.from_yaml(text)
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_get_resource_missing_raises(self):
        """Get resource missing raises."""
        with pytest.raises(InvalidInventoryError, match="No resource"):
            build_inventory().get_resource("nope")


class TestInternalReviewRegressions:
    """Regression tests for internal adversarial/convention review findings."""

    def test_new_preserves_union_discriminators(self):
        """new() works on models holding discriminated unions."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=10.0),)
        )
        renamed = path.new(name="renamed")
        assert renamed.name == "renamed"
        assert isinstance(renamed.optical_components[0], inv.FiberSegment)

    def test_new_preserves_normalized_pool(self):
        """new() keeps resources that normalization moved into the pool."""
        acq = inv.Acquisition(
            code="RAW", interrogator=inv.Interrogator(resource_id="int-1")
        )
        array = inv.FiberArray(code="L001", acquisitions=(acq,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        updated = inventory.new(resource_id="doc-1")
        assert updated.get_resource("int-1").resource_id == "int-1"

    def test_equality_with_nested_nat(self):
        """Structurally identical trees with unset times compare equal."""

        def make():
            return inv.FiberArray(code="F", acquisitions=(inv.Acquisition(code="A"),))

        assert make() == make()

    def test_replace_finds_rebuilt_handle(self):
        """replace() matches an equal-but-rebuilt object, not just identity."""
        inventory = build_inventory()
        old = build_inventory().networks[0].fiber_arrays[0].acquisitions[0]
        updated = inventory.replace(old, old.new(gauge_length=99.0))
        acq = updated.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.gauge_length == 99.0

    def test_replace_normalizes_inline_resources(self):
        """A replacement carrying an inline resource gets normalized."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        new = old.new(interrogator=inv.Interrogator(resource_id="int-9"))
        updated = inventory.replace(old, new)
        assert updated.get_resource("int-9").resource_id == "int-9"
        acq = updated.networks[0].fiber_arrays[0].acquisitions[0]
        assert acq.interrogator == "int-9"

    def test_replace_rejects_dangling_reference(self):
        """Replace rejects dangling reference."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        with pytest.raises(ValidationError, match="Dangling"):
            inventory.replace(old, old.new(interrogator="no-such-id"))

    def test_replace_finds_channel(self):
        """Replace finds channel."""
        chan = inv.Channel(code="HHZ")
        station = inv.Station(code="S1", channels=(chan,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", stations=(station,)),)
        )
        updated = inventory.replace(chan, chan.new(data_type="strain"))
        got = updated.networks[0].stations[0].channels[0]
        assert got.data_type == "strain"

    def test_keyless_dict_resource_adopts_key(self):
        """A dict resource without resource_id adopts its pool key."""
        pytest.importorskip("yaml")
        inventory = inv.Inventory(
            resources={"cab-1": {"type": "Cable", "name": "mycable"}}
        )
        assert inventory.get_resource("cab-1").resource_id == "cab-1"
        loaded = inv.Inventory.from_yaml(inventory.to_yaml())
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_duplicate_array_codes_raise(self):
        """Same-code overlapping fiber arrays fail network check."""
        net = inv.Network(
            code="DAS",
            fiber_arrays=(
                inv.FiberArray(code="L001", start_time="2020-01-01"),
                inv.FiberArray(code="L001", start_time="2021-01-01"),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Duplicate fiber array"):
            net.check()

    def test_mixed_dimensionality_geometry_raises(self):
        """Segments with different coordinate dims fail the path check."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(
                    distance=(0.0, 10.0), coordinates=((0.0, 0.0), (1.0, 1.0))
                ),
                inv.Geometry(
                    distance=(20.0, 30.0),
                    coordinates=((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)),
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="dimensionalities"):
            path.check()

    def test_inventory_function_dispatch(self):
        """dc.inventory handles bad input with clear errors."""
        with pytest.raises(InvalidInventoryError, match="No such inventory file"):
            dc.inventory("does_not_exist.yaml")
        with pytest.raises(InvalidInventoryError, match="Could not get"):
            dc.inventory(123)
        existing = build_inventory()
        assert dc.inventory(existing) is existing

    def test_core_namespace_export(self):
        """Core namespace export."""
        assert Inventory is inv.Inventory


class TestCodexReviewRegressions:
    """Regression tests for counterpart (Codex) review findings."""

    def test_replace_reaches_path_track_items(self):
        """A recalibrated geometry segment is replaceable in place."""
        inventory = build_inventory()
        old = inventory.networks[0].fiber_arrays[0].optical_paths[0].geometry[0]
        new = old.new(name="recalibrated")
        updated = inventory.replace(old, new)
        got = updated.networks[0].fiber_arrays[0].optical_paths[0].geometry[0]
        assert got.name == "recalibrated"

    def test_reference_type_mismatch_raises(self):
        """A string ref resolving to the wrong resource type raises."""
        cable = inv.Cable(resource_id="cab-1")
        acq = inv.Acquisition(code="RAW", interrogator="cab-1")
        array = inv.FiberArray(code="L001", acquisitions=(acq,))
        with pytest.raises(ValidationError, match="expected one of"):
            inv.Inventory(
                resources=[cable],
                networks=(inv.Network(code="DAS", fiber_arrays=(array,)),),
            )

    def test_nested_null_not_equal_to_value(self):
        """A nested null array is not equal to a non-null one."""
        a = dc.PatchAttrs(foo={"x": np.array([np.nan])})
        b = dc.PatchAttrs(foo={"x": np.array([1.0])})
        assert a != b

    def test_nonfinite_interval_values_raise(self):
        """Nonfinite interval values raise."""
        with pytest.raises(ValidationError):
            inv.CouplingCondition(
                start_distance=np.nan, end_distance=10.0, coupling_type="trench"
            )
        with pytest.raises(ValidationError, match="finite"):
            inv.Geometry(distance=(0.0, np.inf), coordinates=((0.0, 0.0), (1.0, 1.0)))
        with pytest.raises(ValidationError, match="finite"):
            inv.DistanceMap(channel=(0.0, np.inf), distance=(0.0, 1.0))


class TestCoverageCompleteness:
    """Exercise remaining branches so the patch stays fully covered."""

    def test_duplicate_coordinate_labels_raise(self):
        """Duplicate coordinate labels raise."""
        with pytest.raises(ValidationError, match="unique"):
            inv.CoordinateReferenceSystem(coordinate_labels=("x", "x"))

    def test_distance_map_length_mismatch_raises(self):
        """Distance map length mismatch raises."""
        with pytest.raises(ValidationError, match="same length"):
            inv.DistanceMap(channel=(1.0, 2.0), distance=(5.0,))

    def test_distance_map_empty_raises(self):
        """Distance map empty raises."""
        with pytest.raises(ValidationError, match="at least one"):
            inv.DistanceMap(channel=(), distance=())

    def test_distance_map_non_increasing_raises(self):
        """Distance map non increasing raises."""
        with pytest.raises(ValidationError, match="input values"):
            inv.DistanceMap(channel=(2.0, 1.0), distance=(5.0, 6.0))
        with pytest.raises(ValidationError, match="distance values"):
            inv.DistanceMap(channel=(1.0, 2.0), distance=(6.0, 5.0))

    def test_coordinates_at_without_geometry(self):
        """Coordinates at without geometry."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=10.0),)
        )
        assert np.all(np.isnan(path.coordinates_at([5.0])))

    def test_select_drops_out_of_range_geometry(self):
        """Selection drops segments entirely outside the clip."""
        seg = dict(coordinates=((0.0, 0.0), (1.0, 1.0)))
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            geometry=(
                inv.Geometry(distance=(0.0, 20.0), **seg),
                inv.Geometry(distance=(80.0, 100.0), **seg),
            ),
        )
        piece = path.select(distance=(40.0, 60.0))
        assert piece.geometry == ()

    def test_add_rejects_non_path(self):
        """Add rejects non path."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=10.0),)
        )
        with pytest.raises(TypeError):
            _ = path + 5
        with pytest.raises(TypeError):
            _ = 5 + path

    def test_resolve_multiple_paths_raises(self):
        """An unchecked inventory with concurrent paths fails resolve."""
        array = inv.FiberArray(
            code="L001",
            acquisitions=(inv.Acquisition(code="RAW"),),
            optical_paths=(
                inv.OpticalPath(name="a", start_time="2020-01-01"),
                inv.OpticalPath(name="b", start_time="2021-01-01"),
            ),
        )
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        with pytest.raises(InvalidInventoryError, match="2 optical paths"):
            inventory.resolve("DAS.L001..RAW", time="2022-01-01")

    def test_replace_network_and_station_and_path(self):
        """Replace works at network, station, and path levels."""
        station = inv.Station(code="S1")
        path = inv.OpticalPath(name="p", start_time="2020-01-01")
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        net = inv.Network(code="DAS", fiber_arrays=(array,), stations=(station,))
        inventory = inv.Inventory(networks=(net,))
        got = inventory.replace(station, station.new(name="renamed"))
        assert got.networks[0].stations[0].name == "renamed"
        got = inventory.replace(path, path.new(name="renamed"))
        assert got.networks[0].fiber_arrays[0].optical_paths[0].name == "renamed"
        got = inventory.replace(net, net.new(name="renamed"))
        assert got.networks[0].name == "renamed"

    def test_replace_missing_resource_raises(self):
        """Replace missing resource raises."""
        inventory = build_inventory()
        cable = inv.Cable(resource_id="ghost")
        with pytest.raises(InvalidInventoryError, match="not found"):
            inventory.replace(cable, cable.new(name="x"))

    def test_from_yaml_non_mapping_raises(self):
        """From yaml non mapping raises."""
        pytest.importorskip("yaml")
        with pytest.raises(InvalidInventoryError, match="mapping"):
            inv.Inventory.from_yaml("- 1\n- 2\n")

    def test_model_equality_foreign_types(self):
        """Models compare False to unrelated types and True to equal dumps."""
        acq = inv.Acquisition(code="RAW")
        assert acq != 5
        assert acq == acq.model_dump()

    def test_equality_array_shape_mismatch(self):
        """Nested arrays of different shapes compare unequal."""
        a = dc.PatchAttrs(foo={"x": np.array([1.0, 2.0])})
        b = dc.PatchAttrs(foo={"x": np.array([1.0])})
        assert a != b

    def test_equality_sequence_length_mismatch(self):
        """Nested sequences of different lengths compare unequal."""
        a = inv.Network(code="DAS", fiber_arrays=(inv.FiberArray(code="A"),))
        b = inv.Network(code="DAS")
        assert a != b

    def test_replace_fiber_array(self):
        """Replace fiber array."""
        inventory = build_inventory()
        array = inventory.networks[0].fiber_arrays[0]
        got = inventory.replace(array, array.new(name="renamed"))
        assert got.networks[0].fiber_arrays[0].name == "renamed"

    def test_values_equal_branches(self):
        """Null-pattern and key mismatches compare unequal."""
        assert not _values_equal(np.array([np.nan]), np.array([1.0]))
        assert _values_equal(np.array([1.0, np.nan]), np.array([1.0, np.nan]))
        assert not _values_equal({"a": 1}, {"b": 1})
        assert _values_equal((1.0, np.nan), (1.0, np.nan))


class TestTypeTag:
    """The serialization type tag is invisible to users."""

    def test_hidden_from_repr(self):
        """Hidden from repr."""
        cable = inv.Cable(resource_id="c1", name="c")
        assert "type" not in repr(cable)

    def test_present_in_dump(self):
        """Present in dump."""
        assert inv.Cable(resource_id="c1").model_dump()["type"] == "Cable"

    def test_wrong_tag_rejected(self):
        """Wrong tag rejected."""
        with pytest.raises(ValidationError):
            inv.Cable(resource_id="c1", type="Enclosure")


class TestUniformAttachments:
    """Every inventory object carries description and extra_fields."""

    def test_description_on_previous_gaps(self):
        """Descriptions on previous gaps."""
        acq = inv.Acquisition(code="RAW", description="tap tested twice")
        path = inv.OpticalPath(description="post-repair epoch")
        assert acq.description and path.description

    def test_yaml_omits_empty_fields(self):
        """Empty strings, dicts, and tuples do not serialize."""
        pytest.importorskip("yaml")
        inventory = build_inventory()
        text = inventory.to_yaml()
        assert "description:" not in text
        assert "extra_fields:" not in text
        loaded = inv.Inventory.from_yaml(text)
        assert loaded.model_dump(mode="json") == inventory.model_dump(mode="json")

    def test_extra_fields_contents_survive(self):
        """User values inside extra_fields are kept verbatim, even empty."""
        pytest.importorskip("yaml")
        acq = inv.Acquisition(code="RAW", extra_fields={"vendor_flag": ""})
        array = inv.FiberArray(code="L001", acquisitions=(acq,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        loaded = inv.Inventory.from_yaml(inventory.to_yaml())
        got = loaded.networks[0].fiber_arrays[0].acquisitions[0]
        assert got.extra_fields == {"vendor_flag": ""}


class TestStationXmlAlignment:
    """Fields added for FDSN StationXML import fidelity."""

    def test_network_epochs(self):
        """Networks carry validity epochs like every other container."""
        net = inv.Network(code="XX", start_time="2020-01-01", end_time="2022-01-01")
        assert net.is_effective_at("2021-06-01")
        assert not net.is_effective_at("2022-01-01")

    def test_identifiers_on_citable_levels(self):
        """Network, FiberArray, and Station carry citation identifiers."""
        doi = "doi:10.7914/SN/XX"
        net = inv.Network(code="XX", identifiers=(doi,))
        array = inv.FiberArray(code="L001", identifiers=(doi,))
        station = inv.Station(code="VA01", identifiers=(doi,))
        assert net.identifiers == array.identifiers == station.identifiers == (doi,)

    def test_channel_orientation_and_depth(self):
        """Channels hold azimuth, dip, and burial depth."""
        chan = inv.Channel(code="BHN", azimuth=0.0, dip=0.0, depth=100.0)
        assert (chan.azimuth, chan.dip, chan.depth) == (0.0, 0.0, 100.0)


class TestPointMarkers:
    """Zero-length intervals are point markers."""

    def test_aerial_coupling_type(self):
        """Aerial coupling type."""
        cond = inv.CouplingCondition(
            start_distance=0.0, end_distance=100.0, coupling_type="aerial"
        )
        assert cond.coupling_type == "aerial"

    def test_point_clamp_inside_span_is_legal(self):
        """A point marker inside a covered span does not count as overlap."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    start_distance=0.0, end_distance=80.0, coupling_type="trench"
                ),
                inv.CouplingCondition(
                    start_distance=40.0,
                    end_distance=40.0,
                    coupling_type="other",
                    description="clamp point",
                ),
            ),
        )
        assert path.check() is path

    def test_point_annotation(self):
        """Point annotation."""
        anno = inv.OpticalPathAnnotation(
            start_distance=350.0, end_distance=350.0, label="wellhead"
        )
        assert anno.interval == (350.0, 350.0)

    def test_reversed_interval_rejected(self):
        """An end before the start is rejected."""
        with pytest.raises(ValidationError, match="must not precede"):
            inv.CouplingCondition(
                start_distance=10.0, end_distance=5.0, coupling_type="trench"
            )


class TestOpticalLoss:
    """Unified loss/reflectance with measurement provenance."""

    def test_scalar_loss_no_provenance(self):
        """Plain numbers with no measurement records are legal."""
        splice = inv.Splice(loss_db=0.08, reflectance_db=-55.0)
        assert splice.loss_db == 0.08

    def test_shared_measurement_pooled_once(self):
        """One OTDR run backs many components through one pool entry."""
        run = inv.OpticalMeasurement(
            resource_id="otdr-1",
            method="otdr",
            wavelength=1550.0,
            pulse_width=1e-8,
            direction="forward",
        )
        path = inv.OpticalPath(
            optical_components=(
                inv.FiberSegment(
                    optical_length=1000.0, loss_db=0.3, loss_measurement=run
                ),
                inv.Splice(
                    loss_db=0.05,
                    loss_measurement=run,
                    reflectance_db=-60.0,
                    reflectance_measurement=run,
                ),
            ),
        )
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        assert list(inventory.resources) == ["otdr-1"]
        comps = (
            inventory.networks[0].fiber_arrays[0].optical_paths[0].optical_components
        )
        assert comps[0].loss_measurement == "otdr-1"
        assert comps[1].reflectance_measurement == "otdr-1"

    def test_multi_wavelength_pairs(self):
        """Tuple losses pair elementwise with their measurements."""
        sheet_1550 = inv.OpticalMeasurement(
            resource_id="ds-1550", method="datasheet", wavelength=1550.0
        )
        sheet_1310 = inv.OpticalMeasurement(
            resource_id="ds-1310", method="datasheet", wavelength=1310.0
        )
        seg = inv.FiberSegment(
            optical_length=2000.0,
            loss_db=(0.6, 0.7),
            loss_measurement=(sheet_1550, sheet_1310),
        )
        assert seg.attenuation_db_per_km == (0.3, 0.35)

    def test_tuple_loss_requires_tuple_measurements(self):
        """Tuple loss requires tuple measurements."""
        with pytest.raises(ValidationError, match="equal-length"):
            inv.FiberSegment(optical_length=10.0, loss_db=(0.1, 0.2))

    def test_length_mismatch_raises(self):
        """Length mismatch raises."""
        with pytest.raises(ValidationError, match="has 2 values"):
            inv.FiberSegment(
                optical_length=10.0,
                loss_db=(0.1, 0.2),
                loss_measurement=("m1", "m2", "m3"),
            )

    def test_dangling_measurement_ref_raises(self):
        """Dangling measurement ref raises."""
        seg = inv.FiberSegment(
            optical_length=10.0, loss_db=0.1, loss_measurement="no-such-run"
        )
        path = inv.OpticalPath(optical_components=(seg,))
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        with pytest.raises(ValidationError, match="Dangling"):
            inv.Inventory(networks=(inv.Network(code="DAS", fiber_arrays=(array,)),))

    def test_measurement_data_file_normalizes(self):
        """A measurement's trace-file resource lands in the pool."""
        trace = inv.ExternalResource(resource_id="trace-1", uri="file://x.sor")
        run = inv.OpticalMeasurement(resource_id="otdr-1", data=trace)
        inventory = inv.Inventory(resources=[run])
        assert inventory.get_resource("otdr-1").data == "trace-1"
        assert inventory.get_resource("trace-1").uri == "file://x.sor"

    def test_path_measurements_normalize(self):
        """Path-level measurement refs land in the pool."""
        run = inv.OpticalMeasurement(resource_id="otdr-1", method="otdr")
        path = inv.OpticalPath(measurements=(run,))
        array = inv.FiberArray(code="L001", optical_paths=(path,))
        inventory = inv.Inventory(
            networks=(inv.Network(code="DAS", fiber_arrays=(array,)),)
        )
        got = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        assert got.measurements == ("otdr-1",)


class TestCrsShape:
    """Units pair with axes; WKT carries non-registry definitions."""

    def test_default_units_match_default_labels(self):
        """The default CRS has honest per-axis units."""
        crs = inv.CoordinateReferenceSystem()
        assert crs.units == ("degree", "degree", "meter")

    def test_units_length_mismatch_raises(self):
        """Units length mismatch raises."""
        with pytest.raises(ValidationError, match="one entry per"):
            inv.CoordinateReferenceSystem(units=("degree",))

    def test_local_frame_with_wkt(self):
        """Local frame with wkt."""
        crs = inv.CoordinateReferenceSystem(
            authority="LOCAL",
            code="mine-grid-7",
            coordinate_labels=("x", "y", "z"),
            units=("meter", "meter", "meter"),
            wkt='ENGCRS["mine grid",EDATUM["portal"],CS[Cartesian,3]]',
        )
        assert crs.wkt.startswith("ENGCRS")
