"""Tests for the DASDAE inventory model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

import dascore as dc
from dascore.core import inventory as inv
from dascore.exceptions import InvalidInventoryError

SPEC_PATH = Path(__file__).parent / "dasdae_inventory_spec.yml"

# Implementation-only fields allowed beyond the spec (serialization/discrimination).
ALLOWED_EXTRA_FIELDS = {"type"}
# Spec nodes not implemented as public classes (internal/diagram-only nodes).
SKIPPED_SPEC_NODES = {"References"}
# Union nodes: spec base checked against a representative member class.
UNION_NODES = {"OpticalComponent": "FiberSegment"}


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
                distance=0.0, optical_length=200.0, coupling_type="trench"
            ),
        ),
        annotations=(
            inv.OpticalPathAnnotation(
                distance=0.0, optical_length=100.0, label="east"
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


class TestSpecConformance:
    """The implementation must carry every field the spec declares."""

    spec = yaml.safe_load(SPEC_PATH.read_text())

    @pytest.mark.parametrize("node_id", list(spec["nodes"]))
    def test_fields_match_spec(self, node_id):
        """Each spec attribute exists as a field or property."""
        node = self.spec["nodes"][node_id]
        label = node.get("label", node_id)
        if label in SKIPPED_SPEC_NODES:
            pytest.skip(f"{label} is not a public class")
        cls = getattr(inv, UNION_NODES.get(label, label))
        implemented = set(cls.model_fields)
        implemented |= {
            name for name in dir(cls) if isinstance(getattr(cls, name, None), property)
        }
        spec_fields = {
            attr["name"]
            for attr in node.get("attributes", [])
            if not attr["name"].startswith("<")
        }
        missing = spec_fields - implemented
        assert not missing, f"{label} missing spec fields: {sorted(missing)}"

    @pytest.mark.parametrize("node_id", list(spec["nodes"]))
    def test_no_undeclared_fields(self, node_id):
        """Model fields beyond the spec are limited to a known allowlist."""
        node = self.spec["nodes"][node_id]
        label = node.get("label", node_id)
        if label in SKIPPED_SPEC_NODES or label in UNION_NODES:
            pytest.skip(f"{label} is not a plain public class")
        cls = getattr(inv, label)
        spec_fields = {a["name"] for a in node.get("attributes", [])}
        extras = set(cls.model_fields) - spec_fields - ALLOWED_EXTRA_FIELDS
        assert not extras, f"{label} has undeclared fields: {sorted(extras)}"


class TestGeometry:
    """Geometry segment rules."""

    def test_requires_two_points(self):
        """Requires two points."""
        with pytest.raises(ValidationError, match="at least two"):
            inv.Geometry(distance=(1.0,), coordinates=((0.0, 0.0),))

    def test_strictly_increasing(self):
        """Strictly increasing."""
        with pytest.raises(ValidationError, match="strictly increasing"):
            inv.Geometry(
                distance=(1.0, 1.0), coordinates=((0.0, 0.0), (1.0, 1.0))
            )

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
        geo = inv.Geometry(
            distance=(10.0, 20.0), coordinates=((0.0, 0.0), (1.0, 1.0))
        )
        out = geo.interpolate([5.0, 25.0])
        assert np.all(np.isnan(out))


class TestPathTracks:
    """Track-kind rules: tiling, function tracks, set track."""

    def test_partial_coverage_is_legal(self):
        """Partial coverage is legal."""
        path = build_inventory().networks[0].fiber_arrays[0].optical_paths[0]
        assert path.validate() is path

    def test_coupling_overlap_raises(self):
        """Coupling overlap raises."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    distance=0.0, optical_length=60.0, coupling_type="trench"
                ),
                inv.CouplingCondition(
                    distance=50.0, optical_length=20.0, coupling_type="conduit"
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="Overlapping coupling"):
            path.validate()

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
            path.validate()

    def test_annotations_overlap_freely(self):
        """Annotations overlap freely."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            annotations=(
                inv.OpticalPathAnnotation(
                    distance=0.0, optical_length=60.0, label="a"
                ),
                inv.OpticalPathAnnotation(
                    distance=50.0, optical_length=20.0, label="b"
                ),
            ),
        )
        assert path.validate() is path

    def test_out_of_bounds_raises(self):
        """Out of bounds raises."""
        path = inv.OpticalPath(
            optical_components=(inv.FiberSegment(optical_length=100.0),),
            coupling=(
                inv.CouplingCondition(
                    distance=90.0, optical_length=60.0, coupling_type="trench"
                ),
            ),
        )
        with pytest.raises(InvalidInventoryError, match="extends past"):
            path.validate()

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
            inv.DistanceMap(
                channel=(1.0,), instrument_distance=(1.0,), distance=(5.0,)
            )
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
            array.validate()

    def test_concurrent_paths_different_locations_legal(self):
        """Location-scoped paths: similar co-located fibers share an array."""
        array = inv.FiberArray(
            code="L001",
            optical_paths=(
                inv.OpticalPath(name="a", location_code="00", start_time="2020-01-01"),
                inv.OpticalPath(name="b", location_code="01", start_time="2020-01-01"),
            ),
        )
        assert array.validate() is array

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
        assert array.validate() is array

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
            array.validate()

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
        assert array.validate() is array

    def test_station_fiber_code_collision_raises(self):
        """Station fiber code collision raises."""
        network = inv.Network(
            code="DAS",
            fiber_arrays=(inv.FiberArray(code="L001", start_time="2020-01-01"),),
            stations=(inv.Station(code="L001", start_time="2020-06-01"),),
        )
        with pytest.raises(InvalidInventoryError, match="share code"):
            network.validate()


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
                inv.Acquisition(
                    code="RAW", start_time="2021-01-01", gauge_length=20.0
                ),
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
        array = inv.FiberArray(
            code="L001", acquisitions=(inv.Acquisition(code="RAW"),)
        )
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

    def test_select_preserves_absolute_distances(self, path):
        """Select preserves absolute distances."""
        piece = path.select(distance=(50.0, 150.0))
        assert piece.start_distance == 50.0
        assert np.isclose(piece.optical_length, 100.0)
        assert piece.geometry[0].distance[0] == 50.0
        piece.validate()

    def test_split_and_rejoin(self, path):
        """Split and rejoin."""
        left, right = path.split_at(100.0)
        assert left.end_distance == right.start_distance == 100.0
        joined = left + right
        assert np.isclose(joined.optical_length, path.optical_length)
        joined.validate()

    def test_reverse_involution(self, path):
        """Reverse involution."""
        twice = path.reverse().reverse()
        assert twice.model_dump(mode="json") == path.model_dump(mode="json")

    def test_reverse_rewrites_all_tracks(self, path):
        """Reverse rewrites all tracks."""
        rev = path.reverse()
        # 0-100 annotation on a 0-250 path becomes 150-250.
        assert np.isclose(rev.annotations[0].distance, 150.0)
        # 0-200 coupling becomes 50-250.
        assert np.isclose(rev.coupling[0].distance, 50.0)
        rev.validate()

    def test_empty_selection_raises(self, path):
        """Empty selection raises."""
        with pytest.raises(Exception, match="Empty distance selection"):
            path.select(distance=(300.0, 400.0))


class TestInventory:
    """Whole-inventory behaviors."""

    def test_validate(self):
        """Validate."""
        assert build_inventory().validate() is not None

    def test_duplicate_network_codes_raise(self):
        """Duplicate network codes raise."""
        net = inv.Network(code="DAS")
        with pytest.raises(InvalidInventoryError, match="unique"):
            inv.Inventory(networks=(net, net)).validate()

    def test_yaml_roundtrip(self, tmp_path):
        """Yaml roundtrip."""
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
        assert isinstance(dc.inventory(), inv.Inventory)
        path = tmp_path / "inv.yaml"
        build_inventory().to_yaml(path)
        assert isinstance(dc.inventory(path), inv.Inventory)

    def test_crs_reserved_labels_raise(self):
        """Crs reserved labels raise."""
        with pytest.raises(ValidationError, match="reserved"):
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
            inv.Acquisition(
                code="RAW", start_time="2022-01-01", end_time="2021-01-01"
            )

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

    def test_dynamic_coordinate_fields(self):
        """Stations accept CRS-label-named fields per the spec."""
        crs = inv.CoordinateReferenceSystem()
        station = inv.Station(
            code="VA01", longitude=2.1, latitude=48.8, elevation=115.0
        )
        assert station.get_coordinates(crs) == (2.1, 48.8, 115.0)

    def test_dynamic_fields_disagreeing_with_tuple_raise(self):
        """Two spellings of coordinates must agree."""
        crs = inv.CoordinateReferenceSystem()
        station = inv.Station(
            code="VA01",
            coordinates=(2.1, 48.8, 115.0),
            longitude=9.9, latitude=48.8, elevation=115.0,
        )
        with pytest.raises(InvalidInventoryError, match="disagree"):
            station.get_coordinates(crs)

    def test_unknown_dynamic_fields_raise(self):
        """Unknown dynamic fields raise."""
        crs = inv.CoordinateReferenceSystem()
        station = inv.Station(code="VA01", northing=1.0)
        with pytest.raises(InvalidInventoryError, match="do not match CRS"):
            station.get_coordinates(crs)
