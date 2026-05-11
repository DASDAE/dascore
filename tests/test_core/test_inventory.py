"""Tests for inventory objects and manifests."""

from __future__ import annotations

import io
import re

import pytest

from dascore.config import set_config
from dascore.core.inventory import (
    Acquisition,
    Cable,
    Connector,
    CoordinateReferenceSystem,
    CouplingCondition,
    CreationInfo,
    Enclosure,
    ExternalResource,
    FiberArray,
    FiberSegment,
    Geometry,
    Interrogator,
    Inventory,
    Network,
    OpticalPath,
    OpticalPathAnnotation,
    Splice,
)
from dascore.exceptions import ParameterError


def make_inventory() -> Inventory:
    """Return a small resolved inventory graph."""
    interrogator = Interrogator(
        resource_id="interrogator_1",
        model="DAS",
        serial_number="SN001",
    )
    configuration = Acquisition(
        code="RAW",
        location_code="00",
        data_category="DAS",
        data_type="strain_rate",
        data_units="m/s",
        interrogator=interrogator,
        acquisition_sample_rate=100.0,
        pulse_width=1.0e-8,
        comment="Initial acquisition.",
    )
    crs = CoordinateReferenceSystem(
        resource_id="crs_1",
        authority="LOCAL",
        code="test_axis",
        axis_order=("x",),
        units="m",
    )
    geometry = Geometry(
        optical_length=2.0,
        geometry_type="linear",
        coordinates=((0.0,), (2.0,)),
    )
    fiber = FiberSegment(optical_length=2.0)
    annotation = OpticalPathAnnotation(distance=(0.0, 2.0), label="well")
    path = OpticalPath(
        optical_length=2.0,
        coordinate_reference_system=crs,
        optical_components=(fiber,),
        geometries=(geometry,),
        annotations=(annotation,),
    )
    fiber_array = FiberArray(
        code="FA",
        acquisitions=(configuration,),
        optical_paths=(path,),
        tag="raw",
    )
    network = Network(
        code="XX",
        description="Test network",
        fiber_arrays=(fiber_array,),
    )
    return Inventory(resource_id="inv_1", networks=(network,))


def get_indexed(inventory, item_type):
    """Return indexed item mappings of one type."""
    type_name = item_type if isinstance(item_type, str) else item_type.__name__
    return tuple(item for item in inventory._indexed_items if item["type"] == type_name)


def get_path(inventory):
    """Return the first optical path in the test inventory."""
    return inventory.networks[0].fiber_arrays[0].optical_paths[0]


class TestInventoryObjects:
    """Tests for canonical storage and resolved object views."""

    def test_stores_runtime_ids_but_not_public_ids_for_internal_objects(self):
        """Internal topology/interval objects should use runtime ids."""
        inventory = make_inventory()
        items = inventory._indexed_items
        network = get_indexed(inventory, Network)[0]
        fiber_array = get_indexed(inventory, FiberArray)[0]
        config = get_indexed(inventory, Acquisition)[0]
        geometry = get_indexed(inventory, Geometry)[0]
        interrogator = get_indexed(inventory, Interrogator)[0]
        path = get_indexed(inventory, OpticalPath)[0]

        assert len(items) == 9
        assert network["id"] == "/network/XX"
        assert network["fiber_array_ids"] == (fiber_array["id"],)
        assert fiber_array["acquisition_ids"] == (config["id"],)
        assert fiber_array["optical_path_ids"] == (path["id"],)
        assert path["geometry_ids"] == (geometry["id"],)
        assert "resource_id" not in network
        assert "resource_id" not in fiber_array
        assert "resource_id" not in geometry
        assert interrogator["resource_id"] == "interrogator_1"
        assert path["coordinate_reference_system_id"] == "crs_1"
        assert config["interrogator_resource_id"] == "interrogator_1"

    def test_no_public_item_collection_property(self):
        """The old item collection property should not be public."""
        inventory = make_inventory()

        assert not hasattr(inventory, "items")

    def test_names_are_display_metadata(self):
        """Names should round-trip without replacing codes or runtime ids."""
        path = OpticalPath(name="Main trench path")
        geometry = Geometry(name="East trench geometry")
        fiber_array = FiberArray(
            code="FORGE",
            name="North trench array",
            optical_paths=(path.revise(geometries=(geometry,)),),
        )
        network = Network(
            code="UU",
            name="Utah FORGE DAS",
            fiber_arrays=(fiber_array,),
        )

        inventory = Inventory(networks=(network,))
        item = get_indexed(inventory, Network)[0]
        resolved = inventory.networks[0]

        assert item["code"] == "UU"
        assert item["name"] == "Utah FORGE DAS"
        assert resolved.fiber_arrays[0].name == "North trench array"
        assert resolved.fiber_arrays[0].optical_paths[0].name == "Main trench path"
        assert (
            resolved.fiber_arrays[0].optical_paths[0].geometries[0].name
            == "East trench geometry"
        )

    def test_only_assembly_items_have_validity_times(self):
        """Subcomponent items should not carry data-time validity intervals."""
        for model in (Geometry, FiberSegment, Interrogator):
            with pytest.raises(ValueError, match="Extra inputs"):
                model.model_validate({"start_time": "2020-01-01"})

        for model in (Geometry, FiberArray, Interrogator):
            with pytest.raises(ValueError, match="Extra inputs"):
                model.model_validate({"creation_time": "2020-01-01"})

        FiberArray(start_time="2020-01-01")
        Acquisition(start_time="2020-01-01")
        OpticalPath(start_time="2020-01-01")

    def test_items_resolve_relationships(self):
        """Resolved items should expose object links, not id fields."""
        network = make_inventory().networks[0]
        fiber_array = network.fiber_arrays[0]

        assert network.inventory_id == "/network/XX"
        assert fiber_array.inventory_id.endswith("/fiber_arrays/0")
        interrogator = fiber_array.acquisitions[0].interrogator
        assert interrogator.model == "DAS"
        assert interrogator.resource_id == "interrogator_1"
        crs = fiber_array.optical_paths[0].coordinate_reference_system
        assert crs.axis_order == ("x",)
        assert fiber_array.optical_paths[0].annotations[0].label == "well"
        assert "acquisition_ids" not in fiber_array.__class__.model_fields

    def test_shared_linked_resources_are_stored_once(self):
        """Repeated linked public objects should not duplicate canonical items."""
        interrogator = Interrogator(resource_id="interrogator_1")
        config_1 = Acquisition(code="A", interrogator=interrogator)
        config_2 = Acquisition(code="B", interrogator=interrogator)
        inventory = Inventory(
            networks=(
                Network(
                    code="XX",
                    fiber_arrays=(
                        FiberArray(code="FA1", acquisitions=(config_1,)),
                        FiberArray(code="FA2", acquisitions=(config_2,)),
                    ),
                ),
            )
        )

        interrogators = tuple(
            item
            for item in inventory.resources.values()
            if isinstance(item, Interrogator)
        )

        assert len(interrogators) == 1

    def test_generated_resource_ids_only_apply_to_public_items(self):
        """Shareable public items get generated ids; internal items do not."""
        with set_config(inventory_resource_id_prefix="smi:local/custom/"):
            inventory = Inventory(
                networks=(
                    Network(
                        code="XX",
                        fiber_arrays=(
                            FiberArray(
                                code="FA",
                                acquisitions=(
                                    Acquisition(
                                        code="RAW",
                                        interrogator=Interrogator(model="DAS"),
                                    ),
                                ),
                            ),
                        ),
                    ),
                )
            )

        interrogator = next(iter(inventory.resources.values()))
        network = get_indexed(inventory, Network)[0]

        assert re.fullmatch(
            r"smi:local/custom/interrogator/[0-9a-f]{32}",
            interrogator.resource_id,
        )
        assert "resource_id" not in network

    def test_internal_ids_keep_identical_siblings_distinct(self):
        """Content-identical siblings should not collapse into one runtime node."""
        fibers = (
            FiberSegment(optical_length=1.0, fiber_type="single_mode"),
            FiberSegment(optical_length=1.0, fiber_type="single_mode"),
        )
        path = OpticalPath(optical_length=2.0, optical_components=fibers)
        inventory = Inventory(
            networks=(
                Network(
                    code="XX",
                    fiber_arrays=(FiberArray(code="FA", optical_paths=(path,)),),
                ),
            )
        )

        items = get_indexed(inventory, FiberSegment)
        ids = {item["id"] for item in items}

        assert len(items) == 2
        assert len(ids) == 2

    def test_explicit_resource_ids_override_generation(self):
        """Explicit public resource ids should be preserved unchanged."""
        inventory = Inventory(
            resources={"external:iu001": Interrogator(resource_id="external:iu001")}
        )

        item = get_indexed(inventory, Interrogator)[0]

        assert item["resource_id"] == "external:iu001"

    def test_external_resource_roundtrip(self):
        """External resources should store public ids for out-of-model resources."""
        resource = ExternalResource(
            uri="https://example.com/assets/instrument-123",
            name="Field asset registry entry",
            description="Maintained outside DASCore.",
        )
        inventory = Inventory(resources={"": resource})
        item = get_indexed(inventory, ExternalResource)[0]
        resolved = next(iter(inventory.resources.values()))

        assert re.fullmatch(
            r"smi:local/dascore/external_resource/[0-9a-f]{32}",
            item["resource_id"],
        )
        assert resolved.uri == "https://example.com/assets/instrument-123"
        assert Inventory.from_dict(inventory.to_dict()).to_dict() == inventory.to_dict()

    def test_cable_specification_uses_external_resource(self):
        """Cable specifications should store as links to external resources."""
        specification = ExternalResource(
            resource_id="epsilon_spec",
            uri="https://example.com/epsilon",
            name="Epsilon product page",
        )
        cable = Cable(
            resource_id="cable_1",
            model="Epsilon Sensor",
            specification=specification,
        )
        inventory = Inventory(resources={"cable_1": cable})
        cable_item = get_indexed(inventory, Cable)[0]
        out = inventory.resources["cable_1"]

        assert cable_item["specification_resource_id"] == "epsilon_spec"
        assert "specification" not in cable_item
        assert out.specification.uri == "https://example.com/epsilon"

    def test_optical_component_container_supports_enclosures(self):
        """Optical components can be housed by generic enclosures."""
        enclosure = Enclosure(resource_id="closure_1", name="Launch box")
        fiber = FiberSegment(optical_length=10.0, container=enclosure)
        splice = Splice(container=enclosure)
        connector = Connector(container=enclosure)
        path = OpticalPath(optical_components=(fiber, splice, connector))
        inventory = Inventory(
            networks=(
                Network(
                    code="XX",
                    fiber_arrays=(FiberArray(code="FA", optical_paths=(path,)),),
                ),
            )
        )
        fiber_item = get_indexed(inventory, FiberSegment)[0]
        splice_item = get_indexed(inventory, Splice)[0]
        out = get_path(inventory).optical_components[2]

        assert fiber_item["container_id"] == "closure_1"
        assert splice_item["container_id"] == "closure_1"
        assert out.container.name == "Launch box"

    def test_coordinate_reference_system_defaults_to_lat_lon(self):
        """A bare CRS should describe WGS84 geographic coordinates."""
        crs = CoordinateReferenceSystem()

        assert crs.authority == "EPSG"
        assert crs.code == "4326"
        assert crs.axis_order == ("latitude", "longitude", "elevation")
        assert crs.units == "degree"

    def test_optical_path_defaults_to_wgs84_crs(self):
        """Lat/lon geometries should not require explicit CRS boilerplate."""
        geometry = Geometry(coordinates=((40.0, -111.0, 1500.0),))
        path = OpticalPath(geometries=(geometry,))
        inventory = Inventory(
            networks=(
                Network(
                    code="XX",
                    fiber_arrays=(FiberArray(code="FA", optical_paths=(path,)),),
                ),
            )
        )
        path_crs = path.coordinate_reference_system
        crs = inventory.resources["EPSG:4326"]
        path_item = get_indexed(inventory, OpticalPath)[0]

        assert path_crs is not None
        assert path_crs.resource_id == "EPSG:4326"
        assert crs.resource_id == "EPSG:4326"
        assert crs.axis_order == ("latitude", "longitude", "elevation")
        assert path_item["coordinate_reference_system_id"] == "EPSG:4326"

    def test_coordinate_reference_system_supports_custom_project_crs(self):
        """Custom project CRS definitions should round-trip structured forms."""
        crs = CoordinateReferenceSystem(
            resource_id="crs_custom",
            authority="LOCAL",
            code="project_enu",
            crs_wkt='ENGCRS["Project ENU"]',
            projjson={"type": "EngineeringCRS", "name": "Project ENU"},
            grid_mapping={"grid_mapping_name": "latitude_longitude"},
            origin=(40.0, -111.0, 1500.0),
            axis_order=("x", "y", "z"),
            units="m",
            vertical_datum="local ground surface",
        )
        inventory = Inventory(resources={"crs_custom": crs})

        out = inventory.resources["crs_custom"]

        assert out.authority == "LOCAL"
        assert out.code == "project_enu"
        assert out.projjson["type"] == "EngineeringCRS"
        assert Inventory.from_dict(inventory.to_dict()).to_dict() == inventory.to_dict()

    def test_acquisition_extra_fields_roundtrip(self):
        """Acquisition extra fields should support simple scalar metadata."""
        config = Acquisition(
            code="RAW",
            extra_fields={
                "native_header_key": "native_header_value",
                "pulse_width_ns": 100,
                "laser_power": 0.5,
                "phase_tracking": True,
            },
        )
        inventory = Inventory(
            networks=(
                Network(
                    code="XX",
                    fiber_arrays=(FiberArray(code="FA", acquisitions=(config,)),),
                ),
            )
        )
        item = get_indexed(inventory, Acquisition)[0]
        out = inventory.networks[0].fiber_arrays[0].acquisitions[0]

        assert item["extra_fields"]["native_header_key"] == "native_header_value"
        assert out.extra_fields["pulse_width_ns"] == 100

    def test_acquisition_extra_fields_reject_nested_values(self):
        """Extra fields should stay flat and JSON-scalar-like."""
        with pytest.raises(ValueError):
            Acquisition(extra_fields={"nested": {"bad": "value"}})

    def test_replace_replaces_internal_node(self):
        """Extracted internal items can be replaced by hidden id."""
        inventory = make_inventory()
        geometry = get_path(inventory).geometries[0]
        out = inventory.replace(geometry.revise(name="corrected"))

        geometries = get_path(out).geometries

        assert len(geometries) == 1
        assert geometries[0].name == "corrected"

    def test_replace_internal_item(self):
        """Replacing an item swaps the stored object."""
        inventory = make_inventory().new(creation_info=CreationInfo(version="3"))
        old = inventory.networks[0].fiber_arrays[0]
        new = old.revise(tag="processed")
        out = inventory.replace(new)

        latest = out.networks[0].fiber_arrays[0]

        assert latest.tag == "processed"
        assert isinstance(latest.optical_paths[0], OpticalPath)
        assert out.creation_info.version == "4"

    def test_replace_public_resource_updates_references(self):
        """Replacing a shared resource should update resolved references."""
        inventory = make_inventory().new(creation_info=CreationInfo(version="8"))
        interrogator = inventory.resources["interrogator_1"]

        out = inventory.replace(interrogator.revise(model="DAS-2"))
        acquisition = out.networks[0].fiber_arrays[0].acquisitions[0]

        assert out.resources["interrogator_1"].model == "DAS-2"
        assert acquisition.interrogator.model == "DAS-2"
        assert out.creation_info.version == "9"

    def test_replace_internal_item_can_add_supporting_resource(self):
        """Replacing an internal object may add a newly referenced resource."""
        inventory = make_inventory()
        acquisition = inventory.networks[0].fiber_arrays[0].acquisitions[0]
        new_interrogator = Interrogator(
            resource_id="interrogator_2",
            model="DAS-2",
        )

        out = inventory.replace(acquisition.revise(interrogator=new_interrogator))
        resolved = out.networks[0].fiber_arrays[0].acquisitions[0]

        assert set(out.resources) >= {"interrogator_1", "interrogator_2"}
        assert resolved.interrogator.resource_id == "interrogator_2"
        assert resolved.interrogator.model == "DAS-2"

    def test_replace_does_not_append_epoch(self):
        """Replace swaps the item even when start time changes."""
        old = (
            make_inventory()
            .networks[0]
            .fiber_arrays[0]
            .revise(
                start_time="2020-01-01",
                end_time="2021-01-01",
                tag="old",
            )
        )
        inventory = Inventory(networks=(Network(code="XX", fiber_arrays=(old,)),))
        new = old.revise(start_time="2021-01-01", end_time=None, tag="new")

        out = inventory.replace(new)

        latest = out.networks[0].fiber_arrays[0]
        assert latest.tag == "new"

    def test_bare_internal_child_without_identity_raises(self):
        """Internal child objects need tree context or inventory identity."""
        with pytest.raises(ParameterError, match="must be nested"):
            Inventory(networks=(Geometry(optical_length=1.0),))

    def test_public_resource_id_cannot_change_type(self):
        """Public resource ids should be globally unique to one object type."""
        with pytest.raises(ValueError, match="used for both"):
            cable = Cable(resource_id="same_id")
            interrogator = Interrogator(resource_id="same_id")
            acquisition = Acquisition(code="RAW", interrogator=interrogator)
            fiber = FiberSegment(container=cable)
            path = OpticalPath(optical_components=(fiber,))
            Inventory(
                networks=(
                    Network(
                        code="XX",
                        fiber_arrays=(
                            FiberArray(
                                code="FA",
                                acquisitions=(acquisition,),
                                optical_paths=(path,),
                            ),
                        ),
                    ),
                )
            )

    def test_replace_requires_existing_item(self):
        """Replace should raise when the item is not already in the inventory."""
        fiber_array = FiberArray(code="missing")

        with pytest.raises(KeyError, match="No inventory item"):
            make_inventory().replace(fiber_array)

    def test_missing_reference_raises_on_load(self):
        """Broken id links should be caught when inventories are created."""
        with pytest.raises(ValueError, match="missing"):
            Inventory(
                networks=(
                    {
                        "type": "Network",
                        "code": "XX",
                        "id": "/network/XX",
                        "fiber_array_ids": ("missing",),
                    },
                )
            )


class TestInventorySerialization:
    """Tests for inventory JSON/YAML serialization."""

    def test_creation_info_is_inventory_metadata(self):
        """Inventory provenance should live in a CreationInfo object."""
        inventory = Inventory(
            creation_info=CreationInfo(
                agency_id="UU",
                author="field_team",
                creation_time="2024-01-01",
                update_time="2024-02-01",
                version="7",
            ),
        )

        out = inventory.to_dict()

        assert "creation_time" not in out
        assert out["creation_info"]["agency_id"] == "UU"
        assert out["creation_info"]["author"] == "field_team"
        assert out["creation_info"]["version"] == "7"

    def test_new_increments_creation_info_version(self):
        """New should increment integer-like content versions by default."""
        inventory = Inventory(creation_info=CreationInfo(version="7"))

        out = inventory.new(comment="updated")
        first = Inventory().new(comment="first")
        explicit = inventory.new(
            comment="explicit",
            creation_info=CreationInfo(version="100"),
        )

        assert out.creation_info.version == "8"
        assert first.creation_info.version == "1"
        assert explicit.creation_info.version == "100"

    def test_dict_roundtrip(self):
        """The nested manifest should round trip through dicts."""
        inventory = make_inventory()

        def assert_no_runtime_id(value):
            if isinstance(value, dict):
                assert "id" not in value
                for item in value.values():
                    assert_no_runtime_id(item)
            elif isinstance(value, list | tuple):
                for item in value:
                    assert_no_runtime_id(item)

        out = inventory.to_dict()
        assert out["format"] == "fas_inventory"
        assert "items" not in out
        assert_no_runtime_id(out)
        assert Inventory.from_dict(out).to_dict() == out

    def test_serialized_resources_are_keyed_by_resource_id(self):
        """Resource payloads should not repeat their mapping key."""
        inventory = make_inventory()

        resources = inventory.to_dict()["resources"]

        assert "interrogator_1" in resources
        assert "resource_id" not in resources["interrogator_1"]
        assert resources["interrogator_1"]["type"] == "Interrogator"

    def test_resource_mapping_key_must_match_payload_id(self):
        """Resource mappings should reject ambiguous ids."""
        with pytest.raises(ValueError, match="mapping keys"):
            Inventory(
                resources={
                    "interrogator_1": {
                        "type": "Interrogator",
                        "resource_id": "interrogator_2",
                    }
                }
            )

    def test_json_roundtrip_string(self):
        """Inventories can dump to and load from JSON strings."""
        inventory = make_inventory()

        assert Inventory.from_json(inventory.to_json()).to_dict() == inventory.to_dict()

    def test_json_roundtrip_stream(self):
        """Inventories can dump to and load from readable streams."""
        inventory = make_inventory()
        stream = io.StringIO()
        inventory.to_json(stream)
        stream.seek(0)

        assert Inventory.from_json(stream).to_dict() == inventory.to_dict()

    def test_yaml_roundtrip_string(self):
        """Inventories can dump to and load from YAML streams."""
        inventory = make_inventory()
        stream = io.StringIO(inventory.to_yaml())

        assert Inventory.from_yaml(stream).to_dict() == inventory.to_dict()

    def test_extra_manifest_fields_are_forbidden(self):
        """Inventory construction should reject unknown manifest fields."""
        with pytest.raises(ValueError, match="Extra inputs"):
            Inventory.model_validate({"unexpected": True})

    def test_rejects_missing_or_wrong_manifest_format(self):
        """Loaded manifests should identify the FAS Inventory format."""
        with pytest.raises(ValueError, match="fas_inventory"):
            Inventory.from_dict({"schema_version": 1, "networks": []})
        with pytest.raises(ValueError, match="fas_inventory"):
            Inventory.from_dict(
                {"format": "other_inventory", "schema_version": 1, "networks": []}
            )


class TestOpticalPath:
    """Tests for optical path helpers using resolved objects."""

    def test_unknown_interval_items_need_only_optical_length(self):
        """Path tracks can represent unknown intervals with only optical_length."""
        fiber = FiberSegment(optical_length=10.0)
        geometry = Geometry(optical_length=10.0)
        coupling = CouplingCondition(optical_length=10.0)
        path = OpticalPath(
            optical_length=10.0,
            optical_components=(fiber,),
            geometries=(geometry,),
            coupling_conditions=(coupling,),
        )

        assert path.validate() is path
        assert fiber.fiber_type == ""
        assert geometry.coordinates == ()
        assert coupling.coupling_type == ""

    def test_annotations_support_open_distance_intervals(self):
        """Annotation distances should accept DASCore-style open bounds."""
        full = OpticalPathAnnotation(distance=(..., ...), label="all")
        left = OpticalPathAnnotation(distance=(None, 1.0), label="left")
        right = OpticalPathAnnotation(distance=(1.0, None), label="right")
        path = OpticalPath(optical_length=2.0, annotations=(full, left, right))
        selected = path.select(distance=(0.5, 1.5))

        assert full.distance == (None, None)
        assert left.distance == (None, 1.0)
        assert right.distance == (1.0, None)
        assert tuple(annotation.distance for annotation in selected.annotations) == (
            (0.0, 1.0),
            (0.0, 0.5),
            (0.5, 1.0),
        )

    def test_validate(self):
        """Track lengths should validate against path length."""
        path = get_path(make_inventory())

        assert path.validate() is path

    def test_select_distance(self):
        """Selecting a distance interval should clip child tracks."""
        path = get_path(make_inventory())
        out = path.select(distance=(0.5, 1.5))

        assert out.optical_length == 1.0
        assert out.coordinate_reference_system == path.coordinate_reference_system
        assert out.geometries[0].optical_length == 1.0
        assert out.annotations[0].distance == (0.0, 1.0)

    def test_reverse(self):
        """Reversing a path should reverse child tracks."""
        path = get_path(make_inventory())
        out = path.reverse()

        assert out.optical_length == path.optical_length
        assert out.coordinate_reference_system == path.coordinate_reference_system
        assert out.annotations[0].distance == (0.0, 2.0)

    def test_add_preserves_matching_crs(self):
        """Concatenating paths should keep the shared CRS."""
        path = get_path(make_inventory())
        out = path + path

        assert out.optical_length == 4.0
        assert out.coordinate_reference_system == path.coordinate_reference_system

    def test_add_rejects_mismatched_crs(self):
        """Concatenating paths with different CRS definitions is ambiguous."""
        path = get_path(make_inventory())
        other_crs = CoordinateReferenceSystem(
            resource_id="crs_2",
            authority="LOCAL",
            code="other_axis",
            axis_order=("y",),
            units="m",
        )
        other = path.revise(coordinate_reference_system=other_crs)

        with pytest.raises(ParameterError, match="different CRS"):
            path + other
