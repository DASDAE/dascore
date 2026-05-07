"""Tests for inventory records and manifests."""

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
        resource_id="cfg_1",
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
        resource_id="geo_1",
        optical_length=2.0,
        geometry_type="linear",
        coordinate_reference_system=crs,
        coordinates=((0.0,), (2.0,)),
    )
    fiber = FiberSegment(resource_id="fiber_1", optical_length=2.0)
    annotation = OpticalPathAnnotation(
        resource_id="anno_1",
        distance=(0.0, 2.0),
        label="well",
    )
    path = OpticalPath(
        resource_id="path_1",
        optical_length=2.0,
        optical_components=(fiber,),
        geometries=(geometry,),
        annotations=(annotation,),
    )
    fiber_array = FiberArray(
        resource_id="fiber_array_1",
        acquisitions=(configuration,),
        optical_paths=(path,),
        tag="raw",
    )
    network = Network(
        resource_id="network_1",
        code="XX",
        description="Test network",
        fiber_arrays=(fiber_array,),
    )
    return Inventory(resource_id="inv_1", records=(network,))


class TestInventoryRecords:
    """Tests for canonical storage and resolved record views."""

    def test_stores_canonical_id_records(self):
        """Object graphs should be stored as id-only canonical records."""
        inventory = make_inventory()
        records = inventory.get_record_dicts()
        network = inventory.get_record_dicts(record_ids="network_1")[0]
        fiber_array = inventory.get_record_dicts(record_ids="fiber_array_1")[0]
        config = inventory.get_record_dicts(record_ids="cfg_1")[0]
        resolved_config = inventory.get_records(record_ids="cfg_1")[0]
        interrogator = inventory.get_record_dicts(record_ids="interrogator_1")[0]
        path = inventory.get_record_dicts(record_ids="path_1")[0]

        assert len(records) == 9
        assert network["fiber_array_ids"] == ("fiber_array_1",)
        assert fiber_array["acquisition_ids"] == ("cfg_1",)
        assert fiber_array["optical_path_ids"] == ("path_1",)
        assert "acquisitions" not in fiber_array
        assert config["code"] == "RAW"
        assert config["location_code"] == "00"
        assert config["data_category"] == "DAS"
        assert config["data_type"] == "strain_rate"
        assert config["data_units"] == "m / s"
        assert config["acquisition_sample_rate"] == 100.0
        assert config["pulse_width"] == 1.0e-8
        assert config["comment"] == "Initial acquisition."
        assert resolved_config.code == "RAW"
        assert resolved_config.location_code == "00"
        assert resolved_config.data_category == "DAS"
        assert resolved_config.data_type == "strain_rate"
        assert resolved_config.acquisition_sample_rate == 100.0
        assert resolved_config.pulse_width == 1.0e-8
        assert resolved_config.comment == "Initial acquisition."
        assert config["interrogator_resource_id"] == "interrogator_1"
        assert "interrogator" not in config
        assert interrogator["resource_id"] == "interrogator_1"
        assert path["geometry_ids"] == ("geo_1",)
        assert path["annotation_ids"] == ("anno_1",)
        assert "geometries" not in path

    def test_records_property_cannot_mutate_storage(self):
        """The records property should not expose mutable internal state."""
        inventory = make_inventory()
        record = inventory.records[0]

        record["resource_id"] = "changed"

        assert inventory.get_record_dicts()[0]["resource_id"] != "changed"

    def test_names_are_display_metadata(self):
        """Names should round-trip without replacing codes or resource ids."""
        path = OpticalPath(resource_id="path_1", name="Main trench path")
        geometry = Geometry(resource_id="geo_1", name="East trench geometry")
        fiber_array = FiberArray(
            resource_id="array_1",
            name="North trench array",
            optical_paths=(path.model_copy(update={"geometries": (geometry,)}),),
        )
        network = Network(
            resource_id="network_1",
            code="FORGE",
            name="Utah FORGE DAS",
            fiber_arrays=(fiber_array,),
        )

        inventory = Inventory(records=(network,))
        record = inventory.get_record_dicts(record_ids="network_1")[0]
        resolved = inventory.get_records(record_ids="network_1")[0]

        assert record["resource_id"] == "network_1"
        assert record["code"] == "FORGE"
        assert record["name"] == "Utah FORGE DAS"
        assert resolved.fiber_arrays[0].name == "North trench array"
        assert resolved.fiber_arrays[0].optical_paths[0].name == "Main trench path"
        assert (
            resolved.fiber_arrays[0].optical_paths[0].geometries[0].name
            == "East trench geometry"
        )

    def test_only_assembly_records_have_validity_times(self):
        """Subcomponent records should not carry data-time validity intervals."""
        for model in (Geometry, FiberSegment, Interrogator):
            with pytest.raises(ValueError, match="Extra inputs"):
                model.model_validate({"start_time": "2020-01-01"})

        FiberArray(start_time="2020-01-01")
        Acquisition(start_time="2020-01-01")
        OpticalPath(start_time="2020-01-01")

    def test_get_records_resolves_relationships(self):
        """Resolved records should expose object links, not id fields."""
        network = make_inventory().get_records(record_ids="network_1")[0]
        fiber_array = make_inventory().get_records(record_ids="fiber_array_1")[0]

        assert network.fiber_arrays[0].resource_id == "fiber_array_1"
        interrogator = fiber_array.acquisitions[0].interrogator
        assert interrogator.model == "DAS"
        assert interrogator.resource_id == "interrogator_1"
        crs = fiber_array.optical_paths[0].geometries[0].coordinate_reference_system
        assert crs.axis_order == ("x",)
        assert fiber_array.optical_paths[0].annotations[0].label == "well"
        assert "acquisition_ids" not in fiber_array.__class__.model_fields

    def test_shared_linked_records_are_stored_once(self):
        """Repeated linked objects should not duplicate canonical records."""
        interrogator = Interrogator(resource_id="interrogator_1")
        config_1 = Acquisition(resource_id="cfg_1", interrogator=interrogator)
        config_2 = Acquisition(resource_id="cfg_2", interrogator=interrogator)
        inventory = Inventory(
            records=(
                FiberArray(
                    resource_id="fiber_array_1",
                    acquisitions=(config_1,),
                ),
                FiberArray(
                    resource_id="fiber_array_2",
                    acquisitions=(config_2,),
                ),
            )
        )

        interrogators = inventory.get_record_dicts(record_types=Interrogator)

        assert len(interrogators) == 1

    def test_generated_resource_ids_are_random_local_ids(self):
        """Equivalent object graphs without explicit ids should get random ids."""

        def make_generated_inventory():
            crs = CoordinateReferenceSystem(
                authority="LOCAL",
                code="generated_axis",
                axis_order=("x",),
                units="m",
            )
            geometry = Geometry(
                optical_length=2.0,
                geometry_type="linear",
                coordinate_reference_system=crs,
                coordinates=((0.0,), (2.0,)),
            )
            path = OpticalPath(optical_length=2.0, geometries=(geometry,))
            fiber_array = FiberArray(optical_paths=(path,), tag="raw")
            network = Network(code="XX", fiber_arrays=(fiber_array,))
            return Inventory(records=(network,))

        inv_1 = make_generated_inventory()
        inv_2 = make_generated_inventory()
        records = inv_1.get_record_dicts()
        fiber_array = inv_1.get_record_dicts(record_types=FiberArray)[0]

        assert inv_1.to_dict() != inv_2.to_dict()
        assert all(record["resource_id"] for record in records)
        assert re.fullmatch(
            r"smi:local/dascore/fiber_array/[0-9a-f]{32}",
            fiber_array["resource_id"],
        )

    def test_generated_resource_id_prefix_comes_from_config(self):
        """Generated inventory resource ids should honor runtime config."""
        with set_config(inventory_resource_id_prefix="smi:local/custom/"):
            inventory = Inventory(records=(Interrogator(model="DAS"),))

        record = inventory.get_record_dicts(record_types=Interrogator)[0]

        assert re.fullmatch(
            r"smi:local/custom/interrogator/[0-9a-f]{32}",
            record["resource_id"],
        )

    def test_generated_resource_ids_keep_identical_siblings_distinct(self):
        """Content-identical siblings should not collapse into one resource."""
        fibers = (
            FiberSegment(optical_length=1.0, fiber_type="single_mode"),
            FiberSegment(optical_length=1.0, fiber_type="single_mode"),
        )
        path = OpticalPath(optical_length=2.0, optical_components=fibers)
        inventory = Inventory(records=(path,))

        records = inventory.get_record_dicts(record_types=FiberSegment)
        resource_ids = {record["resource_id"] for record in records}

        assert len(records) == 2
        assert len(resource_ids) == 2

    def test_generated_resource_ids_share_reused_object_instances(self):
        """Reused object instances should keep one generated resource id."""
        interrogator = Interrogator(model="DAS")
        config_1 = Acquisition(interrogator=interrogator)
        config_2 = Acquisition(interrogator=interrogator)
        inventory = Inventory(
            records=(
                FiberArray(acquisitions=(config_1,)),
                FiberArray(acquisitions=(config_2,)),
            )
        )

        interrogators = inventory.get_record_dicts(record_types=Interrogator)
        configs = inventory.get_record_dicts(record_types=Acquisition)

        assert len(interrogators) == 1
        assert {config["interrogator_resource_id"] for config in configs} == {
            interrogators[0]["resource_id"]
        }

    def test_explicit_resource_ids_override_generation(self):
        """Explicit resource ids should be preserved unchanged."""
        inventory = Inventory(records=(Interrogator(resource_id="external:iu001"),))

        record = inventory.get_record_dicts(record_types=Interrogator)[0]

        assert record["resource_id"] == "external:iu001"

    def test_external_resource_roundtrip(self):
        """External resources should store ids for out-of-model resources."""
        resource = ExternalResource(
            uri="https://example.com/assets/instrument-123",
            name="Field asset registry entry",
            description="Maintained outside DASCore.",
        )
        inventory = Inventory(records=(resource,))
        record = inventory.get_record_dicts(record_types=ExternalResource)[0]
        resolved = inventory.get_records(record_types=ExternalResource)[0]

        assert re.fullmatch(
            r"smi:local/dascore/external_resource/[0-9a-f]{32}",
            record["resource_id"],
        )
        assert resolved.uri == "https://example.com/assets/instrument-123"
        assert Inventory.from_dict(inventory.to_dict()).to_dict() == inventory.to_dict()

    def test_external_resource_is_not_accepted_by_unrelated_relationships(self):
        """Relationships should stay explicitly typed."""
        with pytest.raises(ValueError, match="missing"):
            Inventory(
                records=(
                    ExternalResource(resource_id="external:cable"),
                    {
                        "type": "FiberSegment",
                        "resource_id": "fiber",
                        "container_id": "external:cable",
                    },
                )
            )

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
        inventory = Inventory(records=(cable,))
        cable_record = inventory.get_record_dicts(record_ids="cable_1")[0]
        out = inventory.get_records(record_ids="cable_1")[0]

        assert cable_record["specification_resource_id"] == "epsilon_spec"
        assert "specification" not in cable_record
        assert out.specification.uri == "https://example.com/epsilon"

    def test_optical_component_container_supports_enclosures(self):
        """Optical components can be housed by generic enclosures."""
        specification = ExternalResource(
            resource_id="closure_spec",
            uri="https://example.com/closure",
        )
        enclosure = Enclosure(
            resource_id="closure_1",
            name="Launch box",
            specification=specification,
        )
        fiber = FiberSegment(
            resource_id="launch_fiber",
            optical_length=10.0,
            container=enclosure,
        )
        splice = Splice(resource_id="splice_1", container=enclosure)
        connector = Connector(resource_id="connector_1", container=enclosure)
        inventory = Inventory(records=(fiber, splice, connector))
        fiber_record = inventory.get_record_dicts(record_ids="launch_fiber")[0]
        splice_record = inventory.get_record_dicts(record_ids="splice_1")[0]
        closure_record = inventory.get_record_dicts(record_ids="closure_1")[0]
        out = inventory.get_records(record_ids="connector_1")[0]

        assert fiber_record["container_id"] == "closure_1"
        assert splice_record["container_id"] == "closure_1"
        assert closure_record["specification_resource_id"] == "closure_spec"
        assert out.container.name == "Launch box"

    def test_coordinate_reference_system_defaults_to_lat_lon(self):
        """A bare CRS should describe WGS84 geographic coordinates."""
        crs = CoordinateReferenceSystem()

        assert crs.authority == "EPSG"
        assert crs.code == "4326"
        assert crs.axis_order == ("latitude", "longitude", "elevation")
        assert crs.units == "degree"

    def test_geometry_defaults_to_wgs84_crs(self):
        """Lat/lon geometries should not require explicit CRS boilerplate."""
        geometry = Geometry(coordinates=((40.0, -111.0, 1500.0),))
        inventory = Inventory(records=(geometry,))
        geometry_crs = geometry.coordinate_reference_system
        crs = inventory.get_records(record_types=CoordinateReferenceSystem)[0]
        geometry_record = inventory.get_record_dicts(record_types=Geometry)[0]

        assert geometry_crs is not None
        assert geometry_crs.resource_id == "EPSG:4326"
        assert crs.resource_id == "EPSG:4326"
        assert crs.axis_order == ("latitude", "longitude", "elevation")
        assert geometry_record["coordinate_reference_system_id"] == "EPSG:4326"

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
        inventory = Inventory(records=(crs,))

        out = inventory.get_records(record_ids="crs_custom")[0]

        assert out.authority == "LOCAL"
        assert out.code == "project_enu"
        assert out.projjson["type"] == "EngineeringCRS"
        assert Inventory.from_dict(inventory.to_dict()).to_dict() == inventory.to_dict()

    def test_get_records_filters_by_type(self):
        """Type filters should return current records of the requested classes."""
        records = make_inventory().get_records(
            record_types=(Geometry, OpticalPathAnnotation)
        )

        assert {record.type for record in records} == {
            "Geometry",
            "OpticalPathAnnotation",
        }

    def test_acquisition_extra_fields_roundtrip(self):
        """Acquisition extra fields should support simple scalar metadata."""
        config = Acquisition(
            resource_id="cfg_extra",
            extra_fields={
                "native_header_key": "native_header_value",
                "pulse_width_ns": 100,
                "laser_power": 0.5,
                "phase_tracking": True,
            },
        )
        inventory = Inventory(records=(config,))
        record = inventory.get_record_dicts(record_ids="cfg_extra")[0]
        out = inventory.get_records(record_ids="cfg_extra")[0]

        assert record["extra_fields"]["native_header_key"] == "native_header_value"
        assert out.extra_fields["pulse_width_ns"] == 100

    def test_acquisition_extra_fields_reject_nested_values(self):
        """Extra fields should stay flat and JSON-scalar-like."""
        with pytest.raises(ValueError):
            Acquisition(
                resource_id="cfg_extra",
                extra_fields={"nested": {"bad": "value"}},
            )

    def test_put_matching_start_replaces_epoch(self):
        """Putting the same resource/start replaces the stored epoch."""
        inventory = make_inventory()
        old = inventory.get_records(record_ids="fiber_array_1")[0]
        new = old.model_copy(update={"tag": "processed"})
        out = inventory.put_records(records=(new,))

        history = out.get_record_dicts(record_ids="fiber_array_1", include_history=True)
        latest = out.get_records(record_ids="fiber_array_1")[0]

        assert [record["tag"] for record in history] == ["processed"]
        assert latest.tag == "processed"
        assert isinstance(latest.optical_paths[0], OpticalPath)

    def test_put_different_start_adds_epoch(self):
        """Putting a different start time adds another validity epoch."""
        old = make_inventory().get_records(record_ids="fiber_array_1")[0]
        old = old.model_copy(
            update={
                "start_time": "2020-01-01",
                "end_time": "2021-01-01",
                "tag": "old",
            }
        )
        inventory = Inventory(records=(old,))
        new = old.model_copy(
            update={
                "start_time": "2021-01-01",
                "end_time": None,
                "tag": "new",
            }
        )

        out = inventory.put_records(records=(new,))

        history = out.get_record_dicts(record_ids="fiber_array_1", include_history=True)
        latest = out.get_records(record_ids="fiber_array_1")[0]
        before = out.get_records(record_ids="fiber_array_1", time="2020-06-01")[0]
        after = out.get_records(record_ids="fiber_array_1", time="2021-06-01")[0]
        assert [record["tag"] for record in history] == ["old", "new"]
        assert latest.tag == "new"
        assert before.tag == "old"
        assert after.tag == "new"

    def test_capping_open_epoch_replaces_open_record(self):
        """A cap with the same start replaces an open-ended record."""
        fiber_array = make_inventory().get_records(record_ids="fiber_array_1")[0]
        open_epoch = fiber_array.model_copy(
            update={"start_time": "2020-01-01", "tag": "open"}
        )
        inventory = Inventory(records=(open_epoch,))
        capped = open_epoch.model_copy(update={"end_time": "2021-01-01"})

        out = inventory.put_records(records=(capped,))

        inside = out.get_records(record_ids="fiber_array_1", time="2020-06-01")[0]
        assert inside.tag == "open"
        with pytest.raises(KeyError, match="valid at"):
            out.get_records(record_ids="fiber_array_1", time="2022-01-01")

    def test_fiber_array_stores_multiple_optical_paths(self):
        """Fiber arrays should carry a discoverable set of optical path epochs."""
        old = OpticalPath(
            resource_id="path_old",
            start_time="2020-01-01",
            end_time="2021-01-01",
            optical_length=1.0,
        )
        new = OpticalPath(
            resource_id="path_new",
            start_time="2021-01-01",
            optical_length=2.0,
        )
        fiber_array = FiberArray(
            resource_id="fiber_array_1",
            optical_paths=(old, new),
        )
        inventory = Inventory(records=(fiber_array,))
        record = inventory.get_record_dicts(record_ids="fiber_array_1")[0]
        out = inventory.get_records(record_ids="fiber_array_1")[0]

        assert record["optical_path_ids"] == ("path_old", "path_new")
        assert [path.resource_id for path in out.optical_paths] == [
            "path_old",
            "path_new",
        ]

    def test_get_record_dicts_time_and_history_conflict(self):
        """A request cannot ask for both one validity time and all history."""
        with pytest.raises(ParameterError, match="mutually exclusive"):
            make_inventory().get_record_dicts(
                record_ids="fiber_array_1",
                time="2020-01-01",
                include_history=True,
            )

    def test_resource_id_cannot_change_type(self):
        """Resource ids should be globally unique to one record type."""
        inventory = make_inventory()

        with pytest.raises(ValueError, match="already used"):
            inventory.put_records(
                records=(Interrogator(resource_id="fiber_array_1", model="bad"),)
            )

        with pytest.raises(ValueError, match="used for both"):
            Inventory(
                records=(
                    FiberArray(resource_id="same_id"),
                    Interrogator(resource_id="same_id"),
                )
            )

    def test_put_records_requires_sequence(self):
        """The API should keep records explicit instead of accepting one scalar."""
        fiber_array = make_inventory().get_records(record_ids="fiber_array_1")[0]

        with pytest.raises(ParameterError, match="sequence"):
            make_inventory().put_records(records=fiber_array)

    def test_missing_reference_raises_on_load(self):
        """Broken id links should be caught when inventories are created."""
        with pytest.raises(ValueError, match="missing"):
            Inventory(
                records=(
                    {
                        "type": "FiberArray",
                        "resource_id": "fiber_array_1",
                        "optical_path_ids": ("missing",),
                    },
                )
            )

    def test_missing_network_fiber_array_reference_raises_on_load(self):
        """Broken network fiber array links should be caught on load."""
        with pytest.raises(ValueError, match="missing"):
            Inventory(
                records=(
                    {
                        "type": "Network",
                        "resource_id": "network_1",
                        "fiber_array_ids": ("missing",),
                    },
                )
            )

    def test_canonical_mappings_reject_object_relationship_fields(self):
        """Canonical dict inputs should use id fields, not object fields."""
        with pytest.raises(ValueError, match="object relationship fields"):
            Inventory(
                records=(
                    {
                        "type": "FiberArray",
                        "resource_id": "fiber_array_1",
                        "optical_paths": (
                            {
                                "type": "OpticalPath",
                                "resource_id": "path_1",
                            },
                        ),
                    },
                )
            )


class TestInventorySerialization:
    """Tests for inventory JSON/YAML serialization."""

    def test_dict_roundtrip(self):
        """The canonical manifest should round trip through dicts."""
        inventory = make_inventory()

        assert Inventory.from_dict(inventory.to_dict()).to_dict() == inventory.to_dict()

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

    def test_rejects_old_manifest_aliases(self):
        """The breaking API should reject old manifest aliases."""
        with pytest.raises(ValueError, match="Extra inputs"):
            Inventory.model_validate({"inventory_id": "old", "records": []})
        with pytest.raises(ValueError, match="Extra inputs"):
            Inventory.model_validate({"objects": []})


class TestOpticalPath:
    """Tests for optical path helpers using resolved objects."""

    def test_unknown_interval_records_need_only_optical_length(self):
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
        path = OpticalPath(
            optical_length=2.0,
            annotations=(full, left, right),
        )
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
        path = make_inventory().get_records(record_ids="path_1")[0]

        assert path.validate() is path

    def test_select_distance(self):
        """Selecting a distance interval should clip child tracks."""
        path = make_inventory().get_records(record_ids="path_1")[0]
        out = path.select(distance=(0.5, 1.5))

        assert out.optical_length == 1.0
        assert out.geometries[0].optical_length == 1.0
        assert out.annotations[0].distance == (0.0, 1.0)

    def test_reverse(self):
        """Reversing a path should reverse child tracks."""
        path = make_inventory().get_records(record_ids="path_1")[0]
        out = path.reverse()

        assert out.optical_length == path.optical_length
        assert out.annotations[0].distance == (0.0, 2.0)
