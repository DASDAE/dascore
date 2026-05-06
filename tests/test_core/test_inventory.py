"""Tests for inventory records and manifests."""

from __future__ import annotations

import io

import pytest

from dascore.core.inventory import (
    AcquisitionConfiguration,
    CoordinateReferenceSystem,
    FiberArray,
    FiberSegment,
    Geometry,
    Interrogator,
    Inventory,
    Network,
    OpticalPath,
    OpticalPathAnnotation,
)
from dascore.exceptions import ParameterError


def make_inventory() -> Inventory:
    """Return a small resolved inventory graph."""
    interrogator = Interrogator(
        resource_id="interrogator_1",
        interrogator_id="IU001",
        model="DAS",
        serial_number="SN001",
    )
    configuration = AcquisitionConfiguration(
        resource_id="cfg_1",
        interrogator=interrogator,
        sample_rate=100.0,
    )
    crs = CoordinateReferenceSystem(resource_id="crs_1", axis_order=("x",))
    geometry = Geometry(
        resource_id="geo_1",
        length=2.0,
        geometry_type="linear",
        coordinate_reference_system=crs,
        coordinates=((0.0,), (2.0,)),
    )
    fiber = FiberSegment(resource_id="fiber_1", length=2.0)
    annotation = OpticalPathAnnotation(
        resource_id="anno_1",
        start_distance=0.0,
        end_distance=2.0,
        label="well",
    )
    path = OpticalPath(
        resource_id="path_1",
        length=2.0,
        optical_components=(fiber,),
        geometries=(geometry,),
        annotations=(annotation,),
    )
    fiber_array = FiberArray(
        resource_id="fiber_array_1",
        acquisition_configuration=configuration,
        optical_path=path,
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
        interrogator = inventory.get_record_dicts(record_ids="interrogator_1")[0]
        path = inventory.get_record_dicts(record_ids="path_1")[0]

        assert len(records) == 9
        assert network["fiber_array_ids"] == ("fiber_array_1",)
        assert fiber_array["acquisition_configuration_id"] == "cfg_1"
        assert fiber_array["optical_path_id"] == "path_1"
        assert "acquisition_configuration" not in fiber_array
        assert config["interrogator_resource_id"] == "interrogator_1"
        assert "interrogator" not in config
        assert interrogator["interrogator_id"] == "IU001"
        assert path["geometry_ids"] == ("geo_1",)
        assert path["annotation_ids"] == ("anno_1",)
        assert "geometries" not in path

    def test_records_property_cannot_mutate_storage(self):
        """The records property should not expose mutable internal state."""
        inventory = make_inventory()
        record = inventory.records[0]

        record["resource_id"] = "changed"

        assert inventory.get_record_dicts()[0]["resource_id"] != "changed"

    def test_get_records_resolves_relationships(self):
        """Resolved records should expose object links, not id fields."""
        network = make_inventory().get_records(record_ids="network_1")[0]
        fiber_array = make_inventory().get_records(record_ids="fiber_array_1")[0]

        assert network.fiber_arrays[0].resource_id == "fiber_array_1"
        interrogator = fiber_array.acquisition_configuration.interrogator
        assert interrogator.model == "DAS"
        assert interrogator.interrogator_id == "IU001"
        assert interrogator.resource_id == "interrogator_1"
        crs = fiber_array.optical_path.geometries[0].coordinate_reference_system
        assert crs.axis_order == ("x",)
        assert fiber_array.optical_path.annotations[0].label == "well"
        assert "acquisition_configuration_id" not in fiber_array.__class__.model_fields

    def test_shared_linked_records_are_stored_once(self):
        """Repeated linked objects should not duplicate canonical records."""
        interrogator = Interrogator(resource_id="interrogator_1")
        config_1 = AcquisitionConfiguration(
            resource_id="cfg_1", interrogator=interrogator
        )
        config_2 = AcquisitionConfiguration(
            resource_id="cfg_2", interrogator=interrogator
        )
        inventory = Inventory(
            records=(
                FiberArray(
                    resource_id="fiber_array_1",
                    acquisition_configuration=config_1,
                ),
                FiberArray(
                    resource_id="fiber_array_2",
                    acquisition_configuration=config_2,
                ),
            )
        )

        interrogators = inventory.get_record_dicts(record_types=Interrogator)

        assert len(interrogators) == 1

    def test_get_records_filters_by_type(self):
        """Type filters should return current records of the requested classes."""
        records = make_inventory().get_records(
            record_types=(Geometry, OpticalPathAnnotation)
        )

        assert {record.type for record in records} == {
            "Geometry",
            "OpticalPathAnnotation",
        }

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
        assert isinstance(latest.optical_path, OpticalPath)

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
                        "optical_path_id": "missing",
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
                        "optical_path": {
                            "type": "OpticalPath",
                            "resource_id": "path_1",
                        },
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

    def test_validate_lengths(self):
        """Track lengths should validate against path length."""
        path = make_inventory().get_records(record_ids="path_1")[0]

        assert path.validate_lengths() is path

    def test_select_distance(self):
        """Selecting a distance interval should clip child tracks."""
        path = make_inventory().get_records(record_ids="path_1")[0]
        out = path.select(distance=(0.5, 1.5))

        assert out.length == 1.0
        assert out.geometries[0].length == 1.0
        assert out.annotations[0].start_distance == 0.0
        assert out.annotations[0].end_distance == 1.0

    def test_reverse(self):
        """Reversing a path should reverse child tracks."""
        path = make_inventory().get_records(record_ids="path_1")[0]
        out = path.reverse()

        assert out.length == path.length
        assert out.annotations[0].start_distance == 0.0
        assert out.annotations[0].end_distance == 2.0
