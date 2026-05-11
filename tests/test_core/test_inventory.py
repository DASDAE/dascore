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
        crs=crs,
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


class TestInventory:
    """Generic tests for Inventory object"""





class TestInventoryNew:
    """Tests for new method of inventory."""
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


class TestInventoryDictConversion:
    """Tests for serializing inv to/from dict."""



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


class TestInventoryJsonSerialization:
    """Tests for reading inventory to and from json. """

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


class TestInventoryYAML:
    """Tests for reading inventory to and from yaml. """


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


class TestAnnotations:
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
        assert out.crs == path.crs
        assert out.geometries[0].optical_length == 1.0
        assert out.annotations[0].distance == (0.0, 1.0)

    def test_reverse(self):
        """Reversing a path should reverse child tracks."""
        path = get_path(make_inventory())
        out = path.reverse()

        assert out.optical_length == path.optical_length
        assert out.crs == path.crs
        assert out.annotations[0].distance == (0.0, 2.0)

    def test_add_preserves_matching_crs(self):
        """Concatenating paths should keep the shared CRS."""
        path = get_path(make_inventory())
        out = path + path

        assert out.optical_length == 4.0
        assert out.crs == path.crs

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
