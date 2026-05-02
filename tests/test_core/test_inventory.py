"""Tests for inventory metadata models."""

from __future__ import annotations

from copy import deepcopy
from io import StringIO

import numpy as np
import pandas as pd
import pytest
from pydantic import Field, ValidationError

import dascore as dc
from dascore.core.inventory import (
    Acquisition,
    Cable,
    ClampPoint,
    Connector,
    CoordinateReferenceSystem,
    Coupler,
    CouplingCondition,
    FiberSegment,
    Geometry,
    Instrument,
    InstrumentConfiguration,
    Inventory,
    InventoryRecord,
    OpticalPath,
    OpticalPathAnnotation,
    Splice,
    Terminator,
    Turnaround,
)
from dascore.exceptions import ParameterError
from dascore.utils.inventory import _DistanceInterval


def make_annotations(*labels_lengths):
    """Return annotations from ordered (label, length) pairs."""
    start = 0.0
    out = []
    for num, (label, length) in enumerate(labels_lengths):
        stop = start + float(length)
        out.append(
            OpticalPathAnnotation(
                resource_id=f"annotation_{num}",
                start_distance=start,
                end_distance=stop,
                label=label,
            )
        )
        start = stop
    return tuple(out)


def get_mixed_manifest():
    """Return a small manifest with several inventory object types."""
    return {
        "dascore_inventory": True,
        "schema_version": 1,
        "inventory_id": "inv_test",
        "author_id": "field_team",
        "objects": [
            {
                "type": "Acquisition",
                "resource_id": "acq_001",
                "instrument_configuration_id": "inst_cfg_001",
                "optical_path_id": "optical_path_001",
                "network": "XX",
                "station": "DAS01",
                "tag": "raw",
            },
            {
                "type": "InstrumentConfiguration",
                "resource_id": "inst_cfg_001",
                "instrument_id": "instrument_001",
                "gauge_length": 10.0,
            },
            {
                "type": "Instrument",
                "resource_id": "instrument_001",
                "instrument_id": "serial_001",
                "instrument_type": "das_interrogator",
            },
            {
                "type": "OpticalPath",
                "resource_id": "optical_path_001",
                "length": 1000.0,
                "optical_component_ids": [
                    "fiber_001",
                    "connector_001",
                    "splice_001",
                    "coupler_001",
                    "turnaround_001",
                    "terminator_001",
                ],
                "geometry_ids": ["geometry_001", "geometry_002"],
                "coupling_condition_ids": ["coupling_001", "coupling_002"],
                "annotation_ids": ["annotation_001", "annotation_002"],
            },
            {
                "type": "OpticalPathAnnotation",
                "resource_id": "annotation_001",
                "start_distance": 0.0,
                "end_distance": 30.0,
                "label": "borehole 12",
            },
            {
                "type": "OpticalPathAnnotation",
                "resource_id": "annotation_002",
                "start_distance": 30.0,
                "end_distance": 1000.0,
                "label": "surface cable",
            },
            {
                "type": "Cable",
                "resource_id": "cable_001",
                "cable_id": "field_cable_001",
                "manufacturer": "Cable Co",
                "model": "Armored SMF",
                "serial_number": "SN-001",
                "cable_type": "armored",
                "specification": "ITU-T G.652.D",
                "cable_construction": "loose_tube_armored",
                "jacket_material": "HDPE",
                "armor_type": "corrugated_steel",
                "outer_diameter": 0.012,
                "minimum_bend_radius": 0.24,
                "maximum_tensile_load": 2700.0,
                "datasheet_uri": "https://example.com/cable.pdf",
                "fiber_count": 4,
            },
            {
                "type": "FiberSegment",
                "resource_id": "fiber_001",
                "length": 998.5,
                "fiber_type": "single_mode",
                "fiber_standard": "ITU-T G.652.D",
                "fiber_index": 1,
                "subunit_id": "tube_1",
                "buffer_type": "loose_buffered",
                "buffer_material": "acrylate",
                "buffer_outer_diameter": 0.00025,
                "color": "blue",
                "group_refractive_index": 1.4682,
                "attenuation_coefficient": 0.2,
                "container_id": "cable_001",
            },
            {
                "type": "Connector",
                "resource_id": "connector_001",
                "length": 0.2,
                "connector_type": "LC",
                "insertion_loss": 0.1,
            },
            {
                "type": "Splice",
                "resource_id": "splice_001",
                "length": 0.0,
                "splice_type": "fusion",
                "insertion_loss": 0.02,
            },
            {
                "type": "Coupler",
                "resource_id": "coupler_001",
                "length": 0.3,
                "coupler_type": "splitter",
                "coupling_ratio": "50/50",
            },
            {
                "type": "Turnaround",
                "resource_id": "turnaround_001",
                "length": 1.0,
                "turnaround_type": "loopback",
            },
            {
                "type": "Terminator",
                "resource_id": "terminator_001",
                "length": 0.0,
                "terminator_type": "angled",
                "reflectance": -60.0,
            },
            {
                "type": "CoordinateReferenceSystem",
                "resource_id": "crs_local_001",
                "crs_type": "local",
                "name": "Field local ENU",
                "definition": "local:field_enu",
                "origin": (500000.0, 4100000.0, 1200.0),
                "axis_order": ("x", "y", "z"),
                "units": "m",
                "vertical_datum": "local",
            },
            {
                "type": "Geometry",
                "resource_id": "geometry_001",
                "geometry_type": "linear",
                "length": 400.0,
                "coordinate_reference_system_id": "crs_local_001",
                "coordinates": [(0.0, 0.0, 0.0), (400.0, 0.0, -10.0)],
            },
            {
                "type": "Geometry",
                "resource_id": "geometry_002",
                "geometry_type": "curve",
                "length": 600.0,
                "coordinate_reference_system_id": "crs_local_001",
                "coordinates": [
                    (400.0, 0.0, -10.0),
                    (600.0, 200.0, -20.0),
                    (800.0, 500.0, -30.0),
                ],
            },
            {
                "type": "CouplingCondition",
                "resource_id": "coupling_001",
                "coupling_type": "buried",
                "medium": "soil",
                "length": 700.0,
                "quality": "good",
                "deployment_type": "trench",
                "installation_method": "buried",
                "contact_medium": "soil",
                "embedment_depth": 1.0,
                "confining_pressure": 12000.0,
                "tension": 100.0,
                "coupling_quality_index": 0.8,
                "quality_method": "tap_test",
            },
            {
                "type": "CouplingCondition",
                "resource_id": "coupling_002",
                "coupling_type": "clamped",
                "medium": "steel",
                "attachment": "clamped",
                "length": 300.0,
                "deployment_type": "borehole",
                "installation_method": "clamped",
                "contact_medium": "tubing",
                "clamp_spacing": 10.0,
                "clamp_points": [
                    {
                        "distance": 0.0,
                        "clamp_type": "band",
                        "attachment": "bolted",
                    },
                    {
                        "distance": 10.0,
                        "clamp_type": "band",
                        "attachment": "bolted",
                    },
                ],
            },
        ],
    }


class TestInventoryRecords:
    """Tests for inventory record models."""

    def test_base_record_semantics(self):
        """Inventory records should expose stable identity and schema fields."""
        record = InventoryRecord(resource_id="thing_1")
        assert record.resource_id == "thing_1"
        assert record.schema_version == 1
        assert pd.isnull(record.effective_time)

    def test_resource_id_is_auto_generated(self):
        """Inventory records should get an id when one is not supplied."""
        record = Acquisition()
        assert record.resource_id.startswith("inventory_")

    def test_records_are_immutable(self):
        """Inventory records should not be mutated in-place."""
        record = Acquisition(resource_id="acq_001")
        with pytest.raises(ValidationError):
            record.resource_id = "new"

    def test_valid_time_interval(self):
        """Records should know whether they are effective at a time."""
        record = FiberSegment(
            resource_id="fiber_001",
            effective_time="2024-01-01",
        )
        assert record.is_effective_at(None)
        assert record.is_effective_at("2024-01-15")
        assert not record.is_effective_at("2023-12-31")

    def test_empty_string_time_normalizes_to_nat(self):
        """DateTime64 should normalize empty string inputs to NaT."""
        record = FiberSegment(resource_id="fiber_001", effective_time="")
        assert pd.isnull(record.effective_time)

    def test_fields_are_self_describing(self):
        """Inventory record models should describe their public fields."""
        summary = Acquisition.get_summary_df()
        assert "resource_id" in summary.index
        assert "instrument_configuration_id" in summary.index
        assert "optical_path_id" in summary.index
        assert summary.loc["resource_id", "description"]
        assert summary.loc["instrument_configuration_id", "description"]

    def test_optical_records_are_self_describing(self):
        """Optical component record models should describe their public fields."""
        optical_path_summary = OpticalPath.get_summary_df()
        cable_summary = Cable.get_summary_df()
        fiber_summary = FiberSegment.get_summary_df()
        config_summary = InstrumentConfiguration.get_summary_df()
        coupling_summary = CouplingCondition.get_summary_df()
        crs_summary = CoordinateReferenceSystem.get_summary_df()
        clamp_summary = ClampPoint.get_summary_df()
        assert "length" in optical_path_summary.index
        assert "optical_component_ids" in optical_path_summary.index
        assert "annotation_ids" in optical_path_summary.index
        assert "cable_construction" in cable_summary.index
        assert "minimum_bend_radius" in cable_summary.index
        assert "buffer_type" in fiber_summary.index
        assert "group_refractive_index" in fiber_summary.index
        assert "attenuation_coefficient" in fiber_summary.index
        assert "first_channel_index" in config_summary.index
        assert "first_channel_distance" in config_summary.index
        assert "coupling_type" in coupling_summary.index
        assert "deployment_type" in coupling_summary.index
        assert "installation_method" in coupling_summary.index
        assert "contact_medium" in coupling_summary.index
        assert "clamp_spacing" in coupling_summary.index
        assert "clamp_points" in coupling_summary.index
        assert "distance" in clamp_summary.index
        assert "clamp_type" in clamp_summary.index
        assert "definition" in crs_summary.index
        assert "axis_order" in crs_summary.index
        assert optical_path_summary.loc["optical_component_ids", "description"]
        assert cable_summary.loc["cable_construction", "description"]
        assert fiber_summary.loc["buffer_type", "description"]
        assert config_summary.loc["first_channel_distance", "description"]
        assert coupling_summary.loc["clamp_points", "description"]
        assert clamp_summary.loc["distance", "description"]


class TestInventory:
    """Tests for the Inventory container."""

    def test_manifest_dispatches_known_types(self):
        """Known object types should validate into specific models."""
        inv = Inventory.model_validate(get_mixed_manifest())
        assert isinstance(inv.objects[0], Acquisition)
        assert isinstance(inv.objects[1], InstrumentConfiguration)
        assert isinstance(inv.objects[2], Instrument)
        assert isinstance(inv.objects[3], OpticalPath)
        assert isinstance(inv.get_records(record_type=Cable)[0], Cable)
        assert isinstance(inv.get_records(record_type=FiberSegment)[0], FiberSegment)
        assert isinstance(inv.get_records(record_type=Connector)[0], Connector)
        assert isinstance(inv.get_records(record_type=Splice)[0], Splice)
        assert isinstance(inv.get_records(record_type=Coupler)[0], Coupler)
        assert isinstance(inv.get_records(record_type=Turnaround)[0], Turnaround)
        assert isinstance(inv.get_records(record_type=Terminator)[0], Terminator)
        assert isinstance(
            inv.get_records(record_type=CoordinateReferenceSystem)[0],
            CoordinateReferenceSystem,
        )
        assert len(inv.get_records(record_type=Geometry)) == 2
        assert isinstance(
            inv.get_records(record_type=CouplingCondition)[0],
            CouplingCondition,
        )
        assert inv.objects[3].optical_component_ids == (
            "fiber_001",
            "connector_001",
            "splice_001",
            "coupler_001",
            "turnaround_001",
            "terminator_001",
        )
        cable = inv.get_records(record_type=Cable)[0]
        fiber = inv.get_records(record_type=FiberSegment)[0]
        couplings = inv.get_records(record_type=CouplingCondition)
        crs = inv.get_records(record_type=CoordinateReferenceSystem)[0]
        geometry = inv.get_records(record_type=Geometry)[0]
        assert fiber.container_id == "cable_001"
        assert cable.minimum_bend_radius == 0.24
        assert cable.datasheet_uri == "https://example.com/cable.pdf"
        assert fiber.buffer_type == "loose_buffered"
        assert fiber.group_refractive_index == 1.4682
        assert fiber.attenuation_coefficient == 0.2
        assert couplings[0].deployment_type == "trench"
        assert couplings[1].clamp_spacing == 10.0
        assert isinstance(couplings[1].clamp_points[0], ClampPoint)
        assert couplings[1].clamp_points[1].distance == 10.0
        assert crs.axis_order == ("x", "y", "z")
        assert geometry.coordinates[0] == (0.0, 0.0, 0.0)
        assert isinstance(
            inv.get_records(record_type=OpticalPathAnnotation)[0],
            OpticalPathAnnotation,
        )

    def test_unknown_type_raises(self):
        """Unknown object types should be rejected."""
        with pytest.raises(ValidationError, match="Unknown inventory object type"):
            Inventory(objects=[{"type": "CustomThing", "resource_id": "custom_001"}])

    def test_channel_map_is_not_registered(self):
        """Channel maps are no longer core inventory object types."""
        with pytest.raises(ValidationError, match="Unknown inventory object type"):
            Inventory(
                objects=[{"type": "ChannelMap", "resource_id": "channel_map_001"}]
            )

    def test_manifest_object_without_resource_id_raises(self):
        """Objects loaded from mappings should require explicit resource ids."""
        with pytest.raises(ValidationError, match="resource_id"):
            Inventory(objects=[{"type": "FiberSegment"}])

    def test_manifest_object_without_type_raises(self):
        """Objects loaded from mappings should require explicit types."""
        with pytest.raises(ValidationError, match="include a type"):
            Inventory(objects=[{"resource_id": "fiber_001"}])

    def test_add_mapping_without_type_raises(self):
        """Inventory.add should require type on plain mappings."""
        with pytest.raises(ValueError, match="include a type"):
            Inventory().add({"resource_id": "fiber_001"})

    def test_inventory_accepts_python_object_without_resource_id(self):
        """Direct Python constructors can still auto-generate resource ids."""
        inv = Inventory(objects=[FiberSegment()])
        assert inv.objects[0].resource_id.startswith("inventory_")

    def test_cable_segment_is_not_registered(self):
        """CableSegment is not part of the active inventory schema."""
        with pytest.raises(ValidationError, match="Unknown inventory object type"):
            Inventory(objects=[{"type": "CableSegment", "resource_id": "cable_001"}])

    def test_optical_length_units_are_rejected(self):
        """Optical path lengths are SI-only and should not accept unit fields."""
        with pytest.raises(ValidationError, match="length_units"):
            Inventory(
                objects=[
                    {
                        "type": "FiberSegment",
                        "resource_id": "fiber_001",
                        "length": 100.0,
                        "length_units": "ft",
                    }
                ]
            )

    def test_fiber_segment_manufacturer_is_rejected(self):
        """Cable-level identity fields should live on Cable objects."""
        with pytest.raises(ValidationError, match="manufacturer"):
            Inventory(
                objects=[
                    {
                        "type": "FiberSegment",
                        "resource_id": "fiber_001",
                        "manufacturer": "Cable Co",
                    }
                ]
            )

    def test_instrument_configuration_units_are_rejected(self):
        """Instrument configuration numeric fields are SI-only."""
        with pytest.raises(ValidationError, match="gauge_length_units"):
            Inventory(
                objects=[
                    {
                        "type": "InstrumentConfiguration",
                        "resource_id": "cfg_001",
                        "gauge_length": 10.0,
                        "gauge_length_units": "m",
                    }
                ]
            )

    def test_contents_and_classes(self):
        """Inventory should index records by resource id and object type."""
        inv = Inventory.model_validate(get_mixed_manifest())
        assert inv.contents["acq_001"] == (inv.objects[0],)
        assert inv.classes["Acquisition"] == ("acq_001",)
        assert inv.classes["OpticalPath"] == ("optical_path_001",)
        assert inv.classes["Cable"] == ("cable_001",)
        assert inv.contents is inv.contents
        assert inv.classes is inv.classes

    def test_get_records_filters_by_type_class(self):
        """Inventory record filtering should accept record classes."""
        inv = Inventory.model_validate(get_mixed_manifest())
        assert inv.get_records(record_type=Acquisition) == (inv.objects[0],)

    def test_view_filters_by_type(self):
        """Record views should optionally filter by record type."""
        inv = Inventory(
            objects=[
                Acquisition(resource_id="shared_id"),
                FiberSegment(resource_id="shared_id", fiber_type="single_mode"),
            ]
        )
        assert isinstance(
            inv.view("shared_id", record_type=Acquisition), Acquisition
        )
        assert isinstance(
            inv.view("shared_id", record_type=FiberSegment), FiberSegment
        )

    def test_view_without_type_uses_latest_matching_record(self):
        """Unfiltered record views should use the latest matching resource id."""
        inv = Inventory(
            objects=[
                Acquisition(resource_id="shared_id", tag="old"),
                FiberSegment(resource_id="shared_id", fiber_type="single_mode"),
            ]
        )
        assert isinstance(inv.view("shared_id"), FiberSegment)

    def test_view_unmapped_plural_relationship_raises(self):
        """Plural id fields should require an explicit relationship mapping."""

        class UnmappedPluralRecord(InventoryRecord):
            """Record with an intentionally unmapped plural relationship."""

            widget_ids: tuple[str, ...] = ()
            widgets: tuple[InventoryRecord, ...] = Field(default=(), exclude=True)

        record = UnmappedPluralRecord(
            resource_id="record_1",
            widget_ids=("widget_1",),
        )
        with pytest.raises(ParameterError, match="plural mapping"):
            Inventory().view(record)

    def test_view_missing_record_with_type_mentions_type(self):
        """Missing view lookups should include requested type when supplied."""
        inv = Inventory()
        with pytest.raises(KeyError, match="record_type"):
            inv.view("missing", record_type=FiberSegment)

    def test_view_time_before_effective_record_raises(self):
        """Time-aware record views should raise before any effective record."""
        inv = Inventory(
            objects=[
                FiberSegment(
                    resource_id="fiber_001",
                    effective_time="2024-01-01",
                ),
            ]
        )
        with pytest.raises(KeyError, match="effective"):
            inv.view("fiber_001", time="2023-01-01")

    def test_inventory_model_validate_accepts_inventory_instance(self):
        """Inventory validation should pass through existing Inventory objects."""
        inv = Inventory()
        assert Inventory.model_validate(inv) is inv

    def test_inventory_model_validate_rejects_non_mapping(self):
        """Inventory validation should pass non-mapping inputs to pydantic."""
        with pytest.raises(ValidationError):
            Inventory.model_validate("not an inventory")

    def test_optical_path_sequences_are_independent(self):
        """Optical paths should keep independent ordered component sequences."""
        inv = Inventory.model_validate(get_mixed_manifest())
        optical_path = inv.view("optical_path_001")
        assert isinstance(optical_path, OpticalPath)
        assert optical_path.length == 1000.0
        assert optical_path.optical_component_ids == (
            "fiber_001",
            "connector_001",
            "splice_001",
            "coupler_001",
            "turnaround_001",
            "terminator_001",
        )
        assert optical_path.geometry_ids == ("geometry_001", "geometry_002")
        assert optical_path.coupling_condition_ids == (
            "coupling_001",
            "coupling_002",
        )
        assert tuple(label.label for label in optical_path.annotations) == (
            "borehole 12",
            "surface cable",
        )
        assert sum(
            label.end_distance - label.start_distance
            for label in optical_path.annotations
        ) == 1000.0

    def test_view_gets_latest_appended_record(self):
        """Record views should use the last matching resource id."""
        inv = Inventory(
            objects=[
                Acquisition(resource_id="acq_001", tag="old"),
                Acquisition(resource_id="acq_001", tag="new"),
            ]
        )
        assert inv.view("acq_001").tag == "new"

    def test_view_can_filter_by_time(self):
        """Time-aware lookup should use effective time before append order."""
        inv = Inventory(
            objects=[
                FiberSegment(
                    resource_id="fiber_001",
                    fiber_type="before",
                    effective_time="2024-01-01",
                ),
                FiberSegment(
                    resource_id="fiber_001",
                    fiber_type="after",
                    effective_time="2024-02-01",
                ),
            ]
        )
        assert inv.view("fiber_001", time="2024-01-15").fiber_type == "before"
        assert inv.view("fiber_001", time="2024-02-15").fiber_type == "after"

    def test_view_ties_use_append_order(self):
        """Later records should supersede earlier records at the same time."""
        inv = Inventory(
            objects=[
                FiberSegment(
                    resource_id="fiber_001",
                    fiber_type="initial",
                    effective_time="2024-01-01",
                ),
                FiberSegment(
                    resource_id="fiber_001",
                    fiber_type="corrected",
                    effective_time="2024-01-01",
                ),
            ]
        )
        assert inv.view("fiber_001", time="2024-01-15").fiber_type == "corrected"

    def test_validate_references_accepts_valid_inventory(self):
        """Reference validation should pass for the mixed manifest."""
        inv = Inventory.model_validate(get_mixed_manifest())
        assert inv.validate_references() is inv

    def test_validate_references_ignores_empty_ids(self):
        """Empty reference ids should be treated as absent references."""
        inv = Inventory(objects=[Acquisition(resource_id="acq_001")])
        assert inv.validate_references() is inv

    def test_validate_references_raises_on_missing_reference(self):
        """Reference validation should report missing referenced records."""
        manifest = deepcopy(get_mixed_manifest())
        manifest["objects"][0]["optical_path_id"] = "missing_path"
        inv = Inventory.model_validate(manifest)
        with pytest.raises(ValueError, match="missing resource_id 'missing_path'"):
            inv.validate_references()

    def test_validate_references_raises_on_wrong_type(self):
        """Reference validation should report wrong-type referenced records."""
        manifest = deepcopy(get_mixed_manifest())
        manifest["objects"][3]["optical_component_ids"] = ["geometry_001"]
        inv = Inventory.model_validate(manifest)
        with pytest.raises(ValueError, match="expected Connector"):
            inv.validate_references()

    def test_validate_references_raises_on_wrong_geometry_type(self):
        """Reference validation should check geometry id target types."""
        manifest = deepcopy(get_mixed_manifest())
        manifest["objects"][3]["geometry_ids"] = ["fiber_001"]
        inv = Inventory.model_validate(manifest)
        with pytest.raises(ValueError, match="expected Geometry"):
            inv.validate_references()

    def test_validate_references_raises_on_wrong_coupling_type(self):
        """Reference validation should check coupling id target types."""
        manifest = deepcopy(get_mixed_manifest())
        manifest["objects"][3]["coupling_condition_ids"] = ["geometry_001"]
        inv = Inventory.model_validate(manifest)
        with pytest.raises(ValueError, match="expected CouplingCondition"):
            inv.validate_references()

    def test_validate_references_can_filter_by_time(self):
        """Time-filtered validation should ignore future records."""
        inv = Inventory(
            objects=[
                OpticalPath(
                    resource_id="optical_path_001",
                    optical_component_ids=("missing_future_fiber",),
                    effective_time="2024-02-01",
                )
            ]
        )
        assert inv.validate_references(time="2024-01-01") is inv
        with pytest.raises(ValueError, match="missing_future_fiber"):
            inv.validate_references(time="2024-02-01")

    def test_view_populates_relationship_fields(self):
        """Viewing a record should populate runtime relationship fields."""
        inv = Inventory.model_validate(get_mixed_manifest())
        acq = inv.view("acq_001")
        assert isinstance(acq, Acquisition)
        assert isinstance(acq.instrument_configuration, InstrumentConfiguration)
        assert isinstance(acq.instrument_configuration.instrument, Instrument)
        assert isinstance(acq.optical_path, OpticalPath)
        assert isinstance(acq.optical_path.optical_components[0], FiberSegment)
        assert isinstance(acq.optical_path.optical_components[0].container, Cable)
        assert isinstance(acq.optical_path.geometries[0], Geometry)
        assert isinstance(
            acq.optical_path.geometries[0].coordinate_reference_system,
            CoordinateReferenceSystem,
        )
        assert isinstance(acq.optical_path.coupling_conditions[0], CouplingCondition)

    def test_view_can_stop_after_immediate_relationships(self):
        """Non-recursive resolution should populate only immediate relationships."""
        inv = Inventory.model_validate(get_mixed_manifest())
        acq = inv.view("acq_001", recursive=False)
        assert isinstance(acq.instrument_configuration, InstrumentConfiguration)
        assert acq.instrument_configuration.instrument is None
        assert isinstance(acq.optical_path, OpticalPath)
        assert not acq.optical_path.optical_components

    def test_view_missing_reference_returns_none(self):
        """Soft resolution should leave missing scalar references as None."""
        acq = Acquisition(resource_id="acq_001", optical_path_id="missing_path")
        inv = Inventory(objects=[acq])
        resolved = inv.view("acq_001")
        assert resolved.optical_path is None

    def test_view_relationship_fields_are_not_dumped(self):
        """Runtime relationships should not be included in manifest dumps."""
        inv = Inventory.model_validate(get_mixed_manifest())
        resolved = inv.view("acq_001")
        dump = resolved.model_dump(mode="json")
        assert "instrument_configuration" not in dump
        assert "optical_path" not in dump
        out = inv.add(resolved)
        text = out.to_yaml()
        assert "instrument_configuration:" not in text
        assert "optical_path:" not in text

    def test_optical_paths_can_be_concatenated(self):
        """Adding optical paths should concatenate component sequences."""
        path_1 = OpticalPath(
            resource_id="path_1",
            length=10.0,
            optical_component_ids=("fiber_1",),
            geometry_ids=("geometry_1",),
            coupling_condition_ids=("coupling_1",),
            annotation_ids=("annotation_0",),
            annotations=make_annotations(("one", 10.0)),
        )
        path_2 = OpticalPath(
            resource_id="path_2",
            length=20.0,
            optical_component_ids=("fiber_2",),
            geometry_ids=("geometry_2",),
            coupling_condition_ids=("coupling_2",),
            annotation_ids=("annotation_0",),
            annotations=make_annotations(("two", 20.0)),
        )
        out = path_1 + path_2
        assert out.resource_id not in {path_1.resource_id, path_2.resource_id}
        assert out.length == 30.0
        assert out.optical_component_ids == ("fiber_1", "fiber_2")
        assert out.geometry_ids == ("geometry_1", "geometry_2")
        assert out.coupling_condition_ids == ("coupling_1", "coupling_2")
        assert tuple(label.label for label in out.annotations) == ("one", "two")

    def test_optical_path_concat_with_unknown_length(self):
        """Concatenated path length should be unknown if either input is unknown."""
        path_1 = OpticalPath(length=10.0)
        path_2 = OpticalPath(length=None)
        assert (path_1 + path_2).length is None

    def test_concat_unknown_left_length_preserves_left_annotations(self):
        """Left annotations do not need shifting when the right path has none."""
        annotations = make_annotations(("left", 5.0))
        path_1 = OpticalPath(length=None, annotations=annotations)
        path_2 = OpticalPath(length=10.0)
        out = path_1 + path_2
        assert out.length is None
        assert out.annotations == annotations
        assert out.annotation_ids == ("annotation_0",)

    def test_concat_unknown_left_length_rejects_right_annotations(self):
        """Right annotations cannot be shifted without the left path length."""
        path_1 = OpticalPath(length=None)
        path_2 = OpticalPath(length=10.0, annotations=make_annotations(("right", 5.0)))
        with pytest.raises(ParameterError, match="requires left path length"):
            path_1 + path_2

    def test_optical_path_concat_shifts_overlapping_annotations(self):
        """Adding paths should preserve overlapping annotation structure."""
        left_annotations = (
            OpticalPathAnnotation(
                resource_id="left_section",
                start_distance=0.0,
                end_distance=10.0,
                label="section",
                category="field_note",
            ),
            OpticalPathAnnotation(
                resource_id="left_quality",
                start_distance=2.0,
                end_distance=8.0,
                label="good coupling",
                category="quality",
            ),
        )
        right_annotations = (
            OpticalPathAnnotation(
                resource_id="right_section",
                start_distance=0.0,
                end_distance=10.0,
                label="section",
                category="field_note",
            ),
            OpticalPathAnnotation(
                resource_id="right_quality",
                start_distance=2.0,
                end_distance=8.0,
                label="good coupling",
                category="quality",
            ),
        )
        out = OpticalPath(length=10.0, annotations=left_annotations) + OpticalPath(
            length=10.0,
            annotations=right_annotations,
        )
        assert tuple(
            (x.label, x.category, x.start_distance, x.end_distance)
            for x in out.annotations
        ) == (
            ("section", "field_note", 0.0, 10.0),
            ("good coupling", "quality", 2.0, 8.0),
            ("section", "field_note", 10.0, 20.0),
            ("good coupling", "quality", 12.0, 18.0),
        )
        assert tuple(x.resource_id for x in out.annotations[:2]) == (
            "left_section",
            "left_quality",
        )
        assert {x.resource_id for x in out.annotations[2:]}.isdisjoint(
            {"right_section", "right_quality"}
        )

    def test_viewed_optical_paths_can_be_concatenated(self):
        """Runtime relationship tuples should also concatenate."""
        inv = Inventory.model_validate(get_mixed_manifest())
        path = inv.view("optical_path_001")
        out = path + path
        assert len(out.optical_components) == 2 * len(path.optical_components)
        assert len(out.geometries) == 2 * len(path.geometries)
        assert len(out.coupling_conditions) == 2 * len(path.coupling_conditions)

    def test_optical_path_add_rejects_other_objects(self):
        """Adding non-optical-path objects should raise TypeError."""
        with pytest.raises(TypeError):
            OpticalPath() + object()
        with pytest.raises(TypeError):
            1 + OpticalPath()

    def test_optical_paths_can_be_summed(self):
        """Summing optical paths should concatenate them."""
        paths = [
            OpticalPath(length=10.0, optical_component_ids=("fiber_1",)),
            OpticalPath(length=20.0, optical_component_ids=("fiber_2",)),
            OpticalPath(length=30.0, optical_component_ids=("fiber_3",)),
        ]
        out = sum(paths)
        assert out.length == 60.0
        assert out.optical_component_ids == ("fiber_1", "fiber_2", "fiber_3")

    def test_optical_path_reverse_flips_ordered_sequences(self):
        """Reversing optical paths should reverse ids and runtime records."""
        inv = Inventory.model_validate(get_mixed_manifest())
        path = inv.view("optical_path_001")
        out = path.reverse()
        assert out.length == path.length
        assert out.resource_id != path.resource_id
        assert out.optical_component_ids == tuple(reversed(path.optical_component_ids))
        assert tuple(x.resource_id for x in out.optical_components) == tuple(
            reversed(path.optical_component_ids)
        )
        assert out.geometry_ids == tuple(reversed(path.geometry_ids))
        assert tuple(x.resource_id for x in out.geometries) == tuple(
            reversed(path.geometry_ids)
        )
        assert out.coupling_condition_ids == tuple(
            reversed(path.coupling_condition_ids)
        )
        assert tuple(x.label for x in out.annotations) == (
            "surface cable",
            "borehole 12",
        )

    def test_optical_path_select_clips_ordered_sequences(self):
        """Selecting by distance should clip all populated interval sequences."""
        path = OpticalPath(
            length=200.0,
            optical_component_ids=("fiber_1", "splice_1", "fiber_2"),
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=100.0),
                Splice(resource_id="splice_1", length=50.0),
                FiberSegment(resource_id="fiber_2", length=50.0),
            ),
            geometry_ids=("geometry_1", "geometry_2"),
            geometries=(
                Geometry(resource_id="geometry_1", length=120.0),
                Geometry(resource_id="geometry_2", length=80.0),
            ),
            coupling_condition_ids=("coupling_1", "coupling_2"),
            coupling_conditions=(
                CouplingCondition(resource_id="coupling_1", length=150.0),
                CouplingCondition(resource_id="coupling_2", length=50.0),
            ),
            annotations=make_annotations(("lead-in", 100.0), ("borehole", 100.0)),
        )
        out = path.select(distance=(50.0, 175.0))
        assert out.length == 125.0
        assert out.optical_component_ids == ("fiber_1", "splice_1", "fiber_2")
        assert tuple(x.length for x in out.optical_components) == (50.0, 50.0, 25.0)
        assert out.geometry_ids == ("geometry_1", "geometry_2")
        assert tuple(x.length for x in out.geometries) == (70.0, 55.0)
        assert out.coupling_condition_ids == ("coupling_1", "coupling_2")
        assert tuple(x.length for x in out.coupling_conditions) == (100.0, 25.0)
        assert tuple(x.label for x in out.annotations) == ("lead-in", "borehole")
        assert tuple(
            x.end_distance - x.start_distance for x in out.annotations
        ) == (50.0, 75.0)
        assert out.annotations[0].resource_id != "annotation_0"
        assert out.annotations[1].resource_id != "annotation_1"

    def test_optical_path_select_preserves_unaffected_annotation_ids(self):
        """Selections should preserve ids only for unchanged annotations."""
        path = OpticalPath(
            length=20.0,
            annotations=make_annotations(("left", 10.0), ("right", 10.0)),
        )
        out = path.select(distance=(0.0, 10.0))
        assert tuple(x.label for x in out.annotations) == ("left",)
        assert out.annotations[0].resource_id == "annotation_0"

    def test_optical_path_select_clips_coupling_clamp_points(self):
        """Selecting paths should shift embedded coupling clamp points."""
        path = OpticalPath(
            length=100.0,
            coupling_conditions=(
                CouplingCondition(
                    resource_id="coupling_1",
                    length=100.0,
                    clamp_points=(
                        ClampPoint(distance=10.0, clamp_type="band"),
                        ClampPoint(distance=25.0, clamp_type="band"),
                        ClampPoint(distance=80.0, clamp_type="band"),
                    ),
                ),
            ),
        )
        out = path.select(distance=(20.0, 90.0))
        assert out.coupling_conditions[0].length == 70.0
        clamp_distances = tuple(
            point.distance for point in out.coupling_conditions[0].clamp_points
        )
        assert clamp_distances == (5.0, 60.0)

    def test_optical_path_split_does_not_duplicate_boundary_clamps(self):
        """Splitting paths should keep clamp points on only one side."""
        path = OpticalPath(
            length=100.0,
            coupling_conditions=(
                CouplingCondition(
                    resource_id="coupling_1",
                    length=100.0,
                    clamp_points=(
                        ClampPoint(distance=40.0, clamp_type="band"),
                        ClampPoint(distance=80.0, clamp_type="band"),
                    ),
                ),
            ),
        )
        left, right = path.split_at(40.0)
        assert not left.coupling_conditions[0].clamp_points
        assert tuple(
            point.distance for point in right.coupling_conditions[0].clamp_points
        ) == (0.0, 40.0)

    def test_optical_path_select_supports_open_end(self):
        """Open-ended distance selection should use the path length."""
        path = OpticalPath(
            length=20.0,
            annotations=make_annotations(("one", 10.0), ("two", 10.0)),
        )
        out = path.select(distance=(5.0, None))
        assert out.length == 15.0
        assert tuple(
            x.end_distance - x.start_distance for x in out.annotations
        ) == (5.0, 10.0)

    def test_optical_path_select_without_distance_returns_self(self):
        """Selecting without kwargs should return the same path."""
        path = OpticalPath(length=10.0)
        assert path.select() is path

    def test_optical_path_select_open_end_requires_length(self):
        """Open-ended selection needs path length."""
        with pytest.raises(ParameterError, match="require optical path length"):
            OpticalPath().select(distance=(0.0, None))

    def test_optical_path_select_rejects_reversed_range(self):
        """Distance selection should reject start after stop."""
        with pytest.raises(ParameterError, match="start must be"):
            OpticalPath(length=10.0).select(distance=(8.0, 2.0))

    def test_optical_path_select_includes_zero_length_intervals(self):
        """Selecting a point should keep zero-length intervals at that point."""
        path = OpticalPath(
            length=10.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=5.0),
                Splice(resource_id="splice_1", length=0.0),
                FiberSegment(resource_id="fiber_2", length=5.0),
            ),
        )
        out = path.select(distance=(5.0, 5.0))
        assert tuple(x.resource_id for x in out.optical_components) == ("splice_1",)
        assert out.optical_components[0].length == 0.0

    def test_optical_path_split_at_distance(self):
        """Optical paths should split into two selected paths."""
        path = OpticalPath(
            length=30.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=10.0),
                FiberSegment(resource_id="fiber_2", length=20.0),
            ),
            annotations=make_annotations(("one", 15.0), ("two", 15.0)),
        )
        left, right = path.split_at(12.0)
        assert left.length == 12.0
        assert right.length == 18.0
        assert tuple(x.length for x in left.optical_components) == (10.0, 2.0)
        assert tuple(x.length for x in right.optical_components) == (18.0,)
        left_2, right_2 = path.split(distance=12.0)
        assert left_2.length == left.length
        assert right_2.length == right.length

    def test_optical_path_select_requires_runtime_records_for_id_only_partial(self):
        """Id-only subcomponent sequences cannot be partially clipped."""
        path = OpticalPath(length=20.0, optical_component_ids=("fiber_1",))
        with pytest.raises(ParameterError, match="runtime records"):
            path.select(distance=(5.0, 10.0))
        out = path.select(distance=(0.0, 20.0))
        assert out.optical_component_ids == ("fiber_1",)

    def test_optical_path_select_rejects_mismatched_ids_and_records(self):
        """Runtime records and id tuples should pair one-to-one."""
        path = OpticalPath(
            length=10.0,
            optical_component_ids=("fiber_1", "fiber_2"),
            optical_components=(FiberSegment(resource_id="fiber_1", length=10.0),),
        )
        with pytest.raises(ParameterError, match="same length"):
            path.select(distance=(0.0, 10.0))

    def test_optical_path_select_requires_known_subcomponent_lengths(self):
        """Populated subcomponent sequences need lengths for distance selection."""
        path = OpticalPath(
            length=20.0,
            optical_components=(FiberSegment(resource_id="fiber_1"),),
        )
        with pytest.raises(ParameterError, match="requires all optical_components"):
            path.select(distance=(0.0, 10.0))

    def test_optical_path_select_rejects_unknown_argument(self):
        """Optical path selection should only accept distance for now."""
        with pytest.raises(ParameterError, match="Unsupported"):
            OpticalPath().select(time=(0, 1))

    def test_optical_path_select_rejects_curve_geometry(self):
        """Selecting path geometry should only support simple geometry for now."""
        path = OpticalPath(
            length=20.0,
            geometry_ids=("geometry_1",),
            geometries=(
                Geometry(
                    resource_id="geometry_1",
                    geometry_type="curve",
                    length=20.0,
                ),
            ),
        )
        with pytest.raises(ParameterError, match="linear or unknown geometry"):
            path.select(distance=(0.0, 10.0))

    def test_get_distance_interval_by_object_and_id(self):
        """Optical paths should report component intervals."""
        fiber_1 = FiberSegment(resource_id="fiber_1", length=10.0)
        fiber_2 = FiberSegment(resource_id="fiber_2", length=20.0)
        geometry = Geometry(
            resource_id="geometry_1",
            geometry_type="linear",
            length=30.0,
        )
        coupling = CouplingCondition(resource_id="coupling_1", length=30.0)
        path = OpticalPath(
            length=30.0,
            optical_component_ids=("fiber_1", "fiber_2"),
            optical_components=(fiber_1, fiber_2),
            geometry_ids=("geometry_1",),
            geometries=(geometry,),
            coupling_condition_ids=("coupling_1",),
            coupling_conditions=(coupling,),
        )
        assert path.get_distance_interval(fiber_2) == (10.0, 30.0)
        assert path.get_distance_interval("fiber_2", kind="optical_component") == (
            10.0,
            30.0,
        )
        assert path.get_distance_interval(geometry) == (0.0, 30.0)
        assert path.get_distance_interval("coupling_1", kind="coupling_condition") == (
            0.0,
            30.0,
        )

    def test_get_distance_interval_requires_kind_for_string_ids(self):
        """String ids need explicit target kind to avoid ambiguous lookup."""
        path = OpticalPath()
        with pytest.raises(ParameterError, match="String resource ids require kind"):
            path.get_distance_interval("fiber_1")
        with pytest.raises(ParameterError, match="kind must be"):
            path.get_distance_interval("fiber_1", kind="cable")
        with pytest.raises(ParameterError, match="Annotations are not valid"):
            path.get_distance_interval("annotation_1", kind="annotation")

    def test_get_distance_interval_rejects_annotation_objects(self):
        """Annotation objects should get an annotation-specific interval error."""
        path = OpticalPath()
        annotation = OpticalPathAnnotation(
            resource_id="annotation_1",
            start_distance=0.0,
            end_distance=10.0,
            label="lead-in",
        )
        with pytest.raises(ParameterError, match="Annotations are not valid"):
            path.get_distance_interval(annotation)

    def test_get_distance_interval_requires_known_record_lengths(self):
        """Interval lookup needs record lengths."""
        path = OpticalPath(
            optical_components=(FiberSegment(resource_id="fiber_1"),),
        )
        with pytest.raises(ParameterError, match="requires all records"):
            path.get_distance_interval("fiber_1", kind="optical_component")

    def test_get_distance_interval_reports_missing_target(self):
        """Interval lookup should report missing targets."""
        path = OpticalPath(
            optical_components=(FiberSegment(resource_id="fiber_1", length=10.0),),
        )
        with pytest.raises(ParameterError, match="was not found"):
            path.get_distance_interval("fiber_2", kind="optical_component")

    def test_remove_optical_component_by_object(self):
        """Removing an optical component should remove its distance interval."""
        fiber_1 = FiberSegment(resource_id="fiber_1", length=100.0)
        splice = Splice(resource_id="splice_1", length=50.0)
        fiber_2 = FiberSegment(resource_id="fiber_2", length=50.0)
        path = OpticalPath(
            length=200.0,
            optical_component_ids=("fiber_1", "splice_1", "fiber_2"),
            optical_components=(fiber_1, splice, fiber_2),
            geometry_ids=("geometry_1", "geometry_2"),
            geometries=(
                Geometry(
                    resource_id="geometry_1",
                    geometry_type="linear",
                    length=120.0,
                ),
                Geometry(
                    resource_id="geometry_2",
                    geometry_type="unknown",
                    length=80.0,
                ),
            ),
            coupling_condition_ids=("coupling_1", "coupling_2"),
            coupling_conditions=(
                CouplingCondition(resource_id="coupling_1", length=150.0),
                CouplingCondition(resource_id="coupling_2", length=50.0),
            ),
            annotations=make_annotations(("lead-in", 100.0), ("borehole", 100.0)),
        )
        out = path.remove_component(splice)
        assert out.length == 150.0
        assert out.optical_component_ids == ("fiber_1", "fiber_2")
        assert tuple(x.length for x in out.optical_components) == (100.0, 50.0)
        assert out.geometry_ids == ("geometry_1", "geometry_2")
        assert tuple(x.length for x in out.geometries) == (100.0, 50.0)
        assert out.coupling_condition_ids == ("coupling_1", "coupling_2")
        assert tuple(x.length for x in out.coupling_conditions) == (100.0, 50.0)
        assert tuple(x.label for x in out.annotations) == ("lead-in", "borehole")
        assert tuple(
            x.end_distance - x.start_distance for x in out.annotations
        ) == (100.0, 50.0)

    def test_remove_coupling_condition_by_id(self):
        """Component removal should accept explicit id keyword targets."""
        path = OpticalPath(
            length=30.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=10.0),
                FiberSegment(resource_id="fiber_2", length=10.0),
                FiberSegment(resource_id="fiber_3", length=10.0),
            ),
            coupling_condition_ids=("coupling_1", "coupling_2", "coupling_3"),
            coupling_conditions=(
                CouplingCondition(resource_id="coupling_1", length=10.0),
                CouplingCondition(resource_id="coupling_2", length=10.0),
                CouplingCondition(resource_id="coupling_3", length=10.0),
            ),
            annotations=make_annotations(
                ("one", 10.0),
                ("two", 10.0),
                ("three", 10.0),
            ),
        )
        out = path.remove_component(coupling_condition="coupling_2")
        assert out.length == 20.0
        assert tuple(x.resource_id for x in out.optical_components) == (
            "fiber_1",
            "fiber_3",
        )
        assert out.coupling_condition_ids == ("coupling_1", "coupling_3")
        assert tuple(x.label for x in out.annotations) == ("one", "three")
        assert out.annotations[0].resource_id == "annotation_0"
        assert out.annotations[1].resource_id != "annotation_2"

    def test_remove_component_splits_spanning_annotation(self):
        """Removing an interior component should split spanning annotations."""
        path = OpticalPath(
            length=30.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=10.0),
                FiberSegment(resource_id="fiber_2", length=10.0),
                FiberSegment(resource_id="fiber_3", length=10.0),
            ),
            annotations=make_annotations(("all", 30.0)),
        )
        out = path.remove_component(optical_component="fiber_2")
        assert out.length == 20.0
        assert tuple(x.label for x in out.annotations) == ("all", "all")
        assert tuple(
            (x.start_distance, x.end_distance) for x in out.annotations
        ) == ((0.0, 10.0), (10.0, 20.0))
        assert {x.resource_id for x in out.annotations}.isdisjoint({"annotation_0"})

    def test_optical_path_reverse_updates_annotation_ids(self):
        """Reversing path annotations should flip intervals and create new ids."""
        path = OpticalPath(
            length=30.0,
            annotations=make_annotations(("one", 10.0), ("two", 20.0)),
        )
        out = path.reverse()
        assert tuple(x.label for x in out.annotations) == ("two", "one")
        assert tuple(
            (x.start_distance, x.end_distance) for x in out.annotations
        ) == ((0.0, 20.0), (20.0, 30.0))
        assert {x.resource_id for x in out.annotations}.isdisjoint(
            {x.resource_id for x in path.annotations}
        )

    def test_optical_path_reverse_requires_length_for_annotations(self):
        """Annotated paths cannot be reversed without a known total length."""
        path = OpticalPath(length=None, annotations=make_annotations(("one", 10.0)))
        with pytest.raises(ParameterError, match="requires optical path length"):
            path.reverse()

    def test_optical_path_reverse_allows_unknown_length_without_annotations(self):
        """Unannotated paths can be reversed even when total length is unknown."""
        path = OpticalPath(length=None, optical_component_ids=("fiber_1", "fiber_2"))
        out = path.reverse()
        assert out.length is None
        assert out.optical_component_ids == ("fiber_2", "fiber_1")

    def test_optical_path_annotation_rejects_empty_interval(self):
        """Annotations must occupy a non-empty optical distance interval."""
        with pytest.raises(ValidationError, match="start_distance"):
            OpticalPathAnnotation(
                resource_id="annotation_1",
                start_distance=1.0,
                end_distance=1.0,
                label="empty",
            )

    def test_distance_interval_boundaries_are_half_open(self):
        """Touching intervals should not count as overlapping."""
        interval = _DistanceInterval(0.0, 10.0)
        assert interval.intersect(_DistanceInterval(10.0, 20.0)) is None
        assert interval.intersect(_DistanceInterval(5.0, 15.0)) == _DistanceInterval(
            5.0,
            10.0,
        )
        assert interval.reverse(30.0) == _DistanceInterval(20.0, 30.0)

    def test_remove_geometry_by_object(self):
        """Geometry targets should remove the matching geometry interval."""
        geometry = Geometry(
            resource_id="geometry_1",
            geometry_type="linear",
            length=10.0,
        )
        path = OpticalPath(
            length=20.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=10.0),
                FiberSegment(resource_id="fiber_2", length=10.0),
            ),
            geometry_ids=("geometry_1", "geometry_2"),
            geometries=(
                geometry,
                Geometry(
                    resource_id="geometry_2",
                    geometry_type="unknown",
                    length=10.0,
                ),
            ),
            coupling_conditions=(
                CouplingCondition(resource_id="coupling_1", length=20.0),
            ),
        )
        out = path.remove_component(geometry)
        assert out.length == 10.0
        assert tuple(x.resource_id for x in out.optical_components) == ("fiber_2",)
        assert out.geometry_ids == ("geometry_2",)
        assert tuple(x.length for x in out.coupling_conditions) == (10.0,)

    def test_remove_coupling_condition_by_object(self):
        """Coupling condition objects should be valid positional targets."""
        coupling = CouplingCondition(resource_id="coupling_1", length=10.0)
        path = OpticalPath(
            length=20.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=10.0),
                FiberSegment(resource_id="fiber_2", length=10.0),
            ),
            coupling_conditions=(
                coupling,
                CouplingCondition(resource_id="coupling_2", length=10.0),
            ),
        )
        out = path.remove_component(coupling)
        assert out.length == 10.0
        assert tuple(x.resource_id for x in out.optical_components) == ("fiber_2",)

    def test_remove_component_requires_one_target(self):
        """Removing path components should receive exactly one target."""
        path = OpticalPath()
        with pytest.raises(ParameterError, match="exactly one"):
            path.remove_component()
        with pytest.raises(ParameterError, match="exactly one"):
            path.remove_component(
                optical_component="fiber_1",
                geometry="geometry_1",
            )

    def test_remove_component_requires_runtime_records_for_id_only_path(self):
        """Id-only paths cannot locate removal intervals without record lengths."""
        path = OpticalPath(
            length=10.0,
            optical_component_ids=("fiber_1",),
        )
        with pytest.raises(ParameterError, match="requires runtime records"):
            path.remove_component(optical_component="fiber_1")

    def test_remove_component_rejects_positional_string(self):
        """Positional string ids are ambiguous and should use explicit keywords."""
        with pytest.raises(ParameterError, match="Pass string resource ids"):
            OpticalPath().remove_component("fiber_1")

    def test_validate_lengths_accepts_matching_sequences(self):
        """Length validation should pass for populated matching sequences."""
        path = OpticalPath(
            length=20.0,
            optical_components=(
                FiberSegment(resource_id="fiber_1", length=10.0),
                FiberSegment(resource_id="fiber_2", length=10.0),
            ),
            geometries=(
                Geometry(
                    resource_id="geometry_1",
                    geometry_type="linear",
                    length=20.0,
                ),
            ),
            coupling_conditions=(
                CouplingCondition(resource_id="coupling_1", length=20.0),
            ),
            annotations=make_annotations(("all", 20.0)),
        )
        assert path.validate_lengths() is path

    def test_validate_lengths_raises_on_mismatch(self):
        """Length validation should report mismatched populated sequences."""
        path = OpticalPath(
            length=20.0,
            optical_components=(FiberSegment(resource_id="fiber_1", length=10.0),),
        )
        with pytest.raises(ParameterError, match="optical_components length"):
            path.validate_lengths()

    def test_validate_lengths_reports_all_mismatches(self):
        """Length validation should report every mismatched populated sequence."""
        path = OpticalPath(
            length=20.0,
            optical_components=(FiberSegment(resource_id="fiber_1", length=10.0),),
            geometries=(
                Geometry(
                    resource_id="geometry_1",
                    geometry_type="linear",
                    length=12.0,
                ),
            ),
            coupling_conditions=(
                CouplingCondition(resource_id="coupling_1", length=14.0),
            ),
        )
        with pytest.raises(ParameterError) as exc_info:
            path.validate_lengths()
        text = str(exc_info.value)
        assert "optical_components length" in text
        assert "geometries length" in text
        assert "coupling_conditions length" in text

    def test_validate_lengths_can_infer_expected_length(self):
        """Length validation should compare populated sequences without path length."""
        path = OpticalPath(
            optical_components=(FiberSegment(resource_id="fiber_1", length=10.0),),
            geometries=(
                Geometry(
                    resource_id="geometry_1",
                    geometry_type="linear",
                    length=8.0,
                ),
            ),
        )
        with pytest.raises(ParameterError, match="geometries length"):
            path.validate_lengths()

    def test_validate_lengths_raises_on_unknown_record_length(self):
        """Length validation should require lengths on populated sequences."""
        path = OpticalPath(
            length=20.0,
            optical_components=(FiberSegment(resource_id="fiber_1"),),
        )
        with pytest.raises(ParameterError, match="all records must have length"):
            path.validate_lengths()

    def test_empty_optical_path_property(self):
        """Optical paths should expose a simple emptiness check."""
        assert OpticalPath().is_empty
        assert OpticalPath(length=0.0).is_empty
        assert not OpticalPath(length=10.0).is_empty
        assert not OpticalPath(optical_component_ids=("fiber_1",)).is_empty

    def test_add_returns_new_inventory(self):
        """Adding records should return a new inventory."""
        inv = Inventory()
        out = inv.add({"type": "Acquisition", "resource_id": "acq_001"})
        assert len(inv.objects) == 0
        assert isinstance(out.objects[0], Acquisition)

    def test_replace_at_appends_new_optical_path_state(self):
        """Replacing a record at a time should append the next resource state."""
        old_fiber = FiberSegment(resource_id="fiber_1", length=10.0)
        old_path = OpticalPath(
            resource_id="path_1",
            length=10.0,
            optical_component_ids=("fiber_1",),
        )
        new_fiber = FiberSegment(resource_id="fiber_2", length=20.0)
        new_path = OpticalPath(
            resource_id="temporary_id",
            length=20.0,
            optical_component_ids=("fiber_2",),
        )
        inv = Inventory(objects=[old_fiber, old_path])
        out = inv.replace_at(
            old_path,
            new_path,
            "2024-01-01",
            new_fiber,
            notes="Replaced damaged fiber.",
            author_id="field_team",
        )
        before = out.view("path_1", time="2023-12-31", record_type=OpticalPath)
        after = out.view("path_1", time="2024-01-01", record_type=OpticalPath)
        assert len(out.objects) == 4
        assert before == old_path
        assert after.resource_id == "path_1"
        assert after.effective_time == pd.Timestamp("2024-01-01").to_datetime64()
        assert after.notes == "Replaced damaged fiber."
        assert after.author_id == "field_team"
        assert after.length == 20.0
        assert out.view("fiber_2", record_type=FiberSegment) == new_fiber

    def test_replace_at_accepts_string_old_and_mapping_support_records(self):
        """Record replacement should accept string ids and mapping support records."""
        inv = Inventory(
            objects=[
                FiberSegment(resource_id="shared_id", length=10.0),
                Acquisition(resource_id="shared_id", tag="old"),
            ]
        )
        out = inv.replace_at(
            "shared_id",
            FiberSegment(resource_id="temporary_id", length=20.0),
            "2024-01-01",
            {
                "type": "Cable",
                "resource_id": "cable_1",
                "cable_id": "field_cable_1",
            },
            record_type=FiberSegment,
        )
        after = out.view("shared_id", time="2024-01-01", record_type=FiberSegment)
        assert after.length == 20.0
        assert isinstance(out.view("cable_1"), Cable)

    def test_replace_at_rejects_wrong_record_type(self):
        """Replacement states should have the same concrete record type."""
        inv = Inventory(objects=[Acquisition(resource_id="acq_1")])
        with pytest.raises(ParameterError, match="must also be Acquisition"):
            inv.replace_at("acq_1", FiberSegment(), "2024-01-01")

    def test_replace_at_can_skip_reference_validation(self):
        """Replacement can skip validation for intentionally partial inventories."""
        inv = Inventory(objects=[OpticalPath(resource_id="path_1")])
        new_path = OpticalPath(
            optical_component_ids=("missing_fiber",),
        )
        out = inv.replace_at("path_1", new_path, "2024-01-01", validate=False)
        assert out.view("path_1", time="2024-01-01").optical_component_ids == (
            "missing_fiber",
        )
        with pytest.raises(ValueError, match="missing_fiber"):
            inv.replace_at("path_1", new_path, "2024-01-01")

    def test_invalid_header_raises(self):
        """Inventory manifests must identify themselves as DASCore inventories."""
        with pytest.raises(ValidationError, match="dascore_inventory"):
            Inventory.model_validate({"dascore_inventory": False, "objects": []})

    def test_from_geometry_table_builds_geometry_inventory(self):
        """Geometry control points should build a path and linear geometries."""
        geometry = pd.DataFrame(
            {
                "distance": [0.0, 10.0, 25.0],
                "x": [1.0, 3.0, 6.0],
                "y": [2.0, 4.0, 8.0],
                "z": [0.0, -5.0, -10.0],
            }
        )
        inv = Inventory.from_geometry_table(geometry, id_prefix="survey")
        crs = inv.get_records(record_type=CoordinateReferenceSystem)[0]
        geometries = inv.get_records(record_type=Geometry)
        path = inv.get_records(record_type=OpticalPath)[0]
        assert crs.resource_id == "survey:crs"
        assert crs.axis_order == ("x", "y", "z")
        assert tuple(item.resource_id for item in geometries) == (
            "survey:geometry:0000",
            "survey:geometry:0001",
        )
        assert tuple(item.length for item in geometries) == (10.0, 15.0)
        assert geometries[0].coordinates == ((1.0, 2.0, 0.0), (3.0, 4.0, -5.0))
        assert geometries[1].coordinate_reference_system_id == "survey:crs"
        assert path.resource_id == "survey:optical_path"
        assert path.length == 25.0
        assert path.geometry_ids == (
            "survey:geometry:0000",
            "survey:geometry:0001",
        )

    def test_from_geometry_table_supports_patch_projection(self):
        """Generated geometry should project onto patch distance once linked."""
        geometry = pd.DataFrame(
            {
                "distance": [10.0, 20.0, 30.0],
                "x": [0.0, 10.0, 30.0],
                "z": [0.0, -5.0, -15.0],
            }
        )
        inv = Inventory.from_geometry_table(geometry, id_prefix="survey").add(
            Acquisition(
                resource_id="acq_1",
                optical_path_id="survey:optical_path",
            )
        )
        patch = dc.Patch(
            data=np.ones((3, 2)),
            coords={"distance": [0.0, 5.0, 20.0], "time": [0, 1]},
            dims=("distance", "time"),
            attrs={"acquisition_id": "acq_1"},
        )
        out = patch.add_inventory_coords(inv, coords=("x", "z"))
        assert np.allclose(out.get_array("x"), [0.0, 5.0, 30.0])
        assert np.allclose(out.get_array("z"), [0.0, -2.5, -15.0])

    def test_from_geometry_table_supports_geographic_axes(self):
        """Coordinate columns should define the generated CRS axis order."""
        geometry = pd.DataFrame(
            {
                "distance": [0.0, 100.0],
                "latitude": [40.0, 41.0],
                "longitude": [-111.0, -110.0],
                "elevation": [1500.0, 1400.0],
            }
        )
        inv = Inventory.from_geometry_table(geometry)
        crs = inv.get_records(record_type=CoordinateReferenceSystem)[0]
        assert crs.axis_order == ("latitude", "longitude", "elevation")

    def test_from_geometry_table_honors_crs_and_path_inputs(self):
        """Supplied CRS and path metadata should be preserved where possible."""
        geometry = pd.DataFrame({"distance": [0.0, 1.0], "x": [0.0, 1.0]})
        crs = {"resource_id": "crs_custom", "name": "local survey"}
        optical_path = OpticalPath(resource_id="path_custom", notes="surveyed")
        inv = Inventory.from_geometry_table(
            geometry,
            crs=crs,
            optical_path=optical_path,
        )
        crs_record = inv.get_records(record_type=CoordinateReferenceSystem)[0]
        path = inv.get_records(record_type=OpticalPath)[0]
        assert crs_record.resource_id == "crs_custom"
        assert crs_record.name == "local survey"
        assert crs_record.axis_order == ("x",)
        assert path.resource_id == "path_custom"
        assert path.notes == "surveyed"
        assert path.length == 1.0

    def test_from_geometry_table_is_deterministic(self):
        """Repeated imports of the same table should produce stable manifests."""
        geometry = pd.DataFrame({"distance": [0.0, 1.0], "x": [0.0, 1.0]})
        inv_1 = Inventory.from_geometry_table(geometry, id_prefix="repeat")
        inv_2 = Inventory.from_geometry_table(geometry, id_prefix="repeat")
        assert inv_1.dump_manifest() == inv_2.dump_manifest()

    @pytest.mark.parametrize(
        ("geometry", "match"),
        [
            (pd.DataFrame({"x": [0.0, 1.0]}), "distance"),
            (pd.DataFrame({"distance": [0.0], "x": [0.0]}), "two control points"),
            (
                pd.DataFrame({"distance": [0.0, 0.0], "x": [0.0, 1.0]}),
                "unique and increasing",
            ),
            (
                pd.DataFrame({"distance": [-1.0, 0.0], "x": [0.0, 1.0]}),
                "non-negative",
            ),
            (
                pd.DataFrame({"distance": [0.0, np.inf], "x": [0.0, 1.0]}),
                "finite",
            ),
            (pd.DataFrame({"distance": [0.0, 1.0]}), "coordinate column"),
            (
                pd.DataFrame({"distance": [0.0, 1.0], "label": ["a", "b"]}),
                "numeric",
            ),
        ],
    )
    def test_from_geometry_table_validates_input(self, geometry, match):
        """Bad geometry tables should raise clear errors."""
        with pytest.raises(ParameterError, match=match):
            Inventory.from_geometry_table(geometry)

    def test_from_geometry_table_requires_dataframe(self):
        """The geometry table should be a pandas DataFrame."""
        with pytest.raises(ParameterError, match="DataFrame"):
            Inventory.from_geometry_table({"distance": [0.0, 1.0], "x": [0.0, 1.0]})

    def test_from_geometry_table_validates_context_inputs(self):
        """CRS and path inputs should be mappings or matching inventory records."""
        geometry = pd.DataFrame({"distance": [0.0, 1.0], "x": [0.0, 1.0]})
        with pytest.raises(ParameterError, match="CoordinateReferenceSystem"):
            Inventory.from_geometry_table(geometry, crs=object())
        with pytest.raises(ParameterError, match="OpticalPath"):
            Inventory.from_geometry_table(geometry, optical_path=object())


class TestInventoryYAML:
    """Tests for inventory YAML IO."""

    def test_to_yaml_returns_string(self):
        """Inventory.to_yaml should return YAML when no destination is supplied."""
        inv = Inventory.model_validate(get_mixed_manifest())
        text = inv.to_yaml()
        assert "dascore_inventory: true" in text
        assert "type: Acquisition" in text

    def test_yaml_round_trip_path(self, tmp_path):
        """Inventories should round-trip through a YAML path."""
        inv = Inventory.model_validate(get_mixed_manifest())
        path = tmp_path / Inventory.default_file_name
        inv.to_yaml(path)
        out = Inventory.from_yaml(path)
        assert out.dump_manifest() == inv.dump_manifest()
        assert isinstance(out.view("optical_path_001"), OpticalPath)

    def test_yaml_round_trip_stream(self):
        """Inventories should round-trip through readable/writable streams."""
        inv = Inventory.model_validate(get_mixed_manifest())
        stream = StringIO()
        inv.to_yaml(stream)
        stream.seek(0)
        out = Inventory.from_yaml(stream)
        assert out.dump_manifest() == inv.dump_manifest()
