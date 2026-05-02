"""Tests for inventory patch processing methods."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core import inventory as inv
from dascore.core.inventory import Acquisition, InstrumentConfiguration, Inventory
from dascore.exceptions import CoordError, ParameterError
from dascore.proc import inventory as proc_inventory


def get_channel_patch(
    channel=(10, 11, 12),
    *,
    acquisition_id="acq_1",
    dim="channel",
    time=None,
):
    """Return a small patch with an integer channel-like dimension."""
    time = np.arange(2) if time is None else np.asarray(time)
    data = np.ones((len(channel), len(time)))
    coords = {dim: np.asarray(channel), "time": time}
    return dc.Patch(
        data=data,
        coords=coords,
        dims=(dim, "time"),
        attrs={"acquisition_id": acquisition_id},
    )


def get_inventory(
    *,
    acquisition_id="acq_1",
    config_id="cfg_1",
    interval=2.0,
    first_channel_index=10,
    first_channel_distance=100.0,
):
    """Return an inventory with one acquisition and instrument configuration."""
    return Inventory(
        objects=[
            Acquisition(
                resource_id=acquisition_id,
                instrument_configuration_id=config_id,
            ),
            InstrumentConfiguration(
                resource_id=config_id,
                spatial_sampling_interval=interval,
                first_channel_index=first_channel_index,
                first_channel_distance=first_channel_distance,
            ),
        ]
    )


def get_inventory_patch(distance=(0.0, 1.0, 2.0), *, acquisition_id="acq_1"):
    """Return a small patch with a distance coordinate."""
    distance = np.asarray(distance, dtype=float)
    data = np.ones((len(distance), 2))
    return dc.Patch(
        data=data,
        coords={"distance": distance, "time": [0, 1]},
        dims=("distance", "time"),
        attrs={"acquisition_id": acquisition_id},
    )


def make_resolved_optical_path_inventory(
    *,
    axis_order=("x", "y", "z"),
    coordinates=((0.0, 0.0, 0.0), (10.0, 0.0, 5.0)),
    geometry_type="linear",
    annotations=(
        inv.OpticalPathAnnotation(
            resource_id="annotation_1",
            start_distance=0.0,
            end_distance=2.0,
            label="line",
        ),
    ),
):
    """Return an inventory with one acquisition-resolved optical path."""
    return inv.Inventory(
        objects=[
            inv.Acquisition(
                resource_id="acq_1",
                instrument_configuration_id="cfg_1",
                optical_path_id="path_1",
            ),
            inv.InstrumentConfiguration(
                resource_id="cfg_1",
                spatial_sampling_interval=1.0,
                first_channel_index=10,
            ),
            inv.OpticalPath(
                resource_id="path_1",
                length=2.0,
                optical_component_ids=("fiber_1",),
                geometry_ids=("geo_1",),
                coupling_condition_ids=("coupling_1",),
                annotation_ids=tuple(item.resource_id for item in annotations),
                annotations=annotations,
            ),
            inv.FiberSegment(
                resource_id="fiber_1",
                length=2.0,
                fiber_type="single_mode",
            ),
            inv.CoordinateReferenceSystem(
                resource_id="crs_1",
                axis_order=axis_order,
            ),
            inv.Geometry(
                resource_id="geo_1",
                length=2.0,
                geometry_type=geometry_type,
                coordinate_reference_system_id="crs_1",
                coordinates=coordinates,
            ),
            inv.CouplingCondition(
                resource_id="coupling_1",
                length=2.0,
                coupling_type="buried",
                quality="good",
            ),
            *annotations,
        ]
    )


def make_attr_inventory():
    """Return an inventory with acquisition, config, and instrument attrs."""
    return inv.Inventory(
        objects=[
            inv.Acquisition(
                resource_id="acq_1",
                instrument_configuration_id="cfg_1",
                network="XX",
                station="DAS1",
                tag="raw",
                data_units="m/s",
            ),
            inv.InstrumentConfiguration(
                resource_id="cfg_1",
                instrument_id="inst_1",
                sample_rate=1000.0,
                gauge_length=10.0,
            ),
            inv.Instrument(
                resource_id="inst_1",
                instrument_id="serial_1",
                serial_number="SN001",
            ),
        ]
    )


class TestDistanceFromInventory:
    """Tests for deriving optical distance from inventory channel metadata."""

    def test_default_channel_dim(self):
        """The default channel dimension should produce a distance coordinate."""
        patch = get_channel_patch()
        out = patch.distance_from_inventory(get_inventory())
        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])
        assert out.coords.dim_map["distance"] == ("channel",)

    def test_custom_index_dim(self):
        """The channel-like dimension can be specified explicitly."""
        patch = get_channel_patch(channel=(0, 1, 2), dim="index")
        inv = get_inventory(first_channel_index=0, first_channel_distance=5.0)
        out = patch.distance_from_inventory(inv, dim="index")
        assert np.allclose(out.get_array("distance"), [5.0, 7.0, 9.0])
        assert out.coords.dim_map["distance"] == ("index",)

    def test_explicit_acquisition_id(self):
        """An explicitly supplied acquisition id should be used."""
        patch = get_channel_patch(acquisition_id="")
        out = patch.distance_from_inventory(get_inventory(), acquisition_id="acq_1")
        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])

    def test_missing_acquisition_id_raises(self):
        """An acquisition id is required."""
        patch = get_channel_patch(acquisition_id="")
        with pytest.raises(ParameterError, match="acquisition_id"):
            patch.distance_from_inventory(get_inventory())

    def test_non_inventory_raises(self):
        """The inventory argument must be an Inventory."""
        patch = get_channel_patch()
        with pytest.raises(ParameterError, match="Inventory"):
            patch.distance_from_inventory({})

    def test_absolute_time_selects_configuration_version(self):
        """Absolute patch time should select the time-effective config."""
        time = np.array(["2020-01-03", "2020-01-04"], "datetime64[D]")
        patch = get_channel_patch(time=time)
        inv = Inventory(
            objects=[
                Acquisition(
                    resource_id="acq_1",
                    instrument_configuration_id="cfg_1",
                ),
                InstrumentConfiguration(
                    resource_id="cfg_1",
                    spatial_sampling_interval=1.0,
                    first_channel_index=10,
                    first_channel_distance=0.0,
                ),
                InstrumentConfiguration(
                    resource_id="cfg_1",
                    effective_time="2020-01-02",
                    spatial_sampling_interval=3.0,
                    first_channel_index=10,
                    first_channel_distance=50.0,
                ),
            ]
        )
        out = patch.distance_from_inventory(inv)
        assert np.allclose(out.get_array("distance"), [50.0, 53.0, 56.0])

    def test_missing_instrument_configuration_raises(self):
        """The acquisition must resolve an instrument configuration."""
        patch = get_channel_patch()
        inv = Inventory(objects=[Acquisition(resource_id="acq_1")])
        with pytest.raises(ParameterError, match="instrument configuration"):
            patch.distance_from_inventory(inv)

    def test_missing_spatial_sampling_interval_raises(self):
        """Spatial sampling interval is required for regular channel mapping."""
        patch = get_channel_patch()
        inv = Inventory(
            objects=[
                Acquisition(
                    resource_id="acq_1",
                    instrument_configuration_id="cfg_1",
                ),
                InstrumentConfiguration(resource_id="cfg_1"),
            ]
        )
        with pytest.raises(ParameterError, match="spatial_sampling_interval"):
            patch.distance_from_inventory(inv)

    def test_non_integer_dim_raises(self):
        """The selected channel-like dimension must contain integers."""
        patch = get_channel_patch(channel=(10.0, 11.0, 12.0))
        with pytest.raises(ParameterError, match="integer"):
            patch.distance_from_inventory(get_inventory())

    def test_missing_dim_uses_patch_coord_validation(self):
        """Missing dimensions should use Patch coordinate validation."""
        patch = get_channel_patch()
        with pytest.raises(CoordError, match="no dimension"):
            patch.distance_from_inventory(get_inventory(), dim="index")

    def test_existing_compatible_distance_is_updated(self):
        """An existing distance coordinate on the same dimension can be replaced."""
        patch = get_channel_patch().update_coords(distance=("channel", [0.0, 0.0, 0.0]))
        out = patch.distance_from_inventory(get_inventory())
        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])

    def test_existing_incompatible_distance_raises(self):
        """A distance coordinate on another dimension is ambiguous."""
        patch = get_channel_patch().update_coords(distance=("time", [0.0, 1.0]))
        with pytest.raises(ParameterError, match="already has a distance"):
            patch.distance_from_inventory(get_inventory())


class TestAddInventoryCoords:
    """Tests for projecting optical path metadata onto patch distance."""

    def test_default_label_projection(self):
        """By default, only labels should be added."""
        patch = get_inventory_patch()
        out = patch.add_inventory_coords(make_resolved_optical_path_inventory())
        assert tuple(out.get_array("label")) == ("line", "line", "line")
        assert out.coords.dim_map["label"] == ("distance",)

    def test_overlapping_annotation_label_projection(self):
        """Overlapping annotations should project tuple-valued labels."""
        patch = get_inventory_patch()
        annotations = (
            inv.OpticalPathAnnotation(
                resource_id="annotation_all",
                start_distance=0.0,
                end_distance=2.0,
                label="all",
            ),
            inv.OpticalPathAnnotation(
                resource_id="annotation_middle",
                start_distance=0.5,
                end_distance=1.5,
                label="middle",
            ),
        )
        out = patch.add_inventory_coords(
            make_resolved_optical_path_inventory(annotations=annotations)
        )
        assert tuple(out.get_array("label")) == ("all", ("all", "middle"), "all")

    def test_annotation_metadata_projection(self):
        """Annotation fields should project like other interval metadata."""
        patch = get_inventory_patch()
        annotations = (
            inv.OpticalPathAnnotation(
                resource_id="annotation_left",
                start_distance=0.0,
                end_distance=1.0,
                label="left",
                category="tap_test",
            ),
            inv.OpticalPathAnnotation(
                resource_id="annotation_right",
                start_distance=1.0,
                end_distance=2.0,
                label="right",
                category="field_note",
            ),
        )
        out = patch.add_inventory_coords(
            make_resolved_optical_path_inventory(annotations=annotations),
            coords=("annotation.category",),
        )
        assert tuple(out.get_array("annotation.category")) == (
            "tap_test",
            "field_note",
            "field_note",
        )

    def test_annotation_projection_includes_final_endpoint(self):
        """The final optical-path endpoint belongs to the final annotation."""
        patch = get_inventory_patch(distance=(2.0,))
        out = patch.add_inventory_coords(make_resolved_optical_path_inventory())
        assert tuple(out.get_array("label")) == ("line",)

    def test_local_cartesian_axes(self):
        """Explicit x/y/z axes should project from local CRS geometry."""
        patch = get_inventory_patch()
        out = patch.add_inventory_coords(
            make_resolved_optical_path_inventory(),
            coords=("label", "x", "y", "z"),
        )
        assert np.allclose(out.get_array("x"), [0.0, 5.0, 10.0])
        assert np.allclose(out.get_array("y"), [0.0, 0.0, 0.0])
        assert np.allclose(out.get_array("z"), [0.0, 2.5, 5.0])
        assert tuple(out.get_array("label")) == ("line", "line", "line")

    def test_default_xyz_axes_without_crs(self):
        """Geometry without CRS should use x/y/z coordinate order."""
        inventory = make_resolved_optical_path_inventory()
        geometry = inv.Geometry(
            resource_id="geo_1",
            length=2.0,
            geometry_type="linear",
            coordinates=((1.0, 2.0, 3.0), (3.0, 4.0, 5.0)),
        )
        inventory = inventory.add(geometry)
        out = get_inventory_patch().add_inventory_coords(inventory, coords=("x", "z"))
        assert np.allclose(out.get_array("x"), [1.0, 2.0, 3.0])
        assert np.allclose(out.get_array("z"), [3.0, 4.0, 5.0])

    def test_label_projection_can_be_empty(self):
        """Distances not covered by annotations should receive an empty label."""
        annotations = (
            inv.OpticalPathAnnotation(
                resource_id="annotation_1",
                start_distance=0.0,
                end_distance=1.0,
                label="left",
            ),
        )
        inventory = make_resolved_optical_path_inventory(annotations=annotations)
        out = get_inventory_patch(distance=(0.5, 1.5)).add_inventory_coords(inventory)
        assert tuple(out.get_array("label")) == ("left", "")


class TestAddInventoryAttrs:
    """Tests for adding attrs from acquisition/instrument inventory context."""

    def test_acquisition_attrs(self):
        """Acquisition attrs should be added to patch attrs."""
        patch = get_inventory_patch()
        out = patch.add_inventory_attrs(
            make_attr_inventory(),
            attrs=("network", "station", "tag", "data_units"),
        )
        assert out.attrs.network == "XX"
        assert out.attrs.station == "DAS1"
        assert out.attrs.tag == "raw"
        assert str(out.attrs.data_units) == "1.0 m / s"

    def test_instrument_configuration_attrs(self):
        """Instrument configuration attrs should resolve by simple names."""
        patch = get_inventory_patch()
        out = patch.add_inventory_attrs(
            make_attr_inventory(),
            attrs=("sample_rate", "gauge_length"),
        )
        assert out.attrs.sample_rate == 1000.0
        assert out.attrs.gauge_length == 10.0

    def test_linked_instrument_attrs(self):
        """Linked instrument attrs should resolve by qualified names."""
        patch = get_inventory_patch()
        out = patch.add_inventory_attrs(
            make_attr_inventory(),
            attrs=("instrument.instrument_id", "instrument.serial_number"),
        )
        assert out.attrs.instrument_instrument_id == "serial_1"
        assert out.attrs.instrument_serial_number == "SN001"

    def test_explicit_acquisition_id(self):
        """An explicitly supplied acquisition id should be used."""
        patch = get_inventory_patch(acquisition_id="")
        out = patch.add_inventory_attrs(
            make_attr_inventory(),
            attrs="network",
            acquisition_id="acq_1",
        )
        assert out.attrs.network == "XX"

    def test_missing_acquisition_id_raises_for_attrs(self):
        """An acquisition id is required."""
        patch = get_inventory_patch(acquisition_id="")
        with pytest.raises(ParameterError, match="acquisition_id"):
            patch.add_inventory_attrs(make_attr_inventory(), attrs="network")

    def test_empty_attr_requests_raise(self):
        """At least one inventory attr must be requested."""
        with pytest.raises(ParameterError, match="At least one"):
            get_inventory_patch().add_inventory_attrs(make_attr_inventory(), attrs=())

    def test_non_string_attr_requests_raise(self):
        """Inventory attr requests must be strings."""
        with pytest.raises(ParameterError, match="non-empty strings"):
            get_inventory_patch().add_inventory_attrs(
                make_attr_inventory(),
                attrs=(1,),
            )

    def test_invalid_qualified_attr_raises(self):
        """Qualified attr names must use known acquisition graph sources."""
        with pytest.raises(ParameterError, match="Invalid inventory attr"):
            get_inventory_patch().add_inventory_attrs(
                make_attr_inventory(),
                attrs="not_a_source.field",
            )

    def test_missing_time_coord_still_resolves_attrs(self):
        """Patch time is optional for resolving inventory attrs."""
        patch = dc.Patch(
            data=np.ones(3),
            coords={"distance": [0.0, 1.0, 2.0]},
            dims=("distance",),
            attrs={"acquisition_id": "acq_1"},
        )
        out = patch.add_inventory_attrs(make_attr_inventory(), attrs="network")
        assert out.attrs.network == "XX"

    def test_time_selects_effective_attrs(self):
        """Absolute patch time should select effective acquisition graph records."""
        time = np.array(["2020-01-03", "2020-01-04"], "datetime64[D]")
        patch = get_inventory_patch().update_coords(time=time)
        inventory = inv.Inventory(
            objects=[
                inv.Acquisition(
                    resource_id="acq_1",
                    instrument_configuration_id="cfg_1",
                    network="old",
                ),
                inv.Acquisition(
                    resource_id="acq_1",
                    effective_time="2020-01-02",
                    instrument_configuration_id="cfg_1",
                    network="new",
                ),
                inv.InstrumentConfiguration(
                    resource_id="cfg_1",
                    instrument_id="inst_1",
                    sample_rate=100.0,
                ),
                inv.InstrumentConfiguration(
                    resource_id="cfg_1",
                    effective_time="2020-01-02",
                    instrument_id="inst_1",
                    sample_rate=200.0,
                ),
                inv.Instrument(
                    resource_id="inst_1",
                    instrument_id="old_serial",
                ),
                inv.Instrument(
                    resource_id="inst_1",
                    effective_time="2020-01-02",
                    instrument_id="new_serial",
                ),
            ]
        )
        out = patch.add_inventory_attrs(
            inventory,
            attrs=("network", "sample_rate", "instrument.instrument_id"),
        )
        assert out.attrs.network == "new"
        assert out.attrs.sample_rate == 200.0
        assert out.attrs.instrument_instrument_id == "new_serial"

    def test_missing_linked_object_raises(self):
        """Missing linked objects should raise by default."""
        patch = get_inventory_patch()
        inventory = inv.Inventory(objects=[inv.Acquisition(resource_id="acq_1")])
        with pytest.raises(ParameterError, match="instrument_configuration"):
            patch.add_inventory_attrs(
                inventory,
                attrs="instrument_configuration.sample_rate",
            )

    def test_missing_linked_object_can_ignore(self):
        """Missing linked objects can be ignored."""
        patch = get_inventory_patch()
        inventory = inv.Inventory(objects=[inv.Acquisition(resource_id="acq_1")])
        out = patch.add_inventory_attrs(
            inventory,
            attrs=("network", "instrument_configuration.sample_rate"),
            on_missing="ignore",
        )
        assert not hasattr(out.attrs, "instrument_configuration_sample_rate")

    def test_missing_field_can_ignore(self):
        """Missing fields can be ignored."""
        patch = get_inventory_patch()
        out = patch.add_inventory_attrs(
            make_attr_inventory(),
            attrs=("network", "not_a_field"),
            on_missing="ignore",
        )
        assert out.attrs.network == "XX"
        assert not hasattr(out.attrs, "not_a_field")

    def test_ambiguous_simple_field_raises(self):
        """Ambiguous simple attrs should require qualified names."""
        patch = get_inventory_patch()
        with pytest.raises(ParameterError, match="ambiguous"):
            patch.add_inventory_attrs(make_attr_inventory(), attrs="instrument_id")

    def test_invalid_policy_raises(self):
        """on_missing should be validated."""
        patch = get_inventory_patch()
        with pytest.raises(ParameterError, match="on_missing"):
            patch.add_inventory_attrs(
                make_attr_inventory(),
                attrs="network",
                on_missing="nan",
            )

    def test_geographic_axes(self):
        """Explicit geographic axes should project from CRS axis order."""
        inventory = make_resolved_optical_path_inventory(
            axis_order=("latitude", "longitude", "elevation"),
            coordinates=((40.0, -111.0, 1500.0), (41.0, -110.0, 1400.0)),
        )
        out = get_inventory_patch().add_inventory_coords(
            inventory,
            coords=("label", "latitude", "longitude", "elevation"),
        )
        assert np.allclose(out.get_array("latitude"), [40.0, 40.5, 41.0])
        assert np.allclose(out.get_array("longitude"), [-111.0, -110.5, -110.0])
        assert np.allclose(out.get_array("elevation"), [1500.0, 1450.0, 1400.0])

    def test_no_geometry_wildcard(self):
        """Geometry axes should be requested explicitly."""
        patch = get_inventory_patch()
        with pytest.raises(ParameterError, match="Geometry axis"):
            patch.add_inventory_coords(
                make_resolved_optical_path_inventory(),
                coords=("geometry",),
            )

    def test_empty_coord_requests_raise(self):
        """At least one inventory coordinate must be requested."""
        with pytest.raises(ParameterError, match="At least one"):
            get_inventory_patch().add_inventory_coords(
                make_resolved_optical_path_inventory(),
                coords=(),
            )

    def test_non_string_coord_requests_raise(self):
        """Inventory coordinate requests must be strings."""
        with pytest.raises(ParameterError, match="non-empty strings"):
            get_inventory_patch().add_inventory_coords(
                make_resolved_optical_path_inventory(),
                coords=(1,),
            )

    def test_invalid_qualified_coord_raises(self):
        """Qualified inventory coordinate names must use known track names."""
        with pytest.raises(ParameterError, match="Invalid inventory coordinate"):
            get_inventory_patch().add_inventory_coords(
                make_resolved_optical_path_inventory(),
                coords=("not_a_track.field",),
            )

    def test_missing_distance_raises(self):
        """A distance coordinate is required before projecting path metadata."""
        patch = get_channel_patch()
        with pytest.raises(ParameterError, match="distance_from_inventory"):
            patch.add_inventory_coords(make_resolved_optical_path_inventory())

    def test_multidimensional_distance_coord_raises(self):
        """Inventory projection requires distance to map to one dimension."""
        class BadCoords:
            def __init__(self):
                self.dim_map = {"distance": ("distance", "time")}

            def __contains__(self, name):
                return name == "distance"

        class BadPatch:
            def __init__(self):
                self.coords = BadCoords()

        with pytest.raises(ParameterError, match="exactly one dimension"):
            proc_inventory._get_distance_values_and_dim(BadPatch())

    def test_distance_coord_array_must_be_one_dimensional(self):
        """A one-dimension distance coord should still expose one-dimensional values."""

        class BadCoords:
            def __init__(self):
                self.dim_map = {"distance": ("distance",)}

            def __contains__(self, name):
                return name == "distance"

        class BadPatch:
            def __init__(self):
                self.coords = BadCoords()

            def get_array(self, name):
                assert name == "distance"
                return np.ones((2, 2))

        with pytest.raises(ParameterError, match="one-dimensional"):
            proc_inventory._get_distance_values_and_dim(BadPatch())

    def test_missing_acquisition_id_raises_for_coords(self):
        """An acquisition id is required."""
        patch = get_inventory_patch(acquisition_id="")
        with pytest.raises(ParameterError, match="acquisition_id"):
            patch.add_inventory_coords(make_resolved_optical_path_inventory())

    def test_missing_optical_path_raises(self):
        """The acquisition must resolve an optical path."""
        patch = get_inventory_patch()
        inventory = inv.Inventory(objects=[inv.Acquisition(resource_id="acq_1")])
        with pytest.raises(ParameterError, match="optical path"):
            patch.add_inventory_coords(inventory)

    def test_unknown_path_length_skips_bounds_check(self):
        """Unknown optical path length should not reject finite distances."""
        inventory = make_resolved_optical_path_inventory().add(
            inv.OpticalPath(
                resource_id="path_1",
                length=None,
                annotation_ids=("annotation_1",),
            )
        )
        out = get_inventory_patch(distance=(100.0,)).add_inventory_coords(inventory)
        assert tuple(out.get_array("label")) == ("",)

    def test_out_of_bounds_distance_raises(self):
        """Known optical path length bounds should be enforced."""
        patch = get_inventory_patch(distance=(-1.0,))
        with pytest.raises(ParameterError, match="optical path length"):
            patch.add_inventory_coords(make_resolved_optical_path_inventory())

    def test_unknown_geometry_returns_nan(self):
        """Unknown geometry should yield NaN numeric coordinates."""
        patch = get_inventory_patch()
        out = patch.add_inventory_coords(
            make_resolved_optical_path_inventory(geometry_type="unknown"),
            coords=("x",),
        )
        assert np.all(np.isnan(out.get_array("x")))

    def test_missing_axis_raises_by_default(self):
        """Explicit axes should raise when the CRS cannot provide them."""
        patch = get_inventory_patch()
        inventory = make_resolved_optical_path_inventory(
            axis_order=("latitude", "longitude"),
            coordinates=((40.0, -111.0), (41.0, -110.0)),
        )
        with pytest.raises(ParameterError, match="axis 'z'"):
            patch.add_inventory_coords(inventory, coords=("z",))

    def test_missing_axis_can_fill_nan(self):
        """Missing geometry axes can be filled with NaN."""
        patch = get_inventory_patch()
        inventory = make_resolved_optical_path_inventory(
            axis_order=("latitude", "longitude"),
            coordinates=((40.0, -111.0), (41.0, -110.0)),
        )
        out = patch.add_inventory_coords(inventory, coords=("z",), on_missing="nan")
        assert np.all(np.isnan(out.get_array("z")))

    def test_no_geometry_records_raise(self):
        """Geometry projection requires geometry records."""
        inventory = make_resolved_optical_path_inventory().add(
            inv.OpticalPath(resource_id="path_1", length=2.0)
        )
        with pytest.raises(ParameterError, match="no geometries"):
            get_inventory_patch().add_inventory_coords(inventory, coords=("x",))

    def test_geometry_records_need_lengths(self):
        """Geometry projection requires interval lengths."""
        inventory = make_resolved_optical_path_inventory().add(
            inv.Geometry(
                resource_id="geo_1",
                length=None,
                geometry_type="linear",
                coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            )
        )
        with pytest.raises(ParameterError, match="requires all records"):
            get_inventory_patch().add_inventory_coords(inventory, coords=("x",))

    def test_geometry_lengths_must_match_path(self):
        """Geometry intervals must agree with known optical path length."""
        inventory = make_resolved_optical_path_inventory().add(
            inv.Geometry(
                resource_id="geo_1",
                length=3.0,
                geometry_type="linear",
                coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
            )
        )
        with pytest.raises(ParameterError, match="does not match"):
            get_inventory_patch().add_inventory_coords(inventory, coords=("x",))

    def test_unsupported_geometry_type_raises(self):
        """Only linear and unknown geometry can project numeric axes."""
        inventory = make_resolved_optical_path_inventory(geometry_type="curved")
        with pytest.raises(ParameterError, match="supports only linear"):
            get_inventory_patch().add_inventory_coords(inventory, coords=("x",))

    def test_single_point_geometry_projects_constant_axis(self):
        """Single-point geometry should project constant coordinates."""
        inventory = make_resolved_optical_path_inventory(
            coordinates=((3.0, 4.0, 5.0),),
        )
        out = get_inventory_patch().add_inventory_coords(inventory, coords=("x",))
        assert np.allclose(out.get_array("x"), [3.0, 3.0, 3.0])

    def test_unselected_geometry_interval_is_skipped(self):
        """Projection should ignore geometry intervals with no distance samples."""
        inventory = inv.Inventory(
            objects=[
                inv.Acquisition(resource_id="acq_1", optical_path_id="path_1"),
                inv.OpticalPath(
                    resource_id="path_1",
                    length=2.0,
                    geometry_ids=("geo_1", "geo_2"),
                ),
                inv.Geometry(
                    resource_id="geo_1",
                    length=1.0,
                    geometry_type="linear",
                    coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
                ),
                inv.Geometry(
                    resource_id="geo_2",
                    length=1.0,
                    geometry_type="linear",
                    coordinates=((10.0, 0.0, 0.0), (11.0, 0.0, 0.0)),
                ),
            ]
        )
        out = get_inventory_patch(distance=(0.25,)).add_inventory_coords(
            inventory,
            coords=("x",),
        )
        assert np.allclose(out.get_array("x"), [0.25])

    def test_metadata_projection(self):
        """Unambiguous interval metadata fields should project onto distance."""
        patch = get_inventory_patch()
        out = patch.add_inventory_coords(
            make_resolved_optical_path_inventory(),
            coords=("fiber_type", "coupling_type", "geometry.geometry_type"),
        )
        assert tuple(out.get_array("fiber_type")) == (
            "single_mode",
            "single_mode",
            "single_mode",
        )
        assert tuple(out.get_array("coupling_type")) == ("buried", "buried", "buried")
        assert tuple(out.get_array("geometry.geometry_type")) == (
            "linear",
            "linear",
            "linear",
        )

    def test_ambiguous_metadata_raises(self):
        """Simple metadata names must be unambiguous across interval tracks."""
        patch = get_inventory_patch()
        with pytest.raises(ParameterError, match="ambiguous"):
            patch.add_inventory_coords(
                make_resolved_optical_path_inventory(),
                coords=("length",),
            )

    def test_interval_metadata_boundary_raise(self):
        """Interior interval boundaries should raise by default."""
        inventory = inv.Inventory(
            objects=[
                inv.Acquisition(
                    resource_id="acq_1",
                    optical_path_id="path_1",
                ),
                inv.OpticalPath(
                    resource_id="path_1",
                    length=2.0,
                    optical_component_ids=("fiber_1", "fiber_2"),
                ),
                inv.FiberSegment(
                    resource_id="fiber_1",
                    length=1.0,
                    fiber_type="single_mode",
                ),
                inv.FiberSegment(
                    resource_id="fiber_2",
                    length=1.0,
                    fiber_type="multi_mode",
                ),
            ]
        )
        with pytest.raises(ParameterError, match="interval boundary"):
            get_inventory_patch(distance=(1.0,)).add_inventory_coords(
                inventory,
                coords=("fiber_type",),
            )

    def test_annotation_boundary_raise_not_needed(self):
        """Annotation boundaries are directly assigned by annotation intervals."""
        patch = get_inventory_patch()
        inventory = make_resolved_optical_path_inventory(
            annotations=(
                inv.OpticalPathAnnotation(
                    resource_id="annotation_left",
                    start_distance=0.0,
                    end_distance=1.0,
                    label="left",
                ),
                inv.OpticalPathAnnotation(
                    resource_id="annotation_right",
                    start_distance=1.0,
                    end_distance=2.0,
                    label="right",
                ),
            )
        )
        out = patch.add_inventory_coords(inventory)
        assert tuple(out.get_array("label")) == ("left", "right", "right")

    def test_boundary_warn(self):
        """Boundary samples can warn and assign to the interval on the right."""
        inventory = inv.Inventory(
            objects=[
                inv.Acquisition(
                    resource_id="acq_1",
                    optical_path_id="path_1",
                ),
                inv.OpticalPath(
                    resource_id="path_1",
                    length=2.0,
                    optical_component_ids=("fiber_1", "fiber_2"),
                ),
                inv.FiberSegment(
                    resource_id="fiber_1",
                    length=1.0,
                    fiber_type="single_mode",
                ),
                inv.FiberSegment(
                    resource_id="fiber_2",
                    length=1.0,
                    fiber_type="multi_mode",
                ),
            ]
        )
        with pytest.warns(UserWarning, match="interval boundary"):
            out = get_inventory_patch(distance=(1.0,)).add_inventory_coords(
                inventory,
                coords=("fiber_type",),
                on_boundary="warn",
            )
        assert tuple(out.get_array("fiber_type")) == ("multi_mode",)

    def test_boundary_ignore(self):
        """Boundary samples can silently assign to the interval on the right."""
        inventory = inv.Inventory(
            objects=[
                inv.Acquisition(
                    resource_id="acq_1",
                    optical_path_id="path_1",
                ),
                inv.OpticalPath(
                    resource_id="path_1",
                    length=2.0,
                    optical_component_ids=("fiber_1", "fiber_2"),
                ),
                inv.FiberSegment(
                    resource_id="fiber_1",
                    length=1.0,
                    fiber_type="single_mode",
                ),
                inv.FiberSegment(
                    resource_id="fiber_2",
                    length=1.0,
                    fiber_type="multi_mode",
                ),
            ]
        )
        out = get_inventory_patch(distance=(1.0,)).add_inventory_coords(
            inventory,
            coords=("fiber_type",),
            on_boundary="ignore",
        )
        assert tuple(out.get_array("fiber_type")) == ("multi_mode",)

    def test_channel_indexed_two_step_workflow(self):
        """Channel-indexed patches should derive distance before inventory coords."""
        patch = get_channel_patch()
        out = patch.distance_from_inventory(make_resolved_optical_path_inventory())
        out = out.add_inventory_coords(
            make_resolved_optical_path_inventory(),
            coords=("label", "x"),
        )
        assert np.allclose(out.get_array("distance"), [0.0, 1.0, 2.0])
        assert np.allclose(out.get_array("x"), [0.0, 5.0, 10.0])
