"""Tests for inventory patch processing methods."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core import inventory as inv
from dascore.exceptions import ParameterError

DATA_SOURCE_ID = "XX.FA.00.RAW"


def get_channel_patch(*, data_source_id=DATA_SOURCE_ID, time=None):
    """Return a small patch with channel and time dimensions."""
    time = np.arange(2) if time is None else np.asarray(time)
    data = np.ones((3, len(time)))
    return dc.Patch(
        data=data,
        coords={"channel": np.asarray([0, 1, 2]), "time": time},
        dims=("channel", "time"),
        attrs={"data_source_id": data_source_id},
    )


def get_distance_patch(*, data_source_id=DATA_SOURCE_ID):
    """Return a small patch with a distance coordinate."""
    data = np.ones((3, 2))
    return dc.Patch(
        data=data,
        coords={"distance": np.asarray([0.0, 1.0, 2.0]), "time": [0, 1]},
        dims=("distance", "time"),
        attrs={"data_source_id": data_source_id},
    )


def get_inventory(*, tag="raw", acquisition_sample_rate=100.0):
    """Return an inventory with fiber_array, config, interrogator, and path links."""
    interrogator = inv.Interrogator(
        resource_id="interrogator_1",
        model="SyntheticInterrogator",
        serial_number="SN001",
    )
    config = inv.Acquisition(
        code="RAW",
        location_code="00",
        data_category="DAS",
        data_type="strain_rate",
        data_units="m/s",
        interrogator=interrogator,
        acquisition_sample_rate=acquisition_sample_rate,
        pulse_width=1.0e-8,
        comment="Test acquisition.",
        spatial_sampling_interval=2.0,
        first_channel_distance=100.0,
    )
    crs = inv.CoordinateReferenceSystem(
        resource_id="crs_1",
        authority="LOCAL",
        code="test_axis",
        axis_order=("x",),
        units="m",
    )
    geometry = inv.Geometry(
        optical_length=2.0,
        geometry_type="linear",
        coordinates=((0.0,), (2.0,)),
    )
    fiber = inv.FiberSegment(
        optical_length=2.0,
        fiber_type="single_mode",
    )
    annotation = inv.OpticalPathAnnotation(
        distance=(0.0, 2.0),
        label="line",
    )
    path = inv.OpticalPath(
        optical_length=2.0,
        crs=crs,
        optical_components=(fiber,),
        geometries=(geometry,),
        annotations=(annotation,),
    )
    fiber_array = inv.FiberArray(
        code="FA",
        acquisitions=(config,),
        optical_paths=(path,),
        tag=tag,
    )
    network = inv.Network(
        code="XX",
        fiber_arrays=(fiber_array,),
    )
    return inv.Inventory(networks=(network,))


class TestDistanceFromInventory:
    """Tests for deriving distance from an acquisition."""

    def test_default_channel_dim(self):
        """The default channel dimension should produce a distance coordinate."""
        out = get_channel_patch().distance_from_inventory(get_inventory())

        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])
        assert out.coords.dim_map["distance"] == ("channel",)

    def test_explicit_data_source_id(self):
        """An explicit data source id should override patch attrs."""
        patch = get_channel_patch(data_source_id="")
        out = patch.distance_from_inventory(
            get_inventory(),
            data_source_id=DATA_SOURCE_ID,
        )

        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])

    def test_missing_data_source_id_raises(self):
        """A data source id is required."""
        with pytest.raises(ParameterError, match="data_source_id"):
            get_channel_patch(data_source_id="").distance_from_inventory(
                get_inventory()
            )

    def test_time_selects_acquisition_history(self):
        """Absolute patch time should select the effective acquisition state."""
        patch = get_channel_patch(
            time=np.array(["2020-01-03", "2020-01-04"], dtype="datetime64[D]")
        )
        inventory = get_inventory()
        config = (
            inventory.networks[0]
            .fiber_arrays[0]
            .acquisitions[0]
            .revise(
                start_time="2020-01-02",
                spatial_sampling_interval=3.0,
                first_channel_distance=50.0,
            )
        )
        inventory = inventory.replace(config)

        out = patch.distance_from_inventory(inventory)

        assert np.allclose(out.get_array("distance"), [50.0, 53.0, 56.0])


class TestAddInventoryCoords:
    """Tests for adding coordinates from an optical path."""

    def test_add_label_and_geometry_axis(self):
        """Inventory coordinates should project over patch distance."""
        out = get_distance_patch().add_inventory_coords(
            get_inventory(),
            coords=("label", "x"),
        )

        assert "label" in out.coords.coord_map
        assert "x" in out.coords.coord_map
        assert tuple(out.get_array("label")) == ("line", "line", "line")
        assert np.allclose(out.get_array("x"), [0.0, 1.0, 2.0])

    def test_missing_geometry_axis_can_fill_nan(self):
        """Missing geometry axes can be projected as NaN values."""
        out = get_distance_patch().add_inventory_coords(
            get_inventory(),
            coords=("y",),
            on_missing="nan",
        )

        assert np.isnan(out.get_array("y")).all()

    def test_path_crs_applies_to_all_geometries(self):
        """Every geometry interval on a path should share the path CRS."""
        inventory = get_inventory()
        path = inventory.networks[0].fiber_arrays[0].optical_paths[0]
        crs = inv.CoordinateReferenceSystem(
            resource_id="crs_xy",
            authority="LOCAL",
            code="xy",
            axis_order=("x", "y"),
            units="m",
        )
        geometries = (
            inv.Geometry(
                optical_length=1.0,
                geometry_type="linear",
                coordinates=((0.0, 10.0), (0.0, 11.0)),
            ),
            inv.Geometry(
                optical_length=1.0,
                geometry_type="linear",
                coordinates=((0.0, 11.0), (0.0, 12.0)),
            ),
        )
        path = path.revise(coordinate_reference_system=crs, geometries=geometries)
        inventory = inventory.replace(path)

        out = get_distance_patch().add_inventory_coords(
            inventory,
            coords=("y",),
            on_boundary="ignore",
        )

        assert np.allclose(out.get_array("y"), [10.0, 11.0, 12.0])


class TestAddInventoryAttrs:
    """Tests for adding attrs from fiber array relationships."""

    def test_add_unqualified_and_qualified_attrs(self):
        """Attrs should resolve across fiber_array, acquisition, and interrogator."""
        out = get_distance_patch().add_inventory_attrs(
            get_inventory(),
            attrs=(
                "tag",
                "acquisition_sample_rate",
                "pulse_width",
                "acquisition.comment",
                "acquisition.code",
                "acquisition.location_code",
                "acquisition.first_channel_distance",
                "interrogator.model",
            ),
        )

        assert out.attrs.tag == "raw"
        assert out.attrs.acquisition_sample_rate == 100.0
        assert out.attrs.pulse_width == 1.0e-8
        assert out.attrs.acquisition_comment == "Test acquisition."
        assert out.attrs.acquisition_code == "RAW"
        assert out.attrs.acquisition_location_code == "00"
        assert out.attrs.acquisition_first_channel_distance == 100.0
        assert out.attrs.interrogator_model == "SyntheticInterrogator"

    def test_ambiguous_attr_raises(self):
        """Ambiguous attrs should require qualification."""
        with pytest.raises(ParameterError, match="ambiguous"):
            get_distance_patch().add_inventory_attrs(
                get_inventory(),
                attrs=("code",),
            )
