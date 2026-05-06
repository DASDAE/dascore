"""Tests for inventory patch processing methods."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core import inventory as inv
from dascore.exceptions import ParameterError


def get_channel_patch(*, fiber_array_id="fiber_array_1", time=None):
    """Return a small patch with channel and time dimensions."""
    time = np.arange(2) if time is None else np.asarray(time)
    data = np.ones((3, len(time)))
    return dc.Patch(
        data=data,
        coords={"channel": np.asarray([10, 11, 12]), "time": time},
        dims=("channel", "time"),
        attrs={"fiber_array_id": fiber_array_id},
    )


def get_distance_patch(*, fiber_array_id="fiber_array_1"):
    """Return a small patch with a distance coordinate."""
    data = np.ones((3, 2))
    return dc.Patch(
        data=data,
        coords={"distance": np.asarray([0.0, 1.0, 2.0]), "time": [0, 1]},
        dims=("distance", "time"),
        attrs={"fiber_array_id": fiber_array_id},
    )


def get_legacy_channel_patch():
    """Return a channel patch with only the old acquisition id extra attr."""
    patch = get_channel_patch(fiber_array_id="")
    return patch.update_attrs(acquisition_id="fiber_array_1")


def get_legacy_distance_patch():
    """Return a distance patch with only the old acquisition id extra attr."""
    patch = get_distance_patch(fiber_array_id="")
    return patch.update_attrs(acquisition_id="fiber_array_1")


def get_inventory(*, tag="raw", sample_rate=100.0):
    """Return an inventory with fiber_array, config, interrogator, and path links."""
    interrogator = inv.Interrogator(
        resource_id="interrogator_1",
        interrogator_id="IU001",
        model="SyntheticInterrogator",
        serial_number="SN001",
    )
    config = inv.AcquisitionConfiguration(
        resource_id="cfg_1",
        interrogator=interrogator,
        sample_rate=sample_rate,
        spatial_sampling_interval=2.0,
        first_channel_index=10,
        first_channel_distance=100.0,
    )
    crs = inv.CoordinateReferenceSystem(resource_id="crs_1", axis_order=("x",))
    geometry = inv.Geometry(
        resource_id="geo_1",
        length=2.0,
        geometry_type="linear",
        coordinate_reference_system=crs,
        coordinates=((0.0,), (2.0,)),
    )
    fiber = inv.FiberSegment(
        resource_id="fiber_1",
        length=2.0,
        fiber_type="single_mode",
    )
    annotation = inv.OpticalPathAnnotation(
        resource_id="annotation_1",
        start_distance=0.0,
        end_distance=2.0,
        label="line",
    )
    path = inv.OpticalPath(
        resource_id="path_1",
        length=2.0,
        optical_components=(fiber,),
        geometries=(geometry,),
        annotations=(annotation,),
    )
    fiber_array = inv.FiberArray(
        resource_id="fiber_array_1",
        acquisition_configuration=config,
        optical_path=path,
        tag=tag,
        data_units="m/s",
    )
    return inv.Inventory(records=(fiber_array,))


class TestDistanceFromInventory:
    """Tests for deriving distance from fiber array configuration."""

    def test_default_channel_dim(self):
        """The default channel dimension should produce a distance coordinate."""
        out = get_channel_patch().distance_from_inventory(get_inventory())

        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])
        assert out.coords.dim_map["distance"] == ("channel",)

    def test_explicit_fiber_array_id(self):
        """An explicit fiber array id should override patch attrs."""
        patch = get_channel_patch(fiber_array_id="")
        out = patch.distance_from_inventory(
            get_inventory(),
            fiber_array_id="fiber_array_1",
        )

        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])

    def test_legacy_acquisition_id_fallback(self):
        """The old acquisition_id extra attr can still link legacy patches."""
        out = get_legacy_channel_patch().distance_from_inventory(get_inventory())

        assert np.allclose(out.get_array("distance"), [100.0, 102.0, 104.0])

    def test_missing_fiber_array_id_raises(self):
        """A fiber array id is required."""
        with pytest.raises(ParameterError, match="fiber_array_id"):
            get_channel_patch(fiber_array_id="").distance_from_inventory(
                get_inventory()
            )

    def test_time_selects_configuration_history(self):
        """Absolute patch time should select the effective configuration state."""
        patch = get_channel_patch(
            time=np.array(["2020-01-03", "2020-01-04"], dtype="datetime64[D]")
        )
        inventory = get_inventory()
        fiber_array = inventory.get_records(record_ids="fiber_array_1")[0]
        config = fiber_array.acquisition_configuration.model_copy(
            update={
                "start_time": "2020-01-02",
                "spatial_sampling_interval": 3.0,
                "first_channel_distance": 50.0,
            }
        )
        inventory = inventory.put_records(records=(config,))

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

    def test_legacy_acquisition_id_fallback(self):
        """Coordinate projection can use the old acquisition_id extra attr."""
        out = get_legacy_distance_patch().add_inventory_coords(
            get_inventory(),
            coords=("label",),
        )

        assert tuple(out.get_array("label")) == ("line", "line", "line")


class TestAddInventoryAttrs:
    """Tests for adding attrs from fiber array relationships."""

    def test_add_unqualified_and_qualified_attrs(self):
        """Attrs should resolve across fiber_array, config, and interrogator."""
        out = get_distance_patch().add_inventory_attrs(
            get_inventory(),
            attrs=(
                "tag",
                "sample_rate",
                "acquisition_configuration.first_channel_index",
                "interrogator.model",
            ),
        )

        assert out.attrs.tag == "raw"
        assert out.attrs.sample_rate == 100.0
        assert out.attrs.acquisition_configuration_first_channel_index == 10
        assert out.attrs.interrogator_model == "SyntheticInterrogator"

    def test_ambiguous_attr_raises(self):
        """Ambiguous attrs should require qualification."""
        with pytest.raises(ParameterError, match="ambiguous"):
            get_distance_patch().add_inventory_attrs(
                get_inventory(),
                attrs=("resource_id",),
            )

    def test_legacy_acquisition_id_fallback(self):
        """Attr projection can use the old acquisition_id extra attr."""
        out = get_legacy_distance_patch().add_inventory_attrs(
            get_inventory(),
            attrs=("tag",),
        )

        assert out.attrs.tag == "raw"
