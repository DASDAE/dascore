"""Tests for waterfall plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import dascore as dc
from dascore.utils.time import is_datetime64


def check_label_units(patch, ax, dims, color="distance"):
    """Ensure patch label units match axis."""
    axis_dict = {0: "xaxis", 1: "yaxis"}
    # dims = []
    # Check coord-inate names
    for coord_name in dims:
        coord = patch.coords.coord_map[coord_name]
        if is_datetime64(coord[0]):
            continue  # just skip datetimes for now.
        index = dims.index(coord_name)
        axis = getattr(ax, axis_dict[index])
        label_text = axis.get_label().get_text().lower()
        assert str(coord.units.units) in label_text
        assert coord_name in label_text
    # check colorbar labels
    # cax = ax.images[-1].colorbar
    coord = patch.coords.coord_map[color]
    yaxis_label = ax.figure.get_children()[-1].yaxis.label.get_text()
    assert str(coord.units.units) in yaxis_label


@pytest.fixture(scope="session")
def patch_random_start(event_patch_1):
    """Get a patch with a random, odd, starttime."""
    random_starttime = dc.to_datetime64("2020-01-02T02:12:11.02232")
    attrs = dict(event_patch_1.attrs)
    coords = {i: v for i, v in event_patch_1.coords.items()}
    time = coords["time"] - coords["time"].min()
    coords["time"] = time + random_starttime
    attrs["time_min"] = coords["time"].min()
    attrs["time_max"] = coords["time"].max()
    patch = event_patch_1.update(attrs=attrs, coords=coords)
    return patch


class TestPlotMap:
    """Tests for map plot."""

    def test_str_input(self, random_patch_with_lat_lon):
        """Call map_fiber plot, return."""
        patch = random_patch_with_lat_lon.set_units(latitude="ft")
        patch = patch.set_units(longitude="m")
        ax = patch.viz.map_fiber("latitude", "longitude")

        caxis_label = ax.figure.get_children()[-1].yaxis.label.get_text()

        # check labels
        assert "latitude" in ax.get_xlabel().lower()
        assert "longitude" in ax.get_ylabel().lower()
        assert "distance" in caxis_label
        assert isinstance(ax, plt.Axes)

    def test_array_inputs(self, random_patch_with_lat_lon):
        """Call map_fiber plot, return."""
        lats = random_patch_with_lat_lon.coords.get_array("latitude")
        lons = random_patch_with_lat_lon.coords.get_array("longitude")
        data = 0.5 * (lats + lons)
        ax = random_patch_with_lat_lon.viz.map_fiber(lats, lons, data)

        assert isinstance(ax, plt.Axes)

    def test_default_parameters(self, random_patch):
        """Call map_fiber plot, return."""
        ax = random_patch.viz.map_fiber()

        # check labels
        assert "distance" in ax.get_ylabel().lower()
        assert "distance" in ax.get_xlabel().lower()
        assert isinstance(ax, plt.Axes)

    def test_colorbar_scale(self, random_patch):
        """Tests for the scaling parameter."""
        ax_scalar = random_patch.viz.map_fiber(scale=0.2)
        assert ax_scalar is not None
        seq_scalar = random_patch.viz.map_fiber(scale=[0.1, 0.3])
        assert seq_scalar is not None

    def test_colorbar_absolute_scale(self, random_patch):
        """Tests for absolute scaling of colorbar."""
        patch = random_patch.new(data=random_patch.data * 100 - 50)
        ax1 = patch.viz.map_fiber(scale_type="absolute", scale=(-50, 50))
        assert ax1 is not None
        ax2 = patch.viz.map_fiber(scale_type="absolute", scale=10)
        assert ax2 is not None

    def test_no_colorbar(self, random_patch):
        """Ensure the colorbar can be disabled."""
        ax = random_patch.viz.map_fiber(cmap=None)
        # ensure no colorbar was created.
        assert len(ax.figure.get_children()) == 2

    def test_units(self, random_patch_with_lat_lon):
        """Test that units show up in labels."""
        # standard units

        pa = random_patch_with_lat_lon.set_units(distance="m/s")
        ax = pa.viz.map_fiber()
        check_label_units(pa, ax, ["distance", "distance"])

        new = pa.set_units(latitude="ft", longitude="m")
        ax = new.viz.map_fiber("latitude", "longitude")
        check_label_units(new, ax, ["latitude", "longitude"])

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        random_patch.viz.map_fiber(show=True)
