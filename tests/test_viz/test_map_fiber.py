"""Tests for map_fiber plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.utils.time import is_datetime64


def check_label_units(patch, ax, dims, color="distance"):
    """Ensure patch label units match axis."""
    axis_dict = {0: "xaxis", 1: "yaxis"}
    # dims = []
    # Check coord-innate names
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
    time = coords["time"].values - coords["time"].min()
    coords["time"] = time + random_starttime
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

    def test_masked_arrays_keep_their_mask(self, random_patch_with_lat_lon):
        """A masked point stays out of the drawing, as scatter would leave it."""
        patch = random_patch_with_lat_lon
        lats = patch.coords.get_array("latitude")
        lons = patch.coords.get_array("longitude")
        mask = np.zeros(len(lats), dtype=bool)
        mask[::3] = True
        color = np.ma.array(np.arange(len(lats), dtype=float), mask=mask)
        ax = patch.viz.map_fiber(np.ma.array(lats, mask=mask), lons, color)
        drawn = ax.collections[0].get_array()
        assert np.array_equal(np.ma.getmaskarray(drawn), mask)
        assert np.array_equal(
            np.ma.getmaskarray(ax.collections[0].get_offsets())[:, 0], mask
        )

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

    def test_unitless_coord_label_has_no_units(self, random_patch_with_lat_lon):
        """An unset unit contributes nothing rather than the word None."""
        ax = (
            random_patch_with_lat_lon.std("time")
            .squeeze()
            .viz.map_fiber("latitude", "longitude", "latitude")
        )
        assert ax.get_figure().axes[-1].get_ylabel() == "latitude"

    def test_show(self, random_patch, shown):
        """Ensure show path is callable."""
        random_patch.viz.map_fiber(show=True)
        assert shown

    def test_array_x_named_y_labels_y_axis(self, random_patch_with_lat_lon):
        """A named coordinate labels its own axis, whatever the other one is."""
        lats = random_patch_with_lat_lon.coords.get_array("latitude")
        ax = random_patch_with_lat_lon.viz.map_fiber(lats, "longitude")
        assert "longitude" in ax.get_ylabel().lower()
        assert ax.get_xlabel() == ""

    def test_coord_label_has_no_magnitude(self, random_patch_with_lat_lon):
        """The bar reads "distance [m]", not the "1 m" a pint quantity prints as."""
        patch = random_patch_with_lat_lon.set_units(distance="m")
        ax = patch.viz.map_fiber("latitude", "longitude", "distance")
        assert ax.get_figure().axes[-1].get_ylabel() == "distance [m]"

    def test_int_scale(self, random_patch):
        """An integer scale is read like the same float, not as one limit."""
        from_int = random_patch.viz.map_fiber(scale=2).collections[0].get_clim()
        from_float = random_patch.viz.map_fiber(scale=2.0).collections[0].get_clim()
        assert from_int == from_float
        # Twice half the range either side of the mean is wider than the data.
        values = random_patch.coords.get_array("distance")
        assert from_int[0] < values.min() and from_int[1] > values.max()

    def test_relative_pair_is_a_fraction_of_the_range(self, random_patch):
        """A relative pair reads as fractions of the color range, as in waterfall."""
        ax = random_patch.viz.map_fiber(scale=[0, 0.5])
        values = ax.collections[0].get_array()
        low, high = ax.collections[0].get_clim()
        assert np.isclose(low, values.min())
        assert np.isclose(high, values.min() + 0.5 * np.ptp(values))

    def test_absolute_number_is_symmetric_about_zero(self, random_patch):
        """A single absolute value is a limit either side of zero, as in waterfall."""
        ax = random_patch.viz.map_fiber(scale=10, scale_type="absolute")
        assert ax.collections[0].get_clim() == (-10, 10)

    def test_unusable_absolute_scale_leaves_autoscale(self, random_patch):
        """An empty or non-finite pair applies no limits rather than failing."""
        expected = random_patch.viz.map_fiber().collections[0].get_clim()
        for scale in ([], (np.nan, 1)):
            ax = random_patch.viz.map_fiber(scale=scale, scale_type="absolute")
            assert ax.collections[0].get_clim() == expected
            assert ax.collections[0].colorbar.extend == "neither"

    def test_empty_arrays_draw_an_empty_map(self):
        """Nothing to draw is still a plot, with its colorbar."""
        empty = np.array([])
        ax = dc.get_example_patch().viz.map_fiber(empty, empty, empty)
        assert len(ax.collections[0].get_offsets()) == 0

    def test_clipped_scale_extends_colorbar(self, random_patch):
        """Limits inside the data range get extend triangles, as in waterfall."""
        values = random_patch.coords.get_array("distance")
        mid = np.mean(values)
        ax = random_patch.viz.map_fiber(scale=(mid - 1, mid + 1), scale_type="absolute")
        assert ax.collections[0].colorbar.extend == "both"


class TestMapFiberColorByData:
    """Tests for coloring the fiber by the patch's own data."""

    @pytest.fixture()
    def reduced_patch(self, random_patch_with_lat_lon):
        """A patch with one value per channel, which is what can be drawn."""
        return random_patch_with_lat_lon.std("time").squeeze()

    def test_data_colors_the_fiber(self, reduced_patch):
        """The data is drawn without having to be pulled out by hand."""
        ax = reduced_patch.viz.map_fiber("latitude", "longitude", "data")
        collection = ax.collections[0]
        assert np.array_equal(collection.get_array(), reduced_patch.data)

    def test_data_label_comes_from_attrs(self, reduced_patch):
        """The patch knows its data type and units, so the bar is labeled."""
        patch = reduced_patch.update_attrs(data_type="velocity").set_units("m/s")
        ax = patch.viz.map_fiber("latitude", "longitude", "data")
        label = ax.get_figure().axes[-1].get_ylabel()
        assert label.startswith("velocity")
        assert "m / s" in label or "m/s" in label

    def test_coord_named_data_still_wins(self, reduced_patch):
        """A coordinate of that name keeps working as it did."""
        values = np.arange(len(reduced_patch.get_array("distance")), dtype=float)
        patch = reduced_patch.update_coords(data=("distance", values))
        ax = patch.viz.map_fiber("latitude", "longitude", "data")
        assert np.array_equal(ax.collections[0].get_array(), values)

    def test_aggregated_patch_is_accepted(self, random_patch_with_lat_lon):
        """An aggregation leaves a length one dim; that is still one per channel."""
        patch = random_patch_with_lat_lon.std("time")
        assert patch.ndim == 2, "the aggregated dimension should be kept"
        ax = patch.viz.map_fiber("latitude", "longitude", "data")
        expected = patch.data.reshape(-1)
        assert np.array_equal(ax.collections[0].get_array(), expected)

    def test_units_without_a_data_type_still_label(self, random_patch_with_lat_lon):
        """An unnamed quantity is still worth putting on the bar."""
        patch = random_patch_with_lat_lon.std("time").squeeze().set_units("m/s")
        assert not patch.attrs.data_type, "this patch should have no data type"
        ax = patch.viz.map_fiber("latitude", "longitude", "data")
        assert ax.get_figure().axes[-1].get_ylabel().strip()

    def test_unreduced_patch_raises(self, random_patch_with_lat_lon):
        """Two dimensions of data cannot color one point per channel."""
        with pytest.raises(ParameterError, match="one value for each"):
            random_patch_with_lat_lon.viz.map_fiber("latitude", "longitude", "data")

    def test_bad_color_names_data_as_an_option(self, reduced_patch):
        """The error should say what the caller probably wanted."""
        with pytest.raises(ParameterError, match="Use 'data'"):
            reduced_patch.viz.map_fiber("latitude", "longitude", "not_a_coord")


class TestMapFiberErrors:
    """Tests for map_fiber input validation."""

    def test_bad_x_coord(self, random_patch):
        """A non-existent x coordinate name should raise."""
        with pytest.raises(ParameterError, match="not found in patch"):
            random_patch.viz.map_fiber("not_a_coord", "time")

    def test_bad_y_coord(self, random_patch):
        """A non-existent y coordinate name should raise."""
        with pytest.raises(ParameterError, match="not found in patch"):
            random_patch.viz.map_fiber("distance", "not_a_coord")

    def test_bad_color_coord(self, random_patch):
        """A non-existent color coordinate name should raise."""
        with pytest.raises(ParameterError, match="not found in patch"):
            random_patch.viz.map_fiber("distance", "time", "not_a_coord")

    def test_bad_scale_type(self, random_patch):
        """An unknown scale_type should raise."""
        with pytest.raises(ParameterError, match="scale_type"):
            random_patch.viz.map_fiber(scale_type="nope", scale=10)

    def test_bad_scale_length(self, random_patch):
        """Three values are limits for nothing, whichever way they are read."""
        with pytest.raises(ParameterError, match="scale must be"):
            random_patch.viz.map_fiber(scale=(1, 2, 3))
        with pytest.raises(ParameterError, match="scale must be"):
            random_patch.viz.map_fiber(scale=(1, 2, 3), scale_type="absolute")
