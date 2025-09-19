"""Tests for waterfall plots."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

import dascore as dc
from dascore.units import get_quantity_str
from dascore.utils.time import is_datetime64, to_timedelta64


def check_label_units(patch, ax):
    """Ensure patch label units match axis."""
    axis_dict = {0: "yaxis", 1: "xaxis"}
    dims = patch.dims
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
    cax = ax.images[-1].colorbar
    yaxis_label = cax.ax.yaxis.label.get_text()
    assert str(patch.attrs.data_units.units) in yaxis_label
    assert str(patch.attrs.data_type) in yaxis_label


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


@pytest.fixture(scope="session")
def patch_two_times(random_patch):
    """Create a patch with two time dims."""
    dist = random_patch.coords.get_array("distance")
    pa = random_patch.update_coords(distance=dc.to_datetime64(dist))
    return pa


class TestWaterfall:
    """Tests for waterfall plot."""

    @pytest.fixture()
    def timedelta_patch(self, random_patch):
        """Make a patch with one dimension dtype of timedelta64."""
        old_coord = random_patch.get_coord("time")
        new_time = to_timedelta64(np.arange(len(old_coord)))
        return random_patch.update_coords(time=new_time)

    def test_returns_axes(self, random_patch):
        """Call waterfall plot, return."""
        # modify patch to include line at start
        data = np.array(random_patch.data)
        data[:100, :100] = 2.0  # create an origin block for testing axis line up
        data[:100, -100:] = -2.0  #
        out = random_patch.new(data=data)
        ax = out.viz.waterfall()

        # check labels
        assert random_patch.dims[0] in ax.get_ylabel().lower()
        assert random_patch.dims[1] in ax.get_xlabel().lower()
        assert isinstance(ax, plt.Axes)

    def test_colorbar_scale(self, random_patch):
        """Tests for the scaling parameter."""
        ax_scalar = random_patch.viz.waterfall(scale=0.2)
        assert ax_scalar is not None
        seq_scalar = random_patch.viz.waterfall(scale=[0.1, 0.3])
        assert seq_scalar is not None

    def test_colorbar_absolute_scale(self, random_patch):
        """Tests for absolute scaling of colorbar."""
        patch = random_patch.new(data=random_patch.data * 100 - 50)
        ax1 = patch.viz.waterfall(scale_type="absolute", scale=(-50, 50))
        assert ax1 is not None
        ax2 = patch.viz.waterfall(scale_type="absolute", scale=10)
        assert ax2 is not None

    def test_doc_intro_example(self, event_patch_1):
        """Simple test to ensure the doc examples can be run."""
        patch = event_patch_1.pass_filter(time=(None, 300))
        _ = patch.viz.waterfall(scale=0.04)
        _ = patch.transpose("distance", "time").viz.waterfall(scale=0.04)

    def test_time_axis_label_int_overflow(self, random_patch):
        """Make sure the time axis labels are correct (windows compatibility)."""
        ax = random_patch.viz.waterfall()
        name = ["y", "x"][random_patch.get_axis("time")]
        # Get the piece of the label corresponding to the starttime
        # WE can just grab the offset text.
        sub_ax = getattr(ax, f"{name}axis")
        plt.tight_layout()  # need to call this to get offset to show up.
        offset_str = sub_ax.get_major_formatter().get_offset()
        min_time = random_patch.coords.get_array("time").min()
        assert str(min_time).startswith(offset_str)

    def test_no_colorbar(self, random_patch):
        """Ensure the colorbar can be disabled."""
        ax = random_patch.viz.waterfall(cmap=None)
        # ensure no colorbar was created.
        assert ax.images[-1].colorbar is None

    def test_units(self, random_patch):
        """Test that units show up in labels."""
        # standard units
        pa = random_patch.set_units("m/s")
        ax = pa.viz.waterfall()
        check_label_units(pa, ax)
        # weird units
        new = pa.set_units(
            "furlongs/fortnight",
            distance="feet",
        )
        ax = new.viz.waterfall()
        check_label_units(new, ax)

    def test_time_no_units(self, patch_two_times):
        """time-like dims shouldn't show units in label."""
        pa = patch_two_times
        dims = pa.dims
        ax = pa.viz.waterfall()
        assert ax.get_xlabel() == dims[1]
        assert ax.get_ylabel() == dims[0]

    def test_patch_with_data_type(self, random_patch):
        """Ensure a patch with data_type titles the colorbar."""
        patch = random_patch.update_attrs(
            data_type="strain rate",
            data_units="1/s",
        )
        ax = patch.viz.waterfall()
        check_label_units(patch, ax)

    def test_timedelta_axis(self, timedelta_patch):
        """Ensure plot works when one axis has timedleta dtype. See #309."""
        # if this doesnt raise it probably works ;)
        ax = timedelta_patch.viz.waterfall()
        assert ax is not None

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        random_patch.viz.waterfall(show=True)

    def test_log(self, random_patch):
        """Ensure log is callable."""
        ax = random_patch.viz.waterfall(log=True)

        # Retrieve the colorbar label
        cb = ax.get_figure().get_axes()[-1]
        cb_label = cb.get_ylabel()

        # Retrieve the expected data type and data units
        data_type = str(random_patch.attrs["data_type"])
        data_units = get_quantity_str(random_patch.attrs.data_units) or ""
        expected_dunits = f" ({data_units})" if data_units else ""

        # Construct the expected label
        expected_label = f"{data_type}{expected_dunits} - log_10"

        # Check if the colorbar label matches the expected label
        assert (
            cb_label == expected_label
        ), f"Expected '{expected_label}', but got '{cb_label}'"
