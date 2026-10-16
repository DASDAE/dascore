"""
What operations along a dimension do to the coordinates associated with it.

The coordinate guidelines (`notes/coordinates.qmd`, Scope) ask that whatever
an operation does along a dimension it does to its associated coordinates,
or drops them with a warning, and that units survive.
"""

from __future__ import annotations

import numpy as np
import pytest

from dascore.exceptions import CoordError
from dascore.units import get_quantity
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning

DROPPED = "dropped associated coordinates"


@pytest.fixture(scope="module")
def patch(random_patch):
    """A patch with unit-bearing coordinates riding time and distance."""
    (nt,) = random_patch.coord_shapes["time"]
    (nd,) = random_patch.coord_shapes["distance"]
    return random_patch.update_coords(
        tcoord=("time", np.arange(nt) * 1.0),
        dcoord=("distance", np.arange(nd) * 2.0),
    ).set_units("m/s", distance="m", tcoord="m", dcoord="ft")


def _assert_rides(out, name, values):
    """The coordinate is still there, with these values and its units."""
    coord = out.coords.coord_map[name]
    assert np.array_equal(coord.values, values)
    assert coord.units == get_quantity("m" if name == "tcoord" else "ft")


class TestIndexed:
    """Operations which index a dimension index its riders the same way."""

    def test_dropna(self, patch):
        """Dropping null samples drops them from the riders too."""
        assert patch.dims[0] == "distance"
        kept = np.arange(patch.shape[0]) % 5 != 0
        nulled = patch.update(data=np.where(kept[:, None], patch.data, np.nan))
        out = nulled.dropna("distance")
        _assert_rides(out, "dcoord", patch.get_array("dcoord")[kept])

    def test_roll_with_coordinate(self, patch):
        """Rolling labels rolls the riders with them, keeping units."""
        out = patch.roll(time=3, samples=True, update_coord=True)
        _assert_rides(out, "tcoord", np.roll(patch.get_array("tcoord"), 3))
        assert np.array_equal(
            out.get_array("time"), np.roll(patch.get_array("time"), 3)
        )
        out = patch.roll(distance=2, samples=True, update_coord=True)
        assert out.get_coord("distance").units == get_quantity("m")

    def test_strided_rolling(self, patch):
        """A rolling step strides the riders as it strides the labels."""
        out = patch.rolling(time=10, step=10, samples=True).mean()
        _assert_rides(out, "tcoord", patch.get_array("tcoord")[::10])

    @pytest.mark.parametrize("left", [True, False])
    def test_binary_alignment(self, patch, left):
        """Aligning two patches subsets the riders of either operand."""
        half = patch.select(time=(0, 100), samples=True)
        out = patch + half if left else half + patch
        _assert_rides(out, "tcoord", patch.get_array("tcoord")[:100])

    def test_where(self, patch):
        """A patch condition aligns the riders like an operator."""
        cond = patch.select(time=(0, 100), samples=True) > 0
        out = patch.where(cond)
        _assert_rides(out, "tcoord", patch.get_array("tcoord")[:100])


class TestDroppedWithWarning:
    """Operations which cannot carry the riders drop them, saying so."""

    @pytest.mark.parametrize(
        "func",
        [
            lambda x: x.mean("time"),
            lambda x: x.first("time"),
            lambda x: x.idxmax("time"),
            lambda x: x.integrate("time", definite=True),
            lambda x: x.phase_weighted_stack("time", transform_dim="distance"),
            lambda x: x.drop_data().update_coords(time=x.get_coord("time")[:10]),
        ],
    )
    def test_reductions(self, patch, func):
        """A dimension reduced or resized drops what rode it, with a warning."""
        with pytest.warns(DASCoreWarning, match="tcoord"):
            out = func(patch)
        assert "tcoord" not in out.coords.coord_map
        _assert_rides(out, "dcoord", patch.get_array("dcoord"))

    def test_correlate(self, patch):
        """Lags are not the samples a rider described, so it goes."""
        with pytest.warns(DASCoreWarning, match="tcoord"):
            out = patch.correlate(distance=[1], samples=True)
        assert "tcoord" not in out.coords.coord_map
        _assert_rides(out, "dcoord", patch.get_array("dcoord"))

    def test_tau_p_keeps_time_riders(self, patch):
        """tau-p replaces distance only; time and what rides it stay."""
        with pytest.warns(DASCoreWarning, match="dcoord"):
            out = patch.tau_p(np.linspace(1000, 3000, 5))
        assert "dcoord" not in out.coords.coord_map
        _assert_rides(out, "tcoord", patch.get_array("tcoord"))

    def test_dispersion(self, patch):
        """The dispersion image replaces both dimensions and their riders."""
        with pytest.warns(DASCoreWarning, match=DROPPED):
            out = patch.dispersion_phase_shift(np.arange(1000, 2000, 100))
        assert not {"tcoord", "dcoord"} & set(out.coords.coord_map)

    def test_strain_rate_edgeless(self, patch):
        """Midpoints are not the channels a rider described."""
        velocity = patch.update_attrs(data_type="velocity")
        with pytest.warns(DASCoreWarning, match="dcoord"):
            out = velocity.velocity_to_strain_rate_edgeless()
        assert out.get_coord("distance").units == get_quantity("m")

    def test_same_length_is_silent(self, patch):
        """Replacing a dimension at its own length keeps every rider."""
        with suppress_warnings(DASCoreWarning, action="error"):
            out = patch.update_coords(time=patch.get_coord("time"))
        _assert_rides(out, "tcoord", patch.get_array("tcoord"))


class TestUnits:
    """Units survive operations which rebuild coordinate values."""

    @pytest.mark.parametrize("reduce", ["mean", "median", "first"])
    def test_aggregated_label(self, patch, reduce):
        """A label aggregated from the dimension keeps its units."""
        with pytest.warns(DASCoreWarning):
            out = patch.aggregate("distance", method="mean", dim_reduce=reduce)
        assert out.get_coord("distance").units == get_quantity("m")

    def test_definite_integral(self, patch):
        """The label a definite integral leaves keeps the dimension's units."""
        with pytest.warns(DASCoreWarning):
            out = patch.integrate("distance", definite=True)
        assert out.get_coord("distance").units == get_quantity("m")

    def test_coord_update_values(self, patch):
        """New values keep the coordinate's units unless others are given."""
        coord = patch.get_coord("distance")
        assert coord.update(values=coord.values[::-1]).units == coord.units

    def test_correlate_source(self, patch):
        """A source picked by value keeps the dimension's units."""
        dist = patch.get_array("distance")
        with pytest.warns(DASCoreWarning):
            out = patch.correlate(distance=[dist[1], dist[3]])
        source = out.get_coord("source_distance")
        assert source.units == get_quantity("m")
        assert np.array_equal(source.values, dist[[1, 3]])


class TestMultiDimensional:
    """A coordinate spanning several dimensions cannot pick samples of one."""

    @pytest.fixture()
    def patch_2d(self, random_patch):
        """A patch with a coordinate on both dimensions."""
        values = np.random.default_rng(0).random(random_patch.shape)
        return random_patch.update_coords(quality=(random_patch.dims, values))

    @pytest.mark.parametrize("method", ["select", "order"])
    def test_select_raises_suggesting_mask(self, patch_2d, method):
        """Selecting by it raises and points to a mask."""
        with pytest.raises(CoordError, match="mask"):
            getattr(patch_2d, method)(quality=(0.1, 0.5))
