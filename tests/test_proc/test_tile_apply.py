"""Tests for applying a function to overlapping windows of a patch."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError, PatchError
from dascore.proc.tile_apply import TileApply
from dascore.units import percent


def identity(tiles):
    """The stack as it came."""
    return tiles


def halve(tiles):
    """Every tile at half amplitude."""
    return tiles / 2


def agc(tiles):
    """Every tile scaled to unit RMS."""
    axes = tuple(range(1, tiles.ndim))
    rms = np.sqrt(np.mean(tiles**2, axis=axes, keepdims=True))
    return tiles / np.where(rms > 0, rms, 1)


def _jitted():
    """Return a numba-compiled per-tile function, or skip."""
    numba = pytest.importorskip("numba")
    from dascore.utils._tiles_numba import _JIT_AVAILABLE  # noqa: PLC0415

    if not _JIT_AVAILABLE:
        pytest.skip("numba is not installed")

    @numba.njit
    def half_tile(tile):
        return tile * 0.5

    return half_tile


@pytest.fixture(scope="module")
def patch():
    """The example patch: 300 channels by 2000 samples at 4 ms."""
    return dc.get_example_patch()


@pytest.fixture(scope="module")
def cube():
    """Three shots of 64 channels by 128 samples: one dimension to batch over."""
    rng = np.random.default_rng(0)
    coords = {
        "shot": np.arange(3),
        "distance": np.arange(64) * 2.0,
        "time": dc.to_datetime64(np.arange(128) * 0.004),
    }
    data = rng.normal(size=(3, 64, 128)).astype(np.float32)
    return dc.Patch(data=data, coords=coords, dims=("shot", "distance", "time"))


class TestOverlapAdd:
    """Tiles blended back under the taper."""

    def test_identity(self, patch):
        """An unchanged stack blends back to the input."""
        out = patch.tile_apply(identity, time=0.064, distance=16)
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        np.testing.assert_allclose(out.data, patch.data, atol=1e-5)

    def test_one_dimension(self, patch):
        """One windowed dimension, every other one batched."""
        out = patch.tile_apply(halve, time=0.064)
        np.testing.assert_allclose(out.data, patch.data / 2, atol=1e-5)

    @pytest.mark.parametrize("taper", ["hann", "triang", ("tukey", 0.5)])
    def test_any_complementary_taper(self, patch, taper):
        """The blend is exact for any taper shape, since the ramps are made so."""
        out = patch.tile_apply(identity, taper=taper, time=32, distance=8, samples=True)
        np.testing.assert_allclose(out.data, patch.data, atol=1e-5)

    def test_overlap_forms(self, patch):
        """A percent, a count, and a mapping all say the same overlap."""
        by_percent = patch.tile_apply(
            halve, overlap=25 * percent, time=64, samples=True
        )
        by_count = patch.tile_apply(halve, overlap=16, time=64, samples=True)
        by_map = patch.tile_apply(halve, overlap={"time": 16}, time=64, samples=True)
        assert by_percent.equals(by_count) and by_count.equals(by_map)

    def test_agc_flattens_amplitude(self, patch):
        """Normalized windows leave every region at about the same level."""
        loud = patch.update(data=patch.data * np.linspace(1, 100, patch.shape[1]))
        out = loud.tile_apply(agc, time=0.128, distance=32)
        first = np.std(out.data[:, :200])
        last = np.std(out.data[:, -200:])
        assert 0.5 < first / last < 2
        assert np.std(loud.data[:, :200]) / np.std(loud.data[:, -200:]) < 0.2

    def test_batches_other_dimensions(self, cube):
        """A dimension not windowed is an independent batch."""
        out = cube.tile_apply(halve, distance=16, time=32, samples=True)
        np.testing.assert_allclose(out.data, cube.data / 2, atol=1e-5)

    def test_history_names_a_partial_and_an_object(self, patch):
        """A partial is named by its function, a callable object by its class."""
        from functools import partial  # noqa: PLC0415

        def scale(tiles, by):
            return tiles * by

        class Scaler:
            def __call__(self, tiles):
                return tiles * 3

        by_partial = patch.tile_apply(partial(scale, by=2), time=64, samples=True)
        by_object = patch.tile_apply(Scaler(), time=64, samples=True)
        assert "scale'" in list(by_partial.attrs.history)[-1]
        assert "0x" not in list(by_partial.attrs.history)[-1]
        assert "function='" in list(by_object.attrs.history)[-1]
        assert "0x" not in list(by_object.attrs.history)[-1]

    def test_stack_needs_no_taper(self, patch):
        """The taper is not built for a stack, so a name nothing knows is not asked."""
        stacked = patch.tile_apply(
            identity, mode="stack", taper="windowsXP", time=64, samples=True
        )
        assert stacked.dims[-1] == "time_offset"

    def test_history_names_the_function(self, patch):
        """History says which function, by name, not by address."""
        out = patch.tile_apply(agc, time=64, samples=True)
        assert "function='agc'" in list(out.attrs.history)[-1]


class TestStack:
    """The tiles themselves."""

    @pytest.fixture(scope="class")
    def stacked(self, patch):
        """The example patch cut into 16 by 16 tiles."""
        return patch.tile_apply(identity, mode="stack", time=0.064, distance=16)

    def test_function_is_applied(self, patch):
        """The stack is what the function made of the tiles, not the raw tiles."""
        raw = patch.tile_apply(identity, mode="stack", time=64, samples=True)
        halved = patch.tile_apply(halve, mode="stack", time=64, samples=True)
        np.testing.assert_allclose(halved.data, raw.data / 2)

    def test_compiled_function_refused(self, patch):
        """A per-tile function has no stack to take."""
        half_tile = _jitted()
        with pytest.raises(ParameterError, match="cannot take it"):
            patch.tile_apply(
                half_tile, mode="stack", time=64, distance=16, samples=True
            )

    def test_dims_and_shape(self, stacked, patch):
        """Tile axes where the dimensions were, offsets at the end in axis order."""
        assert stacked.dims == ("distance", "time", "distance_offset", "time_offset")
        assert stacked.shape[2:] == (16, 16)

    def test_centres_and_edges(self, stacked, patch):
        """The tile axis is the centres; start and stop say where each came from."""
        starts = stacked.get_coord("time_start").values
        stops = stacked.get_coord("time_stop").values
        assert starts[0] == -8 and stops[0] == 8  # one stride before the data
        assert np.all(stops - starts == 16)
        centres = stacked.get_coord("time").values
        step = patch.get_coord("time").step
        expected = patch.get_coord("time").min() + dc.to_timedelta64(
            (starts + 7.5) * dc.to_float(step)
        )
        np.testing.assert_array_equal(centres, expected)

    def test_offsets(self, stacked, patch):
        """The offset within a tile, in the dimension's units from zero."""
        offsets = stacked.get_coord("time_offset").values
        assert len(offsets) == 16
        assert offsets[0] == np.timedelta64(0, "ns")
        assert offsets[1] == patch.get_coord("time").step

    def test_source_is_kept(self, stacked, patch):
        """The coordinate the tiles were cut from travels with them."""
        assert stacked.get_coord("_tile_source_time") == patch.get_coord("time")

    def test_float_coordinate(self, patch):
        """A numeric dimension gets numeric centres and offsets."""
        stacked = patch.tile_apply(identity, mode="stack", distance=16, samples=True)
        assert stacked.get_coord("distance_offset").values[1] == 1.0
        assert stacked.dims == ("distance", "time", "distance_offset")

    def test_descending_coordinate(self, patch):
        """Centres step down a descending coordinate from its first sample."""
        flipped = patch.flip("distance", flip_coords=True)
        stacked = flipped.tile_apply(identity, mode="stack", distance=16, samples=True)
        centres = stacked.get_coord("distance").values
        assert centres[0] > centres[1]
        first = flipped.get_coord("distance").values[0]
        assert centres[0] == first + (-8 + 7.5) * flipped.get_coord("distance").step
        # Offsets count from the tile's first sample, in the coordinate's direction.
        offsets = stacked.get_coord("distance_offset").values
        assert offsets[0] == 0 and offsets[1] == flipped.get_coord("distance").step
        assert stacked.reassemble().equals(flipped, close=True)

    def test_source_name_collision_refused(self, patch):
        """The coordinate the source travels under is claimed too."""
        clashing = patch.update_coords(_tile_source_time=("time", np.zeros(2000)))
        with pytest.raises(ParameterError, match="_tile_source_time"):
            clashing.tile_apply(identity, mode="stack", time=64, samples=True)

    def test_collision_refused(self, patch):
        """A coordinate already called what a stack would call one is refused."""
        clashing = patch.update_coords(time_offset=("time", np.zeros(2000)))
        with pytest.raises(ParameterError, match="already has a coordinate"):
            clashing.tile_apply(identity, mode="stack", time=64, samples=True)


class TestReassemble:
    """Blending a stack back."""

    def test_round_trip_two_dimensions(self, patch):
        """An unchanged stack reassembles to the patch, coordinates and all."""
        stacked = patch.tile_apply(identity, mode="stack", time=0.064, distance=16)
        back = stacked.reassemble()
        assert back.equals(patch, close=True)
        assert not any(k.startswith("_tile") for k in back.coords.coord_map)
        assert not any(k.startswith("_tile") for k in dict(back.attrs))
        assert "_tile_stride_time" in dict(stacked.attrs)

    def test_round_trip_one_dimension(self, patch):
        """And with one dimension windowed."""
        back = patch.tile_apply(
            identity, mode="stack", time=64, samples=True
        ).reassemble()
        assert back.equals(patch, close=True)

    def test_round_trip_batched(self, cube):
        """A dimension not windowed is carried through both ways."""
        stacked = cube.tile_apply(
            identity, mode="stack", distance=16, time=32, samples=True
        )
        assert stacked.dims == (
            "shot",
            "distance",
            "time",
            "distance_offset",
            "time_offset",
        )
        assert stacked.reassemble().equals(cube, close=True)

    def test_coordinates_riding_a_windowed_dimension_come_back(self, patch):
        """A per-sample coordinate along a windowed dimension round-trips."""
        flagged = patch.update_coords(quality=("time", np.arange(2000) % 3))
        stacked = flagged.tile_apply(identity, mode="stack", time=64, samples=True)
        assert "quality" not in stacked.dims
        back = stacked.reassemble()
        assert back.equals(flagged, close=True)
        np.testing.assert_array_equal(
            back.get_coord("quality").values, np.arange(2000) % 3
        )

    def test_edit_between(self, patch):
        """Work done on the stack is what comes back: halve, then blend."""
        stacked = patch.tile_apply(identity, mode="stack", time=64, samples=True)
        back = stacked.update(data=stacked.data / 2).reassemble()
        np.testing.assert_allclose(back.data, patch.data / 2, atol=1e-5)

    def test_reordered_tiles_go_back_where_they_came_from(self, patch):
        """Each tile is placed by its start, whatever order the stack is in."""
        stacked = patch.tile_apply(identity, mode="stack", time=64, samples=True)
        n = stacked.shape[stacked.get_axis("time")]
        reversed_stack = stacked.order(time=np.arange(n)[::-1], samples=True)
        assert reversed_stack.reassemble().equals(patch, close=True)

    def test_dropped_tiles_leave_their_region_quiet(self, patch):
        """A stack with tiles removed blends what is left; the gap stays silent."""
        stacked = patch.tile_apply(identity, mode="stack", time=64, samples=True)
        n = stacked.shape[stacked.get_axis("time")]
        kept = stacked.select(time=(2, n - 2), samples=True)
        back = kept.reassemble()
        assert back.shape == patch.shape
        # The middle is untouched; the ends, whose tiles are gone, are not.
        np.testing.assert_allclose(
            back.data[:, 200:-200], patch.data[:, 200:-200], atol=1e-5
        )
        assert np.abs(back.data[:, :8]).max() == 0

    def test_every_other_tile(self, patch):
        """A stack thinned to alternate tiles blends under the taper it was cut for."""
        stacked = patch.tile_apply(identity, mode="stack", time=64, samples=True)
        n = stacked.shape[stacked.get_axis("time")]
        thinned = stacked.select(time=np.arange(0, n, 2), samples=True)
        back = thinned.reassemble()
        assert back.shape == patch.shape
        assert not any(k.startswith("_tile") for k in dict(back.attrs))
        # Every other tile is gone, so no sample sees more than one: the
        # blend is the kept tiles under their taper, nothing doubled.
        assert np.abs(back.data).max() <= np.abs(patch.data).max() * 1.01

    def test_needs_a_stack(self, patch):
        """A patch nobody tiled cannot be reassembled."""
        with pytest.raises(PatchError, match="has no tiles"):
            patch.reassemble()


class TestEngines:
    """numpy over the stack, numba over one tile at a time."""

    def test_numba_matches_numpy(self, patch):
        """The compiled per-tile function blends to what the stack function does."""
        half_tile = _jitted()
        by_numba = patch.tile_apply(half_tile, time=64, distance=16, samples=True)
        by_numpy = patch.tile_apply(halve, time=64, distance=16, samples=True)
        np.testing.assert_allclose(by_numba.data, by_numpy.data, atol=1e-5)
        assert by_numba.dims == patch.dims

    def test_numba_keeps_the_function_s_dtype(self, patch):
        """A compiled function which makes a real tile complex keeps it complex."""
        numba = pytest.importorskip("numba")
        _jitted()

        @numba.njit
        def to_complex(tile):
            return tile * (1 + 1j)

        out = patch.tile_apply(to_complex, time=64, distance=16, samples=True)
        assert np.iscomplexobj(out.data)
        np.testing.assert_allclose(out.data.imag, patch.data, atol=1e-4)

    def test_driver_runs_in_python(self, patch):
        """The driver gives the same answer uncompiled."""
        half_tile = _jitted()
        from dascore.utils._tiles_numba import _apply_colour_class  # noqa: PLC0415
        from dascore.utils.signal import get_taper  # noqa: PLC0415
        from dascore.utils.tiles import get_tile_plan  # noqa: PLC0415

        data = np.asarray(patch.data[:40, :64], dtype=np.float32)
        plan = get_tile_plan(data.shape, (8, 16), (5, 9))
        taper = get_taper("hann", (8, 16), (3, 7))
        padded, out = plan.pad(data), np.zeros(plan.extended, dtype=np.float32)
        for c0 in range(plan.colours[0]):
            for c1 in range(plan.colours[1]):
                _apply_colour_class.func(
                    padded,
                    out,
                    taper,
                    *plan.size,
                    *plan.stride,
                    *plan.grid,
                    *plan.colours,
                    c0,
                    c1,
                    half_tile,
                )
        np.testing.assert_allclose(plan.crop(out), data / 2, atol=1e-5)

    def test_jitted_function_on_numpy_engine_refused(self, patch):
        """A per-tile function cannot take the stack."""
        half_tile = _jitted()
        with pytest.raises(ParameterError, match="one tile at a time"):
            patch.tile_apply(half_tile, engine="numpy", time=64, samples=True)

    def test_plain_function_on_numba_engine_refused(self, patch):
        """And a stack function cannot be compiled."""
        with pytest.raises(ParameterError, match="numba-compiled function"):
            patch.tile_apply(halve, engine="numba", time=64, distance=16, samples=True)

    @pytest.mark.parametrize("engine", ["auto", "numba"])
    def test_compiled_function_needs_two_dimensions(self, patch, engine):
        """The driver tiles two dimensions, and says so however it was reached."""
        half_tile = _jitted()
        with pytest.raises(ParameterError, match="exactly two windowed dimensions"):
            patch.tile_apply(half_tile, time=64, samples=True, engine=engine)

    def test_numba_engine_without_numba(self, patch, monkeypatch):
        """Asked for by name with numba absent, it says what is missing."""
        import dascore.utils._tiles_numba as driver  # noqa: PLC0415

        half_tile = _jitted()
        monkeypatch.setattr(driver, "_JIT_AVAILABLE", False)
        with pytest.raises(dc.exceptions.MissingOptionalDependencyError):
            patch.tile_apply(half_tile, time=64, distance=16, samples=True)

    def test_unknown_engine_refused(self, patch):
        """A name which is not an engine."""
        with pytest.raises(ParameterError, match="engine must be one of"):
            patch.tile_apply(halve, engine="cuda", time=64, samples=True)


class TestArguments:
    """What the call refuses."""

    def test_unknown_mode(self, patch):
        """A mode which is not one."""
        with pytest.raises(ParameterError, match="mode must be one of"):
            patch.tile_apply(halve, mode="centre", time=64, samples=True)

    def test_needs_a_dimension(self, patch):
        """No window, nothing to tile."""
        with pytest.raises(ParameterError, match="needs a dimension"):
            patch.tile_apply(halve)

    def test_overlap_past_half_refused(self, patch):
        """The taper's ramps would cross."""
        with pytest.raises(ParameterError, match="ramps would cross"):
            patch.tile_apply(halve, overlap=40, time=64, samples=True)

    def test_op_is_the_processor(self, patch):
        """The seam: the call names the processor and the two routes agree."""
        op = dc.proc.tile_apply.op(halve, time=64, samples=True)
        assert isinstance(op, TileApply)
        assert op(patch).equals(patch.tile_apply(halve, time=64, samples=True))

    def test_cannot_be_written_to_a_document(self, patch):
        """An operation holding a function is not one a document can hold."""
        op = dc.proc.tile_apply.op(halve, time=64, samples=True)
        with pytest.raises(ParameterError, match="cannot be written"):
            op.to_dict()
