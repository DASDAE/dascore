"""Tests for correlate patch processing function."""

import warnings

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import ParameterError, UnitError
from dascore.proc.correlate import Correlate
from dascore.units import get_quantity, m
from dascore.utils.time import to_float


class TestCorrelateShift:
    """Tests for the correlation shift function."""

    @pytest.fixture(scope="class")
    def random_patch_odd(self):
        """Create a random patch with odd number of time samples."""
        patch = dc.get_example_patch("random_das", shape=(2, 11))
        return patch

    @pytest.fixture(scope="class")
    def random_patch_even(self):
        """Create a random patch with even number of time samples."""
        patch = dc.get_example_patch("random_das", shape=(2, 10))
        return patch

    def test_auto_correlation(self, random_dft_patch):
        """Perform auto correlation and undo shifting."""
        dft_conj = random_dft_patch.conj()
        dft_sq = random_dft_patch * dft_conj
        idft = dft_sq.idft()
        auto_patch = idft.correlate_shift(dim="time")
        assert np.allclose(np.imag(auto_patch.data), 0)
        assert "lag_time" in auto_patch.dims
        coord_array = auto_patch.get_array("lag_time")
        # ensure the max value happens at zero lag time.
        time_ax = auto_patch.get_axis("lag_time")
        argmax = np.argmax(random_dft_patch.data, axis=time_ax)
        assert np.all(coord_array[argmax] == dc.to_timedelta64(0))

    def test_source_bounds_dropped(self):
        """The DFT/IDFT workflow discards position cells when creating lag bins."""
        values = np.arange(8.0)
        patch = dc.Patch(
            data=np.arange(8.0),
            dims=("x",),
            coords={
                "x": values,
                "x_start": ("x", values - 2),
                "x_stop": ("x", values + 2),
            },
        )
        dft = patch.dft("x")
        restored = (dft * dft.conj()).idft()
        assert "x_start" in restored.coords
        out = restored.correlate_shift("x")
        assert out.dims == ("lag_x",)
        assert "lag_x_start" not in out.coords
        assert "lag_x_stop" not in out.coords
        np.testing.assert_array_equal(out.get_array("lag_x"), np.arange(-4.0, 4.0))

    def test_auto_correlation_odd_coord(self, random_patch_odd):
        """Ensure correlate_shift works when dim's coord length is odd."""
        dft = random_patch_odd.dft(dim="time")
        dft_conj = dft.conj()
        dft_sq = dft * dft_conj
        idft = dft_sq.idft()
        assert isinstance(idft.correlate_shift(dim="time"), dc.Patch)

    def test_auto_correlation_even_coord(self, random_patch_even):
        """Ensure correlate_shift works when dim's coord length is even."""
        dft = random_patch_even.dft(dim="time")
        dft_conj = dft.conj()
        dft_sq = dft * dft_conj
        idft = dft_sq.idft()
        assert isinstance(idft.correlate_shift(dim="time"), dc.Patch)


class TestCorrelateInternal:
    """Tests case of intra-patch correlation function."""

    moveout_velocity = 100

    @pytest.fixture(scope="class")
    def corr_patch(self):
        """Create a patch of sin waves whose correlation can be easily checked."""
        patch = dc.get_example_patch(
            "sin_wav",
            sample_rate=100,
            frequency=range(10, 20),
            duration=5,
            channel_count=10,
        ).taper(time=0.05)
        # normalize energy so autocorrection is 1
        time_axis = patch.get_axis("time")
        data = patch.data
        norm = np.linalg.norm(data, axis=time_axis, keepdims=True)
        return patch.update(data=data / norm)

    @pytest.fixture(scope="session")
    def ricker_moveout_patch(self):
        """Create a patch with a ricker moveout."""
        patch = dc.get_example_patch(
            "ricker_moveout",
            velocity=self.moveout_velocity,
            duration=5,
        )
        return patch

    def test_basic_correlation(self, corr_patch):
        """Ensure correlation works along distance dim."""
        dist = corr_patch.get_coord("distance")[0]
        out = corr_patch.correlate(distance=dist)
        fft_ax = corr_patch.get_axis("time")
        fft_len = len(out.get_coord("lag_time"))
        argmax = np.argmax(out.data, axis=fft_ax)
        max_values = np.max(out.data, axis=fft_ax)
        # since the traces are all the same, the auto-correlation, or zero
        # lag should be maximized.
        assert np.all(np.isclose(argmax, fft_len // 2))
        # the max values should also be near 1 since the patch fixture
        # was normalized.
        assert np.allclose(max_values, 1.0, atol=0.01)
        # the sampling rates should not have changed.
        assert np.isclose(
            to_float(out.get_coord("lag_time").step),
            to_float(corr_patch.get_coord("time").step),
        )
        dstep1 = out.get_coord("distance").step
        dstep2 = corr_patch.get_coord("distance").step
        assert dstep1 == dstep2

    def test_transpose_independent(self, corr_patch):
        """The order of the dims shouldn't affect the result."""
        new_order = corr_patch.dims[::-1]
        patch1 = corr_patch
        patch2 = corr_patch.transpose(*new_order)
        corr1 = patch1.correlate(time=3, samples=True)
        corr2 = patch2.correlate(time=3, samples=True)
        assert corr1.transpose(*corr2.dims).equals(corr2)

    def test_time_lags(self, ricker_moveout_patch):
        """Ensure time lags are consistent with expected velocities."""
        corr = ricker_moveout_patch.correlate(distance=0)
        # get predicted lat times
        argmax = np.argmax(corr.data, axis=0)
        distances = corr.get_coord("distance").values
        lag_times = to_float(corr.get_coord("lag_time").values[argmax])
        # get calculated times, they should be close to lag times
        expected_times = distances / self.moveout_velocity
        assert np.allclose(lag_times.flatten(), expected_times)

    def test_units(self, random_patch):
        """Ensure units can be passed as kwarg params."""
        c_patch = random_patch.correlate(distance=10 * m)
        assert isinstance(c_patch, dc.Patch)

    def test_complex_patch(self, ricker_moveout_patch):
        """Ensure correlate works with a patch that has complex data."""
        data = ricker_moveout_patch.data
        rm_patch = ricker_moveout_patch.new(data=data + data * 1j)
        corr = rm_patch.correlate(distance=0)
        # get predicted lat times
        argmax = np.argmax(corr.data, axis=0)
        distances = corr.get_coord("distance").values
        lag_times = to_float(corr.get_coord("lag_time").values[argmax])
        # get calculated times, they should be close to lag times
        expected_times = distances / self.moveout_velocity
        assert np.allclose(lag_times.flatten(), expected_times)

    def test_correlation_freq_domain_patch(self, corr_patch):
        """
        Test correlation when the input patch is already in the frequency
        domain.
        """
        # perform FFT on the original patch to simulate frequency domain data
        fft_patch = corr_patch.dft("time")
        out = fft_patch.correlate(distance=0, samples=True)
        # The patch should still be complex.
        assert np.issubdtype(out.data.dtype, np.complexfloating)
        # Check if the shape of the output is the same as the original patch
        # plus a dimension 1 for the source.
        assert out.shape == tuple([*corr_patch.shape, 1])

    def test_correlate_decimated_patch(self, corr_patch):
        """Ensure a decimated patch can be correlated."""
        out = corr_patch.decimate(distance=2, filter_type=None).correlate(
            distance=1, samples=True
        )
        assert isinstance(out, dc.Patch)

    def test_correlate_units_raises(self, corr_patch):
        """When the patch doesn't have units an error should raise."""
        patch = corr_patch.set_units(distance=None)
        with pytest.raises(UnitError):
            patch.correlate(distance=0 * m)

    def test_correlate_units(self, corr_patch):
        """When the patch has units it should work to specify them."""
        patch = corr_patch.set_units(distance="m")
        out1 = patch.correlate(distance=1 * m)
        assert isinstance(out1, dc.Patch)
        out2 = patch.correlate(distance=np.array([1, 2]) * m)
        assert isinstance(out2, dc.Patch)

    def test_lag_removed(self, corr_patch):
        """Ensure the removed lag parameter is rejected."""
        with pytest.raises(TypeError, match="lag"):
            corr_patch.correlate(time=1, lag=10, samples=True)


class TestCorrelatePaddedTransform:
    """Correlating a padded transform keeps every lag."""

    @pytest.fixture()
    def patch(self):
        """Two traces, the first leading by 3 samples, of a non-fast length."""
        # 230 is not a fast FFT length, so dft pads it to 231.
        trace = np.zeros(230)
        trace[100] = 1
        coords = {"distance": [0.0, 1.0], "time": np.arange(230) * 0.01}
        data = np.vstack([np.roll(trace, -3), trace])
        return dc.Patch(data=data, dims=("distance", "time"), coords=coords)

    def test_negative_lag_after_padding(self, patch):
        """A 3-sample lead reads as -3 samples when dft pads the record."""
        out = (
            patch.dft("time", real=True)
            .correlate(distance=1, samples=True)
            .idft()
            .correlate_shift("time")
            .squeeze()
        )
        lags = out.get_array("lag_time")
        assert np.isclose(lags[np.argmax(out.data[0])], -0.03)

    def test_only_the_lag_dim_keeps_its_padding(self, random_patch):
        """A dim transformed alongside the correlated one is still trimmed."""
        patch = random_patch.select(distance=(0, 17), time=(0, 230), samples=True)
        spectra = patch.dft(("distance", "time"), real="time")
        out = spectra.correlate(ft_distance=1, samples=True).idft()
        assert out.shape[out.get_axis("distance")] == 17


class TestCorrelatePrecision:
    """Single precision data are correlated in single precision."""

    def test_float32_stays_float32(self, random_patch):
        """A float32 patch correlates to float32, close to the float64 result."""
        single = random_patch.new(data=random_patch.data.astype(np.float32))
        out = single.correlate(distance=0, samples=True)
        expected = random_patch.correlate(distance=0, samples=True)
        assert out.dtype == np.float32
        assert np.allclose(out.data, expected.data, rtol=1e-3, atol=1e-3)


class TestCorrelationUnits:
    """Correlations carry the square of the data units."""

    def test_both_routes_give_squared_units(self, random_patch):
        """Correlate and the dft pipeline both give (m/s)**2 for m/s data."""
        patch = random_patch.update_attrs(data_units="m/s")
        direct = patch.correlate(distance=0, samples=True)
        spectra = patch.dft("time", real=True)
        piped = (spectra * spectra.conj()).idft().correlate_shift("time")
        expected = get_quantity("m/s") ** 2
        for out in (direct, piped):
            assert get_quantity(out.attrs.data_units) == expected

    def test_shift_leaves_units_it_does_not_rescale(self, random_patch):
        """Without weighting undone, or without data units, units are kept."""
        spectra = random_patch.update_attrs(data_units="m/s").dft("time", real=True)
        product = (spectra * spectra.conj()).idft()
        kept = product.correlate_shift("time", undo_weighting=False)
        assert kept.attrs.data_units == product.attrs.data_units
        unitless = random_patch.dft("time", real=True)
        shifted = (unitless * unitless.conj()).idft().correlate_shift("time")
        assert shifted.attrs.data_units is None


class TestCorrelateErrors:
    """Tests for correlate input validation."""

    def test_non_2d_patch_raises(self):
        """Correlate requires a 2D patch."""
        coord = dc.get_coord(start=0, step=1, stop=10)
        patch_1d = dc.Patch(np.arange(10.0), coords={"time": coord}, dims=("time",))
        with pytest.raises(ParameterError, match="2D patch"):
            patch_1d.correlate(time=0, samples=True)


class TestCorrelateND:
    """Patches with more than 2 dims correlate each batch separately."""

    @pytest.fixture()
    def patch_3d(self):
        """A (source, distance, time) patch with data units."""
        rng = np.random.default_rng(42)
        coords = {
            "source": np.arange(3),
            "distance": np.array([0.0, 1.0]),
            "time": np.arange(400) * 0.005,
        }
        dims = ("source", "distance", "time")
        patch = dc.Patch(data=rng.random((3, 2, 400)), coords=coords, dims=dims)
        return patch.update_attrs(data_units="m/s")

    def _assert_matches_2d(self, out, patch, batch):
        """Each batch slice of out equals correlating that 2D slice."""
        for index in np.ndindex(*(len(patch.get_coord(x)) for x in batch)):
            select = dict(zip(batch, index))
            part = patch.select(**select, samples=True).squeeze(batch)
            expected = part.correlate(distance=0, samples=True)
            got = out.select(**select, samples=True).squeeze(batch)
            assert got.dims == expected.dims
            assert np.allclose(got.data, expected.data)
            assert got.update(data=expected.data).equals(expected)

    def test_3d_matches_2d_slices(self, patch_3d):
        """The issue's 3D patch gives the per-source 2D results."""
        out = patch_3d.correlate(distance=0, samples=True, lag_dim="time")
        assert out.dims == ("source", "distance", "lag_time", "source_distance")
        assert get_quantity(out.attrs.data_units) == get_quantity("m/s") ** 2
        self._assert_matches_2d(out, patch_3d, ["source"])

    def test_4d_matches_2d_slices(self, patch_3d):
        """Two batch dims, on either side of the source dim, broadcast."""
        data = np.stack([patch_3d.data, 2 * patch_3d.data], axis=2)
        coords = {**dict(patch_3d.coords.coord_map), "shot": np.array([10, 20])}
        dims = ("source", "distance", "shot", "time")
        patch = dc.Patch(data=data, coords=coords, dims=dims)
        out = patch.correlate(distance=0, samples=True, lag_dim="time")
        assert out.dims == (*dims[:3], "lag_time", "source_distance")
        self._assert_matches_2d(out, patch, ["source", "shot"])

    @pytest.mark.parametrize("pad", (True, False))
    def test_transformed_batch_dim_stays_transformed(self, patch_3d, pad):
        """Only the lag dim is inverted; a transformed batch dim is kept."""
        spectra = patch_3d.dft("source", pad=pad)
        out = spectra.correlate(distance=0, samples=True, lag_dim="time")
        assert out.dims[0] == "ft_source"
        self._assert_matches_2d(out, spectra, ["ft_source"])

    def test_padded_batch_dim_inverts_to_its_length(self, patch_3d):
        """Inverting a padded batch dim after correlating restores its length."""
        # 17 sources, which dft pads to 18.
        data = np.random.default_rng(42).random((17, 2, 400))
        coords = patch_3d.coords.update(source=np.arange(17))
        patch = patch_3d.new(data=data, coords=coords)
        out = patch.dft("source").correlate(distance=0, samples=True, lag_dim="time")
        assert out.idft("source").get_coord("source") == patch.get_coord("source")

    def test_missing_lag_dim_raises(self, patch_3d):
        """A patch with more than 2 dims must name the lag dim."""
        with pytest.raises(ParameterError, match="lag_dim must be one of"):
            patch_3d.correlate(distance=0, samples=True)

    @pytest.mark.parametrize("lag_dim", ("distance", "bob"))
    def test_bad_lag_dim_raises(self, patch_3d, lag_dim):
        """The lag dim must be a dim other than the source dim."""
        with pytest.raises(ParameterError, match="lag_dim must be one of"):
            patch_3d.correlate(distance=0, samples=True, lag_dim=lag_dim)


class TestCorrelateShiftProcessor:
    """What the CorrelateShift class guarantees beyond the framework."""

    def test_metadata_without_data(self, random_patch):
        """The lag coordinate and step come from metadata alone."""
        shift = dc.proc.CorrelateShift(dim="distance")
        out, plan = shift.get_metadata(random_patch.drop_data())
        assert out.dims == ("lag_distance", "time")
        assert out.get_coord("lag_distance").min() == -150
        assert plan == {"axis": 0, "step": 1.0}

    @pytest.mark.parametrize("undo_weighting", (True, False))
    def test_undo_weighting(self, random_patch, undo_weighting):
        """Only undoing the weighting divides the shifted data by the step."""
        ints = random_patch.new(data=np.arange(random_patch.size).reshape(300, -1))
        out = ints.correlate_shift("time", undo_weighting=undo_weighting)
        shifted = np.fft.fftshift(ints.data, axes=1)
        expected = shifted / 0.004 if undo_weighting else shifted
        assert out.dtype == expected.dtype
        assert np.array_equal(out.data, expected)

    @pytest.mark.parametrize("undo_weighting", (True, False))
    def test_single_precision_stays_single(self, random_patch, undo_weighting):
        """float32 data stay float32 when divided by the step, or not."""
        single = random_patch.new(data=random_patch.data.astype(np.float32))
        out = single.correlate_shift("time", undo_weighting=undo_weighting)
        assert out.dtype == np.float32

    @pytest.mark.parametrize("undo_weighting", (True, False))
    @pytest.mark.parametrize(
        "dtype", ("bool", "int16", "float16", "float32", "float64")
    )
    def test_metadata_dtype_is_the_kernels(self, random_patch, dtype, undo_weighting):
        """Metadata states the dtype the shifted data come out in."""
        patch = random_patch.new(data=random_patch.data.astype(dtype))
        shift = dc.proc.CorrelateShift("time", undo_weighting=undo_weighting)
        meta = shift.get_metadata(patch.drop_data())[0]
        assert meta.dtype == shift.run(patch).dtype

    def test_half_precision_does_not_overflow(self, random_patch):
        """Half precision is divided by a fine step without overflowing."""
        half = random_patch.new(data=np.full(random_patch.shape, 1000, np.float16))
        assert np.all(np.isfinite(half.correlate_shift("time").data))


class TestCorrelateMetadata:
    """Correlate works out its result from metadata alone."""

    def test_lags_and_sources(self, random_patch):
        """The lag axis spans the padded length, with one source column."""
        corr = Correlate(distance=2, samples=True)
        out, _ = corr.get_metadata(random_patch.drop_data())
        assert out.dims == ("distance", "lag_time", "source_distance")
        assert out.shape == (300, 4000, 1)
        assert out.dtype == corr(random_patch).dtype == np.float64


class TestCorrelateNormalize:
    """Normalized correlation divides by the trace and source energies."""

    @pytest.fixture()
    def patch(self):
        """A small random patch with data units."""
        patch = dc.get_example_patch("random_das", shape=(6, 51))
        return patch.update_attrs(data_units="m/s")

    def _zero_lag(self, out):
        """Return the zero-lag slice of a correlation along lag_time."""
        lags = to_float(out.get_array("lag_time"))
        return out.select(lag_time=np.argmin(np.abs(lags)), samples=True)

    def test_autocorrelation_peak_is_one(self, patch):
        """Every trace correlated with itself is one at zero lag."""
        out = patch.correlate(distance=list(range(6)), samples=True, normalize=True)
        assert np.allclose(np.diagonal(self._zero_lag(out).squeeze("lag_time").data), 1)

    def test_bounded(self, patch):
        """Every value lies in [-1, 1]."""
        out = patch.correlate(distance=[0, 3], samples=True, normalize=True)
        assert np.abs(out.data).max() <= 1 + 1e-12

    def test_matches_manual_normalization(self):
        """The result equals dividing by the energies by hand."""
        patch = dc.get_example_patch().pass_filter(time=(5, 50))
        energy = (patch**2).sum("time").data
        cc = patch.correlate(distance=0, samples=True).squeeze()
        expected = cc.data / np.sqrt(energy * energy[0])
        out = patch.correlate(distance=0, samples=True, normalize=True).squeeze()
        assert np.allclose(out.data, expected)

    def test_multiple_sources(self, patch):
        """Each source of a batch matches correlating it alone."""
        out = patch.correlate(distance=[1, 4], samples=True, normalize=True)
        for i, source in enumerate([1, 4]):
            single = patch.correlate(distance=source, samples=True, normalize=True)
            assert np.allclose(out.data[..., i], single.data[..., 0])
        assert np.allclose(self._zero_lag(out).data[[1, 4], 0, [0, 1]], 1)

    def test_nd_patch(self):
        """Each batch of an N-D patch is normalized by its own traces."""
        rng = np.random.default_rng(0)
        coords = {"source": [0, 1], "distance": [0.0, 1.0], "time": np.arange(40)}
        patch = dc.Patch(data=rng.random((2, 2, 40)), coords=coords, dims=list(coords))
        kwargs = dict(distance=0, samples=True, normalize=True)
        out = patch.correlate(lag_dim="time", **kwargs)
        for i in range(2):
            part = patch.select(source=i, samples=True).squeeze("source")
            assert np.allclose(out.data[i], part.correlate(**kwargs).data)
        assert np.allclose(self._zero_lag(out).data[:, 0, 0, 0], 1)

    @pytest.mark.parametrize("real", [True, False])
    def test_frequency_domain_matches(self, patch, real):
        """A transformed patch normalizes as the time-domain one does."""
        expected = patch.correlate(distance=[0, 2], samples=True, normalize=True)
        dft = patch.pad(time="correlate").dft("time", real=real)
        out = dft.correlate(distance=[0, 2], samples=True, normalize=True)
        out = out.idft().correlate_shift("time")
        assert np.allclose(out.data, expected.data)
        assert np.allclose(self._zero_lag(out).data[2, 0, 1], 1)
        assert get_quantity(out.attrs.data_units).dimensionless

    def test_zero_trace_gives_zeros(self, patch):
        """A trace with no energy correlates to zeros without a warning."""
        data = patch.data.copy()
        data[2] = 0
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = patch.new(data=data).correlate(
                distance=[0, 2], samples=True, normalize=True
            )
        assert np.all(out.data[2] == 0)
        assert np.all(out.data[..., 1] == 0)

    def test_units_dimensionless(self, patch):
        """The result is dimensionless, with or without coordinate units."""
        out = patch.correlate(distance=0, samples=True, normalize=True)
        assert get_quantity(out.attrs.data_units).dimensionless
        no_units = patch.update_coords(time=np.arange(51)).set_units(time=None)
        out = no_units.correlate(distance=0, samples=True, normalize=True)
        assert get_quantity(out.attrs.data_units) == get_quantity("dimensionless")

    @pytest.mark.parametrize("scale", [1e-13, 1e10])
    def test_float32_amplitude_invariant(self, scale):
        """Tiny or huge single-precision traces still peak at one."""
        time = np.arange(51) * 0.004
        wave = np.exp(-(((time - 0.1) / 0.02) ** 2))
        data = (np.stack([wave, wave[::-1]]) * scale).astype(np.float32)
        coords = {"distance": [0.0, 1.0], "time": time}
        patch = dc.Patch(data=data, coords=coords, dims=("distance", "time"))
        out = patch.correlate(distance=0, samples=True, normalize=True)
        assert out.dtype == np.float32
        assert np.isclose(np.abs(out.data).max(), 1, atol=1e-5)

    def test_nan_source_stays_nan(self, patch):
        """A NaN in the source gives NaN, not zeros, like the default."""
        data = patch.data.copy()
        data[0, 3] = np.nan
        out = patch.new(data=data).correlate(
            distance=[0, 1], samples=True, normalize=True
        )
        assert np.all(np.isnan(out.data[..., 0]))
        assert not np.any(np.isnan(out.data[1, :, 1]))


class TestCorrelateOther:
    """Correlating each trace with the matching trace of another patch."""

    step = 0.01

    def _patch(self, data, time_start=0.0, **coords):
        """Return a patch whose last dimension is time."""
        time = time_start + np.arange(data.shape[-1]) * self.step
        coords = {**coords, "time": time}
        return dc.Patch(data=data, coords=coords, dims=list(coords))

    def _expected(self, out, x, y):
        """Return out's trace beside numpy's full correlation of x with y."""
        lags = np.round(to_float(out.get_array("lag_time")) / self.step)
        full = (lags >= -(len(y) - 1)) & (lags <= len(x) - 1)
        assert np.allclose(out.data[~full], 0)
        return out.data[full], np.correlate(x, y, "full")

    def test_matches_source(self):
        """A source trace as other gives the single-patch result."""
        patch = dc.get_example_patch("random_das", shape=(4, 51))
        expected = patch.correlate(distance=2, samples=True).squeeze()
        other = patch.select(distance=2, samples=True).squeeze()
        out = patch.correlate(other=other, lag_dim="time")
        assert out.coords == expected.coords
        assert np.allclose(out.data, expected.data)

    def test_aligned_patches(self):
        """Each shared channel is correlated with its counterpart."""
        rng = np.random.default_rng(0)
        a = self._patch(rng.random((3, 40)), distance=[0.0, 1.0, 2.0])
        b = self._patch(rng.random((3, 40)), time_start=5, distance=[1.0, 2.0, 3.0])
        out = a.correlate(other=b, lag_dim="time")
        assert np.all(out.get_array("distance") == [1.0, 2.0])
        for i in range(2):
            trace = out.isel(distance=i)
            assert np.allclose(*self._expected(trace, a.data[i + 1], b.data[i]))

    def test_broadcast_dim(self):
        """A dimension only one patch has is broadcast."""
        rng = np.random.default_rng(1)
        a = self._patch(rng.random((2, 2, 30)), source=[0, 1], distance=[0.0, 1.0])
        b = self._patch(rng.random((2, 30)), distance=[0.0, 1.0])
        out = a.correlate(other=b, lag_dim="time")
        assert out.get_coord("source").shape == (2,)
        trace = out.isel(source=1, distance=0)
        assert np.allclose(*self._expected(trace, a.data[1, 0], b.data[0]))

    @pytest.mark.parametrize("offset", [0, 37, 80])
    def test_template_shift(self, offset):
        """A short template peaks at the lag where it sits in the record."""
        template = np.sin(np.arange(20.0))
        data = np.zeros(100)
        data[offset : offset + 20] = template
        record = self._patch(data)
        other = self._patch(template, time_start=3)
        out = record.correlate(other=other)
        lags = to_float(out.get_array("lag_time"))
        assert np.isclose(lags[np.argmax(out.data)], offset * self.step)
        assert np.allclose(*self._expected(out, data, template))
        # Swapping the patches flips the sign.
        swapped = other.correlate(other=record)
        lags = to_float(swapped.get_array("lag_time"))
        assert np.isclose(lags[np.argmax(swapped.data)], -offset * self.step)

    @pytest.mark.parametrize("scale", [1e-6, 3.0, 1e6])
    def test_normalized_template_peaks_at_one(self, scale):
        """An embedded template, at any scale, peaks at one."""
        template = np.sin(np.arange(20.0))
        data = np.zeros(100)
        data[30:50] = template * scale
        out = self._patch(data).correlate(other=self._patch(template), normalize=True)
        assert np.isclose(np.abs(out.data).max(), 1)

    def test_units(self):
        """Data units are the product of both patches', or none."""
        rng = np.random.default_rng(2)
        a = self._patch(rng.random(20)).set_units("m/s")
        b = self._patch(rng.random(10)).set_units("m")
        units = a.correlate(other=b).attrs.data_units
        assert get_quantity(units) == get_quantity("m**2/s")
        normalized = a.correlate(other=b, normalize=True).attrs.data_units
        assert get_quantity(normalized) == get_quantity("1")
        assert (
            a.correlate(other=b.update_attrs(data_units=None)).attrs.data_units is None
        )

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            ({"other": np.ones(10)}, "must be a Patch"),
            ({"distance": 0}, "cannot be combined"),
            ({"samples": True}, "cannot be combined"),
            ({}, "lag_dim must name"),
            ({"lag_dim": "distance"}, "lag_dim must name"),
        ],
    )
    def test_refused(self, kwargs, match):
        """Arguments other cannot be used with are refused."""
        patch = self._patch(np.ones((2, 10)), distance=[0.0, 1.0])
        kwargs = {"other": patch.sel(distance=0.0)} | kwargs
        with pytest.raises(ParameterError, match=match):
            patch.correlate(**kwargs)

    def test_mismatched_step(self):
        """Patches with different steps along lag_dim are refused."""
        patch = self._patch(np.ones(10))
        other = dc.Patch(
            data=np.ones(5), coords={"time": np.arange(5) * 0.02}, dims=["time"]
        )
        with pytest.raises(ParameterError, match=r"step is 0\.01 but other's is 0\.02"):
            patch.correlate(other=other)

    def test_transformed_refused(self):
        """A patch already transformed along lag_dim is refused."""
        patch = self._patch(np.ones(10))
        with pytest.raises(ParameterError, match="untransformed"):
            patch.dft("time").correlate(other=patch, lag_dim="ft_time")

    def test_step_in_other_units(self):
        """A step stated in other units of time is compared as one."""
        rng = np.random.default_rng(3)
        patch = self._patch(rng.random(20)).set_units(time="s")
        other = self._patch(rng.random(10)).set_units(time="s")
        time_ms = other.get_array("time") * 1000
        other_ms = other.update_coords(time=time_ms).set_units(time="ms")
        expected = patch.correlate(other=other)
        assert patch.correlate(other=other_ms).equals(expected)

    def test_step_units_differ(self):
        """Equal magnitudes in different units are different steps."""
        patch = self._patch(np.ones(10)).set_units(time="s")
        with pytest.raises(ParameterError, match=r"0\.01 s but other's is 0\.01 ms"):
            patch.correlate(other=patch.set_units(time="ms"))

    def test_step_units_incompatible(self):
        """Lag coordinates whose units do not convert are refused."""
        patch = self._patch(np.ones(10)).set_units(time="s")
        with pytest.raises(UnitError, match="do not convert"):
            patch.correlate(other=patch.set_units(time="m"))
