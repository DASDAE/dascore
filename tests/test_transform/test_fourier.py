"""Tests for Fourier transforms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import next_fast_len

import dascore as dc
import dascore.proc.coords
from dascore.compat import random_state
from dascore.exceptions import ParameterError, PatchError
from dascore.transform.fourier import dft, idft
from dascore.units import get_quantity, get_quantity_str, second
from dascore.utils.misc import iterate

F_0 = 2
seconds = get_quantity("seconds")


@pytest.fixture(scope="session")
def sin_patch():
    """Get the sine wave patch, set units for testing."""
    patch = (
        dc.get_example_patch("sin_wav", sample_rate=100, duration=3, frequency=F_0)
        .set_units(get_quantity("1.0 V"), time="s", distance="m")
        .update_attrs(data_type="strain_rate")
    )
    return patch


@pytest.fixture(scope="session")
def sin_patch_trimmed(sin_patch):
    """Get the sine wave patch trimmed to a non-fast len along time dim."""
    return sin_patch.select(time=(0, -2), samples=True)


@pytest.fixture(scope="session")
def fft_sin_patch_time(sin_patch):
    """Get the sine wave patch, set units for testing."""
    return dft(sin_patch, dim="time")


@pytest.fixture(scope="session")
def fft_sin_patch_all(sin_patch):
    """Get the sine wave patch, set units for testing."""
    return dft(sin_patch, dim=None)


@pytest.fixture(scope="session")
def ifft_sin_patch_time(fft_sin_patch_time):
    """Get the sine wave patch, set units for testing."""
    return idft(fft_sin_patch_time, dim="ft_time")


@pytest.fixture(scope="session")
def ifft_sin_patch_all(fft_sin_patch_all):
    """Get the sine wave patch, set units for testing."""
    return idft(fft_sin_patch_all, dim=None)


@pytest.fixture(scope="session")
def chirp_patch():
    """Get a patch with a linear chirp."""
    patch = dc.examples.chirp(channel_count=2)
    return patch


@pytest.fixture(scope="session")
def chirp_stft_patch(chirp_patch):
    """Perform sensible stft on patch."""
    out = chirp_patch.stft(time=0.5 * seconds)
    return out.update_attrs(history=[])


@pytest.fixture(scope="session")
def chirp_stft_detrend_patch(chirp_patch):
    """Perform stft with detrend on chirp patch."""
    return chirp_patch.stft(time=100, overlap=10, samples=True, detrend=True)


class TestDiscreteFourierTransform:
    """Forward DFT suite."""

    def _assert_spectral_power_matches_patch(self, patch, out, dims, output):
        """Ensure spectral outputs recover the time-domain mean square."""
        dims = patch.dims if dims is None else tuple(iterate(dims))
        old_axes = tuple(patch.get_axis(dim) for dim in dims)
        ft_axes = tuple(out.get_axis(f"ft_{dim}") for dim in dims)
        expected = np.mean(patch.data**2, axis=old_axes)
        spectral_power = np.sum(out.data, axis=ft_axes)
        if output == "PSD":
            bin_volume = np.prod([abs(out.get_coord(f"ft_{dim}").step) for dim in dims])
            spectral_power = spectral_power * bin_volume

        assert np.allclose(spectral_power, expected)

    def _assert_real_output_matches_full_dft(self, patch, out, dims, output, real):
        """Ensure real spectral outputs are not converted to one-sided spectra."""
        dims = patch.dims if dims is None else tuple(iterate(dims))
        real = dims[-1] if real is True else real
        full = patch.dft(dim=dims, real=False, pad=False, output=output)
        real_axis = full.get_axis(f"ft_{real}")
        nonnegative = full.get_coord(f"ft_{real}").data >= 0
        full_ft_axes = tuple(full.get_axis(f"ft_{dim}") for dim in dims)
        out_ft_axes = tuple(out.get_axis(f"ft_{dim}") for dim in dims)
        expected = np.sum(
            np.compress(nonnegative, full.data, axis=real_axis),
            axis=full_ft_axes,
        )
        spectral_power = np.sum(out.data, axis=out_ft_axes)

        assert np.allclose(spectral_power, expected)

    def test_max_frequency(self, fft_sin_patch_time):
        """Ensure when sin wave is input max freq is correct."""
        assert "ft_time" in fft_sin_patch_time.dims
        patch = fft_sin_patch_time
        freq_dim = patch.get_axis("ft_time")
        ar = np.argmax(np.abs(patch.data), freq_dim)
        assert np.allclose(ar, ar[0])
        freqs = patch.get_coord("ft_time").data
        max_freq = np.abs(freqs[ar[0]])
        assert np.isclose(max_freq, F_0, rtol=0.01)

    def test_units(self, fft_sin_patch_time, sin_patch):
        """Ensure units were transformed as expected."""
        time_units = get_quantity(fft_sin_patch_time.get_coord("time").units)
        ft_time_units = get_quantity(fft_sin_patch_time.get_coord("ft_time").units)
        assert 1 / time_units == ft_time_units
        old_data_units = get_quantity(sin_patch.attrs.data_units)
        new_data_units = get_quantity(fft_sin_patch_time.attrs.data_units)
        assert old_data_units * time_units == new_data_units

    def test_drop_non_dimensional_coordinates(self, random_patch_many_coords):
        """
        Non-dimensional coordinates associated with transformed axis should
        be dropped, but those associated with non-transformed axis should remain.

        A dropped one is parked under a private name for idft to restore;
        it is not a coordinate of the transformed patch either way.
        """
        patch = random_patch_many_coords
        # every coord associated with time should be dropped in output.
        # but time itself should be kept as non-dimensional coord.
        coord_to_drop = set(patch.coords.dim_to_coord_map["time"]) - {"time"}
        coords_to_keep = set(patch.coords.coord_map) - coord_to_drop
        # do dft
        out = dft(patch, "time")
        # ensure kept coords are kept and dropped are dropped.
        new_coords = set(out.coords.coord_map)
        assert coord_to_drop.isdisjoint(new_coords)
        assert coords_to_keep.issubset(new_coords)
        # make sure time has no dimensions
        assert out.coords.dim_map["time"] == ()

    def test_real_fft(self, sin_patch):
        """Ensure real fft works."""
        out = sin_patch.dft("time", real=True)
        coord = out.get_coord("ft_time")
        freq_ax = out.get_axis("ft_time")
        assert coord.min() == 0
        ar = np.argmax(np.abs(out.data), axis=freq_ax)
        assert np.allclose(ar, ar[0])
        max_freq = np.abs(coord.data[ar[0]])
        assert np.isclose(max_freq, F_0, rtol=0.01)
        # data shape should be less than before (since real fft)
        ft_shape = out.coord_shapes["ft_time"][0]
        time_shape = sin_patch.coord_shapes["time"][0]
        assert ft_shape == time_shape // 2 or ft_shape == (time_shape // 2 + 1)

    def test_all_dims(self, fft_sin_patch_all):
        """Ensure fft can be done on all axis."""
        patch = fft_sin_patch_all
        assert all(x.startswith("ft_") for x in patch.dims)

    def test_real_multiple_dims(self, sin_patch):
        """Ensure the real axis can be specified."""
        patch = sin_patch
        out = patch.dft(dim=("distance", "time"), real="distance")
        assert all(x.startswith("ft_") for x in out.dims)
        real_coord = out.get_coord("ft_distance")
        assert real_coord.min() == 0

    def test_parseval(self, sin_patch, fft_sin_patch_time):
        """
        Ensure parseval's theorem holds. This means we have scaled the
        transforms correctly.
        """
        pa1, pa2 = sin_patch, fft_sin_patch_time
        vals1 = (pa1**2).integrate("time", definite=True)
        vals2 = (pa2.abs() ** 2).integrate("ft_time", definite=True)
        assert np.allclose(vals1.data, vals2.data)

    def test_idempotent_single_dim(self, fft_sin_patch_time):
        """
        Ensure dft is idempotent for a single dimension.
        """
        out = fft_sin_patch_time.dft("time")
        assert out.equals(fft_sin_patch_time)

    def test_idempotent_does_not_convert_output(self, fft_sin_patch_time):
        """Ensure output is ignored when no new dimensions are transformed."""
        out = fft_sin_patch_time.dft("time", output="AS")

        assert out.equals(fft_sin_patch_time)
        assert out.attrs["_dft_output"] == "FFT"

    def test_idempotent_all_dims(self, fft_sin_patch_all):
        """
        Ensure dft is idempotent for transforms applied to all dims.
        """
        out = fft_sin_patch_all.dft(dim=("time", "distance"))
        assert out.equals(fft_sin_patch_all)

    def test_transform_single_dim(
        self, sin_patch, fft_sin_patch_time, fft_sin_patch_all
    ):
        """
        Ensure dft is idempotent for time, but untransformed axis still gets
        transformed.
        """
        out = fft_sin_patch_time.dft(dim=("time", "distance"))
        assert not out.equals(fft_sin_patch_time)
        assert np.allclose(out.data, fft_sin_patch_all.data)

    def test_datatype_changed(self, fft_sin_patch_time, sin_patch):
        """Ensure the data_type attr is changed after transform."""
        assert sin_patch.attrs.data_type == "strain_rate"
        assert fft_sin_patch_time.attrs.data_type == "fourier_transform"

    def test_dft_output_attr_set(self, fft_sin_patch_time):
        """Ensure the DFT output type is tracked."""
        assert fft_sin_patch_time.attrs["_dft_output"] == "FFT"

    def test_pad(self, sin_patch_trimmed):
        """Ensure patch is padded when requested and not otherwise."""
        trimmed = sin_patch_trimmed
        old_time_len = trimmed.coord_shapes["time"][0]
        dft_pad = trimmed.dft("time")
        dft_no_pad = trimmed.dft("time", pad=False)
        assert dft_pad.shape != dft_no_pad.shape
        assert dft_pad.coord_shapes["ft_time"][0] == next_fast_len(old_time_len)
        assert dft_no_pad.coord_shapes["ft_time"] == trimmed.coord_shapes["time"]

    def test_display(self, fft_sin_patch_time):
        """Ensure a transformed patch returns a str rep."""
        out = str(fft_sin_patch_time)
        assert isinstance(out, str)
        assert out

    @pytest.mark.parametrize(
        ("output", "data_type"),
        [
            ("FFT", "fourier_transform"),
            ("AS", "amplitude_spectrum"),
            ("PS", "power_spectrum"),
            ("PSD", "power_spectral_density"),
        ],
    )
    def test_output_spectral_representations(self, sin_patch, output, data_type):
        """Ensure supported spectral outputs are returned."""
        fft = sin_patch.dft("time", output="FFT")
        out = sin_patch.dft("time", output=output)

        assert out.attrs.data_type == data_type
        assert out.attrs["_dft_output"] == output
        assert out.data.shape == fft.data.shape

    @pytest.mark.parametrize("real", [False, True])
    def test_amplitude_spectrum_scaling(self, sin_patch, real):
        """Ensure AS recovers harmonic amplitudes."""
        out = sin_patch.dft("time", real=real, pad=False, output="AS")
        time_axis = sin_patch.get_axis("time")
        ft_axis = out.get_axis("ft_time")
        sine_amp = np.ptp(sin_patch.data, axis=time_axis) / 2
        expected = sine_amp / 2

        assert np.allclose(np.max(out.data, axis=ft_axis), expected, rtol=0.005)

    @pytest.mark.parametrize("output", ["PS", "PSD"])
    def test_spectral_power_scaling(self, sin_patch, output):
        """Ensure PS and PSD integrate/sum to time-domain mean square."""
        dims = ("time",)
        out = sin_patch.dft("time", pad=False, output=output)

        self._assert_spectral_power_matches_patch(sin_patch, out, dims, output)

    @pytest.mark.parametrize(
        "dims",
        [
            None,
            ("time", "distance"),
        ],
    )
    @pytest.mark.parametrize("output", ["PS", "PSD"])
    def test_spectral_power_scaling_multiple_dims(self, sin_patch, dims, output):
        """Ensure multi-axis spectral outputs recover mean square."""
        out = sin_patch.dft(dim=dims, pad=False, output=output)

        self._assert_spectral_power_matches_patch(sin_patch, out, dims, output)

    @pytest.mark.parametrize(
        ("dims", "real"),
        [
            (("time",), True),
            (("time", "distance"), "time"),
            (("distance", "time"), "distance"),
        ],
    )
    @pytest.mark.parametrize("output", ["PS", "PSD"])
    def test_real_spectral_outputs_match_full_dft_bins(
        self, sin_patch, dims, real, output
    ):
        """Ensure real spectral outputs keep raw nonnegative-frequency bins."""
        out = sin_patch.dft(dim=dims, real=real, pad=False, output=output)

        self._assert_real_output_matches_full_dft(sin_patch, out, dims, output, real)

    @pytest.mark.parametrize("output", ["PS", "PSD"])
    def test_spectral_power_scaling_padded(self, sin_patch_trimmed, output):
        """Ensure default padded spectral outputs use the padded extent."""
        padded = sin_patch_trimmed.pad(time="fft")
        out = sin_patch_trimmed.dft("time", output=output)

        self._assert_spectral_power_matches_patch(padded, out, ("time",), output)

    def test_spectral_output_units(self, sin_patch):
        """Ensure spectral output units are scaled according to output type."""
        data_units = get_quantity(sin_patch.attrs.data_units)
        time_units = get_quantity(sin_patch.get_coord("time").units)

        as_ = sin_patch.dft("time", pad=False, output="AS")
        ps = sin_patch.dft("time", pad=False, output="PS")
        psd = sin_patch.dft("time", pad=False, output="PSD")

        assert get_quantity(as_.attrs.data_units) == data_units
        assert get_quantity(ps.attrs.data_units) == data_units * data_units
        expected_psd_units = data_units * data_units * time_units
        assert get_quantity(psd.attrs.data_units) == expected_psd_units

    def test_time_dft_coord_uses_hz(self, sin_patch):
        """Ensure transformed time coords use Hz rather than reduced 1/s."""
        out = sin_patch.dft("time", pad=False)

        assert get_quantity(out.get_coord("ft_time").units) == get_quantity("Hz")
        assert get_quantity_str(out.get_coord("ft_time").units) == "Hz"

    def test_strain_rate_psd_units_use_hz(self, sin_patch):
        """Ensure strain-rate PSD units use conventional per-Hz units."""
        patch = sin_patch.set_units("strain/s")

        out = patch.dft("time", real=True, pad=False, output="PSD")

        assert get_quantity(out.attrs.data_units) == get_quantity("(strain/s)**2/Hz")
        assert "Hz" in get_quantity_str(out.attrs.data_units)

    @pytest.mark.parametrize("output", ["AS", "PS", "PSD"])
    def test_spectral_output_without_data_units(self, sin_patch, output):
        """Ensure spectral outputs don't invent units without data units."""
        patch = sin_patch.update_attrs(data_units=None)

        out = patch.dft("time", pad=False, output=output)

        assert out.attrs.data_units is None

    def test_output_is_case_insensitive(self, sin_patch):
        """Ensure output names are normalized before conversion."""
        lower = sin_patch.dft("time", output="as")
        upper = sin_patch.dft("time", output="AS")
        assert lower.equals(upper)

    def test_invalid_output_raises(self, sin_patch):
        """Ensure invalid output types raise."""
        with pytest.raises(ValueError, match="Unknown output"):
            sin_patch.dft("time", output="bad")

    def test_db_true_with_fft_raises(self, sin_patch):
        """Ensure dB conversion is only accepted for spectral outputs."""
        with pytest.raises(ParameterError, match="db=True is only supported"):
            sin_patch.dft("time", output="FFT", db=True)

    @pytest.mark.parametrize(("output", "scale"), [("AS", 20), ("PS", 10), ("PSD", 10)])
    def test_db_output(self, sin_patch, output, scale):
        """Ensure spectral outputs use DASCore's no-reference dB scaling."""
        linear = sin_patch.dft("time", output=output)
        out = sin_patch.dft("time", output=output, db=True)
        eps = np.finfo(linear.data.dtype).eps
        expected = scale * np.log10(linear.data + eps)

        assert get_quantity(out.attrs.data_units) == get_quantity("dB")
        assert out.attrs.data_type == linear.attrs.data_type
        assert np.allclose(out.data, expected)


class TestInverseDiscreteFourierTransform:
    """Inverse DFT suite."""

    def _patches_about_equal(self, patch1, patch2):
        """Ensure patches are about equal in coord manager and data."""
        assert patch1.data.shape == patch2.data.shape
        assert np.allclose(patch1.data, patch2.data)
        cm1 = patch1.coords.drop_disassociated_coords()
        cm2 = patch2.coords.drop_disassociated_coords()
        assert cm1 == cm2

    def test_invertible_1d(self, sin_patch, ifft_sin_patch_time):
        """Ensure pre dft and idft(dft(patch)) are equal."""
        patch1 = sin_patch
        patch2 = ifft_sin_patch_time.real()
        self._patches_about_equal(patch1, patch2)

    def test_invertible_2d(self, sin_patch, ifft_sin_patch_all):
        """Ensure 2d patches are invertible."""
        patch1 = sin_patch
        patch2 = ifft_sin_patch_all.real()
        self._patches_about_equal(patch1, patch2)

    def test_undo_real_dft(self, sin_patch):
        """Ensure real dft is properly handled."""
        pa1 = sin_patch.dft(dim="time", real=True)
        pa2 = pa1.idft().real()
        self._patches_about_equal(sin_patch, pa2)

    def test_raises_on_untransformed_patch(self, sin_patch):
        """Only patches which have been first transformed can be idft'ed."""
        with pytest.raises(NotImplementedError):
            sin_patch.idft("time")

    def test_partial_inverse(self, fft_sin_patch_all, sin_patch):
        """Ensure inverse works on only a single axis."""
        # since we only reverse time it should be the same as forward distance.
        ift = fft_sin_patch_all.idft("time")
        dft = sin_patch.dft("distance")
        self._patches_about_equal(ift, dft)
        # and then if we reverse distance it should be the same as original
        full_inverse = ift.idft("distance")
        self._patches_about_equal(full_inverse, sin_patch)

    def test_data_type_restored(self, fft_sin_patch_time, sin_patch):
        """Ensure data_type attr is restored."""
        out = fft_sin_patch_time.idft("time")
        assert out.attrs.data_type == sin_patch.attrs.data_type

    def test_dft_output_attr_removed_after_idft(self, fft_sin_patch_time):
        """Ensure inverse DFT removes private DFT output metadata."""
        out = fft_sin_patch_time.idft("time")

        assert "_dft_output" not in out.attrs

    @pytest.mark.parametrize("output", ["AS", "PS", "PSD"])
    def test_non_fft_output_cannot_be_inverted(self, sin_patch, output):
        """Ensure non-FFT spectral representations cannot be inverted."""
        out = sin_patch.dft("time", output=output)

        with pytest.raises(ValueError, match="Only dft\\(output='FFT'\\)"):
            out.idft("time")

    def test_undo_padding(self, sin_patch_trimmed):
        """Ensure the padding is undone in idft."""
        dft_patch = sin_patch_trimmed.dft("time")
        idft = dft_patch.idft()
        assert idft.shape == sin_patch_trimmed.shape
        assert np.allclose(np.real(idft.data), sin_patch_trimmed.data)

    def test_undo_padding_rft(self, sin_patch_trimmed):
        """Ensure padded rft still works."""
        dft_patch = sin_patch_trimmed.dft("time", real=True)
        idft = dft_patch.idft()
        assert idft.shape == sin_patch_trimmed.shape
        assert np.allclose(np.real(idft.data), sin_patch_trimmed.data)

    def test_two_coords_wanting_one_parking_space(self):
        """A pair of names which park in the same place is refused."""
        patch = dc.Patch(
            data=random_state.random((4, 5)),
            coords={"a": np.arange(4), "a_associated_b": np.arange(5)},
            dims=("a", "a_associated_b"),
        )
        patch = patch.update_coords(
            b_associated_c=("a", np.arange(4) * 1.0),
            c=("a_associated_b", np.arange(5) * 1.0),
        )
        with pytest.raises(PatchError, match="is where it would go"):
            patch.dft(("a", "a_associated_b"))

    def test_associated_coords_restored(self, random_patch_many_coords):
        """Coordinates on a transformed dim come back with it. See #1041."""
        patch = random_patch_many_coords
        out = patch.dft(dim=None).idft()
        # time2 rides time and lat rides distance; quality spans both, which
        # no single name can park, so it is dropped as it always was.
        for name in ("time2", "lat"):
            assert out.coords.dim_map[name] == patch.coords.dim_map[name]
            assert np.allclose(out.get_array(name), patch.get_array(name))
        assert "quality" not in out.coords.coord_map

    def test_associated_coords_wait_for_their_dim(self, random_patch_many_coords):
        """A dim still in the frequency domain keeps its coords parked."""
        patch = random_patch_many_coords
        out = patch.dft(dim=None).idft("ft_time")
        assert "time2" in out.coords.coord_map
        assert "lat" not in out.coords.coord_map
        assert "lat" in out.idft().coords.coord_map

    def test_padded_round_trip_restores_associated_coords(self, event_patch_1):
        """A transform which pads restores what padding said nothing about."""
        count = event_patch_1.coord_shapes["distance"][0]
        labels = np.array([f"c{x}" for x in range(count)])
        patch = event_patch_1.update_coords(
            label=("distance", labels), idx=("distance", np.arange(count))
        )
        out = patch.dft(("time", "distance")).idft().real()
        assert np.array_equal(out.get_array("label"), labels)
        # Padding widened the integers to hold the NaN it added, and the
        # trim which follows cannot narrow them again.
        assert np.allclose(out.get_array("idx"), np.arange(count))

    def test_real_transform_restores_associated_coords(self, sin_patch):
        """A real transform is a different length, but restores its coords."""
        depth = np.arange(len(sin_patch.get_coord("distance"))) * 2.0
        patch = sin_patch.update_coords(depth=("distance", depth))
        out = patch.dft(("time", "distance"), real="time").idft()
        assert np.allclose(out.get_array("depth"), depth)

    def test_no_extra_attrs_or_coords(self, sin_patch):
        """Ensure no extra attrs or coords remain after round trip."""
        dft = sin_patch.dft(dim=None)
        idft = dft.idft()
        old_attrs = set(dict(sin_patch.attrs).keys())
        new_attrs = set(dict(idft.attrs).keys())
        # Before, there were a lot of ft_* keys added from extra coords.
        diff = new_attrs - old_attrs
        assert not diff, "attr keys shouldn't change"
        # Test no extra coords
        assert set(sin_patch.coords.coord_map) == set(idft.coords.coord_map)


class TestSTFT:
    """Tests for the short-time Fourier transform."""

    def test_numeric_window_with_timedelta_coord(self):
        """
        Stft with a numeric window length should work when the time
        coordinate is timedelta64 (not just datetime64); see #604.
        """
        patch = dc.get_example_patch()
        time = patch.get_coord("time")
        # Convert the time coordinate to timedelta64 (relative to the end).
        new = patch.update_coords(time=time.values - time.values[-1])
        # A numeric window length previously raised a ufunc type error here.
        out_numeric = new.stft(time=0.1)
        out_timedelta = new.stft(time=dc.to_timedelta64(0.1))
        # The numeric form must match the explicit-duration form exactly.
        assert out_numeric.equals(out_timedelta)

    def test_type(self, chirp_stft_patch, chirp_patch):
        """Simply ensure the correct type was returned."""
        patch = chirp_stft_patch
        assert isinstance(patch, dc.Patch)
        assert len(patch.dims) == (len(chirp_patch.dims) + 1)

    def test_coord_units(self, chirp_stft_patch):
        """Ensure the units on the new coord are correct."""
        second = dc.get_quantity("second")
        hz = dc.get_quantity("Hz")
        freq_coord = chirp_stft_patch.get_coord("ft_time")
        time_coord = chirp_stft_patch.get_coord("time")
        assert dc.get_quantity(time_coord.units) == second
        assert dc.get_quantity(freq_coord.units) == hz

    def test_array_window(self, random_patch):
        """Ensure an array can be used as a window function."""
        win = np.ones(100)
        out = random_patch.stft(time=100, taper_window=win, overlap=10, samples=True)
        assert len(out.dims) == (len(random_patch.dims) + 1)

    def test_dft_equiv(self, random_patch):
        """
        Ensure using a boxcar window produces the same as the dft for an equal slice.
        """
        patch = random_patch.select(distance=1, samples=True)
        stft = (
            patch.stft(time=101, overlap=0, taper_window="boxcar", samples=True)
            .select(time=0, samples=True)
            .squeeze()
        )
        # With no overlap the first window is the first 101 samples; the stft
        # refers phase to the window's centre, so magnitudes are compared.
        first = patch.select(time=(0, 101), samples=True)
        first_fft = first.dft("time", real=True, pad=False).squeeze()
        ar1 = stft.data
        ar2 = first_fft.data

        factor = np.abs(ar1) / np.abs(ar2)
        assert np.allclose(factor, 1.0)

    def test_data_units(self, random_patch):
        """Ensure data units match those of dft."""
        patch = random_patch.update_attrs(data_units="m")
        pa1 = patch.dft("time", real=True)
        pa2 = patch.stft(time=1)
        assert pa1.attrs.data_units == pa2.attrs.data_units
        ipa1 = pa1.idft()
        ipa2 = pa2.istft()
        assert ipa1.attrs.data_units == ipa2.attrs.data_units

    def test_none_for_overlap(self, random_patch):
        """Using None for overlap should be supported."""
        out = random_patch.stft(time=1, overlap=None)
        assert isinstance(out, dc.Patch)

    def test_nfft_adds_bins(self, random_patch):
        """A longer FFT samples the spectrum at more, closer frequencies."""
        plain = random_patch.stft(time=100, samples=True)
        padded = random_patch.stft(time=100, samples=True, nfft=256)
        assert len(padded.get_coord("ft_time")) == 256 // 2 + 1
        rate = 1 / dc.to_float(random_patch.get_coord("time").step)
        assert float(padded.get_coord("ft_time").step) == pytest.approx(rate / 256)
        # The window and hop are untouched; only the FFT of each window grew.
        assert padded.attrs["_tile_stride_time"] == plain.attrs["_tile_stride_time"]
        assert padded.get_coord("time") == plain.get_coord("time")
        assert padded.attrs["_stft_mfft"] == 256

    def test_nfft_default_is_the_window(self, random_patch):
        """None means the window length, which is what stft always did."""
        plain = random_patch.stft(time=100, samples=True)
        explicit = random_patch.stft(time=100, samples=True, nfft=None)
        assert plain.attrs["_stft_mfft"] == 100
        assert explicit.equals(plain)

    def test_nfft_in_units(self, random_patch):
        """A quantity is read through the coordinate, whatever samples says."""
        by_count = random_patch.stft(time=100, samples=True, nfft=256)
        by_units = random_patch.stft(time=100, samples=True, nfft=1.024 * second)
        assert by_units.attrs["_stft_mfft"] == 256
        assert by_units.equals(by_count)

    def test_nfft_as_a_timedelta(self, random_patch):
        """A native duration is read through the coordinate like a quantity."""
        out = random_patch.stft(time=100, samples=True, nfft=np.timedelta64(1024, "ms"))
        assert out.attrs["_stft_mfft"] == 256

    def test_fractional_nfft_refused(self, random_patch):
        """256.9 points is not an FFT length; it is not rounded quietly."""
        with pytest.raises(ParameterError, match="whole number of samples"):
            random_patch.stft(time=100, samples=True, nfft=256.9)

    def test_nfft_below_window_refused(self, random_patch):
        """A shorter FFT would drop data, which is not a transform."""
        with pytest.raises(ParameterError, match="at least the window length"):
            random_patch.stft(time=100, samples=True, nfft=50)

    def test_nfft_is_interpolation(self, random_patch):
        """Padding adds bins between the old ones; the old ones are unchanged."""
        plain = random_patch.stft(time=100, samples=True)
        padded = random_patch.stft(time=100, samples=True, nfft=200)
        axis = padded.get_axis("ft_time")
        every_other = np.take(padded.data, np.arange(0, 101, 2), axis=axis)
        assert np.allclose(every_other, plain.data)

    def test_complex_input_round_trips(self, random_patch):
        """Complex data is transformed two-sided, centred, and comes back."""
        patch = random_patch.new(data=random_patch.data * (1 + 1j))
        out = patch.stft(time=16, samples=True)
        freqs = out.get_coord("ft_time").values
        assert len(freqs) == 16
        assert freqs[0] < 0 < freqs[-1]
        assert out.attrs["_stft_fft_mode"] == "centered"
        back = out.istft()
        assert back.equals(patch, close=True)

    def test_non_dim_coord_associated_with_transform(self):
        """See #611."""
        patch = dc.get_example_patch("random_das", shape=(10, 200))
        aux_time = np.arange(len(patch.get_coord("time")), dtype=float)
        aux_dist = np.arange(len(patch.get_coord("distance")), dtype=float)
        patch = patch.update_coords(
            aux_time=("time", aux_time),
            aux_dist=("distance", aux_dist),
        )
        out = patch.stft(time=0.1)
        assert "aux_time" not in out.coords.coord_map
        assert "aux_dist" in out.coords.coord_map


class TestInverseSTFT:
    """Tests for the inverse short-time Fourier transform."""

    @pytest.fixture(scope="session")
    def chirp_round_tripped(self, chirp_stft_patch):
        """Round trip patch through stft."""
        return chirp_stft_patch.istft()

    def test_near_round_trip_1(self, chirp_round_tripped, chirp_patch):
        """Test how well the patch round-tripped through the stft."""
        patch1, patch2 = chirp_round_tripped, chirp_patch
        assert patch1.ndim == patch2.ndim
        assert patch1.dims == patch2.dims
        assert patch1.shape == patch2.shape
        assert patch1.equals(chirp_patch, close=True)

    def test_round_trip_2(self):
        """Another round trip test from the doctests."""
        patch = dc.get_example_patch("chirp")
        # Simple stft with 10 second window and 4 seconds overlap
        pa1 = patch.stft(time=10 * second, overlap=4 * second)
        pa2 = pa1.istft()
        assert pa2.equals(patch, close=True)
        # Ensure stft attrs and coords were cleaned up
        assert not any(k.startswith("_stft") for k in dict(pa2.attrs))
        assert not any(k.startswith("_stft") for k in dict(pa2.coords.coord_map))

    def test_roundtrip_3(self, random_patch):
        """Simple round trip with near default params."""
        patch = random_patch
        stft = patch.stft(time=1)
        istft = stft.istft()
        assert patch.equals(istft, close=True)

    def test_round_trip_with_nfft(self, random_patch):
        """A padded FFT inverts to the same data; the padding is dropped."""
        pa1 = random_patch.stft(time=100, samples=True, nfft=256)
        pa2 = pa1.istft()
        assert pa2.equals(random_patch, close=True)
        assert not any(k.startswith("_stft") for k in dict(pa2.attrs))

    def test_non_transformed_raises(self, random_patch):
        """Test that a patch that hasn't undergone stft can't be used."""
        msg = "undergone stft"
        with pytest.raises(PatchError, match=msg):
            random_patch.istft()

    def test_uninvertible_window_raises(self, random_patch):
        """Abutting hann windows leave gaps: the stft is not invertible."""
        out = random_patch.stft(time=16, samples=True, overlap=None)
        with pytest.raises(ParameterError, match="cannot be inverted"):
            out.istft()

    def test_detrended_raise(self, chirp_stft_detrend_patch):
        """Since detrended stft can't be inverted it should raise."""
        msg = "Inverse stft not possible"
        with pytest.raises(PatchError, match=msg):
            chirp_stft_detrend_patch.istft()


class TestInverseSTFTAssociatedCoords:
    """Tests that istft round-trips patches with associated coordinates."""

    @pytest.fixture(scope="class")
    def patch_with_coords(self):
        """A patch with coords on each dimension and one on no dimension."""
        patch = dc.get_example_patch("random_das", shape=(10, 200))
        dist_len = len(patch.get_coord("distance"))
        time_len = len(patch.get_coord("time"))
        return patch.update_coords(
            depth=("distance", np.arange(dist_len, dtype=float) * 2.0),
            tlabel=("time", np.arange(time_len, dtype=float)),
            note=(None, np.array(["a", "b"])),
        )

    def test_coord_on_untransformed_dim(self, patch_with_coords):
        """A coord on an untransformed dim survives the round trip. See #1039."""
        patch = patch_with_coords.drop_coords("tlabel", "note")
        out = patch.stft(time=0.1).istft()
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert np.allclose(out.data, patch.data)

    def test_coord_on_transformed_dim(self, patch_with_coords):
        """A coord on the transformed dim rides with the windows and comes back."""
        patch = patch_with_coords.drop_coords("depth", "note")
        stft_patch = patch.stft(time=0.1)
        assert "tlabel" not in stft_patch.coords.coord_map
        assert "_tile_source_time__tlabel" in stft_patch.coords.coord_map
        out = stft_patch.istft()
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert np.allclose(out.data, patch.data)

    def test_multiple_associated_coords(self, patch_with_coords):
        """Several associated coords round trip, the transformed one included."""
        patch = patch_with_coords
        out = patch.stft(time=0.1).istft()
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert np.allclose(out.data, patch.data)

    def test_coord_on_untransformed_time(self, patch_with_coords):
        """Transforming distance leaves the time-associated coord intact."""
        patch = patch_with_coords
        out = patch.stft(distance=4).istft()
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert np.allclose(out.data, patch.data)

    def test_coords_on_stft_dims_dropped(self, patch_with_coords):
        """Coords on the frequency or window dims cannot survive the inverse."""
        stft_patch = patch_with_coords.drop_coords("tlabel", "note").stft(time=0.1)
        freq_len = len(stft_patch.get_coord("ft_time"))
        win_len = len(stft_patch.get_coord("time"))
        marked = stft_patch.update_coords(
            snr=("ft_time", np.arange(freq_len, dtype=float)),
            wlabel=("time", np.arange(win_len, dtype=float)),
        )
        out = marked.istft()
        assert {"snr", "wlabel"}.isdisjoint(out.coords.coord_map)
        assert out.coords == stft_patch.istft().coords
