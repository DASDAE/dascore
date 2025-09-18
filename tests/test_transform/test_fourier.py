"""Tests for Fourier transforms."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import next_fast_len

import dascore as dc
import dascore.proc.coords
from dascore.exceptions import PatchError
from dascore.transform.fourier import dft, idft
from dascore.units import get_quantity, second

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
        # make sure time has no dimsensions
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

    def test_datatype_removed(self, fft_sin_patch_time, sin_patch):
        """Ensure the data_type attr is removed after transform."""
        assert sin_patch.attrs.data_type == "strain_rate"
        assert fft_sin_patch_time.attrs.data_type == ""

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
        # The first slice should have 50 padded 0s on the right, and the first
        # 51 samples in the signal.
        padded = patch.pad(time=(50, 0), samples=True).select(
            time=(0, 101), samples=True
        )
        padded_fft = padded.dft("time", real=True, pad=False).squeeze()
        ar1 = stft.data
        ar2 = padded_fft.data

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

    def test_non_transformed_raises(self, random_patch):
        """Test that a patch that hasn't undergone stft can't be used."""
        msg = "undergone stft"
        with pytest.raises(PatchError, match=msg):
            random_patch.istft()

    def test_detrended_raise(self, chirp_stft_detrend_patch):
        """Since detrended stft can't be inverted it should raise."""
        msg = "Inverse stft not possible"
        with pytest.raises(PatchError, match=msg):
            chirp_stft_detrend_patch.istft()
