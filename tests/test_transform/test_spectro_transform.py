"""Tests for the spectrogram transformation."""

from __future__ import annotations

import pytest

import dascore as dc
from dascore.transform.spectro import spectrogram


class TestSpectroTransform:
    """Tests for transforming regular patches into spectrograms."""

    @pytest.fixture()
    def spec_patch(self, random_patch):
        """Simple patch trasnformed to spectrogram."""
        patch = random_patch.set_units("m/s")
        return patch.spectrogram("time")

    def test_units(self, random_patch):
        """Ensure units were properly converted."""
        patch = random_patch.set_units("m/s")
        spec_patch = patch.spectrogram("time")
        # first check coord units
        coord1 = patch.get_coord("time")
        coord2 = spec_patch.get_coord("ft_time")
        units1 = dc.get_quantity(coord1.units)
        units2 = dc.get_quantity(coord2.units)
        assert units1 == 1 / units2
        # then check data units
        data_units1 = dc.get_quantity(patch.attrs.data_units)
        data_units2 = dc.get_quantity(spec_patch.attrs.data_units)
        assert data_units1 * units1 == data_units2

    def test_spec_patch_dimensions(self, spec_patch, random_patch):
        """Ensure expected dimensions now exist."""
        dims = spec_patch.dims
        # dims should have been added
        assert len(dims) > len(random_patch.dims)
        assert set(dims) == (set(random_patch.dims) | {"ft_time"})

    def test_transformed_coord(self, spec_patch, random_patch):
        """
        The start values of transformed dimension should be comparable and the
        units unchanged.
        """
        time_1 = random_patch.get_coord("time")
        time_2 = spec_patch.get_coord("time")
        assert time_1.units == time_2.units

    def test_time_first(self, random_patch):
        """Ensure the spectrogram still works when time dim is first."""
        transposed = random_patch.transpose(*("time", "distance"))
        out = spectrogram(transposed, dim="time")
        assert isinstance(out, dc.Patch)

    def test_transpose(self, random_patch):
        """Ensure when patch dim order is different it still works."""
        out = random_patch.transpose().spectrogram("time")
        assert set(out.dims) == (set(random_patch.dims) | {"ft_time"})

    def test_distance_dim(self, random_patch):
        """Ensure distance dimension works."""
        out = random_patch.transpose().spectrogram("distance")
        assert set(out.dims) == (set(random_patch.dims) | {"ft_distance"})

    def test_time_range_unchanged(self, dispersion_patch):
        """Ensure time axis isn't outside original bounds, see #286."""
        spectro = dispersion_patch.spectrogram(dim="time")
        # the new time on the spectrogram should be contained in the original
        original_time = dispersion_patch.get_coord("time")
        new_time = spectro.get_coord("time")
        assert new_time.min() >= original_time.min()
        assert new_time.max() <= original_time.max()
