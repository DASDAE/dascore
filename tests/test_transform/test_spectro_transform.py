"""
Tests for the spectrogram transformation.
"""
import pytest

import dascore
from dascore.transform.spectro import spectrogram


class TestSpectroTransform:
    """Tests for transforming regular patches into spectrograms."""

    @pytest.fixture()
    def spec_patch(self, random_patch):
        """simple patch trasnformed to spectrogram."""
        return random_patch.tran.spectrogram()

    def test_spec_patch_dimensions(self, spec_patch, random_patch):
        """Ensure expected dimensions now exist."""
        dims = spec_patch.dims
        # dims should have been added
        assert len(dims) > len(random_patch.dims)
        assert set(dims) == (set(random_patch.dims) | {"frequency"})

    def test_time_first(self, random_patch):
        """Ensure the spectrogram still works when time dim is first."""
        transposed = random_patch.transpose(*("time", "distance"))
        out = spectrogram(transposed, dim="time")
        assert isinstance(out, dascore.Patch)
