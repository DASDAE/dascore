"""Tests for the Neubrex DAS format."""

import shutil
from fractions import Fraction

import h5py
import numpy as np
import pytest

from dascore.io.neubrex import NeubrexDASV1
from dascore.io.neubrex.core import NeubrexDASPatchAttrs
from dascore.utils.downloader import fetch


class TestNeubrexDAS:
    """Format-specific tests not covered by the common IO tests."""

    def test_exact_time_step(self, tmp_path):
        """A sampling interval of 1/1024 s gives an exact step."""
        path = tmp_path / "hz_1024.h5"
        shutil.copy(fetch("neubrex_das_1.h5"), path)
        with h5py.File(path, "a") as h5:
            attrs = h5["Acoustic"].attrs
            attrs["TimeSamplingInterval(seconds)"] = np.float32(1 / 1024)
        fiber_io = NeubrexDASV1()
        for coord in (
            fiber_io.read(path)[0].get_coord("time"),
            fiber_io.scan(path)[0].get_coord("time"),
        ):
            assert coord.step_exact == Fraction(1, 1024)

    def test_one_index_of_refraction(self, tmp_path):
        """The file's index of refraction is the one attr naming it."""
        path = tmp_path / "refraction.h5"
        shutil.copy(fetch("neubrex_das_1.h5"), path)
        with h5py.File(path, "a") as h5:
            h5["Acoustic"].attrs["IndexOfRefraction"] = np.float32(1.47)
        attrs = dict(NeubrexDASV1().read(path)[0].attrs)
        assert "index_of_reflection" not in attrs
        assert attrs["index_of_refraction"] == pytest.approx(1.47)

    def test_old_name_accepted(self):
        """The name the field had before is taken as it."""
        attrs = NeubrexDASPatchAttrs(index_of_reflection=1.5)
        assert attrs.index_of_refraction == 1.5
        assert "index_of_reflection" not in dict(attrs)
