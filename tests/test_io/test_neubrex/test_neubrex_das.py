"""Tests for the Neubrex DAS format."""

import shutil
from fractions import Fraction

import h5py
import numpy as np

from dascore.io.neubrex import NeubrexDASV1
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
