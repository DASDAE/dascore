"""The distance coordinate a Silixa V1 (Acoustic) file states."""

import h5py
import numpy as np
import pytest

from dascore.io.silixah5 import SilixaH5V1
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def acoustic_path():
    """Path to the Acoustic-dataset (V1) test file."""
    return fetch("silixa_h5_1.hdf5")


@pytest.fixture(scope="module")
def header(acoustic_path):
    """The distance-related header attrs, straight from the file."""
    with h5py.File(acoustic_path, "r") as fi:
        attrs = fi["Acoustic"].attrs
        return {
            "start": float(attrs["Start Distance (m)"]),
            "stop": float(attrs["Stop Distance (m)"]),
            "step": float(attrs["SpatialResolution[m]"])
            * float(attrs["Fibre Length Multiplier"]),
        }


class TestSilixaV1Distance:
    """Distance runs from the header's Start Distance to its Stop Distance."""

    @pytest.fixture(scope="class")
    def distance(self, acoustic_path):
        """The distance coordinate of the read patch."""
        return SilixaH5V1().read(acoustic_path)[0].get_coord("distance")

    def test_first_channel_at_start_distance(self, distance, header):
        """Start Distance is the first channel, not one cable length past it."""
        assert np.isclose(distance.min(), header["start"])

    def test_last_channel_at_stop_distance(self, distance, header):
        """Start + (n - 1) * step lands on the header's Stop Distance.

        The header rounds the fiber length multiplier to six digits, which
        moves the far end by a few centimetres over eight kilometres.
        """
        assert np.isclose(distance.max(), header["stop"], atol=0.1)

    def test_step(self, distance, header):
        """The step is the spatial resolution scaled by the fiber multiplier."""
        assert np.isclose(distance.step, header["step"])
