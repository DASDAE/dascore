"""Tests for the Uptech HDF5 reader."""

from __future__ import annotations

import h5py
import numpy as np

from dascore.io.uptech import UptechH5V1


def test_uptech_read_and_scan(tmp_path):
    """Read a minimal Uptech export and select distance."""
    path = tmp_path / "sample.hdf5"
    with h5py.File(path, "w") as h5:
        group = h5.create_group("Acquisition")
        data = group.create_dataset("StrainRate", data=np.arange(12.0).reshape(4, 3))
        data.attrs.update(
            acquisition_frequency=2.0,
            fiber_length=12800.0,
            gauge_length=10.0,
            sampling_interval=2.5,
            spatial_resolution=2.0,
        )
        group.create_dataset(
            "Time", data=np.array([1e9, 1e9 + 0.5, 1e9 + 1, 1e9 + 1.5])
        )
    io = UptechH5V1()
    assert io.get_format(path) == ("Uptech_H5", "1")
    patch = io.read(path, distance=(2.5, None))[0]
    assert patch.data.shape == (4, 2)
    assert patch.attrs.data_type == "strain_rate"
    assert io.scan(path)[0].file_format == "Uptech_H5"


def test_uptech_not_detected_without_metadata(tmp_path):
    """Reject a file without the identifying metadata."""
    path = tmp_path / "other.hdf5"
    with h5py.File(path, "w") as h5:
        h5.create_dataset("Acquisition/StrainRate", shape=(2, 2))
        h5.create_dataset("Acquisition/Time", shape=(2,))
    assert not UptechH5V1().get_format(path)
