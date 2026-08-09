"""Tests for the Silixa H5 Carina (netCDF-shell) variant."""

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.netcdf.core import NetCDFCFV18
from dascore.io.silixah5 import SilixaH5V1, SilixaH5V2
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def carina_path():
    """Return the path to the Carina-variant test file (INGV Mt Etna)."""
    return fetch("silixa_h5_ingv_1.h5")


@pytest.fixture(scope="module")
def carina_patch(carina_path):
    """Read the Carina test file into a patch."""
    return SilixaH5V2().read(carina_path)[0]


class TestSilixaCarina:
    """Tests for the Carina variant."""

    def test_get_format(self, carina_path):
        """The file resolves to Silixa_H5 v2 through the full dispatch."""
        assert dc.get_format(carina_path) == ("Silixa_H5", "2")

    def test_v1_does_not_claim(self, carina_path):
        """The Acoustic-dataset reader must not claim a Carina file."""
        assert not SilixaH5V1().get_format(carina_path)

    def test_v2_does_not_claim_v1_file(self):
        """The Carina reader must not claim a V1 (Acoustic) file."""
        path = fetch("silixa_h5_1.hdf5")
        assert not SilixaH5V2().get_format(path)

    def test_time_coord(self, carina_patch):
        """Time comes from the root StartTime (µs epoch) and Samplerate."""
        time = carina_patch.get_coord("time")
        assert time.min() == dc.to_datetime64("2023-02-01T00:21:52.693941")
        assert time.step == dc.to_timedelta64(0.01)

    def test_distance_coord(self, carina_patch):
        """Distance places the 271 stored columns at physical channels 143-413."""
        distance = carina_patch.get_coord("distance")
        assert len(distance) == 271
        step = 1.0 * 1.020952  # SpatialResolution * Fibre Length Multiplier
        assert np.isclose(distance.step, step)
        assert np.isclose(distance.min(), -101.584763 + 143 * step)

    def test_attrs(self, carina_patch):
        """Silixa metadata lands on the patch attrs; data stay unitless counts."""
        attrs = carina_patch.attrs
        assert np.isclose(attrs.gauge_length, 2.041905)
        assert not attrs.data_units
        assert carina_patch.data.dtype == np.int16

    def test_non_bijective_channel_map_raises(self, carina_path, tmp_path):
        """A ChannelMap that loses a data column refuses to guess distances."""
        path = tmp_path / "bad_map.h5"
        shutil.copy(carina_path, path)
        with h5py.File(path, "a") as h5:
            h5["ChannelMap"][143] = -1  # column 0 is no longer mapped
        with pytest.raises(InvalidFiberFileError, match="ChannelMap"):
            SilixaH5V2().read(path)

    def test_gapped_channel_map_distances(self, carina_path, tmp_path):
        """A bijective but non-contiguous map yields per-channel distances."""
        path = tmp_path / "gapped_map.h5"
        shutil.copy(carina_path, path)
        with h5py.File(path, "a") as h5:
            # move column 0 from physical channel 143 to physical channel 50
            h5["ChannelMap"][50] = h5["ChannelMap"][143]
            h5["ChannelMap"][143] = -1
        patch = SilixaH5V2().read(path)[0]
        distance = patch.get_coord("distance")
        step = 1.0 * 1.020952
        assert len(distance) == 271
        assert np.isclose(distance.min(), -101.584763 + 50 * step)

    def test_partial_files_not_claimed(self, tmp_path):
        """Files missing the ChannelMap or the attr family aren't claimed."""
        no_map = tmp_path / "no_map.h5"
        with h5py.File(no_map, "w") as h5:
            h5.create_dataset("Fiber", data=np.zeros((4, 3), dtype=np.int16))
        assert not SilixaH5V2().get_format(no_map)
        no_attrs = tmp_path / "no_attrs.h5"
        with h5py.File(no_attrs, "w") as h5:
            h5.create_dataset("Fiber", data=np.zeros((4, 3), dtype=np.int16))
            h5.create_dataset("ChannelMap", data=np.arange(3, dtype=np.int32))
        assert not SilixaH5V2().get_format(no_attrs)

    def test_bad_samplerate_raises(self, carina_path, tmp_path):
        """A zero Samplerate raises a format error, not ZeroDivisionError."""
        path = tmp_path / "zero_rate.h5"
        shutil.copy(carina_path, path)
        with h5py.File(path, "a") as h5:
            h5.attrs["Samplerate"] = np.array([0.0])
        with pytest.raises(InvalidFiberFileError, match="Samplerate"):
            SilixaH5V2().read(path)

    def test_bad_channel_map_shapes_raise(self, carina_path, tmp_path):
        """A 2-d or all-unmapped ChannelMap raises a format error."""
        two_d = tmp_path / "twod_map.h5"
        shutil.copy(carina_path, two_d)
        with h5py.File(two_d, "a") as h5:
            cm = h5["ChannelMap"][()]
            del h5["ChannelMap"]
            h5.create_dataset("ChannelMap", data=cm.reshape(1, -1))
        with pytest.raises(InvalidFiberFileError, match="one-dimensional"):
            SilixaH5V2().read(two_d)
        unmapped = tmp_path / "unmapped.h5"
        shutil.copy(carina_path, unmapped)
        with h5py.File(unmapped, "a") as h5:
            h5["ChannelMap"][:] = -1
        with pytest.raises(InvalidFiberFileError, match="no channels"):
            SilixaH5V2().read(unmapped)

    def test_group_channel_map_not_claimed(self, carina_path, tmp_path):
        """A ChannelMap that is a group, not a dataset, is not claimed."""
        path = tmp_path / "group_map.h5"
        with h5py.File(carina_path, "r") as src, h5py.File(path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.create_dataset("Fiber", data=src["Fiber"][:10, :])
            dst.create_group("ChannelMap")
        assert not SilixaH5V2().get_format(path)

    def test_netcdf_does_not_claim(self, carina_path):
        """NETCDF_CF must keep rejecting these files (no Conventions attr)."""
        pytest.importorskip("xarray")
        with h5py.File(carina_path, "r") as h5file:
            assert NetCDFCFV18().get_format(h5file) is False
