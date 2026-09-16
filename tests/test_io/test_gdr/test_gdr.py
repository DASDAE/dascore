"""Tests for the GDR file format."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.gdr import GDR_V1
from dascore.io.gdr.utils_das import _get_dims
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def gpr_path():
    """Return the file path to a GDR file."""
    return fetch("gdr_1.h5")


class TestGDR:
    """Misc. tests not covered by common tests."""

    def test_no_snap(self, gpr_path):
        """Ensure snap or no snap produces the same coord for this file."""
        fiber_io = GDR_V1()
        patch1 = fiber_io.read(gpr_path, snap=False)[0]
        patch2 = fiber_io.read(gpr_path, snap=True)[0]
        time_1 = patch1.get_coord("time")
        time_2 = patch2.get_coord("time")
        assert len(time_1) == len(time_2)
        assert np.all(time_1.values == time_2.values)


class TestGetDims:
    """Tests for reading the dimension names the file states."""

    @staticmethod
    def _dataset(dimensions):
        """A stand-in dataset stating these DasDimensions."""
        return SimpleNamespace(attrs={"DasDimensions": dimensions})

    @pytest.mark.parametrize("locus", ["locus", b"locus", np.bytes_("locus")])
    def test_locus_is_distance(self, locus):
        """The file's locus axis is DASCore's distance, however it is stored."""
        assert _get_dims(self._dataset(["time", locus])) == ("time", "distance")

    def test_order_follows_the_file(self):
        """A distance-major file is reported distance-major."""
        assert _get_dims(self._dataset(["locus", "time"])) == ("distance", "time")

    def test_unknown_dimension_raises(self):
        """A dimension name the reader cannot map is not guessed at."""
        with pytest.raises(AssertionError, match="DasDimensions"):
            _get_dims(self._dataset(["time", "bob"]))


class TestSingletonTime:
    """A file with one acquisition time cannot infer its sampling interval."""

    @pytest.mark.parametrize("transpose", [False, True])
    @pytest.mark.parametrize("snap", [None, False, "time", ("distance",)])
    def test_single_time_read_and_scan(self, tmp_path, transpose, snap):
        """Both stored axis orders preserve the single timestamp and samples."""
        path = tmp_path / "singleton.h5"
        data = np.array([[101], [203], [307]], dtype="int16")
        dims = ["locus", "time"]
        if transpose:
            data = data.T
            dims.reverse()
        time = np.array([np.datetime64("2024-01-02T03:04:05.123456789", "ns")])
        with h5py.File(path, "w") as h5:
            meta = h5.create_group("DasMetadata")
            meta.attrs.update(
                MetadataStandard="DAS-RCN v1.10", RawDataStandard="PRODML v2.2"
            )
            interrogator = meta.create_group("Interrogator")
            interrogator.attrs["SerialNumber"] = "singleton-test"
            acquisition = interrogator.create_group("Acquisition")
            acquisition.attrs.update(
                GaugeLength=10.0,
                GaugeLengthUnit="m",
                UnitOfMeasure="1/s",
                SpatialSamplingInterval=2.5,
                SpatialSamplingIntervalUnit="m",
            )
            h5["DasRawData/DasTimeArray"] = time.astype("int64")
            h5["DasRawData/RawData"] = data
            h5["DasRawData/RawData"].attrs["DasDimensions"] = dims
        kwargs = {} if snap is None else {"snap": snap}
        patch = dc.read(path, **kwargs)[0]
        scanned = dc.scan_payloads(path, **kwargs)[0]
        np.testing.assert_array_equal(patch.data, data)
        assert patch.dtype == data.dtype
        for name, expected in (("time", time), ("distance", [0.0, 2.5, 5.0])):
            np.testing.assert_array_equal(patch.get_coord(name).values, expected)
            np.testing.assert_array_equal(scanned.get_coord(name).values, expected)
        bounded = dc.read(path, time=(time[0], time[0]), **kwargs)[0]
        np.testing.assert_array_equal(bounded.data, data)
