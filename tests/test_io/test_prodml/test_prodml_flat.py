"""Tests for the flat layout: one DAS dataset carrying PRODML-style attrs."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.ap_sensing import APSensingV10
from dascore.io.h5simple import H5Simple
from dascore.io.prodml import ProdMLFlatV1, ProdMLV2_0, ProdMLV2_1
from dascore.io.silixah5 import SilixaH5V1, SilixaH5V2
from dascore.utils.downloader import fetch

FIXTURE = "prodml_flat_1.h5"
SIBLINGS = (
    (ProdMLV2_0(), "prodml_2.0.h5"),
    (ProdMLV2_1(), "prodml_2.1.h5"),
    (SilixaH5V1(), "silixa_h5_1.hdf5"),
    (SilixaH5V2(), "silixa_h5_ingv_1.h5"),
    (H5Simple(), "h5_simple_1.h5"),
    (APSensingV10(), "ap_sensing_1.hdf5"),
)


@pytest.fixture(scope="module")
def path():
    """The trimmed real file."""
    return fetch(FIXTURE)


@pytest.fixture(scope="module")
def raw(path):
    """The stored array and the DAS dataset's attrs."""
    with h5py.File(path, "r") as h5:
        return h5["DAS"][:], dict(h5["DAS"].attrs)


@pytest.fixture(scope="module")
def patch(path):
    """The fixture read whole."""
    return dc.read(path)[0]


@pytest.fixture
def copy(path, tmp_path):
    """A writable copy of the fixture."""
    return shutil.copy(path, tmp_path / FIXTURE)


class TestGetFormat:
    """Tests for detecting the flat layout."""

    def test_detected(self, path):
        """The fixture is detected without a format hint."""
        assert dc.get_format(path) == ("PRODML_FLAT", "1")

    @pytest.mark.parametrize("fiber_io, name", SIBLINGS)
    def test_sibling_not_claimed(self, fiber_io, name):
        """Sibling formats' fixtures are not claimed."""
        assert ProdMLFlatV1().get_format(fetch(name)) is False

    @pytest.mark.parametrize("fiber_io, name", SIBLINGS)
    def test_not_claimed_by_sibling(self, fiber_io, name, path):
        """Sibling readers do not claim the fixture."""
        assert fiber_io.get_format(path) is False

    def test_missing_attr(self, copy):
        """A DAS dataset lacking one of the layout's attrs is not claimed."""
        with h5py.File(copy, "r+") as h5:
            del h5["DAS"].attrs["AmpScaling"]
        assert ProdMLFlatV1().get_format(copy) is False

    def test_group_named_das(self, tmp_path):
        """A DAS group carrying the attrs is not claimed."""
        path = tmp_path / "group.h5"
        with h5py.File(fetch(FIXTURE), "r") as src, h5py.File(path, "w") as h5:
            h5.create_group("DAS").attrs.update(src["DAS"].attrs)
        assert ProdMLFlatV1().get_format(path) is False


class TestCoords:
    """Tests for the coordinates the header states."""

    def test_time(self, patch, raw):
        """Time steps at 1/OutputDataRate from PartStartTime to PartEndTime."""
        _, attrs = raw
        time = patch.get_coord("time")
        start = np.datetime64("2022-01-26T04:46:24.601960")
        step = np.timedelta64(200, "us")
        assert attrs["Count"] == len(time) == 400
        assert time.min() == start
        assert time.step == step
        assert time.max() == start + 399 * step
        end = np.datetime64(attrs["PartEndTime"].decode().split("+")[0])
        assert time.max() + step == end

    def test_distance(self, patch, raw):
        """Distance steps at SpatialResolution * FibreLengthMultiplier."""
        _, attrs = raw
        dist = patch.get_coord("distance")
        step = float(attrs["SpatialResolution"]) * float(attrs["FibreLengthMultiplier"])
        assert attrs["NumberOfLoci"] == len(dist) == 40
        assert dist.min() == float(attrs["StartDistance"])
        assert dist.step == step
        assert np.isclose(dist.max(), float(attrs["StartDistance"]) + 39 * step)
        assert np.float32(dist.max() + step) == attrs["StopDistance"]
        assert dc.get_quantity(dist.units) == dc.get_quantity("m")


class TestAttrs:
    """Tests for attrs and data units."""

    def test_units_and_attrs(self, patch):
        """RawDataUnit gives the data units; gauge length and scaling are kept."""
        assert dc.get_quantity(patch.attrs.data_units) == dc.get_quantity("nm/m/s")
        assert patch.attrs.gauge_length == 10.0
        assert patch.attrs.amp_scaling == pytest.approx(3.540039)

    def test_samples_unscaled(self, patch, raw):
        """Samples come back as stored: AmpScaling was applied before storage."""
        assert patch.dtype == np.int16
        np.testing.assert_array_equal(patch.data, raw[0])


class TestRead:
    """Tests for scanning and windowed reads."""

    def test_scan_matches_read(self, path, patch, monkeypatch):
        """A scan reads no samples and agrees with the read."""

        def reject(*args, **kwargs):
            raise AssertionError("samples read")

        monkeypatch.setattr(h5py.Dataset, "__getitem__", reject)
        summary = dc.scan(path)[0]
        assert summary.attrs.data_units == patch.attrs.data_units
        assert summary.coords["time"].min == patch.get_coord("time").min()
        assert summary.coords["distance"].max == patch.get_coord("distance").max()

    def test_window_equals_slice(self, path, patch, monkeypatch):
        """A windowed read equals a slice and reads only its hyperslab."""
        original = h5py.Dataset.__getitem__
        keys = []

        def capture(dataset, key):
            keys.append(key)
            return original(dataset, key)

        time, dist = patch.get_coord("time"), patch.get_coord("distance")
        monkeypatch.setattr(h5py.Dataset, "__getitem__", capture)
        out = dc.read(path, time=(time[10], time[19]), distance=(dist[3], dist[7]))[0]
        assert keys == [(slice(10, 20), slice(3, 8))]
        assert out.equals(patch.isel(time=slice(10, 20), distance=slice(3, 8)))


class TestBadMetadata:
    """Tests for refusing headers the array contradicts."""

    @pytest.mark.parametrize("name", ["Count", "NumberOfLoci"])
    def test_count_mismatch(self, copy, name):
        """A Count or NumberOfLoci not matching the shape is refused."""
        with h5py.File(copy, "r+") as h5:
            h5["DAS"].attrs[name] = np.int32(h5["DAS"].attrs[name] + 1)
        with pytest.raises(InvalidFiberFileError, match="shape"):
            dc.read(copy, file_format="PRODML_FLAT")
