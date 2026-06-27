"""
FEBUS G1 BSL HDF5 specific tests.
"""

import shutil

import h5py
import numpy as np
import pytest
from numpy.testing import assert_allclose

import dascore as dc
from dascore.io.febus import FebusBSLH5V1
from dascore.io.febus.g1utils import _get_bsl_attrs
from dascore.utils.downloader import fetch

BSL_NAME = "febusg1_C2_2026-06-03T17.18.13+0200.bsl.h5"


class TestFebusBSL:
    """Tests for the FEBUS G1 BSL HDF5 reader."""

    parser = FebusBSLH5V1()

    @pytest.fixture(scope="class")
    def bsl_path(self):
        """Path to a G1 BSL HDF5 test file."""
        return fetch(BSL_NAME)

    @pytest.fixture(scope="class")
    def bsl_patch(self, bsl_path):
        """Return the parsed G1 BSL patch."""
        return self.parser.read(bsl_path)[0]

    def test_get_format(self, bsl_path):
        """Ensure the BSL HDF5 format can be auto-detected."""
        assert self.parser.get_format(bsl_path) == (
            self.parser.name,
            self.parser.version,
        )

    def test_future_format_version_not_claimed(self, bsl_path, tmp_path):
        """Future BSL format versions should not be claimed by the v1 reader."""
        new_path = tmp_path / bsl_path.name
        shutil.copy2(bsl_path, new_path)
        with h5py.File(new_path, "a") as fi:
            fi.attrs["formatVersion"] = np.array([2], dtype=np.uint64)
        assert not self.parser.get_format(new_path)

    def test_scan(self, bsl_path):
        """Scan returns one patch attrs object with expected metadata."""
        payloads = self.parser.scan(bsl_path)
        assert len(payloads) == 1
        payload = payloads[0]
        attr = payload["attrs"]
        assert isinstance(attr, dc.PatchAttrs)
        assert "path" not in attr.model_dump()
        assert "file_format" not in attr.model_dump()
        assert "file_version" not in attr.model_dump()
        assert payload["dims"] == ("time", "distance")
        assert attr.data_category == "DSS"
        assert attr.data_type == "strain"
        assert attr.data_units == dc.get_quantity("microstrain")

    def test_private_attrs_without_io_provenance(self, bsl_path):
        """The low-level attrs helper can still omit DASCore IO attrs."""
        with h5py.File(bsl_path) as h5:
            attrs = _get_bsl_attrs(h5)
        assert "file_format" not in attrs
        assert "file_version" not in attrs
        assert "path" not in attrs

    def test_read(self, bsl_patch):
        """Ensure the BSL file is read into a patch with expected shape."""
        assert isinstance(bsl_patch, dc.Patch)
        assert bsl_patch.shape == (120, 100)
        assert bsl_patch.attrs.data_units == dc.get_quantity("microstrain")
        assert "temperature" in bsl_patch.coords.coord_map
        assert bsl_patch.coords.dim_map["temperature"] == ("time",)

    def test_read_attrs_have_io_provenance(self, bsl_path, bsl_patch):
        """Read patch attrs should include path and format provenance."""
        assert bsl_patch.attrs.path == str(bsl_path)
        assert bsl_patch.attrs.file_format == self.parser.name
        assert bsl_patch.attrs.file_version == self.parser.version

    def test_distance_range(self, bsl_patch):
        """Distance should span 50-149 m."""
        dist = bsl_patch.get_coord("distance")
        assert_allclose(dist.min(), 50.0)
        assert_allclose(dist.max(), 149.0)
        assert_allclose(dist.step, 1.0)

    def test_time_coord(self, bsl_patch):
        """Time should be monotonic but irregularly sampled."""
        time = bsl_patch.get_coord("time")
        assert "datetime64" in str(np.dtype(time.dtype))
        assert time.min() == np.datetime64("2026-06-03T15:18:13.422442752")
        assert time.max() == np.datetime64("2026-06-03T15:28:08.829897728")
        assert time.step is None
        assert bsl_patch.summary.get_coord_summary("time").step is None

    def test_select(self, bsl_path, bsl_patch):
        """Partial reads should reduce coords and data consistently."""
        time = bsl_patch.get_coord("time")
        dist = bsl_patch.get_coord("distance")
        out = self.parser.read(
            bsl_path,
            time=(time.values[10], time.values[20]),
            distance=(dist.min() + 5, dist.min() + 10),
        )[0]
        assert out.shape == (11, 6)
        assert out.get_coord("time").min() == time.values[10]
        assert out.get_coord("time").max() == time.values[20]
        assert_allclose(out.get_coord("distance").min(), 55.0)
        assert_allclose(out.get_coord("distance").max(), 60.0)

    def test_out_of_range_selects_empty_spool(self, bsl_path, bsl_patch):
        """Out-of-range time and distance selections should return empty spools."""
        time = bsl_patch.get_coord("time")
        dist = bsl_patch.get_coord("distance")
        assert not len(
            self.parser.read(bsl_path, time=(time.max() + np.timedelta64(1, "s"), ...))
        )
        assert not len(self.parser.read(bsl_path, distance=(dist.max() + 1, ...)))
