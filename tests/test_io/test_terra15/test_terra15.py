"""Misc. tests for Terra15."""

from __future__ import annotations

import shutil
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
import tables

import dascore as dc
from dascore.io.terra15 import Terra15FormatterV4


class TestTerra15:
    """Misc tests for Terra15."""

    @pytest.fixture(scope="class")
    def missing_gps_terra15_hdf5(self, terra15_v5_path, tmp_path_factory):
        """Creates a terra15 file with missing GPS Time."""
        new = tmp_path_factory.mktemp("missing_gps") / "missing.hdf5"
        shutil.copy(terra15_v5_path, new)
        with tables.open_file(new, "a") as fi:
            fi.root.data_product.gps_time._f_remove()
        return new

    def test_missing_gps_time(self, missing_gps_terra15_hdf5):
        """Tests for when GPS time isn't found."""
        patch = dc.read(missing_gps_terra15_hdf5)[0]
        assert isinstance(patch, dc.Patch)
        assert not np.any(pd.isnull(patch.coords.get_array("time")))

    def test_time_slice(self, terra15_v6_path):
        """Ensure time slice within the file works."""
        info = dc.scan_to_df(terra15_v6_path).iloc[0]
        file_t1, file_t2 = info["time_min"], info["time_max"]
        dur = file_t2 - file_t1
        new_dur = dur / 4
        t1, t2 = file_t1 + new_dur, file_t1 + 2 * new_dur
        out = dc.read(terra15_v6_path, time=(t1, t2))[0]
        assert isinstance(out, dc.Patch)
        attrs = out.attrs
        assert attrs.time_min >= t1
        assert attrs.time_max <= t2

    def test_time_slice_no_snap(self, terra15_v6_path):
        """Ensure no snapping returns raw time."""
        info = dc.scan_to_df(terra15_v6_path).iloc[0]
        file_t1, file_t2 = info["time_min"], info["time_max"]
        dur = file_t2 - file_t1
        new_dur = dur / 4
        t1, t2 = file_t1 + new_dur, file_t1 + 2 * new_dur
        out = dc.read(terra15_v6_path, time=(t1, t2), snap_dims=False)[0]
        assert isinstance(out, dc.Patch)
        attrs = out.attrs
        assert attrs.time_min >= t1
        assert attrs.time_max <= t2

    def test_units(self, terra15_das_patch):
        """All units should be defined on terra15 patch."""
        patch = terra15_das_patch
        assert patch.attrs.data_units is not None
        assert patch.attrs.distance_units is not None
        assert patch.attrs.time_units is not None
        assert patch.get_coord("time").units == patch.attrs.time_units
        assert patch.get_coord("distance").units == patch.attrs.distance_units

    def test_hdf5file_not_terra15(self, generic_hdf5):
        """Assert that the generic hdf5 file is not a terra15."""
        parser = Terra15FormatterV4()
        assert not parser.get_format(generic_hdf5)

    def test_unsupported_version_error(self):
        """Test that unsupported Terra15 version raises NotImplementedError."""
        from dascore.io.terra15.utils import _get_version_data_node

        # Create a mock HDF5 root object with unsupported version
        class MockRoot:
            attrs: ClassVar = {"file_version": "999"}  # Unsupported version

        mock_root = MockRoot()

        # Test that it raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Unknown Terra15 version"):
            _get_version_data_node(mock_root)


class TestTerra15Unfinished:
    """Test for reading files with zeroes filled at the end."""

    @pytest.fixture(scope="class")
    def patch_unfinished(self, terra15_das_unfinished_path):
        """Return the patch with zeroes at the end."""
        out = dc.spool(terra15_das_unfinished_path)[0]
        return out

    def test_zeros_gone(self, patch_unfinished):
        """No zeros should exist in the data."""
        data = patch_unfinished.data
        all_zero_rows = np.all(data == 0, axis=1)
        assert not np.any(all_zero_rows)

    def test_monotonic_time(self, patch_unfinished):
        """Ensure the time is increasing."""
        time = patch_unfinished.coords.get_array("time")
        assert np.all(np.diff(time) >= np.timedelta64(0, "s"))
