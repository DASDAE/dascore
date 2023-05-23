"""
Misc. tests for Terra15.
"""
import shutil

import numpy as np
import pandas as pd
import pytest
import tables

import dascore as dc


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
        assert not np.any(pd.isnull(patch.coords["time"]))
