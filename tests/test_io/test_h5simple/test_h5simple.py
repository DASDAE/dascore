"""Tests for simple h5 format."""

from __future__ import annotations

import shutil

import pytest
import tables

import dascore as dc
from dascore.utils.downloader import fetch


class TestH5Simple:
    """Tests for h5simple that aren't covered in common tests."""

    @pytest.fixture(scope="class")
    def h5simple_path(self):
        """Get the path to a h5 simple file."""
        return fetch("h5_simple_1.h5")

    @pytest.fixture(scope="class")
    def h5simple_with_dim_attrs_path(self, tmp_path_factory):
        """Create a h5_simpl which has dimensions specified."""
        basic_path = fetch("h5_simple_2.h5")
        new_path = tmp_path_factory.mktemp("h5simple_dim_attrs") / "simple.h5"

        shutil.copy2(basic_path, new_path)
        with tables.open_file(new_path, "a") as h5:
            h5.root._v_attrs["dims"] = "distance,time"
        return new_path

    def test_no_snap(self, h5simple_path):
        """Ensure when snap is not used it still reads patch."""
        patch = dc.read(h5simple_path, file_format="h5simple", snap=False)[0]
        assert isinstance(patch, dc.Patch)

    def test_dims_in_attrs(self, h5simple_with_dim_attrs_path):
        """Ensure if 'dims' is in attrs it gets used."""
        patch = dc.spool(h5simple_with_dim_attrs_path, file_format="h5simple")[0]
        assert isinstance(patch, dc.Patch)
