"""Tests for simple h5 format."""

from __future__ import annotations

import shutil

import h5py
import numpy as np
import pytest
import tables

import dascore as dc
from dascore.io.h5simple.utils import (
    _get_attr_names,
    _get_root_attrs,
    _iter_root_arrays,
)
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


class TestH5SimpleInternalHelpers:
    """Direct tests for helper branches that still support PyTables fixtures."""

    def test_get_root_attrs_supports_pytables(self, tmp_path):
        """PyTables handles should expose root attrs through the helper."""
        path = tmp_path / "root_attrs.h5"
        with tables.open_file(path, "w") as h5:
            h5.root._v_attrs["dims"] = "distance,time"
            attrs = _get_root_attrs(h5)
            assert attrs.dims == "distance,time"

    def test_iter_root_arrays_supports_pytables(self, tmp_path):
        """PyTables root arrays should still be discoverable by helper code."""
        path = tmp_path / "root_arrays.h5"
        with tables.open_file(path, "w") as h5:
            h5.create_array("/", "data", obj=np.arange(3))
            names = [name for name, _node in _iter_root_arrays(h5)]
        assert names == ["data"]

    def test_get_attr_names_supports_pytables_attrs(self, tmp_path):
        """PyTables attr containers should still expose their stored keys."""
        path = tmp_path / "attr_names.h5"
        with tables.open_file(path, "w") as h5:
            h5.root._v_attrs["dims"] = "distance,time"
            out = _get_attr_names(h5.root._v_attrs)
        assert "dims" in out

    def test_get_root_attrs_supports_h5py(self, tmp_path):
        """h5py files should continue to use the attrs mapping directly."""
        path = tmp_path / "h5py_attrs.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["dims"] = "distance,time"
            attrs = _get_root_attrs(h5)
            assert attrs["dims"] == "distance,time"

    def test_iter_root_arrays_supports_h5py(self, tmp_path):
        """h5py root arrays should still be discoverable by helper code."""
        path = tmp_path / "h5py_root_arrays.h5"
        with h5py.File(path, "w") as h5:
            h5.create_dataset("data", data=np.arange(3))
            names = [name for name, _node in _iter_root_arrays(h5)]
        assert names == ["data"]

    def test_get_attr_names_supports_h5py_attrs(self, tmp_path):
        """h5py attr containers should still expose their stored keys."""
        path = tmp_path / "h5py_attr_names.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["dims"] = "distance,time"
            out = _get_attr_names(h5.attrs)
        assert "dims" in out
