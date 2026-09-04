"""Tests for simple h5 format."""

from __future__ import annotations

import shutil

import h5py
import pytest

import dascore as dc
from dascore.constants import STORAGE_PROVENANCE_ATTRS
from dascore.exceptions import UnknownFiberFormatError
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
        with h5py.File(new_path, "a") as h5:
            h5.attrs["dims"] = "distance,time"
        return new_path

    def test_no_snap(self, h5simple_path):
        """Ensure when snap is not used it still reads patch."""
        patch = dc.read(h5simple_path, file_format="h5simple", snap=False)[0]
        assert isinstance(patch, dc.Patch)

    def test_dims_in_attrs(self, h5simple_with_dim_attrs_path):
        """Ensure if 'dims' is in attrs it gets used."""
        patch = dc.spool(h5simple_with_dim_attrs_path, file_format="h5simple")[0]
        assert isinstance(patch, dc.Patch)

    def test_provenance_in_file_is_not_a_patch_attr(self, h5simple_path, tmp_path):
        """A root attr naming where the bytes live is the spool's, not the patch's.

        The format has no header schema, so every root attr is copied. The
        example files happen to carry none of these, which is why only a
        file written with them shows the leak.
        """
        path = tmp_path / "provenance.h5"
        shutil.copy(h5simple_path, path)
        with h5py.File(path, "r+") as h5:
            h5.attrs["file_version"] = "1"
            h5.attrs["path"] = "/somewhere/original.h5"
        read_names = set(dict(dc.read(path)[0].attrs))
        scan_names = set(dict(dc.scan(path)[0].attrs))
        assert not read_names & set(STORAGE_PROVENANCE_ATTRS)
        # Stripping it in only one of the two is how they came to disagree.
        assert read_names == scan_names

    @pytest.mark.parametrize("name", ["__format__", "format", "file_format"])
    def test_declared_format_is_recognized(self, h5simple_path, tmp_path, name):
        """A root attr naming the format opts the file in, not out."""
        path = tmp_path / f"{name}.h5"
        shutil.copy(h5simple_path, path)
        with h5py.File(path, "r+") as h5:
            h5.attrs[name] = "h5simple"
        assert dc.get_format(path) == ("H5Simple", "1")
        assert isinstance(dc.read(path)[0], dc.Patch)

    def test_other_declared_format_is_rejected(self, h5simple_path, tmp_path):
        """A root attr naming another format still rules h5simple out."""
        path = tmp_path / "other.h5"
        shutil.copy(h5simple_path, path)
        with h5py.File(path, "r+") as h5:
            h5.attrs["format"] = "other"
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(path)
