"""Tests for simple h5 format."""

from __future__ import annotations

import shutil
from functools import wraps

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.constants import STORAGE_PROVENANCE_ATTRS
from dascore.exceptions import UnknownFiberFormatError
from dascore.io.h5simple.core import H5Simple
from dascore.utils.downloader import fetch
from dascore.utils.hdf5 import H5Reader


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

    def test_read_traverses_layout_once(self, h5simple_path, monkeypatch):
        """Metadata and samples share one traversal of the HDF5 root nodes."""
        original = h5py.Group.items
        calls = []

        def items(group):
            if group.name == "/":
                calls.append(1)
            return original(group)

        monkeypatch.setattr(h5py.Group, "items", items)
        patch = dc.read(h5simple_path, file_format="H5Simple")[0]
        assert patch.size
        assert len(calls) == 1

    def test_bounded_read_only_loads_selected_samples(self, tmp_path, monkeypatch):
        """The prepared reader slices storage before loading samples."""
        path = tmp_path / "bounded.h5"
        data = np.arange(20_000).reshape(200, 100)
        with h5py.File(path, "w") as handle:
            handle["raw"] = data
            handle["time"] = np.arange(200, dtype=float)
            handle["distance"] = np.arange(100)
            handle.attrs["dims"] = "time,distance"
        original = h5py.Dataset.__getitem__
        reads = []

        def getitem(dataset, selection):
            result = original(dataset, selection)
            if dataset.name == "/raw":
                reads.append(np.asarray(result).size)
            return result

        monkeypatch.setattr(h5py.Dataset, "__getitem__", getitem)
        patch = dc.read(
            path, file_format="H5Simple", samples=True, time=(2, 4), distance=(3, 5)
        )[0]
        np.testing.assert_array_equal(patch.data, data[2:4, 3:5])
        assert reads == [4]

    def test_subclass_array_override(self, h5simple_path):
        """A format extension's public array hook remains authoritative."""

        class Custom(H5Simple):
            name = "_test_h5simple_custom_array"

            def read_array(self, resource: H5Reader, windows, key=""):
                return super().read_array(resource, windows, key=key) * 2

        expected = H5Simple().read(h5simple_path)[0]
        actual = Custom().read(h5simple_path)[0]
        np.testing.assert_array_equal(actual.data, expected.data * 2)

    def test_subclass_metadata_override(self, h5simple_path):
        """A format extension's public metadata hook is used during reading."""

        class Custom(H5Simple):
            name = "_test_h5simple_custom_metadata"

            def get_metadata(self, resource: H5Reader, *, snap=True):
                return [
                    patch.update_attrs(tag="custom")
                    for patch in super().get_metadata(resource, snap=snap)
                ]

        assert Custom().read(h5simple_path)[0].attrs.tag == "custom"

    @pytest.mark.parametrize("hook", ["get_metadata", "read_array"])
    @pytest.mark.parametrize("on_class", [False, True])
    def test_runtime_wrappers_are_honored(
        self, h5simple_path, monkeypatch, hook, on_class
    ):
        """Runtime instrumentation of either public hook affects shared reading."""
        reader = H5Simple()
        expected = reader.read(h5simple_path)[0]
        owner = H5Simple if on_class else reader
        original = getattr(owner, hook)

        @wraps(original)
        def wrapped(*args, **kwargs):
            out = original(*args, **kwargs)
            if hook == "read_array":
                return out * 2
            return [patch.update_attrs(tag="wrapped") for patch in out]

        monkeypatch.setattr(owner, hook, wrapped)
        actual = reader.read(h5simple_path)[0]
        if hook == "read_array":
            np.testing.assert_array_equal(actual.data, expected.data * 2)
        else:
            assert actual.attrs.tag == "wrapped"

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
    @pytest.mark.parametrize("value", ["h5simple", np.bytes_(b"h5simple")])
    def test_declared_format_is_recognized(self, h5simple_path, tmp_path, name, value):
        """A root attr naming the format opts the file in, not out.

        Files written through PyTables carry bytes rather than str, which
        is why the value is decoded rather than compared as it is stored.
        """
        path = tmp_path / f"{name}_{type(value).__name__}.h5"
        shutil.copy(h5simple_path, path)
        with h5py.File(path, "r+") as h5:
            h5.attrs[name] = value
        assert dc.get_format(path) == ("H5Simple", "1")
        assert isinstance(dc.read(path)[0], dc.Patch)

    def test_disagreeing_declared_formats_are_rejected(self, h5simple_path, tmp_path):
        """Two root attrs naming different formats rule the file out."""
        path = tmp_path / "disagree.h5"
        shutil.copy(h5simple_path, path)
        with h5py.File(path, "r+") as h5:
            h5.attrs["format"] = "h5simple"
            h5.attrs["__format__"] = "other"
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(path)

    def test_other_declared_format_is_rejected(self, h5simple_path, tmp_path):
        """A root attr naming another format still rules h5simple out."""
        path = tmp_path / "other.h5"
        shutil.copy(h5simple_path, path)
        with h5py.File(path, "r+") as h5:
            h5.attrs["format"] = "other"
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(path)
