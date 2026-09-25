"""Tests for simple h5 format."""

from __future__ import annotations

import copy
import shutil
from dataclasses import replace
from functools import wraps
from pathlib import Path

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.constants import STORAGE_PROVENANCE_ATTRS
from dascore.core.lazy_array import LazyArray
from dascore.exceptions import UnknownFiberFormatError
from dascore.io import FiberIO
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

    @pytest.fixture
    def isolated_registry(self, monkeypatch):
        """Keep temporary subclasses out of the shared formatter registry."""
        monkeypatch.setattr(H5Simple, "manager", copy.deepcopy(H5Simple.manager))

    def test_subclass_array_override(self, h5simple_path, isolated_registry):
        """A format extension's public array hook remains authoritative."""

        class Custom(H5Simple):
            name = "_test_h5simple_custom_array"

            def read_array(self, resource: H5Reader, windows, key=""):
                return super().read_array(resource, windows, key=key) * 2

        expected = H5Simple().read(h5simple_path)[0]
        actual = Custom().read(h5simple_path)[0]
        np.testing.assert_array_equal(actual.data, expected.data * 2)

    def test_subclass_metadata_override(self, h5simple_path, isolated_registry):
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

    @pytest.fixture
    def source_registry(self, monkeypatch):
        """Register temporary subclasses where sources find their format."""
        monkeypatch.setattr(FiberIO, "manager", copy.deepcopy(FiberIO.manager))

    def test_subclass_array_override_loads_sources(
        self, h5simple_path, source_registry
    ):
        """A source loads through a subclass's read_array, not the parent's."""

        class Custom(H5Simple):
            name = "_test_h5simple_custom_source"

            def read_array(self, resource: H5Reader, windows=(), key=""):
                return super().read_array(resource, windows, key=key) * 2

        patch = dc.read(h5simple_path, file_format=H5Simple.name)[0]
        source = replace(patch._source, format=Custom.name)
        np.testing.assert_array_equal(source[1:3].load(), patch.data[1:3] * 2)

    def test_override_annotation_loads_sources(self, h5simple_path, source_registry):
        """A source's resource is the type the effective read_array asks for."""
        seen = []

        class Custom(H5Simple):
            name = "_test_h5simple_custom_path"

            def read_array(self, resource: Path, windows=(), key=""):
                seen.append(resource)
                return H5Simple().read_array(resource, windows, key=key)

        patch = dc.read(h5simple_path, file_format=H5Simple.name)[0]
        source = replace(patch._source, format=Custom.name)
        np.testing.assert_array_equal(source.load(), patch.data)
        assert isinstance(seen[0], Path)

    @pytest.mark.parametrize("on_class", [False, True])
    def test_runtime_array_wrapper_loads_sources(
        self, h5simple_path, monkeypatch, on_class, source_registry
    ):
        """A source loads through a wrapped read_array, not the prepared node."""
        # The registry is a copy, so no wrapper outlives the test in it.
        patch = dc.read(h5simple_path, file_format=H5Simple.name)[0]
        expected = np.array(patch.data)
        reader = FiberIO.manager.get_fiberio(format=H5Simple.name, version="1")
        owner = H5Simple if on_class else reader
        original = owner.read_array

        @wraps(original)
        def wrapped(*args, **kwargs):
            return original(*args, **kwargs) * 2

        monkeypatch.setattr(owner, "read_array", wrapped)
        np.testing.assert_array_equal(patch._source.load(), expected * 2)

    @pytest.mark.parametrize("doubled_first", [True, False])
    def test_bases_with_different_hooks(
        self, h5simple_path, source_registry, doubled_first
    ):
        """
        A hook one base defines cannot bypass another base's read_array.

        The doubling base defines its own `_prepare_read`, and H5Simple the
        array hook, which was written for H5Simple's read_array.
        """

        class Doubled(FiberIO):
            name = f"_test_h5simple_doubled_{doubled_first}"

            def _prepare_read(self, manager, snap):
                return FiberIO._prepare_read(self, manager, snap)

            def get_metadata(self, resource: H5Reader, *, snap=True):
                return H5Simple().get_metadata(resource, snap=snap)

            def read_array(self, resource: H5Reader, windows=(), key=""):
                return H5Simple().read_array(resource, windows, key=key) * 2

        bases = (Doubled, H5Simple) if doubled_first else (H5Simple, Doubled)

        class Combined(*bases):
            name = f"{Doubled.name}_combined"

        patch = dc.read(h5simple_path, file_format=H5Simple.name)[0]
        plain = np.array(patch.data)
        expected = plain * 2 if doubled_first else plain
        np.testing.assert_array_equal(Combined().read_array(h5simple_path), expected)
        source = replace(patch._source, format=Combined.name)
        np.testing.assert_array_equal(source[1:3].load(), expected[1:3])
        np.testing.assert_array_equal(Combined().read(h5simple_path)[0].data, expected)

    def test_mixin_array_hook_cannot_bypass_override(
        self, h5simple_path, source_registry
    ):
        """A plain mixin's array hook, paired with nothing, is not trusted."""

        class StaleHook:
            def _prepare_array_reader(self, resource, *, key=""):
                return H5Simple()._prepare_array_reader(resource, key=key)

        class Reader(StaleHook, H5Simple):
            name = "_test_h5simple_mixin_reader"

        class Scaled(Reader):
            name = "_test_h5simple_mixin_scaled"

            def read_array(self, resource: H5Reader, windows=(), key=""):
                return super().read_array(resource, windows, key=key) * 2

        patch = dc.read(h5simple_path, file_format=H5Simple.name)[0]
        expected = np.array(patch.data)[1:3] * 2
        source = replace(patch._source, format=Scaled.name)[1:3]
        np.testing.assert_array_equal(Scaled().read_array(h5simple_path)[1:3], expected)
        np.testing.assert_array_equal(source.load(), expected)

    def test_compatible_array_hook_is_kept(
        self, h5simple_path, source_registry, monkeypatch
    ):
        """A subclass pairing its own array hook with its read_array keeps it."""
        prepared = []

        class Compatible(H5Simple):
            name = "_test_h5simple_compatible"

            def read_array(self, resource: H5Reader, windows=(), key=""):
                return self._prepare_array_reader(resource, key=key)(windows)

            def _prepare_array_reader(self, resource, *, key=""):
                prepared.append(key)
                read = super()._prepare_array_reader(resource, key=key)
                return lambda windows: read(windows) * 2

        original = h5py.Group.items
        traversals = []

        def items(group):
            if group.name == "/":
                traversals.append(1)
            return original(group)

        patch = dc.read(h5simple_path, file_format=H5Simple.name)[0]
        expected = np.array(patch.data)[:6] * 2
        source = replace(patch._source, format=Compatible.name)
        array = LazyArray.from_sources([source[0:2], source[2:4], source[4:6]])
        monkeypatch.setattr(h5py.Group, "items", items)
        np.testing.assert_array_equal(array.load(), expected)
        assert prepared == [""] and len(traversals) == 1

    def test_subclass_keeps_prepared_read(self, h5simple_path, monkeypatch):
        """A subclass which overrides nothing still reuses the parsed layout."""
        monkeypatch.setattr(H5Simple, "manager", copy.deepcopy(H5Simple.manager))

        class Plain(H5Simple):
            name = "_test_h5simple_plain_subclass"

        original = h5py.Group.items
        calls = []

        def items(group):
            if group.name == "/":
                calls.append(1)
            return original(group)

        monkeypatch.setattr(h5py.Group, "items", items)
        assert Plain().read(h5simple_path)[0].size
        assert len(calls) == 1

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


class TestSingletonCoordinates:
    """A one-sample axis has a known position but no inferable interval."""

    @pytest.mark.parametrize("shape", [(1, 3), (4, 1), (1, 1)])
    @pytest.mark.parametrize("snap", [None, False, "time", ("distance",)])
    def test_singleton_read_and_scan(self, tmp_path, shape, snap):
        """Full and bounded reads retain singleton spatial and time values."""
        path = tmp_path / "singleton.h5"
        data = (100 + np.arange(np.prod(shape))).reshape(shape).astype("int16")
        time = 1000.0 + np.arange(shape[0]) * 0.25
        distance = 11.0 + np.arange(shape[1]) * 2.5
        expected_time = np.datetime64(1000, "s") + np.arange(shape[0]) * np.timedelta64(
            250, "ms"
        )
        with h5py.File(path, "w") as h5:
            h5["raw"] = data
            h5["time"] = time
            h5["distance"] = distance
            h5.attrs["dims"] = "time,distance"
        kwargs = {} if snap is None else {"snap": snap}
        patch = dc.read(path, **kwargs)[0]
        scanned = dc.scan_payloads(path, **kwargs)[0]
        np.testing.assert_array_equal(patch.data, data)
        assert patch.dtype == data.dtype
        for name, expected in (("time", expected_time), ("distance", distance)):
            np.testing.assert_array_equal(patch.get_coord(name).values, expected)
            np.testing.assert_array_equal(scanned.get_coord(name).values, expected)
        bounded = dc.read(
            path,
            time=(expected_time[0], expected_time[0]),
            distance=(distance[0], distance[0]),
            **kwargs,
        )[0]
        np.testing.assert_array_equal(bounded.data, data[:1, :1])
