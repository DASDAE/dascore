"""Tests for simple h5 format."""

from __future__ import annotations

import copy
import itertools
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
from dascore.core.source import ArraySource
from dascore.exceptions import UnknownFiberFormatError
from dascore.io import FiberIO
from dascore.io import core as io_core
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

            def _prepare_read(self, manager, snap, keys=frozenset()):
                return FiberIO._prepare_read(self, manager, snap, keys)

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
        assert prepared == [""] and not traversals

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


class TestArrayLookup:
    """Array reads only need the named dataset, regardless of root layout."""

    @pytest.fixture(params=["raw", "data", "extra"])
    def array_file(self, tmp_path, request):
        """Store deterministic samples and optional unrelated root nodes."""
        path = tmp_path / "array.h5"
        data = np.arange(24, dtype=np.float32).reshape(6, 4)
        with h5py.File(path, "w") as resource:
            resource["data" if request.param == "data" else "raw"] = data
            resource["time"] = np.arange(6, dtype=float)
            resource["distance"] = np.arange(4)
            resource.attrs["dims"] = "time,distance"
            if request.param == "extra":
                resource.create_group("data")
                resource["unrelated"] = np.arange(4)
        return path, data

    def test_arrays(self, array_file):
        """Direct and lazy windows return the same stored samples."""
        path, data = array_file
        source = ArraySource(path=str(path), format="H5Simple", version="1").describe(
            data.shape, data.dtype
        )
        assert source.key == ""
        np.testing.assert_array_equal(
            H5Simple().read_array(path, ((1, 4), (1, 3))), data[1:4, 1:3]
        )
        lazy = LazyArray.from_source(source)
        np.testing.assert_array_equal(lazy[1:4, 1:3].load(), data[1:4, 1:3])

    def test_no_walk(self, array_file, monkeypatch):
        """Neither direct nor lazy loading enumerates the root group."""
        path, data = array_file
        source = ArraySource(path=str(path), format="H5Simple", version="1").describe(
            data.shape, data.dtype
        )

        def refuse(*args, **kwargs):
            raise AssertionError("array reading must not walk the file")

        monkeypatch.setattr(h5py.Group, "items", refuse)
        monkeypatch.setattr(h5py.Group, "__iter__", refuse)
        np.testing.assert_array_equal(H5Simple().read_array(path), data)
        np.testing.assert_array_equal(LazyArray.from_source(source).load(), data)

    @pytest.mark.parametrize("names", [(), ("raw", "data"), ("group",)])
    def test_invalid_data_nodes(self, tmp_path, names):
        """Missing, ambiguous, and group-only layouts keep the same error."""
        path = tmp_path / "invalid.h5"
        with h5py.File(path, "w") as resource:
            resource["time"] = np.arange(6)
            for name in names:
                if name == "group":
                    resource.create_group("data")
                else:
                    resource[name] = np.zeros((6, 4))
            with pytest.raises(AssertionError) as error:
                H5Simple().read_array(resource, _pre_cast=True)
            assert str(error.value) == f"{resource} doesn't have exactly one data node."

    @pytest.mark.parametrize("key", ["other", "/other"])
    def test_explicit_key(self, array_file, key):
        """An explicit dataset key takes precedence over the default name."""
        path, data = array_file
        with h5py.File(path, "a") as resource:
            resource["other"] = -data
        np.testing.assert_array_equal(
            H5Simple().read_array(path, ((1, 4),), key=key), -data[1:4]
        )

    @pytest.mark.parametrize("key", ["missing", "group"])
    def test_key_fallback(self, array_file, key):
        """A key which is not a dataset falls back to candidate names."""
        path, data = array_file
        with h5py.File(path, "a") as resource:
            resource.create_group("group")
        np.testing.assert_array_equal(H5Simple().read_array(path, key=key), data)

    def test_stable_ids(self, array_file, monkeypatch):
        """With fixed file identity, patch and lazy ids match the dev baseline."""
        path, _ = array_file
        with h5py.File(path, "a") as resource:
            if "unrelated" in resource:
                del resource["unrelated"]
        monkeypatch.setattr(
            "dascore.io.core.source_identity",
            lambda source: ("/fixed/h5simple.h5", 4096, 123456789),
        )
        patch = dc.read(path)[0]
        lazy = LazyArray.from_source(patch._source)
        assert patch._source.key == ""
        assert patch.attrs.data_id == lazy.data_id == "af8aa5c7443693b5f6af5dd541cc83d4"


def _write_simple(path, data, start=0.0, gaps=False, **options):
    """Write an H5Simple file; `gaps` puts other storage between chunks."""
    with h5py.File(path, "w") as h5:
        raw = h5.create_dataset("raw", shape=data.shape, dtype=data.dtype, **options)
        step = (options.get("chunks") or data.shape)[0]
        for num, row in enumerate(range(0, len(data), step)):
            raw[row : row + step] = data[row : row + step]
            if gaps:  # flushed, so the chunk is placed before the gap
                h5.flush()
                h5[f"gaps/{num}"] = np.zeros(64)
                h5.flush()
        h5["time"] = start + np.arange(len(data), dtype=float)
        h5["distance"] = np.arange(data.shape[1])
        h5.attrs["dims"] = "time,distance"
    return path


def _stored(path, window):
    """What h5py gives for a window of the data node."""
    with h5py.File(path, "r") as h5:
        return h5["raw"][window]


@pytest.fixture
def stored_reads(monkeypatch):
    """Count the lazy loads which open the file through its reader."""
    calls, original = [], io_core._open_stored_reader

    def counted(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(io_core, "_open_stored_reader", counted)
    return calls


# The storage layouts of a (30, 5) array the offset reads cover.
LAYOUTS = {
    "contiguous": {},
    "chunked": {"chunks": (7, 5)},
    "gaps": {"chunks": (7, 5), "gaps": True},
}


class TestOffsetReads:
    """A plain dataset's bytes are read from its offsets, never opening HDF5."""

    @pytest.fixture(params=list(LAYOUTS))
    def path(self, tmp_path, request):
        """A file of each storage layout."""
        data = np.arange(150, dtype=np.float32).reshape(30, 5)
        return _write_simple(tmp_path / "a.h5", data, **LAYOUTS[request.param])

    @pytest.mark.parametrize(
        "window",
        [
            (slice(None),),
            (slice(3, 9),),
            (slice(5, 16),),
            (slice(25, 30),),
            (slice(4, 4),),
            (slice(2, 23), slice(1, 4)),
            (slice(None), slice(3, 5)),
        ],
    )
    @pytest.mark.parametrize("dtype", ["<f4", ">f4", ">i2"])
    def test_equals_hdf5(self, tmp_path, stored_reads, window, dtype):
        """Each window equals what h5py reads, dtype, byte order and all."""
        data = np.arange(150).reshape(30, 5).astype(dtype)
        for name, options in LAYOUTS.items():
            path = _write_simple(tmp_path / f"{name}.h5", data, **options)
            source = dc.read(path)[0]._source
            out, expected = source[window].load(), _stored(path, window)
            assert out.dtype == expected.dtype and out.flags.c_contiguous
            np.testing.assert_array_equal(out, expected)
        assert not stored_reads

    def test_gaps_between_chunks(self, tmp_path):
        """Chunks other storage separates are not read as one run."""
        data = np.ones((30, 5), np.float32)
        path = _write_simple(tmp_path / "a.h5", data, chunks=(7, 5), gaps=True)
        offsets = dc.read(path)[0]._source.offsets
        assert any(b - a != 7 * 5 * 4 for a, b in itertools.pairwise(offsets))

    def test_one_open_for_many_windows(self, path, stored_reads, monkeypatch):
        """A lazy array of several windows of one file opens it once."""
        source, opened = dc.read(path)[0]._source, []
        original = open

        def counted(*args, **kwargs):
            opened.append(args[0])
            return original(*args, **kwargs)

        monkeypatch.setattr("builtins.open", counted)
        array = LazyArray.from_sources([source[20:30], source[0:4], source[9:12]])
        expected = np.concatenate(
            [_stored(path, slice(*x)) for x in ((20, 30), (0, 4), (9, 12))]
        )
        np.testing.assert_array_equal(array.load(), expected)
        assert len(opened) == 1 and not stored_reads

    def test_narrow_window_uses_reader(self, tmp_path, stored_reads):
        """A window of few columns of a wide array is left to the reader."""
        data = np.arange(500, dtype=np.float32).reshape(10, 50)
        path = _write_simple(tmp_path / "wide.h5", data)
        source = dc.read(path)[0]._source
        np.testing.assert_array_equal(source[:, 3:8].load(), data[:, 3:8])
        assert len(stored_reads) == 1

    def test_rewritten_file_uses_reader(self, path, stored_reads):
        """A file changed since its offsets were taken is read by its reader."""
        source = dc.read(path)[0]._source
        data = -np.arange(150, dtype=np.float32).reshape(30, 5)
        _write_simple(path, data, chunks=(3, 5), gaps=True)
        np.testing.assert_array_equal(source[2:20].load(), data[2:20])
        assert len(stored_reads) == 1

    def test_offsets_past_the_end(self, path, stored_reads):
        """Offsets which find too few bytes leave the read to the reader."""
        source = dc.read(path)[0]._source
        source = replace(source, offsets=(source.size_bytes,) * len(source.offsets))
        np.testing.assert_array_equal(source.load(), _stored(path, slice(None)))
        assert len(stored_reads) == 1

    def test_missing_file(self, path):
        """A file gone since its offsets were taken fails as the reader fails."""
        source = dc.read(path)[0]._source
        path.unlink()
        with pytest.raises(FileNotFoundError):
            source.load()

    def test_offsets_are_not_identity(self, path):
        """Where the bytes are says nothing about which array a source is."""
        source = dc.read(path)[0]._source
        bare = replace(source, offsets=(), chunk_rows=0, mtime_ns=None, size_bytes=None)
        assert source.offsets and source == bare and hash(source) == hash(bare)
        assert source[2:5].data_id == bare[2:5].data_id
        assert source[2:5].offsets == source.offsets

    def test_round_trip_dict(self, path):
        """A source's dict gives back its offsets."""
        source = dc.read(path)[0]._source
        assert ArraySource.from_dict(source.to_dict()).offsets == source.offsets


class TestIndexedOffsetReads:
    """Offsets survive an index, so merged lazy loads open no HDF5 file."""

    @pytest.fixture
    def directory(self, tmp_path):
        """Contiguous and chunked files, end to end in time, indexed."""
        data = np.arange(6 * 20 * 4, dtype=np.float64).reshape(120, 4)
        for num in range(6):
            options = {"chunks": (6, 4), "gaps": num % 2} if num > 2 else {}
            part = data[num * 20 : (num + 1) * 20]
            _write_simple(tmp_path / f"{num}.h5", part, start=num * 20.0, **options)
        dc.spool(tmp_path).update()
        return tmp_path, data

    def test_merge(self, directory, stored_reads):
        """A reopened index merges every file from its offsets."""
        path, data = directory
        merged = dc.spool(path).chunk(time=None)[0]
        np.testing.assert_array_equal(merged.data, data)
        assert not stored_reads

    def test_trimmed_merge(self, directory, stored_reads):
        """A selection across files reads only its rows, from the offsets."""
        path, data = directory
        spool = (
            dc.spool(path)
            .select(time=tuple(dc.to_datetime64([15.5, 70.5])))
            .chunk(time=None)
        )
        np.testing.assert_array_equal(spool[0].data, data[16:71])
        assert not stored_reads

    def test_rewritten_file(self, directory, stored_reads):
        """A file rewritten after indexing is not read from stale offsets."""
        path, data = directory
        spool = dc.spool(path)
        _write_simple(path / "4.h5", -data[80:100], start=80.0, chunks=(3, 4))
        merged = spool.chunk(time=None)[0]
        np.testing.assert_array_equal(merged.data[80:100], -data[80:100])
