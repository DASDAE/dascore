"""Tests for the ZARR format (CF-on-zarr through xarray)."""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import ClassVar
from unittest import mock

import fsspec
import numpy as np
import pandas as pd
import pytest
import upath
from fsspec.implementations.local import LocalFileSystem
from fsspec.implementations.memory import MemoryFileSystem
from upath import UPath
from upath.implementations.memory import MemoryPath

import dascore as dc
from dascore.exceptions import (
    MissingOptionalDependencyError,
    ParameterError,
    PatchAttributeError,
)
from dascore.io import core as io_core
from dascore.io.core import source_identity
from dascore.io.index.catalog import PatchCatalog
from dascore.io.index.indexer import scan_unit_stats
from dascore.io.netcdf.utils import COORD_PREFIX, STEP_ATTR
from dascore.io.zarr import ZarrV3
from dascore.io.zarr import core as zarr_core
from dascore.utils.misc import suppress_warnings

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

VERSIONS = ("3", "2")
# Present wherever a coordinate's grid is described beside its labels.
RANGE_KEY = f"{COORD_PREFIX}length"


@pytest.fixture(scope="module")
def zarr_patch():
    """A patch with a 1024 Hz time grid, a non-dim coord and a bool attr."""
    patch = dc.get_example_patch()
    ntime = patch.shape[patch.get_axis("time")]
    start = np.datetime64("2020-01-01T00:00:00.123456789")
    time = start + (np.arange(ntime) * 1e9 / 1024).astype("timedelta64[ns]")
    latitude = np.linspace(40.0, 41.0, patch.shape[patch.get_axis("distance")])
    patch = patch.update_coords(time=time, latitude=("distance", latitude))
    return patch.update_attrs(closed_fiber_loop=True, station="BOB")


@pytest.fixture(scope="module", params=VERSIONS)
def written(request, zarr_patch, tmp_path_factory):
    """Return (version, path) of the patch written as each zarr format."""
    path = tmp_path_factory.mktemp("zarr") / "patch.zarr"
    dc.write(zarr_patch, path, "zarr", file_version=request.param)
    return request.param, path


@pytest.fixture(scope="module", params=VERSIONS)
def foreign_store(request, tmp_path_factory):
    """A store xarray writes with CF time in float seconds and a scaled payload."""
    path = tmp_path_factory.mktemp("zarr") / "foreign.zarr"
    time = np.datetime64("2021-01-01") + np.arange(20) * np.timedelta64(250, "ms")
    data = np.arange(200, dtype=np.float64).reshape(10, 20) / 10
    dataset = xr.Dataset(
        {"data": (("distance", "time"), data)},
        coords={"distance": np.arange(10) * 2.0, "time": time},
    )
    encoding = {
        "data": {"dtype": "int16", "scale_factor": 0.1, "_FillValue": -1},
        "time": {"dtype": "float64", "units": "seconds since 2021-01-01"},
    }
    zarr_format = int(request.param)
    dataset.to_zarr(
        path, zarr_format=zarr_format, consolidated=False, encoding=encoding
    )
    return path, data, time


def _edit_attrs(path, **attrs):
    """Set payload attrs in place and consolidate, as another zarr tool would."""
    zarr.open_group(path, mode="r+")["data"].attrs.update(attrs)
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        zarr.consolidate_metadata(path)


@pytest.fixture
def chunked_store(zarr_patch, tmp_path):
    """A store of hundreds of chunk files beside a DASDAE file."""
    path = tmp_path / "a.zarr"
    dc.write(zarr_patch, path, "zarr", encoding={"data": {"chunks": (10, 100)}})
    dc.write(zarr_patch, tmp_path / "b.h5", "dasdae")
    return path


class TestRoundTrip:
    """A written patch reads back unchanged."""

    def test_patch_equal(self, written, zarr_patch):
        """Data, coords (incl. the non-dim one) and attrs round trip."""
        _, path = written
        patch = dc.read(path)[0]
        assert patch == zarr_patch

    def test_fractional_time_grid(self, written, zarr_patch):
        """The 1024 Hz grid comes back exact, to the nanosecond."""
        _, path = written
        coord = dc.read(path)[0].get_coord("time")
        expected = zarr_patch.get_coord("time")
        assert coord.step == expected.step
        assert np.array_equal(coord.values, expected.values)

    def test_bool_attr(self, written):
        """A bool attr is stored as a bool, not an int."""
        _, path = written
        assert dc.read(path)[0].attrs.closed_fiber_loop is True

    def test_consolidated(self, written):
        """The store carries consolidated metadata."""
        _, path = written
        assert zarr.open_consolidated(path, mode="r")

    @pytest.mark.parametrize("version", VERSIONS)
    def test_data_units(self, zarr_patch, tmp_path, version):
        """Data units are written as a string and read back as units."""
        patch = zarr_patch.set_units("m/s")
        path = tmp_path / "units.zarr"
        dc.write(patch, path, "zarr", file_version=version)
        assert dc.read(path)[0] == patch

    @pytest.mark.parametrize("version", VERSIONS)
    def test_partial_coords(self, tmp_path, version):
        """A dimension without a coordinate is rebuilt beside one with it."""
        path = tmp_path / "partial.zarr"
        dataset = xr.Dataset(
            {"data": (("distance", "time"), np.zeros((3, 4)))},
            coords={"time": np.arange(4.0) / 2},
        )
        dataset.to_zarr(path, zarr_format=int(version), consolidated=False)
        patch = dc.read(path)[0]
        assert np.array_equal(patch.get_array("distance"), np.arange(3))
        assert np.array_equal(patch.get_array("time"), np.arange(4.0) / 2)

    @pytest.mark.parametrize("version", VERSIONS)
    def test_cf_units(self, tmp_path, version):
        """Data units come from ``units``; unknown coord units are dropped."""
        path = tmp_path / "cf.zarr"
        dataset = xr.Dataset(
            {"data": (("distance", "time"), np.zeros((3, 4)), {"units": "s-1"})},
            coords={
                "time": np.arange(4.0),
                "latitude": ("distance", np.arange(3.0), {"units": "bogus"}),
            },
        )
        dataset.to_zarr(path, zarr_format=int(version), consolidated=False)
        with pytest.warns(UserWarning, match="latitude.*bogus"):
            patch = dc.read(path)[0]
        assert patch.attrs.data_units == dc.get_quantity("1/s")
        assert "units" not in patch.attrs.model_dump()
        assert patch.get_coord("latitude").units is None


class TestEncoding:
    """Storage options pass through to the store as xarray's encoding."""

    def test_chunks_shards_compressors(self, zarr_patch, tmp_path):
        """Chunks, shards and compressors land on the v3 payload array."""
        path = tmp_path / "enc.zarr"
        compressor = zarr.codecs.ZstdCodec(level=5)
        encoding = {
            "data": {
                "chunks": (50, 500),
                "shards": (100, 1000),
                "compressors": (compressor,),
            }
        }
        dc.write(zarr_patch, path, "zarr", encoding=encoding)
        array = zarr.open_group(path, mode="r")["data"]
        assert array.chunks == (50, 500)
        assert array.shards == (100, 1000)
        assert array.compressors == (compressor,)
        assert dc.read(path)[0] == zarr_patch


class TestForeignStore:
    """A store written by xarray alone reads with CF decoding applied."""

    def test_read_decodes(self, foreign_store):
        """Scaling and float-seconds time decode on read."""
        path, data, time = foreign_store
        patch = dc.read(path)[0]
        assert np.allclose(patch.data, data)
        assert np.array_equal(patch.get_array("time"), time)

    def test_named_version_believed(self, foreign_store):
        """A store opens under either zarr version named, as its marker says."""
        path, data, _ = foreign_store
        for version in VERSIONS:
            patch = dc.read(path, file_format="ZARR", file_version=version)[0]
            assert np.allclose(patch.data, data)

    def test_read_array_decodes(self, foreign_store):
        """read_array applies the scale factor as read does."""
        path, data, _ = foreign_store
        out = ZarrV3().read_array(path, ((1, 3), None))
        assert np.allclose(out, data[1:3])


class TestGetVersion:
    """Only a zarr store holding a dimensioned payload is claimed."""

    def test_scalar_payload(self, tmp_path):
        """A store whose only variable has no dimension is not ours."""
        path = tmp_path / "scalar.zarr"
        xr.Dataset({"a": ((), 1.0)}).to_zarr(path, zarr_format=3, consolidated=False)
        assert ZarrV3().get_version(path) is None

    def test_missing_zarr(self, zarr_patch, tmp_path, monkeypatch):
        """Without zarr a store is still claimed; reading it names zarr."""
        path = tmp_path / "patch.zarr"
        dc.write(zarr_patch, path, "zarr")
        dc.write(zarr_patch, tmp_path / "patch.h5", "dasdae")
        monkeypatch.setitem(sys.modules, "zarr", None)
        assert dc.get_format(path)[0] == "ZARR"
        with pytest.raises(MissingOptionalDependencyError, match="zarr"):
            dc.read(path)
        with pytest.warns(UserWarning, match="zarr"):
            (summary,) = dc.scan(tmp_path)
        assert summary.source_format == "DASDAE"


class TestDirectorySpool:
    """A directory of stores indexes each store as one source."""

    def test_two_stores(self, zarr_patch, tmp_path):
        """Two stores give two patches and nothing from their chunk files."""
        dc.write(zarr_patch, tmp_path / "a.zarr", "zarr")
        other = zarr_patch.update_coords(
            distance=zarr_patch.get_array("distance") + 1e4
        )
        dc.write(other, tmp_path / "b.zarr", "zarr", file_version="2")
        spool = dc.spool(tmp_path).update()
        contents = spool.get_contents()
        paths = sorted(str(x).rsplit("/", 1)[-1] for x in contents["source_path"])
        assert paths == ["a.zarr", "b.zarr"]
        selected = spool.select(distance=(1e4, None))
        assert len(selected) == 1
        assert selected[0] == other

    @pytest.mark.parametrize("version", VERSIONS)
    def test_attrs_edit_refreshes(self, zarr_patch, tmp_path, version):
        """An attrs-only edit, in v2's hidden files or v3's, is re-indexed."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_patch, path, "zarr", file_version=version)
        spool = dc.spool(tmp_path).update()
        _edit_attrs(path, station="NEW")
        assert spool.update().get_contents()["station"].tolist() == ["NEW"]

    def test_rewrite_refreshes(self, zarr_patch, tmp_path):
        """A store dascore writes again is re-indexed."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_patch, path, "zarr")
        spool = dc.spool(tmp_path).update()
        dc.write(zarr_patch.update_attrs(station="NEW"), path, "zarr")
        assert spool.update().get_contents()["station"].tolist() == ["NEW"]

    def test_chunks_not_walked(self, chunked_store):
        """A store is identified by its metadata files, not by its chunks."""
        with mock.patch.object(Path, "rglob", side_effect=AssertionError):
            assert None not in scan_unit_stats(chunked_store)
            assert None not in source_identity(chunked_store)

    def test_stray_marker_name_is_not_a_store(self, tmp_path):
        """A folder holding a zarr metadata name, but no marker, is walked whole."""
        (tmp_path / ".zattrs").write_text("{}")
        (tmp_path / "a.raw").write_bytes(b"1")
        before = scan_unit_stats(tmp_path)
        (tmp_path / "b.raw").write_bytes(b"2")
        assert scan_unit_stats(tmp_path) != before

    def test_store_spool_writes_no_index(self, zarr_patch, tmp_path):
        """A spool of one store keeps its index outside, writing nothing into it."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_patch, path, "zarr", file_version="2")
        before = sorted(path.rglob("*"))
        assert dc.spool(path).update()[0] == zarr_patch
        assert sorted(path.rglob("*")) == before

    def test_second_update_not_a_change(self, zarr_patch, tmp_path):
        """A store spool's second update does not re-index the store."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_patch, path, "zarr", file_version="2")
        spool = dc.spool(path).update()
        sources = spool.indexer._backend.get_sources
        before = sources()["last_indexed_ns"].max()
        spool.update()
        assert sources()["last_indexed_ns"].max() == before

    def test_rewrite_under_open_spool(self, zarr_patch, tmp_path):
        """A store rewritten under an open spool is re-indexed on update."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_patch, path, "zarr")
        spool = dc.spool(path).update()
        later = zarr_patch.update_coords(time_min=np.datetime64("2030-01-01"))
        dc.write(later, path, "zarr")
        assert spool.update()[0] == later


class TestScanDirectory:
    """A scan of a directory takes each store whole, and survives a bad one."""

    def test_sizing_skips_chunks(self, chunked_store, monkeypatch):
        """The progress total counts a store as one and never lists its chunks."""
        lengths, listed = [], []
        track, scandir = io_core.track, os.scandir

        def _track(*args, length, **kwargs):
            lengths.append(length)
            return track(*args, length=length, **kwargs)

        def _scandir(path):
            listed.append(Path(path))
            return scandir(path)

        monkeypatch.setattr(io_core, "track", _track)
        monkeypatch.setattr(os, "scandir", _scandir)
        assert len(dc.scan(chunked_store.parent)) == 2
        assert lengths == [2]
        assert chunked_store not in listed

    def test_invalid_attr_warns(self, chunked_store):
        """A store whose attrs fail validation is skipped, not fatal."""
        _edit_attrs(chunked_store, acquisition_key="bad key")
        with pytest.warns(UserWarning, match="Failed to scan"):
            (summary,) = dc.scan(chunked_store.parent)
        assert summary.source_format == "DASDAE"


def _edit_ranges(path, edit):
    """Apply ``edit`` to each array's grid description, then consolidate."""
    for _, array in zarr.open_group(path, mode="r+").arrays():
        if RANGE_KEY in array.attrs:
            edit(array.attrs)
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        zarr.consolidate_metadata(path)


class TestRangeAttr:
    """A grid is described beside its labels, so a scan never reads them."""

    def test_scan_reads_no_labels(self, zarr_patch):
        """Scanning reads metadata only, and gets the patch's coordinates."""
        fsspec.register_implementation("zarrcount", CountMemory, clobber=True)
        upath.registry.register_implementation("zarrcount", MemoryPath, clobber=True)
        path = UPath("zarrcount://dascore_zarr_count/range.zarr")
        names = ("time", "distance", "latitude")
        encoding = {x: {"chunks": (100,)} for x in names}
        dc.write(zarr_patch, path, "zarr", encoding=encoding)
        CountMemory.read.clear()
        assert len(dc.scan(path)) == 1
        (meta,) = ZarrV3().get_metadata(path)
        assert not [x for x in CountMemory.read if "/c/" in x]
        assert meta.coords == zarr_patch.coords
        # a read loads payload chunks only
        assert dc.read(path)[0] == zarr_patch
        assert not [x for x in CountMemory.read if any(f"/{n}/c/" in x for n in names)]

    def test_labels_still_written(self, written, zarr_patch):
        """A CF tool reads the labels, beside the description it ignores."""
        _, path = written
        dataset = xr.open_dataset(path, engine="zarr")
        assert RANGE_KEY in dataset["time"].attrs
        assert np.array_equal(dataset["time"].values, zarr_patch.get_array("time"))

    def test_store_without_ranges(self, zarr_patch, tmp_path):
        """A store written before the description reads its labels, as before."""
        path = tmp_path / "old.zarr"
        dc.write(zarr_patch, path, "zarr")

        def _strip(attrs):
            for name in [x for x in attrs if x.startswith(COORD_PREFIX)]:
                del attrs[name]

        _edit_ranges(path, _strip)
        assert dc.read(path)[0] == zarr_patch

    def test_stale_range_ignored(self, zarr_patch, tmp_path):
        """A description of another length than the labels is not believed."""
        path = tmp_path / "stale.zarr"
        dc.write(zarr_patch, path, "zarr")

        def _shorten(attrs):
            attrs[RANGE_KEY] = 3

        _edit_ranges(path, _shorten)
        assert dc.read(path)[0] == zarr_patch

    def test_float32_grid(self, zarr_patch, tmp_path):
        """A float32 grid, which JSON cannot describe exactly, reads its labels."""
        path = tmp_path / "f32.zarr"
        size = zarr_patch.shape[zarr_patch.get_axis("distance")]
        distance = np.arange(size, dtype=np.float32) * np.float32(1.02)
        patch = zarr_patch.update_coords(distance=distance)
        dc.write(patch, path, "zarr")
        out = dc.read(path)[0]
        assert out == patch
        assert out.get_array("distance").tobytes() == distance.tobytes()

    def test_coord_encoding_not_described(self, zarr_patch, tmp_path):
        """A coordinate encoded to other labels reads them, as a CF tool does."""
        path = tmp_path / "enc.zarr"
        size = zarr_patch.shape[zarr_patch.get_axis("distance")]
        patch = zarr_patch.update_coords(distance=np.arange(size) + 0.25)
        encoding = {
            "distance": {"dtype": "int32"},
            "latitude": {"compressors": ()},  # codecs may be lossy
            "time": {"chunks": (100,)},
        }
        dc.write(patch, path, "zarr", encoding=encoding)
        group = zarr.open_group(path, mode="r")
        assert RANGE_KEY not in group["distance"].attrs
        assert RANGE_KEY not in group["latitude"].attrs
        assert RANGE_KEY in group["time"].attrs
        stored = xr.open_dataset(path, engine="zarr")["distance"].values
        assert np.array_equal(dc.read(path)[0].get_array("distance"), stored)

    def test_spool_loaded_once(self, zarr_patch, tmp_path):
        """A file spool's one patch is loaded once to be written."""
        dc.write(zarr_patch, tmp_path / "a.h5", "dasdae")
        spool = dc.spool(tmp_path / "a.h5")
        resolve = PatchCatalog.resolve_row
        with mock.patch.object(
            PatchCatalog, "resolve_row", autospec=True, side_effect=resolve
        ) as spy:
            dc.write(spool, tmp_path / "a.zarr", "zarr")
        assert spy.call_count == 1

    def test_holes_keep_step(self, zarr_patch, tmp_path):
        """Labels with holes store only their step, and read back with them."""
        path = tmp_path / "holes.zarr"
        patch = zarr_patch.unselect(distance=(10, 20), samples=True)
        dc.write(patch, path, "zarr")
        attrs = zarr.open_group(path, mode="r")["distance"].attrs
        assert RANGE_KEY not in attrs and STEP_ATTR in attrs
        assert dc.read(path)[0] == patch


class TestRemote:
    """xarray takes a UPath, so an fsspec store works as a local one."""

    def test_memory_round_trip(self, zarr_patch):
        """A memory:// store writes, scans and reads back."""
        path = UPath("memory://dascore_zarr_test/patch.zarr")
        dc.write(zarr_patch, path, "zarr")
        assert dc.read(path)[0] == zarr_patch
        (summary,) = dc.scan(path.parent)
        assert summary.source_format == "ZARR"

    def test_memory_spool_derived(self, zarr_patch):
        """A memory:// store is a file path, not a live patch, when derived."""
        path = UPath("memory://dascore_zarr_test/derived.zarr")
        dc.write(zarr_patch, path, "zarr")
        spool = dc.spool(path)
        assert spool.chunk(time=None)[0] == zarr_patch
        assert all(x == zarr_patch for x in spool + dc.spool([zarr_patch]))

    def test_window_read_skips_coords(self, zarr_patch):
        """A windowed read loads no other coordinate chunk than time's ends."""
        fsspec.register_implementation("zarrcount", CountMemory, clobber=True)
        upath.registry.register_implementation("zarrcount", MemoryPath, clobber=True)
        path = UPath("zarrcount://dascore_zarr_count/window.zarr")
        encoding = {"time": {"chunks": (100,)}, "distance": {"chunks": (100,)}}
        dc.write(zarr_patch, path, "zarr", encoding=encoding)
        CountMemory.read.clear()
        out = ZarrV3().read_array(path, windows=((0, 5), (0, 5)))
        assert np.array_equal(out, zarr_patch.data[:5, :5])
        # the "/x/c/" filters below match real chunk keys
        assert [x for x in CountMemory.read if "/data/c/" in x]
        last = (zarr_patch.shape[zarr_patch.get_axis("time")] - 1) // 100
        time = {x for x in CountMemory.read if "/time/c/" in x}
        assert {x.rsplit("/", 1)[-1] for x in time} <= {"0", str(last)}
        others = ("/distance/c/", "/latitude/c/")
        assert not [x for x in CountMemory.read if any(o in x for o in others)]


class TestReplace:
    """A write replaces a store, or nothing, and never half of one."""

    @pytest.mark.parametrize("remote", [False, True])
    def test_rewrite_replaces_store(self, zarr_patch, tmp_path, remote):
        """A second write, of another shape, leaves only the second patch."""
        base = UPath("memory://dascore_zarr_replace") if remote else tmp_path
        path = base / "patch.zarr"
        dc.write(zarr_patch, path, "zarr")
        small = zarr_patch.select(time=(None, 10), samples=True)
        dc.write(small, path, "zarr")
        assert dc.read(path)[0] == small
        assert [x.name for x in base.iterdir()] == ["patch.zarr"]

    def test_refuses_other_directory(self, zarr_patch, tmp_path):
        """A directory which is not a store is refused and left alone."""
        (tmp_path / "keep.txt").write_text("keep")
        with pytest.raises(FileExistsError, match="not a zarr store"):
            dc.write(zarr_patch, tmp_path, "zarr")
        assert [x.name for x in tmp_path.iterdir()] == ["keep.txt"]

    def test_refuses_file(self, zarr_patch, tmp_path):
        """A file at the path is refused and left alone."""
        path = tmp_path / "patch.zarr"
        path.write_text("keep")
        with pytest.raises(FileExistsError, match="not a zarr store"):
            dc.write(zarr_patch, path, "zarr")
        assert path.read_text() == "keep"

    def test_failed_write_keeps_store(self, zarr_patch, tmp_path, monkeypatch):
        """A write failing after its store is built leaves the old store alone."""
        path = tmp_path / "patch.zarr"
        dc.write(zarr_patch, path, "zarr")
        to_zarr = xr.Dataset.to_zarr

        def _fail(self, *args, **kwargs):
            to_zarr(self, *args, **kwargs)
            raise RuntimeError("killed")

        monkeypatch.setattr(xr.Dataset, "to_zarr", _fail)
        with pytest.raises(RuntimeError, match="killed"):
            dc.write(zarr_patch.update_attrs(station="NEW"), path, "zarr")
        assert dc.read(path)[0] == zarr_patch
        assert [x.name for x in tmp_path.iterdir()] == ["patch.zarr"]

    def test_failed_promotion_restores_store(self, zarr_patch, tmp_path, monkeypatch):
        """A failed move of the new store into place moves the old one back."""
        path = tmp_path / "patch.zarr"
        dc.write(zarr_patch, path, "zarr")
        mv = LocalFileSystem.mv

        def _fail(self, source, target, **kwargs):
            if str(source).endswith(".partial"):
                raise OSError("no space")
            return mv(self, source, target, **kwargs)

        monkeypatch.setattr(LocalFileSystem, "mv", _fail)
        with pytest.raises(OSError, match="no space"):
            dc.write(zarr_patch.update_attrs(station="NEW"), path, "zarr")
        assert dc.read(path)[0] == zarr_patch

    def test_partial_promotion_restores_store(self, zarr_patch, tmp_path, monkeypatch):
        """A move that copies part of the new store before failing is undone."""
        path = tmp_path / "patch.zarr"
        dc.write(zarr_patch, path, "zarr")
        mv = LocalFileSystem.mv

        def _fail(self, source, target, **kwargs):
            if str(source).endswith(".partial"):
                mv(self, source, target, **kwargs)  # the copy lands, then fails
                raise OSError("interrupted")
            return mv(self, source, target, **kwargs)

        monkeypatch.setattr(LocalFileSystem, "mv", _fail)
        with pytest.raises(OSError, match="interrupted"):
            dc.write(zarr_patch.update_attrs(station="NEW"), path, "zarr")
        assert dc.read(path)[0] == zarr_patch
        assert not (path / ".patch.zarr.old").exists()

    def test_killed_write_leftover(self, zarr_patch, tmp_path):
        """Stores a killed write leaves behind are not scanned, then cleared."""
        path = tmp_path / "patch.zarr"
        dc.write(zarr_patch, path, "zarr")
        for name in (".patch.zarr.partial", ".patch.zarr.old"):
            ZarrV3().write(zarr_patch, tmp_path / name)
        assert len(dc.scan(tmp_path)) == 1
        dc.write(zarr_patch, path, "zarr")
        assert [x.name for x in tmp_path.iterdir()] == ["patch.zarr"]


@pytest.fixture(scope="module")
def zarr_spool(zarr_patch):
    """Two patches apart in time, and a third named as the first; data differ."""
    time = zarr_patch.get_coord("time")
    later = zarr_patch.update_coords(time_min=time.max() + np.timedelta64(1, "s"))
    later = later.update(data=later.data * 2)
    distance = zarr_patch.get_array("distance") + 1e4
    twin = zarr_patch.update_coords(distance=distance).update_attrs(station="SUE")
    twin = twin.update(data=twin.data + 1)
    return dc.spool([zarr_patch, later, twin])


@pytest.fixture(scope="module", params=VERSIONS)
def multi_written(request, zarr_spool, tmp_path_factory):
    """Return (path, names) of the spool written as each zarr format."""
    path = tmp_path_factory.mktemp("zarr") / "spool.zarr"
    dc.write(zarr_spool, path, "zarr", file_version=request.param)
    return path, list(zarr_spool.get_patch_names())


def _store(path, groups, version="3"):
    """Write a store of xarray datasets by group path, "/" being the root."""
    tree = xr.DataTree.from_dict(groups)
    with suppress_warnings(UserWarning, message="Consolidated metadata"):
        tree.to_zarr(path, mode="w", zarr_format=int(version))
    return path


class AuthMemory(MemoryFileSystem):
    """A memory filesystem which needs a token, as an object store might."""

    protocol = ("zarrauth",)

    def __init__(self, *args, token=None, **kwargs):
        if token != "secret":
            raise PermissionError("no token")
        super().__init__(*args, **kwargs)


class CountMemory(MemoryFileSystem):
    """A memory filesystem which records the objects read from it."""

    protocol = ("zarrcount",)
    read: ClassVar[list[str]] = []

    def cat_file(self, path, start=None, end=None, **kwargs):
        """Record the path, then read it."""
        self.read.append(path)
        return super().cat_file(path, start, end, **kwargs)


class TestMultiPatch:
    """A spool of several patches is one group per patch, round tripped."""

    def test_round_trip(self, multi_written, zarr_spool):
        """Each patch reads back equal, in name order."""
        path, _ = multi_written
        first, later, twin = zarr_spool
        assert list(dc.read(path)) == [first, twin, later]

    def test_keys(self, multi_written):
        """Patch keys are the group names; a repeated name is suffixed."""
        path, names = multi_written
        keys = sorted(x.source_patch_key for x in dc.scan(path))
        assert names[0] == names[2]
        assert keys == sorted([names[0], names[1], f"{names[0]}__1"])

    def test_datatree(self, multi_written, zarr_spool):
        """Xarray opens the store as a tree with one node per patch."""
        path, _ = multi_written
        tree = xr.open_datatree(path, engine="zarr")
        assert len(tree.children) == len(zarr_spool)
        for node in tree.children.values():
            assert node["data"].dims == zarr_spool[0].dims

    def test_select_reads_one_group(self, zarr_spool, tmp_path):
        """Loading one patch never reads another patch's payload chunks."""
        path = tmp_path / "spool.zarr"
        dc.write(zarr_spool, path, "zarr")
        spool = dc.spool(path).update()
        _, later, _ = zarr_spool
        selected = spool.select(time=(later.get_coord("time").min(), None))
        (key,) = selected.get_contents()["source_patch_key"]
        others = set(zarr.open_group(path, mode="r").group_keys()) - {key}
        for chunk in (x for g in others for x in (path / g / "data").rglob("*")):
            if chunk.is_file() and chunk.name != "zarr.json":
                chunk.write_bytes(b"not a chunk")
        assert selected[0] == later
        with pytest.raises(RuntimeError):  # the spoiled chunks are unreadable
            dc.read(path)

    def test_keyed_read_skips_listing(self, multi_written, zarr_spool, monkeypatch):
        """A keyed read opens its group without listing the others."""
        path, names = multi_written
        monkeypatch.setattr(zarr_core, "_patches", None)
        out = ZarrV3().read_array(path, key=names[1])
        assert np.array_equal(out, zarr_spool[1].data)

    def test_stale_consolidated_range(self, zarr_spool, tmp_path):
        """A group's own metadata wins over an older consolidated copy."""
        path = tmp_path / "stale.zarr"
        dc.write(zarr_spool, path, "zarr")
        root = zarr.open_group(path, mode="r+", use_consolidated=False)
        for _, group in root.groups():
            attrs = group["time"].attrs  # not consolidated after
            attrs[RANGE_KEY] = 3
        first, later, twin = zarr_spool
        assert list(dc.read(path)) == [first, twin, later]

    def test_key_required(self, multi_written):
        """read_array needs a key, and a known one, on a multi-patch store."""
        path, _ = multi_written
        with pytest.raises(PatchAttributeError, match="several patches"):
            ZarrV3().read_array(path)
        with pytest.raises(PatchAttributeError, match="No patch named"):
            ZarrV3().read_array(path, key="bob")

    def test_encoding_each_group(self, zarr_spool, tmp_path):
        """One per-variable encoding lands on every patch's payload."""
        path = tmp_path / "enc.zarr"
        dc.write(zarr_spool, path, "zarr", encoding={"data": {"chunks": (50, 500)}})
        group = zarr.open_group(path, mode="r")
        chunks = {group[x]["data"].chunks for x in group.group_keys()}
        assert chunks == {(50, 500)}

    def test_slash_in_name(self, zarr_spool, tmp_path):
        """A "/" in a patch name makes no nested group; the patch round trips."""
        spool = dc.spool([x.update_attrs(tag="a/b") for x in zarr_spool])
        path = tmp_path / "slash.zarr"
        dc.write(spool, path, "zarr")
        assert all("/" not in x.source_patch_key for x in dc.scan(path))
        assert len(dc.read(path)) == len(spool)

    def test_empty_names(self, zarr_spool, tmp_path, monkeypatch):
        """Empty names (e.g. from a source named ".h5") do not name the root."""
        blank = pd.Series([""] * len(zarr_spool))
        monkeypatch.setattr(zarr_core, "get_patch_names", lambda _: blank)
        path = tmp_path / "blank.zarr"
        dc.write(zarr_spool, path, "zarr")
        assert len(dc.read(path)) == len(zarr_spool)

    def test_raw_zarr_group_skipped(self, zarr_spool, tmp_path):
        """A child group of plain zarr arrays is not a patch and is skipped."""
        path = tmp_path / "raw.zarr"
        dc.write(zarr_spool, path, "zarr")
        group = zarr.open_group(path, mode="a")
        group.create_group("raw").create_array("values", data=np.arange(3))
        zarr.consolidate_metadata(path)
        assert len(dc.read(path)) == len(zarr_spool)
        with pytest.raises(PatchAttributeError, match="No patch named"):
            ZarrV3().read_array(path, key="raw")

    def test_streams(self, zarr_spool, tmp_path, monkeypatch):
        """Each patch is converted and written before the next is converted."""
        events = []
        to_cf, to_zarr = zarr_core.spool_to_cf_dataset, xr.Dataset.to_zarr

        def _to_cf(*args):
            events.append("convert")
            return to_cf(*args)

        def _to_zarr(*args, **kwargs):
            events.append("write")
            return to_zarr(*args, **kwargs)

        monkeypatch.setattr(zarr_core, "spool_to_cf_dataset", _to_cf)
        monkeypatch.setattr(xr.Dataset, "to_zarr", _to_zarr)
        dc.write(zarr_spool, tmp_path / "s.zarr", "zarr")
        assert events == ["convert", "write"] * len(zarr_spool)

    def test_leftover_groups_cleared(self, zarr_spool, tmp_path):
        """Groups a killed write left in the staging store are not kept."""
        path = tmp_path / "spool.zarr"
        old = dc.spool([x.update_attrs(tag="old") for x in zarr_spool])
        ZarrV3().write(old, tmp_path / ".spool.zarr.partial")
        dc.write(zarr_spool, path, "zarr")
        assert len(dc.read(path)) == len(zarr_spool)

    def test_rewrite_refreshes(self, zarr_spool, zarr_patch, tmp_path):
        """A grouped store rewritten as one patch is re-indexed."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_spool, path, "zarr")
        spool = dc.spool(tmp_path).update()
        assert len(spool) == 3
        dc.write(zarr_patch, path, "zarr")
        assert list(spool.update()) == [zarr_patch]

    def test_storage_options(self, zarr_spool):
        """A remote store is listed with its path's storage options."""
        fsspec.register_implementation("zarrauth", AuthMemory, clobber=True)
        upath.registry.register_implementation("zarrauth", MemoryPath, clobber=True)
        path = UPath("zarrauth://dascore_zarr_auth/spool.zarr", token="secret")
        dc.write(zarr_spool, path, "zarr")
        assert len(dc.read(path)) == len(zarr_spool)

    @pytest.mark.parametrize(
        ("name", "file_format"),
        [("auth_spool.zarr", "zarr"), ("auth_spool.h5", "dasdae")],
    )
    def test_spool_loads_with_storage_options(self, zarr_spool, name, file_format):
        """A remote spool's patches load with the path's options, also derived."""
        fsspec.register_implementation("zarrauth", AuthMemory, clobber=True)
        upath.registry.register_implementation("zarrauth", MemoryPath, clobber=True)
        path = UPath(f"zarrauth://dascore_zarr_auth/{name}", token="secret")
        dc.write(zarr_spool, path, file_format)
        spool = dc.spool(path)
        _, later, twin = zarr_spool
        assert spool.select(station="SUE")[0] == twin
        start = later.get_coord("time").min()
        assert spool.select(time=(start, None), station="BOB")[0] == later
        assert all(x.data.size for x in spool.chunk(time=1))
        assert all(x.data.size for x in spool + dc.spool([twin]))
        # A mixed view sent to another process keeps the options too.
        view = pickle.loads(pickle.dumps((spool + dc.spool([twin])).select()))
        assert all(x.data.size for x in view)

    def test_scan_reads_root_once(self, zarr_spool):
        """A scan parses the consolidated root once, not once per group."""
        fsspec.register_implementation("zarrcount", CountMemory, clobber=True)
        upath.registry.register_implementation("zarrcount", MemoryPath, clobber=True)
        path = UPath("zarrcount://dascore_zarr_count/root_once.zarr")
        dc.write(zarr_spool, path, "zarr")
        CountMemory.read.clear()
        assert len(ZarrV3().get_metadata(path)) == len(zarr_spool)
        assert (
            sum(x.endswith("root_once.zarr/zarr.json") for x in CountMemory.read) == 1
        )

    def test_remote_spool_reads_one_group(self, zarr_spool):
        """
        A remote store is a spool; scanning reads no payload chunk, and
        loading one patch reads only that patch's.
        """
        fsspec.register_implementation("zarrcount", CountMemory, clobber=True)
        upath.registry.register_implementation("zarrcount", MemoryPath, clobber=True)
        path = UPath("zarrcount://dascore_zarr_count/spool.zarr")
        dc.write(zarr_spool, path, "zarr", encoding={"data": {"chunks": (10, 100)}})
        CountMemory.read.clear()
        spool = dc.spool(path)
        assert not [x for x in CountMemory.read if "/data/c/" in x]
        # A format 3 store is not probed for format 2 metadata files.
        assert not [x for x in CountMemory.read if "/.z" in x]
        _, later, _ = zarr_spool
        selected = spool.select(time=(later.get_coord("time").min(), None))
        (key,) = selected.get_contents()["source_patch_key"]
        assert selected[0] == later
        groups = {x.split("/")[-5] for x in CountMemory.read if "/data/c/" in x}
        assert groups == {key}


class TestForeignGroups:
    """Stores other tools write: every group, at any depth, holds patches."""

    @pytest.mark.parametrize("version", VERSIONS)
    def test_nested_groups(self, tmp_path, version):
        """Payloads at the root and in nested groups are each a patch."""
        data = xr.Dataset({"data": (("distance", "time"), np.ones((3, 4)))})
        aux = xr.Dataset({"temp": ("sensor", np.arange(3.0))})
        groups = {"/": data, "/aux": aux, "/meta": xr.Dataset(attrs={"a": 1})}
        path = _store(tmp_path / "s.zarr", groups | {"/x/y": data * 2}, version)
        keys = sorted(x.source_patch_key for x in dc.scan(path))
        assert keys == ["aux", "data", "x/y"]
        assert ZarrV3().read_array(path, key="x/y").sum() == 24

    @pytest.mark.parametrize("version", VERSIONS)
    def test_groups_without_payload_skipped(self, tmp_path, version):
        """Child groups holding no payload are not patches."""
        data = xr.Dataset({"data": (("distance", "time"), np.ones((3, 4)))})
        meta = xr.Dataset(attrs={"a": 1})
        groups = {"/a_meta": meta, "/p1": data, "/p2": data * 2, "/z_meta": meta}
        groups["/coords"] = xr.Dataset(coords={"x": np.arange(3)})
        x = ("x", np.arange(3.0), {"bounds": "x_bnds"})
        bounds = {"x_bnds": (("x", "nv"), np.zeros((3, 2)))}
        groups["/bounds"] = xr.Dataset(bounds, coords={"x": x})
        path = _store(tmp_path / "s.zarr", groups, version)
        assert sorted(x.source_patch_key for x in dc.scan(path)) == ["p1", "p2"]
        assert len(dc.read(path)) == 2


@pytest.fixture(scope="module", params=VERSIONS)
def level_store(request, tmp_path_factory):
    """
    A group of several statistics on one grid, beside their coordinates,
    an auxiliary coordinate, a bounds table and a lower-rank helper.
    """
    time = np.datetime64("2024-01-01") + np.arange(4) * np.timedelta64(60, "s")
    bounds = np.stack([time, time + np.timedelta64(60, "s")], axis=1)
    dims = ("time", "distance")
    level = xr.Dataset(
        {
            "hi_max": (dims, np.arange(12.0).reshape(4, 3)),
            "hi_mean": (dims, np.arange(12.0).reshape(4, 3) / 2),
            "count": ("time", np.arange(4)),
            "time_bnds": (("time", "nv"), bounds),
        },
        coords={
            "time": ("time", time, {"bounds": "time_bnds"}),
            "distance": np.arange(3) * 10.0,
            "latitude": ("distance", np.arange(3) + 40.0),
        },
    )
    path = tmp_path_factory.mktemp("zarr") / "levels.zarr"
    return _store(path, {"/levels/1min": level}, request.param), level


class TestSeveralPayloads:
    """A group of several payloads is a patch each, keyed by its path."""

    def test_keys(self, level_store):
        """Only the payloads are patches; helpers and coordinates are not."""
        path, _ = level_store
        keys = sorted(x.source_patch_key for x in dc.scan(path))
        assert keys == ["levels/1min/hi_max", "levels/1min/hi_mean"]

    def test_read_each(self, level_store):
        """Each payload reads back with the group's coordinates."""
        path, level = level_store
        for name in ("hi_max", "hi_mean"):
            data = ZarrV3().read_array(path, key=f"levels/1min/{name}")
            assert np.array_equal(data, level[name].values)
        patches = {x.attrs.tag: x for x in dc.read(path)}
        assert set(patches) == {"hi_max", "hi_mean"}  # told apart, not merged
        patch = patches["hi_mean"]
        assert np.array_equal(patch.data, level["hi_mean"].values)
        assert np.array_equal(patch.get_array("latitude"), level["latitude"].values)

    def test_group_key_ambiguous(self, level_store):
        """A group of several payloads names none of them."""
        path, _ = level_store
        with pytest.raises(PatchAttributeError, match="No patch named"):
            ZarrV3().read_array(path, key="levels/1min")
        with pytest.raises(PatchAttributeError, match="No patch named"):
            ZarrV3().read_array(path, key="levels/1min/count")


def _halves(patch, dim="time"):
    """Split a patch in two along ``dim``, at a sample no chunk ends on."""
    size = patch.shape[patch.get_axis(dim)]
    first = patch.select(**{dim: (None, size // 2 + 3)}, samples=True)
    return first, patch.select(**{dim: (size // 2 + 3, None)}, samples=True)


class TestAppend:
    """append adds patches to a store, or extends one along a dimension."""

    @pytest.mark.parametrize("version", VERSIONS)
    def test_add_patch(self, zarr_spool, tmp_path, version):
        """A patch is added beside the stored ones, which are not rewritten."""
        first, later, twin = zarr_spool
        path = tmp_path / "add.zarr"
        dc.write(first, path, "zarr", file_version=version)
        meta = {"zarr.json", ".zarray", ".zattrs", ".zgroup", ".zmetadata"}
        chunks = [x for x in path.rglob("*") if x.is_file() and x.name not in meta]
        before = {x: x.stat().st_mtime_ns for x in chunks}
        assert chunks
        dc.write(dc.spool([later, twin]), path, "zarr", append=True)
        assert {x: x.stat().st_mtime_ns for x in chunks} == before
        assert sorted(dc.read(path), key=str) == sorted(zarr_spool, key=str)
        assert dc.get_format(path) == ("ZARR", version)

    def test_add_repeated_name(self, multi_written, tmp_path):
        """A name already taken is suffixed, as a fresh write does."""
        path, names = multi_written
        copy = tmp_path / "copy.zarr"
        LocalFileSystem().copy(str(path), str(copy), recursive=True)
        dc.write(dc.read(path)[0], copy, "zarr", append=True)
        keys = {x.source_patch_key for x in dc.scan(copy)}
        assert f"{names[0]}__2" in keys and len(keys) == 4

    def test_missing_store_written(self, zarr_patch, tmp_path):
        """With no store at the path, append writes one."""
        path = tmp_path / "new.zarr"
        dc.write(zarr_patch, path, "zarr", append="time")
        assert dc.read(path)[0] == zarr_patch

    def test_remote_add(self, zarr_spool):
        """A patch is added to a store on another filesystem."""
        path = UPath("memory://dascore_zarr_append/add.zarr")
        first, later, _ = zarr_spool
        dc.write(first, path, "zarr")
        dc.write(later, path, "zarr", append=True)
        assert len(dc.read(path)) == 2
        assert [x.name for x in path.parent.iterdir()] == ["add.zarr"]

    @pytest.mark.parametrize("version", VERSIONS)
    def test_extend(self, zarr_patch, tmp_path, version):
        """A patch extended along time reads back whole, its grid described."""
        path = tmp_path / "grow.zarr"
        first, second = _halves(zarr_patch)
        encoding = {"data": {"chunks": (50, 64)}, "time": {"chunks": (64,)}}
        dc.write(first, path, "zarr", file_version=version, encoding=encoding)
        dc.write(second, path, "zarr", append="time")
        assert dc.read(path)[0] == zarr_patch
        attrs = zarr.open_group(path, mode="r")["time"].attrs
        assert attrs[RANGE_KEY] == len(zarr_patch.get_coord("time"))

    @pytest.mark.parametrize("step", [np.int64(2), np.float32(1.02)])
    def test_extend_distance(self, tmp_path, step):
        """A patch grows along a numeric dimension, described or not."""
        patch = dc.get_example_patch()
        size = patch.shape[patch.get_axis("distance")]
        patch = patch.update_coords(distance=np.arange(size, dtype=step.dtype) * step)
        path = tmp_path / "grow.zarr"
        first, second = _halves(patch, "distance")
        dc.write(first, path, "zarr")
        dc.write(second, path, "zarr", append="distance")
        assert dc.read(path)[0] == patch

    def test_failed_add_leaves_store(self, zarr_spool, tmp_path, monkeypatch):
        """An add failing part way leaves the store and no staged group."""
        first, later, _ = zarr_spool
        path = tmp_path / "add.zarr"
        dc.write(first, path, "zarr")

        def _fail(*args, **kwargs):
            raise OSError("no space")

        monkeypatch.setattr(LocalFileSystem, "mv", _fail)
        with pytest.raises(OSError, match="no space"):
            dc.write(later, path, "zarr", append=True)
        assert list(dc.read(path)) == [first]
        assert [x.name for x in tmp_path.iterdir()] == ["add.zarr"]

    def test_extend_twice(self, zarr_patch, tmp_path):
        """Several pieces in one write each extend what the last one left."""
        path = tmp_path / "grow.zarr"
        first, rest = _halves(zarr_patch)
        second, third = _halves(rest)
        dc.write(first, path, "zarr")
        dc.write(dc.spool([second, third]), path, "zarr", append="time")
        assert dc.read(path)[0] == zarr_patch

    def test_partial_move_removed(self, zarr_spool, tmp_path, monkeypatch):
        """A move copying part of a group before failing leaves none of it."""
        first, later, _ = zarr_spool
        path = tmp_path / "add.zarr"
        dc.write(first, path, "zarr")
        mv = LocalFileSystem.mv

        def _fail(self, source, target, **kwargs):
            mv(self, source, target, **kwargs)
            raise OSError("interrupted")

        monkeypatch.setattr(LocalFileSystem, "mv", _fail)
        with pytest.raises(OSError, match="interrupted"):
            dc.write(later, path, "zarr", append=True)
        monkeypatch.undo()
        dc.write(first.update_attrs(tag="x"), path, "zarr", append=True)
        assert len(dc.read(path)) == 2  # the interrupted group never shows

    def test_extend_one_group(self, zarr_spool, tmp_path):
        """Only the patch the new one continues is extended."""
        first, later, twin = zarr_spool
        path = tmp_path / "grow.zarr"
        start, end = _halves(later)
        dc.write(dc.spool([first, start, twin]), path, "zarr")
        dc.write(end, path, "zarr", append="time")
        assert list(dc.read(path)) == [first, twin, later]

    def test_update_sees_append(self, zarr_spool, tmp_path):
        """A spool's update sees an added and an extended patch."""
        first, _, twin = zarr_spool
        path = tmp_path / "grow.zarr"
        start, end = _halves(first)
        dc.write(start, path, "zarr")
        spool = dc.spool(path).update()
        dc.write(end, path, "zarr", append="time")
        assert spool.update()[0] == first
        dc.write(twin, path, "zarr", append=True)
        assert len(spool.update()) == 2

    def test_chunks_before_shape(self, zarr_patch, tmp_path, monkeypatch):
        """Until a shape grows, a reader sees the stored patch exactly."""
        path = tmp_path / "grow.zarr"
        first, second = _halves(zarr_patch)
        dc.write(first, path, "zarr", encoding={"time": {"chunks": (64,)}})
        resize, seen = zarr.Array.resize, []

        def _resize(self, shape):
            if not seen:  # every chunk is written; no shape has changed
                seen.append(dc.read(path)[0])
            return resize(self, shape)

        monkeypatch.setattr(zarr.Array, "resize", _resize)
        dc.write(second, path, "zarr", append="time")
        assert seen == [first]

    def test_unmatched_added(self, zarr_patch, tmp_path):
        """A patch continuing none is added beside; one continuing two raises."""
        path = tmp_path / "grow.zarr"
        first, second = _halves(zarr_patch)
        dc.write(first, path, "zarr")
        gap = second.select(time=(1, None), samples=True)
        other = second.update_attrs(station="SUE")
        moved = second.update_coords(distance=second.get_array("distance") + 1)
        single = second.update(data=second.data.astype(np.float32))
        units = second.set_units("km")
        before = first.select(time=(0, 10), samples=True)
        late = tmp_path / "late.zarr"
        dc.write(second, late, "zarr")
        dc.write(first, late, "zarr", append="time")  # precedes it
        assert sorted(dc.read(late), key=str) == sorted([first, second], key=str)
        for i, patch in enumerate((gap, other, moved, single, units, first)):
            path = tmp_path / f"grow{i}.zarr"
            dc.write(first, path, "zarr")
            dc.write(dc.spool([patch, before]), path, "zarr", append="time")
            out = list(dc.read(path))
            assert len(out) == 3 and first in out and patch in out and before in out
        start, end = _halves(dc.get_example_patch(), "distance")
        holes = end.unselect(distance=(5, 7), samples=True)  # not evenly sampled
        dc.write(start, path, "zarr")
        dc.write(holes, path, "zarr", append="distance")
        assert sorted(dc.read(path), key=str) == sorted([start, holes], key=str)
        twins = tmp_path / "twins.zarr"
        dc.write(dc.spool([first, first.update(data=first.data * 2)]), twins, "zarr")
        with pytest.raises(ParameterError, match="Several patches"):
            dc.write(second, twins, "zarr", append="time")

    def test_associated_coord_refused(self, zarr_patch, tmp_path):
        """A coordinate riding the dimension (latitude) is not extended."""
        first, second = _halves(zarr_patch, "distance")
        path = tmp_path / "grow.zarr"
        dc.write(first, path, "zarr")
        with pytest.raises(NotImplementedError, match="latitude"):
            dc.write(second, path, "zarr", append="distance")
        assert dc.read(path)[0] == first

    def test_shared_coords_refused(self, level_store, tmp_path):
        """One of several payloads on shared coordinates is not extended alone."""
        path = tmp_path / "level.zarr"
        LocalFileSystem().copy(str(level_store[0]), str(path), recursive=True)
        patch = dc.spool(path).select(tag="hi_max")[0]
        step = patch.get_coord("time").step
        later = patch.update_coords(time_min=patch.get_coord("time").max() + step)
        with pytest.raises(NotImplementedError, match="shares 'time'"):
            dc.write(later, path, "zarr", append="time")

    def test_helper_along_dim_refused(self, zarr_patch, tmp_path):
        """A lower-rank array along the dimension is not left short."""
        path = tmp_path / "helper.zarr"
        first, second = _halves(zarr_patch)
        dc.write(first, path, "zarr")
        count = xr.DataArray(np.arange(first.shape[1]), dims="time", name="count")
        count.to_zarr(path, mode="a", consolidated=True)
        with pytest.raises(NotImplementedError, match="count"):
            dc.write(second, path, "zarr", append="time")
        assert dc.read(path)[0] == first

    def test_scaled_payload_refused(self, foreign_store, tmp_path):
        """A payload stored scaled (CF scale_factor) is not extended."""
        path = tmp_path / "foreign.zarr"
        LocalFileSystem().copy(str(foreign_store[0]), str(path), recursive=True)
        patch = dc.read(path)[0]
        step = patch.get_coord("time").step
        later = patch.update_coords(time_min=patch.get_coord("time").max() + step)
        with pytest.raises(NotImplementedError, match="stored encoded"):
            dc.write(later, path, "zarr", append="time")

    @pytest.mark.parametrize("name", ["data", "time"])
    def test_same_dtype_scaled_refused(self, zarr_patch, tmp_path, name):
        """A payload or labels scaled in their own dtype are not extended."""
        path = tmp_path / "scaled.zarr"
        first, second = _halves(zarr_patch.update(data=zarr_patch.data * 1.0))
        dc.write(first, path, "zarr")
        group = zarr.open_group(path, mode="r+")
        group[name].attrs.update(scale_factor=1, add_offset=0)
        with pytest.raises(NotImplementedError, match="stored encoded"):
            dc.write(second, path, "zarr", append="time")

    def test_bad_append(self, zarr_patch, tmp_path):
        """Append is a bool or a dimension the patch has."""
        path = tmp_path / "bad.zarr"
        for value in (1.5, ""):
            with pytest.raises(ParameterError, match="append must be"):
                dc.write(zarr_patch, path, "zarr", append=value)
        dc.write(zarr_patch, path, "zarr")
        with pytest.raises(ParameterError, match="no dimension"):
            dc.write(zarr_patch, path, "zarr", append="bob")
        with pytest.raises(ValueError, match="empty"):
            dc.write(dc.spool([]), path, "zarr", append=True)
