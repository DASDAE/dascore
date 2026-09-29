"""Tests for the ZARR format (CF-on-zarr through xarray)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from fsspec.implementations.local import LocalFileSystem
from upath import UPath

import dascore as dc
from dascore.exceptions import MissingOptionalDependencyError
from dascore.io import core as io_core
from dascore.io.core import source_identity
from dascore.io.index.indexer import scan_unit_stats
from dascore.io.zarr import ZarrV3
from dascore.utils.misc import suppress_warnings

xr = pytest.importorskip("xarray")
zarr = pytest.importorskip("zarr")

VERSIONS = ("3", "2")


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

    def test_index_in_store_is_not_a_change(self, zarr_patch, tmp_path):
        """A spool of one store keeps its index inside and does not re-index."""
        path = tmp_path / "a.zarr"
        dc.write(zarr_patch, path, "zarr", file_version="2")
        spool = dc.spool(path).update()
        sources = spool.indexer._backend.get_sources
        before = sources()["last_indexed_ns"].max()
        spool.update()
        assert sources()["last_indexed_ns"].max() == before


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


class TestRemote:
    """xarray takes a UPath, so an fsspec store works as a local one."""

    def test_memory_round_trip(self, zarr_patch):
        """A memory:// store writes, scans and reads back."""
        path = UPath("memory://dascore_zarr_test/patch.zarr")
        dc.write(zarr_patch, path, "zarr")
        assert dc.read(path)[0] == zarr_patch
        (summary,) = dc.scan(path.parent)
        assert summary.source_format == "ZARR"


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
