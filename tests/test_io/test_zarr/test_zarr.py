"""Tests for the ZARR format (CF-on-zarr through xarray)."""

from __future__ import annotations

import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.exceptions import ParameterError
from dascore.io.zarr import ZarrV2, ZarrV3
from dascore.io.zarr import core as zarr_core

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


@pytest.fixture(scope="module")
def foreign_store(tmp_path_factory):
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
    dataset.to_zarr(path, zarr_format=3, consolidated=False, encoding=encoding)
    return path, data, time


class TestRoundTrip:
    """A written patch reads back unchanged."""

    def test_patch_equal(self, written, zarr_patch):
        """Data, coords (incl. the non-dim one) and attrs round trip."""
        _, path = written
        patch = dc.read(path)[0]
        assert patch == zarr_patch
        assert set(patch.coords.coord_map) == set(zarr_patch.coords.coord_map)

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

    def test_format_and_zarr_format(self, written):
        """get_format and the store both report the written zarr format."""
        version, path = written
        assert dc.get_format(path) == ("ZARR", version)
        assert zarr.open_group(path, mode="r").metadata.zarr_format == int(version)

    def test_time_stored_as_int64_ns(self, written):
        """Time is stored as exact int64 nanoseconds, not a lossy float."""
        _, path = written
        time = zarr.open_group(path, mode="r")["time"]
        assert time.dtype == np.int64
        assert time.attrs["units"].startswith("nanoseconds since")

    def test_read_array_window(self, written, zarr_patch):
        """read_array returns only the requested window."""
        _, path = written
        out = ZarrV3().read_array(path, ((2, 5), (10, 30)))
        assert np.array_equal(out, zarr_patch.data[2:5, 10:30])

    def test_scan(self, written, zarr_patch):
        """A scan describes the store without reading its payload."""
        version, path = written
        (summary,) = dc.scan(path)
        assert summary.source_format == "ZARR"
        assert summary.source_version == version
        assert summary.shape == zarr_patch.shape


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

    def test_v2_chunks(self, zarr_patch, tmp_path):
        """Chunks reach a zarr format 2 store too."""
        path = tmp_path / "enc2.zarr"
        encoding = {"data": {"chunks": (25, 400)}}
        dc.write(zarr_patch, path, "zarr", file_version="2", encoding=encoding)
        assert zarr.open_group(path, mode="r")["data"].chunks == (25, 400)


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

    def test_plain_directory(self, tmp_path):
        """A directory with no zarr marker is not zarr."""
        (tmp_path / "file.txt").write_text("hello")
        assert ZarrV3().get_version(tmp_path) is None

    @pytest.mark.parametrize("zarr_format", [3, 2])
    def test_no_dimensioned_array(self, tmp_path, zarr_format):
        """A store whose arrays carry no dimension names is not ours."""
        path = tmp_path / "nodims.zarr"
        group = zarr.open_group(path, mode="w", zarr_format=zarr_format)
        group.create_array("a", shape=(3,), dtype="f8")
        assert ZarrV3().get_version(path) is None

    def test_scalar_payload(self, tmp_path):
        """A store whose only variable has no dimension is not ours."""
        path = tmp_path / "scalar.zarr"
        xr.Dataset({"a": ((), 1.0)}).to_zarr(path, zarr_format=3, consolidated=False)
        assert ZarrV3().get_version(path) is None

    def test_missing_dependencies(self, written, monkeypatch):
        """Without zarr or xarray installed nothing is claimed."""
        _, path = written
        monkeypatch.setattr(zarr_core, "_has_zarr_deps", lambda: False)
        assert ZarrV2().get_version(path) is None


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


class TestRemote:
    """xarray takes a UPath, so an fsspec store works as a local one."""

    def test_memory_round_trip(self, zarr_patch):
        """A memory:// store writes, scans and reads back."""
        path = UPath("memory://dascore_zarr_test/patch.zarr")
        dc.write(zarr_patch, path, "zarr")
        assert dc.read(path)[0] == zarr_patch
        (summary,) = dc.scan(path.parent)
        assert summary.source_format == "ZARR"


class TestWriteRefusals:
    """The writer holds one contiguous patch per store."""

    @pytest.fixture()
    def two_patch_spool(self, zarr_patch):
        """A spool of two distinct patches."""
        return dc.spool([zarr_patch, zarr_patch.update_attrs(station="OTHER")])

    def test_multi_patch_spool(self, two_patch_spool, tmp_path):
        """dc.write refuses a spool of several patches."""
        with pytest.raises(ParameterError, match="one patch per file"):
            dc.write(two_patch_spool, tmp_path / "multi.zarr", "zarr")

    def test_direct_multi_patch(self, two_patch_spool, tmp_path):
        """The writer itself refuses several patches."""
        with pytest.raises(NotImplementedError, match="Zarr output"):
            ZarrV3().write(two_patch_spool, tmp_path / "multi.zarr")

    def test_gapped_patch(self, tmp_path):
        """A patch with a hole is refused, not written as one store."""
        distance = concat_coords(
            get_coord(start=0.0, stop=10.0, step=1.0),
            get_coord(start=15.0, stop=25.0, step=1.0),
        )
        patch = dc.Patch(
            data=np.zeros((20, 10)),
            coords={"distance": distance, "time": dc.to_datetime64(np.arange(10))},
            dims=("distance", "time"),
        )
        path = tmp_path / "gapped.zarr"
        with pytest.raises(ParameterError, match="one patch per file"):
            dc.write(patch, path, "zarr")
        assert not path.exists()
