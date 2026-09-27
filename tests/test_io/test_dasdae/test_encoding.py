"""Tests for DASDAE write encodings (chunking and compression)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.core.coords import concat_coords, get_coord


def _group(h5):
    """Return the first patch group of an open DASDAE file."""
    waveforms = h5["waveforms"]
    return waveforms[next(iter(waveforms))]


@pytest.fixture(scope="class")
def gzip_encoding():
    """An encoding setting every h5py option on the data."""
    data = {
        "compression": "gzip",
        "compression_opts": 3,
        "shuffle": True,
        "fletcher32": True,
        "chunksizes": (10, 10**9),
    }
    return {"data": data}


class TestDatasetOptions:
    """Tests for what the encoding puts on the written datasets."""

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_data(self, random_patch, tmp_path, gzip_encoding, version):
        """Every option reaches the data, chunks clamp, and it round trips."""
        path = tmp_path / "out.h5"
        dc.write(
            random_patch, path, "DASDAE", file_version=version, encoding=gzip_encoding
        )
        with h5py.File(path) as h5:
            data = _group(h5)["data"]
            assert (data.compression, data.compression_opts) == ("gzip", 3)
            assert data.shuffle and data.fletcher32
            assert data.chunks == (10, random_patch.shape[1])
        assert dc.read(path)[0] == random_patch

    def test_zlib_complevel(self, random_patch, tmp_path):
        """Zlib and complevel translate to gzip and its level."""
        encoding = {"data": {"zlib": True, "complevel": 6}}
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            data = _group(h5)["data"]
            assert (data.compression, data.compression_opts) == ("gzip", 6)

    def test_coord_values(self, random_patch, tmp_path):
        """A coordinate's value array takes its own encoding; others none."""
        encoding = {"time": {"compression": "lzf", "chunksizes": (7,)}}
        path = tmp_path / "out.h5"
        dc.write(random_patch, path, "DASDAE", file_version="1", encoding=encoding)
        with h5py.File(path) as h5:
            group = _group(h5)
            assert group["_coord_time"].compression == "lzf"
            assert group["_coord_time"].chunks == (7,)
            assert group["_coord_distance"].compression is None
            assert group["data"].chunks is None

    def test_v2_range_takes_nothing(self, random_patch, tmp_path):
        """A version 2 range is a zero-length descriptor node, unfiltered."""
        encoding = {"time": {"compression": "gzip"}}
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            assert _group(h5)["_coord_time"].compression is None

    def test_dimensionless_coord(self, random_patch, tmp_path):
        """A coordinate without dims pairs chunksizes with its own axis."""
        patch = random_patch.update_coords(foo=(None, np.arange(3.0) ** 2))
        encoding = {"foo": {"chunksizes": (10,), "compression": "gzip"}}
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            assert _group(h5)["_coord_foo"].chunks == (3,)
        assert dc.read(path)[0] == patch

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_empty_patch(self, random_patch, tmp_path, gzip_encoding, version):
        """A zero-size patch is written unfiltered rather than refused."""
        end = random_patch.get_coord("time").max()
        patch = random_patch.select(time=(end + np.timedelta64(1, "s"), None))
        assert 0 in patch.shape
        path = tmp_path / "out.h5"
        dc.write(patch, path, "DASDAE", file_version=version, encoding=gzip_encoding)
        with h5py.File(path) as h5:
            assert _group(h5)["data"].compression is None

    def test_segment_values(self, random_patch, tmp_path):
        """The encoding reaches each value array of a gapped coordinate."""
        size = random_patch.shape[0]
        values = np.cumsum(np.random.default_rng(0).random(size))
        values[size // 2 :] += 100  # the gap
        halves = values[: size // 2], values[size // 2 :]
        coord = concat_coords(*(get_coord(data=x) for x in halves))
        patch = random_patch.update_coords(distance=coord)
        encoding = {"distance": {"compression": "gzip", "chunksizes": (1000,)}}
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            segments = _group(h5)["_coord_distance"]
            assert len(segments) == 2
            assert all(x.compression == "gzip" for x in segments.values())
            assert all(x.chunks == x.shape for x in segments.values())

    def test_list_compression_opts(self, random_patch, tmp_path):
        """A list compression_opts, as JSON gives, reaches h5py as a tuple."""
        encoding = {"data": {"compression": 32000, "compression_opts": []}}  # lzf
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        assert dc.read(path)[0] == random_patch

    def test_remote_upath(self, random_patch, gzip_encoding):
        """The encoding survives a remote write, which uploads a temp file."""
        path = UPath("memory://dascore/encoding/compressed.h5")
        dc.write(random_patch, path, "DASDAE", encoding=gzip_encoding)
        assert dc.read(path)[0] == random_patch
        with path.open("rb") as raw, h5py.File(raw, "r", driver="fileobj") as h5:
            data = _group(h5)["data"]
            assert (data.compression, data.compression_opts) == ("gzip", 3)
            assert data.chunks == (10, random_patch.shape[1])


class TestValidation:
    """Tests for which encodings are accepted."""

    @pytest.mark.parametrize(
        "data, match",
        [
            ({"zlib": True, "compression": "lzf"}, "'zlib' and 'compression'"),
            ({"complevel": 4, "compression_opts": 5}, "'complevel' and"),
            ({"bob": 1}, "Valid encodings are"),
        ],
    )
    def test_bad_encoding(self, random_patch, tmp_path, data, match):
        """Conflicting or unknown options raise ValueError."""
        with pytest.raises(ValueError, match=match):
            dc.write(random_patch, tmp_path / "o.h5", "DASDAE", encoding={"data": data})

    def test_unknown_variable(self, random_patch, tmp_path):
        """A variable the patch lacks raises xarray's KeyError."""
        encoding = {"tim": {"compression": "gzip"}}
        with pytest.raises(KeyError, match="tim"):
            dc.write(random_patch, tmp_path / "o.h5", "DASDAE", encoding=encoding)

    def test_variable_in_some_patches(self, random_patch, tmp_path):
        """A coordinate only some patches carry takes the encoding in those."""
        tagged = random_patch.update_coords(_quality=(None, np.arange(3.0) ** 2))
        other = random_patch.update_attrs(station="other")
        encoding = {"_quality": {"compression": "gzip"}}
        path = tmp_path / "mixed.h5"
        dc.write(dc.spool([other, tagged]), path, "DASDAE", encoding=encoding)
        assert len(dc.spool(path)) == 2
        with h5py.File(path) as h5:
            nodes = [
                g["_coord__quality"]
                for g in h5["waveforms"].values()
                if "_coord__quality" in g
            ]
            assert [x.compression for x in nodes] == ["gzip"]

    def test_attr_is_not_a_variable(self, random_patch, tmp_path):
        """An attribute ending in _min does not make a variable."""
        patch = random_patch.update_attrs(bob_min=1.0)
        with pytest.raises(KeyError, match="bob"):
            dc.write(patch, tmp_path / "o.h5", "DASDAE", encoding={"bob": {}})

    def test_private_coord(self, random_patch, tmp_path):
        """A coordinate the contents table hides still takes an encoding."""
        patch = random_patch.update_coords(_quality=(None, np.arange(3.0) ** 2))
        encoding = {"_quality": {"compression": "gzip"}}
        path = dc.write(patch, tmp_path / "o.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            assert _group(h5)["_coord__quality"].compression == "gzip"

    def test_xarray_dropped_keys(self, random_patch, tmp_path):
        """Keys xarray drops on write, as an opened file's encoding has, pass."""
        opts = {"zlib": True, "source": "x.nc", "original_shape": (1, 2)}
        path = tmp_path / "o.h5"
        dc.write(random_patch, path, "DASDAE", encoding={"data": opts})
        with h5py.File(path) as h5:
            assert _group(h5)["data"].compression == "gzip"

    def test_scalar_coord(self, random_patch, tmp_path):
        """A scalar coordinate is written unfiltered, as xarray does."""
        patch = random_patch.update_coords(foo=((), np.array(2.0)))
        encoding = {"foo": {"compression": "gzip", "chunksizes": ()}}
        path = tmp_path / "o.h5"
        dc.write(patch, path, "DASDAE", file_version="1", encoding=encoding)
        with h5py.File(path) as h5:
            assert _group(h5)["_coord_foo"].compression is None
        assert dc.read(path)[0] == patch

    def test_spool(self, random_spool, tmp_path, gzip_encoding):
        """One encoding writes every patch of a spool."""
        path = tmp_path / "out.h5"
        dc.write(random_spool, path, "DASDAE", encoding=gzip_encoding)
        assert len(dc.spool(path)) == len(random_spool)


class TestFailedWrite:
    """Tests that a write h5py refuses leaves an existing file as it was."""

    @pytest.fixture(
        params=[
            ({"chunksizes": (0, 10)}, "positive"),
            ({"chunksizes": (-1, 10)}, "negative"),
            ({"chunksizes": (10,)}, "rank"),
            ({"complevel": 4}, "Compression method"),  # as in h5netcdf
            ({"compression": "gzip", "compression_opts": 99}, "GZIP"),
        ]
    )
    def bad_encoding(self, request):
        """An encoding h5py refuses on the data, and its error message."""
        encoding, match = request.param
        return {"data": encoding}, match

    @pytest.mark.parametrize("version", ["1", "2"])
    @pytest.mark.parametrize("rewrite", [False, True])
    def test_file_intact(self, random_patch, tmp_path, bad_encoding, version, rewrite):
        """Appending or rewriting a patch that fails keeps the stored patch."""
        path = dc.write(random_patch, tmp_path / "o.h5", "DASDAE", file_version=version)
        end = random_patch.get_coord("time").max()
        new = random_patch.update_coords(time_min=end + np.timedelta64(1, "s"))
        new = random_patch if rewrite else new
        encoding, match = bad_encoding
        with pytest.raises(Exception, match=match):
            dc.write(new, path, "DASDAE", file_version=version, encoding=encoding)
        with h5py.File(path) as h5:
            assert list(h5["waveforms"]) == [random_patch.get_patch_name()]
            assert "_dascore_partial" not in h5
        spool = dc.spool(path)
        assert len(spool) == 1 and spool[0] == random_patch


class TestKilledWrite:
    """Tests for a partial group left by a write that never finished."""

    def test_leftover_is_replaced(self, random_patch, tmp_path):
        """A later write clears the leftover instead of failing on it."""
        path = tmp_path / "killed.h5"
        dc.write(random_patch, path, "DASDAE")
        with h5py.File(path, "a") as h5:
            h5.create_group("_dascore_partial")
        dc.write(random_patch, path, "DASDAE")
        with h5py.File(path, "r") as h5:
            assert "_dascore_partial" not in h5
        assert dc.spool(path)[0] == random_patch
