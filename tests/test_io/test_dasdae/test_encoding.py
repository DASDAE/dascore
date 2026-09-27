"""Tests for DASDAE write encodings (chunking and compression)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.core.coords import concat_coords, get_coord


def _group(h5):
    """Return the first patch group of an open DASDAE file."""
    return next(iter(h5["waveforms"].values()))


@pytest.fixture(scope="class")
def gzip_encoding():
    """An encoding setting every h5py option on the data."""
    data = {
        "compression": "gzip",
        "compression_opts": 3,
        "shuffle": True,
        "fletcher32": True,
        "chunksizes": (10, 10**9),
        "source": "x.nc",  # xarray drops these two on write
        "original_shape": (1, 2),
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

    def test_coords(self, random_patch, tmp_path):
        """Coordinate arrays take their own encoding; scalars and others none."""
        foo, bar = (None, np.arange(3.0) ** 2), ((), np.array(2.0))
        patch = random_patch.update_coords(_foo=foo, bar=bar)
        encoding = {
            "time": {"compression": "lzf", "chunksizes": (7,)},
            "_foo": {"compression": 32000, "compression_opts": [], "chunksizes": (10,)},
            "bar": {"compression": "gzip", "chunksizes": ()},
        }
        path = tmp_path / "out.h5"
        dc.write(patch, path, "DASDAE", file_version="1", encoding=encoding)
        with h5py.File(path) as h5:
            group = _group(h5)
            time = group["_coord_time"]
            assert (time.compression, time.chunks) == ("lzf", (7,))
            foo = group["_coord__foo"]  # a list compression_opts, as JSON gives
            assert (foo.compression, foo.chunks) == ("lzf", (3,))
            assert group["_coord_bar"].compression is None
            assert group["_coord_distance"].compression is None
            assert group["data"].chunks is None
        assert dc.read(path)[0] == patch

    def test_empty_patch(self, random_patch, tmp_path, gzip_encoding):
        """A zero-size patch is written unfiltered rather than refused."""
        end = random_patch.get_coord("time").max()
        patch = random_patch.select(time=(end + np.timedelta64(1, "s"), None))
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", encoding=gzip_encoding)
        with h5py.File(path) as h5:
            assert _group(h5)["data"].compression is None

    def test_segment_values(self, random_patch, tmp_path):
        """The encoding reaches the values of an irregular multi-run coordinate."""
        size = random_patch.shape[0]
        values = np.cumsum(np.random.default_rng(0).random(size))
        values[size // 2 :] += 100  # the gap
        halves = values[: size // 2], values[size // 2 :]
        coord = concat_coords(*(get_coord(data=x) for x in halves))
        patch = random_patch.update_coords(distance=coord)
        encoding = {"distance": {"compression": "gzip", "chunksizes": (1000,)}}
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            node = _group(h5)["_coord_distance"]
            assert node.compression == "gzip"
            assert node.chunks == node.shape

    def test_variable_in_some_patches(self, random_patch, tmp_path):
        """A coordinate only one patch of a spool carries takes the encoding."""
        tagged = random_patch.update_coords(_quality=(None, np.arange(3.0) ** 2))
        spool = dc.spool([tagged, random_patch.update_attrs(station="other")])
        encoding = {"_quality": {"compression": "gzip"}}
        path = dc.write(spool, tmp_path / "out.h5", "DASDAE", encoding=encoding)
        with h5py.File(path) as h5:
            nodes = [g.get("_coord__quality") for g in h5["waveforms"].values()]
            assert {getattr(x, "compression", "-") for x in nodes} == {"-", "gzip"}


class TestValidation:
    """Tests for which encodings are refused."""

    @pytest.mark.parametrize(
        "encoding, error, match",
        [
            ({"data": {"zlib": True, "compression": "lzf"}}, ValueError, "'zlib' and"),
            ({"data": {"complevel": 4, "compression_opts": 5}}, ValueError, "'compl"),
            ({"data": {"complevel": 0, "compression_opts": 5}}, ValueError, "'compl"),
            ({"data": {"bob": 1}}, ValueError, "Valid encodings are"),
            ({"tim": {}}, KeyError, "tim"),
            ({"bob": {}}, KeyError, "bob"),  # an attr named bob_min is no variable
        ],
    )
    def test_bad_encoding(self, random_patch, tmp_path, encoding, error, match):
        """Conflicting or unknown options and unknown variables raise."""
        patch = random_patch.update_attrs(bob_min=1.0)
        with pytest.raises(error, match=match):
            dc.write(patch, tmp_path / "o.h5", "DASDAE", encoding=encoding)

    @pytest.mark.parametrize(
        "data, match, rewrite",
        [
            ({"chunksizes": (10,)}, "rank", False),
            ({"zlib": True, "complevel": 99}, "GZIP", True),
        ],
    )
    def test_file_intact(self, random_patch, tmp_path, data, match, rewrite):
        """Appending or rewriting a patch h5py refuses keeps the stored patch."""
        path = dc.write(random_patch, tmp_path / "o.h5", "DASDAE")
        end = random_patch.get_coord("time").max() + np.timedelta64(1, "s")
        new = random_patch if rewrite else random_patch.update_coords(time_min=end)
        with pytest.raises(Exception, match=match):
            dc.write(new, path, "DASDAE", encoding={"data": data})
        with h5py.File(path) as h5:
            assert list(h5) == ["waveforms"]
            assert list(h5["waveforms"]) == [random_patch.get_patch_name()]
        assert dc.spool(path)[0] == random_patch

    def test_leftover_is_replaced(self, random_patch, tmp_path):
        """A group left by a killed write is cleared by the next write."""
        path = dc.write(random_patch, tmp_path / "killed.h5", "DASDAE")
        with h5py.File(path, "a") as h5:
            h5.create_group("_dascore_partial")
        dc.write(random_patch, path, "DASDAE")
        with h5py.File(path, "r") as h5:
            assert "_dascore_partial" not in h5
