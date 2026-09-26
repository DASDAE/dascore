"""Tests for DASDAE storage options (chunking and compression)."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from upath import UPath

import dascore as dc
from dascore.core.coords import concat_coords, get_coord
from dascore.exceptions import ParameterError
from dascore.io.dasdae import DASDAEStorage


def _group(h5):
    """Return the first patch group of an open DASDAE file."""
    waveforms = h5["waveforms"]
    return waveforms[next(iter(waveforms))]


@pytest.fixture(scope="class")
def compressed_dict():
    """Storage options exercising every field."""
    return {
        "chunks": {"time": 10, "distance": 1000},
        "compression": "gzip",
        "compression_opts": 3,
        "shuffle": True,
    }


class TestDatasetOptions:
    """Tests for what the storage options put on the written datasets."""

    def test_absent_dim_uses_full_length(self, random_patch, tmp_path):
        """A dim missing from chunks is one chunk along that axis."""
        path = dc.write(
            random_patch, tmp_path / "out.h5", "DASDAE", storage={"chunks": {"time": 7}}
        )
        with h5py.File(path) as h5:
            assert _group(h5)["data"].chunks == (random_patch.shape[0], 7)

    def test_compression_fields(self, random_patch, tmp_path, compressed_dict):
        """Compression, its level and shuffle reach the data dataset."""
        path = dc.write(
            random_patch, tmp_path / "out.h5", "DASDAE", storage=compressed_dict
        )
        with h5py.File(path) as h5:
            data = _group(h5)["data"]
            assert data.compression == "gzip"
            assert data.compression_opts == 3
            assert data.shuffle
            assert data.chunks == (random_patch.shape[0], 10)

    def test_no_storage_is_contiguous(self, random_patch, tmp_path):
        """Without options the data stays contiguous and uncompressed."""
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE")
        with h5py.File(path) as h5:
            data = _group(h5)["data"]
            assert data.chunks is None
            assert data.compression is None

    def test_v1_coord_values_chunked(self, random_patch, tmp_path, compressed_dict):
        """Version 1 writes coordinate values, chunked by the coord's own dim."""
        path = dc.write(
            random_patch,
            tmp_path / "out.h5",
            "DASDAE",
            file_version="1",
            storage=compressed_dict,
        )
        with h5py.File(path) as h5:
            time = _group(h5)["_coord_time"]
            assert time.chunks == (10,)
            assert time.compression == "gzip"

    def test_v2_non_dim_coord_values(self, random_patch, tmp_path, compressed_dict):
        """A version 2 coordinate stored as values gets the options too."""
        size = random_patch.shape[0]
        latitude = np.cumsum(np.random.default_rng(0).random(size))
        patch = random_patch.update_coords(latitude=("distance", latitude))
        storage = dict(compressed_dict, chunks={"distance": 25})
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", storage=storage)
        with h5py.File(path) as h5:
            group = _group(h5)
            assert group["_coord_latitude"].chunks == (25,)
            assert group["_coord_latitude"].shuffle
            # A range is a zero-length descriptor node and takes no options.
            assert group["_coord_time"].chunks is None

    def test_dimensionless_coord(self, random_patch, tmp_path):
        """A coordinate without dims is one chunk along its axis."""
        patch = random_patch.update_coords(foo=(None, np.arange(3.0) ** 2))
        storage = {"chunks": {"time": 10}, "compression": "gzip"}
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", storage=storage)
        with h5py.File(path) as h5:
            assert _group(h5)["_coord_foo"].chunks == (3,)
        assert dc.read(path)[0] == patch

    @pytest.mark.parametrize("version", ["1", "2"])
    def test_empty_patch(self, random_patch, tmp_path, compressed_dict, version):
        """A zero-size patch is written unfiltered rather than refused."""
        end = random_patch.get_coord("time").max()
        patch = random_patch.select(time=(end + np.timedelta64(1, "s"), None))
        assert 0 in patch.shape
        path = tmp_path / "out.h5"
        dc.write(patch, path, "DASDAE", file_version=version, storage=compressed_dict)
        with h5py.File(path) as h5:
            assert _group(h5)["data"].compression is None

    def test_segment_values(self, random_patch, tmp_path):
        """Storage reaches each value array of a gapped coordinate."""
        size = random_patch.shape[0]
        values = np.cumsum(np.random.default_rng(0).random(size))
        values[size // 2 :] += 100  # the gap
        halves = values[: size // 2], values[size // 2 :]
        coord = concat_coords(*(get_coord(data=x) for x in halves))
        patch = random_patch.update_coords(distance=coord)
        path = dc.write(patch, tmp_path / "out.h5", "DASDAE", storage="compressed")
        with h5py.File(path) as h5:
            segments = _group(h5)["_coord_distance"]
            assert len(segments) == 2
            assert all(x.compression == "gzip" for x in segments.values())


class TestStorageForms:
    """Tests for the ways storage can be given."""

    def test_round_trip_v1(self, random_patch, tmp_path, compressed_dict):
        """A compressed, chunked version 1 file reads back as the same patch."""
        path = dc.write(
            random_patch,
            tmp_path / "out.h5",
            "DASDAE",
            file_version="1",
            storage=compressed_dict,
        )
        assert dc.read(path)[0] == random_patch

    def test_preset(self, random_patch, tmp_path):
        """The compressed preset writes gzip level 5 with shuffle."""
        path = dc.write(
            random_patch, tmp_path / "out.h5", "DASDAE", storage="compressed"
        )
        with h5py.File(path) as h5:
            data = _group(h5)["data"]
            assert (data.compression, data.compression_opts) == ("gzip", 5)
            assert data.shuffle

    def test_unknown_preset(self, random_patch, tmp_path):
        """An unknown preset raises listing the valid ones."""
        with pytest.raises(ParameterError, match="compressed"):
            dc.write(random_patch, tmp_path / "out.h5", "DASDAE", storage="bob")

    def test_instance(self, random_patch, tmp_path):
        """An instance passes through."""
        storage = DASDAEStorage(compression="lzf")
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE", storage=storage)
        with h5py.File(path) as h5:
            assert _group(h5)["data"].compression == "lzf"

    def test_remote_upath(self, random_patch, compressed_dict):
        """Options survive a remote write, which uploads a local temp file."""
        path = UPath("memory://dascore/storage/compressed.h5")
        dc.write(random_patch, path, "DASDAE", storage=compressed_dict)
        assert dc.read(path)[0] == random_patch
        with path.open("rb") as raw, h5py.File(raw, "r", driver="fileobj") as h5:
            data = _group(h5)["data"]
            assert (data.compression, data.compression_opts) == ("gzip", 3)
            assert data.chunks == (random_patch.shape[0], 10)

    def test_json_list_opts(self, random_patch, tmp_path):
        """compression_opts read back from JSON as a list reaches h5py."""
        storage = DASDAEStorage(compression=32000, compression_opts=())  # lzf
        storage = DASDAEStorage.model_validate_json(storage.model_dump_json())
        assert storage.compression_opts == []
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE", storage=storage)
        assert dc.read(path)[0] == random_patch


class TestPreWriteCheck:
    """Tests for refusing bad options before anything is written."""

    def test_typo_raises_before_writing(self, random_patch, tmp_path):
        """A misspelled dim raises and leaves no DASDAE file."""
        path = tmp_path / "out.h5"
        with pytest.raises(ParameterError, match="tim"):
            dc.write(random_patch, path, "DASDAE", storage={"chunks": {"tim": 10}})
        if path.exists():
            with h5py.File(path) as h5:
                assert "__format__" not in h5.attrs
                assert "waveforms" not in h5

    def test_bad_h5py_option_keeps_file(self, random_patch, tmp_path):
        """An option h5py refuses raises before an existing file is touched."""
        path = dc.write(random_patch, tmp_path / "out.h5", "DASDAE")
        end = random_patch.get_coord("time").max()
        later = random_patch.update_coords(time_min=end + np.timedelta64(1, "s"))
        storage = {"compression": "gzip", "compression_opts": 99}
        with pytest.raises(ValueError, match="GZIP"):
            dc.write(later, path, "DASDAE", storage=storage)
        assert dc.read(path)[0] == random_patch

    @pytest.mark.parametrize("storage", [None, {"chunks": {"time": 10}}])
    def test_no_trial_without_compression(
        self, random_patch, tmp_path, monkeypatch, storage
    ):
        """Writes without compression options skip the in-memory trial."""

        def _fail(self):
            raise AssertionError("trialled")

        monkeypatch.setattr(DASDAEStorage, "_trial", _fail)
        dc.write(random_patch, tmp_path / "out.h5", "DASDAE", storage=storage)

    def test_spool_dims(self, random_spool, tmp_path):
        """Chunk dims are checked against every patch in a spool."""
        storage = {"chunks": {"time": 10}}
        path = dc.write(random_spool, tmp_path / "out.h5", "DASDAE", storage=storage)
        assert len(dc.spool(path)) == len(random_spool)

    def test_empty_spool(self, tmp_path):
        """An empty spool with chunks just writes."""
        storage = {"chunks": {"time": 10}}
        path = dc.write(dc.spool([]), tmp_path / "out.h5", "DASDAE", storage=storage)
        assert len(dc.spool(path)) == 0
