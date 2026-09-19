"""Tests for ArraySource, the description of a stored array."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.exceptions import InvalidFiberIOError, ParameterError
from dascore.io import core as io_core
from dascore.utils.downloader import fetch


@pytest.fixture(scope="module")
def path():
    """A DASDAE file holding one two-dimensional patch."""
    return fetch("example_dasdae_event_1.h5")


@pytest.fixture(scope="module")
def patch(path):
    """The patch read from the file."""
    return dc.read(path)[0]


@pytest.fixture(scope="module")
def source(patch):
    """The source the framework gave the patch."""
    return patch._source


@pytest.fixture()
def reads(monkeypatch):
    """Count the arrays the loader reads."""
    calls = []
    original = io_core._load_array_source
    monkeypatch.setattr(
        io_core, "_load_array_source", lambda x: calls.append(x) or original(x)
    )
    return calls


class TestDescription:
    """The source describes the array without reading it."""

    def test_matches_patch(self, source, patch):
        """Shape, dtype and dims are the patch's own."""
        assert source.loadable
        assert (source.shape, source.dtype) == (patch.shape, patch.dtype)
        assert source.dims == patch.dims
        assert (source.ndim, source.size) == (patch.data.ndim, patch.data.size)

    def test_reader_source_not_loadable(self):
        """A source naming only a key cannot load, and says so."""
        source = ArraySource(key="a")
        assert not source.loadable
        with pytest.raises(ParameterError, match="does not say enough"):
            source.load()
        with pytest.raises(IndexError):
            source[:1]


class TestLoad:
    """Loading gives exactly the described array."""

    def test_load(self, source, patch, reads):
        """The whole array loads, and numpy can ask for it."""
        assert np.array_equal(source.load(), patch.data)
        out = np.asarray(source, dtype=np.float32)
        assert out.dtype == np.float32 and len(reads) == 2

    def test_stale_description(self, source):
        """An array which is not the one described is refused."""
        wrong = replace(source, shape=(1, 1))
        with pytest.raises(InvalidFiberIOError, match="may have changed"):
            wrong.load()

    def test_address(self, source, path):
        """An address loads a stored array which is not the patch's data."""
        with h5py.File(path) as fi:
            name = next(
                f"{group.name}/_coord_time" for group in fi["waveforms"].values()
            )
            shape, dtype, expected = fi[name].shape, fi[name].dtype, fi[name][3:9]
        coord = replace(source, key="", address=name).describe(shape, dtype)
        assert np.array_equal(coord[3:9].load(), expected)

    def test_address_needs_support(self, random_patch, tmp_path):
        """A format which is not HDF5 refuses an address until it says how."""
        out = tmp_path / "patch.pkl"
        random_patch.io.write(out, "pickle")
        source = replace(dc.read(out)[0]._source, address="data")
        with pytest.raises(NotImplementedError, match="by address"):
            source.load()


class TestMultiPatch:
    """Each patch in a resource loads its own array."""

    @pytest.fixture()
    def two_patch_path(self, tmp_path):
        """A DASDAE file holding two patches with different data."""
        path = tmp_path / "two.h5"
        spool = dc.get_example_spool()
        dc.write(spool[0], path, "dasdae")
        dc.write(spool[1].new(data=spool[1].data * 3.0), path, "dasdae", append=True)
        return path

    def test_each_loads_its_own(self, two_patch_path):
        """The key tells the two arrays apart, and neither loads the other."""
        patches = list(dc.read(two_patch_path))
        assert len(patches) == 2
        sources = [patch._source for patch in patches]
        assert sources[0].key != sources[1].key
        assert sources[0].id != sources[1].id
        assert not np.array_equal(patches[0].data, patches[1].data)
        for patch, source in zip(patches, sources, strict=True):
            assert np.array_equal(source.load(), patch.data)


class TestSlicing:
    """Slicing composes windows and reads nothing."""

    def test_lazy(self, source, patch, reads):
        """Slices compose, and only `load` touches the file."""
        sub = source[10:20, 5:][2:4]
        assert sub.windows == ((12, 14), (5, patch.shape[1]))
        assert not reads
        assert np.array_equal(sub.load(), patch.data[12:14, 5:])

    def test_negative_and_open(self, source, patch):
        """Negative and omitted bounds resolve against the known shape."""
        assert source[-3:].windows[0] == (patch.shape[0] - 3, patch.shape[0])
        assert source[:] == source[slice(0, None, 1)] == source

    @pytest.mark.parametrize("index", [0, slice(None, None, 2), [1, 2]])
    def test_only_contiguous(self, source, index):
        """Anything but a unit-step slice is refused; narrow detaches."""
        with pytest.raises(TypeError, match="step of one"):
            source[index]
        assert not source.narrow(index).loadable

    def test_too_many_indices(self, source):
        """More indices than dimensions is an error."""
        with pytest.raises(IndexError):
            source[:, :, :]


class TestIdentity:
    """Equal sources hash alike; a different array does not."""

    def test_hash_and_id(self, source):
        """The same recipe built twice agrees, in a dict and by digest."""
        other = replace(source)
        assert other == source and hash(other) == hash(source)
        assert other.id == source.id
        assert source[:10][2:5].id == source[2:5].id

    def test_different_arrays(self, source):
        """A selection, a key, an address or a path is a different array."""
        ids = {
            source.id,
            source[:10].id,
            replace(source, key="b").id,
            replace(source, address="a/b").id,
            replace(source, path="elsewhere.h5").id,
        }
        assert len(ids) == 5

    def test_description_not_identity(self, source):
        """Shape and dtype follow from the rest, so they are not hashed."""
        assert replace(source, dtype=np.dtype("int8")).id == source.id


class TestSerialize:
    """A source survives JSON and pickle."""

    def test_json(self, source):
        """The dict form is plain JSON and restores an equal source."""
        sub = source[1:9]
        out = ArraySource.from_dict(json.loads(json.dumps(sub.to_dict())))
        assert out == sub and out.id == sub.id
        assert ArraySource.from_dict(ArraySource(key="a").to_dict()).key == "a"

    def test_pickle(self, source, patch):
        """Another process needs nothing but the source to load."""
        out = pickle.loads(pickle.dumps(source[:4]))
        assert np.array_equal(out.load(), patch.data[:4])


class TestCarry:
    """Which operations keep a source which still loads their data."""

    def test_read_select(self, path, patch):
        """A contiguous read selection is composed onto the source."""
        dist = patch.get_array("distance")
        sub = dc.read(path, distance=(dist[100], dist[200]))[0]
        assert sub._source.windows[0] == (100, 201)
        assert np.array_equal(sub._source.load(), sub.data)

    def test_select(self, patch):
        """Patch selections compose too."""
        time = patch.get_array("time")
        sub = patch.select(time=(time[10], time[50]))
        sub = sub.select(distance=(3, 9), samples=True)
        assert np.array_equal(sub._source.load(), sub.data)

    def test_metadata_only(self, patch):
        """Changing attrs leaves the data, and so the source, alone."""
        assert patch.update_attrs(station="bob")._source == patch._source

    @pytest.mark.parametrize(
        "func",
        [
            lambda x: x.abs(),
            lambda x: x.transpose(),
            lambda x: x.new(data=x.data * 2),
            lambda x: x.rename_coords(time="t"),
            lambda x: x.select(distance=np.array([1, 5]), samples=True),
        ],
    )
    def test_detached(self, patch, func):
        """New values or a new layout keep the provenance and nothing else."""
        source = func(patch)._source
        assert not source.loadable
        assert (source.path, source.key) == (patch._source.path, patch._source.key)
