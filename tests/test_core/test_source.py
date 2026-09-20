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
from dascore.exceptions import (
    InvalidFiberIOError,
    ParameterError,
    PatchAttributeError,
)
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
        """Shape and dtype are the patch's own."""
        assert source.loadable
        assert (source.shape, source.dtype) == (patch.shape, patch.dtype)
        assert (source.ndim, source.size) == (patch.data.ndim, patch.data.size)

    def test_reader_source_not_loadable(self):
        """A source naming only a key cannot load, and says so."""
        source = ArraySource(key="a")
        assert not source.loadable
        with pytest.raises(ParameterError, match="does not say enough"):
            source.load()
        with pytest.raises(IndexError, match="describes no array"):
            source[:1]

    def test_needs_a_window_per_axis(self, source):
        """The reader is given one positional window for each axis."""
        assert not replace(source, windows=source.windows[:1]).loadable


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

    def test_stored_array(self, source, path):
        """An absolute key loads a stored array which is not a patch's data."""
        with h5py.File(path) as fi:
            name = next(f"{x.name}/_coord_time" for x in fi["waveforms"].values())
            shape, dtype, expected = fi[name].shape, fi[name].dtype, fi[name][3:9]
        assert name.startswith("/")
        coord = replace(source, key=name).describe(shape, dtype)
        assert np.array_equal(coord[3:9].load(), expected)

    def test_stored_array_needs_hdf5(self, random_patch, tmp_path):
        """Another kind of resource gives the key to its reader, which refuses."""
        out = tmp_path / "patch.pkl"
        random_patch.io.write(out, "pickle")
        source = replace(dc.read(out)[0]._source, key="/data")
        with pytest.raises(PatchAttributeError, match="No patch named"):
            source.load()


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
        # Pinned: an id written down must still name the same array later.
        fixed = ArraySource("a.h5", "DASDAE", "1", "k", ((2, 5),), (3,), np.dtype("f8"))
        assert fixed.id == "c0e60b662c53454f69b039af7477d3ca"

    def test_different_arrays(self, source):
        """A selection, a key or a path is a different array."""
        located = replace(source, base_id="")
        ids = {
            located.id,
            located[:10].id,
            replace(located, key="b").id,
            replace(located, path="elsewhere.h5").id,
        }
        assert len(ids) == 4

    def test_a_base_id_names_the_array(self, source, patch):
        """Given a base, where the array is kept is not part of which it is."""
        assert source.base_id == source.id == patch.attrs.origin_id
        moved = replace(source, path="elsewhere.h5")
        assert moved.id == source.id and moved[:10].id == source[:10].id
        assert replace(source, base_id="other")[:10].id != source[:10].id

    def test_the_whole_is_the_base(self, source):
        """A window over everything is the array; a prefix is not."""
        assert source[:].id == source.id
        prefix = source[: source.shape[0] - 1]
        assert prefix.id != source.id
        # A prefix is not mistaken for a smaller whole once it is all there is.
        assert prefix[:].id == prefix.id != source.id
        assert source.detach().id == source.id

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


class TestConstant:
    """A source holding a value generates its array and reads nothing."""

    @pytest.fixture()
    def nans(self):
        """A two dimensional block of NaN."""
        return ArraySource.full((2, 3), np.nan)

    def test_full(self, nans, reads):
        """A constant loads without a path, and gives what `np.full` would."""
        assert nans.constant and nans.loadable and not nans.path
        expected = np.full((2, 3), np.nan)
        assert np.array_equal(nans.load(), expected, equal_nan=True)
        assert np.asarray(nans).shape == (2, 3)
        assert not reads

    def test_dtype_and_value(self):
        """The dtype follows the value, and the value is a plain scalar."""
        assert ArraySource.full((3,), 0).load().dtype == np.dtype(int)
        source = ArraySource.full((3,), np.float64(2), dtype="float32")
        assert source.load().dtype == np.dtype("float32")
        assert type(source.value) is float
        assert source == ArraySource.full((3,), 2.0, dtype="float32")

    def test_slice(self, nans, reads):
        """Slicing gives a smaller constant, still without reading."""
        sub = nans[1:, :2]
        assert sub.constant and sub.load().shape == (1, 2)
        assert np.isnan(sub.load()).all() and not reads

    @pytest.mark.parametrize(
        ("value", "dtype"), [([1, 2], None), (None, None), (1j, None), (np.nan, "i8")]
    )
    def test_refused(self, value, dtype):
        """Only a real scalar the dtype can hold makes a constant."""
        with pytest.raises(ParameterError):
            ArraySource.full((2,), value, dtype=dtype)

    def test_id_is_the_contents(self, nans):
        """Value, dtype and shape say which array a constant is; nothing else."""
        assert nans.id == ArraySource.full((2, 3), np.nan).id
        assert nans[:1].id == ArraySource.full((1, 3), np.nan).id
        others = {
            ArraySource.full((2, 3), 0.0).id,
            ArraySource.full((2, 3), np.nan, dtype="float32").id,
            ArraySource.full((3, 3), np.nan).id,
        }
        assert len(others) == 3 and nans.id not in others
        assert replace(nans, base_id="x").id == "x"

    @pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 0])
    def test_json(self, value):
        """A constant survives strict JSON, the non-finite floats included."""
        source = ArraySource.full((2,), value)
        text = json.dumps(source.to_dict(), allow_nan=False)
        out = ArraySource.from_dict(json.loads(text))
        assert out.to_dict() == source.to_dict() and out.id == source.id

    def test_pickle(self, nans):
        """Pickle keeps the value."""
        out = pickle.loads(pickle.dumps(nans))
        assert out.id == nans.id and np.isnan(out.load()).all()

    def test_dict_without_a_value(self, source):
        """A dict written before constants existed still reads back."""
        contents = source.to_dict()
        del contents["value"]
        assert ArraySource.from_dict(contents) == source

    def test_detach(self, nans):
        """Detaching drops the value: the array is no longer that constant."""
        assert not nans.detach().constant
        assert not nans.detach().loadable


class TestCarry:
    """Which operations keep a source which still loads their data."""

    def test_read_select(self, path, patch):
        """A contiguous read selection is composed onto the source."""
        dist = patch.get_array("distance")
        sub = dc.read(path, distance=(dist[100], dist[200]))[0]
        assert sub._source.windows[0] == (100, 201)
        assert np.array_equal(sub._source.load(), sub.data)

    def test_read_not_contiguous(self, path):
        """Samples picked apart are not what the windows alone would load."""
        sub = dc.read(path, distance=np.array([1, 5, 9]), samples=True)[0]
        assert sub.shape[0] == 3 and not sub._source.loadable

    def test_other_data(self, patch):
        """Metadata given data cannot know the source still loads them."""
        out = patch.drop_data().to_patch(patch.data * 2)
        assert not out._source.loadable
        assert out._source == patch._source.detach()

    def test_select(self, patch):
        """Patch selections compose too."""
        time = patch.get_array("time")
        sub = patch.select(time=(time[10], time[50]))
        sub = sub.select(distance=(3, 9), samples=True)
        assert np.array_equal(sub._source.load(), sub.data)

    def test_isel_and_sel(self, patch):
        """A unit-step slice narrows; any other positional index detaches."""
        dist = patch.get_array("distance")
        for sub in (
            patch.isel(time=slice(3, 9)),
            patch.sel(distance=slice(dist[10], dist[50])),
        ):
            assert sub.size and np.array_equal(sub._source.load(), sub.data)
        for index in (3, slice(0, 9, 2), [1, 2], slice(9, 3, -1)):
            assert not patch.isel(time=index)._source.loadable

    def test_meta_dtype(self, patch):
        """Metadata describing another dtype no longer describes the source."""
        meta = patch.drop_data()
        assert meta.update(attrs={"station": "a"})._source.loadable
        assert not meta.update(dtype="float32")._source.loadable

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
        origin = patch._source
        # Which stored array it came from, and no claim to load this one.
        expected = ArraySource(
            origin.path, origin.format, origin.version, origin.key, base_id=origin.id
        )
        assert func(patch)._source == expected
