"""Fixtures and helpers the lazy array tests share."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from itertools import pairwise, product
from types import MappingProxyType

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.io import core as io_core
from dascore.lazy.block import AXIS_FIELDS
from dascore.lazy.recipe import Recipe
from dascore.utils.downloader import fetch
from dascore.utils.identity import operation_context


@pytest.fixture(scope="module")
def path():
    """A DASDAE file holding one two dimensional patch."""
    return fetch("example_dasdae_event_1.h5")


@pytest.fixture(scope="module")
def patch(path):
    """The patch read from the file; its axes have different lengths."""
    return dc.read(path)[0]


@pytest.fixture(scope="module")
def source(patch):
    """The source the framework gave the patch."""
    return patch._source


@pytest.fixture(scope="module")
def lazy(source):
    """The whole of the stored array, as one member."""
    return Recipe.from_source(source)


@pytest.fixture(scope="module")
def two_files(patch, tmp_path_factory):
    """Two DASDAE files holding windows of different lengths."""
    directory = tmp_path_factory.mktemp("lazy_array")
    dim = patch.dims[0]
    out = []
    for index, span in enumerate(((0, 100), (100, None))):
        window = patch.select(**{dim: span}, samples=True)
        path = directory / f"part_{index}.h5"
        dc.write(window, path, "dasdae")
        out.append(path)
    return out


@pytest.fixture(scope="module")
def two_sources(two_files):
    """The source of each of the two files."""
    return [dc.read(x)[0]._source for x in two_files]


@pytest.fixture(scope="module")
def joined(two_sources):
    """The two files, joined along their first axis."""
    return Recipe.from_sources(two_sources)


@pytest.fixture()
def reads(monkeypatch):
    """Record each source the loader reads, whatever reads it."""
    calls = []
    original = io_core._open_array_reader

    @contextmanager
    def recorded(source):
        with original(source) as load:
            yield lambda x: calls.append(x) or load(x)

    monkeypatch.setattr(io_core, "_open_array_reader", recorded)
    return calls


@pytest.fixture(scope="module")
def parts():
    """Three arrays of one shape, each a different constant."""
    return [constant((2, 3, 4), float(x)) for x in range(3)]


def constant(shape, value=1.0, dtype=None):
    """Return a lazy array of one constant member."""
    return Recipe.from_source(ArraySource.full(shape, value, dtype))


def stored(shape, path="/a/b.h5", origin_id="", key="", dtype=np.float32):
    """Return a source for a whole stored array which is never opened."""
    source = ArraySource(
        path=path, format="DASDAE", version="1", origin_id=origin_id, key=key
    )
    return source.describe(shape, dtype)


def placed(sources, starts, shape):
    """Return the array which puts each source at a corner."""
    return Recipe.from_sources(sources, starts=np.array(starts), shape=shape)


def random_boxes(rng, shape, splits):
    """Return boxes which tile a shape, cut from it one split at a time."""
    boxes = [(np.zeros(len(shape), np.int64), np.asarray(shape, np.int64))]
    for _ in range(splits):
        low, high = boxes.pop(int(rng.integers(len(boxes))))
        wide = [x for x in range(len(shape)) if high[x] - low[x] > 1]
        if not wide:
            boxes.append((low, high))
            continue
        axis = int(rng.choice(wide))
        cut = int(rng.integers(low[axis] + 1, high[axis]))
        first, second = high.copy(), low.copy()
        first[axis], second[axis] = cut, cut
        boxes += [(low, first), (second, high)]
    return boxes


def grid(shape, edges):
    """Return the boxes of a grid cut at the given edges of each axis."""
    spans = [list(pairwise(sorted({0, *x, shape[a]}))) for a, x in enumerate(edges)]
    return [
        (np.array([a for a, _ in corner]), np.array([b for _, b in corner]))
        for corner in product(*spans)
    ]


# The format arguments every from_columns test shares.
FORMAT = MappingProxyType({"format": "DASDAE", "version": "1", "source_dtype": "f4"})


def mixed_sources():
    """Files under and outside a base, keys, origins and constants; mixed adds casts."""
    whole = ArraySource(path="/root/a.h5", format="DASDAE", version="1")
    return [
        whole.describe((3, 2), "f4"),
        replace(whole, key="k1").describe((5, 2), "i2")[1:3],
        ArraySource(path="/elsewhere/é.h5", format="H5Simple", version="2").describe(
            (2, 2), "u1"
        ),
        replace(whole, path="/root/b.h5", origin_id="a" * 32).describe((4, 2), "f4")[
            0:2
        ],
        ArraySource.full((2, 2), 1, dtype="i8"),  # int64 on every platform
        ArraySource(filled=True, value=True).describe((1, 2), "f8"),
        ArraySource(filled=True, value=-0.0).describe((1, 2), "f8"),
        ArraySource(filled=True, value=float("nan")).describe((1, 2), "f8"),
    ]


def mixed():
    """Return an array of every kind of member, some cast on the way."""
    casts = [None, "f8", None, "f4", None, None, "f4", None]
    return Recipe.from_sources(mixed_sources(), base_uri="/root/", cast_via=casts)


def many_members(count=300):
    """Return an array of many members over a few files, some paths repeated."""
    return Recipe.from_columns(
        [f"/root/f{x % 97:03d}.h5" for x in range(count)],
        (2, 3),
        format="DASDAE",
        version="1",
        source_dtype=["f4", "i2"] * (count // 2),
        key=[["", "k"][x % 3 == 0] for x in range(count)],
        base_uri="/root/",
    )


# Constant values which compare equal but are distinct sources.
ALIKE_VALUES = (1, 1.0, True, -0.0, 0.0, float("nan"))


def mixed_member(rng, size, axis):
    """Return a random constant, or a window of a file with a key or origin id."""
    kind = int(rng.integers(4))
    if kind == 0:
        value = ALIKE_VALUES[int(rng.integers(len(ALIKE_VALUES)))]
        return ArraySource(filled=True, value=value).describe(tuple(size), "f8")
    whole = ArraySource(
        path=f"/root/f{rng.integers(4)}.h5",
        format="DASDAE",
        version="1",
        key=["", "k"][kind == 2],
        origin_id=["", "a" * 32][kind == 3],
    )
    extent = list(size)
    extent[axis] += 2
    first = int(rng.integers(3))
    window = [slice(None)] * len(size)
    window[axis] = slice(first, first + size[axis])
    return whole.describe(tuple(extent), "f4")[tuple(window)]


def assert_same_members(array, expected, stored=True):
    """Assert two arrays have the same ids, frames, sources and, maybe, storage."""
    assert array.data_id == expected.data_id
    pd.testing.assert_frame_equal(array.to_frame(), expected.to_frame())
    # A nan is not equal to itself, so the sources are compared as written.
    assert repr(array.sources) == repr(expected.sources)
    if not stored:
        return
    first, second = array._block(), expected._block()
    assert first.shape == second.shape and first.dtype == second.dtype
    assert first.concat_axis == second.concat_axis
    for name in AXIS_FIELDS:
        assert first.axes[name].dtype == second.axes[name].dtype
        assert np.array_equal(first.axes[name], second.axes[name])
    ours, theirs = first.members, second.members
    assert np.array_equal(ours.source_row, theirs.source_row)
    assert ours.cast.values == theirs.cast.values
    assert np.array_equal(ours.cast.codes, theirs.cast.codes)


@pytest.fixture()
def recipe_loads(monkeypatch):
    """Record each recipe which is loaded, whatever loads it."""
    calls = []
    original = Recipe.load

    def recorded(self):
        calls.append(self)
        return original(self)

    monkeypatch.setattr(Recipe, "load", recorded)
    return calls


def stored_patch(directory, data, name="stored"):
    """Write the example patch holding `data` and read it back from the file."""
    path = directory / f"{name}.h5"
    dc.write(dc.get_example_patch().new(data=data), path, "dasdae")
    return dc.read(path)[0]


def lazy_patch(stored):
    """The stored patch holding a recipe of its file, under the same attrs."""
    # Inside an operation, so the patch keeps the stored patch's id.
    with operation_context():
        return stored.to_patch(Recipe.from_source(stored._source))


@pytest.fixture(scope="module")
def stored_float(tmp_path_factory):
    """A stored float patch with negative values and every cell distinct."""
    shape = dc.get_example_patch().shape
    values = np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    directory = tmp_path_factory.mktemp("stored_float")
    return stored_patch(directory, values - values.mean())
