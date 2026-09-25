"""Benchmarks for lazy arrays, which are all member table work."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.lazy_array import LazyArray, concat, stack
from dascore.core.source import ArraySource

# How many members the big table holds, and how long each one is.
MEMBERS = 100_000
ROWS = 10
WIDTH = 1001


@pytest.fixture(scope="module")
def sources():
    """A thousand windows of one stored array."""
    whole = ArraySource(
        path="/data/file.h5", format="DASDAE", version="1", origin_id="a" * 32
    ).describe((10_000, WIDTH), np.float32)
    return [whole[x : x + ROWS] for x in range(0, 10_000, ROWS)]


@pytest.fixture(scope="module")
def array(sources):
    """One array of many members, built by joining windows end to end."""
    part = LazyArray.from_sources(sources)
    return concat([part] * (MEMBERS // len(sources)), axis=0)


@pytest.fixture(scope="module")
def scattered(sources):
    """One array of many members whose windows never join up."""
    spread = sources[::2]
    part = LazyArray.from_sources(spread)
    return concat([part] * (MEMBERS // len(spread)), axis=0)


@pytest.fixture(scope="module")
def grid(sources):
    """One array whose members tile a grid rather than stacking up."""
    spread = sources[::2]
    part = LazyArray.from_sources(spread)
    column = concat([part] * (MEMBERS // 2 // len(spread)), axis=0)
    return concat([column, column], axis=1)


@pytest.fixture(scope="module")
def deep():
    """One array of many members along one axis of a twelve axis source."""
    whole = ArraySource(
        path="/data/deep.h5", format="DASDAE", version="1", origin_id="b" * 32
    ).describe((1000, *[2] * 11), np.float32)
    return LazyArray.from_sources([whole[x : x + 1] for x in range(1000)])


@pytest.fixture(scope="module")
def fragments(tmp_path_factory):
    """One small stored file, read as 256 one-row members."""
    path = tmp_path_factory.mktemp("lazy_load") / "fragments.h5"
    patch = dc.get_example_patch("random_das", shape=(256, 1024))
    dc.write(patch, path, "DASDAE")
    source = dc.read(path)[0]._source
    return LazyArray.from_sources([source[x : x + 1] for x in range(256)])


@pytest.fixture(scope="module")
def constants():
    """Many small arrays, each one constant member."""
    return [LazyArray.from_source(ArraySource.full((ROWS, WIDTH), 1.0))] * 1_000


class TestTableBenchmarks:
    """Benchmarks of the operations which run over members."""

    @pytest.mark.benchmark
    def test_build(self, sources):
        """Time building a table from sources."""
        LazyArray.from_sources(sources)

    @pytest.mark.benchmark
    def test_slice_few_members(self, array):
        """Time a window which touches four members of a big table."""
        for _ in range(100):
            array[0 : 4 * ROWS]

    @pytest.mark.benchmark
    def test_slice_every_member(self, array):
        """Time a window which touches every member."""
        array[1 : array.shape[0] - 1]

    @pytest.mark.benchmark
    def test_concat(self, constants):
        """Time joining a thousand arrays in one pass."""
        concat(constants, axis=0)

    @pytest.mark.benchmark
    def test_stack(self, constants):
        """Time stacking a thousand arrays on a new axis."""
        stack(constants, axis=0)

    @pytest.mark.benchmark
    def test_data_id(self, scattered):
        """Time naming an array whose members cannot be merged."""
        table = scattered.table
        table._ids.clear()
        assert scattered.data_id

    @pytest.mark.benchmark
    def test_data_id_grid(self, grid):
        """Time naming an array whose members are not one stack of slabs."""
        grid.table._ids.clear()
        assert grid.data_id

    @pytest.mark.benchmark
    def test_data_id_many_dimensions(self, deep):
        """Time naming an array of twelve axes, one of which was cut."""
        deep.table._ids.clear()
        assert deep.data_id

    @pytest.mark.benchmark
    def test_validate(self, array):
        """Time checking that the members tile the array."""
        array.validate()

    @pytest.mark.benchmark
    def test_rechunk(self, array):
        """Time cutting an array into sixty times bigger pieces."""
        array.rechunk(np.arange(0, array.shape[0] + 1, 60 * ROWS))

    @pytest.mark.benchmark
    def test_to_frame(self, array):
        """Time the long frame of a big table."""
        array.to_frame()


class TestLoadBenchmarks:
    """Benchmarks of reading members from storage."""

    @pytest.mark.benchmark
    def test_load_members_of_one_file(self, fragments):
        """Time loading many small members of one file."""
        fragments.load()
