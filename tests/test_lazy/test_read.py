"""Tests for region loads which merge abutting windows."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

import dascore as dc
from dascore._lazy import read as read_module
from dascore._lazy import recipe as recipe_module
from dascore._lazy.recipe import Recipe, concat
from dascore.core.source import ArraySource
from tests.test_lazy.conftest import FORMAT


@pytest.fixture(scope="module")
def square_source(patch, tmp_path_factory):
    """The source of a DASDAE file holding a square array."""
    path = tmp_path_factory.mktemp("lazy_square") / "square.h5"
    square = patch.select(distance=(0, 20), time=(0, 20), samples=True)
    dc.write(square, path, "dasdae")
    return dc.read(path)[0]._source


class TestCoalescedReads:
    """Abutting windows of one source are read as one window."""

    def test_windows_of_one_file(self, two_sources, patch, reads):
        """Abutting windows of one file take one read."""
        source = two_sources[0]
        many = Recipe.from_sources([source[x : x + 10] for x in range(0, 100, 10)])
        assert np.array_equal(many.load(), patch.data[:100])
        assert len(reads) == 1 and reads[0].windows[0] == (0, 100)

    def test_breaks_split_runs(self, two_sources, patch, reads):
        """A gap, a reordering, a constant or a file change starts a new read."""
        first, second = two_sources
        width = patch.shape[1]
        parts = [first[0:10], first[10:20], first[30:40], first[20:30]]
        parts += [ArraySource.full((2, width), 0.0, dtype=patch.dtype)]
        parts += [first[40:50], second[0:10], second[10:20]]
        expected = [patch.data[a:b] for a, b in [(0, 20), (30, 40), (20, 30)]]
        expected += [np.zeros((2, width)), patch.data[40:50], patch.data[100:120]]
        out = Recipe.from_sources(parts).load()
        assert np.array_equal(out, np.concatenate(expected))
        assert [x.windows[0] for x in reads] == [
            (0, 20),
            (30, 40),
            (20, 30),
            (40, 50),
            (0, 20),
        ]

    def test_casts_split_runs(self, two_sources, patch, reads):
        """Members cast through different dtypes are read apart."""
        source = two_sources[0]
        parts = [source[0:10], source[10:20], source[20:30]]
        array = Recipe.from_sources(parts, dtype="f8", cast_via=[None, "f4", "f4"])
        expected = np.concatenate(
            [patch.data[0:10], patch.data[10:30].astype("f4")]
        ).astype("f8")
        assert np.array_equal(array.load(), expected)
        assert [x.windows[0] for x in reads] == [(0, 10), (10, 30)]

    def test_tiles_merge_along_one_axis(self, two_sources, patch, reads):
        """Tiles merge along one axis only, so each row of tiles takes one read."""
        source = two_sources[0]
        tile = Recipe.from_source
        rows = [
            concat([tile(source[a:b, 0:5]), tile(source[a:b, 5:10])], axis=1)
            for a, b in ((0, 50), (50, 100))
        ]
        grid = concat(rows)
        assert np.array_equal(grid.load(), patch.data[:100, :10])
        assert [x.windows for x in reads] == [((0, 50), (0, 10)), ((50, 100), (0, 10))]

    def test_other_axes_must_match(self, two_sources, patch, reads):
        """Windows which meet on one axis but differ on another are read apart."""
        source = two_sources[0]
        array = Recipe.from_sources([source[0:10, 0:5], source[10:20, 100:105]])
        expected = np.concatenate([patch.data[0:10, 0:5], patch.data[10:20, 100:105]])
        assert np.array_equal(array.load(), expected)
        assert len(reads) == 2

    def test_axis_maps_must_match(self, two_sources, patch, reads):
        """A transposed member is not merged with a plain one of its file."""
        source = two_sources[0]
        plain = Recipe.from_source(source[0:10, 0:10])
        turned = Recipe.from_source(source[0:10, 10:20]).transpose()
        expected = np.concatenate([patch.data[0:10, 0:10], patch.data[0:10, 10:20].T])
        assert np.array_equal(concat([plain, turned]).load(), expected)
        assert len(reads) == 2

    def test_square_transpose_is_not_merged(self, square_source, reads):
        """A transposed window of a square source is not merged with a plain one."""
        data = square_source.load()
        reads.clear()
        plain = Recipe.from_source(square_source[0:5, 0:5])
        turned = Recipe.from_source(square_source[0:5, 5:10]).transpose()
        expected = np.concatenate([data[0:5, 0:5], data[0:5, 5:10].T])
        assert np.array_equal(concat([plain, turned]).load(), expected)
        assert len(reads) == 2

    def test_a_run_turning_is_cut(self, two_sources, patch, reads):
        """Tiles which join along one axis, then another, are read as two runs."""
        whole = two_sources[0]
        boxes = [(0, 1, 0, 1), (0, 2, 1, 2), (1, 2, 0, 1), (2, 3, 0, 1), (2, 3, 1, 2)]
        array = Recipe.from_sources(
            [whole[a:b, c:d] for a, b, c, d in boxes],
            starts=[(a, c) for a, _, c, _ in boxes],
        )
        assert np.array_equal(array.load(), patch.data[0:3, 0:2])
        windows = [x.windows for x in reads]
        assert windows == [
            ((0, 1), (0, 1)),
            ((0, 2), (1, 2)),
            ((1, 3), (0, 1)),
            ((2, 3), (1, 2)),
        ]

    @pytest.mark.parametrize(
        ("dtype", "cast", "count"),
        [("f8", None, 1), ("u4", None, 2), ("f8", "u4", 2), ("f8", "U", 2)],
    )
    def test_conversions_which_vary_are_read_apart(
        self, two_sources, reads, dtype, cast, count
    ):
        """Floats read into integers, or through text, are read a window at a time."""
        source = two_sources[0]
        array = Recipe.from_sources(
            [source[0:1], source[1:2]], dtype=dtype, cast_via=[cast, cast]
        )
        with np.errstate(invalid="ignore"):
            array.load()
        assert len(reads) == count

    @pytest.mark.parametrize(("dtype", "merged"), [("f4", 1), ("S10", 2)])
    def test_only_numbers_merge(self, dtype, merged):
        """Text, whose casts may take their unit from the samples, is read apart."""
        array = Recipe.from_columns(
            ["/a.h5", "/a.h5"],
            (1,),
            start=[[0], [1]],
            extent=2,
            **{**FORMAT, "source_dtype": dtype},
        )
        assert len(read_module._coalesced(array._block())) == merged

    def test_extents_must_match(self):
        """Windows of one source which state different extents are read apart."""
        array = Recipe.from_columns(
            ["/a.h5", "/a.h5"], (4,), start=[[0], [4]], extent=[[8], [9]], **FORMAT
        )
        assert len(read_module._coalesced(array._block())) == 2

    def test_reads_stop_at_the_byte_budget(
        self, two_sources, patch, reads, monkeypatch
    ):
        """A run is cut where a member starts a budget or more into it."""
        source = two_sources[0]
        itemsize = np.dtype(patch.dtype).itemsize
        monkeypatch.setattr(read_module, "_RUN_BYTES", 30 * patch.shape[1] * itemsize)
        spans = [(0, 10), (10, 20), (20, 50), (50, 60), (60, 100)]
        array = Recipe.from_sources([source[a:b] for a, b in spans])
        assert np.array_equal(array.load(), patch.data[:100])
        # Cut where members start in another 30 rows: 0-29, 30-59, 60-89.
        assert [x.windows[0] for x in reads] == [(0, 50), (50, 60), (60, 100)]

    def test_members_over_the_budget_stay_apart(self, two_sources, reads, monkeypatch):
        """Members each bigger than the budget are read one at a time."""
        monkeypatch.setattr(read_module, "_RUN_BYTES", 1)
        array = Recipe.from_sources([two_sources[0][0:10], two_sources[0][10:20]])
        array.load()
        assert len(reads) == 2

    @pytest.mark.parametrize("seed", range(20))
    def test_random_cuts_load_alike(self, two_sources, monkeypatch, seed):
        """Random cuts load the same array with or without coalescing."""
        rng = np.random.default_rng(seed)
        parts = []
        for source in two_sources:
            size = source.shape[0]
            cuts = np.unique(rng.integers(1, size, rng.integers(0, 8)))
            pieces = list(pairwise([0, *cuts.tolist(), size]))
            if rng.random() < 0.3:
                rng.shuffle(pieces)
            parts += [source[a:b, 2:7] for a, b in pieces]
            if rng.random() < 0.3:
                parts.append(ArraySource.full((3, 5), 1.0, dtype=source.dtype))
        array = Recipe.from_sources(parts)
        window = array[int(rng.integers(0, 20)) :]
        got = [array.load(), window.load()]
        monkeypatch.setattr(recipe_module, "_coalesced", lambda block: block)
        assert all(
            np.array_equal(x, y) for x, y in zip(got, [array.load(), window.load()])
        )
