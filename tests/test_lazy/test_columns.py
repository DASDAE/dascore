"""Tests for the dictionary encoded columns and the sources table."""

from __future__ import annotations

import tracemalloc
from dataclasses import replace
from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from dascore.core.source import ArraySource
from dascore.lazy import columns as columns_module
from dascore.lazy import identity as identity_module
from dascore.lazy.columns import SOURCE_FIELDS
from dascore.lazy.recipe import Recipe, RecipeTable, concat
from tests.test_lazy.conftest import (
    ALIKE_VALUES,
    FORMAT,
    assert_same_members,
    constant,
    many_members,
    mixed_member,
    mixed_sources,
    stored,
)


class TestFoldedColumns:
    """Columns of one value throughout cost one value, not one per member."""

    @staticmethod
    def many(count=2000):
        """Return many members over a few files, with one-valued columns."""
        return Recipe.from_columns(
            [f"/data/f{x % 97:03d}.h5" for x in range(count)],
            (2, 3),
            key=[["", "k"][x % 3 == 0] for x in range(count)],
            **FORMAT,
        )

    def test_memory_per_member(self):
        """A large array of like members keeps well under a column per field."""
        paths = [f"/data/f{x:05d}.h5" for x in range(20_000)]
        # Count only allocations made under this file, so other threads
        # alive in the worker don't inflate the total.
        tracemalloc.start(100)
        try:
            array = Recipe.from_columns(paths, (1000, 100), **FORMAT)
            snapshot = tracemalloc.take_snapshot()
            assert len(array) == len(paths)
        finally:
            tracemalloc.stop()
        mine = tracemalloc.Filter(True, __file__, all_frames=True)
        traces = snapshot.filter_traces([mine]).traces
        held = sum(trace.size for trace in traces)
        assert held / len(paths) < 120

    def test_folding_changes_nothing(self, monkeypatch):
        """Folded and unfolded tables give the same arrays."""

        def views():
            array = self.many()
            half = array.shape[0] // 2
            return [
                array,
                array[5:900, 1:],
                concat([array[:half], array[half:]]),
                concat([array, constant(array.shape, 2.0)], axis=1),
                *array.rechunk([0, 7, 333, 600]),
            ]

        folded = views()
        monkeypatch.setattr(columns_module, "_FOLD_ROWS", 10**9)
        for got, expected in zip(folded, views(), strict=True):
            assert got.data_id == expected.data_id
            assert got.sources == expected.sources
            pd.testing.assert_frame_equal(got.to_frame(), expected.to_frame())
            if got.table.members.filled.all():
                assert np.array_equal(got.load(), expected.load())


def reference_encoding(values):
    """Return the distinct values in first-seen order and each one's code."""
    index = {}
    codes = [index.setdefault(x, len(index)) for x in values]
    return list(index), codes


class TestStringColumns:
    """Strings are dictionary encoded in one vectorized pass."""

    values = ("b", "", "a", "b", "é", "", "日本", "a", "é")

    def test_matches_the_loop(self):
        """The distinct values and codes are those a dict would give."""
        column = columns_module._Column.of(self.values)
        distinct, codes = reference_encoding(self.values)
        assert list(column.values) == distinct
        assert column.codes.tolist() == codes
        assert column.codes.dtype == np.int32

    @pytest.mark.parametrize("kind", ["numpy", "pandas"])
    def test_array_input(self, kind):
        """Numpy and pandas strings give plain python strings."""
        array = np.array(self.values)
        if kind == "pandas":
            array = pd.Series(list(self.values)).astype(str)
        column = columns_module._Column.of(array)
        assert list(column.values) == reference_encoding(self.values)[0]
        assert all(type(x) is str for x in column.values)
        assert type(column.codes) is np.ndarray

    def test_scalars_keep_their_type(self):
        """Equal values of another type or sign stay distinct."""
        column = columns_module._Column.of([1, True, 1.0, -0.0, 0.0, None, "1"])
        assert len(column.values) == 7

    @pytest.mark.parametrize("repeats", [1, 100])
    def test_missing_strings_stay_distinct(self, repeats):
        """A missing value among strings is a value of its own."""
        keys = pd.Series(["k1", None, "k2"] * repeats)
        array = Recipe.from_columns("/a", (2,), key=keys, **FORMAT)
        members = array.table.members
        key = members.sources.key
        assert key.codes[members.source_row].tolist() == [0, 1, 2] * repeats
        assert key.values[0] == "k1" and key.values[2] == "k2"
        assert pd.isna(key.values[1])

    @pytest.mark.parametrize("count", [1, 5, 255, 256, 500])
    def test_any_length(self, count):
        """Short and long columns encode alike, in first-seen order."""
        values = [f"v{-x % 7}" for x in range(count)]
        column = columns_module._Column.of(values)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )
        assert all(type(x) is str for x in column.values)

    def test_a_nul_suffix_stays_distinct(self):
        """A string is not merged with its own prefix before a NUL."""
        values = ["a", "a\0b", *[f"v{x}" for x in range(300)]]
        column = columns_module._Column.of(values)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )

    def test_nul_strings_stay_distinct(self):
        """Strings which differ after a NUL byte stay distinct at any length."""
        values = ["a\0b", "a\0c"] * 200
        column = columns_module._Column.of(values)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )

    @pytest.mark.parametrize("count", [3, 300])
    def test_series_index_ignored(self, count):
        """A series is read by position, whatever its index."""
        values = [f"k{x % 2}" for x in range(count)]
        series = pd.Series(values, index=np.arange(count) + 10)
        column = columns_module._Column.of(series)
        assert (list(column.values), column.codes.tolist()) == reference_encoding(
            values
        )

    def test_all_none(self):
        """A column of None is one value."""
        column = columns_module._Column.of([None] * 5)
        assert column.values == (None,)
        assert not column.codes.any()


class TestSourceTable:
    """Members read one sources table, which views share until a join cuts it."""

    def test_views_share_it(self):
        """Slicing, joining and rechunking keep the one sources table."""
        array = many_members()
        sources = array.table.members.sources
        half = array.shape[0] // 2
        views = [
            array[10:500],
            concat([array[:half], array[half:]]),
            *array.rechunk([0, 7, 333, 600]),
            RecipeTable.from_arrays([array[:half], array[half:]])[1],
        ]
        assert all(x.table.members.sources is sources for x in views)

    def test_built_apart_is_built_together(self):
        """Arrays built apart and joined are the array built from all of them."""
        sources = [*mixed_sources(), *mixed_sources()[::-1]]
        together = Recipe.from_sources(sources)
        apart = concat([Recipe.from_sources(x) for x in (sources[:5], sources[5:])])
        assert apart.data_id == together.data_id
        # A nan is not equal to itself, so the sources are compared as written.
        assert repr(apart.sources) == repr(together.sources)
        pd.testing.assert_frame_equal(apart.to_frame(), together.to_frame())

    def test_compacted_when_little_is_read(self):
        """A join of a few members of a big table keeps only the rows read."""
        array = many_members()
        parts = [array[0:4], array[-6:]]
        joined = concat(parts)
        assert len(joined.table.members.sources) < len(array.table.members.sources)
        assert len(joined.table.members.sources) <= 5
        sources = [*parts[0].sources, *parts[1].sources]
        rebuilt = Recipe.from_sources(sources, base_uri="/root/")
        assert joined.data_id == rebuilt.data_id
        assert joined.sources == rebuilt.sources
        pd.testing.assert_frame_equal(joined.to_frame(), rebuilt.to_frame())

    def test_strings_round_trip(self):
        """Paths and origins of any text are read back exactly."""
        texts = ["/root/日本/é.h5", "/root/b.h5", "/a.h5", "/root/"]
        origins = ["", "ü" * 3, "", "a" * 32]
        sources = [
            ArraySource(path=x, format="DASDAE", origin_id=y).describe((2,), "f4")
            for x, y in zip(texts, origins)
        ]
        sources.insert(2, ArraySource.full((2,), 0.0))
        array = Recipe.from_sources(sources, base_uri="/root/")
        assert array.sources == tuple(sources)
        assert array.sources[2].path == ""
        back = Recipe.from_frame(array.to_frame(), array.shape, array.dtype)
        assert back.sources == tuple(sources)
        assert back.data_id == array.data_id
        first = Recipe.from_sources(sources[:2], base_uri="/root/")
        joined = concat([array[4:], first, array[8:]])
        assert joined.sources == (*sources[2:], *sources[:2], sources[-1])

    def test_undecodable_path(self):
        """A path holding an escaped undecodable byte is read back as it was."""
        source = stored((2,), path="/root/\udcff.h5")
        array = Recipe.from_sources([source, source], base_uri="/root/")
        assert array.sources == (source, source)
        joined = concat([array, Recipe.from_source(source), array[2:]])
        assert joined.sources == (source,) * 4

    def test_constant_values_stay_apart(self):
        """Constants of one dtype whose values compare equal are not one source."""
        values = [1, True, 1.0, -0.0, 0.0, float("nan")]
        sources = [
            ArraySource(filled=True, value=x).describe((1, 2), "f8") for x in values
        ]
        array = Recipe.from_sources([stored((1, 2), dtype="f8"), *sources])
        assert len(array.table.members.sources) == len(values) + 1
        ids = [array[0 : x + 2].data_id for x in range(len(values))]
        assert len(set(ids)) == len(values)

    def test_one_source_two_casts(self):
        """Members of one source row keep their own casts."""
        source = stored((2,), path="/a/c.h5")
        casts = [None, "f4", "f8"]
        together = Recipe.from_sources([source] * 3, cast_via=casts, dtype="f8")
        assert len(together.table.members.sources) == 1
        apart = [Recipe.from_sources([source], cast_via=[x], dtype="f8") for x in casts]
        assert together.data_id == concat(apart).data_id
        uncast = Recipe.from_sources([source] * 3, dtype="f8")
        assert together.data_id != uncast.data_id

    def test_whole_member_of_a_later_source(self):
        """A member left whole is named as its own source, whichever row it reads."""
        first = stored((2,), path="/a/f4.h5", dtype="f4")
        second = stored((2,), path="/a/f8.h5", dtype="f8")
        cut = Recipe.from_sources([first, second], dtype="f8")[2:]
        assert cut.data_id == Recipe.from_sources([second]).data_id == second.data_id

    def test_whole_constant_of_a_later_source(self):
        """A constant cut from a mixed array is named as the constant alone."""
        filled = ArraySource.full((2,), 5.0)
        cut = Recipe.from_sources([stored((2,), dtype="f8"), filled])[2:]
        assert cut.data_id == Recipe.from_sources([filled]).data_id

    def test_empty_frame_columns(self):
        """An array of no members has the text columns of any other."""
        array = Recipe.from_sources([stored((2,)), stored((2,), path="/c.h5")])
        expected = array.to_frame().dtypes
        for empty in (array[0:0], concat([array[0:0], constant((0,))])):
            assert empty.to_frame().dtypes.equals(expected)

    def test_join_merges_repeated_sources(self):
        """Windows of one file built apart and joined read one source row."""
        whole = stored((1000,), path="/a/one.h5")
        windows = [whole[x : x + 10] for x in range(0, 1000, 10)]
        joined = concat([Recipe.from_source(x) for x in windows])
        assert len(joined.table.members.sources) == 1
        assert joined.data_id == Recipe.from_sources(windows).data_id

    def test_copies_are_named_once(self, monkeypatch):
        """Copies of one array joined keep, and name, one row per source."""
        copies = [many_members(30) for _ in range(20)]
        expected = Recipe.from_sources([x for y in copies for x in y.sources]).data_id
        joined = concat(copies)
        assert len(joined.table.members.sources) == 30
        hashed = []
        original = identity_module.H
        monkeypatch.setattr(
            identity_module, "H", lambda x, y: hashed.append(x) or original(x, y)
        )
        assert joined.data_id == expected
        assert hashed.count("location") == 30

    def test_most_read_is_kept_whole(self):
        """A join reading most of a table shares it rather than cutting it."""
        paths = [f"/a/f{x}.h5" for x in range(300)]
        array = Recipe.from_columns(paths, (2,), **FORMAT)
        joined = concat([array[:-2], array[:0]])
        assert joined.table.members.sources is array.table.members.sources


def reference_merge(columns):
    """Merge columns a dictionary value at a time, as the loop once did."""
    index, values, parts = {}, [], []
    for column in columns:
        codes, dictionary = column.codes, column.values
        if len(dictionary) > len(codes):
            used, codes = np.unique(codes, return_inverse=True)
            dictionary = [dictionary[x] for x in used.tolist()]
        remap = []
        for value in dictionary:
            key = columns_module._dict_key(value)
            if key not in index:
                index[key] = len(values)
                values.append(value)
            remap.append(index[key])
        parts.append(np.array(remap, np.int64)[codes])
    return values, np.concatenate([np.empty(0, np.int64), *parts])


class TestMergedDictionaries:
    """Joined columns merge every dictionary in one pass, as one loop would."""

    pool = (*ALIKE_VALUES, None, "a", "b", "")

    def random_columns(self, rng):
        """Return columns built apart, and cut from two big shared ones."""
        texts = [f"t{x}" for x in range(400)]
        shared = [
            columns_module._Column.of([self.pool[x] for x in rng.integers(0, 10, 50)]),
            columns_module._Column.of([texts[x] for x in rng.integers(0, 400, 600)]),
        ]
        out = []
        for _ in range(int(rng.integers(0, 12))):
            kind = int(rng.integers(4))
            if kind < 2:
                big = shared[kind]
                rows = rng.integers(0, len(big.codes), int(rng.integers(0, 80)))
                out.append(big.take(rows))
            elif kind == 2:
                picks = rng.integers(0, len(self.pool), int(rng.integers(0, 6)))
                out.append(columns_module._Column.of([self.pool[x] for x in picks]))
            else:
                picks = rng.integers(0, 400, int(rng.integers(200, 400)))
                out.append(columns_module._Column.of([texts[x] for x in picks]))
        return out

    def test_matches_the_loop(self):
        """Values, their order and every code are those the loop gave."""
        rng = np.random.default_rng(22)
        for _ in range(300):
            columns = self.random_columns(rng)
            merged = columns_module._merge_columns(columns)
            values, codes = reference_merge(columns)
            assert repr(list(merged.values)) == repr(values)
            assert merged.codes.tolist() == codes.tolist()
            assert merged.codes.dtype == np.int32

    def test_selected_rows(self):
        """Rows selected on the way merge as those rows taken first."""
        rng = np.random.default_rng(23)
        for _ in range(100):
            columns = self.random_columns(rng)
            rows = [rng.integers(0, len(x.codes), 5) for x in columns if len(x.codes)]
            columns = [x for x in columns if len(x.codes)]
            merged = columns_module._merge_columns(columns, rows)
            taken = [x.take(y) for x, y in zip(columns, rows)]
            expected = columns_module._merge_columns(taken)
            assert repr(merged.values) == repr(expected.values)
            assert np.array_equal(merged.codes, expected.codes)

    def test_shared_dictionary_read_once(self, monkeypatch):
        """Columns sharing one dictionary contribute it once, cut if it came first."""
        big = columns_module._Column.of([f"v{x}" for x in range(50)])
        sizes, encode = [], columns_module._Column.of
        monkeypatch.setattr(
            columns_module._Column, "of", lambda x: sizes.append(len(x)) or encode(x)
        )
        cut = big.take([3, 3])
        merged = columns_module._merge_columns([cut, big, big, cut, big])
        assert sizes == [1 + 50]
        assert merged.values == big.values[3:4] + big.values[:3] + big.values[4:]

    def test_joins_apart_as_built_together(self):
        """Many arrays built apart join to the array built from all their sources."""
        rng = np.random.default_rng(24)
        sources = [mixed_member(rng, (2, 3), 0) for _ in range(400)]
        casts = [[None, "f4"][x] for x in rng.integers(0, 2, len(sources))]
        kwargs = {"base_uri": "/root/", "dtype": "f8"}
        together = Recipe.from_sources(sources, cast_via=casts, **kwargs)
        bounds = [0, *np.sort(rng.choice(np.arange(1, 400), 150, replace=False)), 400]
        apart = concat(
            [
                Recipe.from_sources(sources[a:b], cast_via=casts[a:b], **kwargs)
                for a, b in pairwise(bounds)
            ]
        )
        assert_same_members(apart.validate(), together)
        for name in SOURCE_FIELDS:
            ours = getattr(apart.table.members.sources, name)
            theirs = getattr(together.table.members.sources, name)
            if name in columns_module._STRING_FIELDS:
                assert ours.at(slice(None)) == theirs.at(slice(None))
            else:
                assert repr(ours.values) == repr(theirs.values)
                assert np.array_equal(ours.codes, theirs.codes)

    def test_cut_from_one_table(self):
        """Few members of a big table join to those members built together."""
        sources = [
            ArraySource.full((2, 3), float(x))
            if x % 3 == 0
            else replace(stored((2, 3), path=f"/root/f{x % 5}.h5"), key=f"k{x}")
            for x in range(600)
        ]
        big = Recipe.from_sources(sources, base_uri="/root/")
        other = Recipe.from_sources(sources[5:9], base_uri="/root/")
        parts = [big[x : x + 6] for x in (0, 400, 1100)]
        joined = concat([parts[0], other, parts[1], other, parts[2]])
        rows = [*range(0, 3), *range(5, 9), *range(200, 203), *range(5, 9)]
        rebuilt = Recipe.from_sources(
            [sources[x] for x in [*rows, *range(550, 553)]], base_uri="/root/"
        )
        # The rows read are kept in the order the big table held them.
        assert_same_members(joined.validate(), rebuilt, stored=False)
        # Only the values the rows read are kept.
        kept = joined.table.members.sources
        assert len(kept.key.values) == 1 + 9 and len(kept.value.values) == 1 + 4
