"""Tests for clipping the members of one array's block."""

from __future__ import annotations

from itertools import product

import numpy as np
import pandas as pd
import pytest

from dascore.core.source import ArraySource
from dascore.lazy import block as block_module
from dascore.lazy.recipe import Recipe, concat
from tests.test_lazy.conftest import (
    FORMAT,
    assert_same_members,
    mixed_member,
)


def mixed_stack(rng, count, axis, ndim, laid=True):
    """Return mixed members end to end along an axis, laid there or at corners."""
    sizes = np.full((count, ndim), 2)
    sizes[:, axis] = rng.integers(1, 4, count)
    sources = [mixed_member(rng, x, axis) for x in sizes.tolist()]
    casts = [[None, "f4"][x] for x in rng.integers(0, 2, count)]
    kwargs = {"base_uri": "/root/", "cast_via": casts, "dtype": "f8"}
    if laid:
        return Recipe.from_sources(sources, axis=axis, **kwargs)
    corners = np.zeros((count, ndim), np.int64)
    corners[1:, axis] = np.cumsum(sizes[:-1, axis])
    return Recipe.from_sources(sources, starts=corners, **kwargs)


def random_window(rng, array):
    """Return a random window, often trimming only one axis or on member edges."""
    shape, axis = array.shape, int(rng.integers(array.ndim))
    edges = np.unique(array.to_frame().query("out_axis == @axis")["out_start"])
    out = []
    for dim, size in enumerate(shape):
        if dim != axis and rng.random() < 0.7:
            out.append(slice(None))
            continue
        low, high = np.sort(rng.integers(0, size + 1, 2))
        if rng.random() < 0.3:
            low, high = np.sort(rng.choice([*edges.tolist(), size], 2))
        if rng.random() < 0.1:
            high = low
        out.append(slice(int(low), int(high)))
    return tuple(out)


def takes_ends(view):
    """Whether resolving a view should clip only its first and last member."""
    window = view._view
    if window is None:
        return False
    block, starts, stops = window.block, window.starts, window.stops
    axis = block.concat_axis
    rest = [x for x in range(block.ndim) if x != axis]
    whole = all(starts[x] == 0 and stops[x] == block.shape[x] for x in rest)
    moved = any(starts) or stops != block.shape
    return axis >= 0 and whole and moved and starts[axis] < stops[axis]


class TestClipEnds:
    """A window trimming only the stacked axis clips its end members alone."""

    @pytest.fixture()
    def arrays(self):
        """Arrays stacked along each axis, at corners, and with folded columns."""
        rng = np.random.default_rng(20)
        paths = [f"/root/f{x % 7}.h5" for x in range(300)]
        return [
            mixed_stack(rng, 40, 0, 2),
            mixed_stack(rng, 40, 1, 2),
            mixed_stack(rng, 30, 2, 3),
            mixed_stack(rng, 20, 0, 1),
            mixed_stack(rng, 40, 1, 2, laid=False),
            mixed_stack(rng, 40, 0, 2, laid=False),
            Recipe.from_columns(paths, (2, 3), axis=1, base_uri="/root/", **FORMAT),
            concat([stacked := mixed_stack(rng, 20, 0, 2), stacked], axis=1),
        ]

    def test_matches_general_path(self, arrays, monkeypatch):
        """Random windows, chains and transposes clip as the general path does."""
        rng = np.random.default_rng(21)
        taken, clip_ends = [], block_module._clip_ends
        monkeypatch.setattr(
            block_module, "_clip_ends", lambda *x: taken.append(1) or clip_ends(*x)
        )
        counts = {True: 0, False: 0}
        for array, _ in product(arrays, range(40)):
            steps = [random_window(rng, array)]
            turn, settle = rng.random() < 0.3, rng.random() < 0.3
            if rng.random() < 0.4:
                steps.append(random_window(rng, array[steps[0]]))

            def run(array=array, steps=steps, turn=turn, settle=settle):
                out = array.transpose() if turn else array
                for step in steps:
                    out = out[step]
                    if settle:
                        out.table
                return out

            view = run()
            expected, before = takes_ends(view), len(taken)
            view.table
            assert len(taken) - before == int(expected)
            counts[expected] += 1
            with monkeypatch.context() as patch:
                patch.setattr(block_module, "_trims_ends", lambda *x: False)
                general = run()
                general.table
            assert_same_members(view, general)
            view.validate()
        assert min(counts.values()) > 40

    def test_owns_its_storage(self):
        """A clipped view shares no placement with its base but broadcast values."""
        paths = [f"/root/f{x}.h5" for x in range(300)]
        array = Recipe.from_columns(paths, (4, 3), **FORMAT)

        def placement(table):
            members = table.members
            return [*table.axes.values(), members.source_row, members.cast.codes]

        mine = placement(array[3:-3].table)
        stored = [x for x in mine if any(x.strides)]
        assert len(stored) < len(mine)
        for ours, theirs in product(stored, placement(array.table)):
            assert not np.shares_memory(ours, theirs)

    def test_invalid_members_take_the_general_path(self):
        """Boxes outside the array or empty are clipped as any other window."""
        sources = [ArraySource.full((4, 3), float(x)) for x in range(3)]
        wide = Recipe.from_sources(sources, shape=(12, 2))
        assert wide[1:11].validate().shape == (10, 2)
        frame = wide.to_frame()
        frame.loc[frame["ordinal"] == 1, "out_stop"] = frame["out_start"]
        empty = Recipe.from_frame(frame, (12, 3), "f8")
        assert len(empty[1:11]) == 2

    def test_overlapping_members_take_the_general_path(self, monkeypatch):
        """Members overlapping on the stacked axis are clipped as any others."""
        short = Recipe.from_sources([ArraySource.full((4,), 1)], shape=(2,))
        joined = concat([short, Recipe.from_source(ArraySource.full((4,), 2))])
        fast = joined[3:5]
        fast.table
        monkeypatch.setattr(block_module, "_trims_ends", lambda *_: False)
        general = joined[3:5]
        pd.testing.assert_frame_equal(fast.to_frame(), general.to_frame())
        assert fast.data_id == general.data_id
