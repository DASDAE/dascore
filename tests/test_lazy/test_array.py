"""Tests for a lazy array: a recipe and the operations pending on it."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.source import ArraySource
from dascore.exceptions import PatchError
from dascore.lazy.array import LazyArray
from dascore.lazy.chain import Chain, Node
from dascore.lazy.recipe import Recipe
from dascore.proc.basic import Abs, Imag
from dascore.utils.array_api import backend_name


class Narrowing(dc.PatchProcessor):
    """A kernel which computes something its metadata does not say."""

    name = None
    shape: bool = False

    def kernel(self, data):
        """Halve the dtype, or drop all but the first row."""
        return data[:1] if self.shape else data.astype(np.float32)


@pytest.fixture()
def recipe():
    """A recipe of a constant, negative array."""
    return Recipe.from_source(ArraySource.full((4, 3), -2.0))


def _step(processor, shape, dtype, data_id="step-id", **plan):
    """A settled step."""
    return Node(processor, plan).settled(shape, dtype, data_id)


class TestEmptyChain:
    """With nothing pending the array is the recipe's."""

    def test_facts_are_the_recipes(self, recipe):
        """Shape, dtype, size and id are read from the recipe."""
        array = LazyArray(recipe)
        assert array.recipe is recipe
        assert len(array.chain) == 0
        assert array.shape == recipe.shape == (4, 3)
        assert array.dtype == recipe.dtype
        assert array.ndim == 2
        assert array.size == 12
        assert array.data_id == recipe.data_id

    def test_load_is_the_recipes(self, recipe):
        """Nothing pending, so loading reads the recipe."""
        assert np.array_equal(LazyArray(recipe).load(), recipe.load())

    def test_backend(self, recipe):
        """The array api names the lazy backend."""
        array = LazyArray(recipe)
        assert backend_name(array) == "lazy"
        assert array.__array_namespace__().__name__ == "lazy"


class TestWithSteps:
    """With steps pending the array is the last step's."""

    def test_facts_are_the_last_steps(self, recipe):
        """Shape, dtype and id come from the last step, not the recipe."""
        array = LazyArray(recipe)._appended(_step(Imag(), (4, 3), np.float32, "id"))
        assert array.shape == (4, 3)
        assert array.dtype == np.float32
        assert array.data_id == "id"
        assert array.recipe is recipe

    def test_load_runs_each_step(self, recipe):
        """Each step runs its numpy kernel on the one before."""
        chain = Chain().append(_step(Abs(), (4, 3), np.float64))
        chain = chain.append(_step(Imag(), (4, 3), np.float64))
        array = LazyArray(recipe, chain)
        assert np.array_equal(array.load(), np.zeros((4, 3)))
        assert np.array_equal(np.asarray(array), np.zeros((4, 3)))

    def test_asarray_dtype(self, recipe):
        """Numpy may ask for another dtype."""
        array = LazyArray(recipe)._appended(_step(Abs(), (4, 3), np.float64))
        out = np.asarray(array, dtype=np.int8)
        assert out.dtype == np.int8
        assert np.array_equal(out, np.full((4, 3), 2))

    def test_settled_replaces_the_pending_step(self, recipe):
        """Settling the last step leaves the others and the recipe."""
        first = _step(Abs(), (4, 3), np.float64, "first")
        array = LazyArray(recipe, Chain().append(first))._appended(Node(Imag()))
        settled = array._settled((4, 3), np.float64, "second")
        assert [x.data_id for x in settled.chain] == ["first", "second"]
        assert settled.chain.nodes[0] is first
        assert array.chain.last.pending

    def test_restamped_only_the_given_id(self, recipe):
        """Only a last step settled under the old id takes the new one."""
        array = LazyArray(recipe)._appended(_step(Abs(), (4, 3), np.float64, "old"))
        assert array._restamped("old", "new").data_id == "new"
        assert array._restamped("other", "new") is array
        empty = LazyArray(recipe)
        assert empty._restamped("old", "new") is empty

    def test_repr_counts_steps(self, recipe):
        """The repr says what it is without reading anything."""
        array = LazyArray(recipe)._appended(_step(Abs(), (4, 3), np.float64))
        assert repr(array) == "LazyArray(shape=(4, 3), dtype=float64, steps=1)"


class TestLoadChecks:
    """A step must compute what its metadata stated."""

    def test_pending_step_refused(self, recipe):
        """A step nothing settled says nothing about what it made."""
        array = LazyArray(recipe)._appended(Node(Abs()))
        with pytest.raises(PatchError, match="Abs step was never settled"):
            array.load()

    def test_wrong_dtype(self, recipe):
        """A dtype other than the stated one is refused, naming the processor."""
        array = LazyArray(recipe)._appended(_step(Narrowing(), (4, 3), np.float64))
        with pytest.raises(PatchError, match=r"Narrowing computed .* float32"):
            array.load()

    def test_wrong_shape(self, recipe):
        """So is a shape other than the stated one."""
        step = _step(Narrowing(shape=True), (4, 3), np.float64)
        with pytest.raises(PatchError, match=r"shape \(1, 3\)"):
            LazyArray(recipe)._appended(step).load()


class TestLenAndBool:
    """`len` is the first axis and `bool` is numpy's."""

    def test_len_is_first_axis(self, recipe):
        """Not the number of members, which is the recipe's length."""
        joined = Recipe.from_sources([ArraySource.full((2, 3), 1.0)] * 3)
        assert len(joined) == 3
        assert len(LazyArray(joined)) == 6

    def test_len_of_zero_dimensions(self, recipe):
        """A scalar has no length."""
        array = LazyArray(recipe)._appended(_step(Abs(), (), np.float64))
        with pytest.raises(TypeError, match="unsized"):
            len(array)

    def test_bool_of_many_refused(self, recipe):
        """More than one element is ambiguous; len does not answer for it."""
        with pytest.raises(ValueError, match="12 elements is ambiguous"):
            bool(LazyArray(recipe))

    def test_bool_of_none_refused(self):
        """So is an empty array."""
        empty = Recipe.from_source(ArraySource.full((0,), 1.0))
        with pytest.raises(ValueError, match="0 elements"):
            bool(LazyArray(empty))

    @pytest.mark.parametrize("value", (0.0, 3.0))
    def test_bool_of_one(self, value):
        """One element is its own truth."""
        one = Recipe.from_source(ArraySource.full((1,), value))
        assert bool(LazyArray(one)) is bool(value)
