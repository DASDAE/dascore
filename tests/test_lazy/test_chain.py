"""Tests for the steps pending on a lazy array."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dascore.exceptions import PatchError
from dascore.lazy.chain import Chain, Node
from dascore.proc.basic import Abs, Imag


class TestNode:
    """One step."""

    def test_pending_until_settled(self):
        """A node knows what it made only once settled."""
        node = Node(Abs(), {})
        assert node.pending
        assert node.shape is node.dtype is node.data_id is None
        settled = node.settled([3, 2], "float32", "an-id")
        assert not settled.pending
        assert settled.shape == (3, 2)
        assert settled.dtype == np.dtype("float32")
        assert settled.data_id == "an-id"
        assert settled.processor is node.processor
        assert settled.made() == ((3, 2), np.dtype("float32"), "an-id")
        # Settling makes a new node.
        assert node.pending

    def test_pending_made_nothing(self):
        """A pending step refuses to say what it made."""
        with pytest.raises(PatchError, match="Abs step was never settled"):
            Node(Abs()).made()

    def test_empty_id_is_settled(self):
        """With ids disabled a step's id is empty, which is still settled."""
        assert not Node(Abs()).settled((1,), np.float64, "").pending

    def test_frozen(self):
        """A node is not changed in place."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            Node(Abs()).data_id = "x"

    def test_compared_by_identity(self):
        """A plan may hold arrays, so two nodes are equal only if the same."""
        plan = {"weights": np.arange(3)}
        node = Node(Abs(), plan)
        assert node == node
        assert node != Node(Abs(), plan)


class TestChain:
    """The steps in order."""

    def test_empty(self):
        """An empty chain has no last step."""
        chain = Chain()
        assert len(chain) == 0
        assert chain.last is None
        assert list(chain) == []

    def test_append_makes_a_new_chain(self):
        """Appending leaves the chain it started from as it was."""
        first, second = Node(Abs()), Node(Imag())
        one = Chain().append(first)
        two = one.append(second)
        assert list(one) == [first]
        assert list(two) == [first, second]
        assert two.last is second
