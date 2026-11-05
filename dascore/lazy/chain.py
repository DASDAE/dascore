"""
The operations pending on a lazy array, as a linear chain.

A [`Node`](`dascore.lazy.chain.Node`) is one step: the processor, the plan
its `get_metadata` worked out, and what the step made -- the shape and dtype
of its output and the `data_id` of the patch which holds it. A node is
pending from the moment a kernel appends it until the operation settles
it with those three facts, before the resulting patch is built. A
[`Chain`](`dascore.lazy.chain.Chain`) is the steps in the order they run.
Both are immutable: appending makes a new chain.

Examples
--------
>>> import numpy as np
>>> from dascore.lazy.chain import Chain, Node
>>> from dascore.proc.basic import Abs
>>>
>>> pending = Node(Abs(), {})
>>> assert pending.pending
>>> node = pending.settled((3, 2), np.float64, "an-id")
>>> chain = Chain().append(node)
>>> assert len(chain) == 1 and chain.last is node
>>> assert node.made() == ((3, 2), np.dtype("float64"), "an-id")
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np

from dascore.exceptions import PatchError


# Not compared by value: a plan may hold arrays, and two steps are the same
# step only when they are the same object; their ids say whether they agree.
@dataclass(frozen=True, eq=False)
class Node:
    """
    One pending operation, and what it made once settled.

    Parameters
    ----------
    processor
        The [`PatchProcessor`](`dascore.core.processor.PatchProcessor`)
        whose kernel computes the step.
    plan
        The arguments its kernel takes beyond the data.
    shape, dtype
        The shape and dtype the step's metadata states; None while pending.
    data_id
        The `data_id` of the patch this step produced; None while pending.
    """

    processor: Any
    plan: Mapping[str, Any] = field(default_factory=dict)
    shape: tuple[int, ...] | None = None
    dtype: np.dtype | None = None
    data_id: str | None = None

    @property
    def pending(self) -> bool:
        """Whether the step has not been settled yet."""
        # An id of "" is settled: it is what a step has with ids disabled.
        return self.data_id is None

    def made(self) -> tuple[tuple[int, ...], np.dtype, str]:
        """
        Return the shape, dtype and `data_id` of what the step made.

        Raises
        ------
        PatchError
            If the step is pending, when nothing says what it made yet.
        """
        shape, dtype, data_id = self.shape, self.dtype, self.data_id
        if shape is None or dtype is None or data_id is None:
            name = type(self.processor).__name__
            msg = f"The {name} step was never settled, so nothing says what it made."
            raise PatchError(msg)
        return shape, dtype, data_id

    def settled(self, shape, dtype, data_id: str) -> Node:
        """Return this step with the output it states and the id it was given."""
        return replace(
            self,
            shape=tuple(int(x) for x in shape),
            dtype=np.dtype(dtype),
            data_id=str(data_id),
        )


@dataclass(frozen=True)
class Chain:
    """The steps pending on a lazy array, in the order they run."""

    nodes: tuple[Node, ...] = ()

    def __len__(self) -> int:
        """The number of steps."""
        return len(self.nodes)

    def __iter__(self) -> Iterator[Node]:
        """The steps, first to last."""
        return iter(self.nodes)

    @property
    def last(self) -> Node | None:
        """The step which runs last, or None for an empty chain."""
        return self.nodes[-1] if self.nodes else None

    def append(self, node: Node) -> Chain:
        """Return the chain with one more step at the end."""
        return Chain((*self.nodes, node))
