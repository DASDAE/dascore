"""
A lazy array: a recipe and the operations pending on it.

A [`LazyArray`](`dascore.lazy.array.LazyArray`) is what a patch holds when
its data are not read yet. The [`Recipe`](`dascore.lazy.recipe.Recipe`)
says which stored samples fill which box of the array, and the
[`Chain`](`dascore.lazy.chain.Chain`) holds the operations run on it since,
none of which has computed anything. Its shape, dtype and `data_id` are the
last step's, or the recipe's when nothing is pending, so they are known
without reading. [`load`](`dascore.lazy.array.LazyArray.load`) reads the
recipe and runs each step's NumPy kernel in order, in memory.

Examples
--------
>>> import numpy as np
>>> from dascore.core.source import ArraySource
>>> from dascore.lazy.array import LazyArray
>>> from dascore.lazy.recipe import Recipe
>>>
>>> recipe = Recipe.from_source(ArraySource.full((4, 3), -2.0))
>>> array = LazyArray(recipe)
>>> assert array.shape == (4, 3) and len(array) == 4
>>> assert array.data_id == recipe.data_id
>>> assert np.array_equal(array.load(), np.full((4, 3), -2.0))
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from dascore.exceptions import PatchError
from dascore.lazy.chain import Chain, Node
from dascore.lazy.recipe import Recipe
from dascore.utils.array_api import LAZY_BACKEND


class LazyArray:
    """
    A recipe, and the operations pending on it.

    Parameters
    ----------
    recipe
        Which stored samples fill which box of the array.
    chain
        The operations to run on the recipe's array, in order; none by
        default, when the array is the recipe's.

    Notes
    -----
    No slicing or transposing: a patch whose data are lazy runs only the
    operations which state a chunk contract, and metadata-only ones.
    """

    __slots__ = ("_chain", "_recipe")

    def __init__(self, recipe: Recipe, chain: Chain | None = None):
        self._recipe = recipe
        self._chain = Chain() if chain is None else chain

    @property
    def recipe(self) -> Recipe:
        """Which stored samples the array starts from."""
        return self._recipe

    @property
    def chain(self) -> Chain:
        """The operations pending on the recipe's array, in order."""
        return self._chain

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape of the array `load` returns."""
        last = self._chain.last
        return self._recipe.shape if last is None else last.made()[0]

    @property
    def dtype(self) -> np.dtype:
        """The dtype of the array `load` returns."""
        last = self._chain.last
        return self._recipe.dtype if last is None else last.made()[1]

    @property
    def ndim(self) -> int:
        """The number of dimensions of the array."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """The number of elements in the array."""
        return math.prod(self.shape)

    @property
    def data_id(self) -> str:
        """
        The id of the array `load` returns: the last step's, or the recipe's.

        Each step's is the `data_id` the patch it produced was given, but a
        patch's own `attrs.data_id` is what names the patch: a metadata-only
        operation changes that and leaves the array as it was.
        """
        last = self._chain.last
        return self._recipe.data_id if last is None else last.made()[2]

    def __len__(self) -> int:
        """The length of the first axis, as numpy's is."""
        if not self.shape:
            msg = "len() of unsized object"
            raise TypeError(msg)
        return self.shape[0]

    def __bool__(self) -> bool:
        """The truth of the one element, refused for any other size, as numpy does."""
        if (size := self.size) != 1:
            msg = (
                f"The truth value of an array of {size} elements is "
                "ambiguous. Use a.any() or a.all() on the loaded array."
            )
            raise ValueError(msg)
        return bool(self.load())

    def __repr__(self) -> str:
        name = type(self).__name__
        steps = len(self._chain)
        return f"{name}(shape={self.shape}, dtype={self.dtype}, steps={steps})"

    def _appended(self, node: Node) -> LazyArray:
        """Return the array with one more operation pending."""
        return LazyArray(self._recipe, self._chain.append(node))

    def _settled(self, shape, dtype, data_id: str) -> LazyArray:
        """Return the array with its last step settled; see `Node.settled`."""
        *before, last = self._chain.nodes
        node = last.settled(shape, dtype, data_id)
        return LazyArray(self._recipe, Chain((*before, node)))

    def _restamped(self, before: str, after: str) -> LazyArray:
        """Return the array with its last step's id `after` if it was `before`."""
        last = self._chain.last
        if last is None or last.data_id != before:
            return self
        shape, dtype, _ = last.made()
        return self._settled(shape, dtype, after)

    def load(self) -> np.ndarray:
        """
        Read the recipe and run every pending step, in memory.

        Each step runs its processor's NumPy kernel. The result of each is
        checked against the shape and dtype its metadata stated, which is
        what the patch said it held all along.

        Raises
        ------
        PatchError
            If a step is still pending, or computed an array other than the
            one its metadata stated.
        """
        out = self._recipe.load()
        for node in self._chain:
            stated_shape, stated_dtype, _ = node.made()
            processor = node.processor
            kernel = type(processor).kernel_for("numpy")
            out = kernel(processor, out, **node.plan)
            shape, dtype = tuple(out.shape), np.dtype(out.dtype)
            if shape != stated_shape or dtype != stated_dtype:
                name = type(processor).__name__
                msg = (
                    f"{name} computed an array of shape {shape} and dtype "
                    f"{dtype}, but its metadata stated {stated_shape} and "
                    f"{stated_dtype}. Its get_metadata must state what its "
                    "kernel returns."
                )
                raise PatchError(msg)
        return out

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Load the array for numpy; each call reads and computes it again."""
        out = self.load()
        return out if dtype is None else out.astype(dtype, copy=False)

    def __array_namespace__(self, api_version: str | None = None):
        """Return the namespace which names the lazy backend."""
        return _NAMESPACE


# Names the backend only: the lazy backend runs operations through their
# chunk contracts, never through an array api namespace.
_NAMESPACE = SimpleNamespace(__name__=LAZY_BACKEND)
