"""
A lazy array: a recipe and the operations pending on it.

A [`LazyArray`](`dascore.lazy.array.LazyArray`) is what a patch holds when
its data are not read yet. The [`Recipe`](`dascore.lazy.recipe.Recipe`)
says which stored samples fill which box of the array, and the
[`Chain`](`dascore.lazy.chain.Chain`) holds the operations run on it since,
none of which has computed anything. Its shape, dtype and `data_id` are the
last step's, or the recipe's when nothing is pending, so they are known
without reading. [`load`](`dascore.lazy.array.LazyArray.load`) reads the
recipe and runs each step's kernel for the array's backend in order, a
chunk at a time when the whole would be large; see
[`dascore.lazy.execute`](`dascore.lazy.execute`).

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
from collections.abc import Sequence
from types import SimpleNamespace

import numpy as np

from dascore.backends import ArrayBackend, NumpyBackend, _refuse_mixed
from dascore.exceptions import ParameterError
from dascore.lazy import execute
from dascore.lazy.chain import Chain, Node
from dascore.lazy.recipe import Recipe, concat
from dascore.utils.array_api import (
    LAZY_BACKEND,
    NUMPY_BACKEND,
    _as_numpy_dtype,
    array_namespace,
    asarray_like,
    backend_name,
    decide_fallback,
    is_numpy,
    to_numpy,
)


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
    backend
        The [`ArrayBackend`](`dascore.backends.ArrayBackend`) each chunk is
        computed on and `load` returns an array of; NumPy's by default.

    Notes
    -----
    It cannot be indexed or transposed itself: a patch whose data are lazy
    runs only the operations which state a chunk contract, `select` among
    them, and metadata-only ones.
    """

    __slots__ = ("_backend", "_chain", "_recipe")

    def __init__(
        self,
        recipe: Recipe,
        chain: Chain | None = None,
        backend: ArrayBackend | None = None,
    ):
        self._recipe = recipe
        self._chain = Chain() if chain is None else chain
        self._backend = NumpyBackend() if backend is None else backend

    def _with_backend(self, backend: ArrayBackend) -> LazyArray:
        """
        Return the array computed on another backend.

        A step already pending with no kernel for its library would have
        each chunk converted through NumPy, which the `numpy_fallback`
        setting decides here, as it does for a step chained afterwards.
        """
        if backend == self._backend:
            return self
        old = self._backend
        # The same arrays under the same name, on another device say, run
        # the kernels they ran: there is nothing to ask again.
        same = (backend._native, backend.name) == (old._native, old.name)
        for node in () if same else self._chain:
            if (converted := execute._converted(node.processor, backend)) is not None:
                decide_fallback(
                    f"{converted} has no {backend._native} implementation",
                    "each chunk is converted to numpy to run it",
                )
        return LazyArray(self._recipe, self._chain, backend)

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
        on = str(self._backend)
        on = "" if on == NUMPY_BACKEND else f", backend={on!r}"
        return f"{name}(shape={self.shape}, dtype={self.dtype}, steps={steps}{on})"

    def _appended(self, node: Node) -> LazyArray:
        """Return the array with one more operation pending."""
        return LazyArray(self._recipe, self._chain.append(node), self._backend)

    def _settled(self, shape, dtype, data_id: str) -> LazyArray:
        """Return the array with its last step settled; see `Node.settled`."""
        *before, last = self._chain.nodes
        node = last.settled(shape, dtype, data_id)
        return LazyArray(self._recipe, Chain((*before, node)), self._backend)

    def load(self) -> np.ndarray:
        """
        Read the recipe and run every pending step, returning the array.

        Each step runs the kernel its processor has for this array's
        backend, NumPy's unless another was given, on the whole
        array or, when that would hold an array larger than the
        `lazy_block_size` setting, a chunk at a time; only the samples the
        result needs are read. The array returned is on that library. What
        each step makes is checked against the shape and dtype its metadata
        stated, which is what the patch said it held all along.

        Raises
        ------
        PatchError
            If a step is still pending, or computed an array other than the
            one its metadata stated.
        """
        return execute.load(self)

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Load the array for numpy; each call reads and computes it again."""
        out = to_numpy(self.load())
        return out if dtype is None else out.astype(dtype, copy=False)

    def __array_namespace__(self, api_version: str | None = None):
        """Return the namespace which names the lazy backend."""
        return _NAMESPACE


# Names the backend only: the lazy backend runs operations through their
# chunk contracts, never through an array api namespace.
_NAMESPACE = SimpleNamespace(__name__=LAZY_BACKEND)


def join(arrays: Sequence, axis: int, new_axis: bool = False):
    """
    Join arrays end to end along an axis, on the backend they are on.

    Lazy arrays with nothing pending are a window of what is stored, and
    are joined by their recipes, reading nothing: the result is lazy. Any
    other arrays held on one library are joined there, by the library's
    own `concat`, so they are never taken through NumPy. Arrays of
    different libraries are not joined at all. What cannot be joined either
    way -- lazy arrays with operations pending, a mix of lazy and in-memory
    arrays, or a new axis on lazy arrays -- is read or converted only as
    the `numpy_fallback` setting allows.

    Parameters
    ----------
    arrays
        The arrays to join; they agree on every other axis.
    axis
        The axis to join along.
    new_axis
        Give each array a new axis of length one at the end first, and
        join along it, as stacking does.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.core.source import ArraySource
    >>> from dascore.lazy.array import LazyArray, join
    >>> from dascore.lazy.recipe import Recipe
    >>>
    >>> full = (ArraySource.full((2, 3), x) for x in (1, 2))
    >>> parts = [LazyArray(Recipe.from_source(x)) for x in full]
    >>> joined = join(parts, axis=1)
    >>> assert isinstance(joined, LazyArray) and joined.shape == (2, 6)
    >>> assert np.array_equal(join([np.ones(2), np.zeros(2)], 0), [1, 1, 0, 0])
    """
    arrays = list(arrays)
    # Of one array library, counting a lazy array as what it loads to: no
    # library's arrays are put on another's to be joined.
    _refuse_mixed(arrays, "Joining patches")
    lazy = [isinstance(x, LazyArray) for x in arrays]
    if all(lazy) and not new_axis:
        if (out := _joined_recipes(arrays, axis)) is not None:
            return out
    if any(lazy):
        reason = (
            "a new axis is added to lazy data"
            if new_axis and all(lazy)
            else (
                "it joins lazy data with operations pending, of different "
                "dtypes, or with data in memory"
            )
        )
        decide_fallback(
            f"This concatenation cannot stay lazy: {reason}",
            "its members were read into memory",
        )
        arrays = [x.load() if is_lazy else x for x, is_lazy in zip(arrays, lazy)]
    if all(is_numpy(x) for x in arrays):
        # As NumPy always has: its own promotion of mixed dtypes, and speed.
        parts = [x[..., None] for x in arrays] if new_axis else arrays
        return np.concatenate(parts, axis=axis)
    # Promoted as NumPy promotes the same arrays, from the dtypes they have
    # now: a library may not do so, nor keep every dtype NumPy holds.
    dtype = np.result_type(*(_as_numpy_dtype(x.dtype) for x in arrays))
    # Something NumPy takes only through `__array__`, which may read it.
    if backend_name(arrays[0]) == NUMPY_BACKEND:
        decide_fallback(
            "Arrays NumPy holds only through `__array__` cannot be joined as they are",
            "they were converted to numpy and joined there",
            lazy=False,
        )
        parts = [to_numpy(x) for x in arrays]
        parts = [x[..., None] for x in parts] if new_axis else parts
        return np.concatenate(parts, axis=axis)
    xp = array_namespace(arrays[0])
    if any(_as_numpy_dtype(x.dtype) != dtype for x in arrays):
        arrays = [cast_on_library(x, dtype) for x in arrays]
    if new_axis:
        arrays = [xp.expand_dims(x, axis=-1) for x in arrays]
    return xp.concat(arrays, axis=axis)


def _check_kept(array, dtype: np.dtype) -> None:
    """Raise if a library holds data in another dtype than they were given in."""
    held = _as_numpy_dtype(array.dtype)
    if held != dtype:
        library = backend_name(array)
        msg = (
            f"{library} holds {dtype} data as {held}, which would change "
            f"their values. Configure {library} to keep {dtype}, or convert "
            "the data first."
        )
        raise ParameterError(msg)


def cast_on_library(array, dtype: np.dtype):
    """
    Return an array cast to a NumPy dtype, on the library it is on.

    A library which would hold the values in another dtype, as JAX does
    with 64-bit numbers unless told otherwise, is refused rather than
    trusted to keep them.
    """
    if is_numpy(array):
        return array.astype(dtype, copy=False)
    target = asarray_like(np.empty(0, dtype), array)
    _check_kept(target, dtype)
    return array_namespace(array).astype(array, target.dtype)


def _joined_recipes(arrays: Sequence[LazyArray], axis: int) -> LazyArray | None:
    """Return lazy arrays joined by their recipes, or None if they cannot be."""
    # Of one kind of array already, so on the first one's backend.
    backend = arrays[0]._backend
    if any(len(x.chain) for x in arrays):
        return None
    # Of one dtype only: a recipe joined with another of a wider dtype casts
    # its stored samples straight to it, skipping any rounding its own
    # dtype made, so values could differ from those joined in memory.
    if len({x.dtype for x in arrays}) > 1:
        return None
    # A recipe join promotes the dtype as NumPy joins arrays in memory.
    joined = concat([x.recipe for x in arrays], axis=axis)
    return LazyArray(joined, backend=backend)
