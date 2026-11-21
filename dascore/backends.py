"""
Array backends: how a spool makes the array each of its patches holds.

A patch holds whatever array it was given, and needs no backend for that.
A [`Spool`](`dascore.core.spool.Spool`) reads arrays, so it can be told
which kind to make: [`set_backend`](`dascore.core.spool.Spool.set_backend`)
takes an [`ArrayBackend`](`dascore.backends.ArrayBackend`), or the name of
one. Options, such as the device the arrays go on, are fields of the
instance. A spool with no backend set hands out whatever its files' readers
return.

A backend is registered by subclassing `ArrayBackend` and giving the class a
`name`: the one
[`backend_name`](`dascore.utils.array_api.backend_name`) reports for its
arrays, which is also what kernels are registered under. A package can have
DASCore import its backend the first time the name is asked for, by
declaring the module as a `dascore.kernels` entry point named for the
backend.

Examples
--------
>>> import numpy as np
>>> from dascore.backends import ArrayAPIBackend, LazyBackend, get_backend
>>>
>>> # A name is the backend with its default options.
>>> assert get_backend("numpy").from_array(np.ones(2)).shape == (2,)
>>> # A lazy backend names the one it computes on in brackets.
>>> assert get_backend("lazy[numpy]") == LazyBackend()
>>>
>>> # Another library which follows the array API standard takes a name.
>>> class Strict(ArrayAPIBackend):
...     '''Arrays of array_api_strict.'''
...
...     name = "array_api_strict"
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from dascore.exceptions import ParameterError
from dascore.utils.array_api import (
    LAZY_BACKEND,
    NUMPY_BACKEND,
    _as_numpy_dtype,
    backend_name,
    device,
    is_numpy,
    to_device,
    to_numpy,
)
from dascore.utils.plugins import load_backend_plugins

if TYPE_CHECKING:
    from dascore.lazy.recipe import Recipe

__all__ = [
    "ArrayAPIBackend",
    "ArrayBackend",
    "CupyBackend",
    "JaxBackend",
    "LazyBackend",
    "NumpyBackend",
    "get_backend",
]

# The registered backend classes, by name and alias in lower case.
_BACKENDS: dict[str, type[ArrayBackend]] = {}
# A backend's name, and in brackets what it is handed to make itself from.
_SPELLING = re.compile(r"\s*(\w+)\s*(?:\[(.*)\])?\s*")
# The backend reads hand their recipes to, or None when reads fill memory
# as usual; see `reading_onto`.
_READING: ContextVar[ArrayBackend | None] = ContextVar(
    "dascore_read_backend", default=None
)


class ArrayBackend(BaseModel):
    """
    How the array a patch holds is made.

    A subclass which sets `name` is registered under it. Its options are
    its fields, so two backends are equal when their options are.

    Attributes
    ----------
    name
        What [`backend_name`](`dascore.utils.array_api.backend_name`)
        reports for the backend's arrays. A string names the backend by it,
        whatever the case.
    aliases
        Other names a string may use for the backend.
    lazy
        Whether `from_recipe` returns an array which is not read yet.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    lazy: ClassVar[bool] = False

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register a class which names itself."""
        super().__pydantic_init_subclass__(**kwargs)
        if "name" in cls.__dict__:
            for key in (cls.name, *cls.aliases):
                _BACKENDS[key.lower()] = cls

    @classmethod
    def from_string(cls, argument: str | None = None) -> Self:
        """
        Return the backend a string names.

        Parameters
        ----------
        argument
            What the string gave in square brackets after the name, or None
            when it gave nothing. A backend takes nothing there unless it
            says otherwise, so a string names it only by its name.
        """
        if argument is not None:
            msg = (
                f"The {cls.name} backend takes nothing in square brackets, "
                f"but was given {argument!r}. Make an instance to set options."
            )
            raise ParameterError(msg)
        return cls()

    def from_recipe(self, recipe: Recipe) -> Any:
        """
        Return the array a recipe describes, which a read hands a patch.

        The recipe is read into memory and handed to `from_array`. A
        backend which reads for itself, or does not read yet, overrides
        this.

        Parameters
        ----------
        recipe
            Which stored samples fill which box of the array.
        """
        return self.from_array(recipe.load())

    def from_array(self, array: Any) -> Any:
        """
        Return an array which is held already as one of this backend's.

        Parameters
        ----------
        array
            An array of any backend. One of this backend's own is handed
            back as it is.
        """
        raise NotImplementedError

    def owns(self, array: Any) -> bool:
        """Return True if an array is one this backend would have made."""
        return backend_name(array) == self.name

    def __str__(self) -> str:
        """Return the string which names this backend with default options."""
        return self.name


class NumpyBackend(ArrayBackend):
    """
    Arrays in memory which NumPy handles.

    An array-like which is not of another array library is NumPy's to
    handle, and is left as it is: one which keeps its values some other way,
    compressed say, is not converted to an `ndarray`.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.backends import NumpyBackend
    >>>
    >>> data = np.ones(3)
    >>> assert NumpyBackend().from_array(data) is data
    """

    name = NUMPY_BACKEND

    def from_array(self, array: Any) -> Any:
        """Return an array NumPy handles; one which is already is left alone."""
        return array if self.owns(array) else to_numpy(array)


class ArrayAPIBackend(ArrayBackend):
    """
    Arrays of a library which follows the array API standard.

    A subclass names the library, and is all another library needs.

    Parameters
    ----------
    device
        The device the arrays are put on, as the library itself spells a
        device. None leaves them where the library puts them.

    Attributes
    ----------
    namespace
        The module whose `asarray` makes the library's arrays, where it is
        not the one called `name`.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.backends import ArrayAPIBackend
    >>>
    >>> class Strict(ArrayAPIBackend):
    ...     '''Arrays of array_api_strict.'''
    ...
    ...     name = "array_api_strict"
    >>>
    >>> assert Strict().from_array(np.ones(3)).shape == (3,)
    """

    namespace: ClassVar[str | None] = None

    device: Any = None

    def owns(self, array: Any) -> bool:
        """Return True for the library's arrays on this backend's device."""
        if backend_name(array) != self.name:
            return False
        return self.device is None or device(array) == self.device

    def from_array(self, array: Any) -> Any:
        """
        Return an array on the library, and on the device if one was given.

        An array of another library is taken through NumPy, since that is
        what every library makes an array from.

        Raises
        ------
        ParameterError
            If the library cannot be imported, or holds the values in
            another dtype than they came in.
        """
        if self.owns(array):
            return array
        if backend_name(array) == self.name:
            # Its own already, so only the device is another.
            return to_device(array, self.device)
        # An array NumPy already holds is handed over as it is.
        held = array if is_numpy(array) else to_numpy(array)
        try:
            out = import_module(self.namespace or self.name).asarray(held)
        except (ImportError, AttributeError) as error:
            msg = f"No array library called {self.name!r} could be used: {error}"
            raise ParameterError(msg) from error
        # A library which holds the values in another dtype, as JAX does
        # with 64-bit numbers unless told otherwise, would change them under
        # the same id: refused rather than trusted.
        made = _as_numpy_dtype(out.dtype)
        if held.size and made != held.dtype:
            msg = (
                f"{self.name} holds {held.dtype} data as {made}, which would "
                f"change their values. Configure {self.name} to keep "
                f"{held.dtype}, or convert the data first."
            )
            raise ParameterError(msg)
        return out if self.device is None else to_device(out, self.device)


class JaxBackend(ArrayAPIBackend):
    """
    JAX arrays.

    Examples
    --------
    >>> from dascore.backends import JaxBackend, get_backend
    >>>
    >>> assert get_backend("jax") == JaxBackend()
    """

    name = "jax"
    namespace = "jax.numpy"


class CupyBackend(ArrayAPIBackend):
    """
    CuPy arrays.

    Examples
    --------
    >>> from dascore.backends import CupyBackend, get_backend
    >>>
    >>> assert get_backend("cupy") == CupyBackend()
    """

    name = "cupy"


class LazyBackend(ArrayBackend):
    """
    Arrays which are not read until they are loaded.

    Parameters
    ----------
    compute
        The backend each chunk is computed on, and the loaded array is one
        of. A string gives it in square brackets: ``"lazy[jax]"``.

    Examples
    --------
    >>> from dascore.backends import JaxBackend, LazyBackend, get_backend
    >>>
    >>> assert get_backend("LazyArray[jax]") == LazyBackend(compute=JaxBackend())
    >>> assert str(LazyBackend(compute=JaxBackend())) == "lazy[jax]"
    """

    name = LAZY_BACKEND
    aliases = ("lazyarray",)
    lazy = True

    compute: ArrayBackend = NumpyBackend()

    def model_post_init(self, context: Any) -> None:
        """Refuse a lazy backend to compute on: nothing would ever be read."""
        if self.compute.lazy:
            msg = (
                "A lazy backend computes on an array library, not on another "
                f"lazy backend: {self.compute!r}."
            )
            raise ParameterError(msg)

    @classmethod
    def from_string(cls, argument: str | None = None) -> Self:
        """Return the lazy backend computing on the one named in brackets."""
        return cls() if argument is None else cls(compute=get_backend(argument))

    def from_recipe(self, recipe: Recipe) -> Any:
        """Return the recipe as a lazy array; nothing is read."""
        return _lazy_array_class()(recipe, backend=self.compute)

    def from_array(self, array: Any) -> Any:
        """
        Return a lazy array computed on `compute`, or an array held there.

        An array in memory has nothing to read later, so it is put where
        the chunks of a lazy one are computed. A dask array is another kind
        of promise, and is left to whoever holds it.
        """
        if isinstance(array, _lazy_array_class()):
            return array._with_backend(self.compute)
        if backend_name(array) == "dask":
            return array
        return self.compute.from_array(array)

    def __str__(self) -> str:
        """Return the name, with the backend computed on if it is not NumPy."""
        inner = str(self.compute)
        return self.name if inner == NUMPY_BACKEND else f"{self.name}[{inner}]"


def _lazy_array_class():
    """Return `LazyArray`, whose module imports this one for its backend."""
    # Imported here: a lazy array holds the backend it computes on.
    from dascore.lazy.array import LazyArray  # noqa: PLC0415

    return LazyArray


def get_backend(backend: str | ArrayBackend) -> ArrayBackend:
    """
    Return the backend an instance or a string names.

    Parameters
    ----------
    backend
        A backend, handed back as it is, or a string: the name or an alias
        of a registered backend, which gives it with default options, and
        after it in square brackets whatever that backend takes there.

    Raises
    ------
    ParameterError
        If no backend of that name is registered, or it does not take what
        the brackets gave.

    Examples
    --------
    >>> from dascore.backends import LazyBackend, NumpyBackend, get_backend
    >>>
    >>> assert get_backend("numpy") == NumpyBackend()
    >>> assert get_backend("lazy") == LazyBackend(compute=NumpyBackend())
    >>> assert get_backend(NumpyBackend()) == NumpyBackend()
    """
    if isinstance(backend, ArrayBackend):
        return backend
    found = _SPELLING.fullmatch(str(backend))
    name = found.group(1).lower() if found else None
    if name is not None and name not in _BACKENDS:
        # A package's backend is registered when its module is imported.
        load_backend_plugins(name)
    if found is None or name not in _BACKENDS:
        known = ", ".join(sorted({x.name for x in _BACKENDS.values()}))
        msg = (
            f"{backend!r} is not a backend. The registered backends are: "
            f"{known}. A lazy one names the backend it computes on in square "
            "brackets, as in 'lazy[jax]'."
        )
        raise ParameterError(msg)
    try:
        return _BACKENDS[name].from_string(found.group(2))
    except ValidationError as error:
        # The options a string gave are a parameter like any other.
        raise ParameterError(str(error)) from error


@contextmanager
def reading_onto(backend: ArrayBackend | None) -> Iterator[None]:
    """
    Have reads inside the block ask `backend` for each patch's array.

    None leaves reads as they are, filling memory. So does a backend whose
    `from_recipe` is the default: its arrays are made from what was read,
    and a reader fills memory best itself, a file at a time. A read which
    a lazy backend cannot leave unread is the `numpy_fallback` setting's to
    allow, which left unset raises
    [`LazyOperationError`](`dascore.exceptions.LazyOperationError`) rather
    than read.
    """
    if backend is not None and type(backend).from_recipe is ArrayBackend.from_recipe:
        backend = None
    token = _READING.set(backend)
    try:
        yield
    finally:
        _READING.reset(token)


def read_backend() -> ArrayBackend | None:
    """Return the backend reads hand their recipes to, or None when eager."""
    return _READING.get()
