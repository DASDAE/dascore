"""
Array backends: how a spool makes the array each of its patches holds.

A patch holds whatever array it was given, and needs no backend for that.
A [`Spool`](`dascore.core.spool.Spool`) reads arrays, so it can be told
which kind to make: [`set_backend`](`dascore.core.spool.Spool.set_backend`)
takes an [`ArrayBackend`](`dascore.backends.ArrayBackend`), or the name of
one. Options, such as the device the arrays go on, are fields of the
instance. A spool with no backend set hands out whatever its files' readers
return.

A backend states two things. Its `name` is whatever it likes: what a string
calls it, what its kernels are registered under, and what a package names
its `dascore.kernels` entry point, so DASCore imports the backend the first
time the name is asked for. Its `array_type` is the class of its arrays.
Two backends may make the same arrays under different names: an operation
runs the kernel registered for the name of the patch's backend, and where
there is none, the one the array would run anyway. Subclassing a backend
under another name is all that takes.
"""

from __future__ import annotations

import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from importlib import import_module
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
from pydantic import BaseModel, ConfigDict

from dascore.compat import is_array_like
from dascore.exceptions import IncompatiblePatchError, ParameterError
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

# The registered backend classes, by name.
_BACKENDS: dict[str, type[ArrayBackend]] = {}
# What `backend_name` reports for the arrays a backend class makes; see
# `ArrayBackend._native`.
_NATIVE: dict[type[ArrayBackend], str] = {}
# What a backend is handed to see what it makes: empty, and of a dtype every
# array library keeps, so one which holds 64-bit numbers in 32 is not
# refused before it is given any.
_PROBE = np.empty(0, np.float32)
# A backend's name, and in brackets what it is handed to make itself from.
_SPELLING = re.compile(r"\s*(\w+)\s*(?:\[(.*)\])?\s*")
# The backend reads hand their recipes to, or None when reads fill memory
# as usual; see `reading_onto`.
_READING: ContextVar[ArrayBackend | None] = ContextVar(
    "dascore_read_backend", default=None
)


class ArrayBackend(BaseModel, ABC):
    """
    How the array a patch holds is made, and which kernels run on it.

    A subclass which states a `name` is registered under it. Its
    options are its fields, so two backends are equal when their options are.

    Attributes
    ----------
    name
        Any name of lower-case letters, digits and underscores: what a
        string calls the backend, whatever the string's case, and what its
        kernels are registered under.
    array_type
        The class of the backend's arrays, or a string of where it is, such
        as ``"cupy.ndarray"``, which imports nothing. Its arrays are the
        instances of it. A subclass under another name inherits it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    name: ClassVar[str]
    array_type: ClassVar[type | str]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Register a class which states its name."""
        super().__pydantic_init_subclass__(**kwargs)
        if "name" not in cls.__dict__:
            return
        if not hasattr(cls, "array_type"):
            msg = (
                f"{cls.__name__} states the name {cls.name!r} but no "
                "array_type: a backend says which class its arrays are."
            )
            raise TypeError(msg)
        if not re.fullmatch(r"[a-z0-9_]+", cls.name):
            msg = (
                f"The name {cls.name!r} of {cls.__name__} must be lower-case "
                "letters, digits and underscores: a string names the backend "
                "by it, and kernels are registered under it exactly as it is."
            )
            raise TypeError(msg)
        holder = _BACKENDS.get(cls.name)
        # The same class made again, as a notebook cell run twice makes it,
        # takes its own place; another class does not take a name.
        if holder is not None and _where(holder) != _where(cls):
            msg = f"The backend name {cls.name!r} is taken by {_where(holder)}."
            raise TypeError(msg)
        _BACKENDS[cls.name] = cls

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
            A [`Recipe`](`dascore.lazy.recipe.Recipe`): which stored samples
            fill which box of the array. Its `load` reads them into a NumPy
            array, and the array returned must have its `shape` and `dtype`.
        """
        return self.from_array(recipe.load())

    @abstractmethod
    def from_array(self, array: Any) -> Any:
        """
        Return an array which is held already as one of this backend's.

        Parameters
        ----------
        array
            An array of any backend. One of this backend's own is handed
            back as it is. A spool or patch set to the backend hands it an
            empty one-dimensional `float32` array first, so a backend which
            cannot be used says so then and not at the first patch.
        """

    def _owns(self, array: Any) -> bool:
        """Return True if an array is an instance of the backend's array type."""
        kind: Any = self.array_type
        if isinstance(kind, str):
            # Nothing is imported for it: an array of a library which is
            # not imported yet cannot be one of its arrays.
            top, *rest = kind.split(".")
            kind = sys.modules.get(top)
            for part in rest:
                kind = getattr(kind, part, None)
        return isinstance(kind, type) and isinstance(array, kind)

    def _check(self) -> None:
        """Raise if the backend does not make arrays of its array type."""
        made = self.from_array(_PROBE)
        if not self._owns(made):
            msg = (
                f"The {self.name} backend made a {_where(type(made))}, "
                f"which is not an instance of its array_type, "
                f"{self.array_type!r}. Its kernels would never run: check the "
                "spelling of the array type, and that the library is the one "
                "`from_array` makes arrays with."
            )
            raise ParameterError(msg)

    @property
    def _native(self) -> str:
        """
        Return what `backend_name` reports for the arrays the backend makes.

        Asked only where there is no array yet to ask, as when a step is
        chained on lazy data; found once for a class, from an array it makes.
        """
        kind = type(self)
        if kind not in _NATIVE:
            _NATIVE[kind] = backend_name(self.from_array(_PROBE))
        return _NATIVE[kind]

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
    array_type = np.ndarray

    def from_array(self, array: Any) -> Any:
        """Return an array NumPy handles; one which is already is left alone."""
        # Not only an `ndarray`: any array which is of no other library.
        own = is_array_like(array) and backend_name(array) == NUMPY_BACKEND
        return array if own else to_numpy(array)


class ArrayAPIBackend(ArrayBackend):
    """
    Arrays of a library which follows the array API standard.

    A subclass states its name and the class of the library's arrays,
    and is all another library needs.

    Parameters
    ----------
    device
        The device the arrays are put on, as the library itself spells a
        device. None leaves them where the library puts them.

    Attributes
    ----------
    module
        The module whose `asarray` makes the library's arrays, where it is
        not the package the array type is in.

    Examples
    --------
    >>> import numpy as np
    >>> from dascore.backends import ArrayAPIBackend
    >>>
    >>> class Strict(ArrayAPIBackend):
    ...     '''Arrays of array_api_strict.'''
    ...
    ...     name = "strict"
    ...     array_type = "array_api_strict._array_object.Array"
    >>>
    >>> assert Strict().from_array(np.ones(3)).shape == (3,)
    """

    module: ClassVar[str | None] = None

    device: Any = None

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
        if self._owns(array):
            # Its own already, so only the device may be another.
            there = self.device is None or device(array) == self.device
            return array if there else to_device(array, self.device)
        kind = self.array_type
        package = (kind if isinstance(kind, str) else kind.__module__).split(".")[0]
        # An array NumPy already holds is handed over as it is.
        held = array if is_numpy(array) else to_numpy(array)
        try:
            out = import_module(self.module or package).asarray(held)
        except (ImportError, AttributeError) as error:
            msg = f"No array library called {package!r} could be used: {error}"
            raise ParameterError(msg) from error
        # A library which holds the values in another dtype, as JAX does
        # with 64-bit numbers unless told otherwise, would change them under
        # the same id: refused rather than trusted.
        made = _as_numpy_dtype(out.dtype)
        if made != held.dtype:
            msg = (
                f"{package} holds {held.dtype} data as {made}, which would "
                f"change their values. Configure {package} to keep "
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
    array_type = "jax.Array"
    module = "jax.numpy"


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
    array_type = "cupy.ndarray"


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
    # By where it is: a lazy array holds the backend it computes on, so its
    # module imports this one.
    array_type = "dascore.lazy.array.LazyArray"

    compute: ArrayBackend = NumpyBackend()

    def __init__(self, **options: Any) -> None:
        """Refuse a lazy backend to compute on: nothing would ever be read."""
        super().__init__(**options)
        if isinstance(self.compute, LazyBackend):
            msg = (
                "A lazy backend computes on an array library, not on another "
                f"lazy backend: {self.compute!r}."
            )
            raise ParameterError(msg)

    def _check(self) -> None:
        """Raise if the backend computed on does not make its own arrays."""
        self.compute._check()

    @property
    def _native(self) -> str:
        """Return what the arrays are once loaded: the computing backend's."""
        return self.compute._native

    @classmethod
    def from_string(cls, argument: str | None = None) -> Self:
        """Return the lazy backend computing on the one named in brackets."""
        return cls() if argument is None else cls(compute=get_backend(argument))

    def from_recipe(self, recipe: Recipe) -> Any:
        """Return the recipe as a lazy array; nothing is read."""
        # Imported here: see `array_type`.
        from dascore.lazy.array import LazyArray  # noqa: PLC0415

        return LazyArray(recipe, backend=self.compute)

    def from_array(self, array: Any) -> Any:
        """
        Return a lazy array computed on `compute`, or an array held there.

        An array in memory has nothing to read later, so it is put where
        the chunks of a lazy one are computed. A dask array is another kind
        of promise, and is left to whoever holds it.
        """
        if self._owns(array):
            return array._with_backend(self.compute)
        if backend_name(array) == "dask":
            return array
        return self.compute.from_array(array)

    def __str__(self) -> str:
        """Return the name, with the backend computed on if it is not NumPy."""
        inner = str(self.compute)
        if inner == NUMPY_BACKEND:
            return self.name
        return f"{self.name}[{inner}]"


# The class's own name is a spelling of the backend too.
_BACKENDS["lazyarray"] = LazyBackend
# Asked whether an array is a lazy one, without importing the lazy array.
_LAZY = LazyBackend()


def _settled(backend: ArrayBackend | None, array: Any) -> tuple[Any, Any]:
    """
    Return an array as a backend's own, and the backend it is then of.

    The backend is None when there was none, and when the backend left the
    array as it was without its being one of its own, as NumPy's leaves an
    array-like of no array library. Held in memory, a lazy backend's array
    is of the backend it computes on.
    """
    if backend is None:
        return array, None
    array = backend.from_array(array)
    if isinstance(backend, LazyBackend) and not backend._owns(array):
        backend = backend.compute
    return array, (backend if backend._owns(array) else None)


def _refuse_mixed(arrays, doing: str) -> None:
    """
    Raise if arrays are of different array libraries; read none of them.

    A lazy array is of the library it will be once it is loaded, and an
    array-like of no array library is NumPy's, as it is everywhere else.
    """
    kinds = {x._backend._native if _LAZY._owns(x) else backend_name(x) for x in arrays}
    if len(kinds) > 1:
        msg = (
            f"{doing} needs arrays of one array library, and these are of "
            f"{', '.join(sorted(kinds))}. Load them, or put the patches on "
            "one backend with set_backend, first."
        )
        raise IncompatiblePatchError(msg)


def _where(kind: type) -> str:
    """Return the module and name a class is found under."""
    return f"{kind.__module__}.{kind.__qualname__}"


def get_backend(backend: str | ArrayBackend) -> ArrayBackend:
    """
    Return the backend an instance or a string names.

    Parameters
    ----------
    backend
        A backend, handed back as it is, or a string: the name of a
        registered backend, which gives it with default options, and after
        it in square brackets whatever that backend takes there.

    Raises
    ------
    ParameterError
        If no backend is registered under that name, or it does not take
        what the brackets gave.

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
        # A package's backend is registered when its module is imported. One
        # whose module failed part way is not left registered half made.
        before = dict(_BACKENDS)
        if not load_backend_plugins(name):
            _BACKENDS.clear()
            _BACKENDS.update(before)
    if found is None or name not in _BACKENDS:
        known = ", ".join(sorted({x.name for x in _BACKENDS.values()}))
        msg = (
            f"{backend!r} is not a backend. The registered backends are: "
            f"{known}. A lazy one names the backend it computes on in square "
            "brackets, as in 'lazy[jax]'."
        )
        raise ParameterError(msg)
    return _BACKENDS[name].from_string(found.group(2))


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
    token = _READING.set(backend)
    try:
        yield
    finally:
        _READING.reset(token)


def read_backend() -> ArrayBackend | None:
    """Return the backend reads hand their recipes to, or None when eager."""
    backend = _READING.get()
    if backend is None or type(backend).from_recipe is ArrayBackend.from_recipe:
        return None
    return backend


def spool_backend() -> ArrayBackend | None:
    """Return the backend of the spool a patch is being read for, if it has one."""
    return _READING.get()
