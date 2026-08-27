"""Utilities for binding method namespaces to objects."""

from __future__ import annotations

import functools
import keyword
import warnings
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from threading import RLock
from typing import ClassVar

import pandas as pd

from dascore.exceptions import DASCorePluginError
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import _locked, _reinit_after_fork
from dascore.utils.plugins import maybe_load_entry_point

_PLUGIN_REGISTRY_DIR = Path(__file__).parent.parent / "plugin_registry"


@functools.cache
def _load_plugin_registry(
    entry_point_group: str | None,
) -> FrozenDict[str, tuple[str, str]]:
    """Load plugin registry CSV for the given entry point group.

    Parameters
    ----------
    entry_point_group
        The entry point group string (e.g. "dascore.patch_namespace").

    Returns
    -------
    A mapping of namespace name to (package_name, package_url).
    """
    if entry_point_group is None:
        return FrozenDict()
    # "dascore.patch_namespace" -> "patch"
    stem = entry_point_group.split(".")[-1].replace("_namespace", "")
    csv_path = _PLUGIN_REGISTRY_DIR / f"{stem}.csv"
    if not csv_path.exists():
        return FrozenDict()
    df = pd.read_csv(csv_path)
    return FrozenDict(
        zip(
            df["namespace"],
            zip(df["package_name"], df["package_url"], strict=True),
            strict=True,
        )
    )


def _check_namespace_name(name) -> None:
    """Refuse a namespace name a host could not hand out."""
    if not isinstance(name, str) or not name.isidentifier():
        msg = f"A namespace name must be a Python identifier, got {name!r}."
        raise ValueError(msg)
    if keyword.iskeyword(name):
        # `host.class` is a syntax error however the namespace registered.
        # Soft keywords -- match, type -- are legal attribute names and pass.
        msg = f"A namespace name must not be a Python keyword, got {name!r}."
        raise ValueError(msg)
    if name.startswith("_"):
        # A private name belongs to the host, which resolves its own before
        # the namespaces are searched, so one registered here is unreachable.
        msg = f"A namespace name must be public, got {name!r}."
        raise ValueError(msg)


def _pass_to_host_method(func):
    """Decorator for marking functions as methods on namedspace parent class."""

    @functools.wraps(func)
    def _func(self: _MethodNameSpace, *args, **kwargs):
        # Rebind namespace methods so they act like methods on the host object.
        return func(self._obj, *args, **kwargs)

    return _func


class _NameSpaceMeta(type):
    """Metaclass for namespace class."""

    def __setattr__(cls, key, value):
        if callable(value) and not key.startswith("_"):
            value = _pass_to_host_method(value)
        super().__setattr__(key, value)


class _MethodNameSpace(metaclass=_NameSpaceMeta):
    """
    A namespace for class methods.

    Parameters
    ----------
    name
        The name for this namespace (eg 'viz')
    host_name
        The host name for this namespace (eg "Patch").
    """

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str | None] = None

    # This is where the namespaces are stored.
    _registry: ClassVar[defaultdict[str | None, dict[str, type[_MethodNameSpace]]]] = (
        defaultdict(dict)
    )
    # Guards every read and write of _registry.
    _registry_lock: ClassVar[RLock] = RLock()

    def __init__(self, obj):
        self._obj = obj

    def __init_subclass__(cls, **kwargs):
        """Wrap all public methods."""
        super().__init_subclass__(**kwargs)
        for key, val in vars(cls).items():
            if callable(val):  # passes to _NameSpaceMeta setattr
                setattr(cls, key, val)
        # Register all subclasses.
        if cls.name is not None:
            _check_namespace_name(cls.name)
            with cls._registry_lock:
                registry = cls._registry[cls.entry_point_group]
                existing = registry.get(cls.name)
                if existing is not None and existing is not cls:
                    msg = (
                        f"Namespace collision for group {cls.entry_point_group!r} and "
                        f"name {cls.name!r}: replacing "
                        f"{existing.__module__}.{existing.__name__} with "
                        f"{cls.__module__}.{cls.__name__}."
                    )
                    warnings.warn(msg, UserWarning, stacklevel=2)
                registry[cls.name] = cls

    @classmethod
    @_locked("_registry_lock")
    def _get_namespace_type(cls, group: str | None, name: str):
        """Return the namespace class registered to a group, or None."""
        return cls._registry.get(group, {}).get(name)


@_reinit_after_fork
def _reinit_namespace_locks():
    """Install a fresh registry lock; see _reinit_after_fork."""
    _MethodNameSpace._registry_lock = RLock()


class PatchNameSpace(_MethodNameSpace):
    """A namespace for Patch methods."""

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str] = "dascore.patch_namespace"


class SpoolNameSpace(_MethodNameSpace):
    """A namespace for Spool methods."""

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str] = "dascore.spool_namespace"


class InventoryNameSpace(_MethodNameSpace):
    """A namespace for Inventory methods."""

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str] = "dascore.inventory_namespace"


class AnnotationNameSpace(_MethodNameSpace):
    """A namespace for AnnotationSet methods."""

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str] = "dascore.annotation_namespace"


class NamespaceOwner:
    """
    Mixin for classes with lazily-loadable method namespaces.

    List this before any base which defines its own ``__getattr__``, as a
    pydantic model does. The one the MRO reaches first is the only one
    called, so a host which lists this last resolves no namespace at all.
    """

    _namespace_entry_point_group: ClassVar[str | None] = None
    _namespace_attr_errors: ClassVar[dict[str, str]] = {}

    @classmethod
    def get_registered_namespaces(cls):
        """Return registered method namespaces on the class."""
        with _MethodNameSpace._registry_lock:
            registry = _MethodNameSpace._registry.get(
                cls._namespace_entry_point_group, {}
            )
            return FrozenDict(registry)

    def __getattr__(self, item):
        """Try loading a lazily registered namespace before failing."""
        # A namespace is never private, so a private name is the host's:
        # a pydantic one keeps its private attributes behind its own
        # __getattr__, and looking one up must not import a plugin.
        if not item.startswith("_"):
            # Unknown attribute; try loading the namespaces. Plugin imports
            # must not run while a lock is held.
            group = self._namespace_entry_point_group
            maybe_load_entry_point(group, item)

            # Once loaded the registry should be populated. The wrapper is
            # built per access rather than stored on the host: one kept there
            # rides into a copy still bound to the object it was built for,
            # and rebuilding costs a fraction of any call made through it.
            if namespace_type := _MethodNameSpace._get_namespace_type(group, item):
                return namespace_type(self)

            # Check the registry for a known third-party package providing this.
            plugin_registry = _load_plugin_registry(group)
            if item in plugin_registry:
                package_name, package_url = plugin_registry[item]
                msg = (
                    f"{self.__class__.__name__} has a registered namespace of "
                    f"'{item}' provided by '{package_name}' but it is not "
                    f"installed. Install it from: {package_url}"
                )
                raise DASCorePluginError(msg)

        # The host may resolve names of its own; a pydantic one resolves its
        # private attributes here. Its failure is not the final word: this
        # class has a better message for a name neither of them knows.
        if (parent := getattr(super(), "__getattr__", None)) is not None:
            with suppress(AttributeError):
                return parent(item)

        # If that fails, see if there is anything specific for this name to raise.
        if item in self._namespace_attr_errors:
            msg = self._namespace_attr_errors[item]
        else:
            msg = f"{self.__class__.__name__} has no attribute '{item}'"
        raise AttributeError(msg)
