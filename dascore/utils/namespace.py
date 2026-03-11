"""Utilities for binding method namespaces to objects."""

from __future__ import annotations

import functools
import warnings
from collections import defaultdict
from collections.abc import Mapping
from typing import ClassVar

from dascore.utils.mapping import FrozenDict
from dascore.utils.plugins import maybe_load_entry_point


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
    _registry: ClassVar[Mapping[str, dict]] = defaultdict(dict)

    def __init__(self, obj):
        self._obj = obj

    def __init_subclass__(cls, **kwargs):
        """Wrap all public methods."""
        super().__init_subclass__(**kwargs)
        for key, val in vars(cls).items():
            if callable(val):  # passes to _NameSpaceMeta settattr
                setattr(cls, key, val)
        # Register all subclasses.
        if cls.name is not None:
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


class PatchNameSpace(_MethodNameSpace):
    """A namespace for Patch methods."""

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str] = "dascore.patch_namespace"


class SpoolNameSpace(_MethodNameSpace):
    """A namespace for Spool methods."""

    name: ClassVar[str | None] = None
    entry_point_group: ClassVar[str] = "dascore.spool_namespace"


class NamespaceOwner:
    """Mixin for classes with lazily-loadable method namespaces."""

    _namespace_entry_point_group: str | None = None
    _namespace_attr_errors: ClassVar[dict[str, str]] = {}

    @classmethod
    def get_registered_namespaces(cls):
        """Return registered method namespaces on the class."""
        registry = _MethodNameSpace._registry.get(cls._namespace_entry_point_group, {})
        return FrozenDict(registry)

    def __getattr__(self, item):
        """Try loading a lazily registered namespace before failing."""
        # Unknown attribute; try loading the namespaces.
        maybe_load_entry_point(self._namespace_entry_point_group, item)

        # Once loaded the registry should be populated.
        registry = _MethodNameSpace._registry.get(self._namespace_entry_point_group, {})
        if item in registry:
            return registry[item](self)

        # If that fails, see if there is anything specific for this name to raise.
        if item in self._namespace_attr_errors:
            msg = self._namespace_attr_errors[item]
        else:
            msg = f"{self.__class__.__name__} has no attribute '{item}'"
        raise AttributeError(msg)
