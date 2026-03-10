"""Utilities for binding method namespaces to objects."""

from __future__ import annotations

import functools
import inspect
from importlib.metadata import entry_points
from typing import Any

from dascore.exceptions import ParameterError


def _pass_through_method(func):
    """Decorator for marking functions as methods on namedspace parent class."""

    @functools.wraps(func)
    def _func(self, *args, **kwargs):
        obj = self._obj
        return func(obj, *args, **kwargs)

    return _func


class _NameSpaceMeta(type):
    """Metaclass for namespace class."""

    def __setattr__(cls, key, value):
        if callable(value):
            value = _pass_through_method(value)
        super().__setattr__(key, value)


class MethodNameSpace(metaclass=_NameSpaceMeta):
    """A namespace for class methods."""

    def __init__(self, obj):
        self._obj = obj

    def __init_subclass__(cls, **kwargs):
        """Wrap all public methods."""
        for key, val in vars(cls).items():
            if callable(val):  # passes to _NameSpaceMeta settattr
                setattr(cls, key, val)


class _MethodNamespaceDescriptor:
    """Descriptor for binding a method namespace to an object."""

    def __init__(self, namespace_cls: type[MethodNameSpace]):
        self._namespace_cls = namespace_cls

    def __get__(self, obj, owner=None):
        if obj is None:
            return self._namespace_cls
        return self._namespace_cls(obj)


def register_method_namespace(
    owner_cls: type,
    name: str,
    namespace_cls: type[MethodNameSpace],
    *,
    overwrite: bool = False,
) -> type[MethodNameSpace]:
    """Register a namespace on a class."""
    if not isinstance(name, str) or not name.isidentifier():
        msg = f"{name!r} is not a valid namespace name."
        raise ParameterError(msg)
    if not inspect.isclass(namespace_cls) or not issubclass(
        namespace_cls, MethodNameSpace
    ):
        msg = "namespace_cls must be a MethodNameSpace subclass."
        raise ParameterError(msg)

    sentinel = object()
    registry_name = "_method_namespace_registry"
    registry = dict(getattr(owner_cls, registry_name, {}))
    existing = inspect.getattr_static(owner_cls, name, sentinel)
    current = registry.get(name)
    if existing is not sentinel and current is not namespace_cls and not overwrite:
        msg = f"{owner_cls.__name__}.{name} already exists."
        raise ParameterError(msg)

    registry[name] = namespace_cls
    setattr(owner_cls, registry_name, registry)
    setattr(owner_cls, name, _MethodNamespaceDescriptor(namespace_cls))
    return namespace_cls


def get_registered_method_namespaces(owner_cls: type) -> dict[str, type[MethodNameSpace]]:
    """Return registered method namespaces for a class."""
    return dict(getattr(owner_cls, "_method_namespace_registry", {}))


class NamespaceManager:
    """Manage lazily loaded method namespaces from entry points."""

    def __init__(self, owner_cls: type, entry_point_group: str):
        self._owner_cls = owner_cls
        self._entry_point_group = entry_point_group
        self._loaded_names: set[str] = set()

    @functools.cached_property
    def _eps(self) -> dict[str, Any]:
        """Return cached entry points keyed by namespace name."""
        return {x.name: x.load for x in entry_points(group=self._entry_point_group)}

    def load_plugin(self, name: str) -> bool:
        """Load a namespace plugin by name if it is available."""
        if name in self._loaded_names:
            return name in get_registered_method_namespaces(self._owner_cls)
        if name not in self._eps:
            return False
        namespace_cls = self._eps[name]()
        register_method_namespace(self._owner_cls, name, namespace_cls)
        self._loaded_names.add(name)
        return True
