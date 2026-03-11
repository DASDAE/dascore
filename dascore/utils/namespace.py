"""Utilities for binding method namespaces to objects."""

from __future__ import annotations

import functools
import inspect
from typing import ClassVar

from dascore.exceptions import ParameterError
from dascore.utils.plugins import get_entry_point_loaders, load_entry_point


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


def _register_namespace_subclass(
    cls: type[MethodNameSpace],
    registry: dict[str, type[MethodNameSpace]],
    base_name: str,
):
    """Register a namespace subclass by its declared name."""
    name = cls.__dict__.get("name")
    if name is None:
        return
    if not isinstance(name, str) or not name.isidentifier():
        msg = f"{name!r} is not a valid namespace name."
        raise ParameterError(msg)
    current = registry.get(name)
    if current is not None and current is not cls:
        msg = f"{base_name}.{name} is already registered."
        raise ParameterError(msg)
    registry[name] = cls


class PatchNameSpace(MethodNameSpace):
    """A namespace for Patch methods."""

    name: ClassVar[str | None] = None
    _namespace_subclasses: ClassVar[dict[str, type[MethodNameSpace]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_namespace_subclass(cls, PatchNameSpace._namespace_subclasses, "Patch")


class SpoolNameSpace(MethodNameSpace):
    """A namespace for Spool methods."""

    name: ClassVar[str | None] = None
    _namespace_subclasses: ClassVar[dict[str, type[MethodNameSpace]]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        _register_namespace_subclass(
            cls, SpoolNameSpace._namespace_subclasses, "BaseSpool"
        )


class _MethodNamespaceDescriptor:
    """Descriptor for binding a method namespace to an object."""

    def __init__(self, namespace_cls: type[MethodNameSpace]):
        self._namespace_cls = namespace_cls

    def __get__(self, obj, owner=None):
        if obj is None:
            return self._namespace_cls
        return self._namespace_cls(obj)


def _bind_method_namespace(
    owner_cls: type,
    name: str,
    namespace_cls: type[MethodNameSpace],
    *,
    overwrite: bool = False,
) -> type[MethodNameSpace]:
    """
    Bind a namespace class to an owner class under an attribute name.

    This is the low-level step which turns a namespace subclass into an
    accessible attribute such as ``Patch.viz`` or ``spool.my_ext``.
    It performs three tasks:

    1. Validate that ``name`` is a legal attribute name.
    2. Validate that ``namespace_cls`` matches the expected namespace base
       for ``owner_cls`` (for example, ``PatchNameSpace`` for ``Patch``).
    3. Attach a descriptor to ``owner_cls`` and record the binding in the
       owner's namespace registry.

    Parameters
    ----------
    owner_cls
        The class which should expose the namespace attribute.
    name
        The attribute name users will access on the owner.
    namespace_cls
        The namespace class to bind to ``owner_cls``.
    overwrite
        If ``True``, replace an existing binding for ``name``.

    Returns
    -------
    type[MethodNameSpace]
        The namespace class that was bound.
    """
    if not isinstance(name, str) or not name.isidentifier():
        msg = f"{name!r} is not a valid namespace name."
        raise ParameterError(msg)
    expected_base = getattr(owner_cls, "_namespace_base_class", MethodNameSpace)
    if not inspect.isclass(namespace_cls) or not issubclass(
        namespace_cls, expected_base
    ):
        msg = f"namespace_cls must be a {expected_base.__name__} subclass."
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


def _get_registered_method_namespaces(
    owner_cls: type,
) -> dict[str, type[MethodNameSpace]]:
    """Return registered method namespaces for a class."""
    return dict(getattr(owner_cls, "_method_namespace_registry", {}))


def _get_namespace_subclasses(
    namespace_base: type[MethodNameSpace],
) -> dict[str, type[MethodNameSpace]]:
    """Return known namespace subclasses for a namespace base."""
    return dict(getattr(namespace_base, "_namespace_subclasses", {}))


def _get_namespace_owner_cls(owner_cls: type) -> type:
    """Return the canonical class used to bind namespaces."""
    # Bind onto the shared owner class (eg BaseSpool) rather than each
    # concrete runtime subclass so one binding is reused everywhere.
    for cls in owner_cls.__mro__:
        if cls.__dict__.get("_namespace_entry_point_group") is not None:
            return cls
    return owner_cls


def load_namespace(owner_cls: type, name: str) -> bool:
    """Load and register a namespace from entry points if available."""
    owner_cls = _get_namespace_owner_cls(owner_cls)
    if name in _get_registered_method_namespaces(owner_cls):
        return True
    namespace_base = getattr(owner_cls, "_namespace_base_class", MethodNameSpace)
    namespace_cls = _get_namespace_subclasses(namespace_base).get(name)
    if namespace_cls is not None:
        # Namespace subclasses self-register by name at import time; delay the
        # actual descriptor binding until the attribute is first requested.
        _bind_method_namespace(owner_cls, name, namespace_cls)
        return True
    group = getattr(owner_cls, "_namespace_entry_point_group", None)
    if not group:
        return False
    if name not in get_entry_point_loaders(group):
        return False
    _bind_method_namespace(owner_cls, name, load_entry_point(group, name))
    return True


class NamespaceOwner:
    """Mixin for classes with lazily-loadable method namespaces."""

    _namespace_entry_point_group: str | None = None
    _namespace_attr_errors: ClassVar[dict[str, str]] = {}
    _namespace_base_class: type[MethodNameSpace] = MethodNameSpace

    @classmethod
    def get_registered_namespaces(cls):
        """Return registered method namespaces on the class."""
        return _get_registered_method_namespaces(cls)

    def __getattr__(self, item):
        """Try loading a lazily registered namespace before failing."""
        if load_namespace(self.__class__, item):
            descriptor = inspect.getattr_static(self.__class__, item)
            return descriptor.__get__(self, self.__class__)
        if item in self.__class__._namespace_attr_errors:
            raise AttributeError(self.__class__._namespace_attr_errors[item])
        msg = f"{self.__class__.__name__!r} object has no attribute {item!r}"
        raise AttributeError(msg)
