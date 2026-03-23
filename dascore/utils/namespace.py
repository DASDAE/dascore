"""Utilities for binding method namespaces to objects."""

from __future__ import annotations

import functools
import warnings
from collections import defaultdict
from pathlib import Path
from typing import ClassVar

import pandas as pd

from dascore.exceptions import DASCorePluginError
from dascore.utils.mapping import FrozenDict
from dascore.utils.plugins import maybe_load_entry_point

_PLUGIN_REGISTRY_DIR = Path(__file__).parent.parent / "plugin_registry"


@functools.cache
def _load_plugin_registry(entry_point_group: str | None) -> dict[str, tuple[str, str]]:
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
        return {}
    # "dascore.patch_namespace" -> "patch"
    stem = entry_point_group.split(".")[-1].replace("_namespace", "")
    csv_path = _PLUGIN_REGISTRY_DIR / f"{stem}.csv"
    if not csv_path.exists():
        return {}
    df = pd.read_csv(csv_path)
    return dict(
        zip(
            df["namespace"],
            zip(df["package_name"], df["package_url"], strict=True),
            strict=True,
        )
    )


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
            instance = registry[item](self)
            self.__dict__[item] = instance
            return instance

        # Check plugin registry for a known third-party package that provides this.
        plugin_registry = _load_plugin_registry(self._namespace_entry_point_group)
        if item in plugin_registry:
            package_name, package_url = plugin_registry[item]
            msg = (
                f"{self.__class__.__name__} has a registered namespace of '{item}' "
                f"provided by '{package_name}' but it is not installed. "
                f"Install it from: {package_url}"
            )
            raise DASCorePluginError(msg)

        # If that fails, see if there is anything specific for this name to raise.
        if item in self._namespace_attr_errors:
            msg = self._namespace_attr_errors[item]
        else:
            msg = f"{self.__class__.__name__} has no attribute '{item}'"
        raise AttributeError(msg)
