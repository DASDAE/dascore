"""
Accessors for DASCore Patches/Spools.
"""

from __future__ import annotations

import functools

from dascore.exceptions import AccessorRegistrationError


class _Accessor:
    """
    Non-caching descriptor that attaches an accessor class to a host object.

    Accessing through the class returns the accessor class itself (useful for
    introspection and documentation).  Accessing through an instance
    instantiates the accessor with that instance and returns it.

    AttributeError raised inside the accessor ``__init__`` is converted to
    RuntimeError so that Python's attribute lookup machinery does not silently
    swallow the message.
    """

    def __init__(self, name: str, accessor_class: type) -> None:
        self._name = name
        self._accessor_class = accessor_class

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor_class
        try:
            return self._accessor_class(obj)
        except AttributeError as err:
            raise RuntimeError(
                f"Accessor {self._name!r} is not valid for this "
                f"{type(obj).__name__}: {err}"
            ) from err


def _register_accessor(name: str, host_class: type):
    """
    Return a decorator that registers an accessor on *host_class* under *name*.

    Parameters
    ----------
    name
        The attribute name to register on *host_class*.
    host_class
        The class to attach the accessor descriptor to.
    """
    if not name.isidentifier():
        raise AccessorRegistrationError(f"{name!r} is not a valid Python identifier.")
    if name.startswith("_"):
        raise AccessorRegistrationError(
            f"Accessor name {name!r} must not start with an underscore."
        )

    def decorator(accessor_class: type) -> type:
        # Idempotent: same class already registered under this name → no-op.
        existing = host_class.__dict__.get(name)
        if (
            isinstance(existing, _Accessor)
            and existing._accessor_class is accessor_class
        ):
            return accessor_class
        # Any other existing attribute (own or inherited) is a hard conflict.
        if hasattr(host_class, name):
            raise AccessorRegistrationError(
                f"Cannot register accessor {name!r} on {host_class.__name__}: "
                f"'{name}' is already an attribute of {host_class.__name__}."
            )
        setattr(host_class, name, _Accessor(name, accessor_class))
        # Track registered names so __dir__ can surface them.
        if "_accessors" not in host_class.__dict__:
            host_class._accessors = set()
        host_class._accessors.add(name)
        return accessor_class

    return decorator


def register_patch_accessor(name: str):
    """
    Register an accessor on Patch under the given name.

    Parameters
    ----------
    name
        The attribute name to register on Patch.

    Examples
    --------
    >>> import dascore as dc
    >>> @dc.register_patch_accessor("qc")
    ... class QCAccessor:
    ...     def __init__(self, patch):
    ...         self._patch = patch
    ...         if "time" not in patch.dims:
    ...             raise AttributeError("qc accessor requires a time dimension")
    ...
    ...     def snr(self):
    ...         ...
    """
    from dascore.core.patch import Patch

    return _register_accessor(name, Patch)


def register_spool_accessor(name: str):
    """
    Register an accessor on BaseSpool under the given name.

    Parameters
    ----------
    name
        The attribute name to register on BaseSpool.

    Examples
    --------
    >>> import dascore as dc
    >>> @dc.register_spool_accessor("event")
    ... class EventAccessor:
    ...     def __init__(self, spool):
    ...         self._spool = spool
    ...
    ...     def by_time(self, start, stop):
    ...         ...
    """
    from dascore.core.spool import BaseSpool

    return _register_accessor(name, BaseSpool)


def _attach_to_patch_accessor(namespace: str, func: callable) -> None:
    """
    Attach *func* as a method on the named patch accessor namespace.

    Creates and registers a dynamic accessor class on Patch if one does not
    already exist for *namespace*.  *func* must follow the ``patch_function``
    convention: its first positional argument is a Patch instance.

    Parameters
    ----------
    namespace
        The accessor name under which *func* will appear on Patch.
    func
        A callable with signature ``(patch, *args, **kwargs)``.  Its
        ``__name__`` becomes the method name on the accessor.
    """
    from dascore.core.patch import Patch

    existing = Patch.__dict__.get(namespace)
    if isinstance(existing, _Accessor):
        acc_cls = existing._accessor_class
    elif not hasattr(Patch, namespace):
        # Build a minimal dynamic accessor class and register it.
        def _init(self, patch):
            self._patch = patch

        acc_cls = type(
            f"_{namespace.capitalize()}PatchAccessor",
            (),
            {"__init__": _init},
        )
        _register_accessor(namespace, Patch)(acc_cls)
    else:
        raise AccessorRegistrationError(
            f"Cannot attach to namespace {namespace!r}: "
            f"'{namespace}' is already a non-accessor attribute of Patch."
        )

    # Bridge: translate accessor self → patch so the wrapped function
    # still receives the Patch as its first argument.
    @functools.wraps(func)
    def _method(self, *args, **kwargs):
        return func(self._patch, *args, **kwargs)

    setattr(acc_cls, func.__name__, _method)
