"""
The registry which lets a serialized document name the class it holds.

A document states its class in an ``object_type`` key holding a registered
name, never an import path. A dotted path would weld stored documents to
today's module layout and make reading one an arbitrary-import surface;
a registered name costs the same to write and has neither property.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping

from dascore.exceptions import InvalidModelTagError
from dascore.utils.plugins import get_entry_point_loaders

# The key a document states its class in. Specific enough that no reader's
# header and no user's extra attribute is expected to spell it, which is why
# the validator may consume it wherever it appears.
TAG_FIELD = "object_type"

NAMESPACE_SEP = ":"

# DASCore's own models are registered bare, so a hand-authored file says
# `object_type: Cable` and a plugin's says `object_type: myplugin:Square`.
DASCORE_NAMESPACE = "dascore"

FIBER_IO_GROUP = "dascore.fiber_io"

# `[namespace:]ClassName[-x.y.z]`. A python class name holds neither ":"
# nor "-", so the three parts can never be read for one another. Nothing
# reads a version today; the grammar only keeps room for one, where an
# absent version will mean the earliest.
TAG_PATTERN = re.compile(r"^(?:[a-z_][\w.]*:)?[A-Za-z_]\w*(?:-\d+\.\d+\.\d+)?$")

_REGISTRY: dict[str, type] = {}

# Set once the io plugins have been swept looking for an unresolved tag.
_plugins_swept = False


def _derive_namespace(cls: type) -> str:
    """Return the namespace a class registers under."""
    # Derived rather than declared: it makes a plugin's models namespaced
    # with no ceremony, and leaves no way to squat a bare name.
    return cls.__module__.split(".", 1)[0]


def get_model_tag(cls: type) -> str:
    """
    Return the tag which names a model class in a document.

    An out-of-tree class is namespaced by the package which declares it;
    DASCore's own classes are bare.
    """
    namespace = _derive_namespace(cls)
    if namespace == DASCORE_NAMESPACE:
        return cls.__name__
    return f"{namespace}{NAMESPACE_SEP}{cls.__name__}"


def register_model(cls: type) -> None:
    """
    Add a model class to the registry under its derived tag.

    Classes declared inside a function are skipped: nothing can resolve a
    name which only exists while its enclosing call runs, and two of them
    sharing a name is neither a mistake nor resolvable.
    """
    if "<locals>" in cls.__qualname__:
        return
    tag = get_model_tag(cls)
    existing = _REGISTRY.get(tag)
    if existing is not None and _identity(existing) != _identity(cls):
        _report_collision(tag, existing, cls)
    # A module re-imported under the same name replaces its own entry.
    _REGISTRY[tag] = cls


def _identity(cls: type) -> tuple[str, str]:
    """Return what makes a class the same class across a re-import."""
    return (cls.__module__, cls.__qualname__)


def _report_collision(tag: str, existing: type, new: type) -> None:
    """Complain that two different classes want one tag."""
    msg = (
        f"Two models claim the tag {tag!r}: {existing.__module__}."
        f"{existing.__qualname__} and {new.__module__}.{new.__qualname__}. "
        "A tag must name one class; rename one of them."
    )
    # DASCore's own names are its own to keep unique, and a test pins it.
    # Out of tree the collision may be between two packages a user merely
    # installed, which they cannot fix by renaming, so it warns and the
    # last registration wins -- as duplicate entry points already do.
    if _derive_namespace(new) == DASCORE_NAMESPACE:
        raise InvalidModelTagError(msg)
    warnings.warn(msg, UserWarning, stacklevel=2)


def _sweep_plugin_modules() -> None:
    """Import the io plugins, defining any models they declare."""
    global _plugins_swept
    if _plugins_swept:
        return
    _plugins_swept = True
    for loader in get_entry_point_loaders(FIBER_IO_GROUP).values():
        try:
            loader()
        except Exception:
            # A plugin which cannot be imported has no models to find. It is
            # not reported here: FiberIO warns about the same plugin when it
            # loads formats, and a failure to resolve one tag is not the
            # place to announce an unrelated broken install.
            continue


def resolve_model_tag(tag: str) -> type | None:
    """
    Return the class a tag names, or None if nothing registers it.

    Raises if the tag is not a legal tag at all, which is a malformed
    document rather than an unknown class.
    """
    if not isinstance(tag, str) or not TAG_PATTERN.match(tag):
        msg = (
            f"{tag!r} is not a legal {TAG_FIELD}. Expected a registered name, "
            "optionally namespaced, eg 'Cable' or 'myplugin:Square'."
        )
        raise InvalidModelTagError(msg)
    if (cls := _REGISTRY.get(tag)) is not None:
        return cls
    # A format's models only exist once its module is imported, and io
    # modules are imported lazily, so an unknown name is worth one sweep.
    _sweep_plugin_modules()
    return _REGISTRY.get(tag)


def check_tag_matches(cls: type, tag: str) -> None:
    """
    Refuse a document whose tag names a class the one being built is not.

    A tag naming a subclass is accepted: such a document holds everything
    the class being built declares, which is what a caller asking for the
    base class asked for. An unregistered tag is accepted too -- the caller
    named the class, so there is nothing for the document to disagree with.
    """
    declared = resolve_model_tag(tag)
    if declared is None or issubclass(declared, cls):
        return
    msg = (
        f"A document declaring {TAG_FIELD} {tag!r} cannot be read as "
        f"{cls.__name__}: {declared.__name__} is not one."
    )
    raise InvalidModelTagError(msg)


def resolve_tagged_model(
    data: Mapping,
    default: type | None = None,
    source: str | None = None,
) -> type:
    """
    Return the model class a document names.

    Parameters
    ----------
    data
        The document, which states its class in its ``object_type`` key.
    default
        The class to fall back on when the document names one which is not
        registered, usually because a plugin which wrote it is not
        installed. Without one, an unresolved name raises.
    source
        Where the document came from, used in messages.
    """
    tag = data.get(TAG_FIELD) if isinstance(data, Mapping) else None
    where = f" in {source}" if source else ""
    if tag is None:
        if default is None:
            msg = (
                f"The document{where} declares no {TAG_FIELD}, and nothing "
                "else says which class it holds."
            )
            raise InvalidModelTagError(msg)
        return default
    if (cls := resolve_model_tag(tag)) is not None:
        return cls
    msg = (
        f"Nothing registers the {TAG_FIELD} {tag!r}{where}. It was likely "
        "written by a package which is not installed."
    )
    if default is None:
        raise InvalidModelTagError(msg)
    warnings.warn(f"{msg} Reading it as {default.__name__}.", UserWarning)
    return default


def registered_models() -> dict[str, type]:
    """Return the registered classes, keyed by the tag naming each."""
    return dict(_REGISTRY)
