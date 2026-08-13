"""
The registry which lets a serialized document name the class it holds.

A document states its class in an ``object_type`` key holding a registered
name, never an import path. A dotted path would weld stored documents to
today's module layout and make reading one an arbitrary-import surface;
a registered name costs the same to write and has neither property.
"""

from __future__ import annotations

import re
import threading
import warnings
from typing import TypeVar, cast

from dascore.exceptions import InvalidModelTagError
from dascore.utils.plugins import FIBER_IO_GROUP, get_entry_point_loaders

T = TypeVar("T")

# The key a document states its class in.
TAG_FIELD = "object_type"

NAMESPACE_SEP = ":"

# DASCore's own models are registered bare, so a hand-authored file says
# `object_type: Cable` and a plugin's says `object_type: myplugin:Square`.
DASCORE_NAMESPACE = "dascore"

# `[namespace:]ClassName[-x.y.z]`. A python class name holds neither ":"
# nor "-", so the three parts can never be read for one another. The
# namespace accepts a capital because package names do (Terra15, PIL).
# Nothing reads a version today; the grammar only keeps room for one,
# where an absent version will mean the earliest.
TAG_PATTERN = re.compile(r"^(?:[A-Za-z_][\w.]*:)?[A-Za-z_]\w*(?:-\d+\.\d+\.\d+)?$")

_REGISTRY: dict[str, type] = {}

# Tags two different classes have claimed. Kept rather than resolved to
# either: a file written by the first would otherwise be read as the
# second, silently applying the wrong model to it.
_AMBIGUOUS: dict[str, tuple[str, str]] = {}

_sweep_lock = threading.Lock()

# Set once the io plugins have been swept looking for an unresolved tag.
_plugins_swept = False


def _derive_namespace(cls: type) -> str:
    """Return the namespace a class registers under."""
    # Derived rather than declared: it makes a plugin's models namespaced
    # with no ceremony, and leaves no way to squat a bare name.
    return cls.__module__.split(".", 1)[0]


def get_model_tag(cls: type) -> str | None:
    """
    Return the tag which names a model class in a document, if it has one.

    An out-of-tree class is namespaced by the package which declares it;
    DASCore's own classes are bare. None means the class cannot be named:
    a parametrized generic is spelled ``G[int]``, which no document could
    state and this module could not read back. Writing such a tag anyway
    would produce a file DASCore refuses to read.
    """
    namespace = _derive_namespace(cls)
    if namespace == DASCORE_NAMESPACE:
        tag = cls.__name__
    else:
        tag = f"{namespace}{NAMESPACE_SEP}{cls.__name__}"
    return tag if TAG_PATTERN.match(tag) else None


def register_model(cls: type) -> None:
    """
    Add a model class to the registry under its derived tag.

    Classes declared inside a function are skipped: nothing can resolve a
    name which only exists while its enclosing call runs, and two of them
    sharing a name is neither a mistake nor resolvable. So are classes
    whose derived tag is not a legal one; see `get_model_tag`.
    """
    if "<locals>" in cls.__qualname__ or (tag := get_model_tag(cls)) is None:
        return
    existing = _REGISTRY.get(tag)
    if existing is not None and _identity(existing) != _identity(cls):
        _report_collision(tag, existing, cls)
        return
    # A module re-imported under the same name replaces its own entry.
    _REGISTRY[tag] = cls


def _identity(cls: type) -> tuple[str, str]:
    """Return what makes a class the same class across a re-import."""
    return (cls.__module__, cls.__qualname__)


def _spell(cls: type) -> str:
    """Spell a class the way a collision message needs to."""
    return f"{cls.__module__}.{cls.__qualname__}"


def _report_collision(tag: str, existing: type, new: type) -> None:
    """Complain that two different classes want one tag."""
    msg = (
        f"Two models claim the tag {tag!r}: {_spell(existing)} and "
        f"{_spell(new)}. A tag must name one class; rename one of them."
    )
    # DASCore's own names are its own to keep unique, and a test pins it.
    if _derive_namespace(new) == DASCORE_NAMESPACE:
        raise InvalidModelTagError(msg)
    # Out of tree the collision may be between two packages a user merely
    # installed, which they cannot fix by renaming, so importing them both
    # still works. What the tag may not do is quietly resolve to one of
    # them: a file written by the first would then be read as the second.
    _AMBIGUOUS[tag] = (_spell(existing), _spell(new))
    _REGISTRY.pop(tag, None)
    warnings.warn(f"{msg} Documents naming it can no longer be read.", UserWarning)


def _sweep_plugin_modules() -> None:
    """
    Import the io plugins, defining any models they declare.

    Not `FiberIO.manager.load_plugins()`, which does the same thing and
    more: `dascore.io.core` imports `dascore.core.attrs`, which imports
    this module, so naming it here at module scope is a cycle -- and a
    function-level import is banned by PLC0415.
    """
    global _plugins_swept
    if _plugins_swept:
        return
    with _sweep_lock:
        # The flag is set after the imports rather than before, so a thread
        # arriving mid-sweep waits here rather than reading a registry which
        # is still filling and concluding a class is not installed. Sweeping
        # again after that wait costs nothing: every module it names is by
        # then in sys.modules.
        for loader in get_entry_point_loaders(FIBER_IO_GROUP).values():
            try:
                loader()
            except Exception:
                # A plugin which cannot be imported has no models to find.
                # It is not reported here: FiberIO warns about the same
                # plugin when it loads formats, and failing to resolve one
                # tag is not the place to announce a broken install.
                continue
        _plugins_swept = True


def _lookup(tag: object) -> type | None:
    """Return the class a tag names, or None if nothing usable names it."""
    if not isinstance(tag, str) or not TAG_PATTERN.match(tag) or tag in _AMBIGUOUS:
        return None
    if (cls := _REGISTRY.get(tag)) is not None:
        return cls
    # A format's models only exist once its module is imported, and io
    # modules are imported lazily, so an unknown name is worth one sweep.
    _sweep_plugin_modules()
    return _REGISTRY.get(tag)


def resolve_model_tag(tag: str) -> type | None:
    """
    Return the class a tag names, or None if nothing registers it.

    Raises if the tag could never name a class -- it is not a legal tag,
    or two classes have claimed it -- which is a document this process
    cannot read rather than one whose class it merely does not have.
    """
    if not isinstance(tag, str) or not TAG_PATTERN.match(tag):
        msg = (
            f"{tag!r} is not a legal {TAG_FIELD}. Expected a registered name, "
            "optionally namespaced, eg 'Cable' or 'myplugin:Square'."
        )
        raise InvalidModelTagError(msg)
    if tag in _AMBIGUOUS:
        first, second = _AMBIGUOUS[tag]
        msg = (
            f"The {TAG_FIELD} {tag!r} names two classes, {first} and "
            f"{second}, so which one wrote a document cannot be known."
        )
        raise InvalidModelTagError(msg)
    return _lookup(tag)


def check_tag_matches(cls: type, declared: type, tag: str) -> None:
    """
    Refuse a document whose tag names some other class.

    A tag naming a subclass is accepted: such a document holds everything
    the class being built declares, which is what a caller asking for the
    base class asked for.
    """
    if issubclass(declared, cls):
        return
    msg = (
        f"A document declaring {TAG_FIELD} {tag!r} cannot be read as "
        f"{cls.__name__}: {declared.__name__} is not one."
    )
    raise InvalidModelTagError(msg)


def resolve_tagged_model(
    tag: str | None,
    default: type[T] | None = None,
    source: str | None = None,
) -> type[T]:
    """
    Return the model class a tag names.

    Parameters
    ----------
    tag
        What the document declared, or None if it declared nothing.
    default
        The class to fall back on when the tag names one which is not
        registered, usually because a plugin which wrote it is not
        installed. Without one, an unresolved tag raises.
    source
        Where the document came from, named in errors. Deliberately not in
        the warning below, which would otherwise defeat the de-duplication
        that keeps a spool of many files from warning once per file.
    """
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
        # The registry holds every model, not just the caller's kind, so
        # this check is what makes the returned class a `default` at all.
        if default is not None and not issubclass(cls, default):
            msg = (
                f"The document{where} declares {TAG_FIELD} {tag!r}, which is "
                f"not a {default.__name__}."
            )
            raise InvalidModelTagError(msg)
        return cast("type[T]", cls)
    msg = (
        f"Nothing registers the {TAG_FIELD} {tag!r}. It was likely written "
        "by a package which is not installed."
    )
    if default is None:
        raise InvalidModelTagError(f"{msg[:-1]}{where}.")
    warnings.warn(f"{msg} Reading it as {default.__name__}.", UserWarning, stacklevel=2)
    return default


def registered_models() -> dict[str, type]:
    """Return the registered classes, keyed by the tag naming each."""
    return dict(_REGISTRY)
