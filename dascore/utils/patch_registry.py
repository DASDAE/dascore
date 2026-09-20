"""Name, resolve and fingerprint patch functions.

Every patch function is registered under a tag as it is decorated: bare for
DASCore's own (``normalize``), ``package:name`` for a plugin's.
[`fingerprint_call`](`dascore.utils.patch_registry.fingerprint_call`) gives
the id of one call's bound arguments, which a result's ``data_id`` is derived
from.

Examples
--------
>>> import dascore as dc
>>> from dascore.utils.patch_registry import fingerprint_call
>>>
>>> first = fingerprint_call(dc.proc.normalize, (), {"dim": "time"})
>>> assert first == fingerprint_call(dc.proc.normalize, ("time",))
"""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Mapping
from contextlib import suppress
from functools import lru_cache
from typing import Any

from pydantic.fields import FieldInfo

from dascore.exceptions import ParameterError
from dascore.utils.identity import (
    PatchMarker,
    callable_name,
    extract_patches,
    operation_id,
)

# Stands in for the patch while a call is bound to a signature. The bind
# only needs something to put in that slot; nothing ever looks at it.
_PATCH = object()

# Every patch function, by the tag which names it. Filled by
# `patch_function` as it decorates, so a function is registered exactly when
# its module is imported.
_REGISTERED: dict[str, Any] = {}

# Tags two different functions have claimed; see `_report_collision`.
_AMBIGUOUS: dict[str, tuple[str, str]] = {}

# DASCore's own patch functions are named bare, a plugin's by its package.
_DASCORE = "dascore"
_SEPARATOR = ":"

# `[package:]name`, which is what a package name and a python name make.
_TAG = re.compile(r"^(?:[A-Za-z_][\w.]*:)?[A-Za-z_]\w*$")

# What `dascore/__init__.py` leaves until something asks for it.
_DEFERRED_MODULES = ("dascore.viz",)

# Set once the install has been swept looking for an unregistered tag.
_swept = False

# Fingerprints of calls already made, so a loop over a spool pays for the
# digest of one call rather than of every one.
_FINGERPRINTS: dict[Any, str] = {}
_FINGERPRINT_LIMIT = 4096


def patch_function_tag(func) -> str | None:
    """
    Return the tag which names a patch function.

    DASCore's own are bare, a plugin's are namespaced by the package which
    declares them -- the rule `dascore.models.registry` uses for a model,
    and for the same reason: it leaves no way to squat a bare name.

    None means the function cannot be named. One defined inside a call is
    such a case: nothing can resolve a name which exists only while its
    enclosing call runs, and two of them sharing one is neither a mistake
    nor resolvable.
    """
    # A callable object carries neither, and is nameless for that reason
    # rather than for being defined inside a call; both take no tag.
    if "<locals>" in getattr(func, "__qualname__", ""):
        return None
    if not (name := getattr(func, "__name__", "")):
        return None
    namespace = getattr(func, "__module__", "").split(".", 1)[0]
    tag = name if namespace == _DASCORE else f"{namespace}{_SEPARATOR}{name}"
    return tag if _TAG.match(tag) else None


def register_patch_function(func) -> str | None:
    """Add a patch function to the registry under its derived tag."""
    tag = patch_function_tag(func)
    if tag is None:
        return None
    existing = _REGISTERED.get(tag)
    # A module re-imported under the same name replaces its own entry.
    if existing is not None and _spell(existing) != _spell(func):
        _report_collision(tag, existing, func)
        return None
    _REGISTERED[tag] = func
    return tag


def resolve_patch_function(name: str, module: str | None = None):
    """
    Return the patch function a tag names, else say what is missing.

    Nothing a tag names is imported to answer this, since a stored tag
    would otherwise be a way to run whatever it names. The sweep imports
    only what the *install* declares: DASCore's own deferred modules and
    the namespaces plugins register through entry points.

    Parameters
    ----------
    name
        The tag, as `patch_function_tag` spells one.
    module
        Where the function was defined, if known. Used only to say what to
        import.
    """
    if (found := _REGISTERED.get(name)) is not None:
        return found
    if name in _AMBIGUOUS:
        first, second = _AMBIGUOUS[name]
        msg = (
            f"The patch function {name!r} names two functions, {first} and "
            f"{second}, so which one it means cannot be known."
        )
        raise ParameterError(msg)
    _sweep_patch_functions()
    if (found := _REGISTERED.get(name)) is not None:
        return found
    raise ParameterError(_missing(name, module))


def _missing(name: str, module: str | None) -> str:
    """Return the message a tag nothing registers deserves."""
    package, _, leaf = name.rpartition(_SEPARATOR)
    if package == "__main__":
        return (
            f"No patch function {leaf!r} is registered in this process. It was "
            "defined in a script or a notebook session, which nothing can "
            "import; redefine it here and try again."
        )
    if not package:
        return (
            f"No patch function {leaf!r} is registered in this process, and "
            "DASCore defines none by that name."
        )
    where = f" It was defined in {module} --" if module else ""
    return (
        f"No patch function {leaf!r} from package {package!r} is registered in "
        f"this process.{where} import that module (or install {package}) and "
        "try again."
    )


def _spell(func) -> str:
    """Spell a function the way a collision message needs to."""
    module = getattr(func, "__module__", "?")
    return f"{module}.{getattr(func, '__qualname__', '?')}"


def _report_collision(tag: str, existing, new) -> None:
    """Complain that two different functions want one tag."""
    msg = (
        f"Two patch functions claim the tag {tag!r}: {_spell(existing)} and "
        f"{_spell(new)}. A tag must name one function; rename one of them."
    )
    # DASCore's own names are its own to keep unique, and a test pins it.
    if _SEPARATOR not in tag:
        raise ParameterError(msg)
    # Out of tree the collision may be between two packages a user merely
    # installed, which they cannot fix by renaming, so importing them both
    # still works. What the tag may not do is quietly resolve to one of
    # them: a tag stored for the first would then run the second.
    _AMBIGUOUS[tag] = (_spell(existing), _spell(new))
    _REGISTERED.pop(tag, None)
    warnings.warn(f"{msg} The tag no longer resolves to either.", UserWarning)


def _sweep_patch_functions() -> None:
    """
    Import what the install says defines patch functions.

    `dascore/__init__.py` leaves `dascore.viz` until something asks for it,
    and a plugin's namespace is imported when a patch is first asked for
    one, which resolving a tag never does. Both are declared by the
    install, so importing them is not the arbitrary import
    `resolve_patch_function` refuses.
    """
    global _swept
    if _swept:
        return
    # Imported here rather than at module scope: this module is imported
    # while `dascore.utils.patch` is still being imported, by way of the
    # checks it re-exports, and `dascore` is not built yet at that point.
    import importlib  # noqa: PLC0415

    from dascore.utils.namespace import _MethodNameSpace  # noqa: PLC0415
    from dascore.utils.plugins import get_entry_point_loaders  # noqa: PLC0415

    for name in _DEFERRED_MODULES:
        with suppress(ImportError):
            importlib.import_module(name)
    groups: set[str] = set()
    for kind in _MethodNameSpace.__subclasses__():
        group = getattr(kind, "entry_point_group", None)
        if isinstance(group, str) and group:
            groups.add(group)
    for group in sorted(groups):
        for loader in get_entry_point_loaders(group).values():
            # A plugin which will not import defines nothing to find, and
            # one unresolved tag is not the place to announce a bad install.
            with suppress(Exception):
                loader()
    _swept = True


def fingerprint_call(func, args: tuple = (), kwargs: dict | None = None) -> str:
    """
    Return the id of one call to a patch function.

    Positional and keyword spellings of one call bind alike, and an
    argument left at its default is the same call as one which passes it,
    so they are one operation with one id.

    Parameters
    ----------
    func
        The patch function, as decorated.
    args
        The positional arguments of the call, without the patch.
    kwargs
        The keyword arguments of the call.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.utils.patch_registry import fingerprint_call
    >>>
    >>> called = fingerprint_call(dc.proc.normalize, (), {"dim": "time"})
    >>> assert called == fingerprint_call(dc.proc.normalize, ("time",))
    """
    return call_operation(func, args, kwargs or {})[0]


def call_operation(func, args: tuple, kwargs: dict) -> tuple[str, list]:
    """Return the id of a call, and the patches among its arguments."""
    params, patches = call_inputs(func, args, kwargs)
    version = getattr(func, "__version__", "1.0")
    return _memoized_fingerprint(func, _call_name(func), params, version), patches


def call_inputs(func, args: tuple, kwargs: dict) -> tuple[dict, list]:
    """Return a call's parameters, and the patches found anywhere in them."""
    return extract_patches(_bind(func, args, kwargs))


def _memoized_fingerprint(owner, name: str, params: dict, version: str) -> str:
    """
    Return `operation_id(name, params, version)`, cached.

    A loop over a spool makes the same call every time. Hashing the
    parameters costs a few microseconds; the digest of their canonical
    JSON costs several times that.

    Parameters
    ----------
    owner
        What made the call: a patch function or a processor class.
    name
        The operation's name.
    params
        Its parameters, with any patch replaced by its marker.
    version
        Its version.
    """
    try:
        # The owner is in the key, not just its name. For one which has no
        # tag the name ends in `id(owner)`, and CPython reuses an address
        # once the owner is collected -- so a factory making one per call
        # could hand a later one the earlier one's fingerprint. Holding the
        # owner here makes the key exact and keeps the address from being
        # reused underneath it.
        key = (owner, name, version, _as_key(params))
    except TypeError:
        # Something unhashable -- an array argument, most often. Its
        # digest is the honest cost of saying which array it was.
        return operation_id(name, params, version)
    if (found := _FINGERPRINTS.get(key)) is None:
        found = operation_id(name, params, version)
        # Bounded, and simply stops growing rather than evicting: the
        # entries are one small string each, and a process which has made
        # four thousand distinct calls is not one this is hot for.
        if len(_FINGERPRINTS) < _FINGERPRINT_LIMIT:
            _FINGERPRINTS[key] = found
    return found


# The only leaves a cache key may be built from. The rule is not "hashable":
# it is "two of these are the same argument exactly when Python says they are
# equal". A pint quantity fails that -- `1 * m == 100 * cm` and the two hash
# alike, while the encoder spells them differently -- so caching on it
# would give one call two answers depending on what ran first.
_KEYABLE = (str, bytes, int, float, bool, type(None), PatchMarker)

# Beyond this many elements, working the key out costs more than the digest
# it saves.
_KEY_LIMIT = 32


def _as_key(value, budget: int = _KEY_LIMIT):
    """
    Return a hashable stand-in a different value cannot share.

    Raises `TypeError` for anything it cannot key safely or cheaply, which
    is the caller's signal to compute the digest instead of caching it.
    """
    if isinstance(value, Mapping):
        if len(value) > budget:
            raise TypeError(value)
        # The keys are typed too: `{1: "x"}` and `{True: "x"}` are equal
        # mappings to Python and different calls to the encoder.
        return (
            dict,
            tuple(
                (_as_key(k, budget - len(value)), _as_key(v, budget - len(value)))
                for k, v in value.items()
            ),
        )
    if isinstance(value, (list, tuple)):
        if len(value) > budget:
            raise TypeError(value)
        return (
            type(value),
            tuple(_as_key(x, budget - len(value)) for x in value),
        )
    # `type(value) in`, not `isinstance`: a subclass may compare equal to
    # its base and encode differently, which is the trap this exists for.
    if type(value) not in _KEYABLE:
        raise TypeError(value)
    if type(value) is float:
        # `0.0 == -0.0` and the two hash alike, while the encoder keeps
        # the sign; `repr` tells them apart, and NaN from NaN.
        return (float, repr(value))
    return (type(value), value)


def _call_name(func) -> str:
    """
    Return the name a call is fingerprinted under.

    The registry tag when the function has one. When it does not -- a patch
    function defined inside another call -- something was still done to the
    patch, and a `data_id` which did not move would claim it was not.
    So the call is named by where it was written instead: enough to tell it
    from another operation, and honestly not resolvable.
    """
    if (tag := patch_function_tag(func)) is not None:
        return tag
    # By its source rather than its address, which is reused once it is
    # collected. One which holds state its source does not show -- a
    # closure from a factory -- is refused, and its result's id is random.
    return callable_name(getattr(func, "raw_function", func))


# Bounded, and it holds function references: a process which builds patch
# functions in a loop should not keep every one of them, and the closures
# they captured, alive for its lifetime.
@lru_cache(maxsize=2048)
def _signature_of(func) -> inspect.Signature:
    """Return a function's signature, worked out once."""
    return inspect.signature(func)


def _signature(func) -> inspect.Signature:
    """
    Return the signature of the function inside a patch function.

    Cached on the function: `inspect.signature` is not cheap, a signature
    cannot change, and every call which is fingerprinted asks for one.
    """
    inner = getattr(func, "raw_function", func)
    try:
        return _signature_of(inner)
    except TypeError:  # something unhashable; ask the slow way
        return inspect.signature(inner)


def _check(func, args: tuple, kwargs: dict) -> None:
    """Refuse a call the function's signature does not accept."""
    try:
        _signature(func).bind(_PATCH, *args, **kwargs)
    except TypeError as error:
        msg = f"{_call_name(func)} cannot be called that way: {error}"
        raise ParameterError(msg) from error


def _bind(func, args: tuple, kwargs: dict) -> dict:
    """
    Return the arguments of a call as the one mapping they mean.

    Positional and keyword spellings of one call bind alike, an argument
    which only restates its default is left out, the patch is dropped, and
    a `**kwargs` group is spread back out so that a dimension given as an
    extra reads as itself.
    """
    signature = _signature(func)
    _check(func, args, kwargs)
    bound = signature.bind(_PATCH, *args, **kwargs)
    parameters = list(signature.parameters.values())
    # Left out rather than filled in, so that a parameter added later with
    # a default does not change the id of every call made before it.
    out = {
        key: value
        for key, value in bound.arguments.items()
        if not is_default(value, signature.parameters[key].default)
    }
    # The patch is what the operation is given, not part of what it is.
    out.pop(parameters[0].name, None)
    for parameter in parameters:
        if parameter.kind != inspect.Parameter.VAR_KEYWORD:
            continue
        extras = out.pop(parameter.name, {})
        # An extra named for a parameter the signature already has cannot
        # be told from it once both are one mapping. `append_dims` is the
        # live shape: its `*empty_dims` cannot be given by name, so
        # `empty_dims=3` lands in the `**kwargs` group and would overwrite
        # the group it looks like.
        # Against every declared name, not only the ones bound: a
        # positional-only parameter left at its default is absent here.
        if collisions := set(extras) & (set(out) | set(signature.parameters)):
            msg = (
                f"{_call_name(func)} was given {sorted(collisions)} both as a "
                "parameter and as an extra, so the call cannot be written "
                "down as one mapping."
            )
            raise ParameterError(msg)
        out |= extras
    return out


def is_default(value: Any, default: Any) -> bool:
    """Return True if a value only restates the default it was declared with."""
    default = _resolve_default(default)
    if default is inspect.Parameter.empty:
        return False
    if value is default:
        return True
    # Keyed rather than compared: `0.0 == -0.0` and `1 == True`, and each
    # pair is two calls. A list restates a tuple, as the encoder reads them.
    if isinstance(value, list) and isinstance(default, list | tuple):
        value, default = tuple(value), tuple(default)
    try:
        return _as_key(value) == _as_key(default)
    except TypeError:
        return False


def _resolve_default(default: Any) -> Any:
    """Return the value a parameter defaults to."""
    # A function which spells its default `x=Field(default=3)` means 3; the
    # FieldInfo itself would otherwise become the default value.
    if isinstance(default, FieldInfo):
        return default.get_default(call_default_factory=True)
    return default
