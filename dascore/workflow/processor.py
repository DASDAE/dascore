"""
A patch operation, as a task.

Any function decorated with `dascore.patch_function` is one operation, and
[`PatchOp`](`dascore.workflow.processor.PatchOp`) is that operation said as
an object: the function's name and the arguments it was given, which can be
compared, fingerprinted, written to a file and joined into a pipe.

One class stands for every patch function, rather than one class each. What
identifies an operation is its name and its arguments, and a name is a
string; manufacturing ninety classes to hold ninety strings buys nothing and
costs a registry tag, an import and a lookup for each.

[`PatchProcessor`](`dascore.workflow.processor.PatchProcessor`) is the base
for the few operations which are eventually written out by hand, because
they want a kernel seam. When one exists, `fn.op(...)` returns it instead:
one name, one implementation.

An operation is named by a registry tag rather than by where a patch keeps
it, so a function bound to nothing -- one a user has only just written -- is
nameable too, and a plugin's `normalize` can never be read as DASCore's.

Examples
--------
>>> import dascore as dc
>>>
>>> patch = dc.get_example_patch()
>>> op = dc.proc.normalize.op(dim="time")
>>> assert op(patch).equals(patch.normalize(dim="time"))
"""

from __future__ import annotations

import inspect
import re
import warnings
from collections.abc import Mapping
from contextlib import suppress
from functools import cache, cached_property
from typing import Any, ClassVar

from pydantic import Field

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from dascore.workflow.checks import attr_type, check_patch_attrs, check_patch_coords
from dascore.workflow.serialize import digest
from dascore.workflow.task import Task, _resolve_default

# Stands in for the patch while a call is bound to a signature. The bind
# only needs something to put in that slot; nothing ever looks at it.
_PATCH = object()

# The names which have a hand-written class, so `fn.op(...)` can return one.
# Empty here: the first entries arrive with the first plan/kernel split.
_IMPLEMENTATIONS: dict[str, type[PatchProcessor]] = {}

# Every patch function, by the tag which names it in a document. Filled by
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


class PatchProcessor(Task):
    """
    A task which runs on a patch, written out by hand.

    Subclasses declare their parameters as fields and say what a patch has
    to carry for the operation to mean anything. They exist where an
    operation wants a seam a whole function does not have -- a kernel to
    dispatch, a plan to reuse -- and are registered with
    [`register_implementation`](`dascore.workflow.processor.register_implementation`)
    so that the patch function's name reaches them.
    """

    # What the patch must hold. Stated on the class rather than passed to
    # `run`, because it is a property of the operation and not of the call.
    required_dims: ClassVar[tuple[str, ...] | str | None] = None
    required_coords: ClassVar[tuple[str, ...] | str | None] = None
    required_attrs: ClassVar[attr_type] = None
    # What the result is, when the operation changes it; None leaves it as
    # it was, and "" clears it.
    data_type: ClassVar[str | None] = None

    def check(self, patch: PatchType) -> PatchType:
        """
        Refuse a patch which does not carry what the operation needs.

        A subclass which declares `required_dims`, `required_coords` or
        `required_attrs` has to call this from its own `run`; nothing calls
        it for you yet. The framework `run` which will call it arrives with
        the first plan/kernel split.

        Parameters
        ----------
        patch
            The patch to check.

        Returns
        -------
        The patch, unchanged, so the call can stand in front of the work.

        Raises
        ------
        PatchCoordinateError
            If a required dimension or coordinate is missing.
        PatchAttributeError
            If a required attr is missing, or holds a different value.
        """
        check_patch_coords(patch, dims=self.required_dims, coords=self.required_coords)
        return check_patch_attrs(patch, self.required_attrs)


class PatchOp(Task):
    """
    Any patch function, as a task.

    Parameters
    ----------
    name
        What the operation is called on a patch. A dotted name walks a
        namespace, so a plugin's function is `"myplugin.denoise"`.
    kwargs
        The arguments the operation was given, bound against its signature,
        so that two spellings of one call are one mapping.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.workflow import PatchOp
    >>>
    >>> patch = dc.get_example_patch()
    >>> op = PatchOp(name="pass_filter", kwargs={"time": (10, 100)})
    >>> assert op(patch).equals(patch.pass_filter(time=(10, 100)))
    """

    name: str
    kwargs: dict[str, Any] = Field(default_factory=dict)
    # The version the *function* is declared at, not this class's. Held as
    # a field rather than looked up when asked, so a document reproduces
    # the fingerprint it recorded even after the function has moved on --
    # and so the class version stays what every other task's document
    # machinery compares against.
    version: str = ""
    # Where the function was defined when this was written down. Carried so
    # a document naming a function this process does not have can say what
    # to import. Informational only: it is in the document and not in the
    # fingerprint, because moving a function between modules does not make
    # it a different operation.
    module: str = ""

    def __init__(self, **data):
        """
        Build the operation, refusing one no patch could run.

        Checked here rather than in a validator so that the answer is the
        `ParameterError` the rest of DASCore raises, rather than the
        `ValidationError` pydantic wraps one in.
        """
        name = data.get("name")
        if isinstance(name, str):
            # Bound rather than merely resolved: an argument the signature
            # rejects is a broken operation, and it is worth saying so
            # where it was written down rather than when a patch finally
            # reaches it. The result is kept, so an operation written by
            # hand carries the same defaults `.op(...)` fills in and the
            # two are one operation with one fingerprint.
            function = resolve_patch_function(name, data.get("module"))
            args, kwargs = _as_call(function, data.get("kwargs") or {})
            data["kwargs"] = _bind(function, args, kwargs)
            if not data.get("version"):
                data["version"] = getattr(function, "__version__", "1.0")
            if not data.get("module"):
                data["module"] = getattr(function, "__module__", "")
        super().__init__(**data)

    @property
    def node_name(self) -> str:
        """
        Return the name this operation takes as a node of a pipe.

        The operation, not the class: every `PatchOp` is a `PatchOp`, so
        naming nodes after the class would call them all `patch_op`.
        """
        return self.name.replace(_SEPARATOR, "_")

    @cached_property
    def fingerprint(self) -> str:
        """
        Return the digest which identifies this operation and its arguments.

        The same digest `fingerprint_call` gives for the call it stands for,
        because both ask the same function for it.
        """
        return self.fingerprint_at(self.version)

    def fingerprint_at(self, version: str) -> str:
        """
        Return the fingerprint this operation has.

        The version asked for is this *class's*, which every operation
        shares and none is identified by; the one which identifies an
        operation is its own field, and travels in the document with it.
        """
        return _fingerprint(self.name, self.version, self.kwargs)

    def run(self, patch: PatchType) -> Any:
        """Run the operation against a patch."""
        function = resolve_patch_function(self.name, self.module)
        args, kwargs = _as_call(function, self.kwargs)
        # The registered function is the wrapper `patch_function` built,
        # which is the object `Patch.normalize` is and the one a namespace
        # hands out bound to its host. So this is the call a user would have
        # made, and the history it records is the same one.
        return function(patch, *args, **kwargs)

    @classmethod
    def from_call(cls, func, args: tuple = (), kwargs: dict | None = None) -> Task:
        """
        Return the operation a call to a patch function is.

        Parameters
        ----------
        func
            The patch function, as decorated.
        args
            The positional arguments of the call, without the patch.
        kwargs
            The keyword arguments of the call.

        Returns
        -------
        A `PatchOp`, or an instance of the class registered for this name.
        """
        name = op_name(func)
        implementation = _IMPLEMENTATIONS.get(name)
        bound = _bind(func, args, kwargs or {})
        if implementation is not None:
            return implementation(**bound)
        return cls(name=name, kwargs=bound)


def register_implementation(name: str, cls: type[PatchProcessor]) -> None:
    """
    Say that a hand-written class is what a patch function's name means.

    One name has one implementation: after this, `fn.op(...)` returns an
    instance of `cls` rather than a `PatchOp`, so there is a single code
    path whether the operation was reached by name or by class.

    Parameters
    ----------
    name
        The patch function's name, as a patch answers to it.
    cls
        The class which implements it.
    """
    if not issubclass(cls, PatchProcessor):
        msg = (
            f"{cls.__name__} is not a PatchProcessor, so it cannot implement {name!r}."
        )
        raise ParameterError(msg)
    _IMPLEMENTATIONS[name] = cls


def patch_function_tag(func) -> str | None:
    """
    Return the tag which names a patch function in a document.

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

    Nothing a document names is imported to answer this: reading a file
    would otherwise be a way to run whatever it names. The sweep this does
    reach for imports what the *install* declares -- DASCore's own deferred
    modules, and the namespaces plugins register through entry points --
    which is a fixed set, not one a document chooses.

    Parameters
    ----------
    name
        The tag, as `patch_function_tag` spells one.
    module
        Where the function was defined when the document was written, if
        the document says. Used only to say what to import.
    """
    if (found := _REGISTERED.get(name)) is not None:
        return found
    if name in _AMBIGUOUS:
        first, second = _AMBIGUOUS[name]
        msg = (
            f"The patch function {name!r} names two functions, {first} and "
            f"{second}, so which one wrote a document cannot be known."
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
            "import; redefine it here and read again."
        )
    if not package:
        return (
            f"No patch function {leaf!r} is registered in this process, and "
            "DASCore defines none by that name."
        )
    where = f" It was defined in {module} when this was written --" if module else ""
    return (
        f"No patch function {leaf!r} from package {package!r} is registered in "
        f"this process.{where} import that module (or install {package}) and "
        "read again."
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
    # them: a file written by the first would then be read as the second.
    _AMBIGUOUS[tag] = (_spell(existing), _spell(new))
    _REGISTERED.pop(tag, None)
    warnings.warn(f"{msg} Documents naming it can no longer be read.", UserWarning)


def _sweep_patch_functions() -> None:
    """
    Import what the install says defines patch functions.

    `dascore/__init__.py` leaves `dascore.viz` until something asks for it,
    and a plugin's namespace is imported when a patch is first asked for
    one -- and reading a document never asks a patch for anything. Both are
    declared by the install, so importing them is not the arbitrary import
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


def op_name(func) -> str:
    """
    Return the tag which names a patch function, else raise.

    A function which cannot be named has no operation: an operation is
    written down by name, and a name nothing can resolve is not one.
    """
    if (tag := patch_function_tag(func)) is None:
        msg = (
            f"{_spell(func)} cannot be named in a document, so it has no "
            "operation. A patch function defined inside a call is such a "
            "case; define it at the top level of a module instead."
        )
        raise ParameterError(msg)
    return tag


def fingerprint_call(func, args: tuple = (), kwargs: dict | None = None) -> str:
    """
    Return the digest which identifies one call to a patch function.

    The same digest the [`PatchOp`](`dascore.workflow.processor.PatchOp`)
    for that call carries, so a call made as a method and the same call made
    as a task are one operation.

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
    >>> from dascore.workflow import fingerprint_call
    >>>
    >>> called = fingerprint_call(dc.proc.normalize, (), {"dim": "time"})
    >>> assert called == dc.proc.normalize.op(dim="time").fingerprint
    """
    version = getattr(func, "__version__", "1.0")
    name = _call_name(func)
    bound = _without_patches(_bind(func, args, kwargs or {}))
    # Answered from the cache when the same call has been made before,
    # which in a loop over a spool is every call after the first. Hashing
    # the bound arguments costs a few microseconds; the digest of their
    # canonical JSON costs several times that.
    try:
        key = (name, version, _as_key(bound))
    except TypeError:
        # Something unhashable -- an array argument, most often. Its
        # digest is the honest cost of saying which array it was.
        return _fingerprint(name, version, bound)
    if (found := _FINGERPRINTS.get(key)) is None:
        found = _fingerprint(name, version, bound)
        # Bounded, and simply stops growing rather than evicting: the
        # entries are one small string each, and a process which has made
        # four thousand distinct calls is not one this is hot for.
        if len(_FINGERPRINTS) < _FINGERPRINT_LIMIT:
            _FINGERPRINTS[key] = found
    return found


# The only leaves a cache key may be built from. The rule is not "hashable":
# it is "two of these are the same argument exactly when Python says they are
# equal". A pint quantity fails that -- `1 * m == 100 * cm` and the two hash
# alike, while the serializer encodes them differently -- so caching on it
# would give one call two answers depending on what ran first.
_KEYABLE = (str, bytes, int, float, bool, type(None))

# Beyond this many elements, working the key out costs more than the digest
# it saves.
_KEY_LIMIT = 32

# Stands for a patch handed to an operation as an argument.
_PATCH_ARGUMENT = "$patch"


def _without_patches(kwargs: dict) -> dict:
    """
    Return the bound arguments with any patch replaced by a marker.

    A patch given as an argument is not a *parameter* of the operation, it
    is another input to it: `where(cond_patch)` is the same operation
    whichever patch it was handed, and which one it was is said by the ids
    folded from the operands. Encoding it here would also hash a whole
    patch on every call, and warn that it has no encoding of its own.
    """
    # Imported here rather than at module scope: this module is imported
    # while `dascore.utils.patch` is still being imported.
    import dascore as dc  # noqa: PLC0415

    if not any(isinstance(x, dc.Patch) for x in kwargs.values()):
        return kwargs
    return {
        key: _PATCH_ARGUMENT if isinstance(value, dc.Patch) else value
        for key, value in kwargs.items()
    }


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
        # mappings to Python and different calls to the serializer.
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
    return (type(value), value)


def _call_name(func) -> str:
    """
    Return the name a call is fingerprinted under.

    The registry tag when the function has one. When it does not -- a patch
    function defined inside another call -- something was still done to the
    patch, and a `processing_id` which did not move would claim it was not.
    So the call is named by where it was written instead: enough to tell it
    from another operation, and honestly not resolvable, which is why
    `op_name` still refuses it and no `PatchOp` can be built.
    """
    if (tag := patch_function_tag(func)) is not None:
        return tag
    return _spell(func)


def _fingerprint(name: str, version: str, kwargs: dict) -> str:
    """
    Return the digest a name, a version and bound arguments make.

    The serializer's warning about a value it has no encoding for is
    suppressed. It is worth hearing when a *task* is being written to a
    document, which is what it was written for; here it would fire on
    every ordinary call carrying, say, a numpy dtype, and hashing such a
    value by its type is all an id needs of it.
    """
    # Spelled the way `Task.fingerprint_at` spells one, so a pipe holding a
    # PatchOp hashes the same whichever built it.
    with suppress_warnings(
        DASCoreWarning, message="A value of type .* has no encoding"
    ):
        return digest(
            {"task": "dascore:PatchOp", "version": version, "params": {name: kwargs}}
        )


@cache
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


def _canonical(value):
    """
    Return a value in the one shape a document can give back.

    A document holds a sequence as a list, whichever it was written from,
    so a call which is not canonicalized here reads back as a different
    call: its history string says `[10, 100]` where it said `(10, 100)`,
    and `make_broadcastable_to`, whose shape ends up in a set, stops
    working entirely.
    """
    if isinstance(value, (list, tuple)):
        return tuple(_canonical(x) for x in value)
    return value


def _check(func, args: tuple, kwargs: dict) -> None:
    """Refuse a call the function's signature does not accept."""
    try:
        _signature(func).bind(_PATCH, *args, **kwargs)
    except TypeError as error:
        msg = f"{op_name(func)} cannot be called that way: {error}"
        raise ParameterError(msg) from error


def _bind(func, args: tuple, kwargs: dict) -> dict:
    """
    Return the arguments of a call as the one mapping they mean.

    Positional and keyword spellings of one call bind alike, defaults are
    filled in, the patch is dropped, and a `**kwargs` group is spread back
    out so that a dimension given as an extra reads as itself.
    """
    signature = _signature(func)
    _check(func, args, kwargs)
    bound = signature.bind(_PATCH, *args, **kwargs)
    bound.apply_defaults()
    # `_resolve_default` rather than a rule of our own: a default spelled
    # `x=Field(default=3)` means 3, and task.py already says so.
    out = {key: _resolve_default(value) for key, value in bound.arguments.items()}
    parameters = list(signature.parameters.values())
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
        if collisions := set(extras) & set(out):
            msg = (
                f"{op_name(func)} was given {sorted(collisions)} both as a "
                "parameter and as an extra, so the call cannot be written "
                "down as one mapping."
            )
            raise ParameterError(msg)
        out |= extras
    # Canonicalized last, so that the names a `**kwargs` group carried are
    # canonicalized too rather than hiding inside the mapping.
    return {key: _canonical(value) for key, value in out.items()}


def _as_call(func, kwargs: dict) -> tuple[tuple, dict]:
    """
    Return bound arguments as the call which can be made from them.

    A `*args` group cannot be passed by name, and neither can anything in
    front of one, so those go back to being positional. The result is bound
    again, so a mapping which is not a call this function accepts is
    refused here rather than part-way through running it.
    """
    signature = _signature(func)
    parameters = list(signature.parameters.values())[1:]
    packs = any(x.kind == inspect.Parameter.VAR_POSITIONAL for x in parameters)
    args, rest = [], dict(kwargs)
    for parameter in parameters:
        positional = parameter.kind == inspect.Parameter.POSITIONAL_ONLY or (
            packs and parameter.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        )
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            args.extend(rest.pop(parameter.name, ()))
        elif positional and parameter.name in rest:
            args.append(rest.pop(parameter.name))
        elif positional:
            # Absent, so everything after it would land in the wrong slot.
            # `_check` says which name is missing.
            break
    _check(func, tuple(args), rest)
    return tuple(args), rest
