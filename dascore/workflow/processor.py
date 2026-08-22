"""
A patch operation, as a task.

Any function decorated with `dascore.patch_function` is one operation, and
[`PatchOp`](`dascore.workflow.processor.PatchOp`) is that operation said as
an object: the function's name and the arguments it was given, which can be
compared, fingerprinted, and written to a file.

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

import functools
import inspect
import re
import warnings
from collections.abc import Mapping
from contextlib import suppress
from functools import cached_property, lru_cache
from typing import Any, ClassVar

from pydantic import Field, PrivateAttr

from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from dascore.workflow.checks import attr_type, check_patch_attrs, check_patch_coords
from dascore.workflow.meta import PatchMeta
from dascore.workflow.serialize import digest
from dascore.workflow.task import (
    _VERSION_KEY,
    Task,
    _resolve_default,
    _take_ownership,
)

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

    # Everything `patch_function` is given, so that a processor is a whole
    # description of its operation rather than half of one. Something
    # reading a chain of these -- to fuse them, to compile them -- has the
    # decorator nowhere in reach, and needs all of it.
    #
    # What the patch must hold. Stated on the class rather than passed to
    # `run`, because it is a property of the operation and not of the call.
    required_dims: ClassVar[tuple[str, ...] | str | None] = None
    required_coords: ClassVar[tuple[str, ...] | str | None] = None
    required_attrs: ClassVar[attr_type] = None
    # What the result is, when the operation changes it; None leaves it as
    # it was, and "" clears it.
    data_type: ClassVar[str | None] = None
    # How the call is written into the patch's history, or None for an
    # operation which records nothing. `transpose` is the live case.
    history: ClassVar[str | None] = "full"
    # Whether the function's arguments are checked by pydantic on the way
    # in. A processor validates its own fields, so this is here to be
    # reconciled with the decorator rather than acted on.
    validate_call: ClassVar[bool] = False

    # The registry tag this class implements, set by
    # `register_implementation`. Empty until it is registered.
    _patch_function: ClassVar[str] = ""
    # Kernels registered for a particular array backend, by
    # `register_kernel`. Looked up in `__dict__` per class, never
    # inherited wholesale, so a subclass does not silently answer for its
    # parent's backends.
    _kernels: ClassVar[dict[str, Any]] = {}

    def check(self, patch: PatchType) -> PatchType:
        """
        Refuse a patch which does not carry what the operation needs.

        The framework does not call this, and deliberately: a registered
        processor is reached through its patch function, whose decorator
        has already run the same checks from its own declaration of them.
        Running them again from the class's declaration would mean two
        sources of truth which can disagree in silence.
        `register_implementation` reconciles the two instead, at import.

        It is still here for a hand-written `run` which does not go
        through a patch function, which has nothing else to call.

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

    # --- the surface a PatchOp has -----------------------------------
    #
    # A registered processor stands where a `PatchOp` would, so it answers
    # the same questions. Every contract parametrised over all the patch
    # functions -- the fingerprint, the document, the pickle -- then keeps
    # passing without knowing which of the two it got.

    @property
    def name(self) -> str:
        """Return the registry tag of the operation this implements."""
        return self._patch_function

    @property
    def kwargs(self) -> dict:
        """Return the arguments the operation was given."""
        return self._params()

    # The function's version, read once when the operation is built. A
    # `PatchOp` keeps it as a field for the same reason: an operation is
    # what it was when it was written down, so a later bump of the
    # function must not reach back and change what an existing one
    # fingerprints as.
    _captured_version: str = PrivateAttr(default="")

    def model_post_init(self, context) -> None:
        """Record the version the function was at when this was built."""
        function = _REGISTERED.get(self._patch_function)
        version = getattr(function, "__version__", self.__version__)
        object.__setattr__(self, "_captured_version", version)

    @property
    def version(self) -> str:
        """
        Return the version the operation is declared at.

        The patch function's, not this class's. Registration pins the two
        equal, so they can only part when the function is bumped -- and a
        bump has to reach the fingerprint, which is the whole reason a
        version exists.
        """
        return self._captured_version or self.__version__

    def to_dict(self) -> dict:
        """
        Return a document which describes this operation.

        The version written down is the function's, so the document
        fingerprints as it did when it was written even after the
        function has moved on.
        """
        return super().to_dict() | {_VERSION_KEY: self.version}

    @property
    def node_name(self) -> str:
        """Return the name this operation goes by where a task is labelled."""
        return self.name.replace(_SEPARATOR, "_")

    def fingerprint_at(self, version: str) -> str:
        """
        Return the digest which identifies this operation and its arguments.

        Spelled as the `PatchOp` for the same call would spell it, not as
        this class. Registering a hand-written implementation is meant to
        be invisible from outside: were the class name to reach the
        digest, every `processing_id` ever recorded for the operation
        would stop matching, and every stored document with it.
        """
        return _fingerprint(self.name, self.version, self.kwargs)

    def run(self, patch: PatchType) -> Any:
        """
        Run the operation against a patch.

        Through the patch function, not straight into `_apply`: the
        function's decorator is what writes the history and stamps the
        ids, and a processor reached by `.op(...)` has to come out the
        same as one reached by calling the method.
        """
        function = resolve_patch_function(self.name)
        args, kwargs = _as_call(function, self.kwargs)
        return function(patch, *args, **kwargs)

    # --- the seam ----------------------------------------------------

    def derive_meta(self, meta: PatchMeta) -> PatchMeta:
        """
        Return what the result's metadata is.

        Never sees an array, which is what lets something fuse a chain of
        operations without holding any data. The default says the
        operation changes nothing: right for anything elementwise.
        """
        return meta

    def plan_kernel(self, meta: PatchMeta, out_meta: PatchMeta):
        """
        Return the array function this call is, or None to touch no data.

        Both metadata objects are given because a kernel often needs the
        difference between them -- `transpose` wants the permutation which
        takes the old dimension order to the new.

        A kernel registered for the data's backend wins, being someone
        saying they took this operation on there. Failing that, the
        class's own `kernel`, written to the array API standard and so
        able to run on any backend -- unless `fusible` says these
        arguments are outside what the standard promises, in which case
        the class's `numpy_kernel` answers for them instead. A class with
        none of the three is a metadata-only operation and gets None.

        Which kernel runs is settled here rather than inside a kernel so
        that something reading a chain of operations can see what each
        one got without running any of them.
        """
        fallback = not self.fusible
        if (found := _resolve_kernel(type(self), meta.backend, fallback)) is None:
            return None
        return functools.partial(found, self, meta=meta, out_meta=out_meta)

    @property
    def fusible(self) -> bool:
        """
        Whether this operation can be lowered with the ones around it.

        Fusing a chain means compiling the kernels into one pass over the
        data, so it can only include kernels written in the backend's own
        terms. A kernel which reaches for numpy cannot be lowered, and
        neither can an operation which has to see the data and the
        metadata at once -- which is what defining `reconcile` says.

        Answered from the operation's own parameters, never from the
        data: something deciding what to fuse has the chain and no
        arrays, so an answer it has to run the operation to get is no
        answer at all.

        The default cannot see inside a kernel. It reads `reconcile`
        alone and takes the class's own kernel to be portable, so a
        processor whose kernel reaches for numpy -- `Demedian` -- and one
        whose kernel is portable for only some of its arguments --
        `Full`, `FillNa` -- both have to say so by overriding this.
        """
        return type(self).reconcile is PatchProcessor.reconcile

    def reconcile(self, data, meta: PatchMeta) -> PatchMeta:
        """
        Return the metadata the data actually turned out to have.

        Defining this says the operation cannot be fused: it is the one
        step which has to see both halves at once. The default only
        carries the data's dtype back, since a kernel may promote.
        """
        dtype = getattr(data, "dtype", meta.dtype)
        return meta if dtype == meta.dtype else meta.update(dtype=dtype)

    @classmethod
    def _call(cls, patch: PatchType, /, **kwargs) -> PatchType:
        """
        Build the operation from a patch function's arguments and run it.

        This is what a patch function's body calls. Built here rather
        than by the body so that the arrays among the arguments are not
        taken over: a task freezes what it is handed so its fingerprint
        cannot come to describe values it no longer holds, but an
        operation built inside a patch function is run once and thrown
        away, and freezing would reach back and lock the caller's own
        array for the rest of its life.
        """
        token = _take_ownership.set(False)
        try:
            operation = cls(**kwargs)
        finally:
            _take_ownership.reset(token)
        return operation._apply(patch)

    def _apply(self, patch: PatchType) -> PatchType:
        """
        Run the operation, metadata first and then the data.

        This is what a patch function's body calls. It does none of the
        ceremony around an operation -- the checks, the history, the
        lineage ids -- because the patch function's decorator is still
        wrapped around this call and is already doing all of it. Doing it
        here as well would count every operation twice.
        """
        meta = PatchMeta.from_patch(patch)
        out_meta = self.derive_meta(meta)
        kernel = self.plan_kernel(meta, out_meta)
        data = patch.data if kernel is None else kernel(patch.data)
        # An operation which changed neither half did nothing, and hands
        # back the patch it was given rather than an equal one. The
        # decorator reads that as nothing having happened, so no history
        # is written and no id advances -- which is what `conj` on real
        # data and a transpose into the order already held both mean.
        #
        # A kernel says "nothing to do" by handing its argument back, so
        # a kernel must not write into that argument and return it: the
        # result would be a change nothing records. Patch data is marked
        # read-only where the backend allows it, but not every array-like
        # can promise that, so this is a contract rather than a guard --
        # checking it would mean hashing the data on every operation.
        if data is patch.data and out_meta is meta:
            return patch
        return self.reconcile(data, out_meta).to_patch(data)


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
        Return the name this operation goes by where a task is labelled.

        The operation, not the class: every `PatchOp` is a `PatchOp`, so a
        label taken from the class would call them all `patch_op`.
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


def register_kernel(cls: type[PatchProcessor], backend: str):
    """
    Say that a function is how an operation runs on one array backend.

    Used as a decorator. The kernel takes the processor, the data, and
    both metadata objects, and returns an array.

    Parameters
    ----------
    cls
        The processor the kernel belongs to.
    backend
        The backend it is for, as
        [`backend_name`](`dascore.utils.array_api.backend_name`) spells
        it -- "numpy", "cupy", "dask".
    """

    def decorate(func):
        """Record the kernel against the class and hand it back."""
        # Written into this class's own dict, not a ClassVar it shares
        # with its parent, so registering for a subclass cannot answer
        # for the class it derives from.
        cls._kernels = {**cls.__dict__.get("_kernels", {}), backend: func}
        return func

    return decorate


def _resolve_kernel(cls: type[PatchProcessor], backend: str, fallback: bool = False):
    """
    Return the kernel a class runs for one backend, or None if it has none.

    A kernel registered for the backend wins; failing that the class's own
    `kernel`, which is written to the array API standard and so runs on
    any of them. A class which defines neither is metadata-only.

    `fallback` says the arguments are outside what the standard promises,
    so the class's `numpy_kernel` stands in for the generic one. A
    registered kernel still wins over it: whoever registered it took this
    backend on and gets to say what it does with these arguments.
    """
    # One class at a time, every question asked of it before moving up:
    # a subclass which wrote its own `kernel` means it, and a backend
    # kernel registered against its parent must not answer for it.
    for klass in cls.__mro__:
        contents = klass.__dict__
        if (found := contents.get("_kernels", {}).get(backend)) is not None:
            return found
        if fallback and (numpy_kernel := contents.get("numpy_kernel")) is not None:
            return numpy_kernel
        if (generic := contents.get("kernel")) is not None:
            return generic
    return None


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
    function = resolve_patch_function(name)
    _reconcile(name, cls, function)
    _check_fields(name, cls, function)
    cls._patch_function = name
    _IMPLEMENTATIONS[name] = cls


def _reconcile(name: str, cls: type[PatchProcessor], function) -> None:
    """
    Make the class and the decorator say the same thing, or refuse both.

    A processor states what its operation requires so that something
    reading a chain of them has the whole story without the decorator in
    reach. That leaves two places saying it, and two places which say it
    differently are worse than one -- so a class which states a
    requirement must state the one the decorator did, and a class which
    states none takes the decorator's.
    """
    declared = getattr(function, "_declared", {})
    for field, value in declared.items():
        # Asked of the class's own dict, not of `getattr`: a class which
        # states the base default on purpose has still stated it, and
        # silently overwriting that would make the check a formality.
        declares = any(
            field in klass.__dict__
            for klass in cls.__mro__[:-1]
            if klass is not PatchProcessor
        )
        stated = getattr(cls, field, None)
        if not declares:
            setattr(cls, field, value)
            continue
        if stated != value:
            msg = (
                f"{cls.__name__} says {field}={stated!r} and {name!r} says "
                f"{value!r}. A processor and its patch function have to "
                "agree about what the operation requires."
            )
            raise ParameterError(msg)
    if (version := getattr(function, "__version__", None)) != cls.__version__:
        msg = (
            f"{cls.__name__} is version {cls.__version__!r} and {name!r} is "
            f"{version!r}. They fingerprint as one operation, so one version."
        )
        raise ParameterError(msg)


def _check_fields(name: str, cls: type[PatchProcessor], function) -> None:
    """
    Refuse a class which could not be built from a call to its function.

    Caught here rather than where someone calls the patch function: the
    class is built with the call's bound arguments, so a mismatch is a
    pydantic complaint about a name the caller never typed, arriving at
    the wrong moment and pointing at the wrong thing.
    """
    parameters = list(_signature(function).parameters.values())[1:]
    fields = set(cls.model_fields)
    takes_extras = cls.model_config.get("extra") == "allow"
    for parameter in parameters:
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            if not takes_extras:
                msg = (
                    f"{name!r} takes **{parameter.name}, so {cls.__name__} has "
                    'to accept them: set model_config extra="allow".'
                )
                raise ParameterError(msg)
            continue
        if parameter.name not in fields:
            msg = (
                f"{name!r} takes {parameter.name!r} and {cls.__name__} has no "
                "such field, so a call could not be written down as one."
            )
            raise ParameterError(msg)


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
        # The function itself is in the key, not just its name. For one
        # which has no tag the name ends in `id(func)`, and CPython reuses
        # an address once the function is collected -- so a factory making
        # one patch function per call could hand a later one the earlier
        # one's fingerprint. Holding the function here makes the key exact
        # and keeps the address from being reused underneath it.
        key = (func, name, version, _as_key(bound))
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


class _PatchArgument:
    """
    Stands for a patch handed to an operation as an argument.

    A class rather than a string: a caller is free to pass the string
    `"$patch"`, and a marker it could be mistaken for would make that call
    and a call with an actual patch one operation.
    """

    __slots__ = ()

    def __repr__(self):
        """Say what it is, which is what the digest records."""
        return "<patch argument>"


_PATCH_ARGUMENT = _PatchArgument()


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
    patch, and a `processing_id` which did not move would claim it was not.
    So the call is named by where it was written instead: enough to tell it
    from another operation, and honestly not resolvable, which is why
    `op_name` still refuses it and no `PatchOp` can be built.
    """
    if (tag := patch_function_tag(func)) is not None:
        return tag
    # Where it was written, and *which* one: a factory making patch
    # functions gives every one of them the same module and qualname, and
    # two closures over different values are two operations. The identity
    # is process-local, which is honest -- so is the function.
    return f"{_spell(func)}#{id(func):x}"


def _fingerprint(name: str, version: str, kwargs: dict) -> str:
    """
    Return the digest a name, a version and bound arguments make.

    The serializer's warning about a value it has no encoding for is
    suppressed. It is worth hearing when a *task* is being written to a
    document, which is what it was written for; here it would fire on
    every ordinary call carrying, say, a numpy dtype, and hashing such a
    value by its type is all an id needs of it.
    """
    # Spelled the way `Task.fingerprint_at` spells one, so a PatchOp hashes
    # the same whichever route built it.
    with suppress_warnings(
        DASCoreWarning, message="A value of type .* has no encoding"
    ):
        return digest(
            {"task": "dascore:PatchOp", "version": version, "params": {name: kwargs}}
        )


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
