"""Patch operations written as classes.

A [`PatchProcessor`](`dascore.core.processor.PatchProcessor`) subclass is a
whole operation: its fields are the parameters, `get_metadata` works out the
result's metadata and the numbers the kernel needs without touching data, and
`kernel` computes the array. Subclassing registers the operation, so the
class is written once and the patch method is a two-line body which builds
it and runs it.

One of DASCore's own operations which writes no kernel changes metadata
alone, so its method is written in [`PatchMeta`](`dascore.PatchMeta`) and
runs on a patch's metadata as readily as on the patch; one with a kernel is
written in [`Patch`](`dascore.Patch`). A class defined outside DASCore is a
method of neither -- the data-less contract is one DASCore holds itself to,
and nothing outside it can write into `Patch` -- so it declares its own
staticmethod, which is what the registry holds.

Examples
--------
>>> import dascore as dc
>>>
>>> class ScaleExample(dc.PatchProcessor):
...     '''Multiply the data by a factor.'''
...
...     name = None  # so this example claims no name of its own
...     factor: float = 2.0
...
...     def kernel(self, data):
...         return data * self.factor
>>>
>>> patch = dc.get_example_patch()
>>> out = ScaleExample(3)(patch)
>>> assert ScaleExample(3).operation_id == ScaleExample(3.0).operation_id
"""

from __future__ import annotations

import ast
import functools
import inspect
import numbers
import sys
import textwrap
from contextvars import ContextVar
from types import FunctionType
from typing import TYPE_CHECKING, Any, ClassVar, Self, overload

import numpy as np
from pydantic import ConfigDict
from pydantic.alias_generators import to_snake

import dascore as dc
from dascore.config import get_config
from dascore.constants import PatchMetaType, PatchType
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel, model_values
from dascore.utils.attrs import _values_equal
from dascore.utils.identity import (
    callable_name,
    extract_patches,
    ids_enabled,
    narrowed_data_id,
    result_ids,
    stamp,
    warn_random_id,
)
from dascore.utils.patch import (
    _call_str,
    _maybe_add_history_str,
    attr_type,
    check_patch_attrs,
    check_patch_coords,
    check_patch_data,
)
from dascore.utils.patch_registry import (
    _memoized_operation_id,
    _spell,
    is_default,
    patch_function_tag,
    register_patch_function,
)

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs

# Whether the next operation to run writes its call into history and ids.
# A bypass (`patch_function.raw_function`) clears it for the one call it
# wraps; `run` spends it, so nothing further down is affected.
_RECORD: ContextVar[bool] = ContextVar("dascore_record_call", default=True)


class PatchProcessor(DascoreBaseModel):
    """
    An operation on a patch, written as a class.

    Subclasses declare their parameters as fields and override some of:

    - `get_metadata(meta)`: the result's metadata and the arguments `kernel`
      needs, worked out together from metadata alone. Deriving the one
      usually computes the other.
    - `kernel(data, **plan)`: the array computation. None of it may see a
      patch, so a chain of kernels can be compiled.
    - `reconcile(data, out)`: the one hook which sees both halves, for what
      only the computed data can say.

    Each subclass is registered under `name` (snake case of the class name
    unless set). One of DASCore's own is reached through a method of that
    name, written in the body of `Patch` or, for an operation with no
    kernel, `PatchMeta`:

        def scale(self, factor: float = 2.0) -> Self:
            '''Multiply the data by a factor.'''
            return Scale(factor=factor).run(self)

    A real method is one a type checker, an IDE and `help` can all read,
    which a synthesized function is not, and `-> Self` is what `run`
    promises: metadata comes back metadata and a subclass comes back a
    subclass. Every field must appear among its parameters; the method may
    take more, since a body may resolve something before building the
    instance. `cls.patch_function` is that method, and carries the class's
    docstring: the parameters, notes and examples are written once, with
    the class. A class with `name = None` has no method and is registered
    nowhere, and a class outside DASCore declares a staticmethod of that
    name itself, taking the patch positionally.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump when the same fields mean a different result.
    __version__: ClassVar[str] = "1.0"
    # The registry name, and the patch method's; None for neither.
    name: ClassVar[str | None] = None
    # What the patch must hold.
    required_dims: ClassVar[tuple[str, ...] | str | None] = None
    required_coords: ClassVar[tuple[str, ...] | str | None] = None
    required_attrs: ClassVar[attr_type] = None
    # The result's data_type: None keeps the input's, "" clears it.
    data_type: ClassVar[str | None] = None
    # How a call is written into history: "full", "method_name" or None.
    history: ClassVar[str | None] = "full"
    # The fields a call may give positionally, in order; None for all.
    _positional_fields: ClassVar[tuple[str, ...] | None] = None
    # Kernels registered per backend by `register_kernel`, looked up in each
    # class's own `__dict__` so a subclass never answers with its parent's.
    _kernels: ClassVar[dict[str, Any]] = {}
    # The method which runs this operation, found at class creation:
    # written in `Patch` or `PatchMeta` for one of DASCore's own, and
    # declared on the class itself for anything out of tree.
    patch_function: ClassVar[Any] = None

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Check the subclass, and find the method which runs it."""
        super().__pydantic_init_subclass__(**kwargs)
        if stale := {"derive", "plan"} & set(cls.__dict__):
            msg = (
                f"{cls.__name__} defines {', '.join(sorted(stale))}, which "
                "PatchProcessor no longer calls. The two are now one method, "
                "`get_metadata(meta)`, returning the result's metadata and "
                "the kernel's arguments together."
            )
            raise ParameterError(msg)
        if clashes := set(cls.model_fields) & _RESERVED:
            msg = (
                f"{cls.__name__} has fields {sorted(clashes)}, which would "
                "shadow names PatchProcessor itself uses; rename them."
            )
            raise ParameterError(msg)
        if "name" not in cls.__dict__:
            cls.name = to_snake(cls.__name__)
        if cls.name is not None and not cls.name.isidentifier():
            msg = f"{cls.__name__}.name must be a python identifier; got {cls.name!r}."
            raise ParameterError(msg)
        cls.patch_function = None
        if cls.name is not None:
            _check_patch_listing(cls)

    @classmethod
    def _positional_names(cls) -> tuple[str, ...]:
        """Return the fields a call may give positionally, in order."""
        names = tuple(cls.model_fields)
        if cls._positional_fields is None:
            return names
        return tuple(x for x in names if x in cls._positional_fields)

    def __init__(self, /, *args, **kwargs):
        """Take the fields positionally, in the order they are declared."""
        if args:
            names = type(self)._positional_names()
            if len(args) > len(names):
                msg = (
                    f"{type(self).__name__} takes {len(names)} positional "
                    f"argument(s); got {len(args)}."
                )
                raise TypeError(msg)
            given = dict(zip(names, args, strict=False))
            if repeated := sorted(set(given) & set(kwargs)):
                msg = f"{type(self).__name__} got repeated argument(s): {repeated}."
                raise TypeError(msg)
            kwargs = {**given, **kwargs}
        super().__init__(**kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return the validated fields, extras included."""
        return model_values(self)

    @property
    def tag(self) -> str:
        """Return the name this operation is identified by."""
        cls = type(self)
        func = cls.patch_function
        if func is not None and (tag := patch_function_tag(func)) is not None:
            return tag
        # Unregistered or defined inside a call: named by its source, or --
        # one with no source to read -- by which class object it is.
        try:
            return callable_name(cls)
        except ParameterError:
            return f"{_spell(cls)}#{id(cls):x}"

    def _inputs(self) -> tuple[dict, list]:
        """Return the non-default fields, and the patches found among them."""
        fields = type(self).model_fields
        # A field which only restates its default is left out, so a field
        # added later does not change the id of every operation before it.
        given = {
            name: value
            for name, value in self.kwargs.items()
            if name not in fields
            or fields[name].is_required()
            or not is_default(value, fields[name])
        }
        return extract_patches(given)

    def _operation(self) -> tuple[str, list]:
        """Return this operation's id, and the patches among its fields."""
        params, patches = self._inputs()
        found = _memoized_operation_id(type(self), self.tag, params, self.__version__)
        return found, patches

    @property
    def operation_id(self) -> str:
        """Return the id of the tag, version and non-default fields."""
        return self._operation()[0]

    def _identity(self) -> tuple[str, str]:
        """Return the id this operation has as another's parameter."""
        found, patches = self._operation()
        if patches:
            # Its id does not say which patches it holds, so an operation
            # given it could not tell two of them apart.
            msg = f"{type(self).__name__} holds a patch, so it has no id of its own."
            raise ParameterError(msg)
        return "operation", found

    def __eq__(self, other) -> bool:
        """Two processors are equal if they are the same operation."""
        if not isinstance(other, PatchProcessor):
            return NotImplemented
        if type(self) is not type(other):
            return False
        try:
            return self.operation_id == other.operation_id
        except Exception:
            # A field with no faithful spelling: only itself is surely equal.
            return self is other

    def __hash__(self) -> int:
        """Hash a processor the way it compares."""
        try:
            return hash(self.operation_id)
        except Exception:
            return object.__hash__(self)

    @overload
    def __call__(self, patch: PatchType) -> PatchType: ...

    @overload
    def __call__(self, patch: PatchMetaType) -> PatchMetaType: ...

    def __call__(self, patch):
        """Run the operation; see `run`."""
        return self.run(patch)

    @classmethod
    def kernel_for(cls, backend: str):
        """
        Return the kernel this class runs for a backend, or None.

        Each class in the MRO is asked for a kernel registered for the
        backend, then for its own `kernel`, before moving up: a subclass
        which wrote its own kernel means it, and a backend kernel
        registered against its parent must not answer for it.
        """
        for klass in cls.__mro__:
            contents = klass.__dict__
            if (found := contents.get("_kernels", {}).get(backend)) is not None:
                return found
            if (generic := contents.get("kernel")) is not None:
                return generic
        return None

    def check(self, patch: PatchMetaType) -> PatchMetaType:
        """
        Refuse a patch which does not carry what the operation needs.

        Raises
        ------
        PatchDataError
            If the operation computes data and the patch holds none.
        PatchCoordinateError
            If a required dimension or coordinate is missing.
        PatchAttributeError
            If a required attr is missing, or holds a different value.
        """
        # Only an operation which computes data needs any; one without a
        # kernel is metadata all the way through, so it runs on a PatchMeta.
        # Asked of the class, not of this patch's backend: an operation with
        # a kernel for some other backend still means to compute.
        if _has_kernel(type(self)):
            check_patch_data(patch)
        check_patch_coords(patch, dims=self.required_dims, coords=self.required_coords)
        return check_patch_attrs(patch, self.required_attrs)

    def get_metadata(self, meta: dc.PatchMeta) -> tuple[dc.PatchMeta, dict[str, Any]]:
        """
        Return the result's metadata and the arguments `kernel` needs.

        Given a [`PatchMeta`](`dascore.PatchMeta`), so it cannot read
        `.data`. Work through the coord manager and `meta.new`. Returning
        the argument itself says the metadata did not change, which is how
        `run` spots a no-op; an empty dict says the kernel takes nothing
        beyond the data, which most kernels do.
        """
        return meta, {}

    def reconcile(self, data, out: dc.PatchMeta) -> dc.PatchMeta:
        """Return the result's metadata once the data are known; default as is."""
        return out

    @overload
    def run(self, patch: PatchType) -> PatchType: ...

    @overload
    def run(self, patch: PatchMetaType) -> PatchMetaType: ...

    def run(self, patch):
        """
        Run the operation: check, get_metadata, kernel, reconcile, record.

        A patch comes back a patch and metadata comes back metadata, which
        is what the two overloads say. An operation which changes neither
        the metadata nor the data hands back what it was given, and records
        nothing.
        """
        record = _RECORD.get()
        if not record:
            # Spent on this one call, which is the one a bypass wrapped: an
            # operation running another inside itself records that one.
            _RECORD.set(True)
        return self._run(patch, record=record)

    def _run(self, patch: dc.PatchMeta, record: bool) -> dc.PatchMeta:
        """Run the operation; `record=False` writes no history or ids."""
        self.check(patch)
        meta = patch.drop_data() if isinstance(patch, dc.Patch) else patch
        out, plan = self.get_metadata(meta)
        plan = _checked_plan(self, plan)
        # Resolved from the metadata, so an operation with no kernel is
        # settled before anything asks the patch for its data.
        kernel = self.kernel_for(meta.backend)
        if kernel is None:
            if out is meta:
                return self._unchanged(patch, record)
            # The hook which sees both halves runs whether or not a kernel
            # computed anything: the data are the ones which came in, and
            # metadata has none to show.
            unchanged = patch._data if isinstance(patch, dc.Patch) else None
            out = self.reconcile(unchanged, out)
            attrs = out.attrs if not record else self._record(patch, out)
            # The data are whatever they were: a patch puts its own back,
            # and metadata has none to put.
            return patch._reattach(out, attrs)
        # `check` refused metadata for an operation with a kernel, so what
        # is left here holds data; the assert is what says so statically.
        assert isinstance(patch, dc.Patch)
        data = patch.data
        # A kernel says "nothing to do" by handing its argument back, so it
        # must not write into that argument and return it.
        result = kernel(self, data, **plan)
        if out is meta and result is data:
            return self._unchanged(patch, record)
        out = self.reconcile(result, out)
        if record:
            out = out.update(attrs=self._record(patch, out))
        new = out.to_patch(result)
        # Only an operation which set a source of its own says it loads the result.
        if out._source is not meta._source:
            new._source = out._source
        return new

    def _unchanged(self, patch: dc.PatchMeta, record: bool) -> dc.PatchMeta:
        """Return the patch an operation did nothing to; nothing is recorded."""
        # A declared data_type still applies, as it does for a decorated
        # patch function.
        if self.data_type is None or not record:
            return patch
        return patch.update_attrs(data_type=self.data_type)

    def _record(self, patch: dc.PatchMeta, out: dc.PatchMeta) -> PatchAttrs:
        """Return attrs carrying the data_type, history and ids of this call."""
        attrs = out.attrs
        if self.data_type is not None:
            attrs = attrs.update(data_type=self.data_type)
        name = self.name or type(self).__name__
        if self.history is not None and get_config().patch_history != "disabled":
            spelled = _call_str(name, self.kwargs) if self.history == "full" else name
            attrs = _maybe_add_history_str(attrs, spelled)
        if not ids_enabled():
            return stamp(attrs, (), None)
        try:
            operation, others = self._operation()
        except Exception as error:
            # As for a patch function: a field the encoder refuses still
            # made new data, so the result gets a random id.
            warn_random_id(type(self).__name__, error)
            # The patches it holds still say where the data came from.
            operation, others = None, self._inputs()[1]
        members = [patch.attrs, *(x.attrs for x in others)]
        # Only a source the operation set itself says the result is still
        # something that source loads; such a narrowing names a window of
        # the same array rather than something derived from it.
        after = out._source if out._source is not patch._source else None
        window = narrowed_data_id(patch.attrs, patch._source, after)
        return attrs.update(**result_ids(members, operation, data_id=window))


# Names a subclass field may not take: the base's own settings and methods.
_RESERVED = frozenset(x for x in vars(PatchProcessor) if not x.startswith("_")) | {
    "name",
    "data_type",
    "history",
    "patch_function",
}


# Classes created before `dascore.core.patch` has finished importing, which
# is every one of DASCore's own; they are checked once both classes exist to
# be checked against.
_UNCHECKED: list[type[PatchProcessor]] = []
# `[Patch, PatchMeta]`, once there are such classes to check against.
_HOSTS: list[type] = []


def _has_kernel(cls: type[PatchProcessor]) -> bool:
    """Whether anything in the MRO computes data; see `kernel_for`."""
    return any(
        x.__dict__.get("kernel") is not None or x.__dict__.get("_kernels")
        for x in cls.__mro__
    )


def _is_dascores(cls) -> bool:
    """Whether a class is DASCore's own rather than a plugin's."""
    # Spelled as the package a module belongs to, the rule
    # `patch_function_tag` uses to tell DASCore's own names apart.
    return cls.__module__.split(".", 1)[0] == "dascore"


def _meta_hosted(cls) -> bool:
    """Whether `PatchMeta` is the class whose body writes the method."""
    # An operation with no kernel changes metadata alone, so every PatchMeta
    # can run it and Patch inherits it. One with a kernel needs data, so its
    # method belongs to Patch.
    # A trust boundary, not a namespace detail: the data-less contract is
    # one DASCore holds itself to, so a plugin's is never PatchMeta's.
    return _is_dascores(cls) and not _has_kernel(cls)


def _home(host: type) -> str:
    """Return the file a class is written in, for a message to point at."""
    leaf = "patch_meta" if host.__name__ == "PatchMeta" else "patch"
    return f"dascore/core/{leaf}.py"


def _check_patch_listing(cls) -> None:
    """Find an operation's method, check it, and give it what reads it."""
    if not _HOSTS:
        _UNCHECKED.append(cls)
        return
    # Only reached for a named class, which is the one with a method.
    assert cls.name is not None
    own = _is_dascores(cls)
    func = _hosted_method(cls) if own else _declared_method(cls)
    _check_signature(cls, func)
    _check_forwarding(cls, func)
    if own:
        _check_returns_self(cls, func)
    cls.patch_function = _prepare_patch_function(cls, func)
    register_patch_function(cls.patch_function)
    if not own:
        return
    # The operation's long-standing functional spellings, both the method
    # itself rather than a copy of it: `dascore.proc.coords.select` is a
    # documented URL, and `dascore.proc.select` is how a body holding no
    # patch reaches it. Set here because the method is written in a class
    # which cannot exist when either module is read.
    for module in (cls.__module__, "dascore.proc"):
        setattr(sys.modules[module], cls.name, cls.patch_function)


def _hosted_method(cls: type[PatchProcessor]):
    """Return the method DASCore's own class writes for an operation."""
    patch_class, meta_class = _HOSTS
    host, other = (meta_class, patch_class)
    if not _meta_hosted(cls):
        host, other = other, host
    if _method_for(cls, other) is not None:
        because = "writes no kernel" if host is meta_class else "computes data"
        msg = (
            f"{other.__name__} defines {cls.name!r}, but {cls.__name__} "
            f"{because}, so it belongs to {host.__name__}. Move the method "
            f"from {_home(other)} to {_home(host)}."
        )
        raise ParameterError(msg)
    if (func := _method_for(cls, host)) is not None:
        # Checked before anything is stamped onto it: two classes claiming
        # one name would otherwise leave the first one's method pointing at
        # the second, and only then be refused.
        owner = getattr(func, "__processor__", None)
        # Spelled rather than compared by identity, the rule
        # `register_patch_function` already applies: a module reloaded makes
        # a new class object for the same name, and replacing its own entry
        # is what reloading means.
        if owner is not None and _spell(owner) != _spell(cls):
            msg = (
                f"Two classes claim the tag {cls.name!r}: {owner.__name__} "
                f"and {cls.__name__}. {host.__name__}.{cls.name} is "
                f"{owner.__name__}'s, so {cls.__name__} needs a name of its "
                "own, or `name = None` to claim none."
            )
            raise ParameterError(msg)
        return func
    msg = (
        f"{host.__name__} defines no {cls.name!r} method, which "
        f"{cls.__name__} is registered as, so nothing can reach it. Add "
        f"`def {cls.name}(self, ...) -> Self` to {_home(host)}, whose body "
        f"is `return {cls.__name__}(...).run(self)`."
    )
    raise ParameterError(msg)


def _declared_method(cls: type[PatchProcessor]):
    """Return the method a class outside DASCore declares for itself."""
    # Only reached for a named class, which is the one with a method.
    assert cls.name is not None
    # Nothing outside DASCore can write into `Patch`, and the data-less
    # contract which puts a method on `PatchMeta` is one DASCore holds only
    # itself to, so the class carries its own doorway. Its own body, never
    # a parent's: a subclass reaching one would stamp the parent's function
    # with its own class.
    if not isinstance(cls.__dict__.get(cls.name), staticmethod):
        msg = (
            f"{cls.__name__} is registered as {cls.name!r} and is not "
            f"DASCore's own, so it must declare `@staticmethod def "
            f"{cls.name}(patch, /, ...)` itself, whose parameters are the "
            "operation's. A subclass which is a variation rather than an "
            f"operation of its own sets `name = None`: inheriting a parent's "
            "method would build the parent, not this class."
        )
        raise ParameterError(msg)
    # Resolved through the class, which is the plain function a staticmethod
    # holds: the object itself would never be handed the patch.
    return getattr(cls, cls.name)


def _subclasses(cls):
    """Yield every subclass of a class, at any depth."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _recheck_for_kernel(cls) -> None:
    """Refuse a kernel which would move an operation between the classes."""
    # Nothing is listed yet, and the deferred check asks the question fresh.
    if not _HOSTS:
        return
    patch_class, meta_class = _HOSTS
    # `kernel_for` walks the MRO, so this kernel answers for every subclass
    # as well: each of DASCore's own which metadata still lists belongs on
    # `Patch` now, and a class body is not something a process can rewrite.
    moved = [
        x.name
        for x in (cls, *_subclasses(cls))
        if x.name is not None
        and _is_dascores(x)
        and _method_for(x, meta_class) is not None
    ]
    if not moved:
        return
    msg = (
        f"A kernel registered for {cls.__name__} gives {moved} data to "
        f"compute, so their methods must move from {_home(meta_class)} to "
        f"{_home(patch_class)}. Move them first, or write the kernel into "
        "the class body where it is read at class creation."
    )
    raise ParameterError(msg)


def check_patch_listings(patch_class, meta_class) -> None:
    """Check the classes made before there was anything to check against."""
    _HOSTS[:] = [patch_class, meta_class]
    unchecked, _UNCHECKED[:] = list(_UNCHECKED), []
    for cls in unchecked:
        _check_patch_listing(cls)


def register_kernel(cls: type[PatchProcessor], backend: str):
    """
    Say that a function is how an operation runs on one array backend.

    Used as a decorator. The kernel has the class's own `kernel` signature,
    `(processor, data, **plan)`, and returns an array.

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
        previous = cls.__dict__.get("_kernels")
        cls._kernels = {**cls.__dict__.get("_kernels", {}), backend: func}
        try:
            # A class body which wrote no kernel looked metadata-only
            # when its method was checked; this kernel says otherwise, for
            # the class and for everything which inherits it.
            _recheck_for_kernel(cls)
        except Exception:
            # Refused, so the class is left as it was rather than
            # holding a kernel whose method is on the wrong class.
            if previous is None:
                del cls._kernels
            else:
                cls._kernels = previous
            raise
        return func

    return decorate


def _method_for(cls: type[PatchProcessor], host: type):
    """Return the method a class's host defines for it, or None."""
    # Only reached for a named class, which is the one with a method.
    assert cls.name is not None
    # Its own body, not what it inherits: `Patch` answers for every name
    # `PatchMeta` defines, which is the point of defining them there.
    found = host.__dict__.get(cls.name)
    return found if isinstance(found, FunctionType) else None


def _check_signature(cls: type[PatchProcessor], func) -> None:
    """Refuse a method whose parameters have drifted from the fields."""
    parameters = inspect.signature(func).parameters
    # One way only: a field the method cannot be given is unreachable, while
    # a parameter which is not a field is a value the body resolves for
    # itself before building the instance.
    if missing := sorted(set(cls.model_fields) - set(parameters)):
        msg = (
            f"{cls.name} does not take {missing}, which the class declares "
            "as fields, so no call can reach them. Add them to the method's "
            "parameters."
        )
        raise ParameterError(msg)
    # A class which takes extras needs somewhere for a call to spell them:
    # `tile_apply(time=0.05)` is a window, not a field, and a method with no
    # var-keyword would refuse it before the class ever saw it.
    if cls.model_config.get("extra") == "allow" and not any(
        x.kind is x.VAR_KEYWORD for x in parameters.values()
    ):
        msg = (
            f"{cls.name} takes no `**kwargs`, but the class allows extras, "
            "so no call can reach them. Add a var-keyword parameter, or set "
            "the class's `extra` to forbid them."
        )
        raise ParameterError(msg)
    positional = cls._positional_names()
    for name, field in cls.model_fields.items():
        parameter = parameters[name]
        # `*args` and `**kwargs` collect rather than default.
        if parameter.kind in {parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD}:
            continue
        # A field the class refuses by position must be refused by the
        # method too, or a call places a value the class would not have.
        if name not in positional and parameter.kind is not parameter.KEYWORD_ONLY:
            msg = (
                f"{cls.name} takes {name} by position, which the class "
                "refuses: it is not in `_positional_fields`. Put a bare `*` "
                "before it, or add it to `_positional_fields`."
            )
            raise ParameterError(msg)
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        if not _values_equal(parameter.default, default):
            msg = (
                f"{cls.name} defaults {name} to {parameter.default!r} where "
                f"{cls.__name__} defaults it to {default!r}; the two must "
                "agree on what unset means."
            )
            raise ParameterError(msg)


def _check_forwarding(cls: type[PatchProcessor], func) -> None:
    """Refuse a body which builds its class without one of its fields."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(func)))
    except (OSError, TypeError, SyntaxError):
        # Nothing to read for a function built at runtime. The parameters
        # and their defaults are still checked, which is the drift a class
        # written in a file can have.
        return
    for node in ast.walk(tree):
        # The call which builds the class, named as the body spells it --
        # bare, or through the module it is imported from.
        if not isinstance(node, ast.Call):
            continue
        spelled = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if spelled != cls.__name__:
            continue
        given = {x.arg for x in node.keywords}
        # `**something` may carry anything, so it answers for every field.
        if None in given:
            return
        if missing := sorted(set(cls.model_fields) - given):
            msg = (
                f"{cls.name} takes {missing} but does not pass them to "
                f"{cls.__name__}, so the values a caller gives are dropped. "
                "Pass each one by name."
            )
            raise ParameterError(msg)
        return


def _check_returns_self(cls: type[PatchProcessor], func) -> None:
    """Refuse a method which claims a kind other than the one it was given."""
    # `run` gives back what it was handed, so metadata comes back metadata
    # and a `Patch` subclass comes back that subclass. Only `Self` says so;
    # a class named outright would flatten both.
    returns = inspect.signature(func).return_annotation
    # Both spellings: a module with postponed annotations hands back the
    # name as a string, and one without hands back `Self` itself.
    if returns in ("Self", Self):
        return
    spelled = (
        "is not annotated"
        if returns is inspect.Signature.empty
        else f"is annotated `-> {returns}`"
    )
    msg = (
        f"{cls.name} {spelled}, but `run` gives back the kind it was given, "
        "which only `-> Self` says. Annotate it `-> Self`."
    )
    raise ParameterError(msg)


def _make_bypass(func):
    """Return the operation without the history and the ids."""

    @functools.wraps(func)
    def raw_function(patch, /, *args, **kwargs):
        """Run the operation, recording nothing of this call."""
        # The method builds the processor itself, so the flag is where the
        # two meet. `run` spends it, so an operation which runs another
        # inside itself still records that one.
        token = _RECORD.set(False)
        try:
            return func(patch, *args, **kwargs)
        finally:
            _RECORD.reset(token)

    # `wraps` marks a wrapper as a view of what it wraps, and two things
    # read that mark: `inspect.signature` to see the real parameters
    # through it, which `call_operation_id` needs, and the docs builder to
    # resolve a wrapper back to what it wraps. Left in place, the second
    # makes `select.raw_function` resolve to `select` and gives every
    # converted operation a page holding a table of itself. Written out
    # rather than pointed at, so the signature survives and the pointer
    # does not: a bypass is a second way to call the operation, not a
    # stand-in for it.
    # `Any`, because `wraps` types its result as a view of the wrapped
    # function, which is the very thing being written out here.
    bypass: Any = raw_function
    bypass.__signature__ = inspect.signature(func)
    del bypass.__wrapped__
    # The path which actually resolves, so a process pool can pickle it:
    # `functools.wraps` copied the method's, which names no attribute of
    # its own. Distinct per operation, so two bypasses given as arguments
    # to another operation id as two callables rather than one closure.
    raw_function.__qualname__ = f"{func.__qualname__}.raw_function"
    return raw_function


def _prepare_patch_function(cls: type[PatchProcessor], func):
    """Give a class's method what the framework and the docs read."""
    # `Any`, because the attributes below are ones a plain function lacks.
    out: Any = func
    # The registry tags an operation by the function's name, which must be
    # the one the class registered: a doorway written as
    # `scale = staticmethod(_scale_impl)` would otherwise be tagged
    # `package:_scale_impl`, which nothing resolves and no history replays.
    out.__name__ = cls.name
    # Built after the rename and before the attributes below, because
    # `functools.wraps` copies both the name and the function's `__dict__`,
    # and the attributes would then point the bypass back at itself.
    raw = _make_bypass(func)
    out.__version__ = cls.__version__
    # The operation is documented once, with the processor: its parameters,
    # notes and examples are there, and the method's own docstring is the
    # one-line summary a reader of the class wants.
    out.__doc__ = cls.__doc__
    # `_history` is read by `record_call`; `__processor__` by the docs
    # builder, which reads the class for the source file and lines.
    out._history = cls.history
    out.__processor__ = cls
    # What a decorated function's `.func` always was: the operation without
    # the history and ids, for a body calling another operation.
    out.func = out.raw_function = raw
    return out


def _checked_plan(processor: PatchProcessor, plan: dict[str, Any]) -> dict[str, Any]:
    """Refuse a plan holding anything but numbers, indices, or arrays."""
    for key, value in plan.items():
        if not _is_plain(value):
            msg = (
                f"{type(processor).__name__}.get_metadata returned "
                f"{key}={value!r}; a plan may hold only ints, floats, bools, "
                "slices, None, Ellipsis, tuples or lists of those, and "
                "numeric arrays, so the kernel never sees a patch."
            )
            raise ParameterError(msg)
    return plan


def _is_plain(value) -> bool:
    """
    Whether a plan value is one a kernel may take.

    Numbers and numeric arrays are the computation's; slices, None,
    Ellipsis and sequences of them are how an index into the data is
    spelled. Anything else -- a patch, a coord manager, a string -- would
    put metadata back in front of a kernel.
    """
    if value is None or value is Ellipsis:
        return True
    if isinstance(value, numbers.Number | np.bool_):
        return True
    if isinstance(value, slice):
        return all(_is_plain(x) for x in (value.start, value.stop, value.step))
    if isinstance(value, tuple | list):
        return all(_is_plain(x) for x in value)
    return isinstance(value, np.ndarray) and value.dtype.kind in "biufc"
