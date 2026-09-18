"""Patch operations written as classes.

A [`PatchProcessor`](`dascore.core.processor.PatchProcessor`) subclass is a
whole operation: its fields are the parameters, `get_metadata` works out the
result's metadata and the numbers the kernel needs without touching data, and
`kernel` computes the array. Subclassing registers the operation and
generates its patch function (`cls.patch_function`), so the class is written
once and the function is never hand-written.

One of DASCore's own operations which writes no kernel changes metadata
alone, so it is bound onto [`PatchMeta`](`dascore.PatchMeta`) and runs on a
patch's metadata as readily as on the patch. An operation defined outside
DASCore is bound to neither class and is called through `dc.proc`: the
data-less contract is one DASCore holds itself to.

Examples
--------
>>> import dascore as dc
>>>
>>> class ScaleExample(dc.PatchProcessor):
...     '''Multiply the data by a factor.'''
...
...     factor: float = 2.0
...
...     def kernel(self, data):
...         return data * self.factor
>>>
>>> patch = dc.get_example_patch()
>>> out = ScaleExample(3)(patch)
>>> assert out.equals(ScaleExample.patch_function(patch, factor=3))
>>> assert ScaleExample(3).fingerprint == ScaleExample(3.0).fingerprint
"""

from __future__ import annotations

import inspect
import numbers
from typing import TYPE_CHECKING, Any, ClassVar, overload

import numpy as np
from pydantic import ConfigDict
from pydantic.alias_generators import to_snake

import dascore as dc
from dascore.config import get_config
from dascore.constants import PatchMetaType, PatchType
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.utils.identity import ids_enabled
from dascore.utils.patch import (
    _call_str,
    _maybe_add_history_str,
    _stamp_ids,
    attr_type,
    check_patch_attrs,
    check_patch_coords,
    check_patch_data,
)
from dascore.utils.patch_registry import (
    _memoized_fingerprint,
    _spell,
    _without_patches,
    patch_function_tag,
    register_patch_function,
)
from dascore.utils.serialize import model_values

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs


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
    unless set) and gets a generated patch function, `cls.patch_function`:
    `(patch, <fields in declaration order>)`, with `*name` for the field
    `_var_positional` names and `**kwargs` when `extra="allow"`. A class
    with `name = None` gets neither.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump when the same fields mean a different result.
    __version__: ClassVar[str] = "1.0"
    # The registry name, and the generated function's; None for neither.
    name: ClassVar[str | None] = None
    # What the patch must hold.
    required_dims: ClassVar[tuple[str, ...] | str | None] = None
    required_coords: ClassVar[tuple[str, ...] | str | None] = None
    required_attrs: ClassVar[attr_type] = None
    # The result's data_type: None keeps the input's, "" clears it.
    data_type: ClassVar[str | None] = None
    # How a call is written into history: "full", "method_name" or None.
    history: ClassVar[str | None] = "full"
    # The field a `*args` group fills in the generated function, if any.
    _var_positional: ClassVar[str | None] = None
    # The fields a call may give positionally, in order; None for all.
    _positional_fields: ClassVar[tuple[str, ...] | None] = None
    # Kernels registered per backend by `register_kernel`, looked up in each
    # class's own `__dict__` so a subclass never answers with its parent's.
    _kernels: ClassVar[dict[str, Any]] = {}
    # Generated for each named subclass.
    patch_function: ClassVar[Any] = None
    # The signature a call binds against: the fields, without the patch.
    _call_signature: ClassVar[inspect.Signature]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Register the subclass and generate its patch function."""
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
        cls._call_signature = _call_signature(cls)
        cls.patch_function = None
        if cls.name is not None:
            cls.patch_function = _make_patch_function(cls, cls.name)
            register_patch_function(cls.patch_function)
            _bind_patch_function(cls)

    def __init__(self, /, *args, **kwargs):
        """Bind positional arguments as the generated function would."""
        if args:
            bound = type(self)._call_signature.bind(*args, **kwargs).arguments
            kwargs = {**bound.pop("kwargs", {}), **bound}
        super().__init__(**kwargs)

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return the validated fields, extras included."""
        return model_values(self)

    @property
    def tag(self) -> str:
        """Return the name this operation is fingerprinted under."""
        cls = type(self)
        func = cls.patch_function
        if func is not None and (tag := patch_function_tag(func)) is not None:
            return tag
        # Unregistered or defined inside a call: named by where it was
        # written and which class it is, which is honestly process-local.
        return f"{_spell(cls)}#{id(cls):x}"

    @property
    def fingerprint(self) -> str:
        """Return the digest of the tag, version and validated fields."""
        return _memoized_fingerprint(
            type(self), self.tag, _without_patches(self.kwargs), self.__version__
        )

    def __eq__(self, other) -> bool:
        """Two processors are equal if they are the same operation."""
        if not isinstance(other, PatchProcessor):
            return NotImplemented
        return type(self) is type(other) and self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        """Hash a processor the way it compares."""
        return hash(self.fingerprint)

    @overload
    def __call__(self, patch: PatchType) -> PatchType: ...

    @overload
    def __call__(self, patch: dc.PatchMeta) -> dc.PatchMeta: ...

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
    def run(self, patch: dc.PatchMeta) -> dc.PatchMeta: ...

    def run(self, patch):
        """
        Run the operation: check, get_metadata, kernel, reconcile, record.

        A patch comes back a patch and metadata comes back metadata, which
        is what the two overloads say. An operation which changes neither
        the metadata nor the data hands back what it was given, and records
        nothing.
        """
        return self._run(patch, record=True)

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
            attrs = out.attrs if not record else self._record(patch, out.attrs)
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
        if not record:
            return out.to_patch(result)
        return out.update(attrs=self._record(patch, out.attrs)).to_patch(result)

    def _unchanged(self, patch: dc.PatchMeta, record: bool) -> dc.PatchMeta:
        """Return the patch an operation did nothing to; nothing is recorded."""
        # A declared data_type still applies, as it does for a decorated
        # patch function.
        if self.data_type is None or not record:
            return patch
        return patch.update_attrs(data_type=self.data_type)

    def _record(self, patch: dc.PatchMeta, attrs: PatchAttrs) -> PatchAttrs:
        """Return attrs carrying the data_type, history and ids of this call."""
        if self.data_type is not None:
            attrs = attrs.update(data_type=self.data_type)
        name = self.name or type(self).__name__
        if self.history is not None and get_config().patch_history != "disabled":
            spelled = _call_str(name, self.kwargs) if self.history == "full" else name
            attrs = _maybe_add_history_str(attrs, spelled)
        if not ids_enabled():
            return attrs
        try:
            fingerprint = self.fingerprint
        except Exception:
            # As for a patch function: a field the serializer cannot encode
            # means no ids for this call, never a failed call.
            return attrs
        others = [x for x in self.kwargs.values() if isinstance(x, dc.Patch)]
        return _stamp_ids(patch, attrs, fingerprint, others)


# Names a subclass field may not take: the base's own settings and methods.
_RESERVED = frozenset(x for x in vars(PatchProcessor) if not x.startswith("_")) | {
    "name",
    "data_type",
    "history",
    "patch_function",
}


# Classes whose function is generated before `dascore.core.patch` has
# finished importing, which is most of DASCore's own; the module drains them
# once both host classes exist.
_PENDING: list[type[PatchProcessor]] = []
# `[Patch, PatchMeta]`, once there are such classes to bind to.
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
    """Whether a generated function is bound onto `PatchMeta`."""
    # An operation with no kernel changes metadata alone, so every PatchMeta
    # can run it and Patch inherits it. One with a kernel needs data, and
    # `Patch` lists those in its body by hand rather than growing them at
    # import: a class whose surface is written down is one a type checker,
    # an IDE and a reader can all see whole.
    # A trust boundary, not a namespace detail: the data-less contract is
    # one DASCore holds itself to, so a plugin never lands on PatchMeta.
    return _is_dascores(cls) and not _has_kernel(cls)


def _bind_patch_function(cls) -> None:
    """Bind a generated function onto `PatchMeta`, or check `Patch` lists it."""
    if not _HOSTS:
        _PENDING.append(cls)
        return
    meta_class = _HOSTS[1]
    if _meta_hosted(cls):
        existing = getattr(meta_class, cls.name, None)
        if existing is not None and not hasattr(existing, "__processor__"):
            msg = (
                f"{cls.__name__} would bind {cls.name!r} onto "
                f"{meta_class.__name__}, which already means something else "
                "there; rename the class or give it a free `name`."
            )
            raise ParameterError(msg)
        setattr(meta_class, cls.name, cls.patch_function)
        return
    # A kernel, whether written in the body or registered later: metadata
    # cannot run this, so it must not still answer for it.
    if meta_class.__dict__.get(cls.name) is cls.patch_function:
        delattr(meta_class, cls.name)


def _check_patch_lists(cls, patch_class) -> None:
    """Refuse one of DASCore's own operations which `Patch`'s body omits."""
    # Nothing out of tree can be written into Patch's body, so there is
    # nothing to check: a plugin's operation is reached through the registry
    # and `dc.proc`, as it was before any of this was bound anywhere.
    if not _is_dascores(cls):
        return
    # Its own body, not what it inherits: an operation still bound onto
    # PatchMeta answers `getattr` here right up until it is unbound.
    if patch_class.__dict__.get(cls.name) is cls.patch_function:
        return
    msg = (
        f"{cls.__name__} computes data, so it is a {patch_class.__name__} "
        f"method, but {patch_class.__name__} does not list {cls.name!r}. Add "
        f"`{cls.name} = dascore.proc.{cls.name}` to dascore/core/patch.py."
    )
    raise ParameterError(msg)


def _subclasses(cls):
    """Yield every subclass of a class, at any depth."""
    for sub in cls.__subclasses__():
        yield sub
        yield from _subclasses(sub)


def _rebind_for_kernel(cls) -> None:
    """Re-site a class, and its descendants, around a new kernel."""
    # Nothing is bound yet, and the pending drain asks the question fresh.
    if not _HOSTS:
        return
    # `kernel_for` walks the MRO, so this kernel answers for every subclass
    # as well: one bound as metadata-only before now belongs where the data
    # are. A subclass with a kernel of its own is already there, and an
    # unnamed one was never bound at all.
    patch_class = _HOSTS[0]
    for klass in (cls, *_subclasses(cls)):
        if klass.patch_function is None:
            continue
        # Asked before anything moves: unbinding it from PatchMeta leaves
        # nothing behind unless Patch's body lists it, so say which line is
        # missing rather than deleting a public method quietly.
        if not _meta_hosted(klass):
            _check_patch_lists(klass, patch_class)
        _bind_patch_function(klass)


def bind_pending_patch_functions(patch_class, meta_class) -> None:
    """Bind the functions generated before their host classes existed."""
    _HOSTS[:] = [patch_class, meta_class]
    pending, _PENDING[:] = list(_PENDING), []
    for cls in pending:
        _bind_patch_function(cls)
        # Asked only of what was written before the tree finished importing,
        # which is DASCore's own source and nothing else: a plugin, a
        # notebook or a docstring example cannot add a line to `Patch` and
        # is not asked to.
        if not _meta_hosted(cls):
            _check_patch_lists(cls, patch_class)


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
            # A class body which wrote no kernel looked metadata-only when
            # its function was generated; this kernel says otherwise, for
            # the class and for everything which inherits it.
            _rebind_for_kernel(cls)
        except Exception:
            # Refused, so the class is left as it was rather than holding a
            # kernel which nothing is bound to run.
            if previous is None:
                del cls._kernels
            else:
                cls._kernels = previous
            raise
        return func

    return decorate


def _call_signature(cls: type[PatchProcessor]) -> inspect.Signature:
    """Return the signature a call binds against: the fields, then extras."""
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = []
    for name, field in cls.model_fields.items():
        if name == cls._var_positional:
            parameters.append(inspect.Parameter(name, inspect.Parameter.VAR_POSITIONAL))
            kind = inspect.Parameter.KEYWORD_ONLY
            continue
        default = (
            inspect.Parameter.empty
            if field.is_required()
            else field.get_default(call_default_factory=True)
        )
        if cls._positional_fields is not None and name not in cls._positional_fields:
            kind = inspect.Parameter.KEYWORD_ONLY
        annotation = field.annotation or inspect.Parameter.empty
        # A required field after a defaulted one (a subclass adding one) can
        # only be given by name.
        if default is inspect.Parameter.empty and any(
            x.default is not inspect.Parameter.empty for x in parameters
        ):
            kind = inspect.Parameter.KEYWORD_ONLY
        parameters.append(
            inspect.Parameter(name, kind, default=default, annotation=annotation)
        )
    if cls.model_config.get("extra") == "allow":
        parameters.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _make_patch_function(cls: type[PatchProcessor], name: str):
    """Return the patch function a processor class generates."""

    def build(args, kwargs):
        """Bind a call against the signature, as a function's would be."""
        bound = cls._call_signature.bind(*args, **kwargs).arguments
        extras = dict(bound.pop("kwargs", {}))
        # `bind` fills every field but the `*args` one by name, so that is
        # the only field which can turn up among the extras, and it does so
        # only when a caller named something after it -- a coordinate called
        # `empty_dims`, say. Handed to the constructor it would land on the
        # field, so it is set as the extra `bind` already said it was.
        shadowed: dict[str, Any] = {}
        group = cls._var_positional
        if group is not None and group in extras:
            shadowed[group] = extras.pop(group)
        out = cls(**extras, **bound)
        if shadowed:
            # `bind` only had extras to give because the model allows them,
            # so pydantic has somewhere to put this one.
            assert out.__pydantic_extra__ is not None
            out.__pydantic_extra__.update(shadowed)
        return out

    # The patch is positional-only, so a field or an extra may be named
    # `patch`: `rename_coords(patch="renamed")`.
    def patch_function(patch, /, *args, **kwargs):
        return build(args, kwargs).run(patch)

    def bypass(patch, /, *args, **kwargs):
        return build(args, kwargs)._run(patch, record=False)

    # `Any`, because the attributes below are ones a plain function lacks.
    func: Any = patch_function
    patch = inspect.Parameter(
        "patch", inspect.Parameter.POSITIONAL_ONLY, annotation="PatchType"
    )
    signature = cls._call_signature
    func.__signature__ = signature.replace(
        parameters=[patch, *signature.parameters.values()],
        return_annotation="PatchType",
    )
    func.__name__ = name
    # The class attribute it is, so pickle finds it for any class defined at
    # module level, and two classes claiming one name collide in the
    # registry rather than one silently replacing the other.
    func.__qualname__ = f"{cls.__qualname__}.patch_function"
    func.__module__ = cls.__module__
    func.__doc__ = cls.__doc__
    func.__version__ = cls.__version__
    # `_history` is read by `record_call`; `__processor__` by the docs
    # builder, which reads the class for the source file and lines.
    func._history = cls.history
    func.__processor__ = cls
    # What a decorated function's `.func` always was: the operation without
    # the history and ids, for a body calling another operation.
    raw: Any = bypass
    raw.__signature__ = func.__signature__
    # Named for its class, so two bypasses given as arguments to another
    # operation fingerprint as two callables, not one closure.
    raw.__name__ = name
    raw.__qualname__ = f"{cls.__qualname__}.patch_function.raw_function"
    raw.__module__ = cls.__module__
    func.func = func.raw_function = raw
    return func


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
