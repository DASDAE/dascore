"""Patch operations written as classes.

A [`PatchProcessor`](`dascore.core.processor.PatchProcessor`) subclass is a
whole operation: its fields are the parameters, `derive` and `plan` work out
the result's metadata and the numbers the kernel needs without touching data,
and `kernel` computes the array. Subclassing registers the operation and
generates its patch function (`cls.patch_function`), so the class is written
once and the function is never hand-written.

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
import re
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from pydantic import ConfigDict

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.utils.array_api import backend_name
from dascore.utils.identity import ids_enabled
from dascore.utils.patch import (
    _maybe_add_history_str,
    attr_type,
    call_str,
    check_patch_attrs,
    check_patch_coords,
    check_patch_data,
    stamp_ids,
)
from dascore.utils.patch_registry import (
    _spell,
    _without_patches,
    memoized_fingerprint,
    patch_function_tag,
    register_patch_function,
    resolve_patch_function,
)
from dascore.utils.serialize import model_values

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs
    from dascore.core.coordmanager import CoordManager


class PatchProcessor(DascoreBaseModel):
    """
    An operation on a patch, written as a class.

    Subclasses declare their parameters as fields and override some of:

    - `derive(patch)`: the result's metadata, from a patch without data.
    - `plan(patch, out)`: the numbers `kernel` needs, as a dict of ints,
      floats, bools, tuples of those, or numeric arrays.
    - `kernel(data, **plan)`: the array computation. None of it may see a
      patch, so a chain of kernels can be compiled.
    - `reconcile(data, out)`: the one hook which sees both halves.

    Each subclass is registered under `name` (snake case of the class name
    unless set; None registers nothing) and gets a generated patch function,
    `cls.patch_function`, whose signature is the fields in declaration
    order.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bump when the same fields mean a different result.
    __version__: ClassVar[str] = "1.0"
    # The patch method this class is; None registers nothing.
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
        if "name" not in cls.__dict__:
            cls.name = _snake_case(cls.__name__)
        cls._call_signature = _call_signature(cls)
        cls.patch_function = None
        if cls.name is not None:
            cls.patch_function = _make_patch_function(cls, cls.name)
            register_patch_function(cls.patch_function)

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
        return memoized_fingerprint(
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

    def __call__(self, patch: PatchType) -> PatchType:
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

    def check(self, patch: PatchType) -> PatchType:
        """
        Refuse a patch which does not carry what the operation needs.

        Raises
        ------
        PatchDataError
            If the patch holds no data.
        PatchCoordinateError
            If a required dimension or coordinate is missing.
        PatchAttributeError
            If a required attr is missing, or holds a different value.
        """
        check_patch_data(patch)
        check_patch_coords(patch, dims=self.required_dims, coords=self.required_coords)
        return check_patch_attrs(patch, self.required_attrs)

    def derive(self, patch: PatchType) -> PatchType:
        """
        Return the result's metadata, as a patch without data.

        Given a patch without data, so it may not read `.data`; one which
        must is an operation that cannot be planned without its values.
        Work through the coord manager and `patch.new`, never through patch
        methods. The default says nothing changes.
        """
        return patch

    def plan(self, patch: PatchType, out: PatchType) -> dict[str, Any]:
        """Return the keyword arguments `kernel` needs; see the class docs."""
        return {}

    def reconcile(self, data, out: PatchType) -> PatchType:
        """Return the result's metadata once the data are known; default as is."""
        return out

    def run(self, patch: PatchType) -> PatchType:
        """
        Run the operation: check, derive, plan, kernel, reconcile, record.

        An operation which changes neither the metadata nor the data hands
        back the patch it was given, and records nothing.
        """
        self.check(patch)
        described = patch.drop_data()
        out = self.derive(described)
        plan = _checked_plan(self, self.plan(described, out))
        data = patch.data
        kernel = self.kernel_for(backend_name(data))
        # A kernel says "nothing to do" by handing its argument back, so it
        # must not write into that argument and return it.
        result = data if kernel is None else kernel(self, data, **plan)
        if out is described and result is data:
            return patch
        out = self.reconcile(result, out)
        return out.new(data=result, attrs=self._record(patch, out.attrs))

    def _record(self, patch: PatchType, attrs: PatchAttrs) -> PatchAttrs:
        """Return attrs carrying the data_type, history and ids of this call."""
        if self.data_type is not None:
            attrs = attrs.update(data_type=self.data_type)
        name = self.name or type(self).__name__
        if self.history == "full":
            attrs = _maybe_add_history_str(attrs, call_str(name, self.kwargs))
        elif self.history is not None:
            attrs = _maybe_add_history_str(attrs, name)
        if ids_enabled():
            fields = self.kwargs.values()
            others = [x for x in fields if isinstance(x, dc.Patch)]
            attrs = stamp_ids(patch, attrs, self.fingerprint, others)
        return attrs


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
        cls._kernels = {**cls.__dict__.get("_kernels", {}), backend: func}
        return func

    return decorate


def _snake_case(name: str) -> str:
    """Return `RenameCoords` as `rename_coords`."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _call_signature(cls: type[PatchProcessor]) -> inspect.Signature:
    """Return the signature a call binds against: the fields, then extras."""
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    parameters = []
    for name, field in cls.model_fields.items():
        if name == cls._var_positional:
            parameters.append(inspect.Parameter(name, inspect.Parameter.VAR_POSITIONAL))
            kind = inspect.Parameter.KEYWORD_ONLY
            continue
        default = inspect.Parameter.empty if field.is_required() else field.default
        parameters.append(inspect.Parameter(name, kind, default=default))
    if cls.model_config.get("extra") == "allow":
        parameters.append(inspect.Parameter("kwargs", inspect.Parameter.VAR_KEYWORD))
    return inspect.Signature(parameters)


def _make_patch_function(cls: type[PatchProcessor], name: str):
    """Return the patch function a processor class generates."""

    def patch_function(patch, *args, **kwargs):
        return cls(*args, **kwargs).run(patch)

    # `Any`, because the attributes below are ones a plain function lacks.
    func: Any = patch_function
    patch = inspect.Parameter("patch", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    signature = cls._call_signature
    func.__signature__ = signature.replace(
        parameters=[patch, *signature.parameters.values()]
    )
    func.__name__ = name
    prefix = cls.__qualname__.rpartition(".")[0]
    func.__qualname__ = f"{prefix}.{name}" if prefix else name
    func.__module__ = cls.__module__
    func.__doc__ = cls.__doc__
    func.__version__ = cls.__version__
    # Read by `record_call` and by the docs builder, which files the
    # function under its class's module.
    func._history = cls.history
    func.__processor__ = cls
    func.raw_function = func
    return func


def _checked_plan(processor: PatchProcessor, plan: dict[str, Any]) -> dict[str, Any]:
    """Refuse a plan holding anything but numbers, tuples of them, or arrays."""
    for key, value in plan.items():
        if not _is_plain(value):
            msg = (
                f"{type(processor).__name__}.plan returned {key}={value!r}; a "
                "plan may hold only ints, floats, bools, tuples of those, and "
                "numeric arrays, so the kernel never sees a patch."
            )
            raise ParameterError(msg)
    return plan


def _is_plain(value) -> bool:
    """Whether a plan value is a number, a tuple of plain values, or an array."""
    if isinstance(value, numbers.Number):
        return True
    if isinstance(value, tuple):
        return all(_is_plain(x) for x in value)
    return isinstance(value, np.ndarray) and value.dtype.kind in "biufc"


# --- the old seam, kept only for tile_apply and adaptive_spectral_filter ---
#
# Both need a window in their kernels, which a plan cannot yet carry; they
# move onto `PatchProcessor` in the next change, and this goes with them.


@dataclass(frozen=True, slots=True)
class PatchMeta:
    """
    Everything a patch carries except its data.

    Parameters
    ----------
    coords
        The coordinates, which carry the dimensions and the shape too.
    attrs
        The patch's attributes.
    dtype
        What the data are.
    backend
        Which array library the data belong to.
    """

    coords: CoordManager
    attrs: PatchAttrs
    dtype: Any
    backend: str = "numpy"
    patch_type: Any = None

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimension names, in order."""
        return self.coords.dims

    @property
    def shape(self) -> tuple[int, ...]:
        """The shape the data have."""
        return self.coords.shape

    @property
    def ndim(self) -> int:
        """How many dimensions the patch has."""
        return len(self.coords.dims)

    def get_axis(self, dim: str) -> int:
        """Return the axis a dimension name refers to."""
        return self.coords.get_axis(dim)

    @classmethod
    def from_patch(cls, patch) -> PatchMeta:
        """Return what a patch is, apart from its values."""
        data = patch.data
        return cls(
            coords=patch.coords,
            attrs=patch.attrs,
            dtype=data.dtype,
            backend=backend_name(data),
            patch_type=type(patch),
        )

    def update(self, **kwargs) -> PatchMeta:
        """Return metadata with some of it changed."""
        return replace(self, **kwargs)

    def to_patch(self, data):
        """Return the patch this metadata and some data make."""
        patch_type = self.patch_type or dc.Patch
        return patch_type(data=data, coords=self.coords, attrs=self.attrs)


class _MetaProcessor(DascoreBaseModel):
    """The old metadata/kernel seam: `derive_meta`, `kernel(data, meta, out)`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # The patch function this class is the body of.
    _patch_function: ClassVar[str] = ""

    @property
    def kwargs(self) -> dict[str, Any]:
        """Return the validated fields, extras included."""
        return model_values(self)

    def __call__(self, patch: PatchType) -> PatchType:
        """Run the operation through its patch function."""
        return resolve_patch_function(self._patch_function)(patch, **self.kwargs)

    def derive_meta(self, meta: PatchMeta) -> PatchMeta:
        """Return the result's metadata; by default, unchanged."""
        return meta

    def _apply(self, patch: PatchType) -> PatchType:
        """Run the operation; the patch function records history and ids."""
        meta = PatchMeta.from_patch(patch)
        out_meta = self.derive_meta(meta)
        data = self.kernel(patch.data, meta, out_meta)  # ty: ignore[unresolved-attribute]
        dtype = getattr(data, "dtype", out_meta.dtype)
        if dtype != out_meta.dtype:
            out_meta = out_meta.update(dtype=dtype)
        return out_meta.to_patch(data)
