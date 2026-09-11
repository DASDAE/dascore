"""Split a patch operation into its metadata and its array kernel.

A [`PatchProcessor`](`dascore.core.processor.PatchProcessor`) derives the
result's metadata separately from running its array kernel, so a chain of
operations can be planned without loading data. A class is registered for a
patch function's name with
[`register_implementation`](`dascore.core.processor.register_implementation`),
and backend packages add kernels with
[`register_kernel`](`dascore.core.processor.register_kernel`).

Examples
--------
>>> import dascore as dc
>>> from dascore.proc.basic import Normalize
>>>
>>> patch = dc.get_example_patch()
>>> assert Normalize(dim="time")(patch).equals(patch.normalize(dim="time"))
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, replace
from functools import cached_property
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ConfigDict

import dascore as dc
from dascore.constants import PatchType
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.utils.array_api import backend_name
from dascore.utils.patch import attr_type, check_patch_attrs, check_patch_coords
from dascore.utils.patch_registry import (
    _as_call,
    _fingerprint,
    _signature,
    resolve_patch_function,
)
from dascore.utils.serialize import model_values

if TYPE_CHECKING:
    from dascore.core.attrs import PatchAttrs
    from dascore.core.coordmanager import CoordManager

# The names which have a hand-written class.
_IMPLEMENTATIONS: dict[str, type[PatchProcessor]] = {}


class PatchProcessor(DascoreBaseModel):
    """
    An operation on a patch, written out by hand.

    Subclasses declare their parameters as fields and say what a patch has
    to carry for the operation to mean anything. They exist where an
    operation wants a seam a whole function does not have -- a kernel to
    dispatch, a plan to reuse -- and are registered with
    [`register_implementation`](`dascore.core.processor.register_implementation`)
    so that the patch function's name reaches them.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Bumped whenever the same parameters should mean a different answer.
    __version__: ClassVar[str] = "1.0"

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

    @property
    def name(self) -> str:
        """Return the registry tag of the operation this implements."""
        return self._patch_function

    @property
    def kwargs(self) -> dict:
        """Return the arguments the operation was given."""
        return model_values(self)

    @cached_property
    def fingerprint(self) -> str:
        """
        Return the digest which identifies this operation and its arguments.

        Spelled as the patch function's call is (`fingerprint_call`), not
        as this class, so both routes stamp the same `processing_id`.
        """
        return _fingerprint(self.name, self.__version__, self.kwargs)

    def __eq__(self, other) -> bool:
        """Two processors are equal if they are the same operation."""
        if not isinstance(other, PatchProcessor):
            return NotImplemented
        return type(self) is type(other) and self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        """Hash a processor the way it compares."""
        return hash(self.fingerprint)

    def __call__(self, patch: PatchType) -> Any:
        """Run the operation; see `run`."""
        return self.run(patch)

    def run(self, patch: PatchType) -> Any:
        """
        Run the operation against a patch.

        Through the patch function, not straight into `_apply`: the
        function's decorator is what writes the history and stamps the
        ids, so both routes record the same thing.
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

        The default finds a kernel registered for the data's backend, and
        failing that the class's own `kernel`, which is written to the
        array API standard and so runs on any of them. A class with no
        kernel at all is a metadata-only operation and gets None.
        """
        if (found := _resolve_kernel(type(self), meta.backend)) is None:
            return None
        return functools.partial(found, self, meta=meta, out_meta=out_meta)

    def reconcile(self, data, meta: PatchMeta) -> PatchMeta:
        """
        Return the metadata the data actually turned out to have.

        Defining this says the operation cannot be fused: it is the one
        step which has to see both halves at once. The default only
        carries the data's dtype back, since a kernel may promote.
        """
        dtype = getattr(data, "dtype", meta.dtype)
        return meta if dtype == meta.dtype else meta.update(dtype=dtype)

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


def _resolve_kernel(cls: type[PatchProcessor], backend: str):
    """
    Return the kernel a class runs for one backend, or None if it has none.

    A kernel registered for the backend wins; failing that the class's own
    `kernel`, which is written to the array API standard and so runs on
    any of them. A class which defines neither is metadata-only.
    """
    # One class at a time, both questions asked of it before moving up:
    # a subclass which wrote its own `kernel` means it, and a backend
    # kernel registered against its parent must not answer for it.
    for klass in cls.__mro__:
        contents = klass.__dict__
        if (found := contents.get("_kernels", {}).get(backend)) is not None:
            return found
        if (generic := contents.get("kernel")) is not None:
            return generic
    return None


def register_implementation(name: str, cls: type[PatchProcessor]) -> None:
    """
    Say that a hand-written class is what a patch function's name means.

    One name has one implementation, so there is a single code path
    whether the operation was reached by name or by class.

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


# A dataclass rather than a model, so building one revalidates no coord
# manager.
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
        What the data are. Held separately because a kernel may promote --
        `normalize` divides, so integers come back as floats -- and the
        metadata cannot always say in advance what it will be.
    backend
        Which array library the data belong to, as
        [`backend_name`](`dascore.utils.array_api.backend_name`) spells
        it. This is how a kernel is chosen without looking at an array.
    """

    coords: CoordManager
    attrs: PatchAttrs
    dtype: Any
    backend: str = "numpy"
    # What to build the result with. A patch function used to go through
    # `patch.new`, which builds `self.__class__`, so a subclass survived
    # its own operations; naming `Patch` here would quietly take that
    # away. None means whatever `dascore.Patch` is.
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
        """
        Return the patch this metadata and some data make.

        The coords check the data's shape on the way in, so a kernel
        which returned the wrong shape is refused here rather than
        somewhere later and stranger.
        """
        patch_type = self.patch_type or dc.Patch
        return patch_type(data=data, coords=self.coords, attrs=self.attrs)
