"""
Tests for PatchProcessor: the seam, the generated function, and kernels.
"""

from __future__ import annotations

import inspect
import pickle
import subprocess
import sys
from types import FunctionType
from typing import Any, ClassVar, Self

import numpy as np
import pytest
from pydantic import ConfigDict, Field, ValidationError

import dascore as dc
from dascore.constants import PatchMetaType, PatchType
from dascore.core import processor as processor_module
from dascore.core.processor import PatchProcessor, register_kernel
from dascore.exceptions import (
    ParameterError,
    PatchAttributeError,
    PatchCoordinateError,
    PatchDataError,
)
from dascore.models import ArrayLike
from dascore.proc.basic import Abs, Normalize, _known_real
from dascore.utils.docs import compose_docstring
from dascore.utils.identity import encode
from dascore.utils.patch_registry import patch_function_tag, resolve_patch_function


class SeamScale(PatchProcessor):
    """Multiply the data by a factor."""

    factor: float = 2.0

    @staticmethod
    def seam_scale(patch: PatchType, /, factor: float = 2.0) -> PatchType:
        """Scale the patch."""
        return SeamScale(factor=factor).run(patch)

    def kernel(self, data):
        """Scale every sample."""
        return data * self.factor


class SeamSum(PatchProcessor):
    """Sum along a dimension, keeping it with length one."""

    dim: str = "time"

    @staticmethod
    def seam_sum(patch: PatchType, /, dim: str = "time") -> PatchType:
        """Sum along a dimension."""
        return SeamSum(dim=dim).run(patch)

    def get_metadata(self, meta):
        """The summed dimension keeps one sample; say which axis."""
        coord = meta.get_coord(self.dim)
        out = meta.new(coords=meta.coords.update(**{self.dim: coord[:1]}))
        return out, {"axis": meta.get_axis(self.dim)}

    def kernel(self, data, *, axis):
        """Sum along it."""
        return data.sum(axis=axis, keepdims=True)


class SeamExtras(PatchProcessor):
    """Take a positional group and keyword extras."""

    flag: bool = False
    model_config = ConfigDict(extra="allow", frozen=True)

    @staticmethod
    def seam_extras(
        patch: PatchMetaType, /, *names: str, flag: bool = False, **kwargs
    ) -> PatchMetaType:
        """Take the names positionally and anything else as an extra."""
        named = {**dict.fromkeys(names, True), **kwargs}
        return SeamExtras(flag=flag, **named).run(patch)


def _seam_aliased_impl(patch, /, factor: float = 2.0):
    """Scale, under a name the class does not register."""
    return SeamAliased(factor=factor).run(patch)


class SeamAliased(SeamScale):
    """Declare a doorway whose function is named something else."""

    seam_aliased = staticmethod(_seam_aliased_impl)


class SeamHidden(PatchProcessor):
    """A processor with no patch function."""

    name = None


class SeamNeedsVelocity(PatchProcessor):
    """Declare requirements, and change the data_type."""

    required_dims = ("time",)
    required_attrs: ClassVar = {"data_type": "velocity"}
    data_type = "acceleration"

    @staticmethod
    def seam_needs_velocity(patch: PatchType, /) -> PatchType:
        """Copy the data of a velocity patch."""
        return SeamNeedsVelocity().run(patch)

    def kernel(self, data):
        """Return a copy."""
        return data + 0


def _host_method(host, name, monkeypatch, run=None):
    """Write a real patch method into a host, as its class body would."""

    def method(self) -> Self:
        """Run the operation, or hand back what it was given."""
        return self if run is None else run(self)

    method.__name__ = name
    method.__qualname__ = f"{host.__name__}.{name}"
    monkeypatch.setattr(host, name, method, raising=False)
    return method


@pytest.fixture(scope="module")
def patch():
    """A patch whose every cell differs, so a wrong axis cannot pass."""
    base = dc.get_example_patch()
    values = np.arange(base.size, dtype=np.float64).reshape(base.shape)
    return base.new(data=values * values[::-1])


class TestTheSeam:
    """What `run` does for a subclass which writes only a kernel."""

    def test_it_is_callable(self, patch):
        """The processor and its generated function give the same patch."""
        out = SeamScale(3)(patch)
        assert np.allclose(out.data, patch.data * 3)
        assert out.equals(SeamScale.patch_function(patch, factor=3))

    def test_it_writes_history(self, patch):
        """The call is spelled from the validated fields."""
        out = SeamScale(3)(patch)
        assert out.attrs.history[-1] == "seam_scale(factor='3.0')"

    def test_history_by_method_name(self, patch):
        """`history = "method_name"` records the name alone."""

        class Named(SeamScale):
            """Record only the name."""

            history = "method_name"

            @staticmethod
            def named(patch: PatchType, /, factor: float = 2.0) -> PatchType:
                """Scale, recording the name alone."""
                return Named(factor=factor).run(patch)

        assert Named()(patch).attrs.history[-1] == "named"

    def test_it_advances_the_ids(self, patch):
        """Which data stays; what was done moves."""
        out = SeamScale(3)(patch)
        assert out.attrs.origin_id == patch.attrs.origin_id
        assert out.attrs.data_id != patch.attrs.data_id

    def test_the_method_and_the_class_stamp_alike(self, patch):
        """One route, so one history and one id."""
        by_method = patch.normalize("time")
        by_class = Normalize(dim="time")(patch)
        assert by_method.attrs.history == by_class.attrs.history
        assert by_method.attrs.data_id == by_class.attrs.data_id

    def test_get_metadata(self, patch):
        """A shape change comes back with the axis the kernel gets."""
        out = SeamSum("time")(patch)
        axis = patch.get_axis("time")
        assert out.shape[axis] == 1
        assert np.allclose(out.data, patch.data.sum(axis=axis, keepdims=True))

    def test_reconcile_sees_the_result(self, patch):
        """The one hook after the kernel; not reached on a no-op."""

        class Tagged(SeamScale):
            """Tag the result with its dtype."""

            name = None

            def reconcile(self, data, out):
                """Record what the kernel produced."""
                return out.update_attrs(station=str(data.dtype))

        assert Tagged()(patch).attrs.station == "float64"
        assert Tagged(factor=1)(patch).attrs.station == "float64"

        class Untouched(SeamHidden):
            """A no-op which would tag the result."""

            name = None

            def reconcile(self, data, out):
                """Would fail the test if reached."""
                raise AssertionError

        assert Untouched()(patch) is patch

    def test_reconcile_runs_without_a_kernel(self, patch):
        """Metadata changed, so the hook which sees both halves still runs."""

        class Tagging(PatchProcessor):
            """Rename a coordinate and tag what the data turned out to be."""

            name = None

            def get_metadata(self, meta):
                """Rename the distance coordinate, so something changed."""
                return meta.rename_coords(distance="depth"), {}

            def reconcile(self, data, out):
                """Record what the data were, which only this hook sees."""
                seen = "none" if data is None else str(data.dtype)
                return out.update_attrs(station=seen)

        assert Tagging()(patch).attrs.station == str(patch.data.dtype)
        # Metadata has no data to show it, and says so rather than lying.
        assert Tagging()(patch.drop_data()).attrs.station == "none"

    def test_a_no_op_hands_back_the_patch(self, patch):
        """Nothing changed, so nothing is recorded."""
        assert dc.proc.transpose(patch, *patch.dims) is patch
        assert SeamHidden()(patch) is patch

    def test_a_patch_subclass_survives(self, patch):
        """An operation on a subclass gives back that subclass."""

        class _Sub(dc.Patch):
            """A patch which is more than a patch."""

        sub = _Sub(data=patch.data, coords=patch.coords)
        assert type(sub.abs()) is _Sub
        assert type(sub.transpose()) is _Sub
        assert type(sub.rename_coords(distance="depth")) is _Sub


class TestReviewFindings:
    """Edges the counterpart review found."""

    def test_two_classes_one_name_collide(self):
        """A second class taking a DASCore name is refused, not swapped in."""
        with pytest.raises(ParameterError, match="claim the tag"):

            class Other(PatchProcessor):
                """Claim `normalize` from another class."""

                name = "normalize"
                __module__ = "dascore.proc.basic"
                __qualname__ = "Other"

                def kernel(self, data):
                    """Stand in for the operation whose name this claims."""
                    return data

        # Refused before anything was stamped, so the method is untouched.
        assert dc.Patch.normalize.__processor__ is Normalize
        assert resolve_patch_function("normalize") is Normalize.patch_function

    @pytest.mark.parametrize("name", ["patch", "self"])
    def test_a_coordinate_named_for_the_receiver(self, name):
        """`self` is positional-only, so a coordinate may be named for it."""
        patch = dc.Patch(data=np.arange(3.0), coords={name: np.arange(3)}, dims=(name,))
        assert patch.rename_coords(**{name: "renamed"}).dims == ("renamed",)
        assert patch.select(**{name: (0, 1), "samples": True}).shape == (1,)
        assert patch.drop_data().update_coords(**{name: [3, 4, 5]}).dims == (name,)

    def test_a_narrow_subclass(self, patch):
        """A subclass whose `__init__` takes no dtype still transposes."""

        class _Sub(dc.Patch):
            def __init__(self, data=None, coords=None, dims=None, attrs=None):
                super().__init__(data=data, coords=coords, dims=dims, attrs=attrs)

        sub = _Sub(patch.data, coords=patch.coords)
        assert type(sub.transpose("time", "distance")) is _Sub
        assert type(sub.rename_coords(distance="depth")) is _Sub

    def test_a_required_field_after_a_default(self, patch):
        """A subclass may add a required field beside an inherited default."""

        class Offset(SeamScale):
            """Scale, then add."""

            name = None
            offset: float

            def kernel(self, data):
                """Scale and add."""
                return data * self.factor + self.offset

        assert np.allclose(Offset(offset=1)(patch).data, patch.data * 2 + 1)
        with pytest.raises(ValidationError):
            Offset()

    def test_a_no_op_still_sets_data_type(self, patch):
        """As a decorated function does, and without recording the call."""

        class Clear(PatchProcessor):
            """Change nothing but the data_type."""

            name = None
            data_type = ""

        tagged = patch.update_attrs(data_type="velocity")
        out = Clear()(tagged)
        assert out.attrs.data_type == ""
        assert out.attrs.history == tagged.attrs.history

    def test_func_skips_the_record(self, patch):
        """`.func` and `raw_function` run the operation without history or ids."""
        for bypass in (dc.proc.normalize.func, dc.proc.normalize.raw_function):
            out = bypass(patch, "time")
            assert out.equals(patch.normalize("time"))
            assert out.attrs.history == patch.attrs.history
            # Nothing was recorded, so nothing names the array it made.
            assert out.attrs.origin_id == patch.attrs.origin_id
            assert out.attrs.data_id not in ("", patch.attrs.data_id)

    def test_signature_carries_annotations(self):
        """The method's own annotations, which static tooling can read."""
        sig = inspect.signature(dc.proc.normalize)
        assert sig.parameters["dim"].annotation == "str"
        assert sig.return_annotation == "Self"
        assert "__signature__" not in vars(dc.proc.normalize)

    def test_bypasses_are_distinct(self):
        """Two bypasses given as arguments are two callables."""
        abs_raw, imag_raw = dc.proc.abs.raw_function, dc.proc.imag.raw_function
        assert encode(abs_raw) != encode(imag_raw)

    def test_the_old_seam_is_refused(self):
        """`derive`/`plan` became `get_metadata`; a stale override must fail."""
        with pytest.raises(ParameterError, match="get_metadata"):

            class Derived(PatchProcessor):
                """A subclass written against the seam before the rename."""

                name = None

                def derive(self, patch):
                    """What `get_metadata` now returns half of."""
                    return patch

        with pytest.raises(ParameterError, match="get_metadata"):

            class Planned(PatchProcessor):
                """The other half of the old seam."""

                name = None

                def plan(self, patch, out):
                    """What `get_metadata` now returns the other half of."""
                    return {}

    def test_a_class_without_its_method_is_refused(self):
        """A named class declares the method; it never borrows a parent's."""
        with pytest.raises(ParameterError, match="must declare"):

            class Borrower(SeamScale):
                """Registered, with no method of its own."""

    def test_the_operation_is_an_ordinary_method(self, patch):
        """Written in the class body, so it binds and takes the patch.

        Nothing is attached at import: `Patch.abs` is the function the
        class body defines, and the framework only checks it and stamps
        what the docs and the registry read.
        """
        assert isinstance(vars(dc.Patch)["abs"], FunctionType)
        assert Abs.patch_function is vars(dc.Patch)["abs"]
        assert patch.abs().equals(dc.Patch.abs(patch))

    def test_a_name_must_be_an_identifier(self):
        """A name the registry cannot tag is refused, not silently skipped."""
        with pytest.raises(ParameterError, match="identifier"):

            class Hyphen(PatchProcessor):
                """Claim a name no function can have."""

                name = "scale-op"

    def test_the_tag_is_the_registered_name(self, patch):
        """A doorway named otherwise still registers under `name`."""
        tag = patch_function_tag(SeamAliased.patch_function)
        assert tag.endswith(":seam_aliased")
        assert resolve_patch_function(tag) is SeamAliased.patch_function

    @pytest.mark.concurrency
    def test_a_module_reloaded_replaces_its_own_entry(self):
        """`%autoreload` redefines a class; that is not two claiming one."""
        # In a process of its own: a reload rebinds what every later test
        # in this one would go on using. Marked so the WebAssembly suite,
        # which has no processes to spawn, deselects it.
        script = (
            "import importlib, dascore as dc\n"
            "importlib.reload(importlib.import_module('dascore.proc.basic'))\n"
            "assert dc.get_example_patch().abs() is not None\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        assert done.returncode == 0, done.stderr

    def test_generated_functions_pickle(self, patch):
        """So a process pool can run them."""
        assert pickle.loads(pickle.dumps(dc.proc.demean)) is dc.proc.demean
        func = SeamScale.patch_function
        assert pickle.loads(pickle.dumps(func)) is func

    def test_bypasses_pickle(self):
        """The bypass is an attribute of the method, which names it."""
        raw = dc.proc.demean.raw_function
        assert pickle.loads(pickle.dumps(raw)) is raw

    def test_a_bad_call_is_a_type_error(self, patch):
        """As it was for a plain function: bound against the signature."""
        with pytest.raises(TypeError, match="dimm"):
            patch.demean(dimm="time")
        with pytest.raises(TypeError, match="dim"):
            patch.standardize()

    def test_a_field_may_not_shadow_the_base(self):
        """A field named for a base setting would corrupt it silently."""
        with pytest.raises(ParameterError, match="shadow"):

            class Shadow(PatchProcessor):
                """Name a field `history`."""

                history: str = "x"

    def test_a_factory_default_must_match_the_method(self):
        """The guard reads what the factory makes, not pydantic's marker."""

        class Factory(PatchProcessor):
            """Default a field through a factory the method spells out."""

            values: tuple = Field(default_factory=lambda: (1, 2))

            @staticmethod
            def factory(
                patch: PatchMetaType, /, values: tuple = (1, 2)
            ) -> PatchMetaType:
                """Forward the values the factory would have made."""
                return Factory(values=values).run(patch)

        try:
            assert Factory().values == (1, 2)
        finally:
            for host in (dc.Patch, dc.PatchMeta):
                if "factory" in vars(host):
                    delattr(host, "factory")

    def test_a_processor_as_a_parameter(self, patch):
        """It is its operation id, unless it holds a patch."""
        from dascore.utils.identity import encode as _encode  # noqa: PLC0415

        first, other = SeamScale(factor=2.0), SeamScale(factor=3.0)
        assert _encode(first) == {"$id": ["operation", first.operation_id]}
        assert _encode(first) != _encode(other)

        class Holder(SeamScale):
            """Hold a patch as a field."""

            name = None
            held: Any = None

        with pytest.raises(ParameterError, match="holds a patch"):
            _encode(Holder(held=patch))

    def test_equality_with_a_field_nothing_spells(self):
        """It compares and hashes, but only as itself."""

        class Odd(SeamScale):
            """Hold a value the encoder refuses."""

            name = None
            thing: Any = None

        first, second = Odd(thing=object()), Odd(thing=object())
        assert first == first and first != second
        assert len({first, second}) == 2

    def test_a_refused_operation_keeps_nested_origins(self, patch):
        """A patch held in a list still says where the data came from."""
        from dascore.utils.identity import fold_origin_ids  # noqa: PLC0415

        class Holding(SeamScale):
            """Hold patches in a list, and something nothing spells."""

            name = None
            items: Any = None
            odd: Any = None

        other = dc.get_example_patch()
        with pytest.warns(dc.warnings.DASCoreWarning, match="random data_id"):
            out = Holding(items=[other], odd=object())(patch)
        expected = fold_origin_ids([patch.attrs.origin_id, other.attrs.origin_id])
        assert out.attrs.origin_id == expected

    def test_a_class_with_no_source(self, patch):
        """Made from text, it is named by which class object it is."""
        made = type("Made", (SeamScale,), {"name": None, "__doc__": "Made up."})
        assert "#" in made().tag
        assert made()(patch).attrs.data_id == made()(patch).attrs.data_id

    def test_an_unencodable_field_gets_a_random_id(self, patch):
        """A field the encoder refuses costs a derived id, not the call."""

        class Dated(SeamScale):
            """Carry a time outside the nanosecond range."""

            name = None
            when: Any = np.datetime64("3000-01-01")

        # Not the default, which is left out of the id unread.
        late = np.datetime64("3000-01-02")
        out = Dated(when=late)(patch)
        assert np.allclose(out.data, patch.data * 2)
        # New data never keeps its input's id, and no two runs share one.
        assert out.attrs.data_id not in ("", patch.attrs.data_id)
        assert out.attrs.data_id != Dated(when=late)(patch).attrs.data_id
        assert out.attrs.origin_id == patch.attrs.origin_id

    def test_a_numpy_bool_in_a_plan(self, patch):
        """Numpy scalars are numbers too."""

        class Flag(SeamScale):
            """Plan a numpy bool."""

            name = None

            def get_metadata(self, meta):
                """Return one."""
                return meta, {"flag": np.bool_(True)}

            def kernel(self, data, *, flag):
                """Use it."""
                return data * self.factor if flag else data

        assert np.allclose(Flag()(patch).data, patch.data * 2)


class TestCheck:
    """`run` refuses what the class declares it cannot take."""

    def test_required_dims(self, patch):
        """A missing dimension is refused."""
        with pytest.raises(PatchCoordinateError):
            SeamNeedsVelocity()(patch.rename_coords(time="t"))

    def test_required_attrs(self, patch):
        """A missing attr value is refused; a present one sets data_type."""
        with pytest.raises(PatchAttributeError):
            SeamNeedsVelocity()(patch)
        out = SeamNeedsVelocity()(patch.update_attrs(data_type="velocity"))
        assert out.attrs.data_type == "acceleration"

    def test_a_backend_only_kernel_still_needs_data(self, patch):
        """`check` asks the class, not this patch's backend."""

        class Elsewhere(PatchProcessor):
            """Its only kernel is for a backend nothing here runs."""

            name = None

        @register_kernel(Elsewhere, "cupy")
        def _elsewhere(processor, data):
            """Never called from these tests."""
            raise AssertionError

        with pytest.raises(PatchDataError):
            Elsewhere()(patch.drop_data())

    def test_no_data(self, patch):
        """An operation which computes data cannot run on metadata alone."""
        with pytest.raises(PatchDataError):
            SeamScale()(patch.drop_data())

    def test_a_kernel_less_operation_runs_on_metadata(self, patch):
        """It never asks for data, so metadata is all it needs."""
        meta = patch.drop_data()
        out = dc.proc.rename_coords(meta, distance="depth")
        assert isinstance(out, dc.PatchMeta) and not isinstance(out, dc.Patch)
        assert out.dims == tuple("depth" if x == "distance" else x for x in patch.dims)

    def test_a_kernel_less_operation_keeps_the_data(self, patch):
        """Applied to a patch it changes the metadata, not the array."""
        out = patch.rename_coords(distance="depth")
        assert out.data is patch.data


class TestSignatureDrift:
    """The method's parameters and the class's fields are one declaration."""

    def test_a_field_the_method_cannot_take_is_refused(self):
        """A field no call can reach is drift worth refusing."""
        with pytest.raises(ParameterError, match="which the class declares"):

            class Missing(PatchProcessor):
                """Declare a field the method leaves out."""

                factor: float = 2.0

                @staticmethod
                def missing(patch: PatchMetaType, /) -> PatchMetaType:
                    """Take nothing, though the class stores a factor."""
                    return patch

    def test_a_parameter_which_is_no_field_is_allowed(self, patch):
        """A body may resolve something before the instance is built."""

        class Resolving(PatchProcessor):
            """Take a spelling of the factor which it does not store."""

            name = None
            factor: float = 2.0

            @staticmethod
            def resolving(patch, /, factor: float = 2.0, double: bool = False):
                """Double the factor before building the instance."""
                return Resolving(factor=factor * (2 if double else 1)).run(patch)

        assert Resolving.resolving(patch, 3, double=True) is not None

    def test_a_field_a_var_parameter_collects(self, patch):
        """`*args` and `**kwargs` gather rather than default, so no default
        of theirs is compared against the field's.
        """

        class Collected(PatchProcessor):
            """Hold the positional group in a field of its own."""

            names: tuple[str, ...] = ()

            @staticmethod
            def collected(patch: PatchMetaType, /, *names: str) -> PatchMetaType:
                """Take the names positionally."""
                return Collected(names=names).run(patch)

        assert Collected.collected(patch, "a", "b") is patch

    def test_a_method_naming_a_class_is_refused(self, monkeypatch):
        """`run` gives back the kind it was given, which only `Self` says."""
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        def seam_flattening(self) -> dc.Patch:
            """Name a class, which a subclass would come back as."""
            return self

        monkeypatch.setattr(dc.Patch, "seam_flattening", seam_flattening, raising=False)

        class SeamFlattening(PatchProcessor):
            """One of DASCore's own, whose method names a class."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamFlattening"

            def kernel(self, data):
                """Compute something, so the method belongs to Patch."""
                return data

        monkeypatch.setattr(processor_module, "_HOSTS", [dc.Patch, dc.PatchMeta])
        with pytest.raises(ParameterError, match="only `-> Self` says"):
            processor_module._check_patch_listing(SeamFlattening)

    def test_extras_a_method_cannot_take_are_refused(self):
        """A class which allows extras needs somewhere to spell them."""
        with pytest.raises(ParameterError, match="takes no `\\*\\*kwargs`"):

            class Extraneous(PatchProcessor):
                """Allow extras the method gives no way to pass."""

                model_config = ConfigDict(extra="allow", frozen=True)

                @staticmethod
                def extraneous(patch, /):
                    """Take nothing, though the class takes anything."""
                    return patch

    def test_a_field_the_class_refuses_by_position_is_refused(self):
        """`_positional_fields` is part of the same one declaration."""
        with pytest.raises(ParameterError, match="`_positional_fields`"):

            class Placed(PatchProcessor):
                """Refuse the factor by position, and take it that way."""

                factor: float = 2.0
                _positional_fields = ()

                @staticmethod
                def placed(patch, /, factor: float = 2.0):
                    """Take positionally what the class takes by name only."""
                    return Placed(factor=factor).run(patch)

    def test_a_field_the_body_drops_is_refused(self):
        """A parameter which never reaches the class is silently ignored."""
        with pytest.raises(ParameterError, match="does not pass them to"):

            class Dropping(PatchProcessor):
                """Take the factor and forget to forward it."""

                factor: float = 2.0

                @staticmethod
                def dropping(patch, /, factor: float = 2.0):
                    """Build the class without the value it was given."""
                    return Dropping().run(patch)

    def test_a_body_with_no_source_is_taken_on_trust(self, patch):
        """A function built at runtime has nothing to read."""
        namespace = {"PatchProcessor": PatchProcessor, "__name__": "seamless"}
        exec(
            "class Sourceless(PatchProcessor):\n"
            "    'Built where no source can be read.'\n"
            "    factor: float = 2.0\n"
            "    @staticmethod\n"
            "    def sourceless(patch, /, factor=2.0):\n"
            "        'Forward nothing, unreadably.'\n"
            "        return patch\n",
            namespace,
        )
        assert namespace["Sourceless"].patch_function is not None

    def test_a_differing_default_is_refused(self):
        """Unset must mean the same on both sides."""
        with pytest.raises(ParameterError, match="must agree on what unset"):

            class Disagreeing(PatchProcessor):
                """Default the same name two ways."""

                factor: float = 2.0

                @staticmethod
                def disagreeing(
                    patch: PatchMetaType, /, factor: float = 3.0
                ) -> PatchMetaType:
                    """Default the factor to something else."""
                    return Disagreeing(factor=factor).run(patch)

    def test_an_array_default(self, patch):
        """A default may be an array, which compares as a whole."""

        class Windowed(PatchProcessor):
            """Default a window the same way on both sides."""

            # A NaN equals nothing, itself included, so an array of them
            # agrees with another only where nulls are compared as nulls.
            window: ArrayLike = np.array([np.nan, 1.0])

            @staticmethod
            def windowed(
                patch: PatchMetaType, /, window=np.array([np.nan, 1.0])
            ) -> PatchMetaType:
                """Take the window the class stores."""
                return Windowed(window=window).run(patch)

        assert Windowed.windowed(patch) is not None
        with pytest.raises(ParameterError, match="must agree on what unset"):

            class Mismatched(PatchProcessor):
                """Default the window two ways."""

                window: ArrayLike = np.ones(3)

                @staticmethod
                def mismatched(
                    patch: PatchMetaType, /, window=np.zeros(3)
                ) -> PatchMetaType:
                    """Default the window to something else."""
                    return Mismatched(window=window).run(patch)


class TestPlan:
    """A plan holds numbers, indices, or numeric arrays only."""

    @pytest.mark.parametrize(
        "value",
        ["time", {1: 2}, np.array(["a"]), object()],
        ids=["string", "dict", "string_array", "object"],
    )
    def test_refused(self, patch, value):
        """Anything else is refused before the kernel runs."""

        class Bad(SeamScale):
            """Plan something a kernel may not take."""

            name = None

            def get_metadata(self, meta):
                """Return the value under test."""
                return meta, {"value": value}

        with pytest.raises(ParameterError, match="plan may hold only"):
            Bad()(patch)

    def test_a_patch_or_coords_are_refused(self, patch):
        """The kernel never sees a patch or a coord manager."""
        for value in (patch, patch.coords):

            class Bad(SeamScale):
                """Plan a patch."""

                name = None

                def get_metadata(self, meta, value=value):
                    """Return the value under test."""
                    return meta, {"value": value}

            with pytest.raises(ParameterError, match="plan may hold only"):
                Bad()(patch)

    def test_allowed(self, patch):
        """Numbers, indices, nested sequences and numeric arrays pass."""

        class Good(PatchProcessor):
            """Plan every allowed kind."""

            name = None

            def get_metadata(self, meta):
                """Return one of each."""
                return meta, {
                    "a": 1,
                    "b": 2.0,
                    "c": True,
                    "d": ((1, 2), 3.0),
                    "e": np.arange(3),
                    "f": slice(1, None, 2),
                    "g": (slice(None), Ellipsis, [1, 2]),
                    "h": None,
                }

            def kernel(self, data, **plan):
                """Ignore the plan, keep the data."""
                return data + 0

        assert Good()(patch).equals(patch)


class TestTransposeMetadata:
    """Transpose works out a new shape from metadata alone."""

    def test_metadata_without_data(self, patch):
        """Coords with another shape, and no data read."""
        processor = dc.proc.Transpose(dims=("time", "distance"))
        out, _ = processor.get_metadata(patch.drop_data())
        assert out.dims == ("time", "distance")
        assert out.shape == patch.shape[::-1]

    def test_the_kernel_permutes(self, patch):
        """The data follow the coords."""
        out = patch.transpose("time", "distance")
        assert np.array_equal(out.data, np.asarray(patch.data).T)


class TestOperationId:
    """The identity of an operation is its validated fields."""

    def test_validated_fields(self):
        """An int and the float it validates to are one operation."""
        assert SeamScale(4).operation_id == SeamScale(4.0).operation_id
        assert SeamScale(4).operation_id == SeamScale(factor=4).operation_id
        assert SeamScale(4).operation_id != SeamScale(5).operation_id

    def test_equal_by_operation(self):
        """Positional and keyword spellings are one processor."""
        assert Normalize("x") == Normalize(dim="x")
        assert hash(Normalize("x")) == hash(Normalize(dim="x"))
        assert hash(Normalize("x")) != hash(Normalize("y"))
        assert Normalize("x") != Normalize("y")
        # A default spelled out is the same operation.
        assert Normalize("x") == Normalize("x", norm="l2")
        assert Normalize("x") != "normalize"

    def test_another_class_is_another_operation(self):
        """Even one with the same fields."""

        class Other(SeamScale):
            """A subclass."""

            name = None

        assert Other() != SeamScale()
        assert Other().operation_id != SeamScale().operation_id

    def test_the_version_counts(self, monkeypatch):
        """A bump makes a new operation."""
        before = SeamScale().operation_id
        monkeypatch.setattr(SeamScale, "__version__", "2.0")
        assert SeamScale().operation_id != before


class TestGeneratedFunction:
    """What subclassing generates."""

    def test_signature_and_defaults(self):
        """The fields in declaration order, with their defaults."""
        sig = inspect.signature(dc.proc.normalize)
        assert list(sig.parameters) == ["self", "dim", "norm", "window", "samples"]
        assert sig.parameters["norm"].default == "l2"
        assert sig.parameters["dim"].default is inspect.Parameter.empty

    def test_star_args_and_extras(self, patch):
        """A real `*args` and `**kwargs`, which the method itself declares."""
        sig = inspect.signature(SeamExtras.patch_function)
        kinds = [x.kind for x in sig.parameters.values()]
        assert kinds[1] == inspect.Parameter.VAR_POSITIONAL
        assert kinds[-1] == inspect.Parameter.VAR_KEYWORD

    def test_a_positional_group_and_a_keyword_of_its_name(self, patch):
        """Python never fills `*args` by keyword, so the real signature parts
        what a synthesized one merged.
        """
        seen = []

        class Grouped(SeamExtras):
            """Record what each spelling was built with."""

            name = None

            def get_metadata(self, meta):
                """Record the extras this call was built with."""
                seen.append(dict(self.model_extra or {}))
                return meta, {}

            @staticmethod
            def grouped(
                patch: PatchMetaType, /, *names: str, **kwargs
            ) -> PatchMetaType:
                """Take the names positionally and anything else as an extra."""
                return Grouped(**{**dict.fromkeys(names, True), **kwargs}).run(patch)

        Grouped.grouped(patch, "a", "b")
        assert seen[-1] == {"a": True, "b": True}
        # A caller naming a coordinate after the group reaches the extras,
        # which no field can intercept.
        Grouped.grouped(patch, names=[1, 2])
        assert seen[-1] == {"names": [1, 2]}

    def test_a_field_given_twice_is_a_type_error(self):
        """Positionally and by name is the same mistake a function makes."""
        with pytest.raises(TypeError, match="repeated argument"):
            SeamScale(3, factor=4)

    def test_too_many_positional_arguments(self):
        """More values than there are fields to take them."""
        with pytest.raises(TypeError, match="positional argument"):
            SeamScale(3, 4)

    def test_positional_fields(self):
        """Fields outside `_positional_fields` can only be given by name."""

        class Named(SeamScale):
            """Take the factor by name only."""

            name = None
            _positional_fields = ()

        with pytest.raises(TypeError):
            Named(3)
        assert Named(factor=3).factor == 3

    def test_names_and_docs(self):
        """Named for the operation, documented by the class."""
        func = Normalize.patch_function
        assert func.__name__ == "normalize"
        # Where the method is written, which is the class it is a method of.
        assert func.__module__ == "dascore.core.patch"
        assert func.__doc__ == Normalize.__doc__
        assert "{sample_explanation}" not in func.__doc__
        assert func.__processor__ is Normalize
        assert dc.proc.normalize is func is dc.Patch.normalize

    def test_a_late_class_composes_its_docstring(self):
        """A class made after `Patch` exists has its method stamped first.

        `compose_docstring` substitutes the class's placeholders, but by
        then the method already holds the copy made at class creation, so
        the copy is made again. In-tree classes are all created while
        `Patch` is still being built, which is why nothing in DASCore's own
        import reaches this; a plugin's class does.
        """

        @compose_docstring(note="substituted")
        class SeamDocumented(PatchProcessor):
            """Hand the patch back.

            {note}
            """

            @staticmethod
            def seam_documented(patch, /):
                """Hand the patch back."""
                return patch

        assert SeamDocumented.patch_function.__doc__ == SeamDocumented.__doc__
        assert "{note}" not in SeamDocumented.patch_function.__doc__

    def test_registered_by_tag(self):
        """DASCore's own are bare; a test module's are namespaced."""
        assert resolve_patch_function("normalize") is Normalize.patch_function
        tag = patch_function_tag(SeamScale.patch_function)
        assert tag.endswith(":seam_scale")
        assert resolve_patch_function(tag) is SeamScale.patch_function

    def test_unnamed_is_not_registered(self):
        """`name = None` gets no function and no tag."""
        assert SeamHidden.patch_function is None
        tag = patch_function_tag(SeamScale.patch_function).replace("scale", "hidden")
        with pytest.raises(ParameterError):
            resolve_patch_function(tag)

    def test_snake_case_name(self):
        """The default name is the class name in snake case."""
        assert dc.proc.RenameCoords.name == "rename_coords"


class TestKernelFor:
    """Which kernel runs, and for which backend."""

    def test_a_registered_kernel_wins_for_its_backend(self, patch):
        """And `run` uses it."""

        class Doubling(PatchProcessor):
            """A processor whose generic kernel does nothing."""

            name = None

            def kernel(self, data):
                """Hand the data back unchanged."""
                return data

        @register_kernel(Doubling, "numpy")
        def _double(processor, data):
            """Double it, so the two can be told apart."""
            return data * 2

        assert Doubling.kernel_for("numpy") is _double
        assert np.array_equal(Doubling()(patch).data, np.asarray(patch.data) * 2)

    def test_another_backend_falls_back_to_the_generic(self):
        """A kernel written to the standard runs wherever it is asked to."""

        class Local(SeamScale):
            """A class to register against, so nothing leaks."""

            name = None

        @register_kernel(Local, "cupy")
        def _never(processor, data):
            """Registered for a backend nothing here uses."""
            raise AssertionError

        assert Local.kernel_for("numpy") is SeamScale.__dict__["kernel"]

    def test_registering_on_a_child_leaves_the_parent(self):
        """A child's kernel is the child's alone."""

        class Parent(SeamScale):
            """The parent."""

            name = None

        class Child(Parent):
            """The child."""

            name = None

        @register_kernel(Child, "cupy")
        def _child(processor, data):
            """The child's cupy kernel."""
            return data

        assert Child.kernel_for("cupy") is _child
        assert Parent.kernel_for("cupy") is SeamScale.__dict__["kernel"]

    def test_a_subclass_kernel_beats_a_parents_backend_kernel(self):
        """A parent's backend kernel must not answer for a subclass."""

        class Parent(PatchProcessor):
            """The class the kernel is registered against."""

            name = None

        @register_kernel(Parent, "numpy")
        def _parent_kernel(processor, data):
            """What the parent does on numpy."""
            return data

        class Child(Parent):
            """A subclass which computes something else entirely."""

            name = None

            def kernel(self, data):
                """The child's own arithmetic."""
                return data

        assert Parent.kernel_for("numpy") is _parent_kernel
        assert Child.kernel_for("numpy") is Child.__dict__["kernel"]

    def test_no_kernel_at_all_is_metadata_only(self):
        """A processor which touches no data says so by defining none."""
        assert SeamHidden.kernel_for("numpy") is None


class TestWhereOperationsAreListed:
    """Both classes write their operations down; the framework checks."""

    def test_metadata_only_is_listed_on_patch_meta(self):
        """Kernel-less, so metadata carries it and `Patch` inherits it."""
        func = dc.proc.RenameCoords.patch_function
        assert vars(dc.PatchMeta)["rename_coords"] is func
        assert "rename_coords" not in vars(dc.Patch)
        assert dc.Patch.rename_coords is func

    def test_a_kernel_is_listed_on_patch(self):
        """Written down, so a reader and a type checker see the whole class."""
        assert vars(dc.Patch)["abs"] is Abs.patch_function
        assert "abs" not in vars(dc.PatchMeta)

    def test_an_unlisted_operation_is_refused(self, monkeypatch):
        """The line nobody can forget silently: import fails and says which."""
        # What the tree looks like mid-import, which is when the check runs.
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        class SeamForgotten(PatchProcessor):
            """One of DASCore's own, with a kernel and no method."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamForgotten"

            def kernel(self, data):
                """Compute something, so Patch has to define it."""
                return data

        assert processor_module._UNCHECKED == [SeamForgotten]
        with pytest.raises(ParameterError, match=r"no 'seam_forgotten' method"):
            processor_module.check_patch_listings(dc.Patch, dc.PatchMeta)

    def test_an_operation_listed_on_the_wrong_class_is_refused(self, monkeypatch):
        """Both directions: the listing says which class, and it must agree."""
        hosts = list(processor_module._HOSTS)
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        class SeamMisplaced(PatchProcessor):
            """Kernel-less, so metadata's, written into Patch instead."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamMisplaced"

        class SeamComputing(PatchProcessor):
            """Kerneled, so Patch's, written into PatchMeta instead."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamComputing"

            def kernel(self, data):
                """Compute something."""
                return data

        _host_method(dc.Patch, "seam_misplaced", monkeypatch)
        _host_method(dc.PatchMeta, "seam_computing", monkeypatch)
        monkeypatch.setattr(processor_module, "_HOSTS", hosts)
        with pytest.raises(ParameterError, match="Patch defines 'seam_misplaced'"):
            processor_module._check_patch_listing(SeamMisplaced)
        with pytest.raises(ParameterError, match="PatchMeta defines 'seam_computing'"):
            processor_module._check_patch_listing(SeamComputing)

    def test_an_out_of_tree_class_is_listed_nowhere(self):
        """A plugin reaches its operation through `dc.proc` and the registry."""
        assert SeamExtras.kernel_for("numpy") is None
        assert "seam_extras" not in vars(dc.Patch)
        assert "seam_extras" not in vars(dc.PatchMeta)
        assert SeamScale.kernel_for("numpy") is not None
        assert "seam_scale" not in vars(dc.Patch)

    def test_a_kernel_registered_before_the_hosts_exist(self, monkeypatch):
        """Mid-import there is nothing to check against; the drain settles it."""
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        class SeamEarly(PatchProcessor):
            """Created before `Patch` exists, as DASCore's own are."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamEarly"

        assert processor_module._UNCHECKED == [SeamEarly]

        @register_kernel(SeamEarly, "numpy")
        def _early(processor, data):
            """Registered while nothing is listed, so nothing is checked."""
            return data

        assert processor_module._UNCHECKED == [SeamEarly]

    def test_a_kernel_registered_later_is_refused(self, patch, monkeypatch):
        """A listing must move between class bodies, which a process cannot do."""
        hosts = list(processor_module._HOSTS)
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        class SeamLate(PatchProcessor):
            """One of DASCore's own, whose method metadata holds."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamLate"

        # What PatchMeta's class body would say, were this real.
        _host_method(dc.PatchMeta, "seam_late", monkeypatch, run=SeamLate().run)
        monkeypatch.setattr(processor_module, "_HOSTS", hosts)
        with pytest.raises(ParameterError, match="must move from"):

            @register_kernel(SeamLate, "numpy")
            def _late(processor, data):
                """Never reached; the check refuses the registration."""
                return data * 2

        # Refused before anything moved, so it still answers.
        assert patch.drop_data().seam_late() is not None
        assert SeamLate.kernel_for("numpy") is None

    def test_a_refused_registration_leaves_the_kernels_alone(self, monkeypatch):
        """A class which already had one keeps exactly the ones it had."""
        hosts = list(processor_module._HOSTS)
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        class SeamPartial(PatchProcessor):
            """One of DASCore's own, whose method metadata holds."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamPartial"

        @register_kernel(SeamPartial, "cupy")
        def _cupy(processor, data):
            """Registered while nothing is listed, so nothing is checked."""
            return data

        before = dict(SeamPartial.__dict__["_kernels"])
        _host_method(dc.PatchMeta, "seam_partial", monkeypatch)
        monkeypatch.setattr(processor_module, "_HOSTS", hosts)
        with pytest.raises(ParameterError, match="must move from"):

            @register_kernel(SeamPartial, "numpy")
            def _numpy(processor, data):
                """Never reached; the check refuses the registration."""
                return data

        assert SeamPartial.__dict__["_kernels"] == before

    def test_a_kernel_registered_later_names_descendants_too(self, monkeypatch):
        """`kernel_for` walks the MRO, so a subclass inherits that kernel."""
        hosts = list(processor_module._HOSTS)
        monkeypatch.setattr(processor_module, "_HOSTS", [])
        monkeypatch.setattr(processor_module, "_UNCHECKED", [])

        class SeamRoot(PatchProcessor):
            """Metadata's, until its kernel is registered."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamRoot"

        class SeamLeaf(SeamRoot):
            """A named subclass, metadata's when it is created."""

            __module__ = "dascore.proc.basic"
            __qualname__ = "SeamLeaf"

        class SeamAnon(SeamRoot):
            """An unnamed subclass, which has no method anywhere."""

            name = None

        for name in ("seam_root", "seam_leaf"):
            _host_method(dc.PatchMeta, name, monkeypatch)
        assert SeamAnon.patch_function is None
        monkeypatch.setattr(processor_module, "_HOSTS", hosts)
        with pytest.raises(ParameterError, match=r"seam_root.*seam_leaf"):

            @register_kernel(SeamRoot, "numpy")
            def _root(processor, data):
                """Never reached; the check names every listing which moves."""
                return data * 2


class TestConversionsKeepTheirAxes:
    """The planned axis is the dimension asked for, not another."""

    @pytest.mark.parametrize("dim", ["time", "distance"])
    @pytest.mark.parametrize("name", ["normalize", "standardize", "demean"])
    def test_along_each_dim(self, patch, name, dim):
        """Every cell differs, so the other axis gives other numbers."""
        other = "distance" if dim == "time" else "time"
        out = getattr(patch, name)(dim)
        assert not np.allclose(out.data, getattr(patch, name)(other).data)
        data = np.asarray(patch.data)
        axis = patch.get_axis(dim)
        mean = data.mean(axis=axis, keepdims=True)
        expected = {
            "demean": data - mean,
            "standardize": (data - mean) / data.std(axis=axis, keepdims=True),
            "normalize": data / np.sqrt((data**2).sum(axis=axis, keepdims=True)),
        }[name]
        assert np.allclose(out.data, expected)

    @pytest.mark.parametrize("name", ["normalize", "standardize"])
    def test_data_type_cleared(self, patch, name):
        """Both declare `data_type = ""`."""
        tagged = patch.update_attrs(data_type="velocity")
        assert getattr(tagged, name)("time").attrs.data_type == ""

    def test_transpose_writes_no_history(self, patch):
        """`history = None`, on the class and on its function."""
        out = patch.transpose("time", "distance")
        assert out.attrs.history == patch.attrs.history
        assert dc.proc.transpose._history is None


class TestKnownReal:
    """Detect real dtypes even on array backends without NumPy dtype.kind."""

    def test_something_which_is_not_an_array(self):
        """No dtype at all says nothing, so the operation runs."""
        assert _known_real(object()) is False

    def test_a_dtype_no_namespace_will_answer_for(self):
        """A dtype with no `kind` from nothing which can be asked is not real."""

        class _Dtype:
            """A dtype which says nothing about what it holds."""

        class _Array:
            """An array-like whose namespace cannot be found."""

            dtype = _Dtype()

        assert _known_real(_Array()) is False
