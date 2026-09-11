"""
Tests for PatchProcessor: the seam, the generated function, and kernels.
"""

from __future__ import annotations

import inspect
import pickle
from typing import ClassVar

import numpy as np
import pytest
from pydantic import ConfigDict, Field

import dascore as dc
from dascore.core.processor import PatchMeta, PatchProcessor, register_kernel
from dascore.exceptions import (
    CoordDataError,
    ParameterError,
    PatchAttributeError,
    PatchCoordinateError,
    PatchDataError,
)
from dascore.proc.basic import Normalize, _known_real
from dascore.utils.patch_registry import patch_function_tag, resolve_patch_function


class SeamScale(PatchProcessor):
    """Multiply the data by a factor."""

    factor: float = 2.0

    def kernel(self, data):
        """Scale every sample."""
        return data * self.factor


class SeamSum(PatchProcessor):
    """Sum along a dimension, keeping it with length one."""

    dim: str = "time"

    def plan(self, patch, out):
        """Say which axis."""
        return {"axis": patch.get_axis(self.dim)}

    def kernel(self, data, *, axis):
        """Sum along it."""
        return data.sum(axis=axis, keepdims=True)

    def derive(self, patch):
        """The summed dimension keeps one sample."""
        coord = patch.get_coord(self.dim)
        return patch.new(coords=patch.coords.update(**{self.dim: coord[:1]}))


class SeamExtras(PatchProcessor):
    """Take a positional group and keyword extras."""

    names: tuple[str, ...] = ()
    flag: bool = False
    model_config = ConfigDict(extra="allow", frozen=True)
    _var_positional = "names"


class SeamHidden(PatchProcessor):
    """A processor with no patch function."""

    name = None


class SeamNeedsVelocity(PatchProcessor):
    """Declare requirements, and change the data_type."""

    required_dims = ("time",)
    required_attrs: ClassVar = {"data_type": "velocity"}
    data_type = "acceleration"

    def kernel(self, data):
        """Return a copy."""
        return data + 0


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

        assert Named()(patch).attrs.history[-1] == "named"

    def test_it_advances_the_ids(self, patch):
        """Which data stays; what was done moves."""
        out = SeamScale(3)(patch)
        assert out.attrs.patch_id == patch.attrs.patch_id
        assert out.attrs.processing_id != patch.attrs.processing_id

    def test_the_method_and_the_class_stamp_alike(self, patch):
        """One route, so one history and one id."""
        by_method = patch.normalize("time")
        by_class = Normalize(dim="time")(patch)
        assert by_method.attrs.history == by_class.attrs.history
        assert by_method.attrs.processing_id == by_class.attrs.processing_id

    def test_derive_and_plan(self, patch):
        """A shape change runs through derive, and the kernel gets the axis."""
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

        assert resolve_patch_function("normalize") is Normalize.patch_function

    def test_a_coordinate_named_patch(self):
        """The patch argument is positional-only, so an extra may be `patch`."""
        patch = dc.Patch(
            data=np.arange(3.0), coords={"patch": np.arange(3)}, dims=("patch",)
        )
        assert patch.rename_coords(patch="renamed").dims == ("renamed",)

    def test_a_narrow_subclass(self, patch):
        """A subclass whose `__init__` takes no dtype still transposes."""

        class _Sub(dc.Patch):
            def __init__(self, data=None, coords=None, dims=None, attrs=None):
                super().__init__(data=data, coords=coords, dims=dims, attrs=attrs)

        sub = _Sub(patch.data, coords=patch.coords)
        assert type(sub.transpose("time", "distance")) is _Sub
        assert type(sub.rename_coords(distance="depth")) is _Sub

    def test_a_required_field_after_a_default(self, patch):
        """A subclass adding a required field makes it keyword-only."""

        class Offset(SeamScale):
            """Scale, then add."""

            name = None
            offset: float

            def kernel(self, data):
                """Scale and add."""
                return data * self.factor + self.offset

        sig = Offset._call_signature
        assert sig.parameters["offset"].kind == inspect.Parameter.KEYWORD_ONLY
        assert np.allclose(Offset(offset=1)(patch).data, patch.data * 2 + 1)

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
            assert out.attrs.processing_id == patch.attrs.processing_id

    def test_signature_carries_annotations(self):
        """Field annotations reach the generated signature."""
        sig = inspect.signature(dc.proc.normalize)
        assert sig.parameters["dim"].annotation is str
        assert sig.return_annotation == "PatchType"

    def test_generated_functions_pickle(self, patch):
        """So a process pool can run them."""
        assert pickle.loads(pickle.dumps(dc.proc.demean)) is dc.proc.demean
        func = SeamScale.patch_function
        assert pickle.loads(pickle.dumps(func)) is func

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

    def test_a_factory_default_shows_its_value(self):
        """Not pydantic's undefined marker."""

        class Factory(PatchProcessor):
            """Default a field through a factory."""

            name = None
            values: tuple = Field(default_factory=lambda: (1, 2))

        assert Factory._call_signature.parameters["values"].default == (1, 2)

    def test_a_numpy_bool_in_a_plan(self, patch):
        """Numpy scalars are numbers too."""

        class Flag(SeamScale):
            """Plan a numpy bool."""

            name = None

            def plan(self, patch, out):
                """Return one."""
                return {"flag": np.bool_(True)}

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

    def test_no_data(self, patch):
        """A patch without data cannot be processed."""
        with pytest.raises(PatchDataError):
            SeamScale()(patch.drop_data())


class TestPlan:
    """A plan holds numbers, tuples of them, or numeric arrays only."""

    @pytest.mark.parametrize(
        "value",
        ["time", None, [1, 2], np.array(["a"])],
        ids=["string", "none", "list", "string_array"],
    )
    def test_refused(self, patch, value):
        """Anything else is refused before the kernel runs."""

        class Bad(SeamScale):
            """Plan something a kernel may not take."""

            def plan(self, patch, out):
                """Return the value under test."""
                return {"value": value}

        with pytest.raises(ParameterError, match="plan may hold only"):
            Bad()(patch)

    def test_a_patch_or_coords_are_refused(self, patch):
        """The kernel never sees a patch or a coord manager."""
        for value in (patch, patch.coords):

            class Bad(SeamScale):
                """Plan a patch."""

                def plan(self, patch, out, value=value):
                    """Return the value under test."""
                    return {"value": value}

            with pytest.raises(ParameterError, match="plan may hold only"):
                Bad()(patch)

    def test_allowed(self, patch):
        """Ints, floats, bools, nested tuples and numeric arrays pass."""

        class Good(PatchProcessor):
            """Plan every allowed kind."""

            name = None

            def plan(self, patch, out):
                """Return one of each."""
                return {
                    "a": 1,
                    "b": 2.0,
                    "c": True,
                    "d": ((1, 2), 3.0),
                    "e": np.arange(3),
                }

            def kernel(self, data, **plan):
                """Ignore the plan, keep the data."""
                return data + 0

        assert Good()(patch).equals(patch)


class TestTransposeDerive:
    """Transpose derives a new shape from a patch without data."""

    def test_derive_without_data(self, patch):
        """Coords with another shape, and no data read."""
        out = dc.proc.Transpose(dims=("time", "distance")).derive(patch.drop_data())
        assert out.dims == ("time", "distance")
        assert out.shape == patch.shape[::-1]

    def test_the_kernel_permutes(self, patch):
        """The data follow the coords."""
        out = patch.transpose("time", "distance")
        assert np.array_equal(out.data, np.asarray(patch.data).T)


class TestFingerprint:
    """The identity of an operation is its validated fields."""

    def test_validated_fields(self):
        """An int and the float it validates to are one operation."""
        assert SeamScale(4).fingerprint == SeamScale(4.0).fingerprint
        assert SeamScale(4).fingerprint == SeamScale(factor=4).fingerprint
        assert SeamScale(4).fingerprint != SeamScale(5).fingerprint

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
        assert Other().fingerprint != SeamScale().fingerprint

    def test_the_version_counts(self, monkeypatch):
        """A bump makes a new operation."""
        before = SeamScale().fingerprint
        monkeypatch.setattr(SeamScale, "__version__", "2.0")
        assert SeamScale().fingerprint != before


class TestGeneratedFunction:
    """What subclassing generates."""

    def test_signature_and_defaults(self):
        """The fields in declaration order, with their defaults."""
        sig = inspect.signature(dc.proc.normalize)
        assert list(sig.parameters) == ["patch", "dim", "norm", "window", "samples"]
        assert sig.parameters["norm"].default == "l2"
        assert sig.parameters["dim"].default is inspect.Parameter.empty

    def test_star_args_and_extras(self, patch):
        """`_var_positional` becomes `*name`; extra="allow" adds `**kwargs`."""
        sig = inspect.signature(SeamExtras.patch_function)
        kinds = [x.kind for x in sig.parameters.values()]
        assert kinds[1] == inspect.Parameter.VAR_POSITIONAL
        assert kinds[-1] == inspect.Parameter.VAR_KEYWORD
        op = SeamExtras("a", "b", flag=True, other=1)
        assert op.kwargs == {"names": ("a", "b"), "flag": True, "other": 1}

    def test_names_and_docs(self):
        """Named for the operation, documented by the class."""
        func = Normalize.patch_function
        assert func.__name__ == "normalize"
        assert func.__module__ == "dascore.proc.basic"
        assert func.__doc__ == Normalize.__doc__
        assert "{sample_explanation}" not in func.__doc__
        assert func.__processor__ is Normalize
        assert dc.proc.normalize is func is dc.Patch.normalize

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

            def kernel(self, data):
                """The child's own arithmetic."""
                return data

        assert Parent.kernel_for("numpy") is _parent_kernel
        assert Child.kernel_for("numpy") is Child.__dict__["kernel"]

    def test_no_kernel_at_all_is_metadata_only(self):
        """A processor which touches no data says so by defining none."""
        assert SeamHidden.kernel_for("numpy") is None


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


class TestPatchMeta:
    """The metadata tile_apply and adaptive_spectral_filter still plan with."""

    @pytest.fixture(scope="class")
    @classmethod
    def meta(cls, patch):
        """The patch without its values."""
        return PatchMeta.from_patch(patch)

    def test_it_carries_the_shape(self, meta, patch):
        """Dims, shape, ndim, dtype, backend and axes."""
        assert (meta.dims, meta.shape, meta.ndim) == (
            patch.dims,
            patch.shape,
            len(patch.dims),
        )
        assert meta.dtype == patch.dtype
        assert meta.backend == "numpy"
        assert meta.get_axis("time") == patch.get_axis("time")

    def test_update_and_back(self, meta, patch):
        """Changing one part changes only that part; data make a patch again."""
        assert meta.update(dtype="float32").coords is meta.coords
        assert meta.to_patch(patch.data).equals(patch)
        with pytest.raises(CoordDataError):
            meta.to_patch(np.asarray(patch.data)[:2])
