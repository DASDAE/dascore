"""
Tests for the plan/kernel seam.

The nine converted operations exercise it from above, and the parity
check proves they still answer what they answered. What is here is the
seam itself: what a processor is allowed to leave out, what it is refused
for getting wrong, and how a kernel for another array backend is found.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import CoordDataError, ParameterError
from dascore.proc.basic import Abs, Normalize, _known_real
from dascore.workflow import PatchMeta, PatchProcessor, Task, register_kernel
from dascore.workflow.processor import (
    _resolve_kernel,
    register_implementation,
)


@pytest.fixture(scope="module")
def patch():
    """A patch to operate on."""
    return dc.get_example_patch()


@pytest.fixture(scope="module")
def meta(patch):
    """What that patch is, apart from its values."""
    return PatchMeta.from_patch(patch)


class TestPatchMeta:
    """A patch with the values taken out."""

    def test_it_carries_the_shape_without_the_data(self, meta, patch):
        """Which is what lets something plan an operation without one."""
        assert meta.dims == patch.dims
        assert meta.shape == patch.shape
        assert meta.ndim == len(patch.dims)
        assert meta.dtype == patch.data.dtype
        assert meta.backend == "numpy"

    def test_an_axis_by_name(self, meta, patch):
        """A kernel wants an axis; the caller said a dimension."""
        assert meta.get_axis("time") == patch.dims.index("time")

    def test_update_leaves_the_rest(self, meta):
        """Changing one part of the metadata changes only that part."""
        other = meta.update(dtype="float32")
        assert other.dtype == "float32"
        assert other.coords is meta.coords
        assert other.attrs is meta.attrs

    def test_back_to_a_patch(self, meta, patch):
        """The metadata and some data make a patch again."""
        assert meta.to_patch(patch.data).equals(patch)

    def test_a_patch_subclass_survives(self, patch):
        """
        An operation on a subclass gives back that subclass.

        A patch function used to build its result with `patch.new`, which
        constructs `self.__class__`; naming `Patch` in the metadata would
        have taken a subclass's own behaviour away from it silently.
        """

        class _Sub(dc.Patch):
            """A patch which is more than a patch."""

        sub = _Sub(data=patch.data, coords=patch.coords, dims=patch.dims)
        assert type(sub.abs()) is _Sub
        assert type(sub.transpose()) is _Sub
        assert type(sub.rename_coords(distance="depth")) is _Sub

    def test_data_of_the_wrong_shape(self, meta, patch):
        """The coords check the data on the way in, so this cannot pass."""
        with pytest.raises(CoordDataError, match="doesnt match the coordinate"):
            meta.to_patch(np.asarray(patch.data)[:2])


class TestKernelResolution:
    """Which function actually runs, and for which backend."""

    def test_a_registered_kernel_wins_for_its_backend(self, patch):
        """Which is the whole point of registering one."""

        class Doubling(PatchProcessor):
            """A processor whose generic kernel does nothing."""

            def kernel(self, data, meta, out_meta):
                """Hand the data back unchanged."""
                return data

        @register_kernel(Doubling, "numpy")
        def _double(processor, data, meta, out_meta):
            """Double it, so the two can be told apart."""
            return data * 2

        assert _resolve_kernel(Doubling, "numpy") is _double
        out = Doubling()._apply(patch)
        assert np.array_equal(np.asarray(out.data), np.asarray(patch.data) * 2)

    def test_another_backend_falls_back_to_the_generic(self):
        """A kernel written to the standard runs wherever it is asked to."""

        class Generic(PatchProcessor):
            """A processor with only a generic kernel."""

            def kernel(self, data, meta, out_meta):
                """Do nothing, visibly."""
                return data

        @register_kernel(Generic, "cupy")
        def _never(processor, data, meta, out_meta):
            """Registered for a backend nothing here uses."""
            raise AssertionError("the cupy kernel should not have been chosen")

        assert _resolve_kernel(Generic, "numpy") is Generic.__dict__["kernel"]

    def test_a_subclass_kernel_beats_a_parents_backend_kernel(self):
        """
        Or a parent could answer for an operation it does not implement.

        A subclass which wrote its own kernel means it; a kernel
        registered against the class it derives from is about the
        parent's operation, not this one.
        """

        class Parent(PatchProcessor):
            """The class the kernel is registered against."""

        @register_kernel(Parent, "numpy")
        def _parent_kernel(processor, data, meta, out_meta):
            """What the parent does on numpy."""
            return data

        class Child(Parent):
            """A subclass which computes something else entirely."""

            def kernel(self, data, meta, out_meta):
                """The child's own arithmetic."""
                return data

        assert _resolve_kernel(Parent, "numpy") is _parent_kernel
        assert _resolve_kernel(Child, "numpy") is Child.__dict__["kernel"]

    def test_no_kernel_at_all_is_metadata_only(self):
        """A processor which touches no data says so by defining none."""

        class MetadataOnly(PatchProcessor):
            """A processor with nothing to compute."""

        assert _resolve_kernel(MetadataOnly, "numpy") is None

    def test_a_metadata_only_processor_passes_the_data_through(self, patch):
        """And hands back the very patch, since nothing changed."""

        class Nothing(PatchProcessor):
            """A processor which does nothing at all."""

        assert Nothing()._apply(patch) is patch


class TestFusibility:
    """
    Whether an operation can be lowered with the ones around it.

    Something deciding what to fuse holds the chain and no arrays, so the
    answer has to come from the operation's parameters. An answer which
    needs the data is no answer at all.
    """

    def test_a_portable_kernel_is_fusible(self):
        """Written in the backend's own terms, so it can be lowered."""
        from dascore.proc.basic import Abs, Normalize

        assert Abs().fusible
        assert Normalize(dim="time", norm="l2").fusible

    def test_a_numpy_kernel_is_not(self):
        """The standard has no median which skips nulls, so this cannot."""
        from dascore.proc.basic import Demedian

        assert not Demedian().fusible

    def test_it_can_depend_on_the_arguments(self):
        """
        Some operations are portable for some of what they accept.

        `full` takes any value numpy would; the standard promises only
        the plain python scalars. `fillna` given a value with a shape
        spends it positionally, which `where` cannot say.
        """
        from dascore.proc.basic import FillNa, Full

        assert Full(fill_value=1.5).fusible
        assert not Full(fill_value=np.float64(1.5)).fusible
        assert FillNa(value=0).fusible
        assert not FillNa(value=[1, 2]).fusible

    def test_the_answer_needs_no_data(self, patch):
        """Asked of the operation, and the patch never offered."""
        from dascore.proc.basic import Full

        operation = Full(fill_value=1.5)
        assert operation.fusible is Full(fill_value=1.5).fusible

    def test_defining_reconcile_says_not_fusible(self):
        """It is the step which has to see both halves at once."""

        class Reconciling(PatchProcessor):
            """A processor which checks the data against the metadata."""

            def kernel(self, data, meta, out_meta):
                """Do nothing, visibly."""
                return data

            def reconcile(self, data, meta):
                """Look at both, which is what cannot be lowered."""
                return meta

        assert not Reconciling().fusible


class TestRegistrationRefuses:
    """What a class is turned away for, at import rather than at use."""

    def test_something_which_is_not_a_processor(self):
        """Only a PatchProcessor can implement an operation."""
        with pytest.raises(ParameterError, match="is not a PatchProcessor"):
            register_implementation("detrend", dict)

    def test_a_class_which_disagrees_about_a_requirement(self):
        """Two places saying it differently is worse than one saying it."""

        class Disagrees(PatchProcessor):
            """Claims a history the function does not declare."""

            dim: str = "time"
            type: str = "linear"
            history = None

        with pytest.raises(ParameterError, match="have to agree"):
            register_implementation("detrend", Disagrees)

    def test_a_class_at_another_version(self):
        """They fingerprint as one operation, so they are one version."""

        class Bumped(PatchProcessor):
            """Says it is a version its function is not."""

            __version__ = "9.9"
            dim: str = "time"
            type: str = "linear"

        with pytest.raises(ParameterError, match="one version"):
            register_implementation("detrend", Bumped)

    def test_a_class_missing_a_parameter(self):
        """A call could not be written down as one."""

        class Missing(PatchProcessor):
            """Has no field for the dimension detrend takes."""

        with pytest.raises(ParameterError, match="no such field"):
            register_implementation("detrend", Missing)

    def test_a_class_which_cannot_take_the_extras(self):
        """A function with **kwargs needs a class which accepts them."""

        class Strict(PatchProcessor):
            """Forbids extras, so the renames could not reach it."""

        with pytest.raises(ParameterError, match='extra="allow"'):
            register_implementation("update_coords", Strict)


class TestTheVersionTravels:
    """An operation is what it was when it was written down."""

    def test_through_a_document(self, monkeypatch):
        """A document says which version wrote it, and that is what it is."""
        op = dc.proc.normalize.op("time")
        document = op.to_dict()
        monkeypatch.setattr(dc.proc.normalize, "__version__", "2.0")
        assert Task.from_dict(document).version == "1.0"
        # A newly built one does see the bump; that is what a version is for.
        assert dc.proc.normalize.op("time").version == "2.0"

    def test_through_a_pickle(self, monkeypatch):
        """Rebuilding in another process must not re-read the function."""
        restored = Task.from_dict(dc.proc.normalize.op("time").to_dict())
        monkeypatch.setattr(dc.proc.normalize, "__version__", "2.0")
        assert pickle.loads(pickle.dumps(restored)).version == "1.0"

    def test_through_an_update(self, monkeypatch):
        """Changing an argument is not changing which version it is."""
        restored = Task.from_dict(dc.proc.normalize.op("time").to_dict())
        monkeypatch.setattr(dc.proc.normalize, "__version__", "2.0")
        assert restored.update(norm="l1").version == "1.0"


class TestKnownReal:
    """
    Whether a dtype alone says the data has no imaginary part.

    The question is asked of every `real`, `conj` and `imag`, and the
    answer decides whether the operation runs at all. A numpy dtype says
    so with `kind`; the standard does not ask a dtype for one, so a
    backend which omits it has to be asked another way.
    """

    def test_something_which_is_not_an_array(self):
        """No dtype at all says nothing, so the operation runs."""
        assert _known_real(object()) is False

    def test_a_dtype_no_namespace_will_answer_for(self):
        """
        A dtype with no `kind`, belonging to nothing which can be asked.

        Not known to be real, so the operation runs -- which is the same
        answer an object array gets, and for the same reason.
        """

        class _Dtype:
            """A dtype which says nothing about what it holds."""

        class _Array:
            """An array-like whose namespace cannot be found."""

            dtype = _Dtype()

        assert _known_real(_Array()) is False


class TestTheSeamIsInvisible:
    """Registering a class must not move anything already recorded."""

    def test_the_fingerprint_is_the_operations(self, patch):
        """Not the class's, or every processing_id would stop matching."""
        by_hand = dc.workflow.PatchOp(
            name="normalize", kwargs={"dim": "time", "norm": "l2"}
        )
        assert Normalize(dim="time", norm="l2").fingerprint == by_hand.fingerprint

    def test_an_operation_answers_what_a_patch_op_answers(self):
        """So a contract written over all of them does not have to care."""
        op = Abs()
        assert op.name == "abs"
        assert op.node_name == "abs"
        assert op.kwargs == {}
        assert op.version == dc.proc.abs.__version__
