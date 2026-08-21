"""
What a patch operation said as a task has to do.

The contracts run over `_calls.CALLS`, which holds one call to every patch
function DASCore defines, so a function added later is covered without this
file being edited.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
from pydantic import Field

import dascore as dc
import dascore.workflow.processor as processor_module
from dascore.exceptions import ParameterError
from dascore.models.registry import registered_models
from dascore.units import get_quantity
from dascore.workflow import (
    PatchOp,
    PatchProcessor,
    Task,
    fingerprint_call,
    register_implementation,
    resolve_patch_function,
)
from dascore.workflow.processor import patch_function_tag, register_patch_function

from ._calls import CALLS, get_patch, patch_functions, resolve

PATCH_FUNCTIONS = patch_functions()

# Named by the function, so a failure says which one.
IDS = [x[0] for x in CALLS]

# The operations whose arguments the serializer refuses. A frame is data,
# not a parameter; the serializer says so, and this records which calls in
# the catalogue land on it.
_NOT_WRITABLE = {"coords_from_df", "add_distance_to"}


def _drawn(axes) -> tuple[int, ...]:
    """Return how much was drawn on a set of axes, if that is what this is."""
    parts = ("images", "lines", "collections", "patches", "texts")
    return tuple(len(getattr(axes, x, ())) for x in parts)


def _same_patch(one, other) -> bool:
    """
    Whether two results are the same result.

    `Patch.equals` compares data with `np.array_equal`, which reports a NaN
    as unequal to itself, so a patch carrying one never equals even itself.
    Several operations here produce NaN legitimately (`dropna`, `stalta`),
    so the comparison is spelled out.

    An operation which draws returns axes rather than a patch. There is no
    equality for those, so what was drawn on them is counted -- which at
    least tells a plot from an empty frame.
    """
    if not isinstance(one, dc.Patch):
        return type(one) is type(other) and _drawn(one) == _drawn(other)
    data, other_data = np.asarray(one.data), np.asarray(other.data)
    if data.shape != other_data.shape or data.dtype != other_data.dtype:
        return False
    if not np.array_equal(data, other_data, equal_nan=data.dtype.kind == "f"):
        return False
    return one.coords == other.coords and one.attrs == other.attrs


@dc.patch_function()
def op_test_scaled(patch, factor=2.0):
    """Scale a patch, for the contracts which need a function of our own."""
    return patch.new(data=patch.data * factor)


@dc.patch_function()
def op_test_field_default(patch, value=Field(default=7)):
    """Take a parameter whose default is spelled as a pydantic Field."""
    return patch.update_attrs(seen=value)


@dc.patch_function()
def op_test_leading_then_group(patch, first, *rest, flag=False):
    """Take one named argument, then a group of them."""
    return patch.update_attrs(seen=f"{first}{rest}{flag}")


@dc.patch_function(version="1.0")
def op_test_versioned(patch, factor=1):
    """Do nothing, at the version it is declared."""
    return patch


@pytest.fixture(autouse=True)
def _close_figures():
    """
    Close whatever the drawing operations opened.

    Several contracts run every catalogued call twice, and four of them
    draw, so without this the suite holds hundreds of open figures.
    """
    yield
    plt.close("all")


@pytest.fixture(scope="module")
def patch():
    """A patch for the contracts which need any patch at all."""
    return dc.get_example_patch("random_das")


class TestEveryPatchFunction:
    """Contracts which hold for all of them."""

    def test_every_patch_function_is_covered(self):
        """A function added later is not covered until it is given a call."""
        named = {x[0] for x in CALLS}
        missing = {x.__name__ for x in PATCH_FUNCTIONS} - named
        assert not missing, f"no call is catalogued for {sorted(missing)}"

    def test_the_walk_found_them_all(self):
        """A floor under everything which runs over the catalogue."""
        assert len(PATCH_FUNCTIONS) >= 89

    @pytest.mark.parametrize("call", CALLS, ids=IDS)
    def test_the_op_does_what_the_call_did(self, call):
        """The operation as a task answers as the operation as a call."""
        name, key, args, kwargs = call
        args = resolve(args)
        func = _function(name)
        target = get_patch(key)
        assert _same_patch(
            func(target, *args, **kwargs), func.op(*args, **kwargs)(target)
        )

    @pytest.mark.parametrize("call", CALLS, ids=IDS)
    def test_the_history_is_the_same(self, call):
        """
        Down to what the patch says was done to it.

        `_same_patch` compares attrs, and history is one, so this is
        already covered -- said again here because it is the property
        which catches an operation that skipped the decorator, and it
        should not quietly go away if that comparison is ever loosened.
        """
        name, key, args, kwargs = call
        args = resolve(args)
        func = _function(name)
        target = get_patch(key)
        direct = func(target, *args, **kwargs)
        if not isinstance(direct, dc.Patch):
            pytest.skip(f"{name} does not return a patch")
        assert func.op(*args, **kwargs)(target).attrs.history == direct.attrs.history

    @pytest.mark.parametrize("call", CALLS, ids=IDS)
    def test_the_call_fingerprints_alike(self, call):
        """One call has one fingerprint, whichever route asks for it."""
        name, _, args, kwargs = call
        args = resolve(args)
        func = _function(name)
        assert (
            fingerprint_call(func, args, kwargs) == func.op(*args, **kwargs).fingerprint
        )

    @pytest.mark.parametrize("call", CALLS, ids=IDS)
    def test_the_op_is_written_down(self, call):
        """What the operation was given survives being written out."""
        name, key, args, kwargs = call
        args = resolve(args)
        op = _function(name).op(*args, **kwargs)
        if name in _NOT_WRITABLE:
            pytest.skip(f"{name} takes an argument no document can hold")
        read_back = Task.from_dict(op.to_dict())
        assert read_back == op
        # Equality is by fingerprint, which does not tell a tuple from the
        # list a document holds it as, so what the operation *does* is
        # asserted too.
        target = get_patch(key)
        assert _same_patch(op(target), read_back(target))

    @pytest.mark.parametrize("name", sorted(_NOT_WRITABLE))
    def test_an_operation_a_document_cannot_hold(self, name):
        """
        An operation given a frame says so rather than writing half of one.

        The serializer refuses a dataframe parameter, and an operation is
        not a special case: it is refused where the document is written.
        """
        call = next(x for x in CALLS if x[0] == name)
        op = _function(name).op(*resolve(call[2]), **call[3])
        with pytest.raises(ParameterError, match="cannot be written"):
            op.to_dict()

    @pytest.mark.parametrize("call", CALLS, ids=IDS)
    def test_the_op_pickles(self, call):
        """An operation handed to another process carries what it is."""
        name, _, args, kwargs = call
        args = resolve(args)
        op = _function(name).op(*args, **kwargs)
        assert pickle.loads(pickle.dumps(op)) == op

    @pytest.mark.parametrize("call", CALLS, ids=IDS)
    def test_the_version_is_the_functions(self, call):
        """
        The operation reports the version its function is declared at.

        Not the class's: `PatchOp` stands for every patch function, so its
        own version says nothing about any of them.
        """
        name, _, args, kwargs = call
        args = resolve(args)
        func = _function(name)
        assert _function(name).op(*args, **kwargs).version == func.__version__


def _function(name):
    """Return the patch function a catalogue entry names."""
    for func in PATCH_FUNCTIONS:
        if func.__name__ == name:
            return func
    msg = f"no patch function called {name!r}"
    raise LookupError(msg)


class TestOneCanonicalCall:
    """Two spellings of one call are one operation."""

    def test_positional_and_keyword(self):
        """Which is what binding against the signature is for."""
        first = dc.proc.normalize.op("time")
        second = dc.proc.normalize.op(dim="time")
        assert first.kwargs == second.kwargs == {"dim": "time", "norm": "l2"}
        assert first.fingerprint == second.fingerprint
        assert first == second

    def test_a_default_spelled_out(self):
        """Giving a default explicitly is the same operation as leaving it."""
        assert dc.proc.normalize.op("time") == dc.proc.normalize.op("time", norm="l2")

    def test_a_star_args_group(self):
        """A `*args` group is one field holding the tuple it is."""
        op = dc.proc.transpose.op("time", "distance")
        assert op.kwargs == {"dims": ("time", "distance")}

    def test_an_extra_is_kept_by_name(self):
        """A dimension the signature does not name is bound under its own."""
        op = dc.proc.pass_filter.op(time=(10, 100))
        assert op.kwargs["time"] == (10, 100)
        assert op.kwargs["corners"] == 4

    def test_a_keyword_only_parameter(self):
        """One which cannot be given positionally still binds."""
        op = dc.transform.dft.op("time", real=True)
        assert op.kwargs["dim"] == "time"
        assert op.kwargs["real"] is True

    def test_a_field_default(self):
        """A default spelled as a pydantic Field is the value it holds."""
        assert op_test_field_default.op().kwargs == {"value": 7}

    def test_different_arguments_are_different_operations(self):
        """Two calls which do different things do not share a fingerprint."""
        assert (
            dc.proc.decimate.op(time=2).fingerprint
            != dc.proc.decimate.op(time=3).fingerprint
        )


class TestRefusedAtConstruction:
    """A broken operation says so where it was written down."""

    def test_an_unknown_dascore_name(self):
        """A bare tag DASCore does not define is not an operation."""
        with pytest.raises(ParameterError, match="DASCore defines none"):
            PatchOp(name="not_a_patch_function")

    def test_an_unknown_package(self):
        """A tag from a package this process lacks says what to import."""
        with pytest.raises(ParameterError, match="install nosuchpkg"):
            PatchOp(name="nosuchpkg:denoise", module="nosuchpkg.filters")

    def test_a_name_from_a_script(self):
        """Which nothing can import, so it says to redefine it instead."""
        with pytest.raises(ParameterError, match="script or a notebook"):
            PatchOp(name="__main__:my_filter")

    def test_a_patch_method_which_is_not_a_patch_function(self):
        """
        Only a registered patch function is an operation.

        `get_axis` and `new` are how a patch is used, not operations on
        one, and a document should not reach into the class through this.
        """
        for name in ("get_axis", "new", "update", "dims"):
            with pytest.raises(ParameterError, match="DASCore defines none"):
                PatchOp(name=name)

    def test_a_function_defined_inside_a_call(self):
        """
        One which exists only while a call runs cannot be named.

        The registry skips it, as the model registry skips a class defined
        the same way, so `op` says so rather than inventing a tag.
        """

        @dc.patch_function()
        def defined_in_here(patch):
            """Exist only for the length of this test."""
            return patch

        with pytest.raises(ParameterError, match="cannot be named in a document"):
            defined_in_here.op()

    def test_an_extra_which_collides_with_a_parameter(self):
        """
        A call which cannot be written as one mapping is refused.

        `append_dims(patch, *empty_dims, **dim_kwargs)` is the live shape:
        `empty_dims` cannot be given by name, so `empty_dims=3` lands in
        the `**kwargs` group and would overwrite the group it looks like.
        """
        with pytest.raises(ParameterError, match="both as a parameter"):
            dc.proc.append_dims.op("a", empty_dims=3)

    def test_an_argument_the_signature_rejects(self):
        """An argument the function does not take fails at construction."""
        with pytest.raises(ParameterError, match="cannot be called that way"):
            dc.proc.normalize.op(dim="time", not_a_parameter=1)

    def test_an_argument_the_signature_rejects_by_hand(self):
        """
        The same, for an operation built without going through a call.

        The required argument is given, so what is refused is the extra
        rather than the absence, which the test below covers.
        """
        with pytest.raises(ParameterError, match="unexpected keyword"):
            PatchOp(name="normalize", kwargs={"dim": "time", "not_a_parameter": 1})

    def test_a_missing_required_argument(self):
        """A parameter with no default has to be given."""
        with pytest.raises(ParameterError, match="cannot be called that way"):
            dc.proc.normalize.op()


class TestCanonicalByHand:
    """An operation written by hand is the one a call would have built."""

    def test_defaults_are_filled_in(self):
        """Or the two would be two operations doing one thing."""
        by_hand = PatchOp(name="detrend", kwargs={"dim": "time"})
        by_call = dc.proc.detrend.op("time")
        assert by_hand.kwargs == by_call.kwargs
        assert by_hand == by_call
        assert by_hand.fingerprint == by_call.fingerprint

    def test_a_name_with_an_implementation(self):
        """
        A hand-built `PatchOp` for an implemented name is the same
        operation, and says so where it counts.

        It is not the same *object*: `.op(...)` gives the class which
        implements the name, and equality asks for one type. The
        fingerprint is what provenance is written from, and that agrees,
        so an id recorded either way still matches.
        """
        by_hand = PatchOp(name="normalize", kwargs={"dim": "time", "norm": "l2"})
        by_call = dc.proc.normalize.op("time")
        assert by_hand.fingerprint == by_call.fingerprint
        assert by_hand.kwargs == by_call.kwargs
        assert type(by_hand) is not type(by_call)

    def test_a_star_args_group_by_hand(self):
        """Including one whose arguments cannot be passed by name."""
        assert PatchOp(
            name="flip", kwargs={"dims": ("time",), "flip_coords": True}
        ) == dc.proc.flip.op("time")


class TestAPositionalBeforeAStarArgs:
    """A shape DASCore has none of, which the binding still has to handle."""

    def test_it_binds_and_runs(self, patch):
        """The leading argument goes back to being positional to run."""
        op = op_test_leading_then_group.op(1, 2, 3, flag=True)
        assert op.kwargs == {"first": 1, "rest": (2, 3), "flag": True}
        direct = op_test_leading_then_group(patch, 1, 2, 3, flag=True)
        assert op(patch).attrs.seen == direct.attrs.seen

    def test_it_is_refused_without_the_leading_argument(self):
        """Without it, everything after would land in the wrong slot."""
        with pytest.raises(ParameterError, match="cannot be called that way"):
            PatchOp(name="tests:op_test_leading_then_group", kwargs={"rest": (2, 3)})


class TestVersions:
    """What a version bump does, and does not, break."""

    def test_an_operation_carries_the_functions_version(self):
        """Not this class's, which is the same for every operation."""
        op = dc.proc.detrend.op("time")
        assert op.version == dc.proc.detrend.__version__
        assert op.to_dict()["params"]["version"] == op.version
        # A processor keeps its version out of the parameters, so that
        # `kwargs` stays the call; the document still records it.
        implemented = dc.proc.normalize.op("time")
        assert implemented.version == dc.proc.normalize.__version__
        assert implemented.to_dict()["version"] == implemented.version

    def test_a_document_reads_back_as_what_was_written(self, monkeypatch):
        """
        A document says which version it was written at, so it reads back
        as the operation it recorded even after the function moved on.
        """
        op = dc.proc.normalize.op("time")
        document = op.to_dict()
        monkeypatch.setattr(dc.proc.normalize, "__version__", "2.0")
        reloaded = PatchOp.from_dict(document)
        assert reloaded.version == "1.0"
        assert reloaded.fingerprint == op.fingerprint

    def test_a_bump_makes_a_new_operation(self, monkeypatch):
        """Which is what a version is for."""
        before = dc.proc.normalize.op("time")
        monkeypatch.setattr(dc.proc.normalize, "__version__", "2.0")
        after = dc.proc.normalize.op("time")
        assert after.version == "2.0"
        assert after.fingerprint != before.fingerprint


class TestTags:
    """What a name is: the tag which names a function, not a patch path."""

    def test_a_dascore_function_is_bare(self):
        """DASCore's own are unqualified, however a patch reaches them."""
        assert dc.proc.normalize.op("time").name == "normalize"

    def test_a_deferred_dascore_function_is_bare_too(self):
        """`viz.waterfall` is a path; the name is `waterfall`."""
        import dascore.viz  # noqa: F401, PLC0415

        assert dc.viz.waterfall.op(show=False).name == "waterfall"

    def test_our_own_is_namespaced_by_its_package(self):
        """A function defined here is `tests:name`, not a bare one."""
        assert op_test_scaled.op().name == "tests:op_test_scaled"

    def test_it_runs_and_round_trips_in_process(self, patch):
        """Which is the point: a user's function is an operation like any."""
        op = op_test_scaled.op(factor=3.0)
        assert np.array_equal(np.asarray(op(patch).data), np.asarray(patch.data) * 3.0)
        assert Task.from_dict(op.to_dict()) == op
        assert op.module == "tests.test_workflow.test_patch_op"

    def test_a_plugin_never_resolves_to_dascores(self):
        """
        A package's `normalize` is its own, and cannot shadow DASCore's.

        The tag carries the package, so the two cannot collide -- which is
        what makes the identity check the first round needed unnecessary.
        """

        def normalize(patch, factor=1):
            """Share a name with a core patch function."""
            return patch

        # Declared as a plugin's would be. `__qualname__` too, or the
        # registry reads it as defined inside this call and skips it.
        normalize.__module__ = "myplugin.filters"
        normalize.__qualname__ = "normalize"
        tagged = dc.patch_function()(normalize)
        assert patch_function_tag(tagged) == "myplugin:normalize"
        assert patch_function_tag(dc.proc.normalize) == "normalize"
        assert resolve_patch_function("normalize") is dc.proc.normalize
        assert resolve_patch_function("myplugin:normalize") is tagged

    @pytest.mark.concurrency
    def test_a_deferred_function_resolves_in_a_fresh_process(self):
        """
        Nothing has imported `dascore.viz`, and the sweep is what finds it.

        The sweep imports what the install declares; it never imports what
        a document names.
        """
        code = (
            "import sys, dascore; "
            "assert 'dascore.viz' not in sys.modules; "
            "from dascore.workflow import PatchOp; "
            "op = PatchOp(name='waterfall', kwargs={'show': False}); "
            "assert op.name == 'waterfall'; "
            "assert 'dascore.viz' in sys.modules"
        )
        subprocess.run(
            [sys.executable, "-c", code], check=True, timeout=120, capture_output=True
        )


class TestTheRegistry:
    """Two functions may not claim one tag."""

    @pytest.fixture(autouse=True)
    def _own_registry(self, monkeypatch):
        """Work against a copy, so a tag left behind changes nothing."""
        monkeypatch.setattr(
            processor_module, "_REGISTERED", dict(processor_module._REGISTERED)
        )
        monkeypatch.setattr(processor_module, "_AMBIGUOUS", {})

    def test_a_function_with_no_name(self):
        """Something callable which is not a function takes no tag."""

        class Callable:
            """A callable object, which has no `__name__`."""

            __qualname__ = "Callable"

            def __call__(self, patch):
                """Do nothing."""
                return patch

        assert patch_function_tag(Callable()) is None

    def test_a_tag_the_sweep_finds(self):
        """
        A tag missing until the install has been swept still resolves.

        The fresh-process test covers the real thing; this covers the
        branch in process, where `dascore.viz` is imported already.
        """

        def late(patch):
            """Arrive only once the sweep has run."""
            return patch

        late.__module__ = "swept.filters"
        late.__name__ = late.__qualname__ = "late"
        monkeypatch = pytest.MonkeyPatch()
        monkeypatch.setattr(processor_module, "_swept", False)
        monkeypatch.setattr(
            processor_module,
            "_sweep_patch_functions",
            lambda: processor_module._REGISTERED.update({"swept:late": late}),
        )
        try:
            assert resolve_patch_function("swept:late") is late
        finally:
            monkeypatch.undo()

    def test_two_dascore_functions_may_not_share_a_tag(self):
        """Ours are ours to keep unique, so the collision is an error."""

        def normalize(patch):
            """Claim a tag DASCore already uses."""
            return patch

        normalize.__module__ = "dascore.somewhere"
        normalize.__qualname__ = "normalize"
        with pytest.raises(ParameterError, match="claim the tag"):
            register_patch_function(normalize)

    def test_two_plugins_may_not_resolve_one_tag(self):
        """
        A user cannot rename another package's function, so both import.

        What the tag may not do is quietly pick one: a document written by
        the first would then be read as the second.
        """

        def denoise(patch):
            """Claim a tag another package also claims."""
            return patch

        denoise.__module__ = "pkga.filters"
        denoise.__qualname__ = "denoise"
        assert register_patch_function(denoise) == "pkga:denoise"

        def other(patch):
            """The other package's function of the same name."""
            return patch

        # `__name__` is what the tag is built from; `__qualname__` is what
        # says whether it was defined inside a call.
        other.__module__ = "pkga.elsewhere"
        other.__name__ = other.__qualname__ = "denoise"
        with pytest.warns(UserWarning, match="claim the tag"):
            register_patch_function(other)
        with pytest.raises(ParameterError, match="names two functions"):
            resolve_patch_function("pkga:denoise")

    def test_a_module_re_imported_keeps_its_entry(self):
        """The same function registered twice is not a collision."""
        tag = register_patch_function(op_test_scaled)
        assert register_patch_function(op_test_scaled) == tag


class TestNodeNames:
    """An operation is labelled for itself, not for `PatchOp`."""

    def test_the_label_is_the_operation(self):
        """Or every operation would be labelled `patch_op`."""
        assert dc.proc.detrend.op("time").node_name == "detrend"
        assert dc.proc.normalize.op("time").node_name == "normalize"

    def test_a_namespaced_operation(self):
        """A package-qualified tag is spelled the way a node name can be."""
        assert op_test_scaled.op().node_name == "tests_op_test_scaled"


class TestDocuments:
    """A saved operation names one class, whatever the operation."""

    def test_the_registry_gains_one_tag(self):
        """
        One class for every patch function, not one class each.

        `PatchOp` is the whole cost of naming any of them in a document.
        """
        tags = {
            tag
            for tag, cls in registered_models().items()
            if isinstance(cls, type) and issubclass(cls, (PatchOp, PatchProcessor))
        }
        # Spelled out rather than counted, so that a class added without
        # a reason to is noticed. The module argues against a class per
        # patch function; these are the exceptions it names -- the ones
        # wanting a kernel seam.
        assert tags == {
            "PatchOp",
            "PatchProcessor",
            "Abs",
            "Conj",
            "Demean",
            "Imag",
            "Normalize",
            "Real",
            "RenameCoords",
            "Standardize",
            "Transpose",
        }

    def test_the_document_names_the_operation(self):
        """The name and the arguments are what a document holds."""
        op = dc.proc.pass_filter.op(time=(10, 100))
        document = op.to_dict()
        assert document["object_type"] == "PatchOp"
        assert document["params"]["name"] == "pass_filter"
        assert document["params"]["kwargs"]["time"] == [10, 100]

    def test_a_document_which_cannot_be_read_says_what_to_import(self):
        """The message is worth nothing if it does not name the function."""
        document = dc.proc.detrend.op("time").to_dict()
        broken = document["params"]
        broken["name"], broken["module"] = "nosuchpkg:denoise", "nosuchpkg.filters"
        with pytest.raises(ParameterError, match="nosuchpkg"):
            PatchOp.from_dict(document)

    def test_operations_run_in_turn(self, patch):
        """Which is what the operations are for."""
        ops = [dc.proc.detrend.op("time"), dc.proc.normalize.op("time")]
        expected = patch.detrend("time").normalize("time")
        out = patch
        for op in ops:
            out = op(out)
        assert _same_patch(out, expected)
        assert [PatchOp.from_dict(x.to_dict()) for x in ops] == ops


class TestFingerprintCall:
    """The digest a call carries."""

    def test_a_call_is_not_fingerprinted_until_something_reads_it(self, patch):
        """
        The decorator does not hash a call's arguments on every call.

        Nothing consumes the answer until patch ids arrive, and taking it
        eagerly costs a share of every call -- around a fifth of one whose
        argument is a large array, since hashing it is proportional to its
        size -- and turns an argument the serializer cannot encode into a
        failure of a call which otherwise worked.
        """
        deep = {}
        deep["self"] = deep

        @dc.patch_function()
        def takes_anything(patch, thing=None):
            """Accept whatever it is given."""
            return patch

        # Would raise RecursionError if the wrapper hashed its arguments.
        assert takes_anything(patch, thing=deep) is not None

    def test_the_version_is_part_of_it(self, monkeypatch):
        """An operation at a new version is a new operation."""
        before = fingerprint_call(op_test_versioned, (), {})
        monkeypatch.setattr(op_test_versioned, "__version__", "2.0")
        assert fingerprint_call(op_test_versioned, (), {}) != before

    def test_the_name_is_part_of_it(self):
        """Two operations given the same arguments are still two."""
        assert fingerprint_call(
            dc.proc.demean, (), {"dim": "time"}
        ) != fingerprint_call(dc.proc.demedian, (), {"dim": "time"})

    def test_equal_quantities_are_two_calls(self):
        """
        A pint quantity is not a cache key.

        `1 * m` and `100 * cm` are equal and hash alike, while the
        serializer encodes them differently. Were the cache keyed on them,
        whichever call ran second would be handed the first one's answer,
        and the same call would fingerprint differently depending on what
        a process happened to do before it.
        """
        one_meter, hundred_cm = get_quantity("1 m"), get_quantity("100 cm")
        # The trap itself: equal to Python, and the same hash.
        assert one_meter == hundred_cm
        assert hash(one_meter) == hash(hundred_cm)
        first = fingerprint_call(dc.proc.select, (), {"distance": one_meter})
        second = fingerprint_call(dc.proc.select, (), {"distance": hundred_cm})
        assert first != second
        # And whichever ran first, both still answer for themselves.
        assert fingerprint_call(dc.proc.select, (), {"distance": one_meter}) == first
        assert fingerprint_call(dc.proc.select, (), {"distance": hundred_cm}) == second

    @pytest.mark.parametrize(
        ("func", "args", "kwargs", "expected"),
        [
            ("abs", (), {}, "9a2280cd3ff6dfad"),
            # With arguments, so that how they are encoded is pinned too
            # and not only the name and the version.
            ("pass_filter", (), {"time": (10, 100)}, "f6eddbc14ba34887"),
            ("transpose", ("time", "distance"), {}, "3678768d58da1f37"),
        ],
    )
    def test_it_is_stable(self, func, args, kwargs, expected):
        """A fingerprint written down last week names the same call today."""
        assert fingerprint_call(getattr(dc.proc, func), args, kwargs) == expected


class TestImplementations:
    """One name, one implementation."""

    def test_a_registered_class_is_what_the_name_means(self, patch, monkeypatch):
        """
        A hand-written class takes over the name it implements.

        Empty in this PR; the first entry arrives with the first
        plan/kernel split, and this is what says the routing works.
        """

        class Doubler(PatchProcessor):
            """A hand-written stand-in for `normalize`."""

            dim: str = "time"
            norm: str = "l2"

            def run(self, patch):
                """Do something a caller could tell apart."""
                return patch.new(data=patch.data * 2)

        # Through the public hook, against a copy of the table: it is a
        # module global, and a name left in it would change every test
        # which runs after this one.
        table = dict(processor_module._IMPLEMENTATIONS)
        monkeypatch.setattr(processor_module, "_IMPLEMENTATIONS", table)
        register_implementation("normalize", Doubler)
        assert table["normalize"] is Doubler
        op = dc.proc.normalize.op("time")
        assert isinstance(op, Doubler)
        assert np.array_equal(np.asarray(op(patch).data), np.asarray(patch.data) * 2)

    def test_the_table_is_left_as_it_was(self):
        """The test above puts the table back, or the next one is wrong."""
        assert "detrend" not in processor_module._IMPLEMENTATIONS
        assert isinstance(dc.proc.detrend.op("time"), PatchOp)

    def test_registering_something_which_is_not_one(self):
        """Only a PatchProcessor can implement an operation."""
        with pytest.raises(ParameterError, match="is not a PatchProcessor"):
            register_implementation("normalize", dict)


class TestPatchProcessor:
    """The base the hand-written classes derive from."""

    def test_a_missing_dimension_is_refused(self, patch):
        """A patch which cannot carry the operation is refused by `check`."""

        class NeedsPressure(PatchProcessor):
            """A processor which wants a dimension the patch lacks."""

            required_dims = ("pressure",)

            def run(self, patch):
                """Check and hand back."""
                return self.check(patch)

        with pytest.raises(dc.exceptions.PatchCoordinateError):
            NeedsPressure()(patch)

    def test_a_present_dimension_passes(self, patch):
        """A patch which carries it goes through untouched."""

        class NeedsTime(PatchProcessor):
            """A processor which wants a dimension the patch has."""

            required_dims = "time"

            def run(self, patch):
                """Check and hand back."""
                return self.check(patch)

        assert NeedsTime()(patch) is patch

    def test_a_required_coord(self, patch):
        """A coordinate is checked the same way a dimension is."""

        class NeedsCoord(PatchProcessor):
            """A processor which wants a coordinate the patch lacks."""

            required_coords = ("no_such_coord",)

            def run(self, patch):
                """Check and hand back."""
                return self.check(patch)

        with pytest.raises(dc.exceptions.PatchCoordinateError, match="no_such_coord"):
            NeedsCoord()(patch)

    def test_a_required_attr(self, patch):
        """As is an attr, whether named alone or against a value."""

        class NeedsVelocity(PatchProcessor):
            """A processor which wants a data_type the patch lacks."""

            required_attrs: ClassVar = {"data_type": "velocity"}

            def run(self, patch):
                """Check and hand back."""
                return self.check(patch)

        with pytest.raises(dc.exceptions.PatchAttributeError):
            NeedsVelocity()(patch)
        assert NeedsVelocity()(patch.update_attrs(data_type="velocity")) is not None
