"""
Tests for the two ids a patch carries.

These cover the rules themselves; the tests which pin where they are
applied live beside the things which apply them.
"""

from __future__ import annotations

import pickle
import warnings

import numpy as np
import pytest

import dascore as dc
import dascore.workflow.processor as processor_module
from dascore.exceptions import ParameterError
from dascore.workflow import Task, fingerprint_call
from dascore.workflow.builtin import ArrayFunc, Concatenate, Stack, Ufunc
from dascore.workflow.identity import (
    NOTHING_DONE,
    advance,
    fold_ids,
    fold_patch_ids,
    fold_processing_ids,
    new_patch_id,
    patch_id_of,
    processing_id_of,
    source_patch_id,
    stamp_combination,
)
from dascore.workflow.processor import _PATCH_ARGUMENT, _as_key, _signature


class TestNewDataId:
    """Data which names no source."""

    def test_each_is_its_own(self):
        """Two arrays are two data, however alike they look."""
        assert new_patch_id() != new_patch_id()

    def test_it_is_a_hex_string(self):
        """The same shape as every other id, so nothing can tell them apart."""
        made = new_patch_id()
        assert isinstance(made, str)
        assert int(made, 16) >= 0


# What a file's identity is made of, and one such identity.
_FIELDS = ("format_name", "version", "path", "key", "size_bytes", "mtime_ns")
_SOURCE = ("DASDAE", "1", "/data/one.h5", 0, 40, 12345)


class TestSourceDataId:
    """Data read from a file."""

    def test_it_is_derived(self):
        """Reading the same file twice is reading the same data."""
        first = source_patch_id(*_SOURCE)
        assert first == source_patch_id(*_SOURCE)

    @pytest.mark.parametrize(
        "changed",
        [
            {"format_name": "TERRA15"},
            {"version": "2"},
            {"path": "/data/two.h5"},
            {"key": 1},
            {"key": None},
            {"size_bytes": 41},
            {"mtime_ns": 999},
            {"size_bytes": None},
            {"mtime_ns": None},
        ],
    )
    def test_every_part_counts(self, changed):
        """Each field names data the others would have called the same."""
        fields = dict(zip(_FIELDS, _SOURCE, strict=True)) | changed
        assert source_patch_id(**fields) != source_patch_id(*_SOURCE)

    def test_a_key_may_be_anything_encodable(self):
        """A reader names its patches however it likes."""
        named = source_patch_id("X", "1", "/a", "channel-3")
        assert named != source_patch_id("X", "1", "/a", 3)

    def test_a_source_which_cannot_be_stat_ed(self):
        """Missing stats are a weaker id, not a different kind of one."""
        made = source_patch_id("X", "1", "/a", 0)
        assert made == source_patch_id("X", "1", "/a", 0, None, None)
        assert len(made) == len(source_patch_id(*_SOURCE))


class TestAdvance:
    """What was done."""

    def test_nothing_done_is_the_starting_point(self):
        """Data which arrived and has not been touched says so."""
        assert NOTHING_DONE == ""

    def test_an_operation_moves_it(self):
        """Which is the whole point of the id."""
        assert advance(NOTHING_DONE, "0123456789abcdef") != NOTHING_DONE

    def test_the_same_route_gives_the_same_answer(self):
        """So two patches processed alike can be told to be alike."""
        once = advance(NOTHING_DONE, "aaaa")
        assert advance(once, "bbbb") == advance(advance(NOTHING_DONE, "aaaa"), "bbbb")

    def test_order_matters(self):
        """Filtering then decimating is not decimating then filtering."""
        first = advance(advance(NOTHING_DONE, "aaaa"), "bbbb")
        second = advance(advance(NOTHING_DONE, "bbbb"), "aaaa")
        assert first != second

    def test_doing_it_twice_is_not_doing_it_once(self):
        """A fold of a fold, so a repeated operation is two operations."""
        once = advance(NOTHING_DONE, "aaaa")
        assert advance(once, "aaaa") != once

    def test_a_different_operation_is_a_different_answer(self):
        """The fingerprint is what distinguishes them."""
        assert advance(NOTHING_DONE, "aaaa") != advance(NOTHING_DONE, "bbbb")


class TestFoldDataIds:
    """Which data, when there was more than one."""

    def test_one_folds_to_itself(self):
        """Combining a patch with nothing leaves the id where it was."""
        assert fold_patch_ids(["only"]) == "only"

    def test_order_is_part_of_it(self):
        """Concatenating a before b is not concatenating b before a."""
        assert fold_patch_ids(["a", "b"]) != fold_patch_ids(["b", "a"])

    def test_repeats_are_part_of_it(self):
        """Stacking a patch with itself is not the patch."""
        assert fold_patch_ids(["a", "a"]) != fold_patch_ids(["a"])

    def test_it_is_derived(self):
        """So the same combination gives the same answer twice."""
        assert fold_patch_ids(["a", "b"]) == fold_patch_ids(["a", "b"])

    def test_it_takes_any_sequence(self):
        """Callers hold their members in whatever they hold them in."""
        assert fold_patch_ids(("a", "b")) == fold_patch_ids(["a", "b"])


class TestFoldProcessingIds:
    """What was done, when the inputs disagree."""

    def test_a_common_route_survives(self):
        """Sixty windows of one file have one history, not sixty."""
        assert fold_processing_ids(["x", "x", "x"]) == "x"

    def test_distinct_routes_fold(self):
        """Combining differently processed data says so."""
        folded = fold_processing_ids(["x", "y"])
        assert folded not in {"x", "y"}

    def test_the_fold_is_stable(self):
        """And says it the same way every time."""
        assert fold_processing_ids(["x", "y"]) == fold_processing_ids(["x", "y"])

    def test_first_seen_order(self):
        """Two orders of the same routes are two answers."""
        assert fold_processing_ids(["x", "y"]) != fold_processing_ids(["y", "x"])

    def test_repeats_do_not_count(self):
        """It is the distinct routes which matter, not how many took each."""
        assert fold_processing_ids(["x", "y", "x"]) == fold_processing_ids(["x", "y"])

    def test_nothing_folds_to_nothing_done(self):
        """An operation given no inputs has nothing to carry forward."""
        assert fold_processing_ids([]) == NOTHING_DONE

    def test_untouched_inputs_stay_untouched(self):
        """Reading two files and putting them together is not processing."""
        assert fold_processing_ids([NOTHING_DONE, NOTHING_DONE]) == NOTHING_DONE


class TestBuiltinTasks:
    """The operations which are not patch functions still have names."""

    def test_a_ufunc_names_which_one(self):
        """Adding is not subtracting."""
        assert Ufunc(name="add").fingerprint != Ufunc(name="subtract").fingerprint

    def test_a_ufunc_names_how_it_was_applied(self):
        """A reduction is not the plain call."""
        plain = Ufunc(name="add")
        assert Ufunc(name="add", method="reduce").fingerprint != plain.fingerprint

    def test_a_reversed_ufunc_is_its_own_operation(self):
        """`1 - patch` is not `patch - 1`."""
        forward = Ufunc(name="subtract")
        assert Ufunc(name="subtract", reversed=True).fingerprint != forward.fingerprint

    def test_a_ufunc_names_its_other_operands(self):
        """Multiplying by two is not multiplying by three."""
        assert Ufunc(name="multiply", operands=(2,)).fingerprint != (
            Ufunc(name="multiply", operands=(3,)).fingerprint
        )

    def test_concatenate_names_what_it_was_given(self):
        """
        Concatenating along time is not concatenating along distance.

        `time=None` is the documented call, and the serializer drops a
        `None` mapping value, so holding the call as a mapping would make
        these one operation. They are held as pairs for that reason.
        """
        assert Concatenate.from_kwargs(time=None).fingerprint != (
            Concatenate.from_kwargs(distance=None).fingerprint
        )

    def test_concatenate_names_how_much(self):
        """And the size, when one was given."""
        assert Concatenate.from_kwargs(time=None).fingerprint != (
            Concatenate.from_kwargs(time=10).fingerprint
        )

    def test_concatenate_keeps_the_order_it_was_given(self):
        """Two dimensions given the other way round is another call."""
        first = Concatenate.from_kwargs(time=None, distance=2)
        assert (
            first.fingerprint
            != Concatenate.from_kwargs(distance=2, time=None).fingerprint
        )

    def test_stack_names_the_varying_dimension(self):
        """Which is the only thing which makes two stacks differ."""
        assert Stack(dim_vary="time").fingerprint != Stack().fingerprint

    def test_an_array_func_names_which_one(self):
        """And what it was given."""
        assert ArrayFunc(name="mean", kwargs={"axis": 0}).fingerprint != (
            ArrayFunc(name="mean").fingerprint
        )

    @pytest.mark.parametrize(
        "task",
        [
            Concatenate.from_kwargs(time=None),
            Stack(dim_vary="time"),
            Ufunc(name="add", operands=(2,)),
            ArrayFunc(name="mean", kwargs={"axis": 0}),
        ],
        ids=["concatenate", "stack", "ufunc", "array_func"],
    )
    def test_they_are_written_down(self, task):
        """A provenance record holds them, so they have to survive one."""
        assert Task.from_dict(task.to_dict()) == task


class TestTheRulesOnRealPatches:
    """The rules, where they are actually applied."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch to operate on."""
        return dc.get_example_patch("random_das")

    def test_data_arrives_with_an_id(self, patch):
        """Everything downstream needs something to carry forward."""
        assert patch.attrs.patch_id
        assert patch.attrs.processing_id == NOTHING_DONE

    def test_two_patches_are_two_data(self):
        """Nothing derives an id from values, so nothing claims they are one."""
        first = dc.get_example_patch("random_das")
        assert first.attrs.patch_id != dc.get_example_patch("random_das").attrs.patch_id

    def test_an_operation_advances_what_was_done(self, patch):
        """Which is what the id is for."""
        out = patch.normalize("time")
        assert out.attrs.processing_id != patch.attrs.processing_id

    def test_an_operation_leaves_which_data_alone(self, patch):
        """Filtering data does not make it other data."""
        assert patch.normalize("time").attrs.patch_id == patch.attrs.patch_id

    def test_a_function_which_rebuilds_still_leaves_it_alone(self, patch):
        """
        Even one which builds its result from scratch.

        Without the wrapper carrying the id across from the input, a body
        which returns a patch it built itself would mint a fresh one and
        claim the data had changed.
        """

        @dc.patch_function()
        def rebuilds(patch):
            """Return a patch built from nothing but the data."""
            return dc.Patch(data=patch.data, coords=patch.coords, dims=patch.dims)

        assert rebuilds(patch).attrs.patch_id == patch.attrs.patch_id
        # And a real one which does the same thing.
        assert patch.fbe(0.5, time=(10, 100)).attrs.patch_id == patch.attrs.patch_id

    def test_the_same_route_gives_the_same_id(self, patch):
        """So two patches processed alike can be told to be alike."""
        first = patch.normalize("time").decimate(time=2)
        second = patch.normalize("time").decimate(time=2)
        assert first.attrs.processing_id == second.attrs.processing_id

    def test_a_different_route_does_not(self, patch):
        """Order included."""
        forward = patch.normalize("time").decimate(time=2)
        backward = patch.decimate(time=2).normalize("time")
        assert forward.attrs.processing_id != backward.attrs.processing_id

    def test_different_arguments_are_a_different_route(self, patch):
        """The operation's fingerprint is what distinguishes them."""
        assert patch.normalize("time").attrs.processing_id != (
            patch.normalize("distance").attrs.processing_id
        )

    def test_a_no_op_records_nothing(self, patch):
        """
        An operation which handed the patch straight back did nothing.

        `select` with no bounds is the live case: it returns the patch it
        was given, and a `processing_id` which moved would say otherwise.
        """
        assert patch.select(time=None).attrs.processing_id == patch.attrs.processing_id

    def test_the_ids_are_not_part_of_equality(self, patch):
        """Two patches with the same data are equal however they were made."""
        assert patch.equals(patch.update_attrs(patch_id="x", processing_id="y"))

    def test_a_raw_constructor_keeps_what_it_was_given(self, patch):
        """You edited it; that is on you."""
        made = patch.update_attrs(patch_id="kept", processing_id="also")
        assert made.attrs.patch_id == "kept"
        assert made.new(data=made.data).attrs.patch_id == "kept"

    def test_combining_patches_folds_which_data(self, patch):
        """Two sources make a third answer, not either of the two."""
        other = patch.update_attrs(patch_id="other")
        merged = dc.utils.attrs.combine_patch_attrs([patch.attrs, other.attrs])
        assert merged.patch_id == fold_patch_ids([patch.attrs.patch_id, "other"])
        # Spelled out, because `not in {...}` is also true of the empty
        # string, which is what dropping the fold entirely would leave.
        assert merged.patch_id
        assert merged.patch_id not in {patch.attrs.patch_id, "other"}

    def test_combining_patches_keeps_a_common_route(self, patch):
        """Windows of one file have one history, not one each."""
        first = patch.normalize("time")
        merged = dc.utils.attrs.combine_patch_attrs([first.attrs, first.attrs])
        assert merged.processing_id == first.attrs.processing_id

    def test_the_ids_survive_a_pickle(self, patch):
        """A patch handed to another process is the same data."""
        out = patch.normalize("time")
        assert pickle.loads(pickle.dumps(out)).attrs.patch_id == out.attrs.patch_id
        assert pickle.loads(pickle.dumps(out)).attrs.processing_id == (
            out.attrs.processing_id
        )

    def test_they_can_be_turned_off(self, patch):
        """A process which does not want them does not pay for them."""
        with dc.config_context(patch_provenance="disabled"):
            made = dc.Patch(data=patch.data, coords=patch.coords, dims=patch.dims)
            assert made.attrs.patch_id == ""
            assert made.normalize("time").attrs.processing_id == ""

    def test_a_summary_does_not_carry_them(self, patch):
        """
        An index does not store an id, so a summary holding one would make
        scanning a file and reading it disagree about the same data.
        """
        summary = patch.summary
        assert summary.attrs.patch_id == ""
        assert summary.attrs.processing_id == ""


class TestTheAwkwardCases:
    """The branches the ordinary path never reaches."""

    def test_no_members_folds_to_nothing(self):
        """A fold given nothing has nothing to say."""
        assert fold_ids([]) == {}

    def test_disabled_folds_to_nothing(self):
        """A process which is not keeping ids does not invent them."""
        patch = dc.get_example_patch()
        with dc.config_context(patch_provenance="disabled"):
            assert fold_ids([patch.attrs, patch.attrs]) == {}

    def test_a_function_defined_inside_a_call_is_still_named(self):
        """
        It cannot be named in a document, but it still did something.

        A `processing_id` which did not move would say it had not.
        """
        patch = dc.get_example_patch()

        @dc.patch_function()
        def only_here(patch):
            """Exist only for the length of this test."""
            return patch.new(data=patch.data + 1)

        out = only_here(patch)
        assert out.attrs.processing_id != patch.attrs.processing_id
        # Named by where it was written, which is honestly not resolvable.
        with pytest.raises(ParameterError, match="cannot be named"):
            only_here.op()

    def test_a_callable_which_cannot_be_hashed(self):
        """Its signature is asked for the slow way rather than cached."""

        class Unhashable:
            """A callable which refuses to be a dict key."""

            __hash__ = None

            def __call__(self, patch, factor=1):
                """Do nothing."""
                return patch

        assert _signature(Unhashable()) is not None


class TestOperationsWhichAreNotPatchFunctions:
    """Concatenating, stacking and ufuncs still have to say what they did."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch to operate on."""
        return dc.get_example_patch("random_das")

    @pytest.fixture(scope="class")
    def spool(self):
        """A spool whose patches are different data."""
        return dc.get_example_spool()

    def test_arithmetic_says_which_operation(self, patch):
        """
        `patch + 1` and `patch - (-1)` give the same data and are not the
        same operation; without the `Ufunc` task they shared an id.
        """
        made = {
            (patch * 2).attrs.processing_id,
            (patch * 3).attrs.processing_id,
            (patch + 5).attrs.processing_id,
            (patch - 7).attrs.processing_id,
            (patch + 1).attrs.processing_id,
            (patch - (-1)).attrs.processing_id,
        }
        assert len(made) == 6

    def test_arithmetic_leaves_which_data_alone(self, patch):
        """Scaling data does not make it other data."""
        assert (patch * 2).attrs.patch_id == patch.attrs.patch_id

    def test_two_patches_fold(self, patch):
        """Adding two patches is data from two sources."""
        other = patch.update_attrs(patch_id="other")
        assert (patch + other).attrs.patch_id == fold_patch_ids(
            [patch.attrs.patch_id, "other"]
        )

    def test_a_unary_ufunc_says_which_one(self, patch):
        """`np.abs` is not `np.sqrt`."""
        assert np.abs(patch).attrs.processing_id != patch.attrs.processing_id
        assert np.abs(patch).attrs.processing_id != (
            np.sqrt(np.abs(patch)).attrs.processing_id
        )

    def test_an_array_function_says_which_one(self, patch):
        """And what it was given."""
        first = np.mean(patch, axis=0).attrs.processing_id
        assert first != patch.attrs.processing_id
        assert first != np.mean(patch, axis=1).attrs.processing_id

    def test_a_patch_argument_is_an_input_not_a_parameter(self, patch):
        """
        Which patch was handed in is said by the ids, not the fingerprint.

        Encoding it would hash a whole patch on every call, and warn that
        a patch has no encoding of its own.
        """
        one = patch.new(data=patch.data > 0.5)
        two = patch.new(data=patch.data < 0.5)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            first = patch.where(one, 0.0)
        assert first.attrs.processing_id == patch.where(two, 0.0).attrs.processing_id

    def test_concatenating_folds_which_data(self, spool):
        """Taking the first patch's id would claim it was the only source."""
        members = {x.attrs.patch_id for x in spool}
        merged = spool.chunk(time=None)[0]
        assert merged.attrs.patch_id not in members
        assert merged.attrs.patch_id

    def test_stacking_folds_which_data(self, spool):
        """As does adding them together."""
        members = {x.attrs.patch_id for x in spool}
        stacked = spool.stack(dim_vary="time")
        assert stacked.attrs.patch_id not in members
        assert stacked.attrs.processing_id != NOTHING_DONE

    def test_the_raw_function_bypass_is_not_recorded(self, patch):
        """
        A body which calls `.raw_function` skips the wrapper, so nothing
        is stamped -- which is what those bypasses are for, and worth
        pinning so a refactor which removes one is a visible change.
        """
        wrapped = dc.proc.normalize(patch, "time")
        raw = dc.proc.normalize.raw_function(patch, "time")
        assert wrapped.attrs.processing_id != raw.attrs.processing_id
        assert raw.attrs.processing_id == patch.attrs.processing_id


class TestCombinationEdges:
    """The branches a combination reaches only in odd cases."""

    def test_disabled_leaves_a_combination_alone(self):
        """A process not keeping ids does not stamp one on a merge."""
        patch = dc.get_example_patch()
        with dc.config_context(patch_provenance="disabled"):
            out = stamp_combination(patch.attrs, [patch.attrs], "abc")
        assert out is patch.attrs

    def test_no_members_leaves_a_combination_alone(self):
        """Nothing went in, so there is nothing to fold."""
        patch = dc.get_example_patch()
        assert stamp_combination(patch.attrs, [], "abc") is patch.attrs

    @pytest.mark.parametrize(
        "value",
        [
            pytest.param([0] * 64, id="a long sequence"),
            pytest.param({str(x): x for x in range(64)}, id="a big mapping"),
        ],
    )
    def test_a_big_argument_is_not_worth_keying(self, value):
        """
        Working the key out would cost more than the digest it saves.

        `TypeError` is how `_as_key` says "do not cache this".
        """
        with pytest.raises(TypeError):
            _as_key(value)


class TestWhatTheReviewsFound:
    """Cases the first round of this PR got wrong."""

    @pytest.fixture(scope="class")
    def patch(self):
        """A patch to operate on."""
        return dc.get_example_patch("random_das")

    def test_every_patch_argument_counts(self, patch):
        """
        `where(cond, other)` uses all of them, so all of them fold.

        Copying the primary patch's id would say a result built from three
        sources was only the first.
        """
        one = patch.new(data=patch.data > 0.5).update_attrs(patch_id="a")
        two = patch.new(data=patch.data < 0.5).update_attrs(patch_id="b")
        assert patch.where(one, 0.0).attrs.patch_id != (
            patch.where(two, 0.0).attrs.patch_id
        )

    def test_provenance_never_fails_a_call(self, patch):
        """
        It is metadata about the work, not the work.

        A self-referential argument cannot be encoded; that is a reason to
        say nothing about the call, not to fail one which otherwise worked.
        """
        loop: dict = {}
        loop["self"] = loop

        @dc.patch_function()
        def takes_anything(patch, thing=None):
            """Accept whatever it is given."""
            return patch.new(data=patch.data)

        out = takes_anything(patch, thing=loop)
        assert out.attrs.processing_id == patch.attrs.processing_id

    def test_attrs_from_before_these_fields(self, patch):
        """
        An old pickle restores a `PatchAttrs` which has neither.

        It must still be usable: unpickling bypasses the constructor, so
        nothing fills the defaults in.
        """
        bare = dc.PatchAttrs()
        held = dict(bare.model_dump())
        held.pop("patch_id"), held.pop("processing_id")
        legacy = dc.PatchAttrs.model_construct(**held)
        made = dc.Patch(data=patch.data, coords=patch.coords, dims=patch.dims)
        assert patch_id_of(legacy) == NOTHING_DONE
        assert processing_id_of(legacy) == NOTHING_DONE
        # The fold reads them without raising, and says the result is data
        # from two places even though one of them could not say which.
        folded = fold_ids([legacy, made.attrs])["patch_id"]
        assert folded and folded != made.attrs.patch_id

    def test_equality_ignores_the_ids_either_way(self, patch):
        """Both spellings of `equals`, not just the default."""
        other = patch.update_attrs(patch_id="x", processing_id="y")
        assert patch.equals(other)
        assert patch.equals(other, only_required_attrs=False)

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            pytest.param(0.0, -0.0, id="signed zero"),
            pytest.param(1, True, id="int and bool"),
            pytest.param(1, 1.0, id="int and float"),
        ],
    )
    def test_the_cache_cannot_confuse_two_arguments(self, first, second):
        """
        Python calls these equal; the serializer does not.

        Caching on one would give a call two answers depending on which
        ran first, which is the one thing an id must never do.
        """
        assert _as_key(first) != _as_key(second)

    def test_a_quantity_is_never_cached(self):
        """`1 * m == 100 * cm` and the two hash alike; they encode apart."""
        from dascore.units import m  # noqa: PLC0415

        with pytest.raises(TypeError):
            _as_key(1 * m)

    def test_two_closures_are_two_operations(self, patch):
        """
        A factory gives every function it makes one module and qualname.

        Naming them by where they were written alone would make
        `make(2)` and `make(3)` one operation.
        """

        def make(factor):
            """Return a patch function which scales by a fixed amount."""

            @dc.patch_function()
            def scale(patch):
                """Scale by whatever this closure captured."""
                return patch.new(data=patch.data * factor)

            return scale

        first, second = make(2), make(3)
        assert first(patch).attrs.processing_id != second(patch).attrs.processing_id

    def test_a_dtype_argument_is_spelled_out(self, patch):
        """
        The serializer has no encoding for a `np.dtype`.

        Hashed by its class it would give every dtype one fingerprint and
        warn on every call, so it is recorded as its string instead.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            single = np.mean(patch, axis=0, dtype=np.dtype("float32"))
            double = np.mean(patch, axis=0, dtype=np.dtype("float64"))
        assert single.attrs.processing_id != double.attrs.processing_id

    def test_the_cache_holds_the_function_it_named(self):
        """
        An unnameable function is named partly by `id(func)`, and CPython
        reuses an address once the function is collected -- so a cache
        entry outliving its function could hand a later one the earlier
        one's fingerprint.

        Asserted on the mechanism rather than by trying to collect one:
        the key holds the function, which is exactly what stops it being
        collected, so a test which deleted it and looked for a collision
        would pass whether or not the fix were there.
        """
        patch = dc.get_example_patch()

        @dc.patch_function()
        def unnameable(patch):
            """Be defined inside a call, so it takes no tag."""
            return patch.new(data=patch.data)

        unnameable(patch)
        held = [k[0] for k in processor_module._FINGERPRINTS]
        assert any(x is unnameable for x in held)

    def test_a_patch_argument_is_not_the_string_that_stands_for_it(self):
        """
        A caller may pass the marker's own spelling as an ordinary value.

        If the marker were that string the two calls would be one
        operation, though their operands are entirely different.
        """
        patch = dc.get_example_patch()
        assert fingerprint_call(dc.proc.where, (patch,), {}) != (
            fingerprint_call(dc.proc.where, ("$patch",), {})
        )
        # It says what it is, which is what the digest records of it.
        assert "patch argument" in repr(_PATCH_ARGUMENT)
