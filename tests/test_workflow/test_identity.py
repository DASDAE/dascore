"""
Tests for the two ids a patch carries.

These cover the rules themselves; the tests which pin where they are
applied live beside the things which apply them.
"""

from __future__ import annotations

import pytest

from dascore.workflow import Task
from dascore.workflow.builtin import ArrayFunc, Concatenate, Stack, Ufunc
from dascore.workflow.identity import (
    NOTHING_DONE,
    advance,
    fold_data_ids,
    fold_processing_ids,
    new_data_id,
    source_data_id,
)


class TestNewDataId:
    """Data which names no source."""

    def test_each_is_its_own(self):
        """Two arrays are two data, however alike they look."""
        assert new_data_id() != new_data_id()

    def test_it_is_a_hex_string(self):
        """The same shape as every other id, so nothing can tell them apart."""
        made = new_data_id()
        assert isinstance(made, str)
        assert int(made, 16) >= 0


class TestSourceDataId:
    """Data read from a file."""

    def test_it_is_derived(self):
        """Reading the same file twice is reading the same data."""
        first = source_data_id("DASDAE", "/data/one.h5", 0)
        assert first == source_data_id("DASDAE", "/data/one.h5", 0)

    @pytest.mark.parametrize(
        ("format_name", "path", "key"),
        [
            ("TERRA15", "/data/one.h5", 0),
            ("DASDAE", "/data/two.h5", 0),
            ("DASDAE", "/data/one.h5", 1),
            ("DASDAE", "/data/one.h5", None),
        ],
    )
    def test_every_part_counts(self, format_name, path, key):
        """The format, the path and the key each name different data."""
        base = source_data_id("DASDAE", "/data/one.h5", 0)
        assert source_data_id(format_name, path, key) != base

    def test_a_key_may_be_anything_encodable(self):
        """A reader names its patches however it likes."""
        assert source_data_id("X", "/a", "channel-3") != source_data_id("X", "/a", 3)


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
        assert fold_data_ids(["only"]) == "only"

    def test_order_is_part_of_it(self):
        """Concatenating a before b is not concatenating b before a."""
        assert fold_data_ids(["a", "b"]) != fold_data_ids(["b", "a"])

    def test_repeats_are_part_of_it(self):
        """Stacking a patch with itself is not the patch."""
        assert fold_data_ids(["a", "a"]) != fold_data_ids(["a"])

    def test_it_is_derived(self):
        """So the same combination gives the same answer twice."""
        assert fold_data_ids(["a", "b"]) == fold_data_ids(["a", "b"])

    def test_it_takes_any_sequence(self):
        """Callers hold their members in whatever they hold them in."""
        assert fold_data_ids(("a", "b")) == fold_data_ids(["a", "b"])


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
