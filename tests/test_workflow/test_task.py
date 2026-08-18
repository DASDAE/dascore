"""Tests for the Task base class and the tasks a function makes."""

from __future__ import annotations

import json
import multiprocessing
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from pydantic import Field, ValidationError

from dascore.exceptions import InvalidModelTagError, ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from dascore.workflow import (
    Task,
    decode,
    encode,
    intern,
    make_function_task_class,
    task,
)


class ScaleTask(Task):
    """Multiply what it is given by a factor."""

    factor: float = 1.0

    def run(self, value):
        """Scale a value."""
        return value * self.factor


class ShiftTask(Task):
    """Add an offset to what it is given."""

    offset: float = 0.0


class VersionedScaleTask(ScaleTask):
    """The same parameters, said to mean something else."""

    __version__ = "2.0"


class TupleFieldTask(Task):
    """A task with a field which declares itself a tuple."""

    values: tuple[int, ...] = ()


class TimedValueTask(Task):
    """A task parametrized by the values a fingerprint has to normalize."""

    when: object = None
    where: object = None


@task
def add_numbers(first, second=1):
    """Add two numbers."""
    return first + second


@task(version="2.0")
def multiply_numbers(first, second=2):
    """Multiply two numbers."""
    return first * second


def every_kind(positional, /, normal=1, *rest, named=2, **extra):
    """Take one of every kind of parameter python has."""
    return (positional, normal, rest, named, extra)


def with_field_default(value=Field(default=7)):
    """Spell a default the way a pydantic model would."""
    return value


def skips_first(given, /, normal=1, *rest, named=2, **extra):
    """Take an input at run time, then one of every other kind."""
    return (given, normal, rest, named, extra)


EveryKind = make_function_task_class(every_kind)
SkipsFirst = make_function_task_class(skips_first, skip_first=True)
WithFieldDefault = make_function_task_class(with_field_default)


def _task_class(name, module):
    """Return a task class which believes it lives in a given module."""
    namespace = {"__annotations__": {"factor": float}, "__module__": module}
    return type(name, (Task,), namespace)


def _fingerprint_of(document):
    """Rebuild a task from its document and return its fingerprint."""
    return Task.from_dict(document).fingerprint


def _fingerprint(task_):
    """Return a task's fingerprint; run in another process."""
    return task_.fingerprint


class TestFingerprint:
    """Tests for what a task's fingerprint answers to."""

    def test_stable_across_constructions(self):
        """The same task built twice fingerprints alike."""
        assert ScaleTask(factor=2).fingerprint == ScaleTask(factor=2).fingerprint

    def test_hard_coded(self):
        """
        A fingerprint does not drift between releases.

        Stored ids depend on this: the same call has to give the same answer
        after a refactor, so the value is pinned rather than merely stable
        within one run.
        """
        assert ScaleTask(factor=2).fingerprint == "761d418c84549e16"

    def test_params_matter(self):
        """Two different calls are two different tasks."""
        assert ScaleTask(factor=2).fingerprint != ScaleTask(factor=3).fingerprint

    def test_default_is_a_value(self):
        """Passing a default explicitly is the same call as leaving it out."""
        assert ScaleTask(factor=1.0).fingerprint == ScaleTask().fingerprint

    def test_class_matters(self):
        """Two classes holding the same fields are two tasks."""
        assert ScaleTask(factor=1).fingerprint != ShiftTask(offset=1).fingerprint

    def test_version_matters(self):
        """A version bump says the same call means something else."""
        assert (
            ScaleTask(factor=2).fingerprint != VersionedScaleTask(factor=2).fingerprint
        )

    def test_survives_a_move(self):
        """Moving a class between modules does not rename it."""
        # Both live in one package under one name, which is exactly the
        # collision the registry warns about; here it is the point.
        with suppress_warnings(UserWarning, message="Two models claim"):
            first = _task_class("MovedScaleTask", "tests.first_home")
            second = _task_class("MovedScaleTask", "tests.second_home")
        assert first(factor=2).fingerprint == second(factor=2).fingerprint

    def test_namespaced(self):
        """A task outside dascore fingerprints under its own namespace."""
        inside = _task_class("NamespacedScaleTask", "tests.tasks")
        outside = _task_class("NamespacedScaleTask", "someplugin.tasks")
        assert inside(factor=2).fingerprint != outside(factor=2).fingerprint

    def test_cached(self):
        """The fingerprint is computed once per instance."""
        task_ = ScaleTask(factor=2)
        assert task_.fingerprint is task_.fingerprint

    def test_time_units_normalized(self):
        """A parameter's fingerprint goes through the serializer."""
        day = TimedValueTask(when=np.datetime64("2020-01-01"))
        nanosecond = TimedValueTask(when=np.datetime64("2020-01-01T00:00:00.000000000"))
        assert day.fingerprint == nanosecond.fingerprint

    def test_none_parameters_dropped(self):
        """A parameter left at None is the same call as one left out."""
        assert TimedValueTask(when=None).fingerprint == TimedValueTask().fingerprint

    def test_array_parameter(self):
        """An array parameter is hashed by its values."""
        first = TimedValueTask(when=np.arange(3))
        assert first.fingerprint != TimedValueTask(when=np.arange(4)).fingerprint
        assert first.fingerprint == TimedValueTask(when=np.arange(3)).fingerprint

    def test_nested_task_parameter(self):
        """A task given to a task is part of its parameters."""
        first = TimedValueTask(when=ScaleTask(factor=2))
        assert first.fingerprint != TimedValueTask(when=ScaleTask(factor=3)).fingerprint
        assert first.fingerprint == TimedValueTask(when=ScaleTask(factor=2)).fingerprint


class TestEquality:
    """Tests for how tasks compare."""

    def test_equal_tasks(self):
        """Two tasks with the same parameters are the same task."""
        assert ScaleTask(factor=2) == ScaleTask(factor=2)

    def test_different_params(self):
        """Two calls of one task are not each other."""
        assert ScaleTask(factor=2) != ScaleTask(factor=3)

    def test_different_classes(self):
        """Two classes holding the same field values are not equal."""
        assert ScaleTask(factor=1) != VersionedScaleTask(factor=1)

    def test_not_a_task(self):
        """Comparison with something else is left to the something else."""
        assert ScaleTask(factor=1).__eq__("ScaleTask(factor=1)") is (NotImplemented)

    def test_hashes_with_equality(self):
        """Equal tasks land in the same place in a set."""
        assert len({ScaleTask(factor=2), ScaleTask(factor=2)}) == 1
        assert len({ScaleTask(factor=2), ScaleTask(factor=3)}) == 2

    def test_hashable_with_array_params(self):
        """A task holding an array hashes, where its fields alone could not."""
        first = TimedValueTask(when=np.arange(3))
        assert hash(first) == hash(TimedValueTask(when=np.arange(3)))


class TestUpdate:
    """Tests for making one task from another."""

    def test_returns_new_task(self):
        """Updating leaves the original alone."""
        first = ScaleTask(factor=2)
        second = first.update(factor=3)
        assert first.factor == 2
        assert second.factor == 3

    def test_validates(self):
        """An update goes through validation."""
        with pytest.raises(ValidationError):
            ScaleTask(factor=2).update(factor="not a number")

    def test_unknown_parameter(self):
        """A task refuses a parameter it does not have."""
        with pytest.raises(ValidationError):
            ScaleTask(factor=2).update(not_a_field=1)


class TestDocuments:
    """Tests for writing a task down and reading it back."""

    def test_round_trip(self):
        """A task rebuilt from its document is the same task."""
        original = ScaleTask(factor=2)
        assert Task.from_dict(original.to_dict()) == original

    def test_names_the_class_by_tag(self):
        """A document names a registered tag, not an import path."""
        document = ScaleTask(factor=2).to_dict()
        assert document["object_type"] == "tests:ScaleTask"

    def test_code_path_is_a_hint(self):
        """The import hint is written but does not change the fingerprint."""
        document = ScaleTask(factor=2).to_dict()
        assert "code_path" in document
        document["code_path"] = "somewhere:else"
        assert _fingerprint_of(document) == ScaleTask(factor=2).fingerprint

    def test_local_class_refused(self):
        """A task class defined in a function cannot be written down."""

        def local(value=1):
            """A function defined inside a test."""
            return value

        with pytest.raises(ParameterError, match="module level"):
            make_function_task_class(local)().to_dict()

    def test_version_change_warns(self):
        """Reading a document written by another version says so."""
        document = ScaleTask(factor=2).to_dict()
        document["version"] = "0.5"
        with pytest.warns(DASCoreWarning, match="version"):
            Task.from_dict(document)

    def test_unknown_tag_raises(self):
        """A tag nothing registers is an error, not an import."""
        document = ScaleTask(factor=2).to_dict()
        document["object_type"] = "NotATaskAnyoneHas"
        with pytest.raises(InvalidModelTagError):
            Task.from_dict(document)

    def test_tag_naming_another_model(self):
        """A tag which names something other than a task is refused."""
        document = ScaleTask(factor=2).to_dict()
        document["object_type"] = "PatchAttrs"
        with pytest.raises(ParameterError, match="not a Task"):
            Task.from_dict(document)

    def test_nested_task_round_trip(self):
        """A task given to a task comes back as itself."""
        original = TimedValueTask(when=ScaleTask(factor=2))
        assert Task.from_dict(original.to_dict()).when == ScaleTask(factor=2)

    def test_nested_task_fingerprint_cannot_decode(self):
        """A nested task hashed for a fingerprint cannot be read back."""
        encoded = encode(ScaleTask(factor=2))
        with pytest.raises(ParameterError, match="cannot be read back"):
            decode(encoded)

    def test_declared_tuple_field(self):
        """A JSON list comes back a tuple where the field says tuple."""
        original = TupleFieldTask(values=(1, 2))
        assert Task.from_dict(original.to_dict()).values == (1, 2)

    def test_undeclared_sequence_stays_a_list(self):
        """A field which promises nothing gets the list a document holds."""
        assert Task.from_dict(TimedValueTask(when=(1, 2)).to_dict()).when == [1, 2]

    def test_time_parameter_round_trip(self):
        """A parameter the serializer tags comes back as itself."""
        original = TimedValueTask(when=np.datetime64("2020-01-01"), where=slice(1, 2))
        rebuilt = Task.from_dict(original.to_dict())
        assert rebuilt.when == original.when
        assert rebuilt.where == original.where

    def test_document_is_json_safe(self):
        """A document holds only what JSON can write."""
        document = TimedValueTask(when=np.arange(3)).to_dict()
        assert json.loads(json.dumps(document)) == document


class TestPickle:
    """Tests for moving a task between processes."""

    def test_round_trip(self):
        """A task survives a pickle."""
        original = ScaleTask(factor=2)
        assert pickle.loads(pickle.dumps(original)) == original

    def test_synthesized_class(self):
        """A class no attribute path names still pickles, by its tag."""
        original = add_numbers(first=1, second=2)
        assert pickle.loads(pickle.dumps(original)) == original

    @pytest.mark.concurrency
    def test_across_processes(self):
        """A task pickled into another process fingerprints the same."""
        original = ScaleTask(factor=2)
        # Spawn rather than the platform default: the task has to survive
        # being rebuilt in an interpreter which shares nothing with this one.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
            out = pool.submit(_fingerprint, original).result()
        assert out == original.fingerprint

    def test_callable_parameter(self):
        """A parameter a document cannot hold still pickles."""
        original = add_numbers(first=1, second=np.mean)
        assert pickle.loads(pickle.dumps(original)) == original

    def test_local_class_refused(self):
        """A class no document could name says so when it is pickled."""

        def local(value=1):
            """A function defined inside a test."""
            return value

        with pytest.raises(ParameterError, match="module level"):
            pickle.dumps(make_function_task_class(local)())


class TestIntern:
    """Tests for sharing one instance of a task."""

    def test_returns_one_instance(self):
        """Equal tasks intern to one object."""
        assert intern(ScaleTask(factor=2)) is intern(ScaleTask(factor=2))

    def test_keeps_them_apart(self):
        """Tasks which are not equal keep their own instances."""
        assert intern(ScaleTask(factor=2)) is not intern(ScaleTask(factor=3))

    def test_first_wins(self):
        """The instance which arrived first is the one handed back."""
        first = ScaleTask(factor=4)
        assert intern(first) is first
        assert intern(ScaleTask(factor=4)) is first


class TestRun:
    """Tests for running a task."""

    def test_base_class_has_no_body(self):
        """A task which does not implement run says so."""
        with pytest.raises(NotImplementedError, match="does not implement run"):
            ShiftTask(offset=1).run(1)

    def test_subclass_body(self):
        """A subclass runs what it implements."""
        assert ScaleTask(factor=2).run(3) == 6


class TestFunctionTasks:
    """Tests for the task a function makes."""

    def test_class_name(self):
        """The class is named for the function, in camel case."""
        assert EveryKind.__name__ == "EveryKind"

    def test_docstring_kept(self):
        """The class carries the function's docstring."""
        assert add_numbers.__doc__ == "Add two numbers."

    def test_runs_the_function(self):
        """Running the task calls the function with its fields."""
        assert add_numbers(first=1, second=2).run() == 3

    def test_defaults(self):
        """A parameter's default becomes the field's default."""
        assert add_numbers(first=1).run() == 2

    def test_required_parameter(self):
        """A parameter with no default is required."""
        with pytest.raises(ValidationError):
            add_numbers()

    def test_version(self):
        """The decorator's version reaches the class."""
        assert multiply_numbers.__version__ == "2.0"
        assert multiply_numbers(first=2).run() == 4

    def test_extra_input_passed_first(self):
        """Arguments given to run are passed before the stored ones."""
        assert SkipsFirst(normal=5).run("given")[0] == "given"

    def test_field_default_resolved(self):
        """A default spelled as a pydantic Field is the value it holds."""
        assert WithFieldDefault().value == 7

    def test_call_plan(self):
        """Only what cannot go by name is passed positionally."""
        # `positional` is positional-only, and `normal` precedes the *args
        # group, so both have to be passed in order; the rest go by name.
        assert EveryKind._positional_names == (
            ("positional", False),
            ("normal", False),
            ("rest", True),
        )
        assert EveryKind._keyword_names == ("named",)

    def test_keyword_parameters_passed_by_name(self):
        """A parameter which can go by name does, whatever its position."""
        cls = make_function_task_class(lambda first=1, second=2: (first, second))
        assert cls._positional_names == ()
        assert cls._keyword_names == ("first", "second")

    def test_all_parameter_kinds(self):
        """Every kind of parameter survives the trip through a task."""
        instance = EveryKind(
            positional="p",
            normal=5,
            rest=(1, 2),
            named=9,
            an_extra="e",
        )
        assert instance.run() == ("p", 5, (1, 2), 9, {"an_extra": "e"})

    def test_extras_are_parameters(self):
        """What a **kwargs group collects is part of the task."""
        first = EveryKind(positional="p", an_extra=1)
        assert first.fingerprint != EveryKind(positional="p", an_extra=2).fingerprint
        assert first.fingerprint != EveryKind(positional="p").fingerprint
        assert Task.from_dict(first.to_dict()) == first
        assert pickle.loads(pickle.dumps(first)) == first

    def test_extras_allowed_only_with_kwargs(self):
        """A function without **kwargs makes a task which refuses extras."""
        with pytest.raises(ValidationError):
            add_numbers(first=1, not_a_parameter=2)

    def test_from_call_matches_construction(self):
        """A task built from a call equals the task built by hand."""
        from_call = EveryKind._from_call(("p", 5, 1, 2), {"named": 9, "an_extra": "e"})
        by_hand = EveryKind(
            positional="p", normal=5, rest=(1, 2), named=9, an_extra="e"
        )
        assert from_call.fingerprint == by_hand.fingerprint

    def test_mutable_default_not_shared(self):
        """A task does not hold the function's own default object."""

        def mutable(value=[]):
            """Take a mutable default, as a careless caller might."""
            return value

        cls = make_function_task_class(mutable)
        first = cls._from_call((), {})
        first.value.append("changed")
        assert cls._from_call((), {}).value == []

    def test_from_call_applies_defaults(self):
        """A call which leaves a parameter out still records its value."""
        assert (
            add_numbers._from_call((1,), {}).fingerprint
            == add_numbers(first=1, second=1).fingerprint
        )

    def test_from_call_skips_first(self):
        """The input a task is given is not one of its parameters."""
        cls = SkipsFirst
        assert "positional" not in cls.model_fields
        # The arguments are the ones left after the input, as _call_args
        # gives them back.
        assert cls._from_call((5,), {}).normal == 5

    def test_shadowing_field_name(self):
        """A parameter named for a model attribute is still a field."""

        def shadowing(copy=True):
            """Take a parameter named after BaseModel.copy."""
            return copy

        cls = make_function_task_class(shadowing)
        assert cls().run() is True


class TestTaskDecorator:
    """Tests for the decorator spelling."""

    def test_bare(self):
        """The decorator works without parentheses."""

        @task
        def bare(value=1):
            """A task made without parentheses."""
            return value

        assert bare(value=2).run() == 2

    def test_with_version(self):
        """The decorator works with arguments."""

        @task(version="3.0")
        def versioned(value=1):
            """A task made with a version."""
            return value

        assert versioned.__version__ == "3.0"


class TestTag:
    """Tests for what names a task."""

    def test_task_tag(self):
        """A task outside dascore is namespaced by its package."""
        assert ScaleTask(factor=1).tag == "tests:ScaleTask"
