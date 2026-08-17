"""Tests for the Task base class and the tasks a function makes."""

from __future__ import annotations

import json
import pickle
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from pydantic import Field, ValidationError

import dascore as dc
from dascore.exceptions import InvalidModelTagError, ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.workflow import (
    Task,
    decode,
    encode,
    intern,
    make_function_task_class,
    task,
)


class Scale(Task):
    """Multiply what it is given by a factor."""

    factor: float = 1.0

    def run(self, value):
        """Scale a value."""
        return value * self.factor


class Shift(Task):
    """Add an offset to what it is given."""

    offset: float = 0.0


class VersionedScale(Scale):
    """The same parameters, said to mean something else."""

    __version__ = "2.0"


class TupleTask(Task):
    """A task with a field which declares itself a tuple."""

    values: tuple[int, ...] = ()


class TimedTask(Task):
    """A task parametrized by the values a fingerprint has to normalize."""

    when: object = None
    where: object = None


@task
def add(first, second=1):
    """Add two numbers."""
    return first + second


@task(version="2.0")
def multiply(first, second=2):
    """Multiply two numbers."""
    return first * second


def every_kind(positional, /, normal=1, *rest, named=2, **extra):
    """Take one of every kind of parameter python has."""
    return (positional, normal, rest, named, extra)


def with_field_default(value=Field(default=7)):
    """Spell a default the way a pydantic model would."""
    return value


EveryKind = make_function_task_class(every_kind)
WithFieldDefault = make_function_task_class(with_field_default)


def _task_class(name, module):
    """Return a task class which believes it lives in a given module."""
    namespace = {"__annotations__": {"factor": float}, "__module__": module}
    return type(name, (Task,), namespace)


def _fingerprint_of(document):
    """Rebuild a task from its document and return its fingerprint."""
    return Task.from_dict(document).fingerprint


class TestFingerprint:
    """Tests for what a task's fingerprint answers to."""

    def test_stable_across_constructions(self):
        """The same task built twice fingerprints alike."""
        assert Scale(factor=2).fingerprint == Scale(factor=2).fingerprint

    def test_hard_coded(self):
        """
        A fingerprint does not drift between releases.

        Stored ids depend on this: the same call has to give the same answer
        after a refactor, so the value is pinned rather than merely stable
        within one run.
        """
        assert Scale(factor=2).fingerprint == "7a9e5fd2be41f9d4"

    def test_params_matter(self):
        """Two different calls are two different tasks."""
        assert Scale(factor=2).fingerprint != Scale(factor=3).fingerprint

    def test_default_is_a_value(self):
        """Passing a default explicitly is the same call as leaving it out."""
        assert Scale(factor=1.0).fingerprint == Scale().fingerprint

    def test_class_matters(self):
        """Two classes holding the same fields are two tasks."""
        assert Scale(factor=1).fingerprint != Shift(offset=1).fingerprint

    def test_version_matters(self):
        """A version bump says the same call means something else."""
        assert Scale(factor=2).fingerprint != VersionedScale(factor=2).fingerprint

    def test_survives_a_move(self):
        """Moving a class between modules does not rename it."""
        # Both live in one package under one name, which is exactly the
        # collision the registry warns about; here it is the point.
        with suppress_warnings(UserWarning, message="Two models claim"):
            first = _task_class("MovedScale", "tests.first_home")
            second = _task_class("MovedScale", "tests.second_home")
        assert first(factor=2).fingerprint == second(factor=2).fingerprint

    def test_namespaced(self):
        """A task outside dascore fingerprints under its own namespace."""
        inside = _task_class("NamespacedScale", "tests.tasks")
        outside = _task_class("NamespacedScale", "someplugin.tasks")
        assert inside(factor=2).fingerprint != outside(factor=2).fingerprint

    def test_cached(self):
        """The fingerprint is computed once per instance."""
        task_ = Scale(factor=2)
        assert task_.fingerprint is task_.fingerprint

    def test_time_units_normalized(self):
        """A parameter's fingerprint goes through the serializer."""
        day = TimedTask(when=np.datetime64("2020-01-01"))
        nanosecond = TimedTask(when=np.datetime64("2020-01-01T00:00:00.000000000"))
        assert day.fingerprint == nanosecond.fingerprint

    def test_none_parameters_dropped(self):
        """A parameter left at None is the same call as one left out."""
        assert TimedTask(when=None).fingerprint == TimedTask().fingerprint

    def test_array_parameter(self):
        """An array parameter is hashed by its values."""
        first = TimedTask(when=np.arange(3))
        assert first.fingerprint != TimedTask(when=np.arange(4)).fingerprint
        assert first.fingerprint == TimedTask(when=np.arange(3)).fingerprint

    def test_nested_task_parameter(self):
        """A task given to a task is part of its parameters."""
        first = TimedTask(when=Scale(factor=2))
        assert first.fingerprint != TimedTask(when=Scale(factor=3)).fingerprint
        assert first.fingerprint == TimedTask(when=Scale(factor=2)).fingerprint


class TestEquality:
    """Tests for how tasks compare."""

    def test_equal_tasks(self):
        """Two tasks with the same parameters are the same task."""
        assert Scale(factor=2) == Scale(factor=2)

    def test_different_params(self):
        """Two calls of one task are not each other."""
        assert Scale(factor=2) != Scale(factor=3)

    def test_different_classes(self):
        """Two classes holding the same field values are not equal."""
        assert Scale(factor=1) != VersionedScale(factor=1)

    def test_not_a_task(self):
        """Something which is not a task is not equal to one."""
        assert Scale(factor=1) != "Scale(factor=1)"

    def test_hashes_with_equality(self):
        """Equal tasks land in the same place in a set."""
        assert len({Scale(factor=2), Scale(factor=2)}) == 1
        assert len({Scale(factor=2), Scale(factor=3)}) == 2

    def test_hashable_with_array_params(self):
        """A task holding an array is still hashable."""
        assert hash(TimedTask(when=np.arange(3))) is not None


class TestUpdate:
    """Tests for making one task from another."""

    def test_returns_new_task(self):
        """Updating leaves the original alone."""
        first = Scale(factor=2)
        second = first.update(factor=3)
        assert first.factor == 2
        assert second.factor == 3

    def test_validates(self):
        """An update goes through validation."""
        with pytest.raises(ValidationError):
            Scale(factor=2).update(factor="not a number")

    def test_unknown_parameter(self):
        """A task refuses a parameter it does not have."""
        with pytest.raises(ValidationError):
            Scale(factor=2).update(not_a_field=1)


class TestDocuments:
    """Tests for writing a task down and reading it back."""

    def test_round_trip(self):
        """A task rebuilt from its document is the same task."""
        original = Scale(factor=2)
        assert Task.from_dict(original.to_dict()) == original

    def test_names_the_class_by_tag(self):
        """A document names a registered tag, not an import path."""
        document = Scale(factor=2).to_dict()
        assert document["object_type"] == "tests:Scale"

    def test_code_path_is_a_hint(self):
        """The import hint is written but does not change the fingerprint."""
        document = Scale(factor=2).to_dict()
        assert "code_path" in document
        document["code_path"] = "somewhere:else"
        assert _fingerprint_of(document) == Scale(factor=2).fingerprint

    def test_unknown_tag_raises(self):
        """A tag nothing registers is an error, not an import."""
        document = Scale(factor=2).to_dict()
        document["object_type"] = "NotATaskAnyoneHas"
        with pytest.raises(InvalidModelTagError):
            Task.from_dict(document)

    def test_tag_naming_another_model(self):
        """A tag which names something other than a task is refused."""
        document = Scale(factor=2).to_dict()
        document["object_type"] = "PatchAttrs"
        with pytest.raises(ParameterError, match="not a Task"):
            Task.from_dict(document)

    def test_nested_task_round_trip(self):
        """A task given to a task comes back as itself."""
        original = TimedTask(when=Scale(factor=2))
        assert Task.from_dict(original.to_dict()).when == Scale(factor=2)

    def test_nested_task_fingerprint_cannot_decode(self):
        """A nested task hashed for a fingerprint cannot be read back."""
        encoded = encode(Scale(factor=2))
        with pytest.raises(ParameterError, match="cannot be read back"):
            decode(encoded)

    def test_declared_tuple_field(self):
        """A JSON list comes back a tuple where the field says tuple."""
        original = TupleTask(values=(1, 2))
        assert Task.from_dict(original.to_dict()).values == (1, 2)

    def test_undeclared_sequence_stays_a_list(self):
        """A field which promises nothing gets the list a document holds."""
        assert Task.from_dict(TimedTask(when=(1, 2)).to_dict()).when == [1, 2]

    def test_time_parameter_round_trip(self):
        """A parameter the serializer tags comes back as itself."""
        original = TimedTask(when=np.datetime64("2020-01-01"), where=slice(1, 2))
        rebuilt = Task.from_dict(original.to_dict())
        assert rebuilt.when == original.when
        assert rebuilt.where == original.where

    def test_document_is_json_safe(self):
        """A document holds only what JSON can write."""
        document = TimedTask(when=np.arange(3)).to_dict()
        assert json.loads(json.dumps(document)) == document


class TestPickle:
    """Tests for moving a task between processes."""

    def test_round_trip(self):
        """A task survives a pickle."""
        original = Scale(factor=2)
        assert pickle.loads(pickle.dumps(original)) == original

    def test_synthesized_class(self):
        """A class no attribute path names still pickles, by its tag."""
        original = add(first=1, second=2)
        assert pickle.loads(pickle.dumps(original)) == original

    def test_across_processes(self):
        """A task rebuilt in another process fingerprints the same."""
        original = Scale(factor=2)
        with ProcessPoolExecutor(max_workers=1) as pool:
            out = pool.submit(_fingerprint_of, original.to_dict()).result()
        assert out == original.fingerprint


class TestIntern:
    """Tests for sharing one instance of a task."""

    def test_returns_one_instance(self):
        """Equal tasks intern to one object."""
        assert intern(Scale(factor=2)) is intern(Scale(factor=2))

    def test_keeps_them_apart(self):
        """Tasks which are not equal keep their own instances."""
        assert intern(Scale(factor=2)) is not intern(Scale(factor=3))

    def test_first_wins(self):
        """The instance which arrived first is the one handed back."""
        first = Scale(factor=4)
        assert intern(first) is first
        assert intern(Scale(factor=4)) is first


class TestRun:
    """Tests for running a task."""

    def test_base_class_has_no_body(self):
        """A task which does not implement run says so."""
        with pytest.raises(NotImplementedError, match="does not implement run"):
            Shift(offset=1).run(1)

    def test_subclass_body(self):
        """A subclass runs what it implements."""
        assert Scale(factor=2).run(3) == 6


class TestFunctionTasks:
    """Tests for the task a function makes."""

    def test_class_name(self):
        """The class is named for the function, in camel case."""
        assert make_function_task_class(every_kind).__name__ == "EveryKind"

    def test_docstring_kept(self):
        """The class carries the function's docstring."""
        assert add.__doc__ == "Add two numbers."

    def test_runs_the_function(self):
        """Running the task calls the function with its fields."""
        assert add(first=1, second=2).run() == 3

    def test_defaults(self):
        """A parameter's default becomes the field's default."""
        assert add(first=1).run() == 2

    def test_required_parameter(self):
        """A parameter with no default is required."""
        with pytest.raises(ValidationError):
            add()

    def test_version(self):
        """The decorator's version reaches the class."""
        assert multiply.__version__ == "2.0"
        assert multiply(first=2).run() == 4

    def test_extra_input_passed_first(self):
        """Arguments given to run are passed before the stored ones."""
        double = make_function_task_class(every_kind, skip_first=True)
        assert double(normal=5).run("given")[0] == "given"

    def test_field_default_resolved(self):
        """A default spelled as a pydantic Field is the value it holds."""
        assert WithFieldDefault().value == 7

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

    def test_extras_allowed_only_with_kwargs(self):
        """A function without **kwargs makes a task which refuses extras."""
        with pytest.raises(ValidationError):
            add(first=1, not_a_parameter=2)

    def test_from_call_matches_construction(self):
        """A task built from a call equals the task built by hand."""
        from_call = EveryKind._from_call(("p", 5, 1, 2), {"named": 9, "an_extra": "e"})
        by_hand = EveryKind(
            positional="p", normal=5, rest=(1, 2), named=9, an_extra="e"
        )
        assert from_call.fingerprint == by_hand.fingerprint

    def test_from_call_applies_defaults(self):
        """A call which leaves a parameter out still records its value."""
        assert (
            add._from_call((1,), {}).fingerprint == add(first=1, second=1).fingerprint
        )

    def test_from_call_skips_first(self):
        """The input a task is given is not one of its parameters."""
        cls = make_function_task_class(every_kind, skip_first=True)
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

    def test_defined_in_a_function_is_not_registered(self):
        """A class which only exists inside a call cannot be named."""

        def local(value=1):
            """A function defined inside a test."""
            return value

        cls = make_function_task_class(local)
        with pytest.raises(InvalidModelTagError):
            Task.from_dict(cls().to_dict())


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

    def test_dascore_tags_are_bare(self):
        """A model registers under a bare name inside dascore."""
        assert dc.PatchAttrs().model_dump(mode="json")["object_type"] == "PatchAttrs"

    def test_task_tag(self):
        """A task outside dascore is namespaced by its package."""
        assert Scale(factor=1).tag == "tests:Scale"
