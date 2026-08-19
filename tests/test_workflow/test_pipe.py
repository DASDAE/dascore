"""Tests for pipes: tasks arranged into the shape of one operation."""

from __future__ import annotations

import pickle
from typing import ClassVar

import pytest
from pydantic import Field, ValidationError

from dascore.exceptions import ParameterError
from dascore.utils.misc import suppress_warnings
from dascore.warnings import DASCoreWarning
from dascore.workflow import Pipe, Task, decode, encode


class AddTask(Task):
    """Add a number to what it is given."""

    value: int = 1

    def run(self, number):
        """Add to a number."""
        return number + self.value


class TimesTask(Task):
    """Multiply what it is given by a number."""

    value: int = 2

    def run(self, number):
        """Multiply a number."""
        return number * self.value


class JoinTask(Task):
    """Add up everything it is given."""

    def run(self, *numbers):
        """Add up numbers."""
        return sum(numbers)


class SubtractTask(Task):
    """Subtract what it is given second from what it is given first."""

    def run(self, first, second):
        """Subtract one number from another."""
        return first - second


class PortedTask(Task):
    """
    A task from another package, carrying declarations of its own.

    Stands for what derzug's ported tasks look like: a subclass which adds
    class level declarations a pipe knows nothing about, and which has to
    round trip through the tag registry all the same.
    """

    inputs_ports: ClassVar[tuple[str, ...]] = ("left", "right")
    output_ports: ClassVar[tuple[str, ...]] = ("result",)

    value: int = 1

    def run(self, number):
        """Add to a number."""
        return number + self.value


class VersionedTask(Task):
    """A task whose version a test moves on after writing a document."""

    value: int = 1

    def run(self, number):
        """Add to a number."""
        return number + self.value


class CarryingTask(Task):
    """A task holding a mapping which looks like a document of its own."""

    meta: dict = Field(default_factory=dict)

    def run(self, number):
        """Hand back what it was given."""
        return number


class ConstantTask(Task):
    """Make a number out of nothing; a source of its own."""

    value: int = 0

    def run(self):
        """Return the number."""
        return self.value


class NestedPipeTask(Task):
    """A task which holds a whole pipe as one of its parameters."""

    inner: Pipe

    def run(self, number):
        """Run the pipe this task holds."""
        return self.inner.run(number)


@pytest.fixture
def chain():
    """A pipe of two tasks, one after the other."""
    return AddTask(value=2) | TimesTask(value=3)


@pytest.fixture
def branched():
    """A pipe whose two branches feed one task."""
    return (AddTask(value=1), TimesTask(value=5)) | JoinTask()


@pytest.fixture
def crossed():
    """
    A pipe fed by two branches, in an order nothing else would pick.

    Its nodes are named so that sorting them gives the other order, and the
    task joining them tells its inputs apart, so the pipe answers
    differently if either the wiring or the source order is lost.
    """
    return (TimesTask(value=5), AddTask(value=1)) | SubtractTask()


@pytest.fixture
def fanned():
    """A pipe which splits into two results and merges neither."""
    return AddTask(value=1) | (TimesTask(value=2), TimesTask(value=3))


@pytest.fixture
def diamond():
    """A pipe which splits into two branches and joins them again."""
    return AddTask(value=1) | (TimesTask(value=2), TimesTask(value=3)) | JoinTask()


class TestBuilding:
    """Tests for putting a pipe together."""

    def test_two_tasks(self, chain):
        """Two tasks joined make a pipe of two."""
        assert isinstance(chain, Pipe)
        assert len(chain) == 2

    def test_extend_a_pipe(self, chain):
        """A pipe takes another task on the end."""
        assert len(chain | AddTask(value=7)) == 3

    def test_prepend_a_task(self, chain):
        """A task in front of a pipe runs first."""
        pipe = AddTask(value=7) | chain
        assert pipe.run(0) == (0 + 7 + 2) * 3

    def test_join_two_pipes(self, chain):
        """Two pipes joined run one after the other."""
        pipe = chain | (AddTask(value=1) | TimesTask(value=2))
        assert len(pipe) == 4

    def test_repeated_task(self):
        """The same task twice is two steps, not one."""
        pipe = AddTask(value=1) | AddTask(value=1)
        assert len(pipe) == 2
        assert pipe.run(0) == 2

    def test_branches_into_a_pipe(self, chain):
        """Two tasks can feed a pipe rather than a single task."""
        pipe = (AddTask(value=1), AddTask(value=2)) | (JoinTask() | TimesTask(value=2))
        assert pipe.run(1) == ((1 + 1) + (1 + 2)) * 2

    def test_joining_something_else(self):
        """A pipe joins tasks and pipes, and says so otherwise."""
        with pytest.raises(ParameterError, match="not str"):
            AddTask() | "not a task"

    def test_joining_nothing(self):
        """A pipe cannot be fed by an empty list of branches."""
        with pytest.raises(ParameterError, match="joined to nothing"):
            () | JoinTask()

    def test_joining_two_fans(self):
        """Several results into several places have no one pairing."""
        with pytest.raises(ParameterError, match="one way to pair them"):
            (AddTask(value=1) | (TimesTask(value=2), TimesTask(value=3))) | (
                (AddTask(value=1), AddTask(value=2)) | JoinTask()
            )

    def test_joining_to_nothing(self):
        """A pipe cannot fan out into an empty list of branches."""
        with pytest.raises(ParameterError, match="joined to nothing"):
            AddTask() | ()


class TestNodeKeys:
    """Tests for what a pipe calls its nodes."""

    def test_named_for_the_task(self, chain):
        """A node is named for its task, said the way a variable is."""
        assert set(chain.tasks) == {"add_task", "times_task"}

    def test_repeated_task_numbered(self):
        """The second copy of a task is numbered rather than merged."""
        pipe = AddTask(value=1) | AddTask(value=2)
        assert list(pipe.tasks) == ["add_task", "add_task_2"]

    def test_key_ignores_parameters(self):
        """Two pipes differing only in a parameter name their nodes alike."""
        first = AddTask(value=1) | TimesTask(value=2)
        second = AddTask(value=9) | TimesTask(value=2)
        assert list(first.tasks) == list(second.tasks)

    def test_relabel(self, chain):
        """A node can be given a name of the caller's own."""
        renamed = chain.relabel(add_task="offset")
        assert set(renamed.tasks) == {"offset", "times_task"}
        assert renamed.run(1) == chain.run(1)

    def test_relabel_keeps_the_fingerprint(self, diamond):
        """A name is for the reader; the pipe is what it always was."""
        renamed = diamond.relabel(
            add_task="start", times_task="left", times_task_2="right"
        )
        assert renamed.fingerprint == diamond.fingerprint
        assert renamed == diamond

    def test_relabel_survives_a_join(self, chain):
        """A name given to a node outlives being joined to something else."""
        pipe = chain.relabel(add_task="offset") | AddTask(value=1)
        assert "offset" in pipe.tasks

    def test_join_keeps_a_pipes_own_names(self):
        """A collision does not push a later node off the name it was given."""
        left = (AddTask(value=1) | TimesTask(value=1)).relabel(add_task="shared")
        right = (AddTask(value=2) | AddTask(value=3)).relabel(
            add_task="shared", add_task_2="shared_2"
        )
        joined = left | right
        # `shared` is taken, so right's own `shared` is numbered past the
        # name its second node holds rather than onto it.
        assert "shared_2" in joined.tasks
        assert joined.get("shared_2") == AddTask(value=3)

    def test_relabel_unknown_node(self, chain):
        """Renaming a node which is not there says so."""
        with pytest.raises(ParameterError, match="no node called"):
            chain.relabel(nowhere="somewhere")

    def test_relabel_onto_a_taken_name(self, chain):
        """A name another node holds is refused."""
        with pytest.raises(ParameterError, match="already has a node"):
            chain.relabel(add_task="times_task")

    def test_relabel_two_nodes_alike(self, chain):
        """Two nodes cannot be given the same name."""
        with pytest.raises(ParameterError, match="cannot both be called"):
            chain.relabel(add_task="same", times_task="same")

    def test_relabel_nodes_holding_one_task(self):
        """
        Two nodes holding the same task, fed alike, still relabel freely.

        This is the shape an order-based fingerprint has to break a tie in,
        and where it would reach for the node's name to do it. Marking a
        node by what feeds it has no tie to break, so renaming these two --
        in an order which sorts the other way -- leaves the pipe alone.
        """
        base = Pipe(
            tasks={
                "src": AddTask(value=0),
                "p": AddTask(value=1),
                "q": AddTask(value=1),
                "left": TimesTask(value=2),
                "right": TimesTask(value=3),
            },
            dependencies={
                "p": ("src",),
                "q": ("src",),
                "left": ("p",),
                "right": ("q",),
            },
            inputs=("src",),
            outputs=("left", "right"),
        )
        renamed = base.relabel(p="zz", q="aa")
        assert renamed.fingerprint == base.fingerprint
        assert renamed.run(1) == base.run(1)

    def test_relabel_swap(self, chain):
        """Two nodes can trade names, which nothing else claims."""
        swapped = chain.relabel(add_task="times_task", times_task="add_task")
        assert swapped.fingerprint == chain.fingerprint
        assert swapped.get("times_task") == AddTask(value=2)


class TestGetAndUpdate:
    """Tests for reading and changing one node of a pipe."""

    def test_get(self, chain):
        """A node hands back the task it holds."""
        assert chain.get("add_task") == AddTask(value=2)

    def test_get_unknown(self, chain):
        """Asking for a node which is not there names the ones which are."""
        with pytest.raises(ParameterError, match="no node called"):
            chain.get("nowhere")

    def test_update(self, chain):
        """Changing a parameter gives back another pipe."""
        updated = chain.update("add_task", value=10)
        assert updated.get("add_task") == AddTask(value=10)
        assert updated.run(1) == (1 + 10) * 3
        # The pipe it came from is untouched.
        assert chain.get("add_task") == AddTask(value=2)

    def test_update_changes_the_fingerprint(self, chain):
        """A pipe holding another task is another pipe."""
        assert chain.update("add_task", value=10).fingerprint != chain.fingerprint

    def test_update_keeps_the_wiring(self, diamond):
        """Only the task changes; the graph around it stays."""
        updated = diamond.update("times_task", value=7)
        assert updated.dependencies == diamond.dependencies
        assert updated.inputs == diamond.inputs
        assert updated.outputs == diamond.outputs

    def test_new(self, chain, fanned):
        """A field can be replaced without going through a dump."""
        swapped = fanned.new(outputs=fanned.outputs[::-1])
        assert swapped.outputs == fanned.outputs[::-1]
        assert swapped.run(1) == fanned.run(1)[::-1]

    def test_update_unknown(self, chain):
        """Changing a node which is not there says so."""
        with pytest.raises(ParameterError, match="no node called"):
            chain.update("nowhere", value=1)


class TestRunning:
    """Tests for running a pipe."""

    def test_order(self, chain):
        """The tasks run in the order they were joined."""
        assert chain.run(1) == (1 + 2) * 3

    def test_calling_runs(self, chain):
        """Calling a pipe runs it, so a pipe can stand for a function."""
        assert chain(1) == chain.run(1)

    def test_branches_share_one_input(self, branched):
        """A pipe which branches gives one input to every branch."""
        assert branched.run(10) == (10 + 1) + (10 * 5)

    def test_branches_take_one_input_each(self, branched):
        """A pipe which branches takes an input per branch."""
        assert branched.run(10, 20) == (10 + 1) + (20 * 5)

    def test_fan_in_order(self, crossed):
        """A task is given its inputs the way they were wired."""
        # Wired (times, add): 10 * 5 - (20 + 1). The other way round it is
        # 21 - 50, so a pipe which loses the order cannot answer this.
        assert crossed.run(10, 20) == 29

    def test_wrong_number_of_inputs(self, branched):
        """A pipe says when it was given inputs it cannot place."""
        with pytest.raises(ParameterError, match="one for each"):
            branched.run(1, 2, 3)

    def test_runs_after_its_inputs(self, branched):
        """A task runs after whatever feeds it, however the pipe is held."""
        order = branched.sorted_nodes()
        for node, given in branched.dependencies.items():
            for upstream in given:
                assert order.index(upstream) < order.index(node)


class TestSources:
    """Tests for a pipe whose sources make their own values."""

    def test_a_pipe_of_sources(self):
        """Several tasks which take nothing can still feed one which does."""
        pipe = (ConstantTask(value=1), ConstantTask(value=2)) | JoinTask()
        assert pipe.run() == 3

    def test_one_source_which_takes_nothing(self):
        """A single source is handed nothing when the pipe is given nothing."""
        assert (ConstantTask(value=4) | TimesTask(value=2)).run() == 8


class TestOutputs:
    """Tests for a pipe which returns more than one thing."""

    def test_fan_out_returns_a_tuple(self, fanned):
        """A fan out with nothing merging it returns one result per output."""
        assert fanned.outputs == ("times_task", "times_task_2")
        assert fanned.run(1) == ((1 + 1) * 2, (1 + 1) * 3)

    def test_diamond_merges(self, diamond):
        """A fan out joined again is one result."""
        assert diamond.outputs == ("join_task",)
        assert diamond.run(1) == (1 + 1) * 2 + (1 + 1) * 3

    def test_diamond_matches_a_hand_built_pipe(self, diamond):
        """A diamond built with `|` is the pipe it would be built by hand."""
        add, left, right = AddTask(value=1), TimesTask(value=2), TimesTask(value=3)
        by_hand = Pipe(
            tasks={"a": add, "l": left, "r": right, "j": JoinTask()},
            dependencies={"l": ("a",), "r": ("a",), "j": ("l", "r")},
            inputs=("a",),
            outputs=("j",),
        )
        assert by_hand.fingerprint == diamond.fingerprint
        assert by_hand.run(1) == diamond.run(1)

    def test_output_order_matters(self, fanned):
        """Which result comes first is part of what a pipe is."""
        swapped = Pipe(
            tasks=fanned.tasks,
            dependencies=fanned.dependencies,
            inputs=fanned.inputs,
            outputs=fanned.outputs[::-1],
        )
        assert swapped.fingerprint != fanned.fingerprint
        assert swapped.run(1) == fanned.run(1)[::-1]

    def test_fan_out_of_pipes(self):
        """The branches of a fan out can be pipes of their own."""
        pipe = AddTask(value=1) | (
            TimesTask(value=2) | AddTask(value=1),
            TimesTask(value=3),
        )
        assert pipe.run(1) == ((1 + 1) * 2 + 1, (1 + 1) * 3)

    def test_fan_in_of_a_fan_out(self, fanned):
        """A pipe of two outputs feeds them, in order, into what follows."""
        assert (fanned | JoinTask()).run(1) == (1 + 1) * 2 + (1 + 1) * 3


class TestFingerprint:
    """Tests for what identifies a pipe."""

    def test_same_pipe(self, chain):
        """The same pipe built twice fingerprints alike."""
        assert chain.fingerprint == (AddTask(value=2) | TimesTask(value=3)).fingerprint

    def test_parameters_matter(self, chain):
        """A pipe holding another task is another pipe."""
        assert chain.fingerprint != (AddTask(value=3) | TimesTask(value=3)).fingerprint

    def test_order_matters(self):
        """The same tasks the other way round are another pipe."""
        first = AddTask(value=2) | TimesTask(value=3)
        second = TimesTask(value=3) | AddTask(value=2)
        assert first.fingerprint != second.fingerprint

    def test_wiring_matters(self):
        """The same tasks wired differently are different pipes."""
        chained = AddTask(value=1) | TimesTask(value=5) | JoinTask()
        branched = (AddTask(value=1), TimesTask(value=5)) | JoinTask()
        assert set(chained.tasks.values()) == set(branched.tasks.values())
        assert set(chained.tasks) == set(branched.tasks)
        assert chained.fingerprint != branched.fingerprint

    def test_input_order_matters(self, branched):
        """Which branch is fed first is part of what a pipe is."""
        swapped = Pipe(
            tasks=branched.tasks,
            dependencies=branched.dependencies,
            inputs=branched.inputs[::-1],
            outputs=branched.outputs,
        )
        assert swapped.fingerprint != branched.fingerprint
        assert swapped.run(10, 20) != branched.run(10, 20)

    def test_a_task_run_twice(self, chain):
        """
        Running a task twice is not the same shape as running it once.

        Both hand what follows the same pair of results, so only how many
        nodes there are tells them apart -- and a task run twice is two
        steps whatever its results look like.
        """
        add, times, join = AddTask(value=1), TimesTask(value=2), JoinTask()
        twice = Pipe(
            tasks={"a": add, "t": times, "t2": times, "j": join},
            dependencies={"t": ("a",), "t2": ("a",), "j": ("t", "t2")},
            inputs=("a",),
            outputs=("j",),
        )
        once = Pipe(
            tasks={"a": add, "t": times, "j": join},
            dependencies={"t": ("a",), "j": ("t", "t")},
            inputs=("a",),
            outputs=("j",),
        )
        assert twice.run(1) == once.run(1)
        assert twice.fingerprint != once.fingerprint

    def test_node_names_do_not_matter(self, branched):
        """A pipe whose nodes are named differently is the same pipe."""
        names = {node: f"node_{index}" for index, node in enumerate(branched.tasks)}
        assert branched.relabel(**names).fingerprint == branched.fingerprint

    def test_task_order_does_not_matter(self, branched):
        """A document is free to list the tasks in any order it likes."""
        shuffled = Pipe(
            tasks=dict(reversed(list(branched.tasks.items()))),
            dependencies=branched.dependencies,
            inputs=branched.inputs,
            outputs=branched.outputs,
        )
        assert shuffled.fingerprint == branched.fingerprint

    def test_equality(self, chain):
        """Two pipes are equal when they hold the same tasks, wired alike."""
        assert chain == AddTask(value=2) | TimesTask(value=3)
        assert chain != AddTask(value=2) | TimesTask(value=4)

    def test_not_a_pipe(self, chain):
        """Comparison with something else is left to the something else."""
        assert chain.__eq__("a pipe") is NotImplemented

    def test_hashes_with_equality(self, chain):
        """Equal pipes land in the same place in a set."""
        assert len({chain, AddTask(value=2) | TimesTask(value=3)}) == 1


class TestValidation:
    """Tests for the checks a pipe makes on itself."""

    def test_missing_task(self):
        """A pipe cannot wire a node it does not hold."""
        with pytest.raises(ValidationError, match="not one of its tasks"):
            Pipe(
                tasks={"add": AddTask()},
                dependencies={"add": ("nowhere",)},
                outputs=("add",),
            )

    def test_missing_output(self):
        """A pipe's output has to be one of its tasks."""
        with pytest.raises(ValidationError, match="output"):
            Pipe(tasks={"add": AddTask()}, inputs=("add",), outputs=("nowhere",))

    def test_no_outputs(self):
        """A pipe which returns nothing is a pipe built wrong."""
        with pytest.raises(ValidationError, match="has to return something"):
            Pipe(tasks={"add": AddTask()}, inputs=("add",), outputs=())

    def test_cycle(self):
        """Tasks which feed each other in a circle are refused."""
        with pytest.raises(ValidationError, match="cycle"):
            Pipe(
                tasks={"one": AddTask(value=1), "two": AddTask(value=2)},
                dependencies={"one": ("two",), "two": ("one",)},
                outputs=("one",),
            )

    def test_stranded_task(self, chain):
        """A task whose output reaches nothing is a pipe built wrong."""
        with pytest.raises(ValidationError, match="every task has to feed"):
            Pipe(
                tasks=dict(chain.tasks) | {"stranded": AddTask(value=99)},
                dependencies=chain.dependencies,
                inputs=(*chain.inputs, "stranded"),
                outputs=chain.outputs,
            )

    def test_a_task_reaching_a_later_output(self):
        """A task is fed when it reaches any output, not only the first."""
        # `times_task_2` reaches only the second result, so a check which
        # looked at one output would call it stranded and refuse to build.
        pipe = AddTask(value=1) | (
            TimesTask(value=2),
            TimesTask(value=3) | AddTask(value=5),
        )
        assert "times_task_2" in pipe.tasks
        assert pipe.run(1) == ((1 + 1) * 2, (1 + 1) * 3 + 5)

    def test_repeated_input(self, chain):
        """A node cannot be fed twice; it would hide one of the sources."""
        with pytest.raises(ValidationError, match="nothing wired into them"):
            Pipe(
                tasks=chain.tasks,
                dependencies=chain.dependencies,
                inputs=(*chain.inputs, *chain.inputs),
                outputs=chain.outputs,
            )

    def test_inputs_must_be_the_unfed_nodes(self, chain):
        """A pipe's inputs are exactly the nodes nothing is wired into."""
        with pytest.raises(ValidationError, match="nothing wired into them"):
            Pipe(
                tasks=chain.tasks,
                dependencies=chain.dependencies,
                inputs=chain.outputs,
                outputs=chain.outputs,
            )


class TestDocuments:
    """Tests for writing a pipe down and reading it back."""

    def test_round_trip(self, chain):
        """A pipe rebuilt from its document is the same pipe."""
        assert Pipe.from_dict(chain.to_dict()) == chain

    def test_branched_round_trip(self, branched):
        """A pipe which branches keeps its shape."""
        rebuilt = Pipe.from_dict(branched.to_dict())
        assert rebuilt == branched
        assert rebuilt.run(10) == branched.run(10)

    @pytest.mark.parametrize("name", ["pipe.json", "pipe.yaml", "pipe.yml", "pipe"])
    def test_save_and_load(self, branched, tmp_path, name):
        """A pipe read back from a file is the same pipe, and runs alike."""
        path = branched.save(tmp_path / name)
        loaded = Pipe.load(path)
        assert loaded == branched
        # Run as well as compared: a document is free to write the tasks in
        # another order, and the answer must not depend on which order.
        assert loaded.run(10, 20) == branched.run(10, 20)

    @pytest.mark.parametrize("name", ["pipe.json", "pipe.yaml"])
    def test_source_order_survives_a_round_trip(self, crossed, tmp_path, name):
        """The branch fed first is still the one fed first after loading."""
        loaded = Pipe.load(crossed.save(tmp_path / name))
        assert loaded.sources() == crossed.sources()
        assert loaded.run(10, 20) == 29

    def test_source_order_survives_a_reordered_document(self, crossed):
        """
        A document is free to list the tasks in any order it likes.

        Which is why the sources are a field of their own: read off the
        tasks instead, this document would hand each branch the other
        branch's input and answer 89.
        """
        document = crossed.to_dict()
        document["tasks"] = dict(reversed(list(document["tasks"].items())))
        loaded = Pipe.from_dict(document)
        assert loaded.sources() == crossed.sources()
        assert loaded.run(10, 20) == 29

    @pytest.mark.parametrize("name", ["pipe.json", "pipe.yaml"])
    def test_output_order_survives_a_round_trip(self, fanned, tmp_path, name):
        """A pipe of two results returns them in the order it was written."""
        loaded = Pipe.load(fanned.save(tmp_path / name))
        assert loaded.outputs == fanned.outputs
        assert loaded.run(1) == fanned.run(1)

    def test_nested_pipe_field(self, chain):
        """A pipe held as a task's parameter is written as a pipe."""
        task = NestedPipeTask(inner=chain)
        document = task.to_dict()
        assert set(document["params"]["inner"]) == {"$pipe"}
        rebuilt = Task.from_dict(document)
        assert rebuilt == task
        assert rebuilt.run(1) == chain.run(1)

    def test_nested_pipe_in_a_pipe(self, chain):
        """A task holding a pipe round trips inside a pipe of its own."""
        pipe = NestedPipeTask(inner=chain) | AddTask(value=1)
        assert Pipe.from_dict(pipe.to_dict()) == pipe

    def test_nested_pipe_fingerprint_cannot_decode(self, chain):
        """A pipe hashed for a fingerprint cannot be read back."""
        with pytest.raises(ParameterError, match="cannot be read back"):
            decode(encode(chain))

    def test_nested_pipe_fingerprint(self, chain):
        """A task holding a pipe is identified by the pipe it holds."""
        first = NestedPipeTask(inner=chain)
        second = NestedPipeTask(inner=chain.update("add_task", value=99))
        assert first.fingerprint != second.fingerprint

    def test_task_from_another_package(self):
        """A task declared elsewhere round trips under its own namespace."""
        task = PortedTask(value=3)
        assert task.tag == "tests:PortedTask"
        assert Task.from_dict(task.to_dict()) == task

    def test_pipe_of_tasks_from_another_package(self):
        """A pipe of such tasks is written and read back the same way."""
        pipe = PortedTask(value=1) | PortedTask(value=2)
        rebuilt = Pipe.from_dict(pipe.to_dict())
        assert rebuilt == pipe
        assert rebuilt.get("ported_task_2").value == 2
        assert rebuilt.run(1) == 4

    def test_yaml_is_yaml(self, chain, tmp_path):
        """A pipe saved as YAML is written as YAML, not as JSON."""
        text = chain.save(tmp_path / "pipe.yaml").read_text()
        assert not text.lstrip().startswith("{")
        assert "tasks:" in text

    def test_unknown_suffix_refused(self, chain, tmp_path):
        """A suffix which names no format is refused, not guessed at."""
        with pytest.raises(ParameterError, match="names no format"):
            chain.save(tmp_path / "pipe.txt")

    def test_unparsable_file(self, chain, tmp_path):
        """A file which does not parse names itself in the error."""
        path = tmp_path / "pipe.json"
        path.write_text("{not json at all")
        with pytest.raises(ParameterError, match="Could not parse JSON"):
            Pipe.load(path)

    def test_unparsable_yaml(self, chain, tmp_path):
        """A YAML file which does not parse says so as YAML."""
        path = tmp_path / "pipe.yaml"
        path.write_text("tasks: [unclosed")
        with pytest.raises(ParameterError, match="Could not parse YAML"):
            Pipe.load(path)

    def test_file_holding_no_document(self, chain, tmp_path):
        """A file holding something other than a document is refused."""
        path = tmp_path / "pipe.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ParameterError, match="describes no workflow"):
            Pipe.load(path)

    def test_document_of_something_else(self):
        """A document which is not a pipe says so rather than raising a KeyError."""
        with pytest.raises(ParameterError, match="describes something else"):
            Pipe.from_dict({"object_type": "Inventory"})

    def test_document_holding_a_graph_which_cannot_run(self, chain):
        """A wire a document lost is an unreadable document, not a bad model."""
        document = chain.to_dict()
        document["dependencies"] = {"times_task": ["nowhere"]}
        with pytest.raises(ParameterError, match="could not be built"):
            Pipe.from_dict(document)

    def test_a_parameter_cannot_fake_a_version(self):
        """
        A parameter which spells a tag and a version is still a parameter.

        Read as a version bump it would switch off the check that the
        document has not been edited, for every task in the pipe.
        """
        faked = {"object_type": VersionedTask(value=1).tag, "version": "0.5"}
        pipe = CarryingTask(meta=faked) | AddTask(value=1)
        document = pipe.to_dict()
        document["tasks"]["add_task"]["params"]["value"] = 99
        with pytest.raises(ParameterError, match="edited it"):
            Pipe.from_dict(document)

    def test_edited_document_refused(self, chain, tmp_path):
        """A document whose fingerprint disagrees with it is refused."""
        document = chain.to_dict()
        document["fingerprint"] = "0" * 16
        with pytest.raises(ParameterError, match="edited it"):
            Pipe.from_dict(document)

    def test_document_written_by_another_version(self, monkeypatch):
        """
        A pipe written before a task changed version still loads.

        The version is what makes a fingerprint differ on purpose, so a
        stored pipe must not be read as an edited one for having one.
        """
        pipe = VersionedTask(value=2) | AddTask(value=3)
        document = pipe.to_dict()
        monkeypatch.setattr(VersionedTask, "__version__", "2.0")
        # Which really is a document whose stored fingerprint no longer
        # matches the pipe it describes.
        moved = VersionedTask(value=2) | AddTask(value=3)
        assert document["fingerprint"] != moved.fingerprint
        with suppress_warnings(DASCoreWarning, message="The document holds"):
            assert Pipe.from_dict(document).run(1) == 6

    def test_an_edit_is_caught_after_a_version_moved_on(self, monkeypatch):
        """
        A version bump does not switch the check on edits off.

        The fingerprint is recomputed at the version the document names,
        which says what it was; a task whose parameters were changed since
        then still does not add up.
        """
        pipe = VersionedTask(value=2) | AddTask(value=3)
        document = pipe.to_dict()
        monkeypatch.setattr(VersionedTask, "__version__", "2.0")
        document["tasks"]["add_task"]["params"]["value"] = 99
        with suppress_warnings(DASCoreWarning, message="The document holds"):
            with pytest.raises(ParameterError, match="edited it"):
                Pipe.from_dict(document)

    def test_nested_version_moved_on(self, monkeypatch):
        """A version bump inside a nested pipe is a version bump."""
        inner = VersionedTask(value=2) | AddTask(value=3)
        pipe = NestedPipeTask(inner=inner) | AddTask(value=1)
        document = pipe.to_dict()
        monkeypatch.setattr(VersionedTask, "__version__", "2.0")
        with suppress_warnings(DASCoreWarning, message="The document holds"):
            assert Pipe.from_dict(document).run(1) == 7

    def test_save_makes_the_directory(self, chain, tmp_path):
        """Saving into a directory which is not there makes it."""
        path = chain.save(tmp_path / "nested" / "deeper" / "pipe.json")
        assert path.exists()

    def test_pickle(self, chain):
        """A pipe survives a pickle, tasks and all."""
        assert pickle.loads(pickle.dumps(chain)) == chain

    def test_pickle_a_fan_out(self, fanned):
        """A pipe of several results keeps them through a pickle."""
        assert pickle.loads(pickle.dumps(fanned)).outputs == fanned.outputs


class TestMermaid:
    """Tests for drawing a pipe."""

    def test_lists_every_task(self, branched):
        """Every node in the pipe is drawn, under the name it holds."""
        text = branched.to_mermaid()
        for name in branched.tasks:
            assert f'"{name}"' in text

    def test_draws_every_edge(self, branched):
        """Every wire in the pipe is drawn."""
        assert text_edges(branched.to_mermaid()) == 2

    def test_chain_edges(self, chain):
        """A chain of two tasks is one edge."""
        assert text_edges(chain.to_mermaid()) == 1

    def test_escapes_a_quoted_name(self, chain):
        """A name holding a quote does not break the label around it."""
        text = chain.relabel(add_task='a "name"').to_mermaid()
        assert '["a #quot;name#quot;"]' in text

    def test_shows_both_leaves(self, fanned):
        """A fan out draws the two results it ends in."""
        text = fanned.to_mermaid()
        for name in fanned.outputs:
            assert f'"{name}"' in text
        assert text_edges(text) == 2


def text_edges(text: str) -> int:
    """Return how many edges a mermaid flowchart holds."""
    return sum("-->" in line for line in text.splitlines())
