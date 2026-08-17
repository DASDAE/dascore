"""Tests for pipes: tasks arranged into the shape of one operation."""

from __future__ import annotations

import pickle
from concurrent.futures import ThreadPoolExecutor

import pytest

from dascore.exceptions import ParameterError
from dascore.workflow import Pipe, Task


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


@pytest.fixture
def chain():
    """A pipe of two tasks, one after the other."""
    return AddTask(value=2) | TimesTask(value=3)


@pytest.fixture
def branched():
    """A pipe whose two branches feed one task."""
    return (AddTask(value=1), TimesTask(value=5)) | JoinTask()


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

    def test_repeated_task_names(self):
        """The second copy of a task is named apart from the first."""
        pipe = AddTask(value=1) | AddTask(value=1)
        first, second = pipe.sorted_nodes()
        assert second.startswith(first) and second != first

    def test_branches_into_a_pipe(self, chain):
        """Two tasks can feed a pipe rather than a single task."""
        pipe = (AddTask(value=1), AddTask(value=2)) | (JoinTask() | TimesTask(value=2))
        assert pipe.run(1) == ((1 + 1) + (1 + 2)) * 2

    def test_joining_something_else(self):
        """A pipe joins tasks and pipes, and says so otherwise."""
        with pytest.raises(ParameterError, match="not str"):
            AddTask() | "not a task"


class TestRunning:
    """Tests for running a pipe."""

    def test_order(self, chain):
        """The tasks run in the order they were joined."""
        assert chain.run(1) == (1 + 2) * 3

    def test_branches_share_one_input(self, branched):
        """A pipe which branches gives one input to every branch."""
        assert branched.run(10) == (10 + 1) + (10 * 5)

    def test_branches_take_one_input_each(self, branched):
        """A pipe which branches takes an input per branch."""
        assert branched.run(10, 20) == (10 + 1) + (20 * 5)

    def test_wrong_number_of_inputs(self, branched):
        """A pipe says when it was given inputs it cannot place."""
        with pytest.raises(ParameterError, match="one for each"):
            branched.run(1, 2, 3)

    def test_map(self, chain):
        """A pipe runs over an iterable."""
        assert chain.map([1, 2]) == [chain.run(1), chain.run(2)]

    def test_map_with_a_client(self, chain):
        """A pipe runs over an iterable with something else's workers."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            assert chain.map([1, 2], client=pool) == [chain.run(1), chain.run(2)]


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
        node = AddTask().fingerprint
        with pytest.raises(ParameterError, match="not one of its tasks"):
            Pipe(
                tasks={node: AddTask()},
                dependencies={node: ("nowhere",)},
                output=node,
            )

    def test_missing_output(self):
        """A pipe's output has to be one of its tasks."""
        node = AddTask().fingerprint
        with pytest.raises(ParameterError, match="output"):
            Pipe(tasks={node: AddTask()}, output="nowhere")

    def test_cycle(self):
        """Tasks which feed each other in a circle are refused."""
        first, second = AddTask(value=1), AddTask(value=2)
        one, two = first.fingerprint, second.fingerprint
        with pytest.raises(ParameterError, match="cycle"):
            Pipe(
                tasks={one: first, two: second},
                dependencies={one: (two,), two: (one,)},
                output=one,
            )

    def test_check_returns_the_pipe(self, chain):
        """Checking a good pipe hands it back."""
        assert chain.check() is chain


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
    def test_save_and_load(self, chain, tmp_path, name):
        """A pipe read back from a file is the same pipe."""
        path = chain.save(tmp_path / name)
        assert Pipe.load(path) == chain

    def test_save_makes_the_directory(self, chain, tmp_path):
        """Saving into a directory which is not there makes it."""
        path = chain.save(tmp_path / "nested" / "deeper" / "pipe.json")
        assert path.exists()

    def test_pickle(self, chain):
        """A pipe survives a pickle, tasks and all."""
        assert pickle.loads(pickle.dumps(chain)) == chain


class TestMermaid:
    """Tests for drawing a pipe."""

    def test_lists_every_task(self, branched):
        """Every task in the pipe is drawn."""
        text = branched.to_mermaid()
        for name in ["AddTask", "TimesTask", "JoinTask"]:
            assert name in text

    def test_draws_every_edge(self, branched):
        """Every wire in the pipe is drawn."""
        assert text_edges(branched.to_mermaid()) == 2

    def test_chain_edges(self, chain):
        """A chain of two tasks is one edge."""
        assert text_edges(chain.to_mermaid()) == 1


def text_edges(text: str) -> int:
    """Return how many edges a mermaid flowchart holds."""
    return sum("-->" in line for line in text.splitlines())
