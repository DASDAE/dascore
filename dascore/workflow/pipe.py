"""
A pipe: several tasks arranged into the shape of one operation.

A [`Pipe`](`dascore.workflow.pipe.Pipe`) is a directed acyclic graph of
[`Task`](`dascore.workflow.task.Task`) objects. It is itself immutable,
fingerprintable and serializable, so a whole processing chain can be
compared, written to a file, handed to another process, and run later
against different data.

Examples
--------
>>> from dascore.workflow import Task
>>>
>>> class PipeAddExample(Task):
...     '''Add a number.'''
...     value: int = 1
...     def run(self, number):
...         return number + self.value
>>>
>>> class PipeTimesExample(Task):
...     '''Multiply by a number.'''
...     value: int = 2
...     def run(self, number):
...         return number * self.value
>>>
>>> pipe = PipeAddExample(value=2) | PipeTimesExample(value=3)
>>> assert pipe.run(1) == 9
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import Field

from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.types import FrozenDictType
from dascore.utils.misc import optional_import
from dascore.workflow.serialize import digest
from dascore.workflow.task import Task

# The suffixes `save` and `load` read as YAML; anything else is JSON.
YAML_SUFFIXES = frozenset({".yaml", ".yml"})

# What separates a node id from the number which makes it unique, when one
# task appears in a pipe more than once.
COPY_SEP = "#"


class Pipe(DascoreBaseModel):
    """
    A directed acyclic graph of tasks.

    Each node is a task, named by its fingerprint, and each edge says which
    node's output becomes one of another node's inputs. Inputs are given to
    a task in the order they are listed, so a task taking two of them --
    concatenating two patches, say -- gets them the way it was wired.

    Build one with ``|`` rather than by hand: ``first | second`` chains two
    tasks, ``pipe | third`` extends a chain, and ``(left, right) | merge``
    feeds two branches into one task.
    """

    tasks: FrozenDictType[str, Task]
    # Only a node with inputs appears here; a source has none.
    dependencies: FrozenDictType[str, tuple[str, ...]] = Field(default_factory=dict)
    output: str

    def __init__(self, **kwargs):
        """Build a pipe, checking it describes a graph which can be run."""
        super().__init__(**kwargs)
        self.check()

    @cached_property
    def fingerprint(self) -> str:
        """
        Return the digest which identifies this pipe.

        It names every task in the pipe and how they are wired, so two pipes
        holding the same tasks in a different order are not the same pipe.
        """
        payload = {
            "tasks": {node: task.fingerprint for node, task in self.tasks.items()},
            "dependencies": {node: list(x) for node, x in self.dependencies.items()},
            "output": self.output,
        }
        return digest(payload)

    def __hash__(self) -> int:
        """Hash a pipe the way it compares."""
        return hash(self.fingerprint)

    def __eq__(self, other) -> bool:
        """Two pipes are equal if they hold the same tasks, wired alike."""
        if not isinstance(other, Pipe):
            return NotImplemented
        return self.fingerprint == other.fingerprint

    def __len__(self) -> int:
        """Return how many tasks the pipe holds."""
        return len(self.tasks)

    def __or__(self, other: Task | Pipe) -> Pipe:
        """Extend this pipe with a task or another pipe."""
        return join(self, other)

    def __ror__(self, other: Task | Sequence[Task | Pipe]) -> Pipe:
        """Feed a task, or several, into this pipe."""
        return join(other, self)

    def run(self, *inputs) -> Any:
        """
        Run every task in order and return what the last one gave back.

        Parameters
        ----------
        *inputs
            What the pipe's sources are given. A pipe with one source hands
            it all of them. A pipe which branches takes one input per
            branch, in the order they were wired, or a single input which
            every branch is given.
        """
        sources = self.sources()
        results = {}
        for node in self.sorted_nodes():
            given = self.dependencies.get(node, ())
            if given:
                arguments = tuple(results[x] for x in given)
            else:
                arguments = _source_arguments(node, sources, inputs)
            results[node] = self.tasks[node].run(*arguments)
        return results[self.output]

    def map(self, iterable: Iterable, client: Any = None) -> list:
        """
        Run the pipe over each item of an iterable.

        Parameters
        ----------
        iterable
            The items to run, each of which is given to the pipe's source.
        client
            Something with a ``map`` -- a `ProcessPoolExecutor`, a dask
            client -- to run them with. The pipe pickles, so a worker in
            another process can rebuild it.
        """
        if client is None:
            return [self.run(x) for x in iterable]
        return list(client.map(self.run, iterable))

    def sources(self) -> tuple[str, ...]:
        """Return the nodes which take their input from the caller."""
        return tuple(x for x in self.sorted_nodes() if not self.dependencies.get(x))

    def sorted_nodes(self) -> tuple[str, ...]:
        """
        Return the nodes in an order which runs each after its inputs.

        Kahn's algorithm, taking the nodes in the order they were added so
        that a pipe always runs its tasks the same way round.
        """
        remaining = {node: len(self.dependencies.get(node, ())) for node in self.tasks}
        downstream: dict[str, list[str]] = {node: [] for node in self.tasks}
        for node, given in self.dependencies.items():
            for upstream in given:
                downstream[upstream].append(node)
        ready = [node for node, count in remaining.items() if not count]
        out = []
        while ready:
            node = ready.pop(0)
            out.append(node)
            for other in downstream[node]:
                remaining[other] -= 1
                if not remaining[other]:
                    ready.append(other)
        if len(out) != len(self.tasks):
            msg = "The pipe's tasks feed into each other in a cycle."
            raise ParameterError(msg)
        return tuple(out)

    def check(self) -> Pipe:
        """
        Check that the pipe describes a graph which can be run.

        Raises `ParameterError` naming what is wrong: an edge from a node
        which is not there, an output which is not there, or a cycle.
        """
        for node, given in self.dependencies.items():
            for name in (node, *given):
                if name not in self.tasks:
                    msg = f"The pipe wires {name!r}, which is not one of its tasks."
                    raise ParameterError(msg)
        if self.output not in self.tasks:
            msg = f"The pipe's output {self.output!r} is not one of its tasks."
            raise ParameterError(msg)
        self.sorted_nodes()
        return self

    def get_provenance(self, **metadata) -> Any:
        """
        Return a record of this pipe and the machine it ran on.

        Parameters
        ----------
        **metadata
            Anything else worth recording alongside it.
        """
        # provenance.py imports this module, so it is named where it is used.
        from dascore.workflow.provenance import Provenance  # noqa: PLC0415

        return Provenance.from_pipe(self, **metadata)

    def to_dict(self) -> dict[str, Any]:
        """Return a document which describes this pipe."""
        return {
            "tasks": {node: task.to_dict() for node, task in self.tasks.items()},
            "dependencies": {node: list(x) for node, x in self.dependencies.items()},
            "output": self.output,
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, document: Mapping) -> Pipe:
        """Return the pipe a document describes."""
        tasks = {
            node: Task.from_dict(value) for node, value in document["tasks"].items()
        }
        dependencies = {
            node: tuple(value)
            for node, value in document.get("dependencies", {}).items()
        }
        return cls(tasks=tasks, dependencies=dependencies, output=document["output"])

    def save(self, path: str | Path) -> Path:
        """
        Write this pipe to a file.

        The suffix picks the format: ``.yaml`` or ``.yml`` write YAML, and
        anything else writes JSON.
        """
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(_dump(self.to_dict(), path), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Pipe:
        """Return the pipe a file holds; see `save`."""
        path = Path(path)
        return cls.from_dict(_parse(path.read_text(encoding="utf-8"), path))

    def to_mermaid(self) -> str:
        """
        Return the pipe as a mermaid flowchart.

        Markdown viewers, this project's docs among them, draw one from the
        text; it is a readable listing on its own besides.
        """
        lines = ["flowchart TD"]
        names = {node: f"n{index}" for index, node in enumerate(self.sorted_nodes())}
        for node, name in names.items():
            label = type(self.tasks[node]).__name__
            lines.append(f"    {name}[{label}]")
        for node, given in self.dependencies.items():
            for upstream in given:
                lines.append(f"    {names[upstream]} --> {names[node]}")
        return "\n".join(lines)


def join(left: Any, right: Any) -> Pipe:
    """Return the pipe which runs one side and then the other."""
    inputs = [_as_pipe(x) for x in (left if isinstance(left, list | tuple) else [left])]
    right = _as_pipe(right)
    tasks: dict[str, Task] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    outputs = []
    for pipe in inputs:
        renamed = _merge(pipe, tasks, dependencies)
        outputs.append(renamed[pipe.output])
    renamed = _merge(right, tasks, dependencies)
    # The first task of the right hand side takes the left hand sides'
    # outputs; anything already wired inside it keeps what it had.
    for node in right.sources():
        dependencies[renamed[node]] = tuple(outputs)
    return Pipe(tasks=tasks, dependencies=dependencies, output=renamed[right.output])


def _as_pipe(value: Any) -> Pipe:
    """Return a task or pipe as a pipe."""
    if isinstance(value, Pipe):
        return value
    if isinstance(value, Task):
        node = value.fingerprint
        return Pipe(tasks={node: value}, output=node)
    msg = f"A pipe joins tasks and pipes, not {type(value).__name__}."
    raise ParameterError(msg)


def _merge(
    pipe: Pipe,
    tasks: dict[str, Task],
    dependencies: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """
    Copy a pipe's nodes into a graph being built, and say what they became.

    A node is named by its task's fingerprint, so the same task twice in one
    chain -- ``detrend | detrend`` -- would be one node and would lose a
    step. The second copy takes a ``#n`` suffix instead.
    """
    renamed = {}
    for node in pipe.sorted_nodes():
        name = node
        copy = 0
        while name in tasks:
            copy += 1
            name = f"{node}{COPY_SEP}{copy}"
        renamed[node] = name
        tasks[name] = pipe.tasks[node]
    for node, given in pipe.dependencies.items():
        dependencies[renamed[node]] = tuple(renamed[x] for x in given)
    return renamed


def _source_arguments(node: str, sources: tuple[str, ...], inputs: tuple) -> tuple:
    """Return what one source node is given to run."""
    if len(sources) == 1:
        return inputs
    # One input for a pipe which branches: every branch gets it, which is
    # what a pipe shaped like a diamond means.
    if len(inputs) == 1:
        return inputs
    if len(inputs) != len(sources):
        msg = (
            f"The pipe has {len(sources)} sources and was given "
            f"{len(inputs)} inputs; give it one for each, or one for all."
        )
        raise ParameterError(msg)
    return (inputs[sources.index(node)],)


def _dump(document: Mapping, path: Path) -> str:
    """Return the text a document is written as, per the path's suffix."""
    if path.suffix.lower() in YAML_SUFFIXES:
        yaml = optional_import("yaml", required_for="writing a pipe as YAML")
        return yaml.safe_dump(document, sort_keys=False)
    return json.dumps(document, indent=2, sort_keys=True)


def _parse(text: str, path: Path) -> Any:
    """Return the document some text holds, per the path's suffix."""
    if path.suffix.lower() in YAML_SUFFIXES:
        yaml = optional_import("yaml", required_for="reading a pipe from YAML")
        return yaml.safe_load(text)
    return json.loads(text)
