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

from collections.abc import Iterable, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.types import FrozenDictType
from dascore.workflow.serialize import digest, read_document, write_document
from dascore.workflow.task import Task

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
    # The nodes the caller feeds, in the order they are fed. Written down
    # rather than read off `tasks`, whose order a document is free to
    # change: sorting the keys of a saved pipe would otherwise send the
    # inputs to the wrong branches.
    inputs: tuple[str, ...] = ()
    output: str

    @model_validator(mode="after")
    def _check_after_building(self) -> Pipe:
        """Check every pipe, however it was built."""
        return self.check()

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
            "inputs": list(self.inputs),
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
        """Return the nodes which take their input from the caller, in order."""
        return self.inputs

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
        # Seeded with the inputs, in their order, so the run order depends
        # on the pipe rather than on how a document happened to be written.
        ready = [x for x in self.inputs if not remaining[x]]
        ready += [x for x, count in remaining.items() if not count and x not in ready]
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
        fed = {x for x in self.tasks if not self.dependencies.get(x)}
        if set(self.inputs) != fed:
            msg = (
                f"The pipe takes its inputs at {sorted(self.inputs)}, but the "
                f"nodes with nothing wired into them are {sorted(fed)}."
            )
            raise ParameterError(msg)
        order = self.sorted_nodes()
        # A task whose output reaches nothing is work the pipe would do and
        # throw away, which is a pipe built wrong rather than a slow one.
        reaching = {self.output}
        for node in reversed(order):
            if node in reaching:
                reaching.update(self.dependencies.get(node, ()))
        if stranded := set(self.tasks) - reaching:
            msg = (
                f"Nothing the pipe returns is built from {sorted(stranded)}; "
                "every task has to feed its output."
            )
            raise ParameterError(msg)
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
            "inputs": list(self.inputs),
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
        out = cls(
            tasks=tasks,
            dependencies=dependencies,
            inputs=tuple(document["inputs"]),
            output=document["output"],
        )
        # A document says what it holds and what it hashed to; a mismatch
        # means it was edited, and hiding that would defeat the fingerprint.
        if (written := document.get("fingerprint")) and written != out.fingerprint:
            msg = (
                f"The document says its fingerprint is {written}, and the "
                f"pipe it describes has {out.fingerprint}."
            )
            raise ParameterError(msg)
        return out

    def save(self, path: str | Path) -> Path:
        """
        Write this pipe to a file.

        The suffix picks the format: ``.yaml`` or ``.yml`` write YAML, and
        anything else writes JSON.
        """
        return write_document(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Pipe:
        """Return the pipe a file holds; see `save`."""
        return cls.from_dict(read_document(Path(path)))

    def to_mermaid(self) -> str:
        """
        Return the pipe as the text of a mermaid flowchart.

        Somewhere which draws mermaid -- a markdown viewer, a notebook, a
        quarto ``{mermaid}`` cell -- turns it into a diagram; on its own it
        is a readable listing of the tasks and the wires between them.
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
    sides = list(left) if isinstance(left, list | tuple) else [left]
    if not sides:
        msg = "A pipe cannot be joined to nothing."
        raise ParameterError(msg)
    upstream = [_as_pipe(x) for x in sides]
    right = _as_pipe(right)
    tasks: dict[str, Task] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    inputs: list[str] = []
    outputs = []
    for pipe in upstream:
        renamed = _merge(pipe, tasks, dependencies)
        inputs.extend(renamed[x] for x in pipe.inputs)
        outputs.append(renamed[pipe.output])
    renamed = _merge(right, tasks, dependencies)
    # Every task the right hand side took its input from now takes the left
    # hand sides' outputs instead; anything wired inside it keeps what it had.
    for node in right.sources():
        dependencies[renamed[node]] = tuple(outputs)
    return Pipe(
        tasks=tasks,
        dependencies=dependencies,
        inputs=tuple(inputs),
        output=renamed[right.output],
    )


def _as_pipe(value: Any) -> Pipe:
    """Return a task or pipe as a pipe."""
    if isinstance(value, Pipe):
        return value
    if isinstance(value, Task):
        node = value.fingerprint
        return Pipe(tasks={node: value}, inputs=(node,), output=node)
    msg = f"A pipe joins tasks and pipes, not {type(value).__name__}."
    raise ParameterError(msg)


def _merge(
    pipe: Pipe,
    tasks: dict[str, Task],
    dependencies: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """
    Copy a pipe's nodes into a graph being built, and say what they became.

    A node which would collide with one already there is renamed; see
    `unique_name`.
    """
    renamed = {}
    for node in pipe.sorted_nodes():
        name = unique_name(node, tasks)
        renamed[node] = name
        tasks[name] = pipe.tasks[node]
    for node, given in pipe.dependencies.items():
        dependencies[renamed[node]] = tuple(renamed[x] for x in given)
    return renamed


def unique_name(fingerprint: str, taken: Mapping[str, Any]) -> str:
    """
    Return a node name nothing in a pipe has claimed yet.

    A node is named by its task's fingerprint, so the same task twice in one
    chain -- ``detrend | detrend`` -- would be one node and would lose a
    step. The second copy takes a ``#n`` suffix instead.
    """
    name = fingerprint
    copy = 0
    while name in taken:
        copy += 1
        name = f"{fingerprint}{COPY_SEP}{copy}"
    return name


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
