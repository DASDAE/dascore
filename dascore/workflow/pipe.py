"""
A pipe: several tasks arranged into the shape of one operation.

A [`Pipe`](`dascore.workflow.pipe.Pipe`) is a directed acyclic graph of
[`Task`](`dascore.workflow.task.Task`) objects. It is itself immutable,
fingerprintable and serializable, so a whole processing chain can be
compared, written to a file, handed to another process, and run later
against different data.

Running one is deliberately boring: each task is given its inputs, in
order, once everything feeding it has run. Nothing here schedules, streams
or fuses; those belong to whatever runs the pipe.

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
>>> assert pipe(1) == 9
"""

from __future__ import annotations

import re
from collections.abc import Container, Mapping, Sequence
from functools import cached_property
from pathlib import Path
from typing import Any

from pydantic import Field, model_validator

from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.types import FrozenDictType
from dascore.workflow.serialize import digest, read_workflow, write_workflow
from dascore.workflow.task import Task, holds_another_version

# Where a camel cased class name breaks into words: before a capital which
# follows a lower case letter or digit, and before the last capital of a run
# of them, so `TrimEdges` and `FFTShift` both read as they are spoken.
_WORD_BREAK = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


class Pipe(DascoreBaseModel):
    """
    A directed acyclic graph of tasks.

    Each node is a task under a name of its own, and each edge says which
    node's output becomes one of another node's inputs. Inputs are given to
    a task in the order they are listed, so a task taking two of them --
    concatenating two patches, say -- gets them the way it was wired.

    Build one with ``|`` rather than by hand: ``first | second`` chains two
    tasks, ``pipe | third`` extends a chain, ``(left, right) | merge`` feeds
    two branches into one task, and ``split | (left, right)`` fans one out
    into two results.
    """

    tasks: FrozenDictType[str, Task]
    # Only a node with inputs appears here; a source has none.
    dependencies: FrozenDictType[str, tuple[str, ...]] = Field(default_factory=dict)
    # The nodes the caller feeds, in the order they are fed. Written down
    # rather than read off `tasks`, whose order a document is free to
    # change: sorting the keys of a saved pipe would otherwise send the
    # inputs to the wrong branches.
    inputs: tuple[str, ...] = ()
    # The nodes whose results the pipe hands back, in the order it hands
    # them back. More than one is a pipe which fans out.
    outputs: tuple[str, ...]

    @model_validator(mode="after")
    def _check_after_building(self) -> Pipe:
        """Check every pipe, however it was built."""
        return self.check()

    @cached_property
    def fingerprint(self) -> str:
        """
        Return the digest which identifies this pipe.

        It is structural rather than nominal: what the pipe is fed and what
        it hands back, each named by how it is arrived at rather than by
        what it is called. Every task reaches an output, so folding the
        outputs folds the whole graph; and no node's name is anywhere in
        it, so naming a node for the reader leaves the pipe the pipe it was.
        """
        marks = self._node_marks()
        payload = {
            "inputs": [marks[x] for x in self.inputs],
            "outputs": [marks[x] for x in self.outputs],
        }
        return digest(payload)

    def _node_marks(self) -> dict[str, str]:
        """
        Return a digest for each node of how its value is arrived at.

        A node's mark folds in the task it holds, the marks of whatever
        feeds it in the order it is fed, and -- for a source -- which of
        the pipe's inputs it takes. So two nodes share a mark exactly when
        they compute the same thing from the same place, whatever either of
        them happens to be called.
        """
        seeds = {node: index for index, node in enumerate(self.inputs)}
        marks: dict[str, str] = {}
        for node in self.sorted_nodes():
            given = self.dependencies.get(node, ())
            marks[node] = digest(
                [
                    self.tasks[node].fingerprint,
                    [marks[x] for x in given],
                    seeds.get(node),
                ]
            )
        return marks

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

    def __or__(self, other: Task | Pipe | Sequence[Task | Pipe]) -> Pipe:
        """Extend this pipe with a task, another pipe, or several of them."""
        return join(self, other)

    def __ror__(self, other: Task | Pipe | Sequence[Task | Pipe]) -> Pipe:
        """Feed a task, or several, into this pipe."""
        return join(other, self)

    def __call__(self, *inputs) -> Any:
        """Run the pipe; see `run`."""
        return self.run(*inputs)

    def run(self, *inputs) -> Any:
        """
        Run every task in order and return what the pipe's outputs gave back.

        Parameters
        ----------
        *inputs
            What the pipe's sources are given. A pipe with one source hands
            it all of them. A pipe which branches takes one input per
            branch, in the order they were wired, or a single input which
            every branch is given. A pipe run with none is a pipe whose
            sources make their own values.

        Returns
        -------
        The result of the pipe's output, or a tuple of them in the order
        `outputs` lists if it has more than one.
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
        out = tuple(results[x] for x in self.outputs)
        return out[0] if len(out) == 1 else out

    def sources(self) -> tuple[str, ...]:
        """Return the nodes which take their input from the caller, in order."""
        return self.inputs

    def get(self, key: str) -> Task:
        """
        Return the task a node holds.

        Examples
        --------
        >>> from dascore.workflow import Task
        >>> class PipeGetExample(Task):
        ...     '''Add a number.'''
        ...     value: int = 1
        ...     def run(self, number):
        ...         return number + self.value
        >>> pipe = PipeGetExample(value=1) | PipeGetExample(value=2)
        >>> assert pipe.get("pipe_get_example").value == 1
        >>> assert pipe.get("pipe_get_example_2").value == 2
        """
        if key not in self.tasks:
            msg = f"The pipe has no node called {key!r}; it holds {sorted(self.tasks)}."
            raise ParameterError(msg)
        return self.tasks[key]

    def update(self, key: str, **kwargs) -> Pipe:
        """
        Return a new pipe with one node's parameters changed.

        The wiring is untouched and the node keeps its name; only the task
        it holds, and therefore the pipe's fingerprint, are new.

        Parameters
        ----------
        key
            Which node to change; see `get`.
        **kwargs
            The parameters to change, as `Task.update` takes them.
        """
        tasks = dict(self.tasks)
        tasks[key] = self.get(key).update(**kwargs)
        return Pipe(
            tasks=tasks,
            dependencies=self.dependencies,
            inputs=self.inputs,
            outputs=self.outputs,
        )

    def relabel(self, **names: str) -> Pipe:
        """
        Return a new pipe with some of its nodes named differently.

        A node's name is for whoever reads the pipe; the fingerprint is
        structural, so renaming one gives back an equal pipe.

        Parameters
        ----------
        **names
            Each node's current name against the name to give it.
        """
        for key, name in names.items():
            self.get(key)
            if name in self.tasks and name not in names:
                msg = f"The pipe already has a node called {name!r}."
                raise ParameterError(msg)
        if len(set(names.values())) != len(names):
            msg = f"Two nodes cannot both be called {sorted(names.values())}."
            raise ParameterError(msg)
        renamed = {node: names.get(node, node) for node in self.tasks}
        return Pipe(
            tasks={renamed[node]: task for node, task in self.tasks.items()},
            dependencies={
                renamed[node]: tuple(renamed[x] for x in given)
                for node, given in self.dependencies.items()
            },
            inputs=tuple(renamed[x] for x in self.inputs),
            outputs=tuple(renamed[x] for x in self.outputs),
        )

    def sorted_nodes(self) -> tuple[str, ...]:
        """
        Return the nodes in an order which runs each after its inputs.

        Kahn's algorithm, seeded with the sources in the order the pipe is
        fed, so a pipe always runs its tasks the same way round. Which of
        two independent branches runs first is not part of what a pipe
        means: tasks are pure, and the fingerprint is built on the shape of
        the graph rather than on any one walk of it.
        """
        remaining = {node: len(self.dependencies.get(node, ())) for node in self.tasks}
        downstream: dict[str, list[str]] = {node: [] for node in self.tasks}
        for node, given in self.dependencies.items():
            for upstream in given:
                downstream[upstream].append(node)
        # The sources are exactly the nodes with nothing wired into them,
        # which `check` has already made sure of.
        ready = list(self.inputs)
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
        if not self.outputs:
            msg = "A pipe has to return something; it names no outputs."
            raise ParameterError(msg)
        for name in self.outputs:
            if name not in self.tasks:
                msg = f"The pipe's output {name!r} is not one of its tasks."
                raise ParameterError(msg)
        fed = {x for x in self.tasks if not self.dependencies.get(x)}
        # Compared by count as well as by name: a node listed twice would
        # be run twice, and would take the place of the source it hides.
        if set(self.inputs) != fed or len(self.inputs) != len(fed):
            msg = (
                f"The pipe takes its inputs at {sorted(self.inputs)}, but the "
                f"nodes with nothing wired into them are {sorted(fed)}."
            )
            raise ParameterError(msg)
        order = self.sorted_nodes()
        # A task whose output reaches nothing is work the pipe would do and
        # throw away, which is a pipe built wrong rather than a slow one.
        reaching = set(self.outputs)
        for node in reversed(order):
            if node in reaching:
                reaching.update(self.dependencies.get(node, ()))
        if stranded := set(self.tasks) - reaching:
            msg = (
                f"Nothing the pipe returns is built from {sorted(stranded)}; "
                "every task has to feed an output."
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
            "outputs": list(self.outputs),
            "fingerprint": self.fingerprint,
        }

    @classmethod
    def from_dict(cls, document: Mapping) -> Pipe:
        """
        Return the pipe a document describes.

        The fingerprint a document carries is checked against the pipe read
        out of it, unless a task has moved on to another version since it
        was written -- which changes the fingerprint by design, and is why
        the check is a guard against an edited file rather than the answer
        to what the pipe is.
        """
        written_tasks = document["tasks"]
        tasks = {node: Task.from_dict(value) for node, value in written_tasks.items()}
        dependencies = {
            node: tuple(value)
            for node, value in document.get("dependencies", {}).items()
        }
        out = cls(
            tasks=tasks,
            dependencies=dependencies,
            inputs=tuple(document["inputs"]),
            outputs=tuple(document["outputs"]),
        )
        written = document.get("fingerprint")
        moved_on = holds_another_version(written_tasks)
        if written and not moved_on and written != out.fingerprint:
            msg = (
                f"The document says its fingerprint is {written}, and the "
                f"pipe it describes has {out.fingerprint}. Something edited "
                "it after it was written."
            )
            raise ParameterError(msg)
        return out

    def save(self, path: str | Path) -> Path:
        """
        Write this pipe to a file.

        The suffix picks the format; see
        [`write_workflow`](`dascore.workflow.serialize.write_workflow`).
        """
        return write_workflow(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Pipe:
        """Return the pipe a file holds; see `save`."""
        return cls.from_dict(read_workflow(Path(path)))

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
            # A name is whatever `relabel` or a document gave the node, so
            # the quotes which hold the label are escaped rather than left
            # for a name holding one to break the diagram with.
            label = node.replace('"', "#quot;")
            lines.append(f'    {name}["{label}"]')
        for node, given in self.dependencies.items():
            for upstream in given:
                lines.append(f"    {names[upstream]} --> {names[node]}")
        return "\n".join(lines)


def join(left: Any, right: Any) -> Pipe:
    """
    Return the pipe which runs one side and then the other.

    Either side may be several tasks or pipes: on the left they fan in,
    each feeding what follows, and on the right they fan out, each fed by
    what came before and each a result of its own.
    """
    upstream = [_as_pipe(x) for x in _sides(left)]
    downstream = [_as_pipe(x) for x in _sides(right)]
    tasks: dict[str, Task] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    inputs: list[str] = []
    given: list[str] = []
    for pipe in upstream:
        renamed = _merge(pipe, tasks, dependencies)
        inputs.extend(renamed[x] for x in pipe.inputs)
        given.extend(renamed[x] for x in pipe.outputs)
    outputs: list[str] = []
    for pipe in downstream:
        renamed = _merge(pipe, tasks, dependencies)
        # Every task the right hand side took its input from now takes the
        # left hand sides' outputs instead; anything wired inside it keeps
        # what it had.
        for node in pipe.sources():
            dependencies[renamed[node]] = tuple(given)
        outputs.extend(renamed[x] for x in pipe.outputs)
    return Pipe(
        tasks=tasks,
        dependencies=dependencies,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def _sides(value: Any) -> list:
    """Return one side of a join as the list of tasks or pipes it stands for."""
    sides = list(value) if isinstance(value, list | tuple) else [value]
    if not sides:
        msg = "A pipe cannot be joined to nothing."
        raise ParameterError(msg)
    return sides


def _as_pipe(value: Any) -> Pipe:
    """Return a task or pipe as a pipe."""
    if isinstance(value, Pipe):
        return value
    if isinstance(value, Task):
        node = default_key(value)
        return Pipe(tasks={node: value}, inputs=(node,), outputs=(node,))
    msg = f"A pipe joins tasks and pipes, not {type(value).__name__}."
    raise ParameterError(msg)


def _merge(
    pipe: Pipe,
    tasks: dict[str, Task],
    dependencies: dict[str, tuple[str, ...]],
) -> dict[str, str]:
    """
    Copy a pipe's nodes into a graph being built, and say what they became.

    A node keeps the name it had, so a name given for the reader survives
    being joined to something else; one which is taken already is numbered.
    """
    renamed = {}
    # The names this pipe has still to place are held against as well as
    # the ones already in the graph: numbering a collision onto a name its
    # own next node wants would push that node off the name it was given.
    pending = set(pipe.tasks)
    for node in pipe.sorted_nodes():
        pending.discard(node)
        name = unique_key(node, tasks.keys() | pending)
        renamed[node] = name
        tasks[name] = pipe.tasks[node]
    for node, given in pipe.dependencies.items():
        dependencies[renamed[node]] = tuple(renamed[x] for x in given)
    return renamed


def default_key(task: Task) -> str:
    """
    Return the name a task takes in a pipe which does not name it.

    The class, said the way a variable is spelled: `Decimate` becomes
    `decimate`. It says what the node is without saying what it was given,
    so changing a parameter leaves every reference to the node -- in
    `update`, in a mermaid diagram, in a provenance record -- where it was.
    """
    return _WORD_BREAK.sub("_", type(task).__name__).lower()


def unique_key(key: str, taken: Container[str]) -> str:
    """
    Return a node name nothing in a pipe has claimed yet.

    Nodes are named for their task, so the same task twice in one chain --
    ``decimate | decimate`` -- would be one node and would lose a step. The
    second copy is numbered instead: ``decimate_2``.
    """
    if key not in taken:
        return key
    copy = 2
    while f"{key}_{copy}" in taken:
        copy += 1
    return f"{key}_{copy}"


def _source_arguments(node: str, sources: tuple[str, ...], inputs: tuple) -> tuple:
    """Return what one source node is given to run."""
    # Nothing for a pipe run with nothing, whatever shape it is: its
    # sources make their own values, as a task declared `inputs=0` does.
    if not inputs:
        return ()
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
