"""
Two records of where data came from.

A [`Provenance`](`dascore.workflow.provenance.Provenance`) is the durable
one: a pipe, the version of DASCore which ran it, and the machine it ran
on, written next to the results and read back later.

A [`ProvenanceNode`](`dascore.workflow.provenance.ProvenanceNode`) is the
live one: one immutable node per step, holding the task which ran and the
nodes which fed it. A patch carries the node it came out of, so its whole
history is reachable from it without a store to look anything up in.
Building the graph as patches are processed is PR 4a's work; this module
is the graph itself.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.workflow.pipe import COPY_SEP, Pipe, _dump, _parse
from dascore.workflow.task import Task


class SourceInfo(DascoreBaseModel):
    """Where a patch was read from."""

    model_config = ConfigDict(frozen=True)

    format: str = ""
    path: str = ""
    # What names this patch inside a file holding several.
    key: str = ""


class ProvenanceNode(DascoreBaseModel):
    """
    One step of what was done to some data.

    A node holds the task which ran, the nodes which produced its inputs,
    and the ids the step produced. Nodes are never edited: a further step
    makes another node which points back at this one, so a graph is shared
    by everything downstream of it rather than copied.
    """

    model_config = ConfigDict(frozen=True)

    # None for a source: nothing was done to it, it was read.
    task: Task | None = None
    parents: tuple[ProvenanceNode, ...] = ()
    # The ids of the inputs, including any whose node is not known.
    input_pairs: tuple[tuple[str, str], ...] = ()
    patch_id: str = ""
    processing_id: str = ""
    source: SourceInfo | None = None
    # Which array backend ran the step; an execution detail, recorded here
    # rather than in the fingerprint, which names the operation.
    backend: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def __hash__(self) -> int:
        """Hash a node by what it produced and what produced it."""
        return hash((self.patch_id, self.processing_id, id(self.task)))

    def walk(self) -> Iterator[ProvenanceNode]:
        """
        Yield every node behind this one and then this one, each once.

        Oldest first, so a node is always yielded after the nodes which fed
        it. A graph which narrows -- many files chunked into one patch --
        shares its ancestors, so a node reached twice is yielded once.
        """
        seen: set[int] = set()
        # Iterative rather than recursive: a long chain of steps is a deep
        # graph, and python's stack is shallower than a spool loop.
        stack: list[tuple[ProvenanceNode, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                yield node
            elif id(node) not in seen:
                seen.add(id(node))
                stack.append((node, True))
                stack.extend((x, False) for x in reversed(node.parents))

    def steps(self) -> tuple[ProvenanceNode, ...]:
        """Return the nodes which did something, oldest first."""
        return tuple(x for x in self.walk() if x.task is not None)

    def sources(self) -> tuple[SourceInfo, ...]:
        """Return where the data behind this node was read from."""
        return tuple(x.source for x in self.walk() if x.source is not None)

    def describe(self) -> str:
        """
        Return a readable listing of what was done, oldest first.

        A step which took many inputs is written once, with how many, so a
        patch chunked from sixty files reads as one line rather than sixty.
        """
        lines = []
        for source in self.sources():
            lines.append(f"read {source.format or 'patch'} {source.path}".rstrip())
        if len(lines) > 1:
            lines = [f"read {len(lines)} sources"]
        for node in self.steps():
            name = type(node.task).__name__
            fan_in = (
                f" over {len(node.parents)} inputs" if len(node.parents) > 1 else ""
            )
            lines.append(f"{name}{fan_in}")
        return "\n".join(lines)

    def to_pipe(self) -> Pipe:
        """
        Return a pipe which would do again what this node records.

        The sources are left out: a pipe describes what to do, and is run
        against whatever data it is given.
        """
        steps = self.steps()
        if not steps:
            msg = "Nothing was done to this data, so there is no pipe to build."
            raise ParameterError(msg)
        names: dict[int, str] = {}
        tasks: dict[str, Task] = {}
        dependencies: dict[str, tuple[str, ...]] = {}
        for node in steps:
            # steps() keeps only the nodes which hold a task.
            assert node.task is not None
            name = _unique_name(node.task.fingerprint, tasks)
            names[id(node)] = name
            tasks[name] = node.task
            given = tuple(names[id(x)] for x in node.parents if x.task is not None)
            if given and len(given) != len(node.parents):
                # Every input has to come from somewhere the pipe can name.
                # A step fed partly by an earlier step and partly by data
                # read straight from a file is the one shape it cannot say;
                # a step fed only by sources becomes the pipe's own source.
                msg = (
                    f"{type(node.task).__name__} took some of its inputs from "
                    "earlier steps and some straight from a source, which a "
                    "pipe has no way to describe."
                )
                raise ParameterError(msg)
            if given:
                dependencies[name] = given
        return Pipe(tasks=tasks, dependencies=dependencies, output=names[id(steps[-1])])

    def to_json(self, indent: int | None = 2) -> str:
        """Return the graph behind this node as JSON text."""
        return json.dumps(self._to_document(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> ProvenanceNode:
        """Return the graph some JSON text holds."""
        return cls._from_document(json.loads(text))

    def _to_document(self) -> dict[str, Any]:
        """Return a document holding this node and everything behind it."""
        # Keyed by identity while it is built, so a node reached twice is
        # written once and read back as one node again.
        order: dict[int, int] = {}
        written: list[dict[str, Any]] = []
        for node in self.walk():
            order[id(node)] = len(written)
            written.append(node._to_entry(order))
        return {"nodes": written, "output": order[id(self)]}

    def _to_entry(self, order: Mapping[int, int]) -> dict[str, Any]:
        """Return this node alone, naming its parents by their place."""
        return {
            "task": self.task.to_dict() if self.task is not None else None,
            "parents": [order[id(x)] for x in self.parents],
            "input_pairs": [list(x) for x in self.input_pairs],
            "patch_id": self.patch_id,
            "processing_id": self.processing_id,
            "source": self.source.model_dump() if self.source is not None else None,
            "backend": self.backend,
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def _from_document(cls, document: Mapping) -> ProvenanceNode:
        """Return the graph a document describes."""
        built: list[ProvenanceNode] = []
        for entry in document["nodes"]:
            source = entry.get("source")
            task = entry.get("task")
            built.append(
                cls(
                    task=Task.from_dict(task) if task is not None else None,
                    parents=tuple(built[x] for x in entry["parents"]),
                    input_pairs=tuple(tuple(x) for x in entry["input_pairs"]),
                    patch_id=entry["patch_id"],
                    processing_id=entry["processing_id"],
                    source=SourceInfo(**source) if source is not None else None,
                    backend=entry["backend"],
                    created_at=datetime.fromisoformat(entry["created_at"]),
                )
            )
        return built[document["output"]]


class Provenance(DascoreBaseModel):
    """
    A record of a pipe and the run which produced some data.

    This is the durable half: what a file written beside the results holds,
    and what makes them reproducible when the objects which made them are
    long gone.
    """

    model_config = ConfigDict(frozen=True)

    pipe: Pipe
    dascore_version: str
    created_at: datetime
    python_version: str
    system_info: Mapping[str, str] = Field(default_factory=dict)
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    # The provenance of whatever this run was given, so a chain of runs
    # reads back as a chain.
    source_provenance: tuple[Provenance, ...] = ()

    @property
    def fingerprint(self) -> str:
        """Return the fingerprint of the pipe this records."""
        return self.pipe.fingerprint

    def __hash__(self) -> int:
        """Hash a record by the pipe it holds."""
        return hash(self.fingerprint)

    @classmethod
    def from_pipe(cls, pipe: Pipe, **metadata) -> Provenance:
        """Return a record of a pipe, this version of DASCore, and this host."""
        return cls(
            pipe=pipe,
            dascore_version=dc.__version__,
            created_at=datetime.now(timezone.utc),
            python_version=platform.python_version(),
            system_info={
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a document which describes this record."""
        return {
            "pipe": self.pipe.to_dict(),
            "fingerprint": self.fingerprint,
            "dascore_version": self.dascore_version,
            "created_at": self.created_at.isoformat(),
            "python_version": self.python_version,
            "system_info": dict(self.system_info),
            "metadata": dict(self.metadata),
            "source_provenance": [x.to_dict() for x in self.source_provenance],
        }

    @classmethod
    def from_dict(cls, document: Mapping) -> Provenance:
        """Return the record a document describes."""
        sources = document.get("source_provenance", ())
        return cls(
            pipe=Pipe.from_dict(document["pipe"]),
            dascore_version=document["dascore_version"],
            created_at=datetime.fromisoformat(document["created_at"]),
            python_version=document["python_version"],
            system_info=document.get("system_info", {}),
            metadata=document.get("metadata", {}),
            source_provenance=tuple(cls.from_dict(x) for x in sources),
        )

    def save(self, path: str | Path) -> Path:
        """
        Write this record to a file.

        The suffix picks the format, as it does for a pipe: ``.yaml`` or
        ``.yml`` write YAML, anything else writes JSON.
        """
        path = Path(path)
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(_dump(self.to_dict(), path), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> Provenance:
        """Return the record a file holds; see `save`."""
        path = Path(path)
        return cls.from_dict(_parse(path.read_text(encoding="utf-8"), path))


def _unique_name(fingerprint: str, taken: Mapping[str, Any]) -> str:
    """Return a node name which nothing in a pipe has claimed yet."""
    # The same task twice in one chain is two steps, and two nodes; see
    # `dascore.workflow.pipe`, which names repeats the same way.
    name = fingerprint
    copy = 0
    while name in taken:
        copy += 1
        name = f"{fingerprint}{COPY_SEP}{copy}"
    return name
