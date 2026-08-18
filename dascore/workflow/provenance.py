"""
Two records of where data came from.

A [`Provenance`](`dascore.workflow.provenance.Provenance`) is the durable
one: a pipe, the version of DASCore which ran it, and the machine it ran
on, written next to the results and read back later.

A [`ProvenanceNode`](`dascore.workflow.provenance.ProvenanceNode`) is the
live one: one immutable node per step, holding the task which ran and the
nodes which fed it. A patch carries the node it came out of, so its whole
history is reachable from it without a store to look anything up in. This
module is the graph; building one as patches are processed comes with the
patch ids, in a later release.
"""

from __future__ import annotations

import json
import platform
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import ConfigDict, Field, ValidationError

import dascore as dc
from dascore.exceptions import ParameterError
from dascore.models.base import DascoreBaseModel
from dascore.models.types import FrozenDictType
from dascore.utils.documents import parse_document
from dascore.workflow.pipe import Pipe, default_key, unique_key
from dascore.workflow.serialize import (
    DOCUMENT,
    decode,
    encode,
    read_workflow,
    write_workflow,
)
from dascore.workflow.task import Task


class SourceInfo(DascoreBaseModel):
    """Where a patch was read from."""

    # A key which is not a field is a document written against another
    # version of this class, which is worth an error rather than a shrug.
    model_config = ConfigDict(extra="forbid")

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

    model_config = ConfigDict(extra="forbid")

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

    def __eq__(self, other) -> bool:
        """
        Two nodes are equal when they stand for the same step.

        Compared on what the step was and what it produced rather than by
        walking the graph behind it: the ids already answer for the whole
        lineage, and a deep comparison would recurse as far as the data has
        been processed.
        """
        if not isinstance(other, ProvenanceNode):
            return NotImplemented
        return self._identity() == other._identity()

    def __hash__(self) -> int:
        """Hash a node the way it compares."""
        return hash(self._identity())

    def _identity(self) -> tuple:
        """Return what makes this node the step it is."""
        task = self.task.fingerprint if self.task is not None else None
        # The pairs as well as the ids the step produced: two steps which
        # ran the same task on different data are not one step, and until
        # the ids are filled in they are all a node has to say so with.
        return (task, self.patch_id, self.processing_id, self.source, self.input_pairs)

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

        Data read from more than one file is one line saying how many
        rather than a line per file, and a step which took several inputs
        says how many, so a patch chunked from sixty files reads as two
        lines rather than sixty-one.
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
        against whatever data it is given. A graph a pipe has no way to say
        raises `ParameterError` rather than giving back a pipe which could
        not run: one where nothing was done, one where a step was fed both
        by an earlier step and straight from a source, one where the steps
        reading from the sources do not each take exactly one input, and
        any shape left which does not make a graph a pipe would accept.
        """
        steps = self.steps()
        if not steps:
            msg = "Nothing was done to this data, so there is no pipe to build."
            raise ParameterError(msg)
        names: dict[int, str] = {}
        tasks: dict[str, Task] = {}
        dependencies: dict[str, tuple[str, ...]] = {}
        # The steps which read straight from a source, and how many inputs
        # each of them took.
        fed: dict[str, int] = {}
        for node in steps:
            # steps() keeps only the nodes which hold a task.
            assert node.task is not None
            name = unique_key(default_key(node.task), tasks)
            names[id(node)] = name
            tasks[name] = node.task
            given = tuple(names[id(x)] for x in node.parents if x.task is not None)
            # What the step was given, which is what it recorded when a
            # parent was not tracked: a node holds a pair per input and a
            # parent only for the ones whose own node is known.
            taken = len(node.input_pairs) or len(node.parents)
            if given and len(given) != taken:
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
            else:
                fed[name] = taken
        # A pipe hands each of its sources one input, so it can describe one
        # step which took several -- a chunk of many files -- but only when
        # that step is the only one reading from the sources.
        if len(fed) > 1 and set(fed.values()) != {1}:
            msg = (
                "This graph has more than one step reading straight from its "
                "sources, and they do not all take one input, which a pipe "
                "has no way to describe."
            )
            raise ParameterError(msg)
        # Anything left which does not make a runnable graph -- steps which
        # never meet, for one -- is this graph having no pipe rather than a
        # pipe refusing to be built.
        try:
            return Pipe(
                tasks=tasks,
                dependencies=dependencies,
                inputs=tuple(fed),
                outputs=(names[id(steps[-1])],),
            )
        except ValidationError as problem:
            msg = f"This graph does not describe a pipe which could run: {problem}"
            raise ParameterError(msg) from problem

    def to_json(self, indent: int | None = 2) -> str:
        """Return the graph behind this node as JSON text."""
        return json.dumps(self._to_document(), indent=indent)

    @classmethod
    def from_json(cls, text: str) -> ProvenanceNode:
        """Return the graph some JSON text holds."""
        # Through the shared parser, so text which does not parse says so
        # the way every other document in DASCore does rather than raising
        # whatever the json module happened to raise.
        document = parse_document(text, "json", label="the provenance text")
        if not isinstance(document, Mapping) or "nodes" not in document:
            msg = "This text holds no record of what was done to any data."
            raise ParameterError(msg)
        return cls._from_document(document)

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

    model_config = ConfigDict(extra="forbid")

    pipe: Pipe
    dascore_version: str
    created_at: datetime
    python_version: str
    system_info: FrozenDictType[str, str] = Field(default_factory=dict)
    metadata: FrozenDictType[str, Any] = Field(default_factory=dict)
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
        """
        Return a document which describes this record.

        Every field is dumped by pydantic, so a field added later is written
        without this having to be edited. The three which pydantic has no
        document form for are excluded from that dump rather than written
        and overwritten: it would raise on a task holding an array, or on a
        time recorded alongside the run, before reaching the line which
        replaces what it produced.
        """
        written = {"pipe", "source_provenance", "metadata"}
        out = self.model_dump(mode="json", exclude=written)
        out["pipe"] = self.pipe.to_dict()
        out["source_provenance"] = [x.to_dict() for x in self.source_provenance]
        # Through the workflow encoding rather than pydantic's: metadata is
        # whatever the caller thought worth recording, which is the same
        # range of values a task's parameters cover.
        out["metadata"] = encode(dict(self.metadata), mode=DOCUMENT)
        out["fingerprint"] = self.fingerprint
        return out

    @classmethod
    def from_dict(cls, document: Mapping) -> Provenance:
        """Return the record a document describes."""
        fields = dict(document)
        # Written for a reader, and derived from the pipe, so it is not one
        # of the fields the record is rebuilt from.
        fields.pop("fingerprint", None)
        if "pipe" not in fields:
            msg = "A record of a run states the pipe it ran; this document has none."
            raise ParameterError(msg)
        sources = fields.pop("source_provenance", ())
        fields["pipe"] = Pipe.from_dict(fields["pipe"])
        fields["source_provenance"] = tuple(cls.from_dict(x) for x in sources)
        fields["metadata"] = decode(fields.get("metadata", {}))
        return cls(**fields)

    def save(self, path: str | Path) -> Path:
        """
        Write this record to a file.

        The suffix picks the format, as it does for a pipe; see
        [`write_workflow`](`dascore.workflow.serialize.write_workflow`).
        """
        return write_workflow(self.to_dict(), Path(path))

    @classmethod
    def load(cls, path: str | Path) -> Provenance:
        """Return the record a file holds; see `save`."""
        return cls.from_dict(read_workflow(Path(path)))
