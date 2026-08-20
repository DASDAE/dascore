"""
Machinery for describing, identifying and composing DASCore operations.

An operation is a [`Task`](`dascore.workflow.task.Task`): a frozen object
whose fields are its parameters, which knows its own fingerprint and can be
written to a document and read back. Several of them joined with ``|`` make
a [`Pipe`](`dascore.workflow.pipe.Pipe`), which is one operation again, and
a [`Provenance`](`dascore.workflow.provenance.Provenance`) records a pipe
and the run which used it.
"""

from __future__ import annotations

from dascore.workflow.serialize import (
    canonical_json,
    combine_hashes,
    decode,
    digest,
    encode,
)
from dascore.workflow.builtin import ArrayFunc, Concatenate, Stack, Ufunc
from dascore.workflow.identity import (
    advance,
    fold_patch_ids,
    fold_processing_ids,
    new_patch_id,
    source_patch_id,
)
from dascore.workflow.pipe import Pipe
from dascore.workflow.processor import (
    PatchOp,
    PatchProcessor,
    fingerprint_call,
    register_implementation,
    resolve_patch_function,
)
from dascore.workflow.provenance import Provenance, ProvenanceNode, SourceInfo
from dascore.workflow.task import Task, intern, make_function_task_class, task
