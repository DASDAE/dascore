"""
Machinery for describing, identifying and composing DASCore operations.

An operation is a [`Task`](`dascore.workflow.task.Task`): a frozen object
whose fields are its parameters, which knows its own fingerprint and can be
written to a document and read back. Alongside them are the two ids a patch
carries -- which data it is, and what was done to it -- which is how a
result says where it came from without a record of the run beside it.
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
from dascore.workflow.meta import PatchMeta
from dascore.workflow.processor import (
    PatchOp,
    PatchProcessor,
    fingerprint_call,
    register_implementation,
    register_kernel,
    resolve_patch_function,
)
from dascore.workflow.task import Task, intern, make_function_task_class, task
