"""Describe, identify, and compose DASCore operations.

A [`Task`](`dascore.workflow.task.Task`) is an immutable operation whose fields
hold its parameters. Tasks can be fingerprinted and serialized. Patches carry
separate IDs for their source data and processing history.
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
