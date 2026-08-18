"""
Machinery for describing, identifying and composing DASCore operations.

An operation is a [`Task`](`dascore.workflow.task.Task`): a frozen object
whose fields are its parameters, which knows its own fingerprint and can be
written to a document and read back.
"""

from __future__ import annotations

from dascore.workflow.serialize import (
    canonical_json,
    combine_hashes,
    decode,
    digest,
    encode,
)
from dascore.workflow.task import Task, intern, make_function_task_class, task
