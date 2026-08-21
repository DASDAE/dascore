"""
The operations DASCore performs which are not patch functions.

Concatenating, stacking and applying a ufunc are things done to patches,
but none of them is written as a `@patch_function`, so none has an `op`.
They still have to be named and fingerprinted, because they are what
advances a patch's `processing_id` -- hence one small `Task` each.

Anything which *is* a patch function needs nothing here:
[`PatchOp`](`dascore.workflow.processor.PatchOp`) already names it, which
is why there is no `Select` in this module.

Examples
--------
>>> import numpy as np
>>> from dascore.workflow.builtin import Ufunc
>>>
>>> added = Ufunc(name="add")
>>> assert added.fingerprint != Ufunc(name="subtract").fingerprint
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from dascore.workflow.task import Task


class Concatenate(Task):
    """
    Putting patches end to end along a dimension.

    Parameters
    ----------
    arguments
        The dimensions concatenated along and how much of each, in the
        order they were given: `(("time", None),)`.
    check_behavior
        What was done about patches which did not fit.
    conflict
        How conflicting attributes and coordinates were settled, for a
        planned concatenation (`Spool.concatenate`); None for the direct
        function, which has no such policy.
    dropped
        Coordinates the output does not carry, which its members held.
        Two outputs of the same members differ when one keeps a
        coordinate the other could not vouch for.

    Notes
    -----
    The arguments are pairs rather than a mapping because here the
    dimension is the *key* and `None` is its documented value, meaning
    "all of it". The canonical serializer drops a `None` mapping value --
    deliberately, so a parameter left at its default is the same call as
    one left out -- which would make concatenating along time and along
    distance one fingerprint. A pair keeps both halves.
    """

    arguments: tuple[tuple[str, Any], ...] = ()
    check_behavior: str | None = "warn"
    conflict: str | None = None
    dropped: tuple[str, ...] = ()

    @classmethod
    def from_kwargs(cls, check_behavior: str | None = "warn", **kwargs) -> Concatenate:
        """Return the task a call to `concatenate_patches` is."""
        return cls(arguments=tuple(kwargs.items()), check_behavior=check_behavior)


class Stack(Task):
    """
    Adding patches together.

    Parameters
    ----------
    dim_vary
        The dimension whose values were allowed to differ, if any.
    check_behavior
        What was done about patches which did not fit.
    """

    dim_vary: str | None = None
    check_behavior: str | None = "warn"


class Ufunc(Task):
    """
    A numpy ufunc applied to a patch.

    Parameters
    ----------
    name
        The ufunc's name, as numpy spells it: `"add"`, `"multiply"`.
    method
        Which way it was applied -- `"__call__"`, `"reduce"`,
        `"accumulate"` -- since a reduction is not the operation the plain
        call is.
    reversed
        Whether the patch was the right operand, so that `1 - patch` is
        not recorded as `patch - 1`.
    operands
        What the other operands were, for the ones which are not patches.
        A patch operand is not written here: it is a parent, and the ids
        already say so.
    kwargs
        Whatever else the call carried.
    """

    name: str
    method: str = "__call__"
    reversed: bool = False
    operands: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = Field(default_factory=dict)


class ArrayFunc(Task):
    """
    A numpy array function applied to a patch.

    Parameters
    ----------
    name
        The function's name, as numpy spells it: `"mean"`, `"concatenate"`.
    args
        The positional arguments, which for a reduction say which axis.
    kwargs
        Whatever the call carried, minus the patches themselves.
    """

    name: str
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = Field(default_factory=dict)
