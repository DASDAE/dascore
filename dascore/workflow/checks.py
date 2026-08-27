"""
What an operation requires of a patch before it runs.

A patch function, and a `PatchProcessor` written by hand for one, may say
which dimensions, coordinates and attributes it needs. Both ask here, so the
answer is the same however the operation was reached.

These live outside `dascore.utils.patch` because `dascore.workflow.processor`
is imported while `dascore.utils.patch` is still being imported -- from its
header, by way of this module -- so importing `dascore.utils.patch` back would
raise on a partially initialized module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dascore.constants import PatchType
from dascore.exceptions import PatchAttributeError, PatchCoordinateError
from dascore.utils.misc import iterate

# What a required-attrs declaration may be: names, or names against values.
attr_type = dict[str, Any] | str | Sequence[str] | None


def check_patch_coords(
    patch: PatchType,
    dims: Sequence[str] | None = None,
    coords: Sequence[str] | None = None,
) -> PatchType:
    """
    Check that a patch has the required coordinates, else raise.

    Parameters
    ----------
    patch
        The input patch
    dims
        A dimension name, or a sequence of them.
    coords
        A coordinate name, or a sequence of them.

    Raises
    ------
    PatchCoordinateError
        If the patch is missing any of them.
    """
    # Each half is skipped when nothing is declared, which is the common
    # case: building the patch's dim and coord sets is the whole cost here,
    # and this runs in front of every patch function.
    # `iterate` rather than the value itself: a single name is spelled as a
    # string everywhere else in DASCore, and iterating one gives characters.
    missing = set()
    if dims:
        missing |= set(iterate(dims)) - set(patch.dims)
    if coords:
        missing |= set(iterate(coords)) - set(patch.coords.coord_map)
    if missing:
        msg = f"patch is missing required coordinates: {tuple(missing)}"
        raise PatchCoordinateError(msg)
    return patch


def check_patch_attrs(patch: PatchType, required_attrs: attr_type) -> PatchType:
    """
    Check for expected attributes.

    Parameters
    ----------
    patch
        The patch to validate
    required_attrs
        The expected attrs. Can be a name, a sequence of them, or a mapping.
        If a mapping, also check that the values are equal.

    Raises
    ------
    PatchAttributeError
        If the patch is missing any of them, or holds a different value.
    """
    if required_attrs is None:
        return patch
    # test that patch attr mapping is equal
    held = dict(patch.attrs)
    wanted = (
        set(required_attrs)
        if isinstance(required_attrs, Mapping)
        else set(iterate(required_attrs))
    )
    # Asked before any value is read: an attr the patch does not carry is a
    # missing attr, not a lookup error from the middle of a comprehension.
    if missing := wanted - set(held):
        msg = f"Patch is missing the following attributes: {missing}"
        raise PatchAttributeError(msg)
    if isinstance(required_attrs, Mapping):
        sub = {i: held[i] for i in required_attrs}
        if sub != dict(required_attrs):
            msg = f"Patch's attrs {sub} are not required attrs: {required_attrs}"
            raise PatchAttributeError(msg)
    return patch
