"""
What a patch is, and what was done to it.

Two ids answer two different questions, and they move independently:

`patch_id` says **which data**. It survives every operation which does not
change what the data is *of* -- filtering, decimating, transposing, changing
units -- and changes only when data from more than one source is combined.

`processing_id` says **what was done**. It advances on every operation, by
folding the operation's fingerprint into the id the input carried, so that
two patches which took the same route from the same source arrive at the
same id and two which did not, do not.

Neither is a random number after the first: they are digests of what came
before, so the same data processed the same way gives the same
`processing_id` on another machine, in another process, next year. That
holds for `patch_id` too once a patch read from a file derives its id from
the source; until then one is minted for each patch built in memory, and
is stable only within the process which built it.

Examples
--------
>>> from dascore.workflow.identity import advance, fold_patch_ids
>>>
>>> # What was done: the operation folds into what came before.
>>> first = advance("", "0123456789abcdef")
>>> assert advance(first, "0123456789abcdef") != first
>>>
>>> # Which data: combining two sources makes a new answer.
>>> assert fold_patch_ids(["a", "b"]) != fold_patch_ids(["b", "a"])
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any
from uuid import uuid4

from dascore.workflow.serialize import combine_hashes, digest

# What a patch carries before anything has been done to it. Not a digest:
# an empty string is the identity element of `advance`, and reads as "this
# is how the data arrived" rather than as an operation which did nothing.
NOTHING_DONE = ""


# The attrs which say which data a patch is and what was done to it. They
# are folded rather than compared wherever patches are combined.
_ID_FIELDS = ("patch_id", "processing_id")


def fold_ids(attrs_list) -> dict[str, Any]:
    """
    Return the ids a patch combined from several inherits.

    Kept members only: a merge which dropped an incompatible patch did not
    use its data, so its id is not part of what this data is.
    """
    if not ids_enabled():
        return {}
    kept = list(attrs_list)
    if not kept:
        return {}
    # `getattr`, for the reason `with_patch_id` gives: attrs unpickled from
    # before these fields existed have neither.
    return {
        "patch_id": fold_patch_ids([patch_id_of(x) for x in kept]),
        "processing_id": fold_processing_ids([processing_id_of(x) for x in kept]),
    }


def stamp_combination(attrs, members, fingerprint: str):
    """
    Return the ids a patch made from several members carries.

    Parameters
    ----------
    attrs
        The attrs the result is being built with.
    members
        The attrs of the patches which actually went into it -- the kept
        ones. A member which was dropped for being incompatible did not
        contribute its data, so it is not part of what this data is.
    fingerprint
        The fingerprint of the operation which combined them.
    """
    if not ids_enabled():
        return attrs
    folded = fold_ids(members)
    if not folded:
        return attrs
    return attrs.update(
        patch_id=folded["patch_id"],
        processing_id=advance(folded["processing_id"], fingerprint),
    )


def patch_id_of(attrs) -> str:
    """Return which data some attrs say they are, or nothing if they cannot."""
    return getattr(attrs, "patch_id", NOTHING_DONE) or NOTHING_DONE


def processing_id_of(attrs) -> str:
    """Return what some attrs say was done, or nothing if they cannot."""
    return getattr(attrs, "processing_id", NOTHING_DONE) or NOTHING_DONE


def ids_enabled() -> bool:
    """Whether this process is keeping track of what a patch is."""
    # Imported here rather than at module scope: `dascore.config` is not
    # built while the workflow package is being imported.
    from dascore import get_config  # noqa: PLC0415

    return get_config().patch_provenance != "disabled"


def with_patch_id(attrs):
    """
    Return attrs which name which data they belong to.

    Minted rather than derived when there is nothing to derive one from: a
    patch built in memory is not the same data as anything else. A reader
    which knows better -- one whose file stores an id, or which can derive
    one from the path -- stamps over this.
    """
    # `getattr`, not attribute access: a `PatchAttrs` unpickled from before
    # these fields existed has neither, and `PatchAttrs.from_dict` hands an
    # instance back untouched rather than revalidating it into one.
    if getattr(attrs, "patch_id", None) or not ids_enabled():
        return attrs
    return attrs.update(patch_id=new_patch_id())


def new_patch_id() -> str:
    """
    Return an id for data which names no source.

    A random one, because there is nothing to derive it from: a patch built
    in memory from an array is not the same data as anything else, and
    saying so is more honest than hashing the values and claiming two
    coincidentally equal arrays are one datum.
    """
    return uuid4().hex


def source_patch_id(format_name: str, path: str, key: object = None) -> str:
    """
    Return the id of data read from a file.

    Derived rather than random, so reading the same file twice -- in two
    processes, on two days -- gives the same answer and the ids in a
    result can be traced back to what they came from.

    Parameters
    ----------
    format_name
        The format the reader was using.
    path
        The resolved path or URI the data came from.
    key
        What names this patch within the file, when a file holds more than
        one. The reader's own key if it has one, else the ordinal.

    Notes
    -----
    An id derived this way is stable for a given archive laid out the same
    way; it is not stable across hosts, because the path is part of it. A
    format which stores an id -- DASDAE does -- keeps that one instead, and
    those are stable everywhere.
    """
    return digest({"format": format_name, "path": path, "key": key})


def advance(processing_id: str, fingerprint: str) -> str:
    """
    Return the processing id an operation leads to.

    Parameters
    ----------
    processing_id
        What the input carried.
    fingerprint
        The operation's fingerprint; see
        [`fingerprint_call`](`dascore.workflow.processor.fingerprint_call`).
    """
    # `combine_hashes` rather than a digest of a mapping: both halves are
    # already digests, and this is what it is for -- an ordered series of
    # them, where the order is part of the answer.
    return combine_hashes([processing_id, fingerprint])


def fold_patch_ids(patch_ids: Sequence[str]) -> str:
    """
    Return the patch id of a patch combined from several.

    Ordered, and not deduplicated: how many patches went in and in what
    order is part of what the data *is*, so stacking a patch with itself is
    not the same datum as the patch alone.

    A single id folds to itself, so an operation which combines one patch
    with nothing leaves the id where it was.
    """
    ids = tuple(patch_ids)
    if len(ids) == 1:
        return ids[0]
    # Data which never said which data it was does not acquire an identity
    # by being combined: folding a pile of empty strings would give every
    # such combination one deterministic id, and they are not one datum.
    if not any(ids):
        return NOTHING_DONE
    return combine_hashes(ids)


def fold_processing_ids(processing_ids: Iterable[str]) -> str:
    """
    Return the processing id which several inputs agree on.

    When every input took the same route, that route is the answer: a
    chunk of sixty windows of one file has one history, not sixty.

    When they did not, the answer is a digest of the distinct routes in
    the order they were first seen -- so combining differently processed
    data says so, and says it the same way every time.
    """
    seen: list[str] = []
    for processing_id in processing_ids:
        if processing_id not in seen:
            seen.append(processing_id)
    if not seen:
        return NOTHING_DONE
    if len(seen) == 1:
        return seen[0]
    return combine_hashes(seen)
