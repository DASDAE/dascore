"""
Utils for working with attributes.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from typing import Literal

import numpy as np
import pandas as pd

import dascore as dc
from dascore.constants import attr_conflict_description
from dascore.exceptions import AttributeMergeError, ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import iterate
from dascore.workflow.identity import _ID_FIELDS, fold_ids

_VALID_CONFLICT_VALUES = ("drop", "raise", "keep_first")


def validate_conflict(conflict: str) -> Literal["drop", "raise", "keep_first"]:
    """Ensure a conflict argument is a supported value."""
    if conflict not in _VALID_CONFLICT_VALUES:
        msg = f"conflict must be one of {_VALID_CONFLICT_VALUES}, got {conflict!r}."
        raise ParameterError(msg)
    return conflict


@compose_docstring(conflict_desc=attr_conflict_description)
def combine_patch_attrs(
    model_list: Sequence[dc.PatchAttrs],
    conflict: Literal["drop", "raise", "keep_first"] = "raise",
    drop_attrs: Sequence[str] | None = None,
) -> dc.PatchAttrs:
    """
    Merge Patch Attributes along a dimension.

    Parameters
    ----------
    model_list
        A list of models.
    conflict
        {conflict_desc}
    drop_attrs
        If provided, attributes which should be dropped.
    """
    validate_conflict(conflict)

    def _to_patch_attrs(model):
        """Normalize supported attr-like inputs to PatchAttrs."""
        if isinstance(model, dc.Patch):
            model = model._attrs
        if isinstance(model, dc.PatchAttrs):
            return model
        return dc.PatchAttrs.from_dict(model)

    def _drop_private_keys(model_dict):
        """Remove private attrs so they don't affect compatibility merges."""
        return {
            key: value for key, value in model_dict.items() if not key.startswith("_")
        }

    def _get_model_dict_list(mod_list):
        """Get list of model dicts with optional dropped attrs."""
        model_dicts = [
            _drop_private_keys(_to_patch_attrs(x).model_dump(exclude_defaults=True))
            for x in mod_list
        ]
        # Taken out before anything compares them: two patches which are
        # different data, or which were processed differently, are exactly
        # what a merge is for, so an id must never make one raise. The fold
        # below decides what the result carries instead.
        model_dicts = [
            {i: v for i, v in x.items() if i not in _ID_FIELDS} for x in model_dicts
        ]
        # drop attributes specified.
        if drop := set(iterate(drop_attrs)):
            model_dicts = [
                {i: v for i, v in x.items() if i not in drop} for x in model_dicts
            ]
        return model_dicts

    def _handle_other_attrs(mod_dict_list):
        """
        Fold the attrs: the members' values must agree, missing included.

        Attrs are scalars, and an empty one (None, NaN, "") is a value
        like any other: it equals another empty one and nothing else. A
        merge combines a collection, which needs one value per attr, so
        differing values are a conflict handled per `conflict`.
        """
        keys = list(dict.fromkeys(key for model in mod_dict_list for key in model))
        out, conflicts = {}, []
        for key in keys:
            values = [
                None if _is_missing(value) else value
                for value in (x.get(key) for x in mod_dict_list)
            ]
            first = values[0]
            agree = all(_values_equal(first, x) for x in values[1:])
            if agree or conflict == "keep_first":
                if first is not None:
                    out[key] = first
            else:
                conflicts.append(key)
        if conflicts and conflict == "raise":
            msg = (
                "Cannot merge models, the following non-dim attrs hold "
                f"conflicting values: {conflicts}. Consider setting the "
                "`conflict` argument for more flexibility in merging "
                "unequal attributes."
            )
            raise AttributeMergeError(msg)
        return [out]

    mod_dict_list = _get_model_dict_list(model_list)
    ids = fold_ids([_to_patch_attrs(x) for x in model_list])
    # History is never compared (processing differing between members is
    # what a merge is for); the first member's is carried, like the ids.
    history = (
        {"history": mod_dict_list[0]["history"]}
        if "history" in mod_dict_list[0]
        else {}
    )
    mod_dict_list = [
        {i: v for i, v in x.items() if i != "history"} for x in mod_dict_list
    ]
    mod_dict_list = _handle_other_attrs(mod_dict_list)
    first = model_list[0]
    first_class = (
        _to_patch_attrs(first).__class__ if not isinstance(first, dict) else dict
    )
    cls = first_class if first_class is not dict else dc.PatchAttrs
    return cls(**{**mod_dict_list[0], **history, **ids})


def _is_missing(value) -> bool:
    """
    True for an attr nobody recorded: None, NaN/NaT, or an empty string.

    The one spelling of "missing": every rule which compares attrs
    normalizes to it first, so a patch which never stated an attr, one
    which stated null, and one which stated "" are the same patch as far
    as kind, the merge conflict policy, and units are concerned.
    """
    return value is None or (np.ndim(value) == 0 and (pd.isnull(value) or value == ""))


def known_only(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Null out the missing values ("" or null) of a frame or series."""
    return values.where(values.notna() & (values != ""))


def _values_equal(value1, value2) -> bool:
    """Compare two attr values; arrays compare as a whole."""
    if isinstance(value1, np.ndarray) or isinstance(value2, np.ndarray):
        return np.array_equal(value1, value2)
    return bool(value1 == value2)


def warn_if_histories_differ(attrs_list: Sequence[dc.PatchAttrs], operation: str):
    """
    Warn when members spliced into one array were processed differently.

    History never decides whether patches combine, but a raw patch merged
    beside a filtered one is usually a mistake, so say so. Used by the
    operations which lay members side by side along a dimension (merge,
    concatenate), not by elementwise ones, where mixing processing is the
    point (a residual is filtered minus raw).
    """
    histories = {tuple(iterate(x.history)) for x in attrs_list}
    if len(histories) > 1:
        msg = (
            f"{operation} patches whose histories differ; the output carries "
            "the first patch's history."
        )
        warnings.warn(msg, UserWarning, stacklevel=3)
