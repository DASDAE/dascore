"""
Utils for working with attributes.
"""

from __future__ import annotations

from collections import ChainMap
from collections.abc import Sequence
from functools import reduce
from typing import Literal

import pandas as pd

import dascore as dc
from dascore.constants import attr_conflict_description
from dascore.exceptions import AttributeMergeError
from dascore.utils.docs import compose_docstring
from dascore.utils.misc import (
    _dict_list_diffs,
    iterate,
)


@compose_docstring(conflict_desc=attr_conflict_description)
def combine_patch_attrs(
    model_list: Sequence[dc.PatchAttrs],
    conflicts: Literal["drop", "raise", "keep_first"] = "raise",
    drop_attrs: Sequence[str] | None = None,
) -> dc.PatchAttrs:
    """
    Merge Patch Attributes along a dimension.

    Parameters
    ----------
    model_list
        A list of models.
    conflicts
        {conflict_desc}
    drop_attrs
        If provided, attributes which should be dropped.
    """

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
        # drop attributes specified.
        if drop := set(iterate(drop_attrs)):
            model_dicts = [
                {i: v for i, v in x.items() if i not in drop} for x in model_dicts
            ]
        return model_dicts

    def _replace_null_with_none(mod_dict_list):
        """Because NaN != NaN we need to replace those values so == works."""
        out = []
        for mod in mod_dict_list:
            out.append(
                {
                    i: (v if (isinstance(v, Sequence) or not pd.isnull(v)) else None)
                    for i, v in mod.items()
                }
            )
        return out

    def _keep_eq(d1, d2):
        """Keep only the values that are equal between d1/d2."""
        out = {}
        for i in set(d1) & set(d2):
            if not d1[i] == d2[i]:
                continue
            out[i] = d1[i]
        return out

    def _handle_other_attrs(mod_dict_list):
        """Check the other attributes and handle based on conflicts param."""
        if conflicts == "keep_first":
            return [dict(ChainMap(*mod_dict_list))]
        no_null_ = _replace_null_with_none(mod_dict_list)
        all_eq = all(no_null_[0] == x for x in no_null_[1:])
        if all_eq:
            return mod_dict_list
        if conflicts == "raise":
            # determine which keys are not equal to help debug.
            uneq_keys = _dict_list_diffs(mod_dict_list)
            msg = (
                "Cannot merge models, the following non-dim attrs are not "
                f"equal: {uneq_keys}. Consider setting the `conflict` or "
                f"`attr_conflict` arguments for more flexibility in merging "
                f"unequal coordinates."
            )
            raise AttributeMergeError(msg)
        final_dict = reduce(_keep_eq, mod_dict_list)
        return [final_dict]

    mod_dict_list = _get_model_dict_list(model_list)
    mod_dict_list = _handle_other_attrs(mod_dict_list)
    first = model_list[0]
    first_class = (
        _to_patch_attrs(first).__class__ if not isinstance(first, dict) else dict
    )
    cls = first_class if first_class is not dict else dc.PatchAttrs
    return cls(**mod_dict_list[0])
