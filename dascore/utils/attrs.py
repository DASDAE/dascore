"""
Utils for working with attributes.
"""

from __future__ import annotations

from collections import ChainMap, defaultdict
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
    is_valid_coord_str,
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

    def _get_model_dict_list(mod_list):
        """Get list of model dicts with optional dropped attrs."""
        model_dicts = [
            _to_patch_attrs(x).model_dump(exclude_defaults=True)
            if not isinstance(x, dict)
            else x
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


def separate_coord_info(
    obj,
    dims: tuple[str, ...] | None = None,
    required: Sequence[str] | None = None,
    cant_be_alone: tuple[str, ...] = ("units", "dtype"),
) -> tuple[dict, dict]:
    """
    Separate coordinate information from mixed attr-like metadata.

    This helper is still needed because DASCore still accepts a few mixed
    metadata shapes internally and in legacy IO paths. In particular, it is
    used to:

    - normalize flat scan/index-style keys such as ``time_min`` and
      ``distance_step`` into coordinate summary payloads
    - split coordinate updates from pure attrs in coord-manager code
    - unpack older nested ``{"coords": {...}}`` metadata payloads

    Supported input shapes include flat coord-style fields such as
    ``{time_min, time_max, time_step, ...}`` and nested coord dictionaries
    such as ``{coords: {time: {min, max, step}}}``.

    Parameters
    ----------
    obj
        The object or model to split.
    dims
        Optional dimension names used to recognize flat coord-style keys.
    required
        If provided, the required attributes (e.g., min, max, step).
    cant_be_alone
        Names which cannot be treated as coord info on their own.

    Returns
    -------
    A tuple of ``(coord_dict, attrs_dict)`` where coordinate-like metadata has
    been separated from the remaining pure attrs.
    """
    coord_summary_fields = tuple(dc.core.CoordSummary.model_fields)

    def _split_coord_key(key, prefixes=None):
        """Split flat coord summary key into coord name and field."""
        prefixes = tuple(iterate(prefixes)) if prefixes is not None else ()
        if prefixes:
            for prefix in sorted(prefixes, key=len, reverse=True):
                prefix_str = f"{prefix}_"
                if key.startswith(prefix_str):
                    field = key[len(prefix_str) :]
                    if field in coord_summary_fields:
                        return prefix, field
            return None
        parts = key.rsplit("_", 1)
        return tuple(parts)

    def _meets_required(coord_dict, strict=True):
        """
        Return True coord dict meets the minimum required keys.

        coord_dict represents potential coordinate fields.

        Strict ensures all required values exist.
        """
        if not coord_dict:
            return False
        if not required and (set(coord_dict) - cant_be_alone):
            return True
        if required or not strict:
            return set(coord_dict).issuperset(required)
        return False

    def _get_dims(obj):
        """Try to ascertain dims from keys in obj."""
        # check first for coord manager
        if isinstance(obj, dict) and hasattr(obj.get("coords", None), "dims"):
            return obj["coords"].dims

        # This object already has dims, just honor it.
        if dims := obj.get("dims", None):
            return tuple(dims.split(",")) if isinstance(dims, str) else dims

        potential_keys = defaultdict(set)
        for key in obj:
            if not is_valid_coord_str(key):
                continue
            coord_name, field = _split_coord_key(key)
            potential_keys[coord_name].add(field)
        return tuple(i for i, v in potential_keys.items() if _meets_required(v))

    def _get_coords_from_top_level(obj, out, dims):
        """First get coord info from top level."""
        for dim in iterate(dims):
            potential_coord = {}
            for key, value in obj.items():
                split = _split_coord_key(key, prefixes=(dim,))
                if split is None:
                    continue
                _, field = split
                potential_coord[field] = value
            if _meets_required(potential_coord, strict=False):
                out[dim] = potential_coord

    def _get_coords_from_coord_level(obj, out):
        """Get coords from coordinate level."""
        coords = obj.get("coords", {})
        if hasattr(coords, "to_summary_dict"):
            coords = coords.to_summary_dict()
        for key, value in coords.items():
            if hasattr(value, "to_summary"):
                value = value.to_summary()
            if hasattr(value, "model_dump"):
                value = value.model_dump()
            if _meets_required(value, strict=False):
                out[key] = value

    def _pop_keys(obj, out):
        """Pop out old keys for attrs, and unused keys from out."""
        # first coord subdict
        obj.pop("coords", None)
        # then top-level
        for coord_name, sub_dict in out.items():
            for thing_name in sub_dict:
                obj.pop(f"{coord_name}_{thing_name}", None)
            if "step" in sub_dict:
                obj.pop(f"d_{coord_name}", None)

    # sequence of short-circuit checks
    coord_dict = {}
    required = set(required) if required is not None else set()
    cant_be_alone = set(cant_be_alone)
    if obj is None:
        return coord_dict, {}
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    obj = dict(obj)
    # Check if dims need to be updated.
    new_dims = _get_dims(obj)
    if new_dims and new_dims != dims:
        dims = new_dims
    # this is already a dict of coord info.
    if dims and set(dims).issubset(set(obj)):
        return obj, {}
    _get_coords_from_coord_level(obj, coord_dict)
    _get_coords_from_top_level(obj, coord_dict, dims)
    _pop_keys(obj, coord_dict)
    return coord_dict, obj
