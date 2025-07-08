"""
Utils for working with attributes.
"""

from __future__ import annotations

import warnings
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
    all_diffs_close_enough,
    get_middle_value,
    is_valid_coord_str,
    iterate,
)


@compose_docstring(conflict_desc=attr_conflict_description)
def combine_patch_attrs(
    model_list: Sequence[dc.PatchAttrs],
    coord_name: str | None = None,
    conflicts: Literal["drop", "raise", "keep_first"] = "raise",
    drop_attrs: Sequence[str] | None = None,
    coord: None | dc.core.coords.BaseCoord = None,
) -> dc.PatchAttrs:
    """
    Merge Patch Attributes along a dimension.

    Parameters
    ----------
    model_list
        A list of models.
    coord_name
        The coordinate, usually a dimension coord, along which to merge.
    conflicts
        {conflict_desc}
    drop_attrs
        If provided, attributes which should be dropped.
    coord
        The coordinate for the new values of dim. This is provided as a
        shortcut if it has already been computed.
    """
    # TODO this is a monstrosity! need to refactor.
    model_fields = dc.core.CoordSummary.model_fields
    eq_coord_fields = set(model_fields) - {"min", "max", "step"}

    def _get_merge_coords(model_dicts):
        """Get the coordinates to merge together."""
        # pop out coord to merge on
        merge_coords = []
        for mod in model_dicts:
            coords = mod.get("coords", {})
            maybe_coord = coords.pop(coord_name, None)
            if maybe_coord is not None:
                merge_coords.append(maybe_coord)
        # dealing with empty coords
        if not merge_coords:
            return {}
        return merge_coords

    def _get_merge_dict_list(merge_coords):
        """Get a list of {attrs: []}."""
        out = defaultdict(list)
        for mdict in merge_coords:
            for key, val in mdict.items():
                out[key].append(val)
        return out

    def _get_new_coord(model_dicts):
        """Get the merged coord to set on all models."""
        merge_coords = _get_merge_coords(model_dicts)
        merge_dict_list = _get_merge_dict_list(merge_coords)
        out = {}
        # empy dicts
        if not merge_dict_list:
            return dict(merge_dict_list)
        # all these should be equal else raise.
        for key in eq_coord_fields:
            if not len(vals := merge_dict_list[key]):
                continue
            if not all(vals[0] == x for x in vals):
                msg = f"Cant merge patch attrs, key {key} not equal."
                raise AttributeMergeError(msg)
            out[key] = vals[0]
        out["min"] = min(merge_dict_list["min"])
        out["max"] = max(merge_dict_list["max"])
        step = None
        if all_diffs_close_enough(steps := merge_dict_list["step"]):
            step = get_middle_value(steps)
        out["step"] = step
        return out

    def _get_model_dict_list(mod_list):
        """Get list of model_dict, merge along dim if specified."""
        model_dicts = [
            x.model_dump(exclude_defaults=True) if not isinstance(x, dict) else x
            for x in mod_list
        ]
        # drop attributes specified.
        if drop := set(iterate(drop_attrs)):
            model_dicts = [
                {i: v for i, v in x.items() if i not in drop} for x in model_dicts
            ]
        # no coordinate to merge on, just return dicts.
        if not coord_name:
            return model_dicts
        # drop models which don't have required coords
        model_dicts = [x for x in model_dicts if coord_name in x.get("coords", {})]
        # coordinate can be determined from existing coords.
        if coord is None:
            new_coord = _get_new_coord(model_dicts)
        else:  # or if one was passed just use its summary.
            new_coord = coord.to_summary()
        if new_coord:
            for mod in model_dicts:
                mod["coords"][coord_name] = new_coord
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
    first_class = model_list[0].__class__
    cls = first_class if not first_class == dict else dc.PatchAttrs
    if not len(mod_dict_list):
        msg = "Failed to combine patch attrs"
        raise AttributeMergeError(msg)
    return cls(**mod_dict_list[0])


def decompose_attrs(attr_list: Sequence[dc.PatchAttrs], exclude=("history",)):
    """Function to decompose attributes into series."""

    def _get_uri_and_hash(model):
        """Pop out this models uri and its hash."""
        uri_key = "uri" if "uri" in model else "path"
        assert uri_key in model, "all models must have uri or path"
        uri = model.pop(uri_key)
        uri_hash = hash(uri)
        return uri, uri_hash

    def _add_coords(coord_dict, dim_dict, coords, uri_hash, dims):
        """Add coordinates to structure."""
        for name, coord in coords.items():
            out_dct = coord_dict if name not in dims else dim_dict
            coord["index"] = uri_hash
            coord["units"] = "" if coord["units"] is None else coord["units"]
            # strip off the [] in datetimes/timedeltas. We will ensure the
            # precision is always ns downstream.
            if dtype := coord.get("dtype"):
                coord["dtype"] = dtype.split("[")[0]
            out_dct[name][coord["dtype"]].append(coord)

    def _add_attrs(attr_dict, model, uri_hash):
        """Add attrs to coordinate structure."""
        for name, value in model.items():
            if name.startswith("_"):
                continue
            entry = {"value": value, "index": uri_hash}
            attr_dict[name][str(type(value))].append(entry)

    def _to_df(dict_list):
        """Convert a dict list to a dataframe."""
        df = pd.DataFrame(dict_list).set_index("index").dropna(how="all")
        # TODO may need to do something faster here, not sure.
        empty_str = df == ""
        null = pd.isnull(df)
        to_keep = ~(empty_str | null).all(axis="columns")
        return df[to_keep]

    def _pandify(decom):
        """Convert all the nested dicts to series."""
        out = {}
        for key, value in decom.items():
            if isinstance(value, list):
                df = _to_df(value)
                if not df.empty:
                    out[key] = df
            else:
                out[key] = _pandify(value)
        return out

    dumped = [x.model_dump(exclude=exclude) for x in attr_list]
    out = {
        "uri": [],
        "attrs": defaultdict(lambda: defaultdict(list)),
        "coords": defaultdict(lambda: defaultdict(list)),
        "dims": defaultdict(lambda: defaultdict(list)),
    }
    for ind, model in enumerate(dumped):
        uri, uri_hash = _get_uri_and_hash(model)
        out["uri"].append({"uri": uri, "index": uri_hash})
        dims = model.get("dims", "").split(",")
        # need to handle coordinates and regular attrs differently
        coords = model.pop("coords", {})
        _add_coords(out["coords"], out["dims"], coords, uri_hash, dims)
        _add_attrs(out["attrs"], model, uri_hash)
    return _pandify(out)


def separate_coord_info(
    obj,
    dims: tuple[str, ...] | None = None,
    required: Sequence[str] | None = None,
    cant_be_alone: tuple[str, ...] = ("units", "dtype"),
) -> tuple[dict, dict]:
    """
    Separate coordinate information from attr dict.

    These can be in the flat-form (ie {time_min, time_max, time_step, ...})
    or a nested coord: {coords: {time: {min, max, step}}

    Parameters
    ----------
    obj
        The object or model to
    dims
        The dimension to look for.
    required
        If provided, the required attributes (e.g., min, max, step).
    cant_be_alone
        names which cannot be on their own.

    Returns
    -------
    coord_dict and attrs_dict.
    """

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
            potential_keys[key.split("_")[0]].add(key.split("_")[1])
        return tuple(i for i, v in potential_keys.items() if _meets_required(v))

    def _get_coords_from_top_level(obj, out, dims):
        """First get coord info from top level."""
        for dim in iterate(dims):
            potential_coord = {
                i.split("_")[1]: v for i, v in obj.items() if is_valid_coord_str(i, dim)
            }
            # nasty hack for handling d_{dim} for backward compatibility.
            if (bad_name := f"d_{dim}") in obj:
                msg = f"d_{dim} is deprecated, use {dim}_step"
                warnings.warn(msg, DeprecationWarning, stacklevel=3)
                potential_coord["step"] = obj[bad_name]

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
        obj["dims"] = new_dims
        dims = new_dims
    # this is already a dict of coord info.
    if dims and set(dims).issubset(set(obj)):
        return obj, {}
    _get_coords_from_coord_level(obj, coord_dict)
    _get_coords_from_top_level(obj, coord_dict, dims)
    _pop_keys(obj, coord_dict)
    if "dims" not in obj and dims is not None:
        obj["dims"] = dims
    return coord_dict, obj
