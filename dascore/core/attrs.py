"""Pydantic schemas used by DASCore."""
from __future__ import annotations

import warnings
from collections import ChainMap
from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from functools import reduce
from typing import Annotated, Literal
from typing import Any

import pandas as pd
from pydantic import ConfigDict, PlainValidator
from pydantic import Field, model_validator
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    basic_summary_attrs,
    max_lens,
    PatchType,
)
from dascore.core.coordmanager import CoordManager
from dascore.core.coords import BaseCoord, CoordRange, CoordSummary
from dascore.exceptions import AttributeMergeError, IncompatiblePatchError
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    all_diffs_close_enough,
    get_middle_value,
    iterate,
    to_str,
)
from dascore.utils.models import (
    DascoreBaseModel,
    UnitQuantity,
    CommaSeparatedStr,
    frozen_dict_validator,
    frozen_dict_serializer,
)

str_validator = PlainValidator(to_str)
_coord_summary_suffixes = set(CoordSummary.model_fields)


def _get_coords_dict(data_dict, fields):
    """
    Add coords dict to data dict, pop out any coordinate attributes.

    For example, if time_min, time_step are in data_dict, these will be
    grouped into the coords sub dict under "time".
    """

    def _get_coord_dict(data_dict):
        """Get a dict for stuffing outputs in."""
        # first handle coords.
        coords = data_dict.get("coords", {})
        if isinstance(coords, CoordManager):
            data_dict["dims"] = coords.dims
            coords = coords.to_summary_dict()
        coords = dict(coords)
        return coords

    def _ensure_all_coord_summaries(coords):
        """Enssure each coordinate entry is a CoordSummary."""
        # then iterate and convert each coordinate
        for i, v in coords.items():
            if isinstance(v, BaseCoord):
                coords[i] = v.to_summary()
            # converts normal dicts to coord summary
            elif not isinstance(v, CoordSummary):
                coords[i] = CoordSummary(**v)
        return coords

    def _get_coords_from_attrs(data_dict, fields, coords):
        """Use fields in attrs to get coord dict."""
        # finally, pull out any attributes on the top level
        extra_fields = set(data_dict) - set(fields)
        new_coords = defaultdict(dict)
        for extra_field in extra_fields:
            if len(split := extra_field.split("_")) != 2:
                continue
            first, second = split
            # handle d_{coord_name}, it should now be {coord_name}_step
            if first == "d":
                first, second = second, "step"
                msg = f"{extra_field} is deprecated, use {first}_{second}"
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
            # if coords were passed we don't want attrs to overwrite them.
            if second in _coord_summary_suffixes:
                # we need to pop out the key either way, but only use
                # it if we haven't already defined the coord summary
                val = data_dict.pop(extra_field)
                if first not in coords:
                    new_coords[first][second] = val
        # now convert new values to Summaries
        for i, v in new_coords.items():
            new_coords[i] = CoordSummary(**v)
        return new_coords

    coords = _get_coord_dict(data_dict)
    coords_with_summaries = _ensure_all_coord_summaries(coords)
    new_coords = _get_coords_from_attrs(data_dict, fields, coords_with_summaries)
    # update coords and return
    coords.update(new_coords)
    data_dict["coords"] = coords
    return data_dict


@compose_docstring(basic_params=basic_summary_attrs)
class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    Parameter
    ---------
    {basic_params}

    Notes
    -----
    These attributes go into the HDF5 index used by dascore. Therefore,
    when they are changed the index version needs to be incremented so
    previous indices are invalidated.
    """

    model_config = ConfigDict(
        title="Patch Summary",
        extra="allow",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = ""
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = ""
    data_units: UnitQuantity | None = None

    instrument_id: str = Field("", max_length=max_lens["instrument_id"])
    cable_id: str = Field("", max_length=max_lens["cable_id"])

    tag: str = Field("", max_length=max_lens["tag"])
    station: str = Field("", max_length=max_lens["station"])
    network: str = Field("", max_length=max_lens["network"])
    history: str | Sequence[str] = Field(default_factory=list)

    dims: CommaSeparatedStr = Field("", max_length=max_lens["dims"])

    coords: Annotated[
        FrozenDict[str, CoordSummary],
        frozen_dict_validator,
        frozen_dict_serializer,
    ] = {}

    @model_validator(mode="before")  # noqa
    @classmethod
    def parse_coord_attributes(cls, data: Any) -> Any:
        """
        Parse the coordinate attributes into coord dict.
        """
        if isinstance(data, dict):
            data = _get_coords_dict(data, cls.model_fields)
        return data

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(self.model_dump())

    def __getattr__(self, item):
        """
        This enables dynamic attributes such as time_min, time_max, etc.
        """
        split = item.split("_")
        # this only works on names like time_max, distance_step, etc.
        if not len(split) == 2:
            return super().__getattr__(item)
        first, second = split
        if first == "d":
            first, second = second, "step"
            msg = f"{item} is depreciated, use {first}_{second} instead."
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
        if first not in self.coords:
            return super().__getattr__(item)
        coord_sum = self.coords[first]
        return getattr(coord_sum, second)

    def get(self, item, default=None):
        """dict-like get method."""
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default

    def items(self):
        """Yield (attribute, values) just like dict.items()."""
        yield from self.model_dump().items()

    def coords_from_dims(self) -> Mapping[str, BaseCoord]:
        """Return coordinates from dimensions assuming evenly sampled."""
        out = {}
        for dim in self.dim_tuple:
            start, stop = self[f"{dim}_min"], self[f"{dim}_max"]
            step = self[f"d_{dim}"]
            out[dim] = CoordRange(start=start, stop=stop + step, step=step)
        return out

    @classmethod
    def from_dict(
        cls,
        attr_map: Mapping | PatchAttrs,
        coord_manager: dc.core.coordmanager.CoordManager | None = None,
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Optionally, give preference to data contained in a
        [`CoordManager`](`dascore.core.coordmanager.CoordManager`).

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains the attr info.
        coord_manager
            A coordinate manager to fill in/overwrite attributes.

        """
        if isinstance(attr_map, cls) and coord_manager is None:
            return attr_map
        out = {} if attr_map is None else dict(attr_map)
        if coord_manager is None:
            return cls(**out)
        out["dims"] = ",".join(coord_manager.dims)
        out["coords"] = coord_manager.to_summary_dict()
        return cls(**out)

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions. The dims attr is a string."""
        return tuple(self.dims.split(","))

    def rename_dimension(self, **kwargs):
        """
        Rename one or more dimensions if in kwargs. Return new PatchAttrs.
        """
        if not (dims := set(kwargs) & set(self.dim_tuple)):
            return self
        new = dict(self)
        new_dims = list(self.dim_tuple)
        for old_name, new_name in {x: kwargs[x] for x in dims}.items():
            new_dims[new_dims.index(old_name)] = new_name
        new["dims"] = tuple(new_dims)
        return self.__class__(**new)

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        out = dict(self)
        out.update(kwargs)
        return self.__class__(**out)

    def drop_private(self) -> Self:
        """Drop all private attributes."""
        contents = dict(self)
        out = {i: v for i, v in contents.items() if not i.startswith("_")}
        return self.__class__(**out)

    def flat_dump(self) -> dict:
        """
        Flatten the coordinates and dump to dict
        """
        out = self.model_dump()
        for coord_name, coord in out.pop("coords").items():
            for name, val in coord.items():
                out[f"{coord_name}_{name}"] = val
        return out


def combine_patch_attrs(
    model_list: Sequence[PatchAttrs],
    coord_name: str | None = None,
    conflicts: Literal["drop", "raise", "keep_first"] = "raise",
    drop_attrs: Sequence[str] | None = None,
    coord: None | BaseCoord = None,
):
    """
    Merge Patch Attributes along a dimension.

    Parameters
    ----------
    model_list
        A list of models.
    coord_name
        The coordinate, usually a dimension coord, along which to merge.
    conflicts
        Indicates how to handle conflicts in attributes other than those
        indicated by dim. If "drop" simply drop conflicting attributes,
        or attributes not shared by all models. If "raise" raise an
        [AttributeMergeError](`dascore.exceptions.AttributeMergeError`] when
        issues are encountered. If "keep_first", just keep the first value
        for each attribute.
    drop_attrs
        If provided, attributes which should be dropped.
    coord
        The coordinate for the new values of dim. This is provided as a
        shortcut if it has already been computed.
    """
    # TODO this is a monstrosity! need to refactor.
    eq_coord_fields = set(CoordSummary.model_fields) - {"min", "max", "step"}

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
        """Get a list of {attrs: []}"""
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
        # make sure at least min/max is defined.
        if not len(merge_dict_list["min"]) == len(merge_dict_list["max"]):
            msg = "Cant merge attributes, min and max must be defined for coord."
            raise AttributeMergeError(msg)
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
        if all_diffs_close_enough((steps := merge_dict_list["step"])):
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
        if not coord_name:
            return model_dicts
        # coordinate can be determined from existing coords.
        if coord is None:
            new_coord = _get_new_coord(model_dicts)
        else:  # or if one was passed just use its summary.
            new_coord = coord.to_summary()
        if new_coord:
            for mod in model_dicts:
                mod["coords"][coord_name] = new_coord
        return model_dicts

    def _replace_null_with_None(mod_dict_list):
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
        no_null_ = _replace_null_with_None(mod_dict_list)
        all_eq = all(no_null_[0] == x for x in no_null_)
        if all_eq:
            return mod_dict_list
        # now the fun part.
        if conflicts == "raise":
            msg = "Cannot merge models, not all of their non-dim attrs are equal."
            raise AttributeMergeError(msg)
        final_dict = reduce(_keep_eq, mod_dict_list)
        return [final_dict]

    mod_dict_list = _get_model_dict_list(model_list)
    mod_dict_list = _handle_other_attrs(mod_dict_list)
    first_class = model_list[0].__class__
    cls = first_class if not first_class == dict else PatchAttrs
    return cls(**mod_dict_list[0])


def merge_compatible_coords_attrs(
    patch1: PatchType, patch2: PatchType, attrs_to_ignore=("history",)
) -> tuple[CoordManager, PatchAttrs]:
    """
    Merge the coordinates and attributes of patches or raise if incompatible.

    The rules for compatibility are:
        - All attrs must be equal other than history.
        - Patches must share the same dimensions, in the same order
        - All dimensional coordinates must be strictly equal
        - If patches share a non-dimensional coordinate they must be equal.
    Any coordinates or attributes contained by a single patch will be included
    in the output.

    Parameters
    ----------
    patch1
        The first patch
    patch2
        The second patch
    attr_ignore
        A sequence of attributes to not consider in equality. Only these
        attributes from the first patch are kept in outputs.
    """

    def _check_dims(dims1, dims2):
        if dims1 == dims2:
            return
        msg = (
            "Patches are not compatible because their dimensions are not equal."
            f" Patch1 dims: {dims1}, Patch2 dims: {dims2}"
        )
        raise IncompatiblePatchError(msg)

    def _check_coords(cm1, cm2):
        cset1, cset2 = set(cm1.coord_map), set(cm2.coord_map)
        shared = cset1 & cset2
        not_equal_coords = []
        for coord in shared:
            coord1 = cm1.coord_map[coord]
            coord2 = cm2.coord_map[coord]
            if coord1 == coord2:
                continue
            not_equal_coords.append(coord)
        if not_equal_coords:
            msg = (
                f"Patches are not compatible. The following shared coordinates "
                f"are not equal {coord}"
            )
            raise IncompatiblePatchError(msg)

    def _merge_coords(coords1, coords2):
        out = {}
        coord_names = set(coords1.coord_map) & set(coords2.coord_map)
        # fast patch to update identical coordinates
        if len(coord_names) == len(coords1.coord_map):
            return coords1
        # otherwise just squish coords from both managers together.
        for name in coord_names:
            coord = coords1 if name in coords1.coord_map else coords2
            dims = coord.dim_map[name]
            out[name] = (dims, coord.coord_map[name])
        return dc.core.coordmanager.get_coord_manager(out, dims=coords1.dims)

    def _merge_models(attrs1, attrs2, coord):
        """Ensure models are equal in the right ways."""
        no_comp_keys = set(attrs_to_ignore)
        if attrs1 == attrs2:
            return attrs1
        dict1, dict2 = dict(attrs1), dict(attrs2)
        # Coords has already handled merging coordinates
        new_coords = coord.to_summary_dict()
        dict1["coords"], dict2["coords"] = new_coords, new_coords
        common_keys = set(dict1) & set(dict2)
        ne_attrs = []
        for key in common_keys:
            if key in no_comp_keys:
                continue
            if dict2[key] != dict1[key]:
                ne_attrs.append(key)
        if ne_attrs:
            msg = (
                "Patches are not compatible because the following attributes "
                f"are not equal. {ne_attrs}"
            )
            raise IncompatiblePatchError(msg)
        return combine_patch_attrs([dict1, dict2], conflicts="keep_first")

    _check_dims(patch1.dims, patch2.dims)
    coord1, coord2 = patch1.coords, patch2.coords
    attrs1, attrs2 = patch1.attrs, patch2.attrs
    _check_coords(coord1, coord2)
    coord_out = _merge_coords(coord1, coord2)
    attrs = _merge_models(attrs1, attrs2, coord_out)
    return coord_out, attrs
