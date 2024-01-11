"""Pydantic schemas used by DASCore."""
from __future__ import annotations

import warnings
from collections import ChainMap, defaultdict
from collections.abc import Mapping, Sequence
from functools import reduce
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, PlainValidator, model_validator
from typing_extensions import Self

import dascore as dc
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    WARN_LEVELS,
    PatchType,
    attr_conflict_description,
    max_lens,
)
from dascore.core.coordmanager import CoordManager
from dascore.core.coords import BaseCoord, CoordSummary
from dascore.exceptions import AttributeMergeError, IncompatiblePatchError
from dascore.utils.docs import compose_docstring
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    _dict_list_diffs,
    all_diffs_close_enough,
    get_middle_value,
    iterate,
    separate_coord_info,
    to_str,
    warn_or_raise,
)
from dascore.utils.models import (
    CommaSeparatedStr,
    DascoreBaseModel,
    UnitQuantity,
    frozen_dict_serializer,
    frozen_dict_validator,
)

str_validator = PlainValidator(to_str)
_coord_summary_suffixes = set(CoordSummary.model_fields)
_coord_required = {"min", "max"}


def _get_coords_dict(data_dict, fields):
    """
    Add coords dict to data dict, pop out any coordinate attributes.

    For example, if time_min, time_step are in data_dict, these will be
    grouped into the coords sub dict under "time".
    """

    def _get_dims(data_dict):
        """Try to get dim tuple."""
        dims = None
        if "dims" in data_dict:
            dims = data_dict["dims"]
        elif hasattr(coord := data_dict.get("coords"), "dims"):
            dims = coord.dims
        if isinstance(dims, str):
            dims = tuple(dims.split(","))
        return dims

    dims = _get_dims(data_dict)
    coord_info, new_attrs = separate_coord_info(
        data_dict, dims, required=("min", "max")
    )
    if "dims" not in new_attrs and dims is not None:
        new_attrs["dims"] = dims

    new_attrs["coords"] = {i: dc.core.CoordSummary(**v) for i, v in coord_info.items()}
    return new_attrs


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    The default attributes are:
    ```{python}
    #| echo: false

    import dascore as dc
    from IPython.display import Markdown

    df_str = (
        dc.PatchAttrs.get_summary_df()
        .reset_index()
        .to_markdown(index=False, stralign="center")
    )
    Markdown(df_str)
    ```
    """

    model_config = ConfigDict(
        title="Patch Summary",
        extra="allow",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = Field(
        description="Describes the quantity being measured.", default=""
    )
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = Field(
        description="Describes the type of data.",
        default="",
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="The units of the data measurements"
    )
    instrument_id: str = Field(
        description="A unique id for the instrument which generated the data.",
        default="",
        max_length=max_lens["instrument_id"],
    )
    acquisition_id: str = Field(
        description="A unique identifier linking this data to an experiment.",
        default="",
        max_length=max_lens["experiment_id"],
    )
    tag: str = Field(
        default="", max_length=max_lens["tag"], description="A custom string field."
    )
    station: str = Field(
        default="", max_length=max_lens["station"], description="A station code."
    )
    network: str = Field(
        default="", max_length=max_lens["network"], description="A network code."
    )
    history: str | Sequence[str] = Field(
        default_factory=list, description="A list of processing performed on the patch."
    )

    dims: CommaSeparatedStr = Field(
        default="",
        max_length=max_lens["dims"],
        description="A tuple of comma-separated dimensions names.",
    )

    coords: Annotated[
        FrozenDict[str, CoordSummary],
        frozen_dict_validator,
        frozen_dict_serializer,
    ] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def parse_coord_attributes(cls, data: Any) -> Any:
        """Parse the coordinate attributes into coord dict."""
        if isinstance(data, dict):
            data = _get_coords_dict(data, cls.model_fields)
            # add dims as coords if dims is not included.
            if "dims" not in data:
                data["dims"] = tuple(data["coords"])
        return data

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(self.model_dump())

    def __getattr__(self, item):
        """Enables dynamic attributes such as time_min, time_max, etc."""
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
            out[dim] = self.coords[dim].to_coord()
        return out

    @classmethod
    def from_dict(
        cls,
        attr_map: Mapping | PatchAttrs,
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Optionally, give preference to data contained in a
        [`CoordManager`](`dascore.core.coordmanager.CoordManager`).

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains the attr info.
        """
        if isinstance(attr_map, cls):
            return attr_map
        out = {} if attr_map is None else attr_map
        return cls(**out)

    @property
    def dim_tuple(self):
        """Return a tuple of dimensions. The dims attr is a string."""
        return tuple(self.dims.split(","))

    def rename_dimension(self, **kwargs):
        """Rename one or more dimensions if in kwargs. Return new PatchAttrs."""
        if not (dims := set(kwargs) & set(self.dim_tuple)):
            return self
        new = self.model_dump(exclude_defaults=True)
        coords = new.get("coords", {})
        new_dims = list(self.dim_tuple)
        for old_name, new_name in {x: kwargs[x] for x in dims}.items():
            new_dims[new_dims.index(old_name)] = new_name
            coords[new_name] = coords.pop(old_name, None)
        new["dims"] = tuple(new_dims)
        return self.__class__(**new)

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        coord_info, attr_info = separate_coord_info(kwargs, dims=self.dim_tuple)
        out = self.model_dump(exclude_unset=True)
        out.update(attr_info)
        out_coord_dict = out["coords"]
        for name, coord_dict in coord_info.items():
            if name not in out_coord_dict:
                out_coord_dict[name] = coord_dict
            else:
                out_coord_dict[name].update(coord_dict)
        # silly check to clear coords
        if not kwargs.get("coords", True):
            out["coords"] = {}
        return self.__class__(**out)

    def drop_private(self) -> Self:
        """Drop all private attributes."""
        contents = dict(self)
        out = {i: v for i, v in contents.items() if not i.startswith("_")}
        return self.__class__(**out)

    def flat_dump(self, dim_tuple=False, exclude=None) -> dict:
        """
        Flatten the coordinates and dump to dict.

        Parameters
        ----------
        dim_tuple
            If True, return dimensional tuple instead of range. EG, the
            output will have {time: (min, max)} rather than
            {time_min: ..., time_max: ...,}. This is useful because it can
            be passed to read, scan, select, etc.
        exclude
            keys to exclude.
        """
        out = self.model_dump(exclude=exclude)
        for coord_name, coord in out.pop("coords").items():
            names = list(coord)
            if dim_tuple:
                names = sorted(set(names) - {"min", "max"})
                out[coord_name] = (coord["min"], coord["max"])
            for name in names:
                out[f"{coord_name}_{name}"] = coord[name]
            # ensure step has right type if nullish
            step_name = f"{coord_name}_step"
            step, start = out[step_name], coord["min"]
            if step is None:
                is_time = isinstance(start, np.datetime64 | np.timedelta64)
                if is_time:
                    out[step_name] = np.timedelta64("NaT")
                elif isinstance(start, float | np.floating):
                    out[step_name] = np.NaN
        return out


@compose_docstring(conflict_desc=attr_conflict_description)
def combine_patch_attrs(
    model_list: Sequence[PatchAttrs],
    coord_name: str | None = None,
    conflicts: Literal["drop", "raise", "keep_first"] = "raise",
    drop_attrs: Sequence[str] | None = None,
    coord: None | BaseCoord = None,
) -> PatchAttrs:
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
    cls = first_class if not first_class == dict else PatchAttrs
    if not len(mod_dict_list):
        msg = "Failed to combine patch attrs"
        raise AttributeMergeError(msg)
    return cls(**mod_dict_list[0])


def check_dims(patch1, patch2, check_behavior: WARN_LEVELS = "raise") -> bool:
    """
    Return True if dimensions of two patches are equal.

    Parameters
    ----------
    patch1
        first patch
    patch2
        second patch
    check_behavior
        String with 'raise' will raise an error if incompatible,
        'warn' will provide a warning, None will do nothing.
    """
    dims1 = patch1.dims
    dims2 = patch2.dims
    if dims1 == dims2:
        return True
    msg = (
        "Patches are not compatible because their dimensions are not equal."
        f" Patch1 dims: {dims1}, Patch2 dims: {dims2}"
    )
    warn_or_raise(msg, exception=IncompatiblePatchError, behavior=check_behavior)
    return False


def check_coords(
    patch1, patch2, check_behavior: WARN_LEVELS = "raise", dim_to_ignore=None
) -> bool:
    """
    Return True if the coordinates of two patches are compatible, else False.

    Parameters
    ----------
    patch1
        patch 1
    patch2
        patch 2
    check_behavior
        String with 'raise' will raise an error if incompatible,
        'warn' will provide a warning.
    dim_to_ignore
        None by default (all coordinates must be identical).
        String specifying a dimension that differences in values,
        but not shape, are allowed.
    """
    cm1 = patch1.coords
    cm2 = patch2.coords
    cset1, cset2 = set(cm1.coord_map), set(cm2.coord_map)
    shared = cset1 & cset2
    not_equal_coords = []
    for coord in shared:
        coord1 = cm1.coord_map[coord]
        coord2 = cm2.coord_map[coord]
        if coord1 == coord2:
            # Straightforward case, coords are identical.
            continue
        elif coord == dim_to_ignore:
            # If dimension that's ok to ignore value differences,
            # check whether shape is the same.
            if coord1.shape == coord2.shape:
                continue
            else:
                not_equal_coords.append(coord)
        else:
            not_equal_coords.append(coord)
    if not_equal_coords and len(shared):
        msg = (
            f"Patches are not compatible. The following shared coordinates "
            f"are not equal {coord}"
        )
        warn_or_raise(msg, exception=IncompatiblePatchError, behavior=check_behavior)
        return False
    return True


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

    check_dims(patch1, patch2)
    check_coords(patch1, patch2)
    coord1, coord2 = patch1.coords, patch2.coords
    attrs1, attrs2 = patch1.attrs, patch2.attrs
    coord_out = _merge_coords(coord1, coord2)
    attrs = _merge_models(attrs1, attrs2, coord_out)
    return coord_out, attrs


def decompose_attrs(attr_list: Sequence[PatchAttrs], exclude=("history",)):
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
