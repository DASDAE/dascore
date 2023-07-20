"""
Utilities for models.
"""
from collections import ChainMap
from functools import _lru_cache_wrapper, cached_property, reduce
from typing import Annotated, Optional, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, WrapValidator
from typing_extensions import Literal, Self

from dascore.compat import array
from dascore.exceptions import AttributeMergeError
from dascore.units import validate_quantity
from dascore.utils.misc import all_diffs_close_enough, get_middle_value, iterate
from dascore.utils.time import to_datetime64, to_timedelta64


class DascoreBaseModel(BaseModel):
    """A base model with sensible configurations."""

    model_config = ConfigDict(
        extra="ignore",
        validate_assignment=True,
        ignored_types=(cached_property, _lru_cache_wrapper),
        frozen=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def new(self, **kwargs) -> Self:
        """Create new instance with some attributed updated."""
        out = dict(self)
        for item, value in kwargs.items():
            out[item] = value
        return self.__class__(**out)

    def __hash__(self):
        return hash(id(self))


def call_validator(callable):
    """Return validator that simply calls a func on a single value."""

    @WrapValidator
    def _wraper(value, handler):
        return handler(callable(value))

    return _wraper


DateTime64 = Annotated[np.datetime64, call_validator(to_datetime64)]
TimeDelta64 = Annotated[np.timedelta64, call_validator(to_timedelta64)]
ArrayLike = Annotated[np.ndarray, call_validator(array)]
DTypeLike = Annotated[str, call_validator(np.dtype)]
UnitQuantity = Annotated[str | None, call_validator(validate_quantity)]


def merge_models(
    model_list: Sequence[BaseModel],
    dim: Optional[str] = None,
    conflicts: Literal["drop", "raise", "keep_first"] = "raise",
    drop_attrs: Optional[Sequence[str]] = None,
    coord=None,
):
    """
    Merge base models along a dimension.

    Parameters
    ----------
    model_list
        A list of models.
    dim
        The dimension along which to merge. If provided, the models must
        all have "{dim}_min" and "{dim}_max" attributes.
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
        The coordinate for the new values of dim.
    """

    def _check_class(mod_list):
        """Convert models to dictionaries."""
        if not len({x.__class__ for x in mod_list}) == 1:
            msg = "Models must all be the same class to merge"
            raise AttributeMergeError(msg)

    def _get_start_stop_step_from_values(model_dicts, model_sets):
        dmin, dmax, dstep = f"{dim}_min", f"{dim}_max", f"d_{dim}"
        expected_attrs = {dmin, dmax}
        if not all([x.issuperset(expected_attrs) for x in model_sets]):
            msg = f"All models do not have required attributes: {expected_attrs}"
            raise AttributeMergeError(msg)
        # now all model dicts just have min/max values set
        min_start = min([x[dmin] for x in model_dicts])
        max_end = max([x[dmax] for x in model_dicts])
        steps = [x[dstep] for x in model_dicts]
        step = None
        # if the steps are "close" we allow them to merge
        if all_diffs_close_enough(steps):
            step = get_middle_value(steps)
        return min_start, max_end, step

    def _get_model_dict_list(mod_list):
        """Get list of model_dict, merge along dim if specified."""
        model_dicts = [dict(x) for x in mod_list]
        # drop attributes specified.
        if drop := set(iterate(drop_attrs)):
            model_dicts = [
                {i: v for i, v in x.items() if i not in drop} for x in model_dicts
            ]
        model_sets = [set(x) for x in model_dicts]
        if not dim:
            return model_dicts
        if coord is None:
            min_start, max_end, step = _get_start_stop_step_from_values(
                model_dicts, model_sets
            )
        else:
            min_start, max_end = coord.min(), coord.max()
            step = coord.step
        for mod in model_dicts:
            mod[f"{dim}_min"], mod[f"{dim}_max"] = min_start, max_end
            mod[f"d_{dim}"] = np.NaN if step is None else step
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
        for i in set(d1) | set(d2):
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

    _check_class(model_list)
    _get_model_dict_list(model_list)
    mod_dict_list = _get_model_dict_list(model_list)
    mod_dict_list = _handle_other_attrs(mod_dict_list)
    cls = model_list[0].__class__
    return cls(**mod_dict_list[0])
