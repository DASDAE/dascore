"""Summary models for patch workflows.

See ['Coordinate Internals'](`docs/notes/coordinate_internals.qmd`) for the
relationship between full coords, exact coord summaries, and flattened index metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field, model_validator

import dascore as dc
from dascore.constants import path_types
from dascore.core.attrs import PatchAttrs
from dascore.core.coords import BaseCoord, CoordSummary, get_coord
from dascore.utils.models import DascoreBaseModel
from dascore.utils.paths import coerce_to_upath, is_pathlike


def _to_coord_summary(value: Any, dims: tuple[str, ...] = ()) -> CoordSummary:
    """Normalize a coordinate summary input."""
    # Summary inputs can already be normalized, coord-like objects, or exact
    # structured summary mappings persisted from a full coord.
    if isinstance(value, CoordSummary):
        return value
    if hasattr(value, "to_summary"):
        value = value.to_summary(dims=dims)
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Mapping):
        value = dict(value)
        if "dims" in value:
            value["dims"] = _normalize_dims(value["dims"])
        if dims and "dims" not in value:
            value["dims"] = dims
    return CoordSummary(**value)


def coord_summary_from_data(
    data: BaseCoord | np.ndarray | Any,
    *,
    dims: tuple[str, ...] = (),
    units=None,
    step=None,
    dtype=None,
) -> CoordSummary:
    """Create a CoordSummary from raw data or an existing coordinate."""
    if isinstance(data, BaseCoord):
        coord = data
    else:
        coord = get_coord(data=data, units=units, step=step, dtype=dtype)
    summary = coord.to_summary(dims=dims)
    if len(coord) == 0:
        summary = CoordSummary(
            min=summary.min,
            max=summary.max,
            step=summary.step,
            dtype=summary.dtype,
            units=units if units is not None else summary.units,
            dims=summary.dims,
            len=0,
        )
    return summary


def _normalize_coord_summary_dtype(
    dtype=None,
    *,
    is_datetime: bool = False,
    is_timedelta: bool = False,
    is_string: bool = False,
    original_dtype: str = "",
) -> str:
    """Normalize external dtype metadata to the CoordSummary dtype string."""
    if is_datetime:
        return "datetime64"
    if is_timedelta:
        return "timedelta64"
    if is_string and original_dtype:
        return str(np.dtype(original_dtype))
    if dtype not in (None, ""):
        return str(np.dtype(dtype))
    return ""


def _coord_summary_to_dict(summary: CoordSummary) -> dict[str, Any]:
    """Return the normalized dict form of a coord summary."""
    return {field: getattr(summary, field) for field in type(summary).model_fields}


def _get_null_step_value(summary: CoordSummary):
    """Return the historical flat-dump sentinel for missing step values."""
    start = summary.min
    if isinstance(start, np.datetime64 | np.timedelta64):
        return np.timedelta64("NaT")
    if isinstance(start, float | np.floating):
        return np.nan
    return None


def _flatten_coord_summary(
    coord_name: str,
    summary: CoordSummary,
    *,
    dim_tuple: bool = False,
    exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Flatten a single coord summary into scan/index-style fields."""
    exclude = set() if exclude is None else exclude
    summary_dict = _coord_summary_to_dict(summary)
    out = {}
    if dim_tuple and coord_name not in exclude:
        out[coord_name] = (summary_dict["min"], summary_dict["max"])
    for field, value in summary_dict.items():
        if field in exclude:
            continue
        if field == "dims":
            value = ",".join(value) if value else ""
        elif field == "step" and value is None:
            value = _get_null_step_value(summary)
        out[f"{coord_name}_{field}"] = value
    return out


def _infer_dims_from_coords(coords: Mapping[str, CoordSummary]) -> tuple[str, ...]:
    """Infer patch dims from summarized coordinates when not provided."""
    out: list[str] = []
    for name, summary in coords.items():
        dims = tuple(getattr(summary, "dims", (name,)))
        if dims == (name,):
            out.append(name)
    return tuple(x for x in out if x)


def _normalize_dims(dims: Any) -> tuple[str, ...]:
    """Normalize dims to a tuple.

    Handle comma-separated strings from serialized data.
    """
    if dims in (None, ""):
        return tuple()
    if isinstance(dims, str):
        return tuple(dims.split(","))
    return tuple(dims)


def _normalize_coord_summary_map(
    coords: Mapping[str, Any],
) -> dict[str, CoordSummary]:
    """Normalize a mapping of coord-like values to CoordSummary objects."""
    return {
        name: _to_coord_summary(
            summary,
            dims=_normalize_dims(summary.get("dims", (name,)))
            if isinstance(summary, Mapping)
            else (),
        )
        for name, summary in coords.items()
    }


def _normalize_source_patch_id(
    attrs: PatchAttrs, source_patch_id: Any = ""
) -> tuple[PatchAttrs, str]:
    """Normalize summary and private attr source ids to one value."""
    summary_source_patch_id = (
        "" if source_patch_id in (None, "") else str(source_patch_id)
    )
    attrs_source_patch_id = str(attrs.get("_source_patch_id", "") or "")
    normalized = summary_source_patch_id or attrs_source_patch_id
    if normalized:
        attrs = attrs.update(_source_patch_id=normalized)
    return attrs, normalized


def _build_patch_summary_payload(
    *,
    attrs: PatchAttrs,
    coords: dict[str, CoordSummary],
    dims: tuple[str, ...] = (),
    shape=(),
    dtype="",
    source_path="",
    source_format="",
    source_version="",
    source_patch_id="",
) -> dict[str, Any]:
    """Build the canonical structured payload used to validate PatchSummary."""
    attrs, source_patch_id = _normalize_source_patch_id(attrs, source_patch_id)
    dims = dims or _infer_dims_from_coords(coords)
    # Only preserve source metadata when the caller already supplied a cheap,
    # path-like reload target. Validation should not touch the filesystem.
    if source_path in (None, ""):
        normalized_source_path = ""
    elif is_pathlike(source_path):
        normalized_source_path = coerce_to_upath(source_path)
    else:
        normalized_source_path = ""
    normalized_source_format = "" if source_format in (None, "") else str(source_format)
    normalized_source_version = (
        "" if source_version in (None, "") else str(source_version)
    )
    # If the source path is not reloadable, drop the paired source metadata too
    # so summary objects only expose coherent reload information.
    if not normalized_source_path:
        normalized_source_format = ""
        normalized_source_version = ""
    return {
        "attrs": attrs,
        "coords": coords,
        "dims": dims,
        "shape": tuple(shape),
        "dtype": str(dtype),
        "source_path": normalized_source_path,
        "source_format": normalized_source_format,
        "source_version": normalized_source_version,
        "source_patch_id": source_patch_id,
    }


class PatchSummary(DascoreBaseModel):
    """A metadata-only summary of a patch."""

    model_config = ConfigDict(title="Patch Summary", extra="ignore", frozen=True)

    attrs: PatchAttrs = Field(default_factory=PatchAttrs)
    coords: dict[str, CoordSummary] = Field(default_factory=dict)
    dims: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    dtype: str = ""
    source_path: path_types = ""
    source_format: str = ""
    source_version: str = ""
    source_patch_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        """Accept structured summary input."""
        if isinstance(data, dc.Patch):
            return cls.from_patch(data).dump_structured()
        # Let pydantic raise the normal validation error for unsupported inputs.
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        # Structured inputs already separate non-coordinate attrs from coordinate
        # summaries, so we only need to normalize nested coord values and dims.
        if "attrs" in data or "coords" in data:
            return _build_patch_summary_payload(
                attrs=PatchAttrs.from_dict(data.get("attrs")),
                coords=_normalize_coord_summary_map(data.get("coords", {})),
                dims=_normalize_dims(data.get("dims", ())),
                shape=data.get("shape", ()),
                dtype=data.get("dtype", ""),
                source_path=data.get("source_path", data.get("path", "")),
                source_format=data.get("source_format", data.get("file_format", "")),
                source_version=data.get("source_version", data.get("file_version", "")),
                source_patch_id=data.get("source_patch_id", ""),
            )
        msg = (
            "PatchSummary requires structured `attrs`/`coords` input. "
            "Flat coord summary fields are no longer supported."
        )
        raise TypeError(msg)

    @classmethod
    def from_patch(cls, patch: dc.Patch) -> PatchSummary:
        """Create a summary from a loaded patch."""
        return cls(
            attrs=patch.attrs,
            coords=patch.coords.to_summary_dict(),
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(np.dtype(patch.data.dtype)),
            source_patch_id=patch.attrs.get("_source_patch_id", ""),
        )

    def dump_structured(self) -> dict[str, Any]:
        """Return the structured representation of the summary."""
        return super().model_dump()

    def flat_dump(self, dim_tuple: bool = False, exclude=None) -> dict[str, Any]:
        """Return a flat dict suitable for indexing/dataframes."""
        exclude = set(() if exclude is None else exclude)
        # Build flattened attrs first, then overlay coord summaries so coord-
        # derived fields win over any attr using the same simplified key.
        out = self.attrs.flat_dump(exclude=exclude)
        summary_meta = {
            "dims": ",".join(self.dims),
            "dtype": self.dtype,
            # TODO: Replace these to source_path, source_format, and source_version
            # when redoing the indexer.
            "path": str(self.source_path) if self.source_path else "",
            "file_format": self.source_format,
            "file_version": self.source_version,
            "source_patch_id": self.source_patch_id,
            "coord_names": ",".join(self.coords),
        }
        out.update(
            {name: value for name, value in summary_meta.items() if name not in exclude}
        )
        for coord_name, summary in self.coords.items():
            if coord_name in exclude:
                continue
            out.update(
                _flatten_coord_summary(
                    coord_name,
                    summary,
                    dim_tuple=dim_tuple,
                    exclude=exclude,
                )
            )
        return out

    @property
    def dim_tuple(self) -> tuple[str, ...]:
        """Return the dimensions as a tuple."""
        return self.dims

    @property
    def summary(self) -> PatchSummary:
        """Return self for symmetry with Patch.summary."""
        return self

    def get_coord_summary(self, coord_name: str) -> CoordSummary:
        """Return the summary object for a single coordinate."""
        return self.coords[coord_name]
