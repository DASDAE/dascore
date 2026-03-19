"""Summary models for metadata-only patch workflows."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ConfigDict, Field, model_validator

import dascore as dc
from dascore.core.attrs import PatchAttrs
from dascore.core.coords import CoordSummary
from dascore.utils.attrs import separate_coord_info
from dascore.utils.models import DascoreBaseModel


def _to_coord_summary(value: Any, dims: tuple[str, ...] = ()) -> CoordSummary:
    """Normalize a coordinate summary input."""
    # Summary inputs can already be normalized, coord-like objects, or plain
    # mappings produced by scan/index code.
    if isinstance(value, CoordSummary):
        return value
    if hasattr(value, "to_summary"):
        value = value.to_summary(dims=dims)
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if isinstance(value, Mapping):
        value = dict(value)
        if dims and "dims" not in value:
            value["dims"] = dims
    return CoordSummary(**value)


def _infer_dims_from_coords(coords: Mapping[str, CoordSummary]) -> tuple[str, ...]:
    """Infer patch dims from summarized coordinates when not provided."""
    out: list[str] = []
    for name, summary in coords.items():
        dims = tuple(getattr(summary, "dims", (name,)))
        if dims == (name,):
            out.append(name)
    return tuple(x for x in out if x)


def _normalize_dims(dims: Any) -> tuple[str, ...]:
    """Normalize dims input from flat or structured summary payloads."""
    if dims in (None, ""):
        return tuple()
    if isinstance(dims, str):
        return tuple(dims.split(","))
    return tuple(dims)


class PatchSummary(DascoreBaseModel):
    """A metadata-only summary of a patch."""

    model_config = ConfigDict(title="Patch Summary", extra="ignore", frozen=True)

    attrs: PatchAttrs = Field(default_factory=PatchAttrs)
    coords: dict[str, CoordSummary] = Field(default_factory=dict)
    dims: tuple[str, ...] = ()
    shape: tuple[int, ...] = ()
    dtype: str = ""
    path: str | Path = ""
    file_format: str = ""
    file_version: str = ""
    source_patch_id: str = ""

    @model_validator(mode="before")
    @classmethod
    def _normalize_input(cls, data: Any) -> Any:
        """Accept flat scan metadata or structured summary input."""
        if isinstance(data, dc.Patch):
            return cls.from_patch(data).dump_structured()
        # Let pydantic raise the normal validation error for unsupported inputs.
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        # Structured inputs already separate non-coordinate attrs from coordinate
        # summaries, so we only need to normalize nested coord values and dims.
        if "attrs" in data or "coords" in data:
            attrs = PatchAttrs.from_dict(data.get("attrs"))
            dims = _normalize_dims(data.get("dims", ()))
            coord_map = {
                name: _to_coord_summary(
                    summary,
                    dims=tuple(summary.get("dims", (name,)))
                    if isinstance(summary, Mapping)
                    else (),
                )
                for name, summary in data.get("coords", {}).items()
            }
            dims = dims or _infer_dims_from_coords(coord_map)
            return {
                "attrs": attrs,
                "coords": coord_map,
                "dims": dims,
                "shape": tuple(data.get("shape", ())),
                "dtype": str(data.get("dtype", "")),
                "path": data.get("path", ""),
                "file_format": data.get("file_format", ""),
                "file_version": data.get("file_version", ""),
                "source_patch_id": data.get("source_patch_id", ""),
            }
        # Flat inputs still use scan/index-style keys such as time_min/time_max.
        # Split those into coordinate summaries plus pure attrs before building
        # the canonical structured payload.
        dims = _normalize_dims(data.get("dims", ""))
        coord_info, attr_info = separate_coord_info(
            {k: v for k, v in data.items() if k != "dims"},
            dims=dims or None,
        )
        attrs = PatchAttrs.from_dict(attr_info)
        coord_map = {
            name: _to_coord_summary(
                summary,
                dims=tuple(summary.get("dims", (name,)))
                if isinstance(summary, Mapping)
                else (),
            )
            for name, summary in coord_info.items()
        }
        dims = dims or _infer_dims_from_coords(coord_map)
        return {
            "attrs": attrs,
            "coords": coord_map,
            "dims": dims,
            "shape": tuple(data.get("shape", ())),
            "dtype": str(data.get("dtype", "")),
            "path": data.get("path", ""),
            "file_format": data.get("file_format", ""),
            "file_version": data.get("file_version", ""),
            "source_patch_id": data.get("source_patch_id", ""),
        }

    @classmethod
    def from_patch(cls, patch: dc.Patch) -> PatchSummary:
        """Create a summary from a loaded patch."""
        return cls.model_construct(
            attrs=patch.attrs,
            coords=patch.coords.to_summary_dict(),
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(np.dtype(patch.data.dtype)),
        )

    def dump_structured(self) -> dict[str, Any]:
        """Return the structured representation of the summary."""
        return super().model_dump()

    def flat_dump(self, dim_tuple: bool = False, exclude=None) -> dict[str, Any]:
        """Return a flat dict suitable for indexing/dataframes."""
        exclude = set(() if exclude is None else exclude)
        attrs = self.attrs
        out = {
            name: getattr(attrs, name)
            for name in type(attrs).model_fields
            if name not in exclude
        }
        extra = getattr(attrs, "__pydantic_extra__", None) or {}
        for name, value in extra.items():
            if name not in exclude:
                out[name] = value
        summary_meta = {
            "dims": ",".join(self.dims),
            "dtype": self.dtype,
            "path": self.path,
            "file_format": self.file_format,
            "file_version": self.file_version,
            "source_patch_id": self.source_patch_id,
            "coord_names": ",".join(self.coords),
        }
        out.update(
            {name: value for name, value in summary_meta.items() if name not in exclude}
        )
        for coord_name, summary in self.coords.items():
            if coord_name in exclude:
                continue
            summary_dict = {}
            for field in type(summary).model_fields:
                if field == "dims":
                    summary_dict[field] = getattr(summary, field)
                else:
                    summary_dict[field] = getattr(summary, field)
            # Some callers still want {(min, max)} tuples for dimensional coords
            # instead of the fully expanded flat summary columns.
            if dim_tuple:
                out[coord_name] = (summary_dict["min"], summary_dict["max"])
            for field, value in summary_dict.items():
                if field in exclude:
                    continue
                if field == "dims":
                    value = ",".join(value) if value else ""
                # Flat summaries historically stored a sentinel step value even
                # when a coord is not evenly sampled. Preserve that behavior for
                # dataframe/index consumers that expect a scalar in *_step.
                if field == "step" and value is None:
                    start = summary_dict["min"]
                    if isinstance(start, np.datetime64 | np.timedelta64):
                        value = np.timedelta64("NaT")
                    elif isinstance(start, float | np.floating):
                        value = np.nan
                out[f"{coord_name}_{field}"] = value
        return out

    def __getitem__(self, item):
        try:
            return getattr(self, item)
        except AttributeError as exc:
            raise KeyError(item) from exc

    def __iter__(self):
        return iter(self.flat_dump())

    def __len__(self):
        return len(self.flat_dump())

    def __getattr__(self, item):
        # Keep summary access ergonomic during the attrs/coords split by falling
        # back to non-coordinate attrs first, then flattened coord fields.
        try:
            return getattr(self.attrs, item)
        except AttributeError:
            split = item.rsplit("_", 1)
            if len(split) == 2:
                coord_name, field = split
                if coord_name in self.coords:
                    coord = self.coords[coord_name]
                    if field in type(coord).model_fields:
                        return getattr(coord, field)
            msg = f"{self.__class__.__name__} has no attribute '{item}'"
            raise AttributeError(msg) from None

    def get(self, item, default=None):
        """Return item if present else default."""
        try:
            return self[item]
        except KeyError:
            return default

    def items(self):
        """Yield summary items like a mapping."""
        yield from self.flat_dump().items()

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

    def get_coord(self, coord_name: str) -> CoordSummary:
        """Return the coordinate summary for compatibility with scan workflows."""
        return self.get_coord_summary(coord_name)
