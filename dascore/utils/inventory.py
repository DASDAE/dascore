"""Utilities for inventory metadata models."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from pydantic import (
    ConfigDict,
    Field,
    computed_field,
)

from dascore.exceptions import ParameterError
from dascore.utils.models import DascoreBaseModel, DateTime64
from dascore.utils.time import to_datetime64

_IntervalTrackName = Literal[
    "optical_component",
    "geometry",
    "coupling_condition",
]


def _get_default_resource_id() -> str:
    """Return a default inventory resource id."""
    return f"inventory_{uuid4().hex}"


class _InventoryRecord(DascoreBaseModel):
    """Base class for immutable records stored in an Inventory."""

    model_config = ConfigDict(
        title="Inventory Record",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    resource_id: str = Field(
        default_factory=_get_default_resource_id,
        description=(
            "Stable identifier for the real-world or logical metadata resource."
        ),
    )
    schema_version: int = Field(
        default=1,
        description=(
            "Version of this record class schema; used for metadata model migration."
        ),
    )
    creation_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Time this metadata record was created.",
    )
    author_id: str = Field(
        default="",
        description="Identifier for the person or process creating the record.",
    )
    notes: str = Field(
        default="", description="Free-form notes describing this metadata record."
    )
    effective_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Time when this metadata state becomes effective.",
    )

    @computed_field
    @property
    def type(self) -> str:
        """Return the inventory type name."""
        return self.__class__.__name__.lstrip("_")

    def is_effective_at(self, time) -> bool:
        """Return True if the record is effective at the supplied time."""
        time = to_datetime64(time)
        if pd.isnull(time):
            return True
        return pd.isnull(self.effective_time) or self.effective_time <= time


def _record_from_mapping(
    data: Mapping[str, Any],
    record_types: Mapping[str, type[_InventoryRecord]],
) -> _InventoryRecord:
    """
    Validate one inventory manifest object against registered models.

    This is the dispatch path for plain mappings loaded from YAML or passed to
    ``Inventory(objects=[...])``. Manifest objects must explicitly include a
    non-empty ``type`` field and ``resource_id`` field. The type value is used
    to choose the matching Pydantic record class from ``record_types``; the
    manifest-only ``type`` key is then removed before validating the remaining
    fields into that class.

    Direct Python construction of inventory record classes does not use this
    helper, so constructors can still auto-generate ``resource_id`` values.
    """
    data = dict(data)
    if not data.get("resource_id"):
        msg = "Inventory manifest objects must include a non-empty resource_id."
        raise ValueError(msg)
    record_type = str(data.get("type", ""))
    if not record_type:
        msg = "Inventory manifest objects must include a type."
        raise ValueError(msg)
    try:
        cls = record_types[record_type]
    except KeyError:
        valid = sorted(record_types)
        msg = f"Unknown inventory object type {record_type!r}. Valid types are {valid}."
        raise ValueError(msg) from None
    data.pop("type", None)
    return cls.model_validate(data)


def _get_resource_id(record_or_id) -> str:
    """Return the resource id for an inventory record or id string."""
    if isinstance(record_or_id, str):
        return record_or_id
    return record_or_id.resource_id


def _coerce_geometry_table(table) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    """Return sorted geometry control points from a dataframe-like table."""
    if not isinstance(table, pd.DataFrame):
        msg = "geometry must be a pandas DataFrame."
        raise ParameterError(msg)
    if "distance" not in table.columns:
        msg = "geometry table must include a 'distance' column."
        raise ParameterError(msg)
    axis_columns = tuple(column for column in table.columns if column != "distance")
    if not axis_columns:
        msg = "geometry table must include at least one coordinate column."
        raise ParameterError(msg)
    non_numeric = tuple(
        column
        for column in ("distance", *axis_columns)
        if not pd.api.types.is_numeric_dtype(table[column])
    )
    if non_numeric:
        msg = "geometry table columns must be numeric: " + ", ".join(non_numeric)
        raise ParameterError(msg)
    data = table.loc[:, ("distance", *axis_columns)].sort_values("distance")
    distances = data["distance"].to_numpy(dtype=float)
    coordinates = data.loc[:, axis_columns].to_numpy(dtype=float)
    if len(distances) < 2:
        msg = "geometry table must include at least two control points."
        raise ParameterError(msg)
    if np.any(~np.isfinite(distances)) or np.any(~np.isfinite(coordinates)):
        msg = "geometry table distances and coordinates must be finite."
        raise ParameterError(msg)
    if np.any(distances < 0):
        msg = "geometry table distances must be non-negative."
        raise ParameterError(msg)
    if np.any(np.diff(distances) <= 0):
        msg = "geometry table distances must be unique and increasing."
        raise ParameterError(msg)
    return distances, axis_columns, coordinates


def _coerce_inventory_record(record, record_type, resource_id: str):
    """Return an inventory record from None, a mapping, or an existing record."""
    if record is None:
        return record_type(resource_id=resource_id)
    if isinstance(record, record_type):
        return record
    if isinstance(record, Mapping):
        data = dict(record)
        data.setdefault("resource_id", resource_id)
        return record_type.model_validate(data)
    msg = f"Expected {record_type.__name__}, mapping, or None."
    raise ParameterError(msg)


def _records_from_geometry_table(
    table,
    *,
    crs,
    optical_path,
    id_prefix: str,
    coordinate_reference_system_type,
    geometry_type,
    optical_path_type,
) -> tuple[_InventoryRecord, ...]:
    """Build CRS, Geometry records, and one OpticalPath from a geometry table."""
    distances, axis_columns, coordinates = _coerce_geometry_table(table)
    local_distances = distances - distances[0]
    crs_record = _coerce_inventory_record(
        crs,
        coordinate_reference_system_type,
        f"{id_prefix}:crs",
    )
    if not crs_record.axis_order:
        crs_record = crs_record.model_copy(update={"axis_order": axis_columns})
    geometry_records = []
    for index, (start, stop) in enumerate(pairwise(local_distances)):
        geometry_records.append(
            geometry_type(
                resource_id=f"{id_prefix}:geometry:{index:04d}",
                length=float(stop - start),
                geometry_type="linear",
                coordinate_reference_system_id=crs_record.resource_id,
                coordinates=(
                    tuple(float(value) for value in coordinates[index]),
                    tuple(float(value) for value in coordinates[index + 1]),
                ),
            )
        )
    path_record = _coerce_inventory_record(
        optical_path,
        optical_path_type,
        f"{id_prefix}:optical_path",
    )
    path_record = path_record.model_copy(
        update={
            "length": float(local_distances[-1]),
            "geometry_ids": tuple(record.resource_id for record in geometry_records),
        }
    )
    return (crs_record, *geometry_records, path_record)


@dataclass(frozen=True)
class _DistanceInterval:
    """Small helper for path-relative distance intervals."""

    start: float
    stop: float

    def intersect(self, other: _DistanceInterval) -> _DistanceInterval | None:
        """Return the overlap with another interval, if any."""
        start = max(self.start, other.start)
        stop = min(self.stop, other.stop)
        if start >= stop:
            return None
        return self.__class__(start, stop)

    def shift(self, offset: float) -> _DistanceInterval:
        """Return this interval shifted by an offset."""
        return self.__class__(self.start + offset, self.stop + offset)

    def reverse(self, length: float) -> _DistanceInterval:
        """Return this interval flipped around a path length."""
        return self.__class__(length - self.stop, length - self.start)


def _new_derived_annotation_id(resource_id: str) -> str:
    """Return a new id for an annotation derived from another annotation."""
    return f"{resource_id}:derived:{uuid4().hex}"


def _annotation_interval(annotation) -> _DistanceInterval:
    """Return the path interval occupied by an annotation."""
    return _DistanceInterval(annotation.start_distance, annotation.end_distance)


def _copy_annotation(annotation, interval: _DistanceInterval, *, preserve_id: bool):
    """Return an annotation copy with updated interval and identity."""
    updates = {
        "start_distance": interval.start,
        "end_distance": interval.stop,
    }
    if not preserve_id:
        updates["resource_id"] = _new_derived_annotation_id(annotation.resource_id)
    return annotation.model_copy(update=updates)


def _shift_annotations(annotations, offset: float, *, preserve_ids: bool = False):
    """Return annotations shifted by an offset."""
    return tuple(
        _copy_annotation(
            annotation,
            _annotation_interval(annotation).shift(offset),
            preserve_id=preserve_ids,
        )
        for annotation in annotations
    )


def _select_annotations(annotations, start: float, stop: float):
    """Return annotations clipped and shifted to a selected interval."""
    selection = _DistanceInterval(start, stop)
    out = []
    for annotation in annotations:
        interval = _annotation_interval(annotation)
        if (overlap := interval.intersect(selection)) is None:
            continue
        new_interval = overlap.shift(-start)
        preserve_id = interval == new_interval
        out.append(_copy_annotation(annotation, new_interval, preserve_id=preserve_id))
    return tuple(out)


def _reverse_annotations(annotations, length: float | None):
    """Return annotations flipped around a path length."""
    if not annotations:
        return ()
    if length is None:
        msg = "Reversing annotations requires optical path length."
        raise ParameterError(msg)
    return tuple(
        _copy_annotation(
            annotation,
            _annotation_interval(annotation).reverse(float(length)),
            preserve_id=False,
        )
        for annotation in reversed(annotations)
    )


def _merge_annotations(left_annotations, right_annotations, left_length: float | None):
    """Return annotations for a concatenated optical path."""
    if not right_annotations:
        return tuple(left_annotations)
    if right_annotations and left_length is None:
        msg = "Concatenating shifted annotations requires left path length."
        raise ParameterError(msg)
    shifted = _shift_annotations(right_annotations, float(left_length))
    return (*left_annotations, *shifted)


@dataclass(frozen=True)
class _IntervalSequence:
    """Private helper for length-based inventory interval sequences."""

    name: _IntervalTrackName
    records: tuple
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate id/runtime pairing."""
        if self.ids and self.records and len(self.ids) != len(self.records):
            msg = "Runtime records and resource id sequences must have same length."
            raise ParameterError(msg)

    @property
    def display_name(self) -> str:
        """Return the user-facing interval track name."""
        return {
            "optical_component": "optical_components",
            "geometry": "geometries",
            "coupling_condition": "coupling_conditions",
        }[self.name]

    @property
    def length(self) -> float | None:
        """Return summed sequence length, or None when no records are present."""
        if not self.records:
            return None
        lengths = tuple(item.length for item in self.records)
        if any(length is None for length in lengths):
            msg = f"Cannot validate {self.display_name}; all records must have length."
            raise ParameterError(msg)
        return sum(lengths)

    def validate_length(
        self,
        expected: float | None,
        *,
        tolerance: float,
    ) -> str | None:
        """Return a mismatch message, or None if length matches expected."""
        actual = self.length
        if actual is None or expected is None:
            return None
        if math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            return None
        return (
            f"{self.display_name} length {actual} does not match optical path "
            f"length {expected}."
        )

    def _check_supported_geometries(self) -> None:
        """Ensure geometry records can be safely clipped by length."""
        if self.name != "geometry":
            return
        unsupported = tuple(
            record.geometry_type
            for record in self.records
            if hasattr(record, "geometry_type")
            and record.geometry_type not in {"", "linear", "unknown"}
        )
        if unsupported:
            unsupported_str = ", ".join(sorted(set(unsupported)))
            msg = (
                "Distance selection currently supports only linear or unknown "
                f"geometry, not {unsupported_str}."
            )
            raise ParameterError(msg)

    def _clip_records(
        self,
        start: float,
        stop: float,
    ) -> tuple[tuple, tuple[int, ...]]:
        """Return length-clipped records and their original positions."""
        out = []
        indexes = []
        position = 0.0
        for index, item in enumerate(self.records):
            length = item.length
            if length is None:
                msg = (
                    f"Distance selection requires all {self.display_name} records "
                    "to have length."
                )
                raise ParameterError(msg)
            next_position = position + float(length)
            if length == 0:
                overlap = 0.0
                include = start <= position <= stop
            else:
                overlap = min(next_position, stop) - max(position, start)
                include = overlap > 0
            if include:
                local_start = max(position, start) - position
                local_stop = min(next_position, stop) - position
                if hasattr(item, "select"):
                    out.append(item.select(distance=(local_start, local_stop)))
                else:
                    out.append(item.model_copy(update={"length": overlap}))
                indexes.append(index)
            position = next_position
        return tuple(out), tuple(indexes)

    def select(
        self,
        start: float,
        stop: float,
        *,
        full_selection: bool,
    ) -> tuple[tuple[str, ...], tuple]:
        """Clip a paired id/runtime sequence by cumulative record length."""
        if self.records:
            self._check_supported_geometries()
            clipped, indexes = self._clip_records(start, stop)
            clipped_ids = (
                tuple(self.ids[index] for index in indexes) if self.ids else ()
            )
            return clipped_ids, clipped
        if self.ids and not full_selection:
            msg = (
                "Partial distance selections require runtime records with lengths. "
                "Use an inventory view before selecting an optical path with id-only "
                f"{self.display_name}."
            )
            raise ParameterError(msg)
        return self.ids, self.records

    def get_interval(self, target) -> tuple[float, float]:
        """Return the distance interval occupied by a target sequence item."""
        if not self.records:
            msg = (
                f"Finding {self.name} intervals requires runtime records with "
                "lengths. Use an inventory view before operating on id-only "
                "optical paths."
            )
            raise ParameterError(msg)
        target_id = _get_resource_id(target)
        position = 0.0
        for index, record in enumerate(self.records):
            length = record.length
            if length is None:
                msg = (
                    f"Finding {self.name} intervals requires all records "
                    "to have length."
                )
                raise ParameterError(msg)
            record_id = self.ids[index] if self.ids else record.resource_id
            next_position = position + float(length)
            if (
                record is target
                or record.resource_id == target_id
                or record_id == target_id
            ):
                return position, next_position
            position = next_position
        msg = f"{self.name} {target_id!r} was not found in this optical path."
        raise ParameterError(msg)
