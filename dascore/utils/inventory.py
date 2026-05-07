"""Utilities for inventory metadata models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
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
        default="",
        description=(
            "Stable inventory identifier for the real-world or logical metadata "
            "resource. DASCore generates random local ids for omitted values when "
            "records are stored in an Inventory."
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
    comment: str = Field(
        default="",
        description="Additional comments.",
    )

    @computed_field
    @property
    def type(self) -> str:
        """Return the inventory type name."""
        return self.__class__.__name__.lstrip("_")


class _TimedInventoryRecord(_InventoryRecord):
    """Base class for records with data-time validity intervals."""

    start_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Start time for which this metadata record is valid.",
    )
    end_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="End time for which this metadata record is valid.",
    )

    def is_effective_at(self, time) -> bool:
        """Return True if this record is valid at the supplied time."""
        time = to_datetime64(time)
        if pd.isnull(time):
            return True
        start = to_datetime64(self.start_time)
        end = to_datetime64(self.end_time)
        after_start = pd.isnull(start) or start <= time
        before_end = pd.isnull(end) or time < end
        return after_start and before_end


def _get_resource_id(record_or_id) -> str:
    """Return the resource id for an inventory record or id string."""
    if isinstance(record_or_id, str):
        return record_or_id
    return record_or_id.resource_id


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
    start, stop = annotation.distance
    if start is None or stop is None:
        msg = "Open annotation intervals require optical path context."
        raise ParameterError(msg)
    return _DistanceInterval(start, stop)


def _resolve_annotation_interval(annotation, length: float | None) -> _DistanceInterval:
    """Return the concrete path interval occupied by an annotation."""
    start, stop = annotation.distance
    start = 0.0 if start is None else float(start)
    if stop is None:
        if length is None:
            msg = "Open-ended annotation intervals require optical path length."
            raise ParameterError(msg)
        stop = float(length)
    else:
        stop = float(stop)
    if length is not None:
        if start >= float(length):
            msg = "Annotation distance start must be less than optical path length."
            raise ParameterError(msg)
        stop = min(stop, float(length))
    return _DistanceInterval(start, stop)


def _copy_annotation(annotation, interval: _DistanceInterval, *, preserve_id: bool):
    """Return an annotation copy with updated interval and identity."""
    updates = {
        "distance": (interval.start, interval.stop),
    }
    if not preserve_id:
        updates["resource_id"] = _new_derived_annotation_id(annotation.resource_id)
    return annotation.model_copy(update=updates)


def _shift_annotations(
    annotations,
    offset: float,
    *,
    length: float | None,
    preserve_ids: bool = False,
):
    """Return annotations shifted by an offset."""
    return tuple(
        _copy_annotation(
            annotation,
            _resolve_annotation_interval(annotation, length).shift(offset),
            preserve_id=preserve_ids,
        )
        for annotation in annotations
    )


def _select_annotations(
    annotations,
    start: float,
    stop: float,
    *,
    length: float | None,
):
    """Return annotations clipped and shifted to a selected interval."""
    selection = _DistanceInterval(start, stop)
    out = []
    for annotation in annotations:
        interval = _resolve_annotation_interval(annotation, length)
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
            _resolve_annotation_interval(annotation, length).reverse(float(length)),
            preserve_id=False,
        )
        for annotation in reversed(annotations)
    )


def _merge_annotations(
    left_annotations,
    right_annotations,
    left_length: float | None,
    right_length: float | None,
):
    """Return annotations for a concatenated optical path."""
    if not right_annotations:
        return tuple(left_annotations)
    if right_annotations and left_length is None:
        msg = "Concatenating shifted annotations requires left path length."
        raise ParameterError(msg)
    shifted = _shift_annotations(
        right_annotations,
        float(left_length),
        length=right_length,
    )
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
    def optical_length(self) -> float | None:
        """Return summed optical length, or None when no records are present."""
        if not self.records:
            return None
        lengths = tuple(item.optical_length for item in self.records)
        if any(length is None for length in lengths):
            msg = (
                f"Cannot validate {self.display_name}; all records must have "
                "optical_length."
            )
            raise ParameterError(msg)
        return sum(lengths)

    def validate_length(
        self,
        expected: float | None,
        *,
        tolerance: float,
    ) -> str | None:
        """Return a mismatch message, or None if length matches expected."""
        actual = self.optical_length
        if actual is None or expected is None:
            return None
        if math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            return None
        return (
            f"{self.display_name} optical_length {actual} does not match optical "
            f"path optical_length {expected}."
        )

    def _check_supported_geometries(self) -> None:
        """Ensure geometry records can be safely clipped by optical length."""
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
        """Return optical-length-clipped records and their original positions."""
        out = []
        indexes = []
        position = 0.0
        for index, item in enumerate(self.records):
            length = item.optical_length
            if length is None:
                msg = (
                    f"Distance selection requires all {self.display_name} records "
                    "to have optical_length."
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
                    out.append(item.model_copy(update={"optical_length": overlap}))
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
        """Clip a paired id/runtime sequence by cumulative optical length."""
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
            length = record.optical_length
            if length is None:
                msg = (
                    f"Finding {self.name} intervals requires all records "
                    "to have optical_length."
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
