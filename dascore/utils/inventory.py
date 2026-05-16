"""Utilities for inventory metadata models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal
from uuid import uuid4
from itertools import accumulate

import numpy as np
from pydantic import (
    Field, ConfigDict,
)

from dascore.exceptions import ParameterError
from dascore.utils.models import CommentableModel
from dascore.utils.misc import sanitize_range_param
from dascore.utils.inventory import _trim_sequence

_IntervalTrackName = Literal[
    "optical_component",
    "geometry",
    "coupling_condition",
]



class _PublicInventoryItem(CommentableModel):
    """Base class for objects with public reusable resource identifiers."""

    resource_id: str = Field(
        description="Stable identifier for a shareable inventory resource.",
        default_factory=lambda: str(uuid4()),
    )
    name: str = Field(
        default="", description="Human-readable resource name."
    )


class _OpticalLengthItem(CommentableModel):
    """Base class for optical path items with SI-only optical lengths."""

    model_config = ConfigDict(extra="forbid")

    optical_length: float = Field(
        default=0,
        description="Optical path interval length in meters.",
    )



def _get_resource_id(item_or_id) -> str:
    """Return the resource id for an inventory object or id string."""
    if isinstance(item_or_id, str):
        return item_or_id
    return getattr(item_or_id, "resource_id", "") or item_or_id.inventory_id


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


def _trim_sequence(
       distance, length_items: tuple[_OpticalLengthItem, ...],
) -> tuple[_OpticalLengthItem, ...]:
    """
    Trim a sequence that has optical lengths.
    """
    segment_stop = np.array(accumulate([x.optical_length for x in length_items]))
    segment_start = np.concatenate([0], segment_stop[:-1])
    # get values to remove, and the new lengths for each component.
    start, stop = sanitize_range_param(distance)
    start = 0 if start is None else start
    stop = segment_stop.max() if stop is None else stop
    new_lengths = segment_stop - segment_start
    adjustments = np.zeros_like(new_lengths)
    # handle ranges completely outside select range.
    new_lengths[((segment_stop < start) | (segment_start > stop))] = 0
    # next handle dynamic adjustments to length
    in_start = (segment_start > start) & (segment_start < start)
    in_stop = (segment_stop > start) & (segment_stop < start)
    adjustments[in_start] = new_lengths[in_start] - (start - segment_start)[in_start]
    adjustments[in_stop] = new_lengths[in_stop] - (segment_stop - stop)[in_stop]
    new_lengths += adjustments
    # package up and return.
    return tuple(
        x.new(optical_length=new_lengths)
        for x, new_len in zip(length_items, new_lengths)
    )


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
    out = annotation.model_copy(update=updates)
    if not preserve_id:
        object.__setattr__(out, "_inventory_id", "")
    return out


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
    items: tuple
    ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate id/runtime pairing."""
        if self.ids and self.items and len(self.ids) != len(self.items):
            msg = "Runtime items and resource id sequences must have same length."
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
        """Return summed optical length, or None when no items are present."""
        if not self.items:
            return 0
        return sum(tuple(item.optical_length for item in self.items))

    def validate_length(
        self,
        expected: float,
        *,
        tolerance: float,
    ) -> str | None:
        """Return a mismatch message, or None if length matches expected."""
        actual = self.optical_length
        if math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
            return None
        return (
            f"{self.display_name} optical_length {actual} does not match optical "
            f"path optical_length {expected}."
        )

    def _check_supported_geometries(self) -> None:
        """Ensure geometry items can be safely clipped by optical length."""
        if self.name != "geometry":
            return
        unsupported = tuple(
            item.geometry_type
            for item in self.items
            if hasattr(item, "geometry_type")
            and item.geometry_type not in {"", "linear", "unknown"}
        )
        if unsupported:
            unsupported_str = ", ".join(sorted(set(unsupported)))
            msg = (
                "Distance selection currently supports only linear or unknown "
                f"geometry, not {unsupported_str}."
            )
            raise ParameterError(msg)

    def _clip_items(
        self,
        start: float,
        stop: float,
    ) -> tuple[tuple, tuple[int, ...]]:
        """Return optical-length-clipped items and their original positions."""
        out = []
        indexes = []
        position = 0.0
        for index, item in enumerate(self.items):
            length = item.optical_length
            if length is None:
                msg = (
                    f"Distance selection requires all {self.display_name} items "
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
        if self.items:
            self._check_supported_geometries()
            clipped, indexes = self._clip_items(start, stop)
            clipped_ids = (
                tuple(self.ids[index] for index in indexes) if self.ids else ()
            )
            return clipped_ids, clipped
        if self.ids and not full_selection:
            msg = (
                "Partial distance selections require runtime items with lengths. "
                "Use an inventory view before selecting an optical path with id-only "
                f"{self.display_name}."
            )
            raise ParameterError(msg)
        return self.ids, self.items

    def get_interval(self, target) -> tuple[float, float]:
        """Return the distance interval occupied by a target sequence item."""
        if not self.items:
            msg = (
                f"Finding {self.name} intervals requires runtime items with "
                "lengths. Use an inventory view before operating on id-only "
                "optical paths."
            )
            raise ParameterError(msg)
        target_id = _get_resource_id(target)
        position = 0.0
        for index, item in enumerate(self.items):
            length = item.optical_length
            if length is None:
                msg = (
                    f"Finding {self.name} intervals requires all items "
                    "to have optical_length."
                )
                raise ParameterError(msg)
            item_id = self.ids[index] if self.ids else _get_resource_id(item)
            next_position = position + float(length)
            if (
                item is target
                or _get_resource_id(item) == target_id
                or item_id == target_id
            ):
                return position, next_position
            position = next_position
        msg = f"{self.name} {target_id!r} was not found in this optical path."
        raise ParameterError(msg)

