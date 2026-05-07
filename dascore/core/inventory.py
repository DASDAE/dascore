"""
DASCore's metadata model.

Inventory is intended to be a mixed-record metadata catalog for describing
patches and related fiber array context. Patch data, coordinates, and attrs
remain the primary source of truth, while Inventory records provide richer
context which can be resolved explicitly by fiber array id and time.
Moreover, Patches can defer to specific fiber arrays for their metadata,
enabling the decoupling of metadata from data for DAS archives.

The Inventory itself is the canonical storage layer. Its ``records`` field is
an id-only collection of plain dictionaries, with relationships stored as ids
such as ``optical_path_ids`` or ``geometry_ids``. This is the durable form used
by ``to_dict``, JSON, and YAML.

Resolved Pydantic records are materialized views over that storage. Calling
``Inventory.get_records`` returns object graphs with relationships resolved,
such as a ``FiberArray`` whose ``optical_paths`` contain ``OpticalPath`` objects.
Those objects are ergonomic views, not the source of truth. To write them back,
use ``Inventory.put_records``; it extracts linked objects recursively, stores
canonical id-only records, and validates the resulting inventory.


## user stories
  1. As a DASCore user with raw or lightly processed patches, I want to attach a data_source_id to a patch so that later processing can resolve the correct inventory context.
  2. As a user with channel-indexed DAS data, I want to derive optical distance from an inventory acquisition so that geometry and fiber metadata can be projected onto the patch.
  3. As a user with an inventory and a distance-indexed patch, I want to add selected optical-path metadata as patch coordinates so that I can process, select, and visualize data by labels, geometry, coupling, or fiber
     properties.
  4. As a user, I want to add selected fiber array, acquisition, and interrogator fields to patch attrs so that downstream workflows can carry compact metadata without embedding the whole inventory.
  5. As a user managing evolving deployments, I want inventory records to be valid over time so that the same data_source_id can resolve to different configurations or paths for different data epochs.
  6. As a user building an inventory, I want to write records as linked Pydantic objects but store them as stable id-based records so that inventories are ergonomic in Python and durable in JSON/YAML.
  7. As a user exchanging or archiving metadata, I want inventories to round-trip through dict, JSON, and YAML so that I can store metadata beside data files.

"""
# ruff: noqa: E501

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, TypeAlias
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator
from typing_extensions import Self

from dascore.config import get_config
from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    path_types,
)
from dascore.exceptions import ParameterError
from dascore.utils.inventory import (
    _IntervalSequence,
    _IntervalTrackName,
    _InventoryRecord,
    _merge_annotations,
    _reverse_annotations,
    _select_annotations,
    _TimedInventoryRecord,
)
from dascore.utils.misc import sanitize_range_param
from dascore.utils.models import DascoreBaseModel, DateTime64, UnitQuantity
from dascore.utils.time import to_datetime64

ExtraFieldValue: TypeAlias = str | int | float | bool


def _inventory_type_slug(record_type: str) -> str:
    """Return a compact snake_case slug for one inventory record type."""
    chars = []
    for index, char in enumerate(record_type):
        if char.isupper() and index:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


class FiberArray(_TimedInventoryRecord):
    """Durable fiber-optic observing identity anchored by an optical path."""

    code: str = Field(default="", description="Station-like fiber array code.")
    name: str = Field(default="", description="Human-readable fiber array name.")
    acquisitions: tuple[Acquisition, ...] = Field(
        default=(),
        description="Acquisitions associated with this fiber array.",
    )
    optical_paths: tuple[OpticalPath, ...] = Field(
        default=(),
        description="Optical paths associated with this fiber array.",
    )
    dims: str = Field(
        default="", description="Comma-separated default patch dimensions."
    )
    tag: str = Field(default="", description="Default tag for acquired patches.")


class Network(_InventoryRecord):
    """FDSN-like organizational container for inventory observing objects."""

    code: str = Field(default="", description="Network code.")
    name: str = Field(default="", description="Human-readable network name.")
    description: str = Field(default="", description="Network description.")
    fiber_arrays: tuple[FiberArray, ...] = Field(
        default=(),
        description="Fiber arrays associated with this network.",
    )
    stations: tuple[None, ...] = ()  # just a stub for now


class Interrogator(_InventoryRecord):
    """DAS interrogator unit used for data collection."""

    manufacturer: str = Field(
        default="", description="Manufacturer name of the interrogator."
    )
    model: str = Field(default="", description="Model number of the interrogator.")
    serial_number: str = Field(
        default="", description="Serial number of the interrogator."
    )
    firmware_version: str = Field(
        default="", description="Firmware version used within the interrogator."
    )
    instrument_type: str = Field(
        default="interrogator",
        description="General instrument category.",
    )


class Acquisition(_TimedInventoryRecord):
    """
    Channel-like DAS acquisition setup.

    `Acquisition` is the inventory-side source of acquisition-derived patch
    metadata. It is not a replacement for [`PatchAttrs`](
    `dascore.core.attrs.PatchAttrs`); instead, patches use `data_source_id` to
    resolve an acquisition and may copy selected acquisition fields onto patch
    attrs. Fields such as `data_type`, `data_category`, `data_units`,
    acquisition sample rate, gauge length, pulse width, and channel-to-distance
    alignment describe the acquisition context. Patch-local values such as
    processing `history`, `tag`, and post-processing unit changes remain on
    `PatchAttrs`.
    """

    code: str = Field(
        default="",
        description="Channel-like acquisition code.",
    )
    location_code: str = Field(
        default="",
        description=(
            "FDSN-style location code used to group or disambiguate acquisition "
            "records within a fiber array. Empty location codes are allowed."
        ),
    )
    data_type: str = Field(
        default="",
        description="Quantity measured or produced by this acquisition.",
    )
    data_category: str = Field(
        default="",
        description="Acquisition family such as DAS, DTS, or DSS.",
    )
    data_units: UnitQuantity | None = Field(
        default=None,
        description="Units of data produced by this acquisition.",
    )
    interrogator: Interrogator | None = Field(
        default=None,
        description="Interrogator used for this acquisition.",
    )
    gauge_length: float | None = Field(
        default=None, description="Gauge length used by the acquisition in meters."
    )
    pulse_rate: float | None = Field(
        default=None, description="Pulse repetition rate for the configuration in Hz."
    )
    pulse_width: float | None = Field(
        default=None, description="Pulse width for the acquisition in seconds."
    )
    acquisition_sample_rate: float | None = Field(
        default=None, description="FDSN-style acquisition sample rate in Hz."
    )
    spatial_sampling_interval: float | None = Field(
        default=None,
        description="Spatial sampling interval between channels in meters.",
    )
    first_channel_distance: float = Field(
        default=0.0,
        description=(
            "Optical path distance in meters corresponding to patch channel/index "
            "position 0. This is a channel-to-path alignment parameter for "
            "primitive or partial file formats; it should not be used instead of "
            "modeling lead-in, telemetry, unknown, or other non-sensing intervals "
            "in OpticalPath when those intervals matter."
        ),
    )
    extra_fields: dict[str, ExtraFieldValue] = Field(
        default_factory=dict,
        description=(
            "Extra acquisition metadata not represented by standardized fields. "
            "This maps to FDSN DAS Metadata native_headers when exporting."
        ),
    )

    @model_validator(mode="after")
    def _validate_data_fields(self) -> Self:
        """Validate acquisition data descriptors."""
        if self.data_type not in VALID_DATA_TYPES:
            msg = (
                f"data_type must be one of {VALID_DATA_TYPES}, not {self.data_type!r}."
            )
            raise ValueError(msg)
        if self.data_category not in VALID_DATA_CATEGORIES:
            msg = (
                f"data_category must be one of {VALID_DATA_CATEGORIES}, "
                f"not {self.data_category!r}."
            )
            raise ValueError(msg)
        return self


class ExternalResource(_InventoryRecord):
    """External resource identified but not otherwise modeled by DASCore."""

    uri: str = Field(default="", description="URI or identifier for the resource.")
    name: str = Field(default="", description="Human-readable resource name.")
    description: str = Field(
        default="", description="Free-form description of the external resource."
    )


class _OpticalLengthRecord(_InventoryRecord):
    """Base class for optical path records with SI-only optical lengths."""

    model_config = ConfigDict(extra="forbid")

    optical_length: float | None = Field(
        default=None,
        description="Optical path interval length in meters.",
    )


class Cable(_InventoryRecord):
    """Physical cable containing one or more fiber segments."""

    name: str = Field(default="", description="Human-readable cable name.")
    owner: str = Field(default="", description="Proprietary owner of the cable.")
    manufacturer: str = Field(default="", description="Manufacturer of the cable.")
    model: str = Field(default="", description="Model name of the cable.")
    serial_number: str = Field(
        default="", description="Manufacturer serial number for the cable."
    )
    specification: ExternalResource | None = Field(
        default=None,
        description="External cable specification, datasheet, or product page.",
    )
    fiber_count: int | None = Field(
        default=None, description="Number of fibers contained in the cable."
    )


class Enclosure(_InventoryRecord):
    """Physical housing for optical components."""

    name: str = Field(default="", description="Human-readable enclosure name.")
    owner: str = Field(default="", description="Proprietary owner of the enclosure.")
    manufacturer: str = Field(default="", description="Manufacturer of the enclosure.")
    model: str = Field(default="", description="Model name of the enclosure.")
    serial_number: str = Field(
        default="", description="Manufacturer serial number for the enclosure."
    )
    specification: ExternalResource | None = Field(
        default=None,
        description="External enclosure specification, datasheet, or product page.",
    )


class OpticalComponent(_OpticalLengthRecord):
    """Base class for physical optical components in an optical path."""

    optical_length: float | None = Field(
        default=None,
        description="Optical component length along the optical path in meters.",
    )
    name: str = Field(default="", description="Human-readable component name.")
    container: Cable | Enclosure | None = Field(
        default=None,
        description="Optional physical container housing this optical component.",
    )


class FiberSegment(OpticalComponent):
    """Physical fiber interval in an optical path."""

    fiber_type: str = Field(
        default="", description="Fiber type, such as single_mode or multi_mode."
    )
    fiber_standard: str = Field(
        default="",
        description=(
            "Fiber standard or grade, such as ITU-T G.652.D or ITU-T G.657.A1."
        ),
    )
    fiber_index: int | None = Field(
        default=None, description="Fiber position or index within the parent cable."
    )
    subunit_id: str = Field(
        default="",
        description=(
            "Identifier for the tube, ribbon, bundle, or other cable subunit "
            "containing this fiber."
        ),
    )
    buffer_type: str = Field(
        default="",
        description=(
            "Per-fiber or subunit buffer construction, such as tight_buffered, "
            "loose_buffered, bare, ribbon, or unknown."
        ),
    )
    buffer_material: str = Field(
        default="", description="Buffer or coating material around this fiber."
    )
    buffer_outer_diameter: float | None = Field(
        default=None, description="Outer diameter of the fiber buffer in meters."
    )
    color: str = Field(
        default="", description="Fiber or buffer color used for field identification."
    )
    group_refractive_index: float | None = Field(
        default=None,
        description=(
            "Group refractive index used to convert optical time-of-flight to "
            "distance."
        ),
    )
    attenuation_coefficient: float | None = Field(
        default=None,
        description="Optical attenuation coefficient in dB/km.",
    )


class Connector(OpticalComponent):
    """Optical connector in an optical path."""

    connector_type: str = Field(default="", description="Connector type.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )
    return_loss: float | None = Field(
        default=None, description="Return loss in dB, if known."
    )


class Splice(OpticalComponent):
    """Optical splice in an optical path."""

    splice_type: str = Field(default="", description="Splice type.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )


class Coupler(OpticalComponent):
    """Optical coupler or splitter in an optical path."""

    coupler_type: str = Field(default="", description="Coupler or splitter type.")
    coupling_ratio: str = Field(default="", description="Coupling ratio, if known.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )


class Turnaround(OpticalComponent):
    """Optical turnaround or loopback component in an optical path."""

    turnaround_type: str = Field(default="", description="Turnaround type.")


class Terminator(OpticalComponent):
    """Optical path terminator."""

    terminator_type: str = Field(default="", description="Terminator type.")
    reflectance: float | None = Field(
        default=None, description="Reflectance in dB, if known."
    )


class CoordinateReferenceSystem(_InventoryRecord):
    """
    Coordinate reference system used by geometry records.

    The default CRS is WGS84 geographic latitude/longitude (`EPSG:4326`). Custom
    projected or local engineering coordinate systems can be represented with a
    local authority/code pair, or with formal definitions in WKT, PROJJSON, or
    CF grid-mapping form.
    """

    authority: str = Field(
        default="EPSG",
        description="CRS authority such as EPSG, OGC, CF, or LOCAL.",
    )
    code: str = Field(
        default="4326",
        description="Authority code or local project code for this CRS.",
    )
    name: str = Field(default="", description="Human-readable CRS name.")
    crs_wkt: str = Field(
        default="",
        description="Well-known text CRS definition, if available.",
    )
    projjson: dict[str, Any] = Field(
        default_factory=dict,
        description="PROJJSON CRS definition, if available.",
    )
    grid_mapping: dict[str, Any] = Field(
        default_factory=dict,
        description="CF grid-mapping attributes, if available.",
    )
    origin: tuple[float, ...] = Field(
        default=(), description="Origin for local coordinate systems."
    )
    axis_order: tuple[str, ...] = Field(
        default=("latitude", "longitude", "elevation"),
        description=(
            "Coordinate axis order, such as latitude, longitude, elevation or x, y, z."
        ),
    )
    units: str = Field(default="degree", description="Coordinate units when known.")
    vertical_datum: str = Field(
        default="", description="Vertical datum or reference surface, if known."
    )


def _default_coordinate_reference_system() -> CoordinateReferenceSystem:
    """Return the default WGS84 geographic CRS used by geometries."""
    return CoordinateReferenceSystem(resource_id="EPSG:4326")


class Geometry(_OpticalLengthRecord):
    """Geometry for an interval of an optical path."""

    name: str = Field(default="", description="Human-readable geometry name.")
    optical_length: float | None = Field(
        default=None,
        description="Optical path interval length described by this geometry in meters.",
    )
    geometry_type: str = Field(
        default="",
        description="Kind of geometry, such as linear, curve, or unknown.",
    )
    coordinate_reference_system: CoordinateReferenceSystem | None = Field(
        default_factory=_default_coordinate_reference_system,
        description="Coordinate reference system used by this geometry.",
    )
    coordinates: tuple[tuple[float, ...], ...] = Field(
        default=(), description="Coordinate points describing the geometry."
    )
    reference: str = Field(
        default="", description="Optional URI, path, or name for external geometry."
    )


class ClampPoint(DascoreBaseModel):
    """Discrete clamp location within a coupling interval."""

    model_config = ConfigDict(
        title="Clamp Point",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    distance: float = Field(
        default=0.0,
        description=(
            "Distance in meters from the start of the coupling interval to the "
            "clamp point."
        ),
    )
    clamp_type: str = Field(
        default="", description="Clamp type or hardware style at this point."
    )
    attachment: str = Field(
        default="", description="Attachment method used by this clamp point."
    )
    comment: str = Field(default="", description="Additional comments.")


class CouplingCondition(_OpticalLengthRecord):
    """Acoustic coupling condition for an interval of an optical path."""

    optical_length: float | None = Field(
        default=None,
        description="Optical path interval length described by this coupling in meters.",
    )
    coupling_type: str = Field(
        default="",
        description=(
            "Kind of acoustic coupling, such as buried, trenched, cemented, "
            "clamped, conduit, suspended, loose, surface_laid, or unknown."
        ),
    )
    medium: str = Field(
        default="", description="Surrounding medium, such as soil, rock, or steel."
    )
    attachment: str = Field(
        default="", description="Attachment method, such as clamped or bonded."
    )
    quality: str = Field(
        default="", description="Qualitative coupling quality, if known."
    )
    deployment_type: str = Field(
        default="",
        description=(
            "Deployment setting, such as borehole, surface, trench, conduit, "
            "structure, or unknown."
        ),
    )
    installation_method: str = Field(
        default="",
        description=(
            "How the cable was installed or coupled, such as cemented, clamped, "
            "buried, pinned, weighted, loose, or unknown."
        ),
    )
    contact_medium: str = Field(
        default="",
        description=(
            "Material directly contacting the cable, such as cement, grout, soil, "
            "sand, casing, tubing, conduit, air, or water."
        ),
    )
    embedment_depth: float | None = Field(
        default=None,
        description="Depth below the local surface or host interface in meters.",
    )
    confining_pressure: float | None = Field(
        default=None,
        description=(
            "Approximate normal pressure coupling the cable to the surrounding "
            "medium in pascals."
        ),
    )
    tension: float | None = Field(
        default=None, description="Approximate axial cable tension in newtons."
    )
    coupling_quality_index: float | None = Field(
        default=None,
        description="Optional normalized or method-specific coupling quality estimate.",
    )
    quality_method: str = Field(
        default="",
        description=(
            "Method used to estimate coupling quality, such as visual inspection, "
            "tap test, coherence, SNR, or unknown."
        ),
    )
    clamp_spacing: float | None = Field(
        default=None, description="Nominal spacing between clamp points in meters."
    )
    clamp_points: tuple[ClampPoint, ...] = Field(
        default=(),
        description=(
            "Discrete clamp locations measured in meters from the start of this "
            "coupling interval."
        ),
    )

    def select(self, *, distance) -> Self:
        """Return this coupling condition selected by local distance."""
        start, stop = OpticalPath._get_distance_limits(distance, self.optical_length)
        clamp_points = tuple(
            point.model_copy(update={"distance": point.distance - start})
            for point in self.clamp_points
            if start <= point.distance < stop
        )
        return self.model_copy(
            update={
                "optical_length": stop - start,
                "clamp_points": clamp_points,
            }
        )


class OpticalPathAnnotation(_InventoryRecord):
    """Named or categorized interval on an optical path."""

    distance: tuple[float | None, float | None] = Field(
        default=(None, None),
        description="DASCore-style optical distance interval for this annotation.",
    )
    label: str = Field(
        default="", description="Human-readable label for this path annotation."
    )
    category: str = Field(
        default="", description="Optional annotation category or namespace."
    )

    @field_validator("distance", mode="before")
    @classmethod
    def _normalize_distance(cls, value):
        """Normalize DASCore open interval sentinels before type validation."""
        start, stop = sanitize_range_param(value)
        start = None if start is None else float(start)
        stop = None if stop is None else float(stop)
        return start, stop

    @model_validator(mode="after")
    def _validate_interval(self) -> Self:
        """Lightly validate the annotation interval."""
        start, stop = self.distance
        if start is not None and start < 0:
            msg = "Annotation distance start must be greater than or equal to 0."
            raise ValueError(msg)
        if stop is not None and stop <= 0:
            msg = "Annotation distance stop must be greater than 0."
            raise ValueError(msg)
        if start is not None and stop is not None and start >= stop:
            msg = "Annotation distance start must be less than distance stop."
            raise ValueError(msg)
        return self


class OpticalPath(_TimedInventoryRecord):
    """Continuous optical path described by independent component sequences."""

    optical_length: float | None = Field(
        default=None, description="Declared total optical path length in meters."
    )
    name: str = Field(default="", description="Human-readable optical path name.")
    optical_components: tuple[OpticalComponent, ...] = Field(
        default=(),
        description="Ordered optical components on this path.",
    )
    geometries: tuple[Geometry, ...] = Field(
        default=(),
        description="Ordered geometries on this path.",
    )
    coupling_conditions: tuple[CouplingCondition, ...] = Field(
        default=(),
        description="Ordered coupling conditions on this path.",
    )
    annotations: tuple[OpticalPathAnnotation, ...] = Field(
        default=(),
        description="Ordered optical path annotations.",
    )

    @staticmethod
    def _add_lengths(length_1: float | None, length_2: float | None) -> float | None:
        """Return summed lengths, or None when either length is unknown."""
        if length_1 is None or length_2 is None:
            return None
        return length_1 + length_2

    @property
    def is_empty(self) -> bool:
        """Return True if this optical path has no known interval length."""
        if self.optical_length is not None:
            return self.optical_length == 0
        return not any(
            (
                self.optical_components,
                self.geometries,
                self.coupling_conditions,
                self.annotations,
            )
        )

    @staticmethod
    def _get_distance_limits(distance, length: float | None) -> tuple[float, float]:
        """Validate and normalize a distance selection."""
        start, stop = sanitize_range_param(distance)
        start = 0.0 if start is None else float(start)
        if stop is None:
            if length is None:
                msg = "Open-ended distance selections require optical path length."
                raise ParameterError(msg)
            stop = float(length)
        else:
            stop = float(stop)
        if start > stop:
            msg = "Distance selection start must be less than or equal to stop."
            raise ParameterError(msg)
        if length is not None:
            start = max(start, 0.0)
            stop = min(stop, float(length))
        stop = max(stop, start)
        return start, stop

    @staticmethod
    def _is_full_distance_selection(
        start: float,
        stop: float,
        length: float | None,
    ) -> bool:
        """Return True when the distance selection covers the whole known path."""
        return length is not None and start == 0.0 and stop == float(length)

    def _get_interval_sequence(self, name: _IntervalTrackName) -> _IntervalSequence:
        """Return an interval sequence wrapper for a named path track."""
        match name:
            case "optical_component":
                return _IntervalSequence(
                    name,
                    records=self.optical_components,
                )
            case "geometry":
                return _IntervalSequence(
                    name,
                    records=self.geometries,
                )
            case "coupling_condition":
                return _IntervalSequence(
                    name,
                    records=self.coupling_conditions,
                )
            case _:
                msg = (
                    "kind must be 'optical_component', 'geometry', "
                    f"or 'coupling_condition', not {name!r}."
                )
                raise ParameterError(msg)

    def _select_interval_sequence(
        self,
        name: _IntervalTrackName,
        start: float,
        stop: float,
        full_selection: bool,
    ) -> tuple[tuple[str, ...], tuple]:
        """Select one named interval sequence."""
        sequence = self._get_interval_sequence(name)
        return sequence.select(start, stop, full_selection=full_selection)

    @staticmethod
    def _get_interval_target(
        component,
        optical_component,
        geometry,
        coupling_condition,
    ) -> tuple[_IntervalTrackName, object]:
        """Normalize path interval target inputs to a single kind and object."""
        targets = {
            "optical_component": optical_component,
            "geometry": geometry,
            "coupling_condition": coupling_condition,
        }
        if component is not None:
            if isinstance(component, OpticalComponent):
                targets["optical_component"] = component
            elif isinstance(component, Geometry):
                targets["geometry"] = component
            elif isinstance(component, CouplingCondition):
                targets["coupling_condition"] = component
            elif isinstance(component, OpticalPathAnnotation):
                msg = "Annotations are not valid get_distance_interval targets."
                raise ParameterError(msg)
            else:
                msg = (
                    "Positional interval targets must be inventory objects. "
                    "Pass string resource ids with optical_component=, geometry=, "
                    "or coupling_condition=."
                )
                raise ParameterError(msg)
        targets = {key: value for key, value in targets.items() if value is not None}
        if len(targets) != 1:
            msg = "exactly one optical path interval target is required."
            raise ParameterError(msg)
        return next(iter(targets.items()))

    def __add__(self, other):
        """Concatenate two optical paths."""
        if not isinstance(other, OpticalPath):
            return NotImplemented
        annotations = _merge_annotations(
            self.annotations,
            other.annotations,
            self.optical_length,
            other.optical_length,
        )
        return self.__class__(
            optical_length=self._add_lengths(self.optical_length, other.optical_length),
            optical_components=(
                *self.optical_components,
                *other.optical_components,
            ),
            geometries=(*self.geometries, *other.geometries),
            coupling_conditions=(
                *self.coupling_conditions,
                *other.coupling_conditions,
            ),
            annotations=annotations,
        )

    def __radd__(self, other):
        """Support summing optical path sequences."""
        if other == 0:
            return self
        return NotImplemented

    def reverse(self) -> Self:
        """Return a new optical path with all ordered subcomponents reversed."""
        annotations = _reverse_annotations(self.annotations, self.optical_length)
        return self.__class__(
            optical_length=self.optical_length,
            optical_components=tuple(reversed(self.optical_components)),
            geometries=tuple(reversed(self.geometries)),
            coupling_conditions=tuple(reversed(self.coupling_conditions)),
            annotations=annotations,
        )

    def select(self, **kwargs) -> Self:
        """Return a new optical path selected by distance."""
        extra = set(kwargs) - {"distance"}
        if extra:
            extra_str = ", ".join(sorted(extra))
            msg = f"Unsupported optical path selection argument(s): {extra_str}."
            raise ParameterError(msg)
        if "distance" not in kwargs:
            return self
        start, stop = self._get_distance_limits(kwargs["distance"], self.optical_length)
        full_selection = self._is_full_distance_selection(
            start, stop, self.optical_length
        )
        _, optical_components = self._select_interval_sequence(
            "optical_component",
            start,
            stop,
            full_selection,
        )
        _, geometries = self._select_interval_sequence(
            "geometry",
            start,
            stop,
            full_selection,
        )
        _, coupling_conditions = self._select_interval_sequence(
            "coupling_condition",
            start,
            stop,
            full_selection,
        )
        annotations = _select_annotations(
            self.annotations,
            start,
            stop,
            length=self.optical_length,
        )
        return self.__class__(
            optical_length=stop - start,
            optical_components=optical_components,
            geometries=geometries,
            coupling_conditions=coupling_conditions,
            annotations=annotations,
        )

    def split_at(self, distance: float) -> tuple[Self, Self]:
        """Split the optical path into two paths at a distance."""
        distance = float(distance)
        return (
            self.select(distance=(0.0, distance)),
            self.select(distance=(distance, None)),
        )

    def split(self, *, distance: float) -> tuple[Self, Self]:
        """Split the optical path by keyword distance."""
        return self.split_at(distance)

    def get_distance_interval(
        self,
        component_or_id,
        *,
        kind: str | None = None,
    ) -> tuple[float, float]:
        """Return the distance interval for a component, geometry, or coupling."""
        if kind is None:
            if isinstance(component_or_id, str):
                msg = (
                    "String resource ids require kind='optical_component', "
                    "'geometry', or 'coupling_condition'."
                )
                raise ParameterError(msg)
            kind, target = self._get_interval_target(component_or_id, None, None, None)
        else:
            target = component_or_id
        if kind == "annotation":
            msg = "Annotations are not valid get_distance_interval targets."
            raise ParameterError(msg)
        sequence = self._get_interval_sequence(kind)
        return sequence.get_interval(target)

    def validate(self, tolerance: float = 1e-9) -> Self:
        """Validate that populated interval sequences match path optical length."""
        expected = self.optical_length
        sequences = tuple(
            self._get_interval_sequence(name)
            for name in (
                "optical_component",
                "geometry",
                "coupling_condition",
            )
        )
        lengths = tuple(sequence.optical_length for sequence in sequences)
        if expected is None:
            known_lengths = tuple(length for length in lengths if length is not None)
            expected = known_lengths[0] if known_lengths else None
        errors = tuple(
            error
            for error in (
                sequence.validate_length(expected, tolerance=tolerance)
                for sequence in sequences
            )
            if error
        )
        if errors:
            msg = "Optical path optical length validation failed:\n" + "\n".join(errors)
            raise ParameterError(msg)
        return self

    def remove_component(
        self,
        component=None,
        *,
        optical_component=None,
        geometry=None,
        coupling_condition=None,
    ) -> Self:
        """Return a new optical path with one component interval removed."""
        target_type, target = self._get_interval_target(
            component,
            optical_component,
            geometry,
            coupling_condition,
        )
        match target_type:
            case "optical_component":
                start, stop = self.get_distance_interval(
                    target,
                    kind="optical_component",
                )
            case "geometry":
                start, stop = self.get_distance_interval(
                    target,
                    kind="geometry",
                )
            case "coupling_condition":
                start, stop = self.get_distance_interval(
                    target,
                    kind="coupling_condition",
                )
        return self.select(distance=(0.0, start)) + self.select(distance=(stop, None))


INVENTORY_RECORD_TYPES: dict[str, type[_InventoryRecord]] = {
    cls.__name__: cls
    for cls in (
        FiberArray,
        Network,
        Cable,
        CoordinateReferenceSystem,
        Enclosure,
        Connector,
        CouplingCondition,
        Coupler,
        ExternalResource,
        FiberSegment,
        Geometry,
        Interrogator,
        Acquisition,
        OpticalPathAnnotation,
        OpticalPath,
        Splice,
        Terminator,
        Turnaround,
    )
}

OPTICAL_COMPONENT_TYPES = (
    Connector,
    Coupler,
    FiberSegment,
    Splice,
    Terminator,
    Turnaround,
)


@dataclass(frozen=True)
class _Relationship:
    """Inventory relationship between an object field and a canonical id field."""

    owner: type[_InventoryRecord]
    field: str
    storage_field: str
    target: type[_InventoryRecord] | tuple[type[_InventoryRecord], ...]
    many: bool = False


INVENTORY_RELATIONSHIPS: tuple[_Relationship, ...] = (
    _Relationship(Network, "fiber_arrays", "fiber_array_ids", FiberArray, many=True),
    _Relationship(
        FiberArray,
        "acquisitions",
        "acquisition_ids",
        Acquisition,
        many=True,
    ),
    _Relationship(
        FiberArray,
        "optical_paths",
        "optical_path_ids",
        OpticalPath,
        many=True,
    ),
    _Relationship(
        Acquisition,
        "interrogator",
        "interrogator_resource_id",
        Interrogator,
    ),
    _Relationship(
        Cable, "specification", "specification_resource_id", ExternalResource
    ),
    _Relationship(
        Enclosure, "specification", "specification_resource_id", ExternalResource
    ),
    _Relationship(
        OpticalComponent,
        "container",
        "container_id",
        (Cable, Enclosure),
    ),
    _Relationship(
        Geometry,
        "coordinate_reference_system",
        "coordinate_reference_system_id",
        CoordinateReferenceSystem,
    ),
    _Relationship(
        OpticalPath,
        "optical_components",
        "optical_component_ids",
        OPTICAL_COMPONENT_TYPES,
        many=True,
    ),
    _Relationship(OpticalPath, "geometries", "geometry_ids", Geometry, many=True),
    _Relationship(
        OpticalPath,
        "coupling_conditions",
        "coupling_condition_ids",
        CouplingCondition,
        many=True,
    ),
    _Relationship(
        OpticalPath,
        "annotations",
        "annotation_ids",
        OpticalPathAnnotation,
        many=True,
    ),
)


class Inventory(DascoreBaseModel):
    """
    A time-aware catalog of DASCore metadata records.

    See the [inventory tutorial](/tutorial/inventory.qmd) for examples of
    enriching patches from inventories and the
    [inventory design note](/notes/inventory_design.qmd) for model rationale.
    """

    model_config = ConfigDict(
        title="Inventory",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    default_file_name: ClassVar[str] = "dascore-inventory.yaml"

    dascore_inventory: bool = Field(
        default=True, description="Marker identifying a DASCore inventory manifest."
    )
    schema_version: int = Field(
        default=1,
        description=(
            "Version of the inventory manifest/envelope schema; used for "
            "manifest migration."
        ),
    )
    resource_id: str = Field(
        default="", description="Optional identifier for this inventory manifest."
    )
    creation_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Time this inventory manifest was created.",
    )
    author_id: str = Field(
        default="", description="Identifier for the manifest author."
    )
    comment: str = Field(default="", description="Additional comments.")
    _records: tuple[dict[str, Any], ...] = PrivateAttr(default_factory=tuple)

    def __init__(self, **data):
        """Create an inventory and store canonical records privately."""
        records = data.pop("records", ())
        super().__init__(**data)
        object.__setattr__(self, "_records", tuple(self._canonicalize_many(records)))
        if not self.dascore_inventory:
            msg = "Inventory manifest must set dascore_inventory to true."
            raise ValueError(msg)
        self._validate_resource_ids()
        self._validate_references()

    @classmethod
    def _relationships_for(cls, record_cls: type[_InventoryRecord]):
        """Return relationship specs for a record class."""
        return tuple(
            rel for rel in INVENTORY_RELATIONSHIPS if issubclass(record_cls, rel.owner)
        )

    @property
    def records(self) -> tuple[dict[str, Any], ...]:
        """Return copies of canonical id-only inventory records."""
        return tuple(dict(record) for record in self._records)

    def new(self, **kwargs) -> Self:
        """Create a new Inventory with updated attributes."""
        if "records" not in kwargs:
            kwargs["records"] = self.records
        return self.__class__(**(self.to_dict() | kwargs))

    @staticmethod
    def _normalize_filter(value) -> tuple | None:
        """Normalize scalar or sequence filters."""
        if value is None:
            return None
        if isinstance(value, str | type):
            return (value,)
        return tuple(value)

    @staticmethod
    def _type_name(record_type) -> str:
        """Return canonical record type name."""
        return record_type if isinstance(record_type, str) else record_type.__name__

    @classmethod
    def _record_cls(cls, record_type: str) -> type[_InventoryRecord]:
        """Return record class for a canonical record type."""
        try:
            return INVENTORY_RECORD_TYPES[record_type]
        except KeyError:
            valid = sorted(INVENTORY_RECORD_TYPES)
            msg = f"Unknown inventory object type {record_type!r}. Valid types are {valid}."
            raise ValueError(msg) from None

    @classmethod
    def _canonicalize_many(cls, records) -> tuple[dict[str, Any], ...]:
        """Return canonical records for mappings or inventory record objects."""
        out = []
        seen = set()
        cache = {}
        for record in records:
            for item in cls._canonicalize_record(record, cache=cache):
                key = cls._record_cache_key(item)
                if key not in seen:
                    out.append(item)
                    seen.add(key)
        return tuple(out)

    @classmethod
    def _canonicalize_record(
        cls,
        record,
        cache=None,
    ) -> tuple[dict[str, Any], ...]:
        """Return supporting records plus one canonical id-only record."""
        records, _ = cls._canonicalize_record_with_id(record, cache=cache)
        return records

    @classmethod
    def _canonicalize_record_with_id(
        cls,
        record,
        cache=None,
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        """Return canonical records and the record id for one inventory object."""
        if isinstance(record, Mapping):
            out = cls._canonicalize_mapping(record)
            return (out,), out["resource_id"]
        if not isinstance(record, _InventoryRecord):
            msg = "records must contain inventory record instances or mappings."
            raise ParameterError(msg)
        if cache is None:
            cache = {}
        if id(record) in cache:
            return cache[id(record)]
        supporting = []
        data = record.model_dump(mode="json", exclude_none=True)
        data["type"] = record.type
        for relationship in cls._relationships_for(record.__class__):
            data.pop(relationship.field, None)
            if relationship.many:
                values = tuple(getattr(record, relationship.field))
                if values:
                    resource_ids = []
                    for value in values:
                        records, resource_id = cls._canonicalize_record_with_id(
                            value,
                            cache=cache,
                        )
                        supporting.extend(records)
                        resource_ids.append(resource_id)
                    data[relationship.storage_field] = tuple(resource_ids)
            elif (value := getattr(record, relationship.field)) is not None:
                records, resource_id = cls._canonicalize_record_with_id(
                    value,
                    cache=cache,
                )
                supporting.extend(records)
                data[relationship.storage_field] = resource_id
        if not data.get("resource_id"):
            data["resource_id"] = cls._generated_resource_id(record.type)
        canonical = cls._canonicalize_mapping(data)
        out = (*supporting, canonical)
        result = (out, canonical["resource_id"])
        cache[id(record)] = result
        return result

    @classmethod
    def _generated_resource_id(
        cls,
        record_type: str,
    ) -> str:
        """Return a generated local resource id for a canonical record."""
        prefix = get_config().inventory_resource_id_prefix
        type_slug = _inventory_type_slug(record_type)
        return f"{prefix}/{type_slug}/{uuid4().hex}"

    @classmethod
    def _canonicalize_mapping(cls, record: Mapping[str, Any]) -> dict[str, Any]:
        """Validate and normalize one canonical id-only record mapping."""
        data = dict(record)
        if not data.get("resource_id"):
            msg = "Inventory records must include a non-empty resource_id."
            raise ValueError(msg)
        record_type = str(data.get("type", ""))
        if not record_type:
            msg = "Inventory records must include a type."
            raise ValueError(msg)
        record_cls = cls._record_cls(record_type)
        relationships = cls._relationships_for(record_cls)
        object_fields = tuple(
            relationship.field
            for relationship in relationships
            if relationship.field in data
        )
        if object_fields:
            fields = ", ".join(repr(field) for field in object_fields)
            msg = (
                "Canonical inventory record mappings must use id relationship "
                f"fields, not object relationship fields: {fields}."
            )
            raise ValueError(msg)
        validate_data = dict(data)
        for relationship in relationships:
            validate_data.pop(relationship.storage_field, None)
        validate_data.pop("type", None)
        record_cls.model_validate(validate_data)
        out = record_cls.model_validate(validate_data).model_dump(
            mode="json", exclude_none=True
        )
        out["type"] = record_type
        for relationship in relationships:
            out.pop(relationship.field, None)
            if relationship.many:
                values = data.get(relationship.storage_field, ())
                if values:
                    out[relationship.storage_field] = tuple(
                        str(value) for value in values
                    )
            elif value := data.get(relationship.storage_field, ""):
                out[relationship.storage_field] = str(value)
        return out

    @staticmethod
    def _record_cache_key(record: Mapping[str, Any]) -> str:
        """Return a stable key for one canonical record mapping."""
        return json.dumps(record, sort_keys=True, default=str)

    @staticmethod
    def _time_key(value) -> str | None:
        """Return a comparable key for a datetime-like value."""
        value = to_datetime64(value)
        return None if pd.isnull(value) else str(value)

    @staticmethod
    def _record_start_key(record: Mapping[str, Any]) -> str | None:
        """Return the validity start key for a canonical record."""
        return Inventory._time_key(record.get("start_time", None))

    @staticmethod
    def _is_valid_at(record: Mapping[str, Any], time) -> bool:
        """Return True if a canonical record is valid at a supplied time."""
        time = to_datetime64(time)
        if pd.isnull(time):
            return True
        start = to_datetime64(record.get("start_time", None))
        end = to_datetime64(record.get("end_time", None))
        after_start = pd.isnull(start) or start <= time
        before_end = pd.isnull(end) or time < end
        return after_start and before_end

    @staticmethod
    def _select_current_record(records: tuple[dict[str, Any], ...], time=None):
        """Return the current record from one resource lineage."""
        if not records:
            return None
        if pd.isnull(to_datetime64(time)):
            return records[-1]
        for record in reversed(records):
            if Inventory._is_valid_at(record, time):
                return record
        return None

    def _validate_resource_ids(self) -> None:
        """Ensure resource ids identify exactly one record type."""
        types = {}
        for record in self._records:
            resource_id = record["resource_id"]
            record_type = record["type"]
            if (old_type := types.get(resource_id)) is None:
                types[resource_id] = record_type
            elif old_type != record_type:
                msg = (
                    f"Inventory resource_id {resource_id!r} is used for both "
                    f"{old_type!r} and {record_type!r}."
                )
                raise ValueError(msg)

    def _current_record(
        self,
        resource_id: str,
        time=None,
        *,
        record_type: str
        | type[_InventoryRecord]
        | tuple[type[_InventoryRecord], ...]
        | None = None,
    ) -> dict[str, Any]:
        """Return the current canonical record for one resource id."""
        type_filters = self._normalize_filter(record_type)
        type_names = (
            None
            if type_filters is None
            else {self._type_name(record_type) for record_type in type_filters}
        )
        all_records = tuple(
            record for record in self._records if record["resource_id"] == resource_id
        )
        records = tuple(
            record
            for record in all_records
            if type_names is None or record["type"] in type_names
        )
        if not records:
            msg = f"No inventory record found for resource_id {resource_id!r}."
            if record_type is not None:
                msg = f"{msg} and record_type {record_type!r}."
            raise KeyError(msg)
        out = self._select_current_record(records, time)
        if out is None:
            msg = (
                f"No inventory record found for resource_id {resource_id!r} "
                f"valid at {to_datetime64(time)!r}."
            )
            raise KeyError(msg)
        return out

    def get_records(
        self,
        record_ids=None,
        *,
        record_types=None,
        time=None,
    ) -> tuple[_InventoryRecord, ...]:
        """Return resolved records filtered by id, type, and time."""
        records = self.get_record_dicts(
            record_ids=record_ids,
            record_types=record_types,
            time=time,
        )
        cache = {}
        return tuple(
            self._resolve_record(record, time=time, cache=cache) for record in records
        )

    def get_record_dicts(
        self,
        record_ids=None,
        *,
        record_types=None,
        time=None,
        include_history: bool = False,
    ) -> tuple[dict[str, Any], ...]:
        """Return canonical id-only records filtered by id, type, and time."""
        if include_history and time is not None:
            msg = "time and include_history=True are mutually exclusive."
            raise ParameterError(msg)
        ids = self._normalize_filter(record_ids)
        type_filters = self._normalize_filter(record_types)
        type_names = (
            None
            if type_filters is None
            else {self._type_name(record_type) for record_type in type_filters}
        )
        records = tuple(
            record
            for record in self._records
            if (ids is None or record["resource_id"] in ids)
            and (type_names is None or record["type"] in type_names)
        )
        if not include_history:
            grouped = defaultdict(list)
            for record in records:
                grouped[record["resource_id"]].append(record)
            selected = []
            for resource_id, values in grouped.items():
                current = self._select_current_record(tuple(values), time)
                if current is None:
                    if ids is not None:
                        msg = (
                            f"No inventory record found for resource_id "
                            f"{resource_id!r} valid at {to_datetime64(time)!r}."
                        )
                        raise KeyError(msg)
                    continue
                selected.append(current)
            records = tuple(selected)
        return tuple(dict(record) for record in records)

    def _validate_references(self, time=None) -> Self:
        """Validate that inventory id references point to expected record types."""
        time = to_datetime64(time)
        errors: list[str] = []

        def _check(
            owner: _InventoryRecord,
            field: str,
            resource_id: str,
            expected_type: type[_InventoryRecord] | tuple[type[_InventoryRecord], ...],
            time=None,
        ) -> None:
            """Record an error if a referenced resource is missing or mistyped."""
            if not resource_id:
                return
            try:
                current = self._current_record(
                    resource_id,
                    time=time,
                    record_type=expected_type,
                )
                record = self._storage_object(current)
            except KeyError:
                errors.append(
                    f"{owner.type} {owner.resource_id!r} field {field!r} "
                    f"references missing resource_id {resource_id!r}."
                )
                return
            if not isinstance(record, expected_type):
                expected = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else ", ".join(cls.__name__ for cls in expected_type)
                )
                errors.append(
                    f"{owner.type} {owner.resource_id!r} field {field!r} "
                    f"references {record.type} {resource_id!r}; expected {expected}."
                )

        records = (
            self.get_record_dicts(time=time)
            if not pd.isnull(time)
            else self.get_record_dicts(include_history=True)
        )
        for obj in records:
            record_time = to_datetime64(obj.get("start_time", None))
            check_time = time if not pd.isnull(time) else record_time
            if pd.isnull(check_time):
                check_time = None
            owner = self._storage_object(obj)
            for relationship in self._relationships_for(owner.__class__):
                if relationship.many:
                    for resource_id in obj.get(relationship.storage_field, ()):
                        _check(
                            owner,
                            relationship.storage_field,
                            resource_id,
                            relationship.target,
                            time=check_time,
                        )
                else:
                    _check(
                        owner,
                        relationship.storage_field,
                        obj.get(relationship.storage_field, ""),
                        relationship.target,
                        time=check_time,
                    )
        if errors:
            joined = "\n".join(errors)
            msg = f"Inventory reference validation failed:\n{joined}"
            raise ValueError(msg)
        return self

    def _storage_object(self, record: Mapping[str, Any]) -> _InventoryRecord:
        """Return unresolved pydantic object for a canonical record."""
        data = dict(record)
        record_type = data.pop("type")
        record_cls = self._record_cls(record_type)
        for relationship in self._relationships_for(record_cls):
            data.pop(relationship.storage_field, None)
        return record_cls.model_validate(data)

    def _resolve_reference(self, resource_id: str, time, record_type=None, cache=None):
        """Resolve one required reference."""
        if not resource_id:
            return None
        record = self._current_record(resource_id, time=time, record_type=record_type)
        return self._resolve_record(record, time=time, cache=cache)

    def _resolve_record(
        self,
        record: Mapping[str, Any],
        time=None,
        cache=None,
    ) -> _InventoryRecord:
        """Return a resolved pydantic object from a canonical record."""
        if cache is None:
            cache = {}
        key = self._record_cache_key(record)
        if key in cache:
            return cache[key]
        record_time = to_datetime64(record.get("start_time", None))
        resolve_time = time if not pd.isnull(to_datetime64(time)) else record_time
        if pd.isnull(resolve_time):
            resolve_time = None
        data = dict(record)
        record_type = data.pop("type")
        record_cls = self._record_cls(record_type)
        for relationship in self._relationships_for(record_cls):
            if relationship.many:
                resource_ids = data.pop(relationship.storage_field, ())
                data[relationship.field] = tuple(
                    self._resolve_reference(
                        resource_id,
                        resolve_time,
                        relationship.target,
                        cache=cache,
                    )
                    for resource_id in resource_ids
                )
                continue
            resource_id = data.pop(relationship.storage_field, "")
            if resource_id:
                data[relationship.field] = self._resolve_reference(
                    resource_id,
                    resolve_time,
                    relationship.target,
                    cache=cache,
                )
        out = record_cls.model_validate(data)
        cache[key] = out
        return out

    def put_records(
        self,
        records,
        *,
        author_id: str | None = None,
        comment: str | None = None,
    ) -> Self:
        """
        Return a new inventory with canonicalized records inserted.

        Records may be canonical id-only mappings or resolved inventory objects.
        Linked inventory objects are recursively extracted and stored as canonical
        records before the object that references them. A record replaces an
        existing epoch when ``resource_id`` and ``start_time`` both match.
        """
        if isinstance(records, _InventoryRecord) or isinstance(records, Mapping):
            msg = (
                "records must be a sequence of inventory record instances or mappings."
            )
            raise ParameterError(msg)
        incoming = list(self._canonicalize_many(records))
        current = list(self.records)

        def _find_matching_epoch(record: Mapping[str, Any]) -> int | None:
            """Return index of matching resource/start epoch, if present."""
            start_key = self._record_start_key(record)
            for index, old in enumerate(current):
                if old["resource_id"] != record["resource_id"]:
                    continue
                if old["type"] != record["type"]:
                    msg = (
                        f"Inventory resource_id {record['resource_id']!r} is "
                        f"already used for {old['type']!r}, not {record['type']!r}."
                    )
                    raise ValueError(msg)
                if self._record_start_key(old) == start_key:
                    return index
            return None

        for record in incoming:
            if author_id is not None:
                record = {**record, "author_id": author_id}
            if comment is not None:
                record = {**record, "comment": comment}
            record = self._canonicalize_mapping(record)
            if (index := _find_matching_epoch(record)) is None:
                current.append(record)
            elif current[index] != record:
                current[index] = record
        return self.new(records=tuple(current))

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Load an inventory from a canonical manifest mapping."""
        return cls.model_validate(data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible canonical manifest mapping."""
        out = self.model_dump(mode="json", exclude_none=True)
        out["records"] = [dict(record) for record in self._records]
        return out

    @classmethod
    def from_json(cls, source: path_types | Any) -> Self:
        """Load an inventory from a JSON path, stream, or string."""
        if hasattr(source, "read"):
            data = json.load(source)
        elif isinstance(source, str | bytes) and str(source).lstrip().startswith("{"):
            data = json.loads(source)
        else:
            with Path(source).open() as fi:
                data = json.load(fi)
        return cls.from_dict(data)

    def to_json(self, destination: path_types | Any | None = None) -> str | None:
        """Write inventory to JSON, or return JSON string if no destination."""
        contents = json.dumps(self.to_dict(), indent=2)
        if destination is None:
            return contents
        if hasattr(destination, "write"):
            destination.write(contents)
        else:
            Path(destination).write_text(contents)
        return None

    @classmethod
    def from_yaml(cls, source: path_types | Any) -> Self:
        """Load an inventory from a YAML path or readable stream."""
        if hasattr(source, "read"):
            data = yaml.safe_load(source) or {}
        else:
            with Path(source).open() as fi:
                data = yaml.safe_load(fi) or {}
        return cls.from_dict(data)

    def to_yaml(self, destination: path_types | Any | None = None) -> str | None:
        """Write inventory to YAML, or return a YAML string if no destination."""
        contents = yaml.safe_dump(
            self.to_dict(),
            sort_keys=False,
            allow_unicode=True,
        )
        if destination is None:
            return contents
        if hasattr(destination, "write"):
            destination.write(contents)
        else:
            Path(destination).write_text(contents)
        return None
