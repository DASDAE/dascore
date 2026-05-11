"""
DASCore's metadata model.

Inventory is intended to be a metadata catalog for describing patches and
related fiber array context. Patch data, coordinates, and attrs remain the
primary source of truth, while Inventory objects provide richer context which
can be resolved explicitly by fiber array id and time.
Moreover, Patches can defer to specific fiber arrays for their metadata,
enabling the decoupling of metadata from data for DAS archives.

The public Inventory shape is intentionally small: document metadata, a flat
``resources`` mapping for shareable objects, and ``networks`` as the root
container. DASCore keeps a private canonical index internally so the serialized
JSON/YAML manifest can stay nested and ergonomic without exposing tree addresses.

Resolved Pydantic objects are materialized views over that storage. A user can
take an object from ``resources`` or ``networks``, call ``.revise(...)``, and
write it back with ``Inventory.replace(...)``. Replacement preserves the
inventory's internal identity and increments the inventory content version in
``creation_info``.


## user stories
  1. As a DASCore user with raw or lightly processed patches, I want to attach a data_source_id to a patch so that later processing can resolve the correct inventory context.
  2. As a user with channel-indexed DAS data, I want to derive optical distance from an inventory acquisition so that geometry and fiber metadata can be projected onto the patch.
  3. As a user with an inventory and a distance-indexed patch, I want to add selected optical-path metadata as patch coordinates so that I can process, select, and visualize data by labels, geometry, coupling, or fiber
     properties.
  4. As a user, I want to add selected fiber array, acquisition, and interrogator fields to patch attrs so that downstream workflows can carry compact metadata without embedding the whole inventory.
  5. As a user managing evolving deployments, I want inventory objects to be valid over time so that the same data_source_id can resolve to different configurations or paths for different data epochs.
  6. As a user building an inventory, I want to write linked Pydantic objects while DASCore stores stable internal identities so that inventories are ergonomic in Python and durable in JSON/YAML.
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
    _InventoryItem,
    _merge_annotations,
    _PublicInventoryItem,
    _reverse_annotations,
    _select_annotations,
    _TimedInventoryItem,
)
from dascore.utils.misc import sanitize_range_param
from dascore.utils.models import DascoreBaseModel, DateTime64, UnitQuantity
from dascore.utils.time import to_datetime64

ExtraFieldValue: TypeAlias = str | int | float | bool


class CreationInfo(DascoreBaseModel):
    """QuakeML-style provenance for an inventory document."""

    model_config = ConfigDict(
        title="CreationInfo",
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    agency_id: str = Field(
        default="",
        description="Identifier for the agency responsible for the inventory.",
    )
    author: str = Field(
        default="",
        description="Author or process responsible for the inventory.",
    )
    creation_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Time this inventory was originally created.",
    )
    update_time: DateTime64 = Field(
        default=np.datetime64("NaT"),
        description="Time this inventory was last updated.",
    )
    version: str = Field(
        default="",
        description="Content version of this inventory document.",
    )


def _inventory_type_slug(item_type: str) -> str:
    """Return a compact snake_case slug for one inventory item type."""
    chars = []
    for index, char in enumerate(item_type):
        if char.isupper() and index:
            chars.append("_")
        chars.append(char.lower())
    return "".join(chars)


class FiberArray(_TimedInventoryItem):
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


class Network(_InventoryItem):
    """FDSN-like organizational container for inventory observing objects."""

    code: str = Field(default="", description="Network code.")
    name: str = Field(default="", description="Human-readable network name.")
    description: str = Field(default="", description="Network description.")
    fiber_arrays: tuple[FiberArray, ...] = Field(
        default=(),
        description="Fiber arrays associated with this network.",
    )
    stations: tuple[None, ...] = ()  # just a stub for now


class Interrogator(_PublicInventoryItem):
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


class Acquisition(_TimedInventoryItem):
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
            "items within a fiber array. Empty location codes are allowed."
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


class ExternalResource(_PublicInventoryItem):
    """External resource identified but not otherwise modeled by DASCore."""

    uri: str = Field(default="", description="URI or identifier for the resource.")
    name: str = Field(default="", description="Human-readable resource name.")
    description: str = Field(
        default="", description="Free-form description of the external resource."
    )


class _OpticalLengthItem(_InventoryItem):
    """Base class for optical path items with SI-only optical lengths."""

    model_config = ConfigDict(extra="forbid")

    optical_length: float | None = Field(
        default=None,
        description="Optical path interval length in meters.",
    )


class Cable(_PublicInventoryItem):
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


class Enclosure(_PublicInventoryItem):
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


class OpticalComponent(_OpticalLengthItem):
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


class CoordinateReferenceSystem(_PublicInventoryItem):
    """
    Coordinate reference system used by optical path geometries.

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


class Geometry(_OpticalLengthItem):
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


class CouplingCondition(_OpticalLengthItem):
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


class OpticalPathAnnotation(_InventoryItem):
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


class OpticalPath(_TimedInventoryItem):
    """Continuous optical path described by independent component sequences."""

    optical_length: float | None = Field(
        default=None, description="Declared total optical path length in meters."
    )
    name: str = Field(default="", description="Human-readable optical path name.")
    coordinate_reference_system: CoordinateReferenceSystem | None = Field(
        default_factory=_default_coordinate_reference_system,
        description="Coordinate reference system shared by geometries on this path.",
    )
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

    @staticmethod
    def _check_compatible_crs(
        left: CoordinateReferenceSystem | None,
        right: CoordinateReferenceSystem | None,
    ) -> CoordinateReferenceSystem | None:
        """Return the shared CRS or raise for incompatible path concatenation."""
        if left is None:
            return right
        if right is None:
            return left
        if left != right:
            msg = "Cannot concatenate optical paths with different CRS definitions."
            raise ParameterError(msg)
        return left

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
                    items=self.optical_components,
                )
            case "geometry":
                return _IntervalSequence(
                    name,
                    items=self.geometries,
                )
            case "coupling_condition":
                return _IntervalSequence(
                    name,
                    items=self.coupling_conditions,
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
        crs = self._check_compatible_crs(
            self.coordinate_reference_system,
            other.coordinate_reference_system,
        )
        annotations = _merge_annotations(
            self.annotations,
            other.annotations,
            self.optical_length,
            other.optical_length,
        )
        out = self.__class__(
            optical_length=self._add_lengths(self.optical_length, other.optical_length),
            coordinate_reference_system=crs,
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
        return out

    def __radd__(self, other):
        """Support summing optical path sequences."""
        if other == 0:
            return self
        return NotImplemented

    def reverse(self) -> Self:
        """Return a new optical path with all ordered subcomponents reversed."""
        annotations = _reverse_annotations(self.annotations, self.optical_length)
        out = self.__class__(
            optical_length=self.optical_length,
            coordinate_reference_system=self.coordinate_reference_system,
            optical_components=tuple(reversed(self.optical_components)),
            geometries=tuple(reversed(self.geometries)),
            coupling_conditions=tuple(reversed(self.coupling_conditions)),
            annotations=annotations,
        )
        object.__setattr__(out, "_inventory_id", self.inventory_id)
        return out

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
        out = self.__class__(
            optical_length=stop - start,
            coordinate_reference_system=self.coordinate_reference_system,
            optical_components=optical_components,
            geometries=geometries,
            coupling_conditions=coupling_conditions,
            annotations=annotations,
        )
        object.__setattr__(out, "_inventory_id", self.inventory_id)
        return out

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


INVENTORY_ITEM_TYPES: dict[str, type[_InventoryItem]] = {
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

PUBLIC_ITEM_TYPES = (
    Cable,
    CoordinateReferenceSystem,
    Enclosure,
    ExternalResource,
    Interrogator,
)

ROOT_ITEM_TYPES = (Network,)


@dataclass(frozen=True)
class _Relationship:
    """Inventory relationship between an object field and a canonical id field."""

    owner: type[_InventoryItem]
    field: str
    storage_field: str
    target: type[_InventoryItem] | tuple[type[_InventoryItem], ...]
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
        OpticalPath,
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
    A time-aware catalog of FAS Inventory metadata objects.

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

    format_name: ClassVar[str] = "fas_inventory"
    get_summary_df: ClassVar[None] = None

    schema_version: int = Field(
        default=1,
        description=(
            "Version of the FAS Inventory manifest/envelope schema; used for "
            "manifest migration."
        ),
    )
    resource_id: str = Field(
        default="", description="Optional identifier for this inventory manifest."
    )
    creation_info: CreationInfo = Field(
        default_factory=CreationInfo,
        description="QuakeML-style creation and update metadata.",
    )
    comment: str = Field(default="", description="Additional comments.")
    resources: dict[str, _PublicInventoryItem] = Field(
        default_factory=dict,
        description="Shareable resources keyed by resource_id.",
    )
    networks: tuple[Network, ...] = Field(
        default=(),
        description="Network containers in this inventory.",
    )
    _items: tuple[dict[str, Any], ...] = PrivateAttr(default_factory=tuple)

    def __init__(self, **data):
        """Create an inventory and store a private canonical index."""
        indexed_items = data.pop("_indexed_items", None)
        resources = self._normalize_resources(data.pop("resources", {}))
        networks = tuple(data.pop("networks", ()))
        items = (*resources, *networks)
        data["resources"] = {}
        data["networks"] = ()
        super().__init__(**data)
        if indexed_items is None:
            indexed_items = self._canonicalize_many(items)
        object.__setattr__(self, "_items", tuple(indexed_items))
        self._validate_ids()
        self._validate_references()
        self._set_public_containers()

    @classmethod
    def _normalize_resources(cls, resources) -> tuple[_PublicInventoryItem, ...]:
        """Return resources from a public resource_id mapping."""
        if resources is None:
            return ()
        if not isinstance(resources, Mapping):
            msg = "resources must be a mapping of resource_id to resource object."
            raise ParameterError(msg)
        out = []
        for resource_id, resource in resources.items():
            resource_id = str(resource_id)
            if isinstance(resource, Mapping):
                data = dict(resource)
                if data.get("resource_id", resource_id) != resource_id:
                    msg = "Resource mapping keys must match payload resource_id values."
                    raise ValueError(msg)
                item_type = data.pop("type")
                data["resource_id"] = resource_id
                resource = cls._item_cls(item_type).model_validate(data)
            if not isinstance(resource, _PublicInventoryItem):
                msg = "Inventory resources must be public resource objects."
                raise ParameterError(msg)
            if not resource.resource_id:
                resource = resource.model_copy(update={"resource_id": resource_id})
            elif resource.resource_id != resource_id:
                msg = "Resource mapping keys must match resource.resource_id values."
                raise ValueError(msg)
            out.append(resource)
        return tuple(out)

    def _set_public_containers(self) -> None:
        """Set public resources and networks from the private canonical index."""
        cache = {}
        resources = {}
        for item in self._items:
            item_cls = self._item_cls(item["type"])
            if item_cls in PUBLIC_ITEM_TYPES:
                resource = self._resolve_item(item, cache=cache)
                resources[resource.resource_id] = resource
        networks = tuple(
            self._resolve_item(item, cache=cache)
            for item in self._items
            if item["type"] == "Network"
        )
        object.__setattr__(self, "resources", resources)
        object.__setattr__(self, "networks", networks)

    @classmethod
    def _relationships_for(cls, item_cls: type[_InventoryItem]):
        """Return relationship specs for an item class."""
        return tuple(
            rel for rel in INVENTORY_RELATIONSHIPS if issubclass(item_cls, rel.owner)
        )

    @property
    def _indexed_items(self) -> tuple[dict[str, Any], ...]:
        """Return copies of canonical id-only inventory items."""
        return tuple(dict(item) for item in self._items)

    def new(self, **kwargs) -> Self:
        """Create a new Inventory with updated attributes."""
        if "items" in kwargs:
            msg = "Use resources or networks to update an Inventory."
            raise ParameterError(msg)
        data = {
            "schema_version": self.schema_version,
            "resource_id": self.resource_id,
            "creation_info": self.creation_info,
            "comment": self.comment,
            "resources": self.resources,
            "networks": self.networks,
        }
        if "creation_info" not in kwargs:
            data["creation_info"] = self._increment_creation_info_version(
                self.creation_info
            )
        return self.__class__(**(data | kwargs))

    def replace(self, item: _InventoryItem) -> Self:
        """Return a new inventory with one existing item replaced."""
        if not isinstance(item, _InventoryItem):
            msg = "replace requires an inventory object."
            raise ParameterError(msg)
        if not isinstance(item, _PublicInventoryItem) and not item.inventory_id:
            msg = f"No inventory item found for identity {item.type!r}."
            raise KeyError(msg)
        incoming = list(self._canonicalize_many((item,)))
        current = list(self._indexed_items)
        target = incoming[-1]
        target_identity = self._item_identity(target)
        found_target = False
        for new_item in incoming:
            identity = self._item_identity(new_item)
            match = None
            for index, old_item in enumerate(current):
                if self._item_identity(old_item) != identity:
                    continue
                if old_item["type"] != new_item["type"]:
                    msg = (
                        f"Inventory identity {identity!r} is already used for "
                        f"{old_item['type']!r}, not {new_item['type']!r}."
                    )
                    raise ValueError(msg)
                match = index
                break
            if match is None:
                if identity == target_identity:
                    msg = f"No inventory item found for identity {identity!r}."
                    raise KeyError(msg)
                current.append(new_item)
                continue
            current[match] = new_item
            found_target = found_target or identity == target_identity
        if not found_target:
            msg = f"No inventory item found for identity {target_identity!r}."
            raise KeyError(msg)
        creation_info = self._increment_creation_info_version(self.creation_info)
        return self.__class__(
            schema_version=self.schema_version,
            resource_id=self.resource_id,
            creation_info=creation_info,
            comment=self.comment,
            _indexed_items=tuple(current),
        )

    @staticmethod
    def _increment_creation_info_version(creation_info: CreationInfo) -> CreationInfo:
        """Return creation info with integer-like content version incremented."""
        version = creation_info.version
        if version == "":
            version = "1"
        else:
            try:
                version = str(int(version) + 1)
            except ValueError:
                return creation_info
        return creation_info.model_copy(update={"version": version})

    @staticmethod
    def _normalize_filter(value) -> tuple | None:
        """Normalize scalar or sequence filters."""
        if value is None:
            return None
        if isinstance(value, str | type):
            return (value,)
        return tuple(value)

    @staticmethod
    def _type_name(item_type) -> str:
        """Return canonical item type name."""
        return item_type if isinstance(item_type, str) else item_type.__name__

    @classmethod
    def _item_cls(cls, item_type: str) -> type[_InventoryItem]:
        """Return item class for a canonical item type."""
        try:
            return INVENTORY_ITEM_TYPES[item_type]
        except KeyError:
            valid = sorted(INVENTORY_ITEM_TYPES)
            msg = (
                f"Unknown inventory object type {item_type!r}. Valid types are {valid}."
            )
            raise ValueError(msg) from None

    @classmethod
    def _canonicalize_many(cls, items) -> tuple[dict[str, Any], ...]:
        """Return canonical items for mappings or inventory item objects."""
        out = []
        seen = set()
        cache = {}
        for index, item in enumerate(items):
            path = cls._root_path(item, index)
            for item in cls._canonicalize_item(item, cache=cache, path=path):
                key = cls._item_cache_key(item)
                if key not in seen:
                    out.append(item)
                    seen.add(key)
        return tuple(out)

    @classmethod
    def _root_path(cls, item, index: int) -> str | None:
        """Return the deterministic internal path for a top-level item."""
        item_type = item.get("type", "") if isinstance(item, Mapping) else item.type
        item_cls = cls._item_cls(str(item_type))
        if (
            not issubclass(item_cls, (*PUBLIC_ITEM_TYPES, *ROOT_ITEM_TYPES))
            and not (isinstance(item, Mapping) and item.get("id", ""))
            and not (not isinstance(item, Mapping) and item.inventory_id)
        ):
            return None
        type_slug = _inventory_type_slug(str(item_type) or "item")
        label = ""
        if isinstance(item, Mapping):
            label = str(item.get("code") or item.get("resource_id") or index)
        elif hasattr(item, "code") and getattr(item, "code"):
            label = str(getattr(item, "code"))
        elif hasattr(item, "resource_id") and getattr(item, "resource_id"):
            label = str(getattr(item, "resource_id"))
        else:
            label = str(index)
        return f"/{type_slug}/{label}"

    @classmethod
    def _canonicalize_item(
        cls,
        item,
        cache=None,
        path: str | None = None,
    ) -> tuple[dict[str, Any], ...]:
        """Return supporting items plus one canonical id-only item."""
        items, _ = cls._canonicalize_item_with_id(item, cache=cache, path=path)
        return items

    @classmethod
    def _canonicalize_item_with_id(
        cls,
        item,
        cache=None,
        path: str | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        """Return canonical items and the item id for one inventory object."""
        if isinstance(item, Mapping):
            if any(rel.field in item for rel in INVENTORY_RELATIONSHIPS):
                items, resource_id = cls._canonicalize_nested_mapping(item, cache, path)
                return items, resource_id
            out = cls._canonicalize_mapping(item, path=path)
            return (out,), out["id"]
        if not isinstance(item, _InventoryItem):
            msg = "items must contain inventory item instances or mappings."
            raise ParameterError(msg)
        if cache is None:
            cache = {}
        if id(item) in cache:
            return cache[id(item)]
        supporting = []
        data = item.model_dump(mode="json", exclude_none=True)
        data["type"] = item.type
        item_id = cls._get_item_id(item, data, path=path)
        data["id"] = item_id
        for relationship in cls._relationships_for(item.__class__):
            data.pop(relationship.field, None)
            if relationship.many:
                values = tuple(getattr(item, relationship.field))
                if values:
                    resource_ids = []
                    for index, value in enumerate(values):
                        child_path = f"{item_id}/{relationship.field}/{index}"
                        items, resource_id = cls._canonicalize_item_with_id(
                            value,
                            cache=cache,
                            path=child_path,
                        )
                        supporting.extend(items)
                        resource_ids.append(resource_id)
                    data[relationship.storage_field] = tuple(resource_ids)
            elif (value := getattr(item, relationship.field)) is not None:
                child_path = f"{item_id}/{relationship.field}"
                items, resource_id = cls._canonicalize_item_with_id(
                    value,
                    cache=cache,
                    path=child_path,
                )
                supporting.extend(items)
                data[relationship.storage_field] = resource_id
        canonical = cls._canonicalize_mapping(data)
        out = (*supporting, canonical)
        result = (out, canonical["id"])
        cache[id(item)] = result
        return result

    @classmethod
    def _canonicalize_nested_mapping(
        cls,
        item: Mapping[str, Any],
        cache=None,
        path: str | None = None,
    ) -> tuple[tuple[dict[str, Any], ...], str]:
        """Canonicalize a nested serialized mapping into flat runtime items."""
        data = dict(item)
        item_type = data.get("type", "")
        item_cls = cls._item_cls(item_type)
        item_id = cls._get_mapping_id(data, item_cls, path=path)
        data["id"] = item_id
        supporting = []
        for relationship in cls._relationships_for(item_cls):
            if relationship.many:
                values = tuple(data.pop(relationship.field, ()))
                if values:
                    ids = []
                    for index, value in enumerate(values):
                        child_path = f"{item_id}/{relationship.field}/{index}"
                        items, child_id = cls._canonicalize_item_with_id(
                            value,
                            cache=cache,
                            path=child_path,
                        )
                        supporting.extend(items)
                        ids.append(child_id)
                    data[relationship.storage_field] = tuple(ids)
            elif relationship.field in data:
                value = data.pop(relationship.field)
                if value is not None:
                    child_path = f"{item_id}/{relationship.field}"
                    items, child_id = cls._canonicalize_item_with_id(
                        value,
                        cache=cache,
                        path=child_path,
                    )
                    supporting.extend(items)
                    data[relationship.storage_field] = child_id
        canonical = cls._canonicalize_mapping(data, path=path)
        return (*supporting, canonical), canonical["id"]

    @classmethod
    def _get_item_id(
        cls,
        item: _InventoryItem,
        data: Mapping[str, Any],
        *,
        path: str | None,
    ) -> str:
        """Return public or runtime id for an item being canonicalized."""
        if item.inventory_id:
            return item.inventory_id
        if isinstance(item, _PublicInventoryItem):
            return data.get("resource_id") or cls._generated_resource_id(item.type)
        if path is None:
            msg = (
                f"{item.type} objects without public resource_id must be nested "
                "or come from an Inventory view before replacement."
            )
            raise ParameterError(msg)
        return path

    @classmethod
    def _get_mapping_id(
        cls,
        data: Mapping[str, Any],
        item_cls: type[_InventoryItem],
        *,
        path: str | None,
    ) -> str:
        """Return id for a canonical or nested mapping."""
        if value := data.get("id", ""):
            return str(value)
        if issubclass(item_cls, _PublicInventoryItem):
            return str(
                data.get("resource_id") or cls._generated_resource_id(data["type"])
            )
        if path is None:
            msg = (
                f"{data['type']} objects without public resource_id must be nested "
                "inside the inventory tree."
            )
            raise ParameterError(msg)
        return path

    @classmethod
    def _generated_resource_id(
        cls,
        item_type: str,
    ) -> str:
        """Return a generated local resource id for a canonical item."""
        prefix = get_config().inventory_resource_id_prefix
        type_slug = _inventory_type_slug(item_type)
        return f"{prefix}/{type_slug}/{uuid4().hex}"

    @classmethod
    def _canonicalize_mapping(
        cls,
        item: Mapping[str, Any],
        *,
        path: str | None = None,
    ) -> dict[str, Any]:
        """Validate and normalize one canonical id-only item mapping."""
        data = dict(item)
        item_type = str(data.get("type", ""))
        if not item_type:
            msg = "Inventory objects must include a type."
            raise ValueError(msg)
        item_cls = cls._item_cls(item_type)
        data["id"] = cls._get_mapping_id(data, item_cls, path=path)
        if issubclass(item_cls, _PublicInventoryItem) and not data.get("resource_id"):
            data["resource_id"] = data["id"]
        relationships = cls._relationships_for(item_cls)
        object_fields = tuple(
            relationship.field
            for relationship in relationships
            if relationship.field in data
        )
        if object_fields:
            fields = ", ".join(repr(field) for field in object_fields)
            msg = (
                "Canonical inventory item mappings must use id relationship "
                f"fields, not object relationship fields: {fields}."
            )
            raise ValueError(msg)
        validate_data = dict(data)
        validate_data.pop("id", None)
        for relationship in relationships:
            validate_data.pop(relationship.storage_field, None)
        validate_data.pop("type", None)
        item_cls.model_validate(validate_data)
        out = item_cls.model_validate(validate_data).model_dump(
            mode="json", exclude_none=True
        )
        out["id"] = data["id"]
        out["type"] = item_type
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
    def _item_cache_key(item: Mapping[str, Any]) -> str:
        """Return a stable key for one canonical item mapping."""
        return json.dumps(item, sort_keys=True, default=str)

    @staticmethod
    def _item_identity(item: Mapping[str, Any]) -> str:
        """Return public identity when present, otherwise runtime id."""
        return item.get("resource_id", "") or item["id"]

    @staticmethod
    def _is_valid_at(item: Mapping[str, Any], time) -> bool:
        """Return True if a canonical item is valid at a supplied time."""
        time = to_datetime64(time)
        if pd.isnull(time):
            return True
        start = to_datetime64(item.get("start_time", None))
        end = to_datetime64(item.get("end_time", None))
        after_start = pd.isnull(start) or start <= time
        before_end = pd.isnull(end) or time < end
        return after_start and before_end

    @staticmethod
    def _select_current_item(items: tuple[dict[str, Any], ...], time=None):
        """Return the current item from one resource lineage."""
        if not items:
            return None
        if pd.isnull(to_datetime64(time)):
            return items[-1]
        for item in reversed(items):
            if Inventory._is_valid_at(item, time):
                return item
        return None

    def _validate_ids(self) -> None:
        """Ensure runtime ids and public resource ids identify one item type."""
        ids = {}
        types = {}
        for item in self._items:
            item_id = item["id"]
            item_type = item["type"]
            if (old_type := ids.get(item_id)) is None:
                ids[item_id] = item_type
            elif old_type != item_type:
                msg = (
                    f"Inventory internal id {item_id!r} is used for both "
                    f"{old_type!r} and {item_type!r}."
                )
                raise ValueError(msg)
            resource_id = item.get("resource_id", "")
            if not resource_id:
                continue
            if (old_type := types.get(resource_id)) is None:
                types[resource_id] = item_type
            elif old_type != item_type:
                msg = (
                    f"Inventory resource_id {resource_id!r} is used for both "
                    f"{old_type!r} and {item_type!r}."
                )
                raise ValueError(msg)

    def _current_item(
        self,
        resource_id: str,
        time=None,
        *,
        item_type: str
        | type[_InventoryItem]
        | tuple[type[_InventoryItem], ...]
        | None = None,
    ) -> dict[str, Any]:
        """Return the current canonical item for one resource id."""
        type_filters = self._normalize_filter(item_type)
        type_names = (
            None
            if type_filters is None
            else {self._type_name(item_type) for item_type in type_filters}
        )
        all_items = tuple(
            item
            for item in self._items
            if item["id"] == resource_id or item.get("resource_id", "") == resource_id
        )
        items = tuple(
            item
            for item in all_items
            if type_names is None or item["type"] in type_names
        )
        if not items:
            msg = f"No inventory item found for resource_id {resource_id!r}."
            if item_type is not None:
                msg = f"{msg} and item_type {item_type!r}."
            raise KeyError(msg)
        out = self._select_current_item(items, time)
        if out is None:
            msg = (
                f"No inventory item found for resource_id {resource_id!r} "
                f"valid at {to_datetime64(time)!r}."
            )
            raise KeyError(msg)
        return out

    def _current_index(self, time=None) -> tuple[dict[str, Any], ...]:
        """Return current canonical items for each inventory identity."""
        grouped = defaultdict(list)
        for item in self._items:
            grouped[self._item_identity(item)].append(item)
        out = []
        for values in grouped.values():
            current = self._select_current_item(tuple(values), time)
            if current is not None:
                out.append(dict(current))
        return tuple(out)

    def _validate_references(self, time=None) -> Self:
        """Validate that inventory id references point to expected item types."""
        time = to_datetime64(time)
        errors: list[str] = []

        def _check(
            owner: _InventoryItem,
            field: str,
            resource_id: str,
            expected_type: type[_InventoryItem] | tuple[type[_InventoryItem], ...],
            time=None,
        ) -> None:
            """Add an error if a referenced resource is missing or mistyped."""
            if not resource_id:
                return
            try:
                current = self._current_item(
                    resource_id,
                    time=time,
                    item_type=expected_type,
                )
                item = self._storage_object(current)
            except KeyError:
                owner_id = getattr(owner, "resource_id", "") or owner.inventory_id
                errors.append(
                    f"{owner.type} {owner_id!r} field {field!r} "
                    f"references missing resource_id {resource_id!r}."
                )
                return
            if not isinstance(item, expected_type):
                owner_id = getattr(owner, "resource_id", "") or owner.inventory_id
                expected = (
                    expected_type.__name__
                    if isinstance(expected_type, type)
                    else ", ".join(cls.__name__ for cls in expected_type)
                )
                errors.append(
                    f"{owner.type} {owner_id!r} field {field!r} "
                    f"references {item.type} {resource_id!r}; expected {expected}."
                )

        items = self._current_index(time=time) if not pd.isnull(time) else self._items
        for obj in items:
            item_time = to_datetime64(obj.get("start_time", None))
            check_time = time if not pd.isnull(time) else item_time
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

    def _storage_object(self, item: Mapping[str, Any]) -> _InventoryItem:
        """Return unresolved pydantic object for a canonical item."""
        data = dict(item)
        item_id = data.pop("id")
        item_type = data.pop("type")
        item_cls = self._item_cls(item_type)
        for relationship in self._relationships_for(item_cls):
            data.pop(relationship.storage_field, None)
        out = item_cls.model_validate(data)
        object.__setattr__(out, "_inventory_id", item_id)
        return out

    def _resolve_reference(self, resource_id: str, time, item_type=None, cache=None):
        """Resolve one required reference."""
        if not resource_id:
            return None
        item = self._current_item(resource_id, time=time, item_type=item_type)
        return self._resolve_item(item, time=time, cache=cache)

    def _resolve_item(
        self,
        item: Mapping[str, Any],
        time=None,
        cache=None,
    ) -> _InventoryItem:
        """Return a resolved pydantic object from a canonical item."""
        if cache is None:
            cache = {}
        key = self._item_cache_key(item)
        if key in cache:
            return cache[key]
        item_time = to_datetime64(item.get("start_time", None))
        resolve_time = time if not pd.isnull(to_datetime64(time)) else item_time
        if pd.isnull(resolve_time):
            resolve_time = None
        data = dict(item)
        item_id = data.pop("id")
        item_type = data.pop("type")
        item_cls = self._item_cls(item_type)
        for relationship in self._relationships_for(item_cls):
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
        out = item_cls.model_validate(data)
        object.__setattr__(out, "_inventory_id", item_id)
        cache[key] = out
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Load an inventory from a FAS Inventory manifest mapping."""
        data = dict(data)
        format_name = data.pop("format", None)
        if format_name != cls.format_name:
            msg = (
                f"Inventory manifest format must be {cls.format_name!r}, "
                f"not {format_name!r}."
            )
            raise ValueError(msg)
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible nested FAS Inventory manifest mapping."""
        out = {
            "format": self.format_name,
            "schema_version": self.schema_version,
            "resource_id": self.resource_id,
            "creation_info": self.creation_info.model_dump(mode="json"),
            "comment": self.comment,
        }
        public_items = [
            self._serialize_item(item)
            for item in self._items
            if self._item_cls(item["type"]) in PUBLIC_ITEM_TYPES
        ]
        network_items = [
            self._serialize_item(item)
            for item in self._items
            if item["type"] == "Network"
        ]
        if public_items:
            out["resources"] = {
                item["resource_id"]: {
                    key: value for key, value in item.items() if key != "resource_id"
                }
                for item in public_items
            }
        out["networks"] = network_items
        return out

    def _serialize_item(self, item: Mapping[str, Any]) -> dict[str, Any]:
        """Return a nested serialized item without runtime ids."""
        data = dict(item)
        data.pop("id", None)
        item_cls = self._item_cls(data["type"])
        for relationship in self._relationships_for(item_cls):
            if relationship.many:
                resource_ids = data.pop(relationship.storage_field, ())
                if resource_ids:
                    data[relationship.field] = [
                        self._serialize_item(self._current_item(resource_id))
                        for resource_id in resource_ids
                    ]
                continue
            resource_id = data.get(relationship.storage_field, "")
            if not resource_id:
                continue
            target = relationship.target
            target_types = target if isinstance(target, tuple) else (target,)
            if all(issubclass(item, _PublicInventoryItem) for item in target_types):
                continue
            data.pop(relationship.storage_field, None)
            data[relationship.field] = self._serialize_item(
                self._current_item(resource_id)
            )
        return data

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
