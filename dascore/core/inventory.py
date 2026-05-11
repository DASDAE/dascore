"""
DASCore's metadata model.

Inventory is intended to be an extension of the stationxml concept
that provides first class supoprt to fiberoptic-based arrays. It provides the
necesisary abstractions to describe:

- The physical optical path including fiber types and connectors
- The coupling conditions and annotations of the system
- The interrogator configuration that produced DAS patches

In addition to being a metadata store, it naturally extends other dascore
patch interactions to include such metadata.

The public Inventory shape is intentionally small: document metadata, a flat
``resources`` mapping for shareable objects, and ``networks`` as the root
container.

DASCore keeps a private canonical index internally so the serialized
JSON/YAML manifest can stay nested and ergonomic without exposing tree addresses.
The tree addresses, however, enable ergonomic updates of tree components.
"""
# ruff: noqa: E501

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, TypeAlias, Literal

import numpy as np
import pandas as pd
from pydantic import ConfigDict, Field, PrivateAttr, field_validator, model_validator
from typing_extensions import Self

from dascore.constants import (
    VALID_DATA_TYPES,
)
from dascore.exceptions import ParameterError
from dascore.utils.inventory import (
    _IntervalTrackName,
    _merge_annotations,
    _PublicInventoryItem,
    _reverse_annotations,
    _select_annotations, _OpticalLengthItem,
    _trim_sequence,
)
from dascore.utils.misc import sanitize_range_param
from dascore.utils.models import DascoreBaseModel, DateTime64, UnitQuantity, CommentableModel, TimeRangedModel
from dascore.core.annotations import OpticalPathAnnotation
from dascore.exceptions import InvalidInventoryError

ExtraFieldValue: TypeAlias = str | int | float | bool

COUPLINGTYPES = Literal["conduit", "trench", "outside borehole casing", "clamp", "wireline", ""]


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


class FiberArray(TimeRangedModel):
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


class Network(CommentableModel):
    """FDSN-like organizational container for inventory observing objects."""

    code: str = Field(default="", description="Network code.")
    name: str = Field(default="", description="Human-readable network name.")
    description: str = Field(default="", description="Network description.")
    stations: tuple[FiberArray, ...] = Field(
        default=(),  # Todo: add stationxml-like station object later.
        description="Fiber arrays or stations associated with this network.",
    )


class Interrogator(_PublicInventoryItem):
    """DAS interrogator unit used for data collection."""

    manufacturer: str = Field(
        default="", description="Manufacturer name of the interrogator."
    )
    model: str = Field(default="", description="Model number of the interrogator.")
    serial_number: str = Field(
        default="", description="Serial number of the interrogator."
    )
    instrument_type: str = Field(
        default="interrogator",
        description="General instrument category.",
    )
    lead_cable_length: float = Field(
        default=0,
        description=
        "The length of lead cable for the interrogator. If the interrogator"
        " internally corrects for this in the patch output files, set to 0."
    )
    take_in_cable_length: float = Field(
        default=0,
        description=
        "The length of take-in cable for the interrogator. If the interrogator"
        " internally corrects for this in the patch output files, set to 0. "
        "Only applicable for closed loop systems."
    )


class Acquisition(TimeRangedModel):
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
    data_type: Literal[VALID_DATA_TYPES] = Field(
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
    firmware_version: str = Field(
        default="",
        description="Firmware version.",
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
    closed_fiber_loop: bool = Field(
        default=False,
        description=(
            "If true, the interrogator is attached to both ends of the "
            "optical path, otherwise just the start."
        )
    )
    extra_fields: dict[str, ExtraFieldValue] = Field(
        default_factory=dict,
        description=(
            "Extra acquisition metadata not represented by standardized fields. "
            "This maps to FDSN DAS Metadata native_headers when exporting."
        ),
    )


class ExternalResource(_PublicInventoryItem):
    """External resource identified but not otherwise modeled by DASCore."""

    uri: str = Field(default="", description="URI or identifier for the resource.")
    name: str = Field(default="", description="Human-readable resource name.")
    description: str = Field(
        default="", description="Free-form description of the external resource."
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

    optical_length: float = Field(
        default=0,
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
    """
    Geometry for an interval of an optical path.

    Simple linear interpolation is assumed between points. The labels of the
    coordinate column are defined by the CRS (coordinate reference system) of
    the OpticalPath.
    """

    name: str = Field(default="", description="Human-readable geometry name.")
    distance: tuple[float] = Field(
        default=(), description="Optical distance which pairs to coordinates."
    )
    coordinates: tuple[tuple[float, ...], ...] = Field(
        default=(), description="Coordinate points describing the geometry."
    )

    @property
    def optical_length(self) -> float:
        """The total optical length in meters."""
        args = self.optical_length if self.optical_length else [0]
        return np.max(args)


class CouplingCondition(_OpticalLengthItem):
    """
    Acoustic coupling condition for an interval of an optical path.

    Can represent many different types of coupling conditions, and point anchoring.
    """

    optical_length: float = Field(
        default=0,
        description="Optical path interval length described by this coupling in meters.",
    )
    coupling_type: COUPLINGTYPES = Field(
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


class OpticalPath(TimeRangedModel):
    """Continuous optical path described by independent component sequences."""

    name: str = Field(default="", description="Human-readable optical path name.")
    crs: CoordinateReferenceSystem = Field(
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
    otdr_traces: tuple[ExternalResource, ...] = Field(
        default=(),
        description="Reference to OTDR traces of this optical path.",
    )

    _length_attrs: ClassVar[tuple[str]] = (
        "optical_components",
        "geometries",
        "coupling_conditions",
    )

    @property
    def optical_length(self):
        return sum([x.optical_length for x in self.optical_components])

    def __add__(self, other):
        """Concatenate two optical paths."""
        if not isinstance(other, OpticalPath):
            return NotImplemented
        if self.crs != other.crs:
            msg = "Cannot concatenate optical paths with different CRS definitions."
            raise ParameterError(msg)
        annotations = _merge_annotations(
            self.annotations,
            other.annotations,
            self.optical_length,
            other.optical_length,
        )
        out = self.__class__(
            crs=self.crs,
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
        return NotImplemented

    def reverse(self) -> Self:
        """Return a new optical path with all ordered subcomponents reversed."""
        annotations = _reverse_annotations(self.annotations, self.optical_length)
        out = self.__class__(
            crs=self.crs,
            optical_components=tuple(reversed(self.optical_components)),
            geometries=tuple(reversed(self.geometries)),
            coupling_conditions=tuple(reversed(self.coupling_conditions)),
            annotations=annotations,
        )
        object.__setattr__(out, "_inventory_id", self.inventory_id)
        return out

    def select(self, *, distance: tuple[float | None, float | None]) -> Self:
        """
        Return a new optical path selected by distance.

        Parameters
        ----------
        distance
            A tuple indicating the distance values to select.
        """
        start, stop = sanitize_range_param(distance)
        optical_components = _trim_sequence(distance, self.optical_components)
        geometries = _trim_sequence(distance, self.geometries)
        coupling_conditions = _trim_sequence(distance, self.coupling_conditions)
        annotations = _select_annotations(
            self.annotations,
            start,
            stop,
            length=self.optical_length,
        )
        out = self.__class__(
            crs=self.crs,
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

    def validate(self, tolerance: float = 1e-9) -> Self:
        """Validate that populated interval sequences match path optical length."""
        expected_length = self.optical_length
        errors = []
        # first check sum of length-like components.
        for attr in self._length_attrs[1:]:
            comp_tuple = getattr(self, attr)
            length = np.sum([x.optical_length for x in comp_tuple])
            if not np.isclose(length, expected_length):
                errors.append(
                    f"Optical_path component {attr} has a total length of: "
                    f"{length} but the optical path has a length of {expected_length}."
                )
        # then check annotation ranges
        for anna in self.annotations:
            vals = np.array([
                x for x in sanitize_range_param(anna.distance)
                if not pd.isna(x)
            ])
            if np.any(vals > expected_length):
                errors.append(
                    f"Annotation {anna} has values outside length of {expected_length}."
                )
        if errors:
            msg = "Optical path validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        return self


class Inventory(DascoreBaseModel):
    """
    A fiber and seismic (FAS) inventory.

    See the [inventory tutorial](/tutorial/inventory.qmd) for examples of
    enriching patches from inventories and the
    [inventory design note](/notes/inventory_design.qmd) for model rationale.
    """
    format_name: ClassVar[str] = "fas_inventory"

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

    def replace(self, component) -> Self:
        """Replace a component in the inventory and return a new Inventory."""
