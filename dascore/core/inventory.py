"""
DASDAE Inventory: DASCore's metadata model for DFOS observing systems.

The inventory extends the StationXML concept with first-class support for
fiber-optic arrays. It describes the physical optical path (fiber, connectors,
splices), the geometry, coupling, and annotation tracks along optical
distance, and the interrogator configurations (acquisitions) that produced
DAS patches. Patches carry a ``data_source_id``
(``network.fiber_array.location.acquisition``) which, together with time,
resolves against an inventory.

Key model rules (see the DASDAE inventory specification):

- Validity intervals are half-open ``[start, end)`` in UTC; an unset (NaT)
  end time means ongoing. The outermost endpoint of a coverage domain is
  included.
- No more than one ``OpticalPath`` per ``(FiberArray, location_code)`` is
  valid at a given time.
- Optical components are the tiling track: they cover the whole path exactly
  once. Geometry and coupling are function tracks: coverage may be partial
  (uncovered distance is undefined), overlap raises. Annotations overlap
  freely.
- Codes use letters, digits, and ``-``; ``.`` is the ``data_source_id``
  separator. All codes are non-empty except ``location_code``.
- ``Acquisition`` maps channels onto path distance through either the affine
  form (``start_distance`` + ``spatial_interval``) or a measured
  ``DistanceMap``; the two are mutually exclusive.
"""

from __future__ import annotations

import itertools
import re
from typing import Annotated, ClassVar, Literal, NamedTuple, TypeAlias
from uuid import uuid4

import numpy as np
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from dascore.constants import VALID_DATA_TYPES
from dascore.exceptions import InvalidInventoryError, ParameterError
from dascore.utils.models import (
    DascoreBaseModel,
    DateTime64,
    InventoryModel,
    TimeRangedModel,
    UnitQuantity,
)

ExtraFieldValue: TypeAlias = str | int | float | bool

VALID_COUPLING_TYPES = (
    "conduit",
    "trench",
    "outside_borehole_casing",
    "wireline",
    "surface",
    "coiled",
    "other",
)
CouplingType = Literal[VALID_COUPLING_TYPES]

_CODE_RE = re.compile(r"[A-Za-z0-9-]+")
_LOCATION_RE = re.compile(r"[A-Za-z0-9-]*")


def _check_code(value: str, name: str, allow_blank: bool = False) -> str:
    """Validate a data_source_id code token."""
    pattern = _LOCATION_RE if allow_blank else _CODE_RE
    if pattern.fullmatch(value) is None:
        allowed = "letters, digits, and '-'"
        blank = " (or blank)" if allow_blank else ""
        msg = f"Invalid {name} {value!r}; codes use {allowed}{blank}."
        raise InvalidInventoryError(msg)
    return value


def _is_strictly_increasing(values) -> bool:
    """Return True if a sequence is strictly increasing."""
    arr = np.asarray(values, dtype=float)
    return bool(len(arr) < 2 or np.all(np.diff(arr) > 0))


class CreationInfo(InventoryModel):
    """QuakeML-style provenance for an inventory document."""

    agency_id: str = Field(default="", description="Responsible agency.")
    author: str = Field(default="", description="Author or process.")
    creation_time: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description="Time this inventory was originally created (UTC).",
    )
    update_time: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description="Time this inventory was last updated (UTC).",
    )
    version: str = Field(default="", description="Content version of the document.")


class CoordinateReferenceSystem(InventoryModel):
    """
    Inventory-wide coordinate reference system.

    All coordinate-bearing metadata is interpreted using this CRS. The default
    is geographic WGS84 3D (EPSG:4979); override it only for exceptional
    frames such as mines or laboratories.
    """

    authority: str = Field(default="EPSG", description="CRS authority.")
    code: str = Field(default="4979", description="Authority code for this CRS.")
    name: str = Field(default="WGS 84 3D", description="Human-readable CRS name.")
    coordinate_labels: tuple[str, ...] = Field(
        default=("longitude", "latitude", "elevation"),
        description="Ordered coordinate axis labels.",
    )
    units: str = Field(default="degree", description="Coordinate units when known.")
    vertical_datum: str = Field(
        default="", description="Vertical datum or reference surface, if known."
    )

    _RESERVED_LABELS: ClassVar[frozenset] = frozenset(
        {"type", "sequence", "segment", "distance"}
    )

    @field_validator("coordinate_labels")
    @classmethod
    def _check_labels(cls, value):
        """Reserved structural column names may not be coordinate labels."""
        bad = set(value) & cls._RESERVED_LABELS
        if bad:
            msg = f"coordinate_labels may not use reserved names: {sorted(bad)}"
            raise InvalidInventoryError(msg)
        return value


class ExternalResource(InventoryModel):
    """External resource identified but not otherwise modeled by DASCore."""

    type: Literal["ExternalResource"] = "ExternalResource"
    resource_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this shareable inventory resource.",
    )
    uri: str = Field(default="", description="URI or identifier for the resource.")
    name: str = Field(default="", description="Human-readable resource name.")
    description: str = Field(default="", description="Free-form description.")


class Interrogator(InventoryModel):
    """DAS interrogator unit used for data collection."""

    type: Literal["Interrogator"] = "Interrogator"
    resource_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this shareable inventory resource.",
    )
    name: str = Field(default="", description="Human-readable resource name.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model number.")
    serial_number: str = Field(default="", description="Serial number.")
    instrument_type: str = Field(
        default="interrogator", description="General instrument category."
    )


class Enclosure(InventoryModel):
    """Physical housing, pipe, duct, conduit, or carrier resource."""

    type: Literal["Enclosure"] = "Enclosure"
    resource_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this shareable inventory resource.",
    )
    name: str = Field(default="", description="Human-readable resource name.")
    enclosure_type: str = Field(
        default="",
        description=(
            "Functional enclosure type, such as splice_box, junction_box, "
            "coupler, turnaround, protective_housing, conduit, pipe, duct, "
            "casing, borehole, or tray."
        ),
    )
    owner: str = Field(default="", description="Proprietary owner.")
    material: str = Field(default="", description="Material of the enclosure.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model name.")
    serial_number: str = Field(default="", description="Serial number.")
    inner_diameter: float | None = Field(
        default=None, description="Inner diameter in meters."
    )
    outer_diameter: float | None = Field(
        default=None, description="Outer diameter in meters."
    )
    specification: ExternalResource | None = Field(
        default=None, description="External specification or datasheet."
    )
    notes: str = Field(default="", description="Additional notes.")


class Cable(InventoryModel):
    """Physical cable containing one or more fiber segments."""

    type: Literal["Cable"] = "Cable"
    resource_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this shareable inventory resource.",
    )
    name: str = Field(default="", description="Human-readable resource name.")
    owner: str = Field(default="", description="Proprietary owner.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model name.")
    serial_number: str = Field(default="", description="Serial number.")
    diameter: float | None = Field(
        default=None, description="Outer cable diameter in meters."
    )
    strength_member: str = Field(
        default="", description="Strength member construction."
    )
    maximum_load: float | None = Field(
        default=None, description="Maximum load in newtons."
    )
    minimum_bend_radius: float | None = Field(
        default=None, description="Minimum bend radius in meters."
    )
    specification: ExternalResource | None = Field(
        default=None, description="External specification or datasheet."
    )
    container: Enclosure | Cable | None = Field(
        default=None, description="Optional containing physical asset."
    )
    fiber_count: int | None = Field(
        default=None, description="Number of fibers contained in the cable."
    )
    notes: str = Field(default="", description="Additional notes.")


_Resource: TypeAlias = Annotated[
    Interrogator | Cable | Enclosure | ExternalResource,
    Field(discriminator="type"),
]


class _OpticalComponentBase(InventoryModel):
    """Base class for physical optical components in an optical path."""

    optical_length: float = Field(
        default=0.0,
        ge=0.0,
        description="Optical component length along the optical path in meters.",
    )
    name: str = Field(default="", description="Human-readable component name.")


class FiberSegment(_OpticalComponentBase):
    """Length of optical fiber within a cable, patch cord, or other run."""

    type: Literal["FiberSegment"] = "FiberSegment"
    container: Cable | None = Field(
        default=None, description="Cable containing this fiber."
    )
    fiber_index: int | None = Field(
        default=None, description="Fiber position within the parent cable."
    )
    color: str | None = Field(default=None, description="Fiber color code.")
    fiber_type: str = Field(
        default="", description="Fiber type, such as single_mode or multi_mode."
    )
    fiber_standard: str = Field(
        default="", description="Fiber standard or grade, such as ITU-T G.652.D."
    )
    buffer_type: str = Field(
        default="", description="Buffer construction, such as tight_buffered."
    )
    attenuation_db_per_km: float | None = Field(
        default=None, description="Optical attenuation in dB/km."
    )
    attenuation_wavelength: float | None = Field(
        default=None, description="Wavelength of the attenuation value in nm."
    )
    rayleigh_backscatter_window: tuple[float, float] | None = Field(
        default=None, description="Rayleigh backscatter window in nm."
    )
    center_wavelength: float | None = Field(
        default=None, description="Center wavelength in nm."
    )
    notes: str = Field(default="", description="Additional notes.")


class Connector(_OpticalComponentBase):
    """Optical connector in an optical path."""

    type: Literal["Connector"] = "Connector"
    container: Enclosure | None = Field(
        default=None, description="Enclosure housing this connector."
    )
    connector_type: str = Field(default="", description="Connector type.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )
    notes: str = Field(default="", description="Additional notes.")


class Splice(_OpticalComponentBase):
    """Optical splice in an optical path."""

    type: Literal["Splice"] = "Splice"
    container: Enclosure | None = Field(
        default=None, description="Enclosure housing this splice."
    )
    splice_type: str = Field(default="", description="Splice type, such as fusion.")
    insertion_loss: float | None = Field(
        default=None, description="Insertion loss in dB, if known."
    )
    notes: str = Field(default="", description="Additional notes.")


class Terminator(_OpticalComponentBase):
    """Optical path terminator."""

    type: Literal["Terminator"] = "Terminator"
    container: Enclosure | None = Field(
        default=None, description="Enclosure housing this terminator."
    )
    termination_type: str = Field(
        default="", description="Termination type, such as open, capped, or angled."
    )
    reflectance: float | None = Field(
        default=None, description="Return loss or reflectance, if known."
    )
    notes: str = Field(default="", description="Additional notes.")


OpticalComponent: TypeAlias = Annotated[
    FiberSegment | Connector | Splice | Terminator,
    Field(discriminator="type"),
]


class Geometry(InventoryModel):
    """
    Geometry for an interval of an optical path.

    A geometry is a piecewise segment placed by its ``distance`` array: at
    least two strictly increasing optical distances, each paired with the
    coordinate at that point (interpreted using the inventory CRS). Coverage
    is the half-open span of the array; there is no separate length field. A
    coil is a segment whose coordinates repeat while distance advances.
    Interpolation between points is piecewise linear in the CRS and never
    crosses segments; uncovered distance has undefined coordinates.
    """

    name: str = Field(default="", description="Human-readable geometry name.")
    distance: tuple[float, ...] = Field(
        description=(
            "Optical distances paired to coordinates; at least two strictly "
            "increasing values whose span is the segment's coverage."
        ),
    )
    coordinates: tuple[tuple[float, ...], ...] = Field(
        description="Coordinate points; same length as distance.",
    )

    @model_validator(mode="after")
    def _validate_geometry(self) -> Self:
        """Enforce paired, strictly increasing control points."""
        if len(self.distance) < 2:
            msg = "Geometry requires at least two distance control points."
            raise InvalidInventoryError(msg)
        if not _is_strictly_increasing(self.distance):
            msg = "Geometry distance values must be strictly increasing."
            raise InvalidInventoryError(msg)
        if len(self.coordinates) != len(self.distance):
            msg = "Geometry coordinates and distance must have the same length."
            raise InvalidInventoryError(msg)
        dims = {len(coord) for coord in self.coordinates}
        if len(dims) > 1:
            msg = "Geometry coordinate points must share one dimensionality."
            raise InvalidInventoryError(msg)
        return self

    @property
    def distance_span(self) -> tuple[float, float]:
        """The (start, end) optical distance covered by this segment."""
        return (self.distance[0], self.distance[-1])

    def interpolate(self, distances) -> np.ndarray:
        """
        Return coordinates at the requested optical distances.

        Distances outside this segment's coverage return NaN rows. Coverage is
        half-open ``[first, last)``; inclusion of the outermost track endpoint
        is handled by the caller (`OpticalPath.coordinates_at`).
        """
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        coords = np.asarray(self.coordinates, dtype=float)
        out = np.full((len(dist), coords.shape[1]), np.nan)
        start, end = self.distance_span
        inside = (dist >= start) & (dist < end)
        for dim in range(coords.shape[1]):
            out[inside, dim] = np.interp(
                dist[inside], np.asarray(self.distance), coords[:, dim]
            )
        return out


class CouplingCondition(InventoryModel):
    """
    Acoustic coupling condition for an interval of an optical path.

    Placed by start ``distance`` plus ``optical_length`` (the annotation
    idiom). Coverage may be partial and conditions may not overlap.
    """

    distance: float = Field(
        description="Start optical distance for this coupling condition in meters."
    )
    optical_length: float = Field(
        gt=0.0, description="Interval length described by this coupling in meters."
    )
    coupling_type: CouplingType = Field(description="Controlled coupling category.")
    medium: str = Field(default="", description="Surrounding medium.")
    attachment: str = Field(default="", description="Attachment method.")
    depth: float | None = Field(
        default=None,
        description=(
            "Depth in meters, positive down, relative to the local surface, "
            "when relevant."
        ),
    )
    notes: str = Field(default="", description="Additional notes.")

    @property
    def interval(self) -> tuple[float, float]:
        """The (start, end) optical distance covered by this condition."""
        return (self.distance, self.distance + self.optical_length)


class OpticalPathAnnotation(InventoryModel):
    """Named interval on an optical path; annotations may overlap freely."""

    distance: float = Field(
        description="Start optical distance for this annotation in meters."
    )
    optical_length: float = Field(
        gt=0.0, description="Interval length described by this annotation in meters."
    )
    label: str = Field(default="", description="Label for this interval.")
    notes: str = Field(default="", description="Additional notes.")

    @property
    def interval(self) -> tuple[float, float]:
        """The (start, end) optical distance covered by this annotation."""
        return (self.distance, self.distance + self.optical_length)


class DistanceMap(InventoryModel):
    """
    Measured control-point map from a channel-like coordinate onto optical
    path distance.

    The map is a function, not a roster of existing channels. Exactly one
    input axis is populated: ``channel`` when the interrogator reports channel
    numbers, ``instrument_distance`` when it reports its own nominal meters.
    Values between control points are piecewise-linearly interpolated;
    channels outside the covered range are undefined (NaN). A single-point
    map takes its slope from the acquisition's nominal ``spatial_interval``.
    """

    channel: tuple[float, ...] | None = Field(
        default=None,
        description="Interrogator-reported channel numbers; strictly increasing.",
    )
    instrument_distance: tuple[float, ...] | None = Field(
        default=None,
        description="Interrogator-reported nominal distances; strictly increasing.",
    )
    distance: tuple[float, ...] = Field(
        description="Optical path distances at the control points; increasing.",
    )

    @model_validator(mode="after")
    def _validate_map(self) -> Self:
        """Enforce one input axis and paired increasing control points."""
        inputs = [self.channel, self.instrument_distance]
        populated = [x for x in inputs if x is not None]
        if len(populated) != 1:
            msg = (
                "DistanceMap requires exactly one input axis: "
                "channel or instrument_distance."
            )
            raise InvalidInventoryError(msg)
        source = populated[0]
        if len(source) != len(self.distance):
            msg = "DistanceMap input and distance must have the same length."
            raise InvalidInventoryError(msg)
        if len(source) < 1:
            msg = "DistanceMap requires at least one control point."
            raise InvalidInventoryError(msg)
        if not _is_strictly_increasing(source):
            msg = "DistanceMap input values must be strictly increasing."
            raise InvalidInventoryError(msg)
        if not _is_strictly_increasing(self.distance):
            msg = "DistanceMap distance values must be strictly increasing."
            raise InvalidInventoryError(msg)
        return self

    @property
    def source_values(self) -> tuple[float, ...]:
        """The populated input axis values."""
        out = self.channel if self.channel is not None else self.instrument_distance
        return out

    def map_to_distance(self, values, slope: float | None = None) -> np.ndarray:
        """
        Map channel-like values onto optical path distance.

        Parameters
        ----------
        values
            Channel numbers or instrument distances (per the populated axis).
        slope
            Distance per input unit, used only for single-point maps
            (normally the acquisition's nominal ``spatial_interval``).
        """
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        source = np.asarray(self.source_values, dtype=float)
        dist = np.asarray(self.distance, dtype=float)
        if len(source) == 1:
            if slope is None:
                msg = (
                    "A single-point DistanceMap requires a slope "
                    "(the acquisition's spatial_interval)."
                )
                raise InvalidInventoryError(msg)
            return dist[0] + (vals - source[0]) * slope
        out = np.interp(vals, source, dist)
        out[(vals < source[0]) | (vals > source[-1])] = np.nan
        return out


class Acquisition(TimeRangedModel):
    """
    Time-aware, channel-like DAS acquisition setup.

    ``(location_code, code)`` pairs are unique within a ``FiberArray`` for
    overlapping time ranges. The ``location_code`` names the optical path
    lineage this acquisition interrogates. ``start_distance`` and
    ``distance_map`` are mutually exclusive channel-resolution mechanisms.
    """

    code: str = Field(description="Channel-like acquisition code.")
    location_code: str = Field(
        default="",
        description=(
            "FDSN-style location code naming the optical path lineage this "
            "acquisition interrogates; may be blank, as in FDSN."
        ),
    )
    data_type: Literal[VALID_DATA_TYPES] = Field(
        default="", description="Quantity measured or produced."
    )
    data_category: str = Field(
        default="", description="Acquisition family such as DAS, DTS, or DSS."
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="Units of data produced."
    )
    interrogator: Interrogator | None = Field(
        default=None, description="Interrogator used for this acquisition."
    )
    interrogator_port: str | None = Field(
        default=None,
        description=(
            "Instrument connector or launch port that fed this acquisition, "
            "for multiplexed interrogators. Descriptive only."
        ),
    )
    gauge_length: float | None = Field(
        default=None, description="Gauge length in meters."
    )
    pulse_rate: float | None = Field(
        default=None, description="Pulse repetition rate in Hz."
    )
    pulse_width: float | None = Field(
        default=None, description="Pulse width in seconds."
    )
    sample_rate: float | None = Field(
        default=None, description="FDSN-style acquisition sample rate in Hz."
    )
    spatial_interval: float | None = Field(
        default=None,
        description=(
            "Nominal spatial sampling interval between channels in meters. "
            "Participates in channel resolution only as the affine slope, or "
            "as the slope of a single-point distance_map."
        ),
    )
    start_distance: float | None = Field(
        default=None,
        description=(
            "Optical path distance corresponding to channel position 0. "
            "Mutually exclusive with distance_map."
        ),
    )
    distance_map: DistanceMap | None = Field(
        default=None,
        description=(
            "Measured map from interrogator-reported channels or distances "
            "onto optical path distance. Mutually exclusive with "
            "start_distance."
        ),
    )
    closed_fiber_loop: bool = Field(
        default=False,
        description="True when the interrogator is attached to both path ends.",
    )
    extra_fields: dict[str, ExtraFieldValue] = Field(
        default_factory=dict,
        description="Extra acquisition metadata not represented by the model.",
    )

    @field_validator("code")
    @classmethod
    def _check_acq_code(cls, value):
        return _check_code(value, "acquisition code")

    @field_validator("location_code")
    @classmethod
    def _check_location(cls, value):
        return _check_code(value, "location code", allow_blank=True)

    @model_validator(mode="after")
    def _check_resolution_mechanism(self) -> Self:
        """start_distance and distance_map are mutually exclusive."""
        if self.start_distance is not None and self.distance_map is not None:
            msg = (
                "Acquisition defines at most one channel-resolution mechanism: "
                "start_distance and distance_map are mutually exclusive."
            )
            raise InvalidInventoryError(msg)
        return self

    def channel_to_distance(self, values) -> np.ndarray:
        """
        Map channel-like patch coordinates onto optical path distance.

        Uses the measured ``distance_map`` when present, else the affine
        ``start_distance`` + ``spatial_interval`` form.
        """
        if self.distance_map is not None:
            return self.distance_map.map_to_distance(
                values, slope=self.spatial_interval
            )
        if self.start_distance is None or self.spatial_interval is None:
            msg = (
                "Acquisition defines no channel-resolution mechanism; set "
                "start_distance and spatial_interval, or a distance_map."
            )
            raise InvalidInventoryError(msg)
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        return self.start_distance + vals * self.spatial_interval


def _intervals_overlap(intervals: list[tuple[float, float]]) -> tuple | None:
    """Return the first overlapping pair of half-open intervals, or None."""
    ordered = sorted(intervals)
    for first, second in itertools.pairwise(ordered):
        if second[0] < first[1]:
            return first, second
    return None


class OpticalPath(TimeRangedModel):
    """
    Continuous optical path described by independent tracks.

    Optical components tile ``[start_distance, start_distance + optical
    length)``. Geometry and coupling are function tracks (partial coverage,
    no overlap); annotations overlap freely. No more than one path per
    ``(FiberArray, location_code)`` is valid at a time.
    """

    name: str = Field(default="", description="Human-readable optical path name.")
    start_distance: float = Field(
        default=0.0,
        description=(
            "Origin of this path's optical-distance axis in meters; 0 for "
            "whole paths. Set by select and split_at so pieces keep absolute "
            "optical distances."
        ),
    )
    location_code: str = Field(
        default="",
        description=(
            "FDSN-style location code naming this path lineage within its "
            "fiber array; blank by default."
        ),
    )
    optical_components: tuple[OpticalComponent, ...] = Field(
        default=(), description="Ordered optical components on this path."
    )
    geometry: tuple[Geometry, ...] = Field(
        default=(), description="Piecewise geometry segments on this path."
    )
    coupling: tuple[CouplingCondition, ...] = Field(
        default=(), description="Coupling conditions on this path."
    )
    annotations: tuple[OpticalPathAnnotation, ...] = Field(
        default=(), description="Annotations on this path."
    )
    otdr_traces: tuple[ExternalResource, ...] = Field(
        default=(), description="References to OTDR traces of this path."
    )

    @field_validator("location_code")
    @classmethod
    def _check_location(cls, value):
        return _check_code(value, "location code", allow_blank=True)

    @property
    def optical_length(self) -> float:
        """Total optical length, computed from the optical components."""
        return float(sum(x.optical_length for x in self.optical_components))

    @property
    def end_distance(self) -> float:
        """The end of this path's optical-distance axis."""
        return self.start_distance + self.optical_length

    def component_intervals(self) -> tuple[tuple[float, float], ...]:
        """Return each component's (start, end) on the absolute axis."""
        out, position = [], self.start_distance
        for comp in self.optical_components:
            nxt = position + comp.optical_length
            out.append((position, nxt))
            position = nxt
        return tuple(out)

    def coordinates_at(self, distances) -> np.ndarray:
        """
        Return CRS coordinates at the requested optical distances.

        Uncovered distance returns NaN rows. Segment coverage is half-open;
        the outermost covered endpoint of the geometry track is included.
        """
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        if not self.geometry:
            return np.full((len(dist), 1), np.nan)
        ndim = len(self.geometry[0].coordinates[0])
        out = np.full((len(dist), ndim), np.nan)
        for segment in self.geometry:
            seg_coords = segment.interpolate(dist)
            filled = ~np.isnan(seg_coords[:, 0])
            out[filled] = seg_coords[filled]
        # The outermost endpoint of the geometry coverage domain is included.
        outer = max(seg.distance_span[1] for seg in self.geometry)
        at_outer = dist == outer
        if np.any(at_outer):
            segment = max(self.geometry, key=lambda s: s.distance_span[1])
            coords = np.asarray(segment.coordinates, dtype=float)
            out[at_outer] = coords[-1]
        return out

    def coupling_at(self, distances) -> list[CouplingCondition | None]:
        """Return the coupling condition covering each distance, or None."""
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        outer = max((c.interval[1] for c in self.coupling), default=None)
        out: list[CouplingCondition | None] = [None] * len(dist)
        for cond in self.coupling:
            start, end = cond.interval
            for i, d in enumerate(dist):
                covered = start <= d < end or (d == end == outer)
                if covered:
                    out[i] = cond
        return out

    def validate(self, tolerance: float = 1e-9) -> Self:
        """
        Validate track rules for this path.

        Checks that geometry and coupling stay within path bounds and do not
        overlap (partial coverage is legal), and that annotations stay within
        bounds. Component tiling is inherent to the cumulative layout.
        """
        errors = []
        start, end = self.start_distance, self.end_distance
        geo_spans = [seg.distance_span for seg in self.geometry]
        coup_spans = [c.interval for c in self.coupling]
        anno_spans = [a.interval for a in self.annotations]
        for name, spans in (
            ("geometry", geo_spans),
            ("coupling", coup_spans),
            ("annotations", anno_spans),
        ):
            for lo, hi in spans:
                if lo < start - tolerance or hi > end + tolerance:
                    errors.append(
                        f"{name} interval ({lo}, {hi}) extends past path "
                        f"span ({start}, {end})."
                    )
        for name, spans in (("geometry", geo_spans), ("coupling", coup_spans)):
            overlap = _intervals_overlap(spans)
            if overlap is not None:
                errors.append(
                    f"Overlapping {name} intervals {overlap[0]} and "
                    f"{overlap[1]}; {name} is a function track."
                )
        if errors:
            msg = "Optical path validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        return self

    def select(self, *, distance: tuple[float | None, float | None]) -> Self:
        """
        Return a new path clipped to a distance interval.

        Absolute optical distances are preserved: the piece's
        ``start_distance`` becomes the clip start, never zero.
        """
        lo = self.start_distance if distance[0] is None else float(distance[0])
        hi = self.end_distance if distance[1] is None else float(distance[1])
        lo = max(lo, self.start_distance)
        hi = min(hi, self.end_distance)
        if hi <= lo:
            msg = f"Empty distance selection ({distance})."
            raise ParameterError(msg)
        components = []
        for comp, (c_lo, c_hi) in zip(
            self.optical_components, self.component_intervals()
        ):
            new_lo, new_hi = max(c_lo, lo), min(c_hi, hi)
            if new_hi > new_lo or (c_lo == c_hi and lo <= c_lo < hi):
                length = max(new_hi - new_lo, 0.0)
                components.append(comp.model_copy(update={"optical_length": length}))
        geometry = []
        for seg in self.geometry:
            s_lo, s_hi = seg.distance_span
            if s_hi <= lo or s_lo >= hi:
                continue
            new_lo, new_hi = max(s_lo, lo), min(s_hi, hi)
            if new_hi <= new_lo:
                continue
            dist = np.asarray(seg.distance, dtype=float)
            inside = (dist > new_lo) & (dist < new_hi)
            new_dist = np.concatenate([[new_lo], dist[inside], [new_hi]])
            coords = np.asarray(seg.coordinates, dtype=float)
            new_coords = np.stack(
                [
                    np.interp(new_dist, dist, coords[:, dim])
                    for dim in range(coords.shape[1])
                ],
                axis=1,
            )
            geometry.append(
                seg.model_copy(
                    update={
                        "distance": tuple(new_dist),
                        "coordinates": tuple(map(tuple, new_coords)),
                    }
                )
            )
        coupling = _clip_intervals(self.coupling, lo, hi)
        annotations = _clip_intervals(self.annotations, lo, hi)
        return self.model_copy(
            update={
                "start_distance": lo,
                "optical_components": tuple(components),
                "geometry": tuple(geometry),
                "coupling": tuple(coupling),
                "annotations": tuple(annotations),
            }
        )

    def split_at(self, distance: float) -> tuple[Self, Self]:
        """Split the optical path into two pieces at a distance."""
        distance = float(distance)
        return (
            self.select(distance=(None, distance)),
            self.select(distance=(distance, None)),
        )

    def reverse(self) -> Self:
        """
        Return the path traversed from the far end.

        Every distance-bearing track is rewritten against the reversed axis;
        the absolute span is unchanged.
        """
        start, end = self.start_distance, self.end_distance

        def flip(d):
            return start + end - d

        geometry = []
        for seg in self.geometry:
            dist = np.asarray(seg.distance, dtype=float)
            coords = np.asarray(seg.coordinates, dtype=float)
            geometry.append(
                seg.model_copy(
                    update={
                        "distance": tuple(flip(dist)[::-1]),
                        "coordinates": tuple(map(tuple, coords[::-1])),
                    }
                )
            )
        geometry.sort(key=lambda s: s.distance[0])
        coupling = sorted(
            (
                c.model_copy(update={"distance": flip(c.interval[1])})
                for c in self.coupling
            ),
            key=lambda c: c.distance,
        )
        annotations = sorted(
            (
                a.model_copy(update={"distance": flip(a.interval[1])})
                for a in self.annotations
            ),
            key=lambda a: a.distance,
        )
        return self.model_copy(
            update={
                "optical_components": tuple(reversed(self.optical_components)),
                "geometry": tuple(geometry),
                "coupling": tuple(coupling),
                "annotations": tuple(annotations),
            }
        )

    def __add__(self, other) -> Self:
        """Concatenate paths, rewriting the second onto the combined axis."""
        if not isinstance(other, OpticalPath):
            return NotImplemented
        offset = self.end_distance - other.start_distance
        geometry = tuple(
            seg.model_copy(
                update={"distance": tuple(d + offset for d in seg.distance)}
            )
            for seg in other.geometry
        )
        coupling = tuple(
            c.model_copy(update={"distance": c.distance + offset})
            for c in other.coupling
        )
        annotations = tuple(
            a.model_copy(update={"distance": a.distance + offset})
            for a in other.annotations
        )
        return self.model_copy(
            update={
                "optical_components": (
                    *self.optical_components,
                    *other.optical_components,
                ),
                "geometry": (*self.geometry, *geometry),
                "coupling": (*self.coupling, *coupling),
                "annotations": (*self.annotations, *annotations),
                "otdr_traces": (*self.otdr_traces, *other.otdr_traces),
            }
        )

    def __radd__(self, other):
        """Support sum() over path sequences."""
        if other == 0:
            return self
        return NotImplemented


def _clip_intervals(items, lo: float, hi: float) -> list:
    """Clip start+optical_length interval items to [lo, hi), dropping empties."""
    out = []
    for item in items:
        start, end = item.interval
        new_lo, new_hi = max(start, lo), min(end, hi)
        if new_hi <= new_lo:
            continue
        out.append(
            item.model_copy(
                update={"distance": new_lo, "optical_length": new_hi - new_lo}
            )
        )
    return out


class ResponseStage(InventoryModel):
    """A single stage of a station-channel response."""

    name: str = Field(default="", description="Stage name.")
    description: str = Field(default="", description="Stage description.")
    gain: float | None = Field(default=None, description="Stage gain.")
    extra_fields: dict[str, ExtraFieldValue] = Field(
        default_factory=dict, description="Extra stage metadata."
    )


class Response(InventoryModel):
    """Station-specific response model associated with a channel."""

    sensitivity: float | None = Field(
        default=None, description="Overall channel sensitivity."
    )
    input_units: UnitQuantity | None = Field(
        default=None, description="Physical units before conversion."
    )
    output_units: UnitQuantity | None = Field(
        default=None, description="Physical units after conversion."
    )
    stages: tuple[ResponseStage, ...] = Field(
        default=(), description="Ordered response stages."
    )
    extra_fields: dict[str, ExtraFieldValue] = Field(
        default_factory=dict, description="Extra response metadata."
    )


class Channel(TimeRangedModel):
    """Station-level time-series stream identity."""

    code: str = Field(description="Channel code used in data source identifiers.")
    location_code: str = Field(
        default="", description="FDSN-style location code; may be blank."
    )
    coordinates: tuple[float, ...] | None = Field(
        default=None,
        description="Coordinates ordered by the inventory CRS coordinate_labels.",
    )
    data_type: Literal[VALID_DATA_TYPES] = Field(
        default="", description="Quantity measured or produced."
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="Units of data produced."
    )
    response: Response | None = Field(
        default=None, description="Optional response metadata."
    )

    @field_validator("code")
    @classmethod
    def _check_chan_code(cls, value):
        return _check_code(value, "channel code")

    @field_validator("location_code")
    @classmethod
    def _check_location(cls, value):
        return _check_code(value, "location code", allow_blank=True)


class Station(TimeRangedModel):
    """Point-like observing identity under a network."""

    code: str = Field(description="Station code used in data source identifiers.")
    name: str = Field(default="", description="Human-readable station name.")
    coordinates: tuple[float, ...] | None = Field(
        default=None,
        description="Coordinates ordered by the inventory CRS coordinate_labels.",
    )
    channels: tuple[Channel, ...] = Field(
        default=(), description="Channels associated with this station."
    )
    notes: str = Field(default="", description="Additional notes.")

    @field_validator("code")
    @classmethod
    def _check_sta_code(cls, value):
        return _check_code(value, "station code")


class FiberArray(TimeRangedModel):
    """
    Durable fiber-optic observing identity.

    A fiber array groups similar, co-located fiber routes; each route is a
    location code carrying one optical path at a time.
    """

    code: str = Field(description="Station-like fiber array code.")
    name: str = Field(default="", description="Human-readable fiber array name.")
    acquisitions: tuple[Acquisition, ...] = Field(
        default=(), description="Acquisitions associated with this fiber array."
    )
    optical_paths: tuple[OpticalPath, ...] = Field(
        default=(), description="Optical paths associated with this fiber array."
    )
    notes: str = Field(default="", description="Additional notes.")

    @field_validator("code")
    @classmethod
    def _check_fa_code(cls, value):
        return _check_code(value, "fiber array code")

    def validate(self) -> Self:
        """
        Validate epoch rules for this fiber array.

        No more than one optical path per location code may be valid at a
        time, and acquisition (location_code, code) pairs must be unique for
        overlapping time ranges. Contained paths are also validated.
        """
        errors = []
        by_location: dict[str, list[OpticalPath]] = {}
        for path in self.optical_paths:
            by_location.setdefault(path.location_code, []).append(path)
        for location, paths in by_location.items():
            for i, first in enumerate(paths):
                for second in paths[i + 1 :]:
                    if first.overlaps(second):
                        errors.append(
                            f"Optical paths {first.name!r} and {second.name!r} "
                            f"overlap in time for location {location!r}."
                        )
        by_key: dict[tuple[str, str], list[Acquisition]] = {}
        for acq in self.acquisitions:
            by_key.setdefault((acq.location_code, acq.code), []).append(acq)
        for key, acqs in by_key.items():
            for i, first in enumerate(acqs):
                for second in acqs[i + 1 :]:
                    if first.overlaps(second):
                        errors.append(
                            f"Acquisition epochs for {key} overlap in time."
                        )
        if errors:
            msg = f"Fiber array {self.code!r} validation failed:\n" + "\n".join(
                errors
            )
            raise InvalidInventoryError(msg)
        for path in self.optical_paths:
            path.validate()
        return self


class Network(InventoryModel):
    """FDSN-like organizational container below an inventory."""

    code: str = Field(description="Network code.")
    name: str = Field(default="", description="Human-readable network name.")
    description: str = Field(default="", description="Network description.")
    fiber_arrays: tuple[FiberArray, ...] = Field(
        default=(), description="Fiber arrays in this network."
    )
    stations: tuple[Station, ...] = Field(
        default=(), description="Stations in this network."
    )
    notes: str = Field(default="", description="Additional notes.")

    @field_validator("code")
    @classmethod
    def _check_net_code(cls, value):
        return _check_code(value, "network code")

    def validate(self) -> Self:
        """
        Validate code rules for this network.

        Station and fiber-array codes must be disjoint for overlapping time
        ranges; contained fiber arrays are also validated.
        """
        errors = []
        for array in self.fiber_arrays:
            for station in self.stations:
                if array.code == station.code and array.overlaps(station):
                    errors.append(
                        f"Fiber array and station share code {array.code!r} "
                        "for overlapping time ranges."
                    )
        if errors:
            msg = f"Network {self.code!r} validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        for array in self.fiber_arrays:
            array.validate()
        return self


class ResolvedContext(NamedTuple):
    """The inventory objects a data_source_id + time resolve to."""

    network: Network
    fiber_array: FiberArray
    acquisition: Acquisition
    optical_path: OpticalPath | None


class Inventory(DascoreBaseModel):
    """
    Top-level DASDAE inventory manifest.

    Stores document metadata, shareable resources keyed by resource_id, the
    inventory-wide coordinate reference system, and network containers.
    """

    schema_version: int = Field(
        default=1, description="Version of the inventory manifest envelope."
    )
    resource_id: str = Field(
        default="", description="Optional identifier for the inventory manifest."
    )
    creation_info: CreationInfo = Field(
        default_factory=CreationInfo,
        description="QuakeML-style creation and update metadata.",
    )
    resources: dict[str, _Resource] = Field(
        default_factory=dict,
        description="Shareable resources keyed by resource_id.",
    )
    coordinate_reference_system: CoordinateReferenceSystem = Field(
        default_factory=CoordinateReferenceSystem,
        description="Inventory-wide CRS used by all coordinate-bearing metadata.",
    )
    networks: tuple[Network, ...] = Field(
        default=(), description="Network containers in the inventory."
    )

    @field_validator("resources", mode="before")
    @classmethod
    def _key_resources(cls, value):
        """Accept an iterable of resources, keying them by resource_id."""
        if isinstance(value, dict):
            return value
        return {resource.resource_id: resource for resource in value}

    def validate(self) -> Self:
        """Validate the whole inventory tree."""
        codes = [net.code for net in self.networks]
        if len(codes) != len(set(codes)):
            msg = f"Network codes must be unique; got {codes}."
            raise InvalidInventoryError(msg)
        for net in self.networks:
            net.validate()
        return self

    def resolve(self, data_source_id: str, time=None) -> ResolvedContext:
        """
        Resolve a data_source_id (and time) to its inventory context.

        Parameters
        ----------
        data_source_id
            A dotted ``network.fiber_array.location.acquisition`` identifier.
        time
            The instant to resolve at; required whenever any epoch level has
            more than one candidate.
        """
        parts = data_source_id.split(".")
        if len(parts) != 4:
            msg = (
                f"data_source_id {data_source_id!r} must have exactly four "
                "dot-separated parts."
            )
            raise InvalidInventoryError(msg)
        net_code, array_code, location, acq_code = parts
        networks = [x for x in self.networks if x.code == net_code]
        if len(networks) != 1:
            msg = f"{data_source_id!r} resolves to {len(networks)} networks."
            raise InvalidInventoryError(msg)
        network = networks[0]
        arrays = [
            x
            for x in network.fiber_arrays
            if x.code == array_code and x.is_effective_at(time)
        ]
        if len(arrays) != 1:
            msg = f"{data_source_id!r} resolves to {len(arrays)} fiber arrays."
            raise InvalidInventoryError(msg)
        array = arrays[0]
        acqs = [
            x
            for x in array.acquisitions
            if x.code == acq_code
            and x.location_code == location
            and x.is_effective_at(time)
        ]
        if len(acqs) != 1:
            msg = f"{data_source_id!r} resolves to {len(acqs)} acquisitions."
            raise InvalidInventoryError(msg)
        paths = [
            x
            for x in array.optical_paths
            if x.location_code == location and x.is_effective_at(time)
        ]
        if len(paths) > 1:
            msg = f"{data_source_id!r} resolves to {len(paths)} optical paths."
            raise InvalidInventoryError(msg)
        path = paths[0] if paths else None
        return ResolvedContext(network, array, acqs[0], path)

    def replace(self, old, new) -> Self:
        """
        Return a new inventory with one component replaced.

        This is the correction mechanism: the change applies in place and
        retroactively. ``old`` is matched by equality anywhere in the tree.
        """
        replaced = 0

        def swap(items):
            nonlocal replaced
            out = []
            for item in items:
                if item == old:
                    out.append(new)
                    replaced += 1
                else:
                    out.append(item)
            return tuple(out)

        networks = []
        for net in swap(self.networks):
            if net == new and replaced:
                networks.append(net)
                continue
            arrays = []
            for array in swap(net.fiber_arrays):
                if array == new:
                    arrays.append(array)
                    continue
                arrays.append(
                    array.model_copy(
                        update={
                            "acquisitions": swap(array.acquisitions),
                            "optical_paths": swap(array.optical_paths),
                        }
                    )
                )
            networks.append(
                net.model_copy(
                    update={
                        "fiber_arrays": tuple(arrays),
                        "stations": swap(net.stations),
                    }
                )
            )
        if not replaced:
            msg = "Component to replace was not found in the inventory."
            raise InvalidInventoryError(msg)
        return self.model_copy(update={"networks": tuple(networks)})

    def to_yaml(self, path=None) -> str:
        """Serialize this inventory to YAML, optionally writing to a path."""
        import yaml

        data = self.model_dump(mode="json", exclude_none=True)
        out = yaml.safe_dump(data, sort_keys=False)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(out)
        return out

    @classmethod
    def from_yaml(cls, source) -> Self:
        """Load an inventory from a YAML string or file path."""
        import os

        import yaml

        text = source
        if isinstance(source, os.PathLike) or (
            isinstance(source, str) and "\n" not in source and os.path.exists(source)
        ):
            with open(source) as fh:
                text = fh.read()
        return cls(**yaml.safe_load(text))


def inventory(source=None) -> Inventory:
    """
    Load or create a DASDAE inventory.

    Parameters
    ----------
    source
        A YAML file path or string; if None, returns an empty inventory.
    """
    if source is None:
        return Inventory()
    return Inventory.from_yaml(source)
