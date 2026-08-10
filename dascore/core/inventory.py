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
from typing import Annotated, Any, Literal, NamedTuple, TypeAlias, get_args
from uuid import uuid4

import numpy as np
from pydantic import AfterValidator, Field, field_validator, model_validator
from typing_extensions import Self

from dascore.constants import DataCategory, DataType
from dascore.exceptions import InvalidInventoryError, ParameterError
from dascore.utils.misc import optional_import
from dascore.utils.models import (
    DateTime64,
    InventoryModel,
    TimeRangedModel,
    UnitQuantity,
)

CouplingType = Literal[
    "conduit",
    "trench",
    "outside_borehole_casing",
    "wireline",
    "surface",
    "aerial",
    "coiled",
    "other",
]
VALID_COUPLING_TYPES = get_args(CouplingType)

# Coordinates are stored on the canonical (x, y, z) axes; these labels are
# resolvable aliases whose meaning the inventory CRS declares.
VALID_COORDINATE_LABELS = (
    "x",
    "y",
    "z",
    "latitude",
    "longitude",
    "elevation",
    "easting",
    "northing",
)

_CODE_RE = re.compile(r"[A-Za-z0-9-]+")
_LOCATION_RE = re.compile(r"[A-Za-z0-9-]*")


def _check_code(value: str, allow_blank: bool = False) -> str:
    """Validate a data_source_id code token."""
    pattern = _LOCATION_RE if allow_blank else _CODE_RE
    if pattern.fullmatch(value) is None:
        blank = " (or blank)" if allow_blank else ""
        msg = f"Invalid code {value!r}; codes use letters, digits, and '-'{blank}."
        raise InvalidInventoryError(msg)
    return value


# Code tokens used in data_source_id; location codes alone may be blank.
CodeStr = Annotated[str, AfterValidator(_check_code)]
LocationCodeStr = Annotated[
    str, AfterValidator(lambda value: _check_code(value, allow_blank=True))
]
# Stable identifier for a shareable inventory resource.
ResourceIdStr = Annotated[
    str,
    Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this shareable inventory resource.",
    ),
]


def _type_tag(name: str):
    """
    Return the serialization-only type tag field for a tagged model class.

    The tag drives union dispatch and the authoring format's type
    declaration in serialized YAML/JSON. Users never set it: it defaults to
    the class name, the Literal annotation rejects any other value, and it
    is hidden from repr.
    """
    return Field(default=name, repr=False)


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
        description=(
            "Time the content was last updated (UTC); by convention set "
            "alongside a version bump when a correction is made."
        ),
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
        description=(
            "Meaning of the canonical (x, y, z) axes, in order. Labels come "
            "from the controlled coordinate vocabulary."
        ),
    )
    units: str = Field(default="degree", description="Coordinate units when known.")
    vertical_datum: str = Field(
        default="", description="Vertical datum or reference surface, if known."
    )

    @field_validator("coordinate_labels")
    @classmethod
    def _check_labels(cls, value):
        """Labels come from the controlled vocabulary and are unique."""
        bad = set(value) - set(VALID_COORDINATE_LABELS)
        if bad:
            msg = (
                f"coordinate_labels {sorted(bad)} not in the coordinate "
                f"vocabulary {VALID_COORDINATE_LABELS}."
            )
            raise InvalidInventoryError(msg)
        if len(set(value)) != len(value):
            msg = f"coordinate_labels must be unique; got {value}."
            raise InvalidInventoryError(msg)
        return value

    def axis_index(self, label: str) -> int:
        """
        Return the canonical axis position for a coordinate label.

        ``x``, ``y``, and ``z`` are always the canonical axes; any other
        label resolves only when this CRS defines it in coordinate_labels.
        """
        canonical = {"x": 0, "y": 1, "z": 2}
        if label in canonical:
            index = canonical[label]
            if index >= len(self.coordinate_labels):
                msg = f"This CRS has no {label!r} axis (labels: "
                msg += f"{self.coordinate_labels})."
                raise InvalidInventoryError(msg)
            return index
        if label in self.coordinate_labels:
            return self.coordinate_labels.index(label)
        msg = (
            f"Coordinate label {label!r} is not defined by this CRS "
            f"(labels: {self.coordinate_labels})."
        )
        raise InvalidInventoryError(msg)


class ExternalResource(InventoryModel):
    """External resource identified but not otherwise modeled by DASCore."""

    type: Literal["ExternalResource"] = _type_tag("ExternalResource")
    resource_id: ResourceIdStr
    uri: str = Field(default="", description="URI or identifier for the resource.")
    name: str = Field(default="", description="Human-readable resource name.")
    description: str = Field(default="", description="Free-form description.")


class OpticalMeasurement(InventoryModel):
    """
    A characterization of the optical path or its components.

    One record captures the conditions a loss or reflectance number was
    obtained under — the conditions live here, stated once, and every
    number derived from the same run references the same record. A
    datasheet claim is a legitimate record (method="datasheet").
    """

    type: Literal["OpticalMeasurement"] = _type_tag("OpticalMeasurement")
    resource_id: ResourceIdStr
    name: str = Field(default="", description="Human-readable measurement name.")
    method: str = Field(
        default="",
        description=(
            "How the values were obtained: otdr, power_meter, "
            "splicer_estimate, datasheet, or other."
        ),
    )
    time: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description="Time of the measurement (UTC).",
    )
    wavelength: float | None = Field(
        default=None, description="Measurement wavelength in nm."
    )
    pulse_width: float | None = Field(
        default=None, description="OTDR pulse width in seconds."
    )
    direction: str = Field(
        default="", description="Measurement direction: forward or reverse."
    )
    data: ExternalResource | str | None = Field(
        default=None, description="The trace or report file, as a resource."
    )


class Interrogator(InventoryModel):
    """DAS interrogator unit used for data collection."""

    type: Literal["Interrogator"] = _type_tag("Interrogator")
    resource_id: ResourceIdStr
    name: str = Field(default="", description="Human-readable resource name.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model number.")
    serial_number: str = Field(default="", description="Serial number.")
    instrument_type: str = Field(
        default="interrogator", description="General instrument category."
    )


class Enclosure(InventoryModel):
    """Physical housing, pipe, duct, conduit, or carrier resource."""

    type: Literal["Enclosure"] = _type_tag("Enclosure")
    resource_id: ResourceIdStr
    name: str = Field(default="", description="Human-readable resource name.")
    enclosure_type: str = Field(
        default="",
        description=(
            "Functional enclosure type, such as splice_box, junction_box, "
            "coupler, turnaround, protective_housing, conduit, pipe, duct, "
            "casing, borehole, or tray."
        ),
    )
    material: str = Field(default="", description="Material of the enclosure.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model name.")
    inner_diameter: float | None = Field(
        default=None, description="Inner diameter in meters."
    )
    specification: ExternalResource | str | None = Field(
        default=None, description="External specification or datasheet."
    )


class Cable(InventoryModel):
    """Physical cable containing one or more fiber segments."""

    type: Literal["Cable"] = _type_tag("Cable")
    resource_id: ResourceIdStr
    name: str = Field(default="", description="Human-readable resource name.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model name.")
    diameter: float | None = Field(
        default=None, description="Outer cable diameter in meters."
    )
    specification: ExternalResource | str | None = Field(
        default=None, description="External specification or datasheet."
    )
    container: Enclosure | Cable | str | None = Field(
        default=None, description="Optional containing physical asset."
    )
    fiber_count: int | None = Field(
        default=None, description="Number of fibers contained in the cable."
    )


_Resource: TypeAlias = Annotated[
    Interrogator | Cable | Enclosure | ExternalResource | OpticalMeasurement,
    Field(discriminator="type"),
]


class _OpticalComponentBase(InventoryModel):
    """
    Base class for physical optical components in an optical path.

    Every component carries a unified one-way transmission ``loss_db`` and
    return ``reflectance_db`` (the two quantities an OTDR trace shows per
    event), each paired with the measurement record that produced it.
    Multi-wavelength values are equal-length tuples paired elementwise
    with their measurements, which carry the wavelengths.
    """

    optical_length: float = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Optical component length along the optical path in meters.",
    )
    name: str = Field(default="", description="Human-readable component name.")
    loss_db: float | tuple[float, ...] | None = Field(
        default=None,
        description="One-way transmission loss across this component in dB.",
    )
    loss_measurement: (
        OpticalMeasurement | str | tuple[OpticalMeasurement | str, ...] | None
    ) = Field(
        default=None,
        description="Measurement record(s) the loss value(s) came from.",
    )
    reflectance_db: float | tuple[float, ...] | None = Field(
        default=None,
        description="Return loss (reflectance) of this component in dB.",
    )
    reflectance_measurement: (
        OpticalMeasurement | str | tuple[OpticalMeasurement | str, ...] | None
    ) = Field(
        default=None,
        description="Measurement record(s) the reflectance value(s) came from.",
    )

    @model_validator(mode="after")
    def _check_measurement_pairing(self) -> Self:
        """Tuple values pair elementwise with tuple measurement records."""
        for quantity in ("loss", "reflectance"):
            value = getattr(self, f"{quantity}_db")
            meas = getattr(self, f"{quantity}_measurement")
            value_seq = isinstance(value, tuple)
            meas_seq = isinstance(meas, tuple)
            if value_seq != meas_seq:
                msg = (
                    f"Multi-valued {quantity}_db requires an equal-length "
                    f"{quantity}_measurement tuple (each value needs the "
                    "record carrying its wavelength), and vice versa."
                )
                raise InvalidInventoryError(msg)
            if value_seq and len(value) != len(meas):
                msg = (
                    f"{quantity}_db has {len(value)} values but "
                    f"{quantity}_measurement has {len(meas)}."
                )
                raise InvalidInventoryError(msg)
        return self


class FiberSegment(_OpticalComponentBase):
    """Length of optical fiber within a cable, patch cord, or other run."""

    type: Literal["FiberSegment"] = _type_tag("FiberSegment")
    container: Cable | str | None = Field(
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

    @property
    def attenuation_db_per_km(self) -> float | tuple[float, ...] | None:
        """The familiar per-length loss rate, derived from loss_db."""
        if self.loss_db is None or not self.optical_length:
            return None
        km = self.optical_length / 1000.0
        if isinstance(self.loss_db, tuple):
            return tuple(x / km for x in self.loss_db)
        return self.loss_db / km


class Connector(_OpticalComponentBase):
    """Optical connector in an optical path."""

    type: Literal["Connector"] = _type_tag("Connector")
    container: Enclosure | str | None = Field(
        default=None, description="Enclosure housing this connector."
    )
    connector_type: str = Field(default="", description="Connector type.")


class Splice(_OpticalComponentBase):
    """Optical splice in an optical path."""

    type: Literal["Splice"] = _type_tag("Splice")
    container: Enclosure | str | None = Field(
        default=None, description="Enclosure housing this splice."
    )
    splice_type: str = Field(default="", description="Splice type, such as fusion.")


class Terminator(_OpticalComponentBase):
    """Optical path terminator."""

    type: Literal["Terminator"] = _type_tag("Terminator")
    container: Enclosure | str | None = Field(
        default=None, description="Enclosure housing this terminator."
    )
    termination_type: str = Field(
        default="", description="Termination type, such as open, capped, or angled."
    )


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
        if not np.all(np.isfinite(self.distance)):
            msg = "Geometry distance values must be finite."
            raise InvalidInventoryError(msg)
        if not _is_strictly_increasing(self.distance):
            msg = "Geometry distance values must be strictly increasing."
            raise InvalidInventoryError(msg)
        if len(self.coordinates) != len(self.distance):
            msg = "Geometry coordinates and distance must have the same length."
            raise InvalidInventoryError(msg)
        dims = {len(coord) for coord in self.coordinates}
        if len(dims) > 1 or 0 in dims:
            msg = "Geometry coordinate points must share one nonzero dimensionality."
            raise InvalidInventoryError(msg)
        return self

    @property
    def interval(self) -> tuple[float, float]:
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
        start, end = self.interval
        inside = (dist >= start) & (dist < end)
        for dim in range(coords.shape[1]):
            out[inside, dim] = np.interp(
                dist[inside], np.asarray(self.distance), coords[:, dim]
            )
        return out


class _IntervalModel(InventoryModel):
    """
    Base for items covering the half-open interval [start, end) of optical
    distance, matching the start/end idiom of time epochs and how interval
    bounds read off OTDR and interrogator displays.

    Equal start and end make the item a point marker (e.g. a clamp or a
    labeled spot): it documents a location but covers no distance, so it
    never participates in coverage, enrichment, or overlap checks.
    """

    start_distance: float = Field(
        allow_inf_nan=False,
        description="Start optical distance of this interval in meters.",
    )
    end_distance: float = Field(
        allow_inf_nan=False,
        description=(
            "End optical distance of this interval in meters; equal to "
            "start_distance for a point marker."
        ),
    )

    @model_validator(mode="after")
    def _check_interval_order(self) -> Self:
        """The end may not precede the start."""
        if self.end_distance < self.start_distance:
            msg = (
                f"end_distance {self.end_distance} must not precede "
                f"start_distance {self.start_distance}."
            )
            raise InvalidInventoryError(msg)
        return self

    @property
    def optical_length(self) -> float:
        """The interval length in meters."""
        return self.end_distance - self.start_distance

    @property
    def interval(self) -> tuple[float, float]:
        """The (start, end) optical distance covered by this item."""
        return (self.start_distance, self.end_distance)


class CouplingCondition(_IntervalModel):
    """
    Acoustic coupling condition for an interval of an optical path.

    Covers ``[start_distance, end_distance)``. Coverage may be partial
    and conditions may not overlap.
    """

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


class OpticalPathAnnotation(_IntervalModel):
    """Named interval on an optical path; annotations may overlap freely."""

    label: str = Field(default="", description="Label for this interval.")


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
        if not (np.all(np.isfinite(source)) and np.all(np.isfinite(self.distance))):
            msg = "DistanceMap control points must be finite."
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
        assert out is not None  # guaranteed by the model validator
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
    On export to FDSN DAS metadata, ``extra_fields`` maps to native_headers.
    """

    code: CodeStr = Field(description="Channel-like acquisition code.")
    location_code: LocationCodeStr = Field(
        default="",
        description=(
            "FDSN-style location code naming the optical path lineage this "
            "acquisition interrogates; may be blank, as in FDSN."
        ),
    )
    data_type: DataType = Field(
        default="", description="Quantity measured or produced."
    )
    data_category: DataCategory = Field(
        default="", description="Acquisition family such as DAS, DTS, or DSS."
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="Units of data produced."
    )
    interrogator: Interrogator | str | None = Field(
        default=None,
        description=(
            "Interrogator used for this acquisition; an object or a "
            "resource_id reference."
        ),
    )
    interrogator_port: str | None = Field(
        default=None,
        description=(
            "Instrument connector or launch port that fed this acquisition, "
            "for multiplexed interrogators. Descriptive only."
        ),
    )
    firmware_version: str = Field(
        default="",
        description="Interrogator firmware version during this acquisition.",
    )
    software_version: str = Field(
        default="",
        description="Acquisition software version during this acquisition.",
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


def _overlapping_epochs(items, key) -> list[tuple]:
    """Return (key, first, second) for time-overlapping items sharing a key."""
    groups: dict = {}
    for item in items:
        groups.setdefault(key(item), []).append(item)
    out = []
    for group_key, group in groups.items():
        for first, second in itertools.combinations(group, 2):
            if first.overlaps(second):
                out.append((group_key, first, second))
    return out


def _intervals_overlap(intervals: list[tuple[float, float]]) -> tuple | None:
    """Return the first overlapping pair of half-open intervals, or None.

    Empty (point) intervals cover nothing and cannot overlap.
    """
    ordered = sorted(x for x in intervals if x[0] < x[1])
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
        allow_inf_nan=False,
        description=(
            "Origin of this path's optical-distance axis in meters; 0 for "
            "whole paths. Set by select and split_at so pieces keep absolute "
            "optical distances."
        ),
    )
    location_code: LocationCodeStr = Field(
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
    measurements: tuple[OpticalMeasurement | str, ...] = Field(
        default=(),
        description="OTDR and other optical measurements of this whole path.",
    )

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
        outer = max(seg.interval[1] for seg in self.geometry)
        at_outer = dist == outer
        if np.any(at_outer):
            segment = max(self.geometry, key=lambda s: s.interval[1])
            coords = np.asarray(segment.coordinates, dtype=float)
            out[at_outer] = coords[-1]
        return out

    def check(self, tolerance: float = 1e-9) -> Self:
        """
        Check track rules for this path.

        Checks that geometry and coupling stay within path bounds and do not
        overlap (partial coverage is legal), and that annotations stay within
        bounds. Component tiling is inherent to the cumulative layout.
        """
        errors = []
        start, end = self.start_distance, self.end_distance
        geo_spans = [seg.interval for seg in self.geometry]
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
        dims = {len(seg.coordinates[0]) for seg in self.geometry}
        if len(dims) > 1:
            errors.append(
                f"Geometry segments mix coordinate dimensionalities {sorted(dims)}."
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
            at_outer = c_lo == hi == self.end_distance
            if new_hi > new_lo or (c_lo == c_hi and (lo <= c_lo < hi or at_outer)):
                length = max(new_hi - new_lo, 0.0)
                components.append(comp.model_copy(update={"optical_length": length}))
        geometry = []
        for seg in self.geometry:
            s_lo, s_hi = seg.interval
            if s_hi <= lo or s_lo >= hi:
                continue
            new_lo, new_hi = max(s_lo, lo), min(s_hi, hi)
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
        the absolute span is unchanged. Because intervals stay half-open on
        the left, exact interval endpoints swap membership under reversal
        (measure-zero; ``reverse().reverse()`` is exact).
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

        def flip_item(item):
            update = {
                "start_distance": flip(item.end_distance),
                "end_distance": flip(item.start_distance),
            }
            return item.model_copy(update=update)

        coupling = sorted(
            (flip_item(c) for c in self.coupling),
            key=lambda c: c.start_distance,
        )
        annotations = sorted(
            (flip_item(a) for a in self.annotations),
            key=lambda a: a.start_distance,
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
            seg.model_copy(update={"distance": tuple(d + offset for d in seg.distance)})
            for seg in other.geometry
        )

        def shift_item(item):
            update = {
                "start_distance": item.start_distance + offset,
                "end_distance": item.end_distance + offset,
            }
            return item.model_copy(update=update)

        coupling = tuple(shift_item(c) for c in other.coupling)
        annotations = tuple(shift_item(a) for a in other.annotations)
        return self.model_copy(
            update={
                "optical_components": (
                    *self.optical_components,
                    *other.optical_components,
                ),
                "geometry": (*self.geometry, *geometry),
                "coupling": (*self.coupling, *coupling),
                "annotations": (*self.annotations, *annotations),
                "measurements": (*self.measurements, *other.measurements),
            }
        )


def _clip_intervals(items, lo: float, hi: float) -> list:
    """Clip start+optical_length interval items to [lo, hi), dropping empties."""
    out = []
    for item in items:
        start, end = item.interval
        new_lo, new_hi = max(start, lo), min(end, hi)
        if new_hi <= new_lo:
            continue
        out.append(
            item.model_copy(update={"start_distance": new_lo, "end_distance": new_hi})
        )
    return out


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


class Channel(TimeRangedModel):
    """Station-level time-series stream identity."""

    code: CodeStr = Field(description="Channel code used in data source identifiers.")
    location_code: LocationCodeStr = Field(
        default="", description="FDSN-style location code; may be blank."
    )
    coordinates: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Coordinates on the canonical (x, y, z) axes; the inventory CRS "
            "coordinate_labels declare their meaning."
        ),
    )
    sample_rate: float | None = Field(default=None, description="Sample rate in Hz.")
    data_type: DataType = Field(
        default="", description="Quantity measured or produced."
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="Units of data produced."
    )
    response: Response | None = Field(
        default=None, description="Optional response metadata."
    )


class Station(TimeRangedModel):
    """Point-like observing identity under a network."""

    code: CodeStr = Field(description="Station code used in data source identifiers.")
    name: str = Field(default="", description="Human-readable station name.")
    coordinates: tuple[float, ...] | None = Field(
        default=None,
        description=(
            "Coordinates on the canonical (x, y, z) axes; the inventory CRS "
            "coordinate_labels declare their meaning."
        ),
    )
    channels: tuple[Channel, ...] = Field(
        default=(), description="Channels associated with this station."
    )


class FiberArray(TimeRangedModel):
    """
    Durable fiber-optic observing identity.

    A fiber array groups similar, co-located fiber routes; each route is a
    location code carrying one optical path at a time.
    """

    code: CodeStr = Field(description="Station-like fiber array code.")
    name: str = Field(default="", description="Human-readable fiber array name.")
    acquisitions: tuple[Acquisition, ...] = Field(
        default=(), description="Acquisitions associated with this fiber array."
    )
    optical_paths: tuple[OpticalPath, ...] = Field(
        default=(), description="Optical paths associated with this fiber array."
    )

    def check(self) -> Self:
        """
        Check epoch rules for this fiber array.

        No more than one optical path per location code may be valid at a
        time, and acquisition (location_code, code) pairs must be unique for
        overlapping time ranges. Contained paths are also validated.
        """
        errors = []
        for loc, first, second in _overlapping_epochs(
            self.optical_paths, lambda x: x.location_code
        ):
            errors.append(
                f"Optical paths {first.name!r} and {second.name!r} "
                f"overlap in time for location {loc!r}."
            )
        for key, *_ in _overlapping_epochs(
            self.acquisitions, lambda x: (x.location_code, x.code)
        ):
            errors.append(f"Acquisition epochs for {key} overlap in time.")
        if errors:
            msg = f"Fiber array {self.code!r} validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        for path in self.optical_paths:
            path.check()
        return self


class Network(InventoryModel):
    """FDSN-like organizational container below an inventory."""

    code: CodeStr = Field(description="Network code.")
    name: str = Field(default="", description="Human-readable network name.")
    description: str = Field(default="", description="Network description.")
    fiber_arrays: tuple[FiberArray, ...] = Field(
        default=(), description="Fiber arrays in this network."
    )
    stations: tuple[Station, ...] = Field(
        default=(), description="Stations in this network."
    )

    def check(self) -> Self:
        """
        Check code rules for this network.

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
        for kind, items in (
            ("fiber array", self.fiber_arrays),
            ("station", self.stations),
        ):
            for code, *_ in _overlapping_epochs(items, lambda x: x.code):
                errors.append(
                    f"Duplicate {kind} code {code!r} for overlapping time ranges."
                )
        if errors:
            msg = f"Network {self.code!r} validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        for array in self.fiber_arrays:
            array.check()
        return self


def _drop_empty(value, _in_extras=False):
    """
    Recursively drop empty strings, mappings, and sequences from a dump.

    All model fields default to empty values, so pruning them is lossless on
    reload. User-supplied ``extra_fields`` contents are kept verbatim.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            pruned = (
                item
                if _in_extras
                else _drop_empty(item, _in_extras=key == "extra_fields")
            )
            if pruned in ("", {}, []) and not _in_extras:
                continue
            out[key] = pruned
        return out
    if isinstance(value, list):
        return [_drop_empty(item) for item in value]
    return value


class ResolvedContext(NamedTuple):
    """The inventory objects a data_source_id + time resolve to."""

    network: Network
    fiber_array: FiberArray
    acquisition: Acquisition
    optical_path: OpticalPath | None


class Inventory(InventoryModel):
    """
    Top-level DASDAE inventory manifest.

    Stores document metadata, shareable resources keyed by resource_id, the
    inventory-wide coordinate reference system, and network containers.
    """

    schema_version: int = Field(
        default=1, description="Version of the inventory manifest envelope."
    )
    resource_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="Identifier for the inventory manifest.",
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
            out = {}
            for key, resource in value.items():
                if isinstance(resource, dict):
                    rid = resource.get("resource_id")
                    if rid is None:
                        resource = {**resource, "resource_id": key}
                        rid = key
                else:
                    rid = getattr(resource, "resource_id", None)
                if rid is not None and rid != key:
                    msg = f"Resource key {key!r} disagrees with resource_id {rid!r}."
                    raise InvalidInventoryError(msg)
                out[key] = resource
            return out
        out = {}
        for resource in value:
            if resource.resource_id in out:
                msg = f"Duplicate resource_id {resource.resource_id!r}."
                raise InvalidInventoryError(msg)
            out[resource.resource_id] = resource
        return out

    @model_validator(mode="after")
    def _normalize_resources(self) -> Self:
        """
        Normalize shareable resources into the flat pool.

        Inline resource objects anywhere in the tree move into ``resources``
        (keyed by ``resource_id``) and the fields that held them keep the id
        string. Two inline definitions of one id must be equal or this
        raises; id references that resolve to nothing raise.
        """
        pool: dict[str, Any] = {}
        string_refs: list[tuple[str, str, tuple]] = []
        measurement_refs = {
            "loss_measurement": (OpticalMeasurement,),
            "reflectance_measurement": (OpticalMeasurement,),
        }
        ref_fields = {
            FiberSegment: {"container": (Cable,), **measurement_refs},
            Connector: {"container": (Enclosure,), **measurement_refs},
            Splice: {"container": (Enclosure,), **measurement_refs},
            Terminator: {"container": (Enclosure,), **measurement_refs},
            Cable: {
                "container": (Enclosure, Cable),
                "specification": (ExternalResource,),
            },
            Enclosure: {"specification": (ExternalResource,)},
            Acquisition: {"interrogator": (Interrogator,)},
            OpticalMeasurement: {"data": (ExternalResource,)},
        }

        def pool_add(rid, resource):
            if rid in pool and pool[rid] != resource:
                msg = f"Resource {rid!r} is defined twice with different content."
                raise InvalidInventoryError(msg)
            pool[rid] = resource

        def register(value, field="resource", allowed=()):
            if value is None:
                return None
            if isinstance(value, str):
                string_refs.append((value, field, allowed))
                return value
            pool_add(value.resource_id, normalize(value))
            return value.resource_id

        def normalize(obj):
            """Rewrite an object's resource-valued fields to id references."""
            fields = next(
                (f for cls, f in ref_fields.items() if isinstance(obj, cls)), {}
            )
            updates = {}
            for field, allowed in fields.items():
                value = getattr(obj, field)
                if isinstance(value, tuple):
                    new_value = tuple(register(x, field, allowed) for x in value)
                    if new_value == value:
                        continue
                else:
                    new_value = register(value, field, allowed)
                    if new_value is value:
                        continue
                updates[field] = new_value
            return obj.model_copy(update=updates) if updates else obj

        for rid, resource in self.resources.items():
            pool_add(rid, normalize(resource))

        def norm_path(path):
            return path.model_copy(
                update={
                    "optical_components": tuple(
                        normalize(c) for c in path.optical_components
                    ),
                    "measurements": tuple(
                        register(x, "measurements", (OpticalMeasurement,))
                        for x in path.measurements
                    ),
                }
            )

        networks = tuple(
            net.model_copy(
                update={
                    "fiber_arrays": tuple(
                        arr.model_copy(
                            update={
                                "acquisitions": tuple(
                                    normalize(a) for a in arr.acquisitions
                                ),
                                "optical_paths": tuple(
                                    norm_path(p) for p in arr.optical_paths
                                ),
                            }
                        )
                        for arr in net.fiber_arrays
                    ),
                }
            )
            for net in self.networks
        )
        dangling = sorted({r for r, *_ in string_refs if r not in pool})
        if dangling:
            msg = f"Dangling resource references: {dangling}."
            raise InvalidInventoryError(msg)
        for rid, field, allowed in string_refs:
            if allowed and not isinstance(pool[rid], allowed):
                names = tuple(x.__name__ for x in allowed)
                msg = (
                    f"Resource reference {rid!r} for {field!r} resolves to "
                    f"{type(pool[rid]).__name__}, expected one of {names}."
                )
                raise InvalidInventoryError(msg)
        object.__setattr__(self, "resources", pool)
        object.__setattr__(self, "networks", networks)
        self.__pydantic_fields_set__.update({"resources", "networks"})
        return self

    def get_resource(self, resource_id: str):
        """Return the shareable resource registered under a resource_id."""
        try:
            return self.resources[resource_id]
        except KeyError:
            msg = f"No resource with resource_id {resource_id!r}."
            raise InvalidInventoryError(msg) from None

    def check(self) -> Self:
        """Check the whole inventory tree against the model rules."""
        codes = [net.code for net in self.networks]
        if len(codes) != len(set(codes)):
            msg = f"Network codes must be unique; got {codes}."
            raise InvalidInventoryError(msg)
        for net in self.networks:
            net.check()
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

        def exactly_one(matches, kind):
            if len(matches) != 1:
                msg = f"{data_source_id!r} resolves to {len(matches)} {kind}."
                raise InvalidInventoryError(msg)
            return matches[0]

        network = exactly_one(
            [x for x in self.networks if x.code == net_code], "networks"
        )
        array = exactly_one(
            [
                x
                for x in network.fiber_arrays
                if x.code == array_code and x.is_effective_at(time)
            ],
            "fiber arrays",
        )
        acqs = [
            exactly_one(
                [
                    x
                    for x in array.acquisitions
                    if x.code == acq_code
                    and x.location_code == location
                    and x.is_effective_at(time)
                ],
                "acquisitions",
            )
        ]
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
        retroactively. ``old`` is matched by equality at any addressable
        level: pooled resources, networks, stations, channels, fiber arrays,
        acquisitions, optical paths, and path track items (components,
        geometry, coupling, annotations). Singletons such as the CRS or a
        distance map are corrected with ``new()`` on their parent. ``new``
        must be the same type as ``old``. Note that resource normalization
        rewrites inline resource objects to id references at construction, so
        match against the stored (normalized) object — e.g. an acquisition
        whose ``interrogator`` is the id string — not a pre-construction
        handle holding the inline object.
        """
        if type(new) is not type(old):
            msg = (
                f"Replacement type {type(new).__name__} does not match "
                f"{type(old).__name__}."
            )
            raise InvalidInventoryError(msg)
        if isinstance(old, Interrogator | Cable | Enclosure | ExternalResource):
            matches = [r for r, res in self.resources.items() if res == old]
            if not matches:
                msg = "Component to replace was not found in the inventory."
                raise InvalidInventoryError(msg)
            if new.resource_id != old.resource_id:
                msg = (
                    "Resource corrections must keep the same resource_id "
                    f"({old.resource_id!r} != {new.resource_id!r}); id "
                    "references throughout the tree would dangle."
                )
                raise InvalidInventoryError(msg)
            pool = dict(self.resources)
            for rid in matches:
                pool[rid] = new
            return self.new(resources=pool)
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
                paths = []
                for path in swap(array.optical_paths):
                    if path == new:
                        paths.append(path)
                        continue
                    paths.append(
                        path.model_copy(
                            update={
                                "optical_components": swap(path.optical_components),
                                "geometry": swap(path.geometry),
                                "coupling": swap(path.coupling),
                                "annotations": swap(path.annotations),
                            }
                        )
                    )
                arrays.append(
                    array.model_copy(
                        update={
                            "acquisitions": swap(array.acquisitions),
                            "optical_paths": tuple(paths),
                        }
                    )
                )
            stations = []
            for station in swap(net.stations):
                if station == new:
                    stations.append(station)
                    continue
                stations.append(
                    station.model_copy(update={"channels": swap(station.channels)})
                )
            networks.append(
                net.model_copy(
                    update={
                        "fiber_arrays": tuple(arrays),
                        "stations": tuple(stations),
                    }
                )
            )
        if not replaced:
            msg = "Component to replace was not found in the inventory."
            raise InvalidInventoryError(msg)
        return self.new(networks=tuple(networks))

    def to_yaml(self, path=None) -> str:
        """Serialize this inventory to YAML, optionally writing to a path."""
        yaml = optional_import("yaml", required_for="YAML inventory serialization")

        data = _drop_empty(self.model_dump(mode="json", exclude_none=True))
        out = yaml.safe_dump(data, sort_keys=False)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(out)
        return out

    @classmethod
    def from_yaml(cls, source) -> Self:
        """
        Load an inventory from a YAML string or file path.

        Parameters
        ----------
        source
            A path to a YAML file, or YAML text (must contain a newline to
            be treated as text rather than a missing path).
        """
        import os

        text = source
        if isinstance(source, os.PathLike) or (
            isinstance(source, str) and "\n" not in source
        ):
            if not os.path.exists(source):
                msg = f"No such inventory file: {source!r}."
                raise InvalidInventoryError(msg)
            with open(source) as fh:
                text = fh.read()
        yaml = optional_import("yaml", required_for="YAML inventory serialization")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            msg = f"Could not parse an inventory mapping from {source!r}."
            raise InvalidInventoryError(msg)
        return cls(**data)


def inventory(source=None) -> Inventory:
    """
    Load or create a DASDAE inventory.

    Parameters
    ----------
    source
        An existing Inventory (returned as is), a YAML file path or YAML
        text, or None for an empty inventory.

    Examples
    --------
    >>> import dascore as dc
    >>> empty = dc.inventory()
    >>> assert dc.inventory(empty) is empty
    """
    import os

    if source is None:
        return Inventory()
    if isinstance(source, Inventory):
        return source
    if isinstance(source, str | os.PathLike):
        return Inventory.from_yaml(source)
    msg = f"Could not get an inventory from {source!r}."
    raise InvalidInventoryError(msg)
