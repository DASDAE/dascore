"""
DASDAE Inventory: DASCore's metadata model for DFOS observing systems.

The inventory extends the StationXML concept with first-class support for
fiber-optic arrays. It describes the physical optical path (fiber, connectors,
splices), the geometry, coupling, and annotation tracks along optical
distance, and the interrogator configurations (acquisitions) that produced
patches. Patches carry a ``acquisition_key``
(``network.fiber_array.location.acquisition``) which, together with time,
resolves against an inventory.

Each object documents the rules it enforces.
"""

from __future__ import annotations

import itertools
import os
from collections.abc import Mapping, Sized
from functools import cache
from types import MappingProxyType, UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    TypeAlias,
    Union,
    get_args,
    get_origin,
)
from uuid import uuid4

import numpy as np
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    field_validator,
    model_validator,
)
from typing_extensions import Self

from dascore.constants import DataCategory, DataType
from dascore.exceptions import InvalidInventoryError, ParameterError
from dascore.models import (
    DateTime64,
    FiniteFloat,
    FrozenDictType,
    InventoryModel,
    TimeRangedModel,
    UnitQuantity,
)
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    check_code,
    is_strictly_monotonic,
    optional_import,
    validate_acquisition_key,
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
CoordinateLabel = Literal[
    "x",
    "y",
    "z",
    "latitude",
    "longitude",
    "elevation",
    "depth",
    "easting",
    "northing",
]
VALID_COORDINATE_LABELS = get_args(CoordinateLabel)

# Code tokens used in acquisition_key; location codes alone may be blank.
# The token rule is shared with PatchAttrs.acquisition_key so a code legal
# in one is legal in the other.
CodeStr = Annotated[str, AfterValidator(check_code)]
# Sensor orientation, in the ranges seismology already uses.
Azimuth = Annotated[float, Field(ge=0, lt=360, allow_inf_nan=False)]
Dip = Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
LocationCodeStr = Annotated[
    str, AfterValidator(lambda value: check_code(value, allow_blank=True))
]
# Stable identifier for a shareable inventory resource.
ResourceIdStr = Annotated[
    str,
    Field(
        default_factory=lambda: str(uuid4()),
        description="Stable identifier for this shareable inventory resource.",
    ),
]


def _annotation_value(value):
    """
    Normalize an annotation value so its Python type survives validation.

    Numpy scalars are unwrapped: pydantic's smart union resolves every numpy
    scalar to float, which would turn a mask element into a numeric group.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        msg = f"Annotation value must be finite; got {value}."
        raise InvalidInventoryError(msg)
    return value


# The value kind decides an annotation group's shape, so it must be exact.
AnnotationValue = Annotated[
    str | bool | int | float, BeforeValidator(_annotation_value)
]


def _object_type_tag(name: str):
    """
    Return the serialization-only ``object_type`` field of a union member.

    Every model states its class when serialized (see
    [dascore.models.registry](`dascore.models.registry`)), but pydantic must
    pick a class for these before an object exists, so the models sharing a
    union declare the tag as a real field and the base class leaves them
    alone. Users never set it: it defaults to the class name, the Literal
    annotation rejects any other value, and it is hidden from repr.
    """
    return Field(default=name, repr=False)


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
    is geographic WGS84 3D (EPSG:4979).
    """

    authority: str = Field(default="EPSG", description="CRS authority.")
    code: str = Field(default="4979", description="Authority code for this CRS.")
    name: str = Field(default="WGS 84 3D", description="Human-readable CRS name.")
    coordinate_labels: tuple[CoordinateLabel, ...] = Field(
        default=("longitude", "latitude", "elevation"),
        description=(
            "Meaning of the canonical (x, y, z) axes, in order. Labels come "
            "from the controlled coordinate vocabulary."
        ),
    )
    units: tuple[str, ...] = Field(
        default=("degree", "degree", "meter"),
        description=(
            "Units of the canonical axes, in order; same length as coordinate_labels."
        ),
    )
    vertical_datum: str = Field(
        default="", description="Vertical datum or reference surface, if known."
    )
    wkt: str = Field(
        default="",
        description=(
            "WKT2 definition for frames an authority code cannot describe "
            "(local, derived, or compound CRSs); empty for registry CRSs."
        ),
    )

    @field_validator("coordinate_labels")
    @classmethod
    def _check_labels(cls, value):
        """Labels are unique; the vocabulary itself is enforced by the type."""
        if len(set(value)) != len(value):
            msg = f"coordinate_labels must be unique; got {value}."
            raise InvalidInventoryError(msg)
        return value

    @field_validator("coordinate_labels")
    @classmethod
    def _check_label_count(cls, value):
        """Coordinates are stored on the canonical (x, y, z) axes."""
        if not 1 <= len(value) <= 3:
            msg = (
                "A CRS declares one to three axes, matching the canonical "
                f"(x, y, z) storage; got {value}."
            )
            raise InvalidInventoryError(msg)
        return value

    @model_validator(mode="after")
    def _check_units_length(self) -> Self:
        """Units pair with the axes, one per coordinate label."""
        if len(self.units) != len(self.coordinate_labels):
            msg = (
                f"units {self.units} must have one entry per coordinate "
                f"label {self.coordinate_labels}."
            )
            raise InvalidInventoryError(msg)
        return self

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

    object_type: Literal["ExternalResource"] = _object_type_tag("ExternalResource")
    resource_id: ResourceIdStr
    uri: str = Field(default="", description="URI or identifier for the resource.")
    name: str = Field(default="", description="Human-readable resource name.")


class OpticalMeasurement(InventoryModel):
    """
    A characterization of the optical path or its components.

    One record captures the conditions a loss or reflectance number was
    obtained under — the conditions live here, stated once, and every
    number derived from the same run references the same record. A
    datasheet claim is a legitimate record (method="datasheet").
    """

    object_type: Literal["OpticalMeasurement"] = _object_type_tag("OpticalMeasurement")
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
    wavelength: FiniteFloat | None = Field(
        default=None, description="Measurement wavelength in nm."
    )
    pulse_width: FiniteFloat | None = Field(
        default=None, description="OTDR pulse width in seconds."
    )
    direction: str = Field(
        default="", description="Measurement direction: forward or reverse."
    )
    data: ExternalResource | str | None = Field(
        default=None, description="The trace or report file, as a resource."
    )


class Interrogator(InventoryModel):
    """DFOS interrogator unit used for data collection."""

    object_type: Literal["Interrogator"] = _object_type_tag("Interrogator")
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

    object_type: Literal["Enclosure"] = _object_type_tag("Enclosure")
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
    inner_diameter: FiniteFloat | None = Field(
        default=None, description="Inner diameter in meters."
    )
    specification: ExternalResource | str | None = Field(
        default=None, description="External specification or datasheet."
    )


class Cable(InventoryModel):
    """Physical cable containing one or more fiber segments."""

    object_type: Literal["Cable"] = _object_type_tag("Cable")
    resource_id: ResourceIdStr
    name: str = Field(default="", description="Human-readable resource name.")
    manufacturer: str = Field(default="", description="Manufacturer name.")
    model: str = Field(default="", description="Model name.")
    diameter: FiniteFloat | None = Field(
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
    Field(discriminator="object_type"),
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

    _identity_field: ClassVar[str] = "name"
    optical_length: FiniteFloat = Field(
        default=0.0,
        ge=0.0,
        allow_inf_nan=False,
        description="Optical component length along the optical path in meters.",
    )
    name: str = Field(default="", description="Human-readable component name.")
    loss_db: FiniteFloat | tuple[FiniteFloat, ...] | None = Field(
        default=None,
        description="One-way transmission loss across this component in dB.",
    )
    loss_measurement: (
        OpticalMeasurement | str | tuple[OpticalMeasurement | str, ...] | None
    ) = Field(
        default=None,
        description="Measurement record(s) the loss value(s) came from.",
    )
    reflectance_db: FiniteFloat | tuple[FiniteFloat, ...] | None = Field(
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

    object_type: Literal["FiberSegment"] = _object_type_tag("FiberSegment")
    container: Cable | str | None = Field(
        default=None, description="Cable containing this fiber."
    )
    fiber_number: int | None = Field(
        default=None, description="Fiber position within the parent cable."
    )
    fiber_color: str | None = Field(default=None, description="Fiber color code.")
    fiber_type: str = Field(
        default="", description="Fiber type, such as single_mode or multi_mode."
    )
    fiber_standard: str = Field(
        default="", description="Fiber standard or grade, such as ITU-T G.652.D."
    )
    refractive_index: FiniteFloat | None = Field(
        default=None,
        description=(
            "Effective group index used to convert optical time of flight to "
            "distance; the OTDR group index setting."
        ),
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

    object_type: Literal["Connector"] = _object_type_tag("Connector")
    container: Enclosure | str | None = Field(
        default=None, description="Enclosure housing this connector."
    )
    connector_type: str = Field(default="", description="Connector type.")


class Splice(_OpticalComponentBase):
    """Optical splice in an optical path."""

    object_type: Literal["Splice"] = _object_type_tag("Splice")
    container: Enclosure | str | None = Field(
        default=None, description="Enclosure housing this splice."
    )
    splice_type: str = Field(default="", description="Splice type, such as fusion.")


class Terminator(_OpticalComponentBase):
    """Optical path terminator."""

    object_type: Literal["Terminator"] = _object_type_tag("Terminator")
    container: Enclosure | str | None = Field(
        default=None, description="Enclosure housing this terminator."
    )
    termination_type: str = Field(
        default="", description="Termination type, such as open, capped, or angled."
    )


OpticalComponent: TypeAlias = Annotated[
    FiberSegment | Connector | Splice | Terminator,
    Field(discriminator="object_type"),
]


class Geometry(InventoryModel):
    """
    Geometry for an interval of an optical path.

    A geometry is a piecewise segment placed by its ``distance`` array: at
    least two strictly increasing optical distances, each paired with the
    coordinate at that point (interpreted using the inventory CRS). Coverage
    is the half-open span of the array; there is no separate length field. A
    coil, or other "clump", is a segment whose coordinates repeat while
    distance advances.
    Interpolation between points is piecewise linear in the CRS and never
    crosses segments; uncovered distance has undefined coordinates.
    """

    _identity_field: ClassVar[str] = "name"
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
        _check_control_points(self.distance, "Geometry distance", minimum=2)
        if len(self.coordinates) != len(self.distance):
            msg = "Geometry coordinates and distance must have the same length."
            raise InvalidInventoryError(msg)
        dims = {len(coord) for coord in self.coordinates}
        if len(dims) > 1 or 0 in dims:
            msg = "Geometry coordinate points must share one nonzero dimensionality."
            raise InvalidInventoryError(msg)
        if not np.all(np.isfinite(np.asarray(self.coordinates, dtype=float))):
            msg = "Geometry coordinate values must be finite."
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


def interval_masks(values, intervals) -> list[np.ndarray]:
    """
    Return, per interval, the mask of values that interval covers.

    Coverage is half-open, ``[start, end)``, with one exception: the end of
    a coverage run belongs to the interval ending there when no half-open
    interval claims it, so the last point of a run is not left out. Point
    markers (equal start and end) cover nothing.
    """
    values = np.asarray(values, dtype=float)
    spans = [(lo, hi) for lo, hi in intervals]
    claimed = np.zeros(len(values), dtype=bool)
    for lo, hi in spans:
        if lo < hi:
            claimed |= (values >= lo) & (values < hi)
    out = []
    for lo, hi in spans:
        if lo >= hi:  # a point marker covers nothing
            out.append(np.zeros(len(values), dtype=bool))
            continue
        mask = (values >= lo) & (values < hi)
        out.append(mask | ((values == hi) & ~claimed))
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

    _identity_field: ClassVar[str] = "coupling_type"
    coupling_type: CouplingType = Field(description="Controlled coupling category.")
    medium: str = Field(default="", description="Surrounding medium.")
    attachment: str = Field(default="", description="Attachment method.")
    depth: FiniteFloat | None = Field(
        default=None,
        description=(
            "Depth in meters, positive down, relative to the local surface, "
            "when relevant."
        ),
    )


class OpticalPathAnnotation(_IntervalModel):
    """
    Key/value annotation attached to an interval of an optical path.

    ``group`` names the variable and ``value`` is its state over the
    interval, so a bare flag is simply a boolean value. String and numeric
    groups are single valued and their intervals may not overlap; boolean
    groups state membership and may overlap freely. Point markers (equal
    start and end) cover nothing, so they are exempt from that rule.
    """

    group: str = Field(default="", description="Name of the annotated variable.")
    value: AnnotationValue = Field(
        default=True, description="Value of the variable over this interval."
    )

    @field_validator("value")
    @classmethod
    def _reject_empty_string(cls, value):
        """An annotation whose value is empty states nothing.

        It would also be indistinguishable from an uncovered channel, since
        a string coordinate spells absence as the empty string.
        """
        if isinstance(value, str) and not value:
            msg = (
                "An annotation value may not be the empty string; it would "
                "state nothing and would read as an uncovered channel."
            )
            raise ValueError(msg)
        return value


# The coordinates a DistanceMap may be written in, in preference order.
DISTANCE_MAP_AXES = ("channel", "instrument_distance")


def _check_control_points(values, what: str, minimum: int = 1) -> None:
    """Check an axis of control points: enough of them, finite, increasing."""
    if len(values) < minimum:
        plural = "" if minimum == 1 else "s"
        msg = f"{what} requires at least {minimum} control point{plural}."
        raise InvalidInventoryError(msg)
    if not np.all(np.isfinite(values)):
        msg = f"{what} values must be finite."
        raise InvalidInventoryError(msg)
    if not is_strictly_monotonic(values, increasing=True):
        msg = f"{what} values must be strictly increasing."
        raise InvalidInventoryError(msg)


class DistanceMap(InventoryModel):
    """
    Measured control-point map from a channel-like coordinate onto optical
    path distance.

    The map is a function, not a roster of existing channels. It states one
    set of control points in whichever input coordinates were measured:
    ``channel`` when the interrogator reports channel numbers,
    ``instrument_distance`` when it reports its own nominal meters, or both
    when the same points are known in both, which lets one acquisition
    serve patches whose axes differ. Values between control points are
    piecewise-linearly interpolated; channels outside the covered range are
    undefined (NaN). A single-point map states an origin and takes its
    slope from the axis it is read on: the acquisition's nominal
    ``spatial_interval`` on the channel axis, and one meter of path per
    interrogator meter on the instrument_distance axis.
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
        description=(
            "Optical path distances at the control points; strictly increasing."
        ),
    )

    @model_validator(mode="after")
    def _validate_map(self) -> Self:
        """Enforce paired, increasing control points on every input axis."""
        if not self.axes:
            msg = (
                "DistanceMap requires at least one input axis: "
                "channel or instrument_distance."
            )
            raise InvalidInventoryError(msg)
        _check_control_points(self.distance, "DistanceMap distance")
        for axis in self.axes:
            source = getattr(self, axis)
            if len(source) != len(self.distance):
                msg = (
                    f"DistanceMap {axis} and distance must have the same "
                    "length; they are the same control points."
                )
                raise InvalidInventoryError(msg)
            _check_control_points(source, f"DistanceMap {axis}")
        self._check_axes_agree()
        return self

    def _check_axes_agree(self) -> None:
        """
        Check that two input axes describe one interrogator.

        An interrogator samples at a fixed spacing, so its channel numbers
        and its own meters are related by a constant. Axes which imply a
        varying spacing describe no instrument, and reading the map on one
        axis would then contradict reading it on the other.
        """
        if len(self.axes) < 2 or len(self.distance) < 3:
            return
        channel = np.asarray(self.channel, dtype=float)
        instrument = np.asarray(self.instrument_distance, dtype=float)
        ratios = np.diff(instrument) / np.diff(channel)
        if not np.allclose(ratios, ratios[0], rtol=1e-6, atol=0):
            msg = (
                "DistanceMap channel and instrument_distance imply a channel "
                f"spacing which varies along the fiber ({ratios.min()} to "
                f"{ratios.max()} interrogator meters per channel); one "
                "interrogator samples at a fixed spacing."
            )
            raise InvalidInventoryError(msg)

    @property
    def axes(self) -> tuple[str, ...]:
        """The input axes this map is written in, in preference order."""
        return tuple(x for x in DISTANCE_MAP_AXES if getattr(self, x) is not None)

    def source_values(self, axis: str | None = None) -> tuple[float, ...]:
        """Return the control points on one input axis."""
        axis = self.axes[0] if axis is None else axis
        if axis not in DISTANCE_MAP_AXES:
            msg = (
                f"{axis!r} is not a DistanceMap input axis; the axes are "
                f"{DISTANCE_MAP_AXES}."
            )
            raise InvalidInventoryError(msg)
        out = getattr(self, axis)
        if out is None:
            msg = f"This DistanceMap is not written in {axis!r}; it has {self.axes}."
            raise InvalidInventoryError(msg)
        return out

    def map_to_distance(
        self, values, axis: str | None = None, slope: float | None = None
    ) -> np.ndarray:
        """
        Map channel-like values onto optical path distance.

        Parameters
        ----------
        values
            Channel numbers or instrument distances.
        axis
            The input axis ``values`` are on; the map's first axis by
            default.
        slope
            Distance per input unit, used only for single-point maps. The
            caller supplies it in the units of that axis; see
            ``Acquisition.channel_to_distance``.
        """
        vals = np.atleast_1d(np.asarray(values, dtype=float))
        source = np.asarray(self.source_values(axis), dtype=float)
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
    Time-aware, channel-like DFOS acquisition setup.

    ``(location_code, code)`` pairs are unique within a ``FiberArray`` for
    overlapping time ranges. The ``location_code`` names the optical path
    lineage this acquisition interrogates. The ``distance_map`` is the one
    channel-resolution mechanism: a single control point states an origin
    (and, on the channel axis, takes ``spatial_interval`` as its slope),
    and more points describe a measured, bending relationship.
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
    gauge_length: FiniteFloat | None = Field(
        default=None, description="Gauge length in meters."
    )
    pulse_rate: FiniteFloat | None = Field(
        default=None, description="Pulse repetition rate in Hz."
    )
    pulse_width: FiniteFloat | None = Field(
        default=None, description="Pulse width in seconds."
    )
    sample_rate: FiniteFloat | None = Field(
        default=None, description="FDSN-style acquisition sample rate in Hz."
    )
    spatial_interval: FiniteFloat | None = Field(
        default=None,
        description=(
            "Nominal spatial sampling interval between channels in meters. "
            "Participates in channel resolution only as the slope of a "
            "single-point distance_map read on the channel axis."
        ),
    )
    distance_map: DistanceMap | None = Field(
        default=None,
        description=(
            "Map from interrogator-reported channels or distances onto "
            "optical path distance."
        ),
    )
    closed_fiber_loop: bool = Field(
        default=False,
        description="True when the interrogator is attached to both path ends.",
    )

    @model_validator(mode="before")
    @classmethod
    def _reject_start_distance(cls, data):
        """Explain the removed affine form rather than 'extra inputs'."""
        if isinstance(data, Mapping) and "start_distance" in data:
            msg = (
                "Acquisition no longer takes start_distance; the distance_map "
                "is the one channel-resolution mechanism. Write "
                f"start_distance: {data['start_distance']} as distance_map: "
                f"{{channel: [0], distance: [{data['start_distance']}]}}, which "
                "states the same origin and takes spatial_interval as its slope."
            )
            raise InvalidInventoryError(msg)
        return data

    def channel_to_distance(self, values, axis: str | None = None) -> np.ndarray:
        """
        Map channel-like patch coordinates onto optical path distance.

        Parameters
        ----------
        values
            The interrogator-reported coordinate to place on the path.
        axis
            Which of the map's input axes ``values`` are on; the map's
            first axis by default.
        """
        if (dist_map := self.distance_map) is None:
            msg = (
                f"Acquisition {self.code!r} defines no distance_map, so its "
                "channels cannot be placed on the optical path."
            )
            raise InvalidInventoryError(msg)
        axis = dist_map.axes[0] if axis is None else axis
        # The slope carries the units of the axis being read: meters of path
        # per channel on the channel axis, and per interrogator meter --
        # nominally one -- on the instrument_distance axis.
        slope = self.spatial_interval if axis == "channel" else 1.0
        return dist_map.map_to_distance(values, axis=axis, slope=slope)


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


_MIXED_DIMS_MSG = "Geometry segments mix coordinate dimensionalities {dims}."


def _track_identity_fields() -> Mapping[str, str]:
    """
    Map each typed track of an optical path to the field its name means.

    `coupling="trench"` asks about coupling_type; every other field of a
    track is reached by its qualified name (`coupling.medium`). Each
    track model declares which of its fields is its identity, so the
    pairing lives beside the field rather than in a list to keep in step
    with it.
    """
    out = {}
    for track, info in OpticalPath.model_fields.items():
        # A track is a tuple of items; the path's scalar fields are not.
        if get_origin(info.annotation) is not tuple:
            continue
        for model in _annotation_members(get_args(info.annotation)[0]):
            field = getattr(model, "_identity_field", None)
            if field is None:
                continue
            # The model names one of its own fields, or the map it builds
            # would point at nothing.
            assert field in model.model_fields, (model, field)
            out[track] = field
    return MappingProxyType(out)


# Names an annotation group may not take: a group becomes a patch coordinate
# at enrichment, where it would shadow one of these.
RESERVED_GROUP_NAMES = frozenset(
    {"time", "distance", "channel", "instrument_distance"}
    | {"optical_components", "geometry", "coupling", "annotations"}
    | set(VALID_COORDINATE_LABELS)
)


def _times_equal(time1, time2) -> bool:
    """Compare two epoch times, treating unset (NaT) times as equal."""
    null1, null2 = np.isnat(time1), np.isnat(time2)
    if null1 or null2:
        return bool(null1 and null2)
    return bool(time1 == time2)


def _annotation_kind(value) -> str:
    """Return the value kind which decides an annotation group's shape."""
    if isinstance(value, bool):  # bool before int; bool is an int subclass
        return "boolean"
    if isinstance(value, str):
        return "string"
    return "numeric"


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
    no overlap); annotations overlap as their group's value kind allows. No
    more than one path per ``(FiberArray, location_code)`` is valid at a
    time.
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

        Uncovered distance returns NaN rows. Segment coverage is half-open,
        with the end of each coverage run included: a distance on a
        segment's last control point belongs to that segment unless another
        segment claims it.
        """
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        if not self.geometry:
            return np.full((len(dist), 1), np.nan)
        dims = {len(seg.coordinates[0]) for seg in self.geometry}
        if len(dims) > 1:
            raise InvalidInventoryError(_MIXED_DIMS_MSG.format(dims=sorted(dims)))
        out = np.full((len(dist), dims.pop()), np.nan)
        masks = interval_masks(dist, [x.interval for x in self.geometry])
        for segment, mask in zip(self.geometry, masks, strict=True):
            if not np.any(mask):
                continue
            # interpolate() reports its own coverage, which excludes the
            # run end the mask includes, so fill that from the last point.
            seg_coords = segment.interpolate(dist[mask])
            last = np.asarray(segment.coordinates, dtype=float)[-1]
            seg_coords[np.isnan(seg_coords[:, 0])] = last
            out[mask] = seg_coords
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
            errors.append(_MIXED_DIMS_MSG.format(dims=sorted(dims)))
        for name, spans in (("geometry", geo_spans), ("coupling", coup_spans)):
            overlap = _intervals_overlap(spans)
            if overlap is not None:
                errors.append(
                    f"Overlapping {name} intervals {overlap[0]} and "
                    f"{overlap[1]}; {name} is a function track."
                )
        errors.extend(self._check_annotation_groups())
        if errors:
            msg = "Optical path validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        return self

    def _check_annotation_groups(self) -> list[str]:
        """Check that each annotation group holds one kind of value."""
        groups: dict[str, list] = {}
        for annotation in self.annotations:
            groups.setdefault(annotation.group, []).append(annotation)
        errors = []
        for group in sorted(set(groups) & RESERVED_GROUP_NAMES):
            errors.append(
                f"Annotation group {group!r} is a reserved name; a group "
                "becomes a coordinate and cannot shadow a structural "
                "coordinate, a typed track, or a coordinate label."
            )
        for group, items in groups.items():
            kinds = {_annotation_kind(x.value) for x in items}
            if len(kinds) > 1:
                errors.append(
                    f"Annotation group {group!r} mixes {sorted(kinds)} values; "
                    "a group holds one kind of value."
                )
                continue
            if kinds == {"boolean"}:  # membership groups may overlap
                continue
            overlap = _intervals_overlap([x.interval for x in items])
            if overlap is not None:
                errors.append(
                    f"Overlapping intervals {overlap[0]} and {overlap[1]} in "
                    f"annotation group {group!r}; only boolean groups, which "
                    "state membership, may overlap."
                )
        return errors

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
            self.optical_components, self.component_intervals(), strict=True
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
        outer = self.end_distance
        coupling = _clip_intervals(self.coupling, lo, hi, outer)
        annotations = _clip_intervals(self.annotations, lo, hi, outer)
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
        """
        Concatenate paths, rewriting the second onto the combined axis.

        Both paths must share a lineage and an epoch: the result carries the
        left path's ``location_code``, ``start_time``, and ``end_time``, so
        combining across either would misattribute the right path's metadata.
        """
        if not isinstance(other, OpticalPath):
            return NotImplemented
        differing = [
            name
            for name, same in (
                ("location_code", self.location_code == other.location_code),
                ("start_time", _times_equal(self.start_time, other.start_time)),
                ("end_time", _times_equal(self.end_time, other.end_time)),
            )
            if not same
        ]
        if differing:
            msg = (
                "Optical paths are only concatenable within one lineage and "
                f"epoch; {differing} differ. The result would advertise the "
                "left path's identity for both."
            )
            raise InvalidInventoryError(msg)
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


def _clip_intervals(items, lo: float, hi: float, outer: float | None = None) -> list:
    """
    Clip interval items to [lo, hi), dropping those left with no coverage.

    Point markers cover nothing but are not nothing: they survive when they
    fall inside the clip, or on its outermost included endpoint.
    """
    out = []
    for item in items:
        start, end = item.interval
        if start == end:
            if lo <= start < hi or (outer is not None and start == hi == outer):
                out.append(item)
            continue
        new_lo, new_hi = max(start, lo), min(end, hi)
        if new_hi <= new_lo:
            continue
        out.append(
            item.model_copy(update={"start_distance": new_lo, "end_distance": new_hi})
        )
    return out


class Response(InventoryModel):
    """Station-specific response model associated with a channel."""

    sensitivity: FiniteFloat | None = Field(
        default=None, description="Overall channel sensitivity."
    )
    input_units: UnitQuantity | None = Field(
        default=None, description="Physical units before conversion."
    )
    output_units: UnitQuantity | None = Field(
        default=None, description="Physical units after conversion."
    )


def _check_finite_coordinates(value):
    """Coordinates name a position; nan and inf name nothing."""
    if value is not None and not np.all(np.isfinite(np.asarray(value, dtype=float))):
        msg = f"Coordinate values must be finite; got {value}."
        raise InvalidInventoryError(msg)
    return value


# Coordinates on the canonical axes, as declared by the inventory CRS.
Coordinates = Annotated[
    tuple[float, ...] | None, AfterValidator(_check_finite_coordinates)
]


class Channel(TimeRangedModel):
    """Station-level time-series stream identity."""

    code: CodeStr = Field(description="Channel code used in data source identifiers.")
    location_code: LocationCodeStr = Field(
        default="", description="FDSN-style location code; may be blank."
    )
    coordinates: Coordinates = Field(
        default=None,
        description=(
            "Coordinates on the canonical (x, y, z) axes; the inventory CRS "
            "coordinate_labels declare their meaning."
        ),
    )
    sample_rate: FiniteFloat | None = Field(
        default=None, description="Sample rate in Hz."
    )
    azimuth: Azimuth | None = Field(
        default=None,
        description=(
            "Sensor component azimuth in degrees clockwise from north, [0, 360)."
        ),
    )
    dip: Dip | None = Field(
        default=None,
        description=(
            "Sensor component dip in degrees down from horizontal, [-90, 90]."
        ),
    )
    depth: FiniteFloat | None = Field(
        default=None,
        description="Sensor depth in meters below the local surface; positive down.",
    )
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
    identifiers: tuple[str, ...] = Field(
        default=(),
        description=(
            "Scheme-prefixed identifier URIs naming this object, e.g. "
            "'doi:10.7914/SN/XX'."
        ),
    )
    coordinates: Coordinates = Field(
        default=None,
        description=(
            "Coordinates on the canonical (x, y, z) axes; the inventory CRS "
            "coordinate_labels declare their meaning."
        ),
    )
    channels: tuple[Channel, ...] = Field(
        default=(), description="Channels associated with this station."
    )

    def check(self) -> Self:
        """
        Check channel rules for this station.

        Channel ``(location_code, code)`` identities, which name a stream,
        must be unique for overlapping time ranges.
        """
        errors = [
            f"Duplicate channel identity {key} for overlapping time ranges."
            for key, *_ in _overlapping_epochs(
                self.channels, lambda x: (x.location_code, x.code)
            )
        ]
        if errors:
            msg = f"Station {self.code!r} validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        return self


class FiberArray(TimeRangedModel):
    """
    Durable fiber-optic observing identity.

    A fiber array groups similar, co-located fiber routes; each route is a
    location code carrying one optical path at a time.
    """

    code: CodeStr = Field(description="Station-like fiber array code.")
    name: str = Field(default="", description="Human-readable fiber array name.")
    identifiers: tuple[str, ...] = Field(
        default=(),
        description=(
            "Scheme-prefixed identifier URIs naming this object, e.g. "
            "'doi:10.7914/SN/XX'."
        ),
    )
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


class Network(TimeRangedModel):
    """FDSN-like organizational container below an inventory."""

    code: CodeStr = Field(description="Network code.")
    name: str = Field(default="", description="Human-readable network name.")
    identifiers: tuple[str, ...] = Field(
        default=(),
        description=(
            "Scheme-prefixed identifier URIs naming this object, e.g. "
            "'doi:10.7914/SN/XX'."
        ),
    )
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
        ranges; contained fiber arrays and stations are also validated.
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
        for station in self.stations:
            station.check()
        return self


# Fields whose default is not the empty value; dropping them would reload as
# something else. An annotation value defaults to True, so a pruned empty
# string would come back as a boolean and change its group's kind.
_UNPRUNED_KEYS = frozenset({"value"})


def _drop_empty(value, _in_extras=False):
    """
    Recursively drop empty strings, mappings, and sequences from a dump.

    Pruned fields default to the empty value they held, so reload is
    lossless; fields in ``_UNPRUNED_KEYS`` and user-supplied ``extra_fields``
    contents are kept verbatim.
    """
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            pruned = (
                item
                if _in_extras
                else _drop_empty(item, _in_extras=key == "extra_fields")
            )
            empty = pruned in ("", {}, []) and key not in _UNPRUNED_KEYS
            if empty and not _in_extras:
                continue
            out[key] = pruned
        return out
    if isinstance(value, list):
        return [_drop_empty(item) for item in value]
    return value


class ResolvedContext(NamedTuple):
    """The inventory objects an acquisition_key + time resolve to."""

    network: Network
    fiber_array: FiberArray
    acquisition: Acquisition
    optical_path: OpticalPath | None


class InventoryNames(NamedTuple):
    """
    The names an inventory could contribute to a patch.

    Split by where each lands: `attrs` are the scalar facts about the
    observing system, `coords` the names which take a value per channel.
    """

    attrs: tuple[str, ...]
    coords: tuple[str, ...]


# Fields which name or tag a record rather than describe it: the type
# discriminator and resource id a shareable record carries, and the codes
# locating an acquisition (a patch's acquisition_key already states them).
_IDENTITY_FIELDS = frozenset({"object_type", "resource_id", "code", "location_code"})

# What a distance-ranged track carries to place itself, which is its extent
# rather than a value the channels inside it take.
_EXTENT_FIELDS = frozenset(_IntervalModel.model_fields) - frozenset(
    InventoryModel.model_fields
)


# The generics a field annotation uses to say it holds many values. Only
# these are unwrapped further, so `tuple[float, ...]` stays a tuple rather
# than reading as the float it contains.
_COLLECTION_ORIGINS = (tuple, list, set, frozenset, dict)


def _annotation_members(annotation) -> list:
    """
    Return the alternatives a possibly-optional union annotation admits.

    `Annotated` is transparent: this file writes several of its unions as
    `Annotated[A | B, Field(discriminator=...)]`, and the wrapper must not
    hide what they hold. `None` is dropped, since "or nothing" says what
    the field does when it is unset rather than what it can hold.
    """
    if get_origin(annotation) is Annotated:
        return _annotation_members(get_args(annotation)[0])
    if get_origin(annotation) in (Union, UnionType):
        out = [
            x for member in get_args(annotation) for x in _annotation_members(member)
        ]
        return [x for x in out if x is not type(None)]
    return [annotation]


def _is_value_field(info) -> bool:
    """
    Return True when a field could hold one value describing one thing.

    A field which may hold another inventory model is a reference to a
    second record rather than a fact of this one, and one which can only
    hold a collection has no single value to state, or to give a channel.
    """
    members = _annotation_members(info.annotation)
    if any(isinstance(x, type) and issubclass(x, InventoryModel) for x in members):
        return False
    return any(get_origin(x) not in _COLLECTION_ORIGINS for x in members)


def _value_shape(item, field: str) -> str | None:
    """
    Return how an item states a field: as one value, several, or not at all.

    A field stated as several values -- a multi-wavelength loss, say --
    has none to give a channel, however scalar its annotation reads.
    """
    value = getattr(item, field, None)
    if value is None or (isinstance(value, str) and not value):
        return None
    return "many" if isinstance(value, Sized) and not isinstance(value, str) else "one"


@cache
def _value_field_names(model) -> tuple[str, ...]:
    """
    Return the fields of a model which state a fact about one thing.

    Cached on the class: the answer is a property of the model, and an
    inventory asked for its names walks every item of every track.
    """
    structural = frozenset(TimeRangedModel.model_fields) | _IDENTITY_FIELDS
    structural |= _EXTENT_FIELDS
    return tuple(
        name
        for name, info in model.model_fields.items()
        if name not in structural and _is_value_field(info)
    )


TRACK_IDENTITY_FIELDS = _track_identity_fields()


# The observing-system facts, read off the models rather than listed, so a
# field added to either automatically becomes something an inventory can
# contribute. Pinned to INVENTORY_ATTRS by a test: the two are one
# vocabulary, and a new field has to reach the readers as well.
_SYSTEM_FACT_NAMES = tuple(
    sorted(
        list(_value_field_names(Acquisition))
        + [f"interrogator.{x}" for x in _value_field_names(Interrogator)]
    )
)


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
    resources: FrozenDictType[str, _Resource] = Field(
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
        # Mapping, not dict, at both levels: FrozenDict is not a dict, so a
        # pool would fall through to the iterable-of-resources branch and a
        # record would be read as an object, hiding a key that disagrees.
        if isinstance(value, Mapping):
            out = {}
            for key, resource in value.items():
                if isinstance(resource, Mapping):
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
        # object.__setattr__ bypasses the field validator, so freeze here.
        object.__setattr__(self, "resources", FrozenDict(pool))
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

    def get_names(self) -> InventoryNames:
        """
        Return the names this inventory could contribute to a patch.

        These are the names
        [`Patch.enrich`](`dascore.proc.inventory.enrich`) can be asked for
        and selection can query, split by where each lands: `attrs` for
        the observing-system facts, which are scalar per patch, and
        `coords` for the names taking a value per channel. `attrs` is the
        same for every inventory, since the models decide it; `coords`
        depends on what this inventory holds.

        Contributing a name is not promising a value for it: a name is
        listed when some acquisition or optical path could define it, and
        one which none does simply resolves to nothing.

        Examples
        --------
        >>> from dascore.examples import inventory_patch_pair
        >>>
        >>> _, inventory = inventory_patch_pair()
        >>> names = inventory.get_names()
        >>> assert "gauge_length" in names.attrs
        >>> assert "interrogator.model" in names.attrs
        >>> assert "coupling" in names.coords  # its coupling_type
        >>> assert "coupling.medium" in names.coords
        """
        return InventoryNames(attrs=_SYSTEM_FACT_NAMES, coords=self._coord_names())

    def _optical_paths(self):
        """Iterate every optical path this inventory holds."""
        for network in self.networks:
            for array in network.fiber_arrays:
                yield from array.optical_paths

    def _coord_names(self) -> tuple[str, ...]:
        """
        Return the per-channel names this inventory's paths could define.

        The CRS's axes and optical distance come from the model, while the
        annotation groups and the tracks which are actually described come
        from the paths themselves: an inventory with no coupling track has
        no coupling to select on.
        """
        labels = self.coordinate_reference_system.coordinate_labels
        # Both spellings of the same axes: the canonical storage names and
        # whatever this CRS declares they mean.
        out = dict.fromkeys(["distance", *("x", "y", "z")[: len(labels)], *labels])
        groups: dict[str, None] = {}
        tracks: dict[str, dict[str, None]] = {}
        shapes: dict[str, set[str]] = {}
        for path in self._optical_paths():
            groups.update(dict.fromkeys(x.group for x in path.annotations if x.group))
            for track in TRACK_IDENTITY_FIELDS:
                for item in getattr(path, track):
                    fields = tracks.setdefault(track, {})
                    names = _value_field_names(type(item))
                    fields.update(dict.fromkeys(names))
                    for field in names:
                        if (shape := _value_shape(item, field)) is not None:
                            shapes.setdefault(f"{track}.{field}", set()).add(shape)
        # Dropped only where nothing states it as one value: another path
        # may record a scalar where this one records several, and that
        # path can still give it to a channel.
        unusable = {x for x, kinds in shapes.items() if kinds == {"many"}}
        for track, fields in tracks.items():
            out[track] = None
            names = (f"{track}.{x}" for x in fields)
            out.update(dict.fromkeys(x for x in names if x not in unusable))
        out.update(groups)
        return tuple(out)

    def check(self) -> Self:
        """Check the whole inventory tree against the model rules."""
        errors = [
            f"Duplicate network code {code!r} for overlapping time ranges."
            for code, *_ in _overlapping_epochs(self.networks, lambda x: x.code)
        ]
        errors.extend(self._check_coordinate_widths())
        if errors:
            msg = "Inventory validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        for net in self.networks:
            net.check()
        return self

    def _check_coordinate_widths(self) -> list[str]:
        """Check coordinates carry one value per axis the CRS declares."""
        crs = self.coordinate_reference_system
        axes = len(crs.coordinate_labels)
        errors = []

        def check_width(width, what):
            if width != axes:
                errors.append(
                    f"{what} has {width} coordinate values but the inventory "
                    f"CRS declares {axes} axes {crs.coordinate_labels}."
                )

        for net in self.networks:
            for array in net.fiber_arrays:
                for path in array.optical_paths:
                    for segment in path.geometry:
                        what = f"Geometry {segment.name!r}"
                        check_width(len(segment.coordinates[0]), what)
            for station in net.stations:
                if station.coordinates is not None:
                    check_width(len(station.coordinates), f"Station {station.code!r}")
                for channel in station.channels:
                    if channel.coordinates is None:
                        continue
                    what = f"Channel {station.code!r}.{channel.code!r}"
                    check_width(len(channel.coordinates), what)
        return errors

    def resolve(self, acquisition_key: str, time=None) -> ResolvedContext:
        """
        Resolve an acquisition_key (and time) to its inventory context.

        Parameters
        ----------
        acquisition_key
            A dotted ``network.fiber_array.location.acquisition`` identifier.
        time
            The instant to resolve at; required whenever any epoch level has
            more than one candidate.
        """
        # The same validator PatchAttrs uses, so a key legal in one is legal
        # in the other and a malformed one is told apart from an unknown one.
        # It accepts the empty string, which is how a patch spells "no
        # identity" -- legal to carry, but nothing to resolve.
        try:
            acquisition_key = validate_acquisition_key(acquisition_key)
        except ValueError as error:
            raise InvalidInventoryError(str(error)) from error
        if not acquisition_key:
            msg = "Cannot resolve an empty acquisition_key; it names no entry."
            raise InvalidInventoryError(msg)
        net_code, array_code, location, acq_code = acquisition_key.split(".")

        def exactly_one(matches, kind):
            if len(matches) != 1:
                msg = f"{acquisition_key!r} resolves to {len(matches)} {kind}."
                raise InvalidInventoryError(msg)
            return matches[0]

        network = exactly_one(
            [
                x
                for x in self.networks
                if x.code == net_code and x.is_effective_at(time)
            ],
            "networks",
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
            msg = f"{acquisition_key!r} resolves to {len(paths)} optical paths."
            raise InvalidInventoryError(msg)
        path = paths[0] if paths else None
        return ResolvedContext(network, array, acqs[0], path)

    def replace(self, old, new) -> Self:
        """
        Return a new inventory with one component replaced.

        This is the correction mechanism: the change applies in place and
        retroactively. ``old`` is matched by equality at any addressable
        level: networks, stations, channels, fiber arrays, acquisitions,
        optical paths, and path track items (components, geometry, coupling,
        annotations); pooled resources are addressed by their resource_id.
        An ``old`` matching more than one item is ambiguous and raises.
        Singletons such as the CRS or a distance map are corrected with
        ``new()`` on their parent. ``new`` must be the same type as ``old``.
        Note that resource normalization rewrites inline resource objects to
        id references at construction, so match against the stored
        (normalized) object — e.g. an acquisition whose ``interrogator`` is
        the id string — not a pre-construction handle holding the inline
        object.
        """
        if type(new) is not type(old):
            msg = (
                f"Replacement type {type(new).__name__} does not match "
                f"{type(old).__name__}."
            )
            raise InvalidInventoryError(msg)
        if isinstance(old, get_args(get_args(_Resource)[0])):
            if self.resources.get(old.resource_id) != old:
                msg = "Component to replace was not found in the inventory."
                raise InvalidInventoryError(msg)
            if new.resource_id != old.resource_id:
                msg = (
                    "Resource corrections must keep the same resource_id "
                    f"({old.resource_id!r} != {new.resource_id!r}); id "
                    "references throughout the tree would dangle."
                )
                raise InvalidInventoryError(msg)
            pool = dict(self.resources) | {old.resource_id: new}
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
        if replaced > 1:
            msg = (
                f"{type(old).__name__} to replace matches {replaced} items; "
                "equal items are indistinguishable, so set a distinguishing "
                "field (a name, or a description on items without one) on the "
                "intended item before correcting it."
            )
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

        A loaded document is checked before it is returned: reading one asks
        whether it is a valid inventory, and a document which fails should
        fail at its source rather than as a confusing error later. Building
        an inventory in memory stays unchecked until ``check`` is called, so
        it can be assembled a piece at a time.

        Parameters
        ----------
        source
            A path to a YAML file, or YAML text (must contain a newline to
            be treated as text rather than a missing path).
        """
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
        return cls(**data).check()
