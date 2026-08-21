"""
DASDAE Inventory: DASCore's metadata model for DFOS observing systems.

The inventory extends the StationXML concept with first-class support for
fiber-optic arrays. It describes the physical optical path (fiber, connectors,
splices), the geometry, coupling, and label tracks along optical
distance, and the interrogator configurations (acquisitions) that produced
patches. Patches carry a ``acquisition_key``
(``network.fiber_array.location.acquisition``) which, together with time,
resolves against an inventory.

Each object documents the rules it enforces.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sized
from contextlib import suppress
from functools import cache
from pathlib import Path
from types import MappingProxyType, UnionType
from typing import (
    Annotated,
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    Self,
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
from dascore.utils.documents import (
    dump_document,
    parse_document,
    write_text_document,
)
from dascore.utils.intervals import (
    clip_intervals,
    interval_masks,
    intervals_overlap,
    normalize_value,
    value_kind,
)
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import (
    check_code,
    is_strictly_monotonic,
    validate_acquisition_key,
)
from dascore.utils.namespace import NamespaceOwner

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


def _label_value(value):
    """Normalize a label value so its Python type survives validation."""
    return normalize_value(value, error=InvalidInventoryError)


# A label states membership by carrying no value, so a value, when there is
# one, is text or a number; its kind decides the group's shape.
LabelValue = Annotated[str | int | float, BeforeValidator(_label_value)]


def _object_type_tag(name: str):
    """
    Return the serialization-only ``object_type`` field of a union member.

    Every model states its class in a json document (see
    [dascore.models.registry](`dascore.models.registry`)), but pydantic must
    pick a union member's class before an object exists, so these models
    declare the tag as a real field and the base class leaves them alone.

    Users never set it: it defaults to the class name, the Literal
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
    Measured curves along an interval of an optical path.

    A geometry is a piecewise segment placed by its ``distance`` array: at
    least two strictly increasing optical distances, each paired with one
    value of every column the segment states. Coverage is the half-open span
    of the array; there is no separate length field. A coil, or other
    "clump", is a segment whose columns repeat while distance advances.

    A column whose name the inventory CRS declares -- or the canonical
    ``x``, ``y``, ``z`` alias of one -- is that position axis, and takes the
    CRS's units. Every other column is a numeric quantity along the fiber in
    its own right, carrying its own entry in ``units``: borehole depth where
    the CRS is spent on easting/northing/elevation, pipeline chainage, fiber
    azimuth.

    Interpolation between points is piecewise linear and never crosses
    segments; uncovered distance has undefined values. Two segments may
    cover the same distance as long as they state different columns -- a
    depth survey and an azimuth survey of one borehole -- and then they
    share a name, being two measurements of one stretch of fiber.

    Examples
    --------
    >>> from dascore.core.inventory import Geometry
    >>>
    >>> # A surveyed run, placed by the CRS's axes.
    >>> trench = Geometry(
    ...     name="trench",
    ...     distance=(0.0, 100.0),
    ...     coordinates={"x": (0.0, 86.6), "y": (0.0, 0.0), "z": (-0.5, -0.5)},
    ... )
    >>>
    >>> # A curve which is not a position at all.
    >>> chainage = Geometry(
    ...     name="chainage",
    ...     distance=(0.0, 100.0),
    ...     coordinates={"chainage": (1200.0, 1290.0)},
    ...     units={"chainage": "m"},
    ... )
    """

    _identity_field: ClassVar[str] = "name"
    name: str = Field(default="", description="Human-readable geometry name.")
    distance: tuple[float, ...] = Field(
        description=(
            "Optical distances paired to values; at least two strictly "
            "increasing values whose span is the segment's coverage."
        ),
    )
    coordinates: FrozenDictType[str, tuple[float, ...]] = Field(
        description=(
            "Columns measured along this segment, keyed by name; each holds "
            "one value per distance."
        ),
    )
    units: FrozenDictType[str, str] = Field(
        default_factory=dict,
        description="Units of the columns which are not position axes.",
    )

    @model_validator(mode="after")
    def _validate_geometry(self) -> Self:
        """Enforce paired, strictly increasing control points."""
        _check_control_points(self.distance, "Geometry distance", minimum=2)
        if not self.coordinates:
            msg = "Geometry states no columns, so it describes nothing."
            raise InvalidInventoryError(msg)
        wrong = sorted(
            name
            for name, values in self.coordinates.items()
            if len(values) != len(self.distance)
        )
        if wrong:
            msg = (
                f"Geometry column(s) {wrong} do not have one value per "
                f"distance; distance states {len(self.distance)}."
            )
            raise InvalidInventoryError(msg)
        for name, values in self.coordinates.items():
            if not np.all(np.isfinite(np.asarray(values, dtype=float))):
                msg = f"Geometry column {name!r} must hold finite values."
                raise InvalidInventoryError(msg)
        # Units name a column or nothing; the alternative is a unit sitting
        # on a column the segment never states, which no reader would find.
        if orphaned := sorted(set(self.units) - set(self.coordinates)):
            msg = f"Geometry states units for {orphaned}, which it has no column for."
            raise InvalidInventoryError(msg)
        return self

    @property
    def interval(self) -> tuple[float, float]:
        """The (start, end) optical distance covered by this segment."""
        return (self.distance[0], self.distance[-1])

    def interpolate(self, distances) -> dict[str, np.ndarray]:
        """
        Return each column's values at the requested optical distances.

        Distances outside this segment's coverage return NaN. Coverage is
        half-open ``[first, last)``; inclusion of the outermost track endpoint
        is handled by the caller (`OpticalPath.coordinates_at`).
        """
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        start, end = self.interval
        inside = (dist >= start) & (dist < end)
        knots = np.asarray(self.distance, dtype=float)
        out = {}
        for name, values in self.coordinates.items():
            column = np.full(len(dist), np.nan)
            column[inside] = np.interp(
                dist[inside], knots, np.asarray(values, dtype=float)
            )
            out[name] = column
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


class OpticalPathLabel(_IntervalModel):
    """
    Label attached to an interval of an optical path.

    ``group`` names the variable and ``value`` is its state over the
    interval. A label with no value states membership, and membership
    groups may overlap freely; a label with a value makes its group single
    valued, so string and numeric groups may not overlap. A group states
    membership or holds one kind of value, never both. Point markers (equal
    start and end) cover nothing and are exempt from the overlap rule.
    """

    group: str = Field(default="", description="Name of the labelled variable.")
    value: LabelValue | None = Field(
        default=None,
        description=(
            "Value of the variable over this interval; unset for a label "
            "which states membership."
        ),
    )

    @field_validator("value")
    @classmethod
    def _reject_empty_string(cls, value):
        """A label whose value is empty states nothing.

        It would also be indistinguishable from an uncovered channel, since
        a string coordinate spells absence as the empty string.
        """
        if isinstance(value, str) and not value:
            msg = (
                "A label value may not be the empty string; it would "
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


def _epoch_span(item) -> str:
    """Name an epoch's range the way an error can show it."""
    start = "the beginning" if np.isnat(item.start_time) else str(item.start_time)
    end = "ongoing" if np.isnat(item.end_time) else str(item.end_time)
    return f"{start} to {end}"


def _containment_errors(parent, children, what: str, name) -> list[str]:
    """
    Return errors for children whose epoch reaches outside their container.

    A bound a child leaves unset defers to its container rather than
    reaching past it: an optical path stating no start begins when its
    fiber array does. A bound it states has to fall inside the range its
    container states, or the child describes time its container says did
    not happen -- and since resolution walks down from the container, that
    is metadata no query can reach.
    """
    start, end = parent.start_time, parent.end_time
    errors = []
    for child in children:
        before = not np.isnat(start) and not np.isnat(child.start_time)
        before = before and child.start_time < start
        after = not np.isnat(end) and not np.isnat(child.end_time)
        after = after and child.end_time > end
        if before or after or not child.overlaps(parent):
            errors.append(
                f"{what} {name(child)} is valid {_epoch_span(child)}, outside "
                f"its container's {_epoch_span(parent)}."
            )
    return errors


def axis_columns(segment, crs) -> dict[str, int]:
    """
    Return which of a segment's columns name which canonical axis.

    A column resolves to an axis when the CRS declares its name, or when it
    is the canonical ``x``/``y``/``z`` alias of an axis the CRS has. Every
    other column is a quantity along the fiber rather than a position.
    """
    out = {}
    for name in segment.coordinates:
        with suppress(InvalidInventoryError):
            out[name] = crs.axis_index(name)
    return out


def _placed(segment, name: str, distances) -> np.ndarray:
    """
    One column of a segment, over distances already claimed for it.

    ``interpolate`` stops at its own half-open coverage, which excludes the
    run end a mask includes, so that point is filled from the last control
    point rather than left NaN.
    """
    column = segment.interpolate(distances)[name]
    column[np.isnan(column)] = segment.coordinates[name][-1]
    return column


def _axis_set_errors(segment, axes: Mapping[str, int], crs) -> list[str]:
    """Return what is wrong with the set of axes one segment states."""
    what = f"Geometry {segment.name!r}" if segment.name else "A geometry"
    spellings: dict[int, list[str]] = {}
    for name, index in axes.items():
        spellings.setdefault(index, []).append(name)
    errors = [
        f"{what} states axis {crs.coordinate_labels[index]!r} twice, as "
        f"{sorted(names)}."
        for index, names in sorted(spellings.items())
        if len(names) > 1
    ]
    if axes and len(spellings) != len(crs.coordinate_labels):
        errors.append(
            f"{what} states the axes {sorted(axes)} but the inventory CRS "
            f"declares {list(crs.coordinate_labels)}; a segment states every "
            "axis or none of them."
        )
    return errors


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


# Names a label group may not take: a group becomes a patch coordinate
# at enrichment, where it would shadow one of these, and a stamped attr
# at expansion, where it would replace the patch's identity.
RESERVED_GROUP_NAMES = frozenset(
    {"time", "distance", "channel", "instrument_distance", "acquisition_key"}
    | {"optical_components", "geometry", "coupling", "labels"}
    | set(VALID_COORDINATE_LABELS)
)


# The reserved names a geometry column may not take. The coordinate labels
# are left out of it: a column named for one is how a segment states that
# axis, and one the CRS does not declare is free to be a column of its own.
_RESERVED_COLUMN_NAMES = RESERVED_GROUP_NAMES - set(VALID_COORDINATE_LABELS)


def _times_equal(time1, time2) -> bool:
    """Compare two epoch times, treating unset (NaT) times as equal."""
    null1, null2 = np.isnat(time1), np.isnat(time2)
    if null1 or null2:
        return bool(null1 and null2)
    return bool(time1 == time2)


class OpticalPath(TimeRangedModel):
    """
    Continuous optical path described by independent tracks.

    Optical components tile ``[start_distance, start_distance + optical
    length)``. Geometry and coupling are function tracks (partial coverage,
    no overlap); labels overlap as their group's value kind allows. No
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
    labels: tuple[OpticalPathLabel, ...] = Field(
        default=(), description="Labels on this path."
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

    def coordinates_at(self, distances, crs) -> np.ndarray:
        """
        Return CRS coordinates at the requested optical distances.

        The CRS is what decides a column is an axis, and only those are
        assembled: a segment naming none of them contributes no position,
        however much else it measures, and does not decide the position
        track's coverage either. Uncovered distance returns NaN rows.
        Coverage is half-open, with the end of each run included: a distance
        on a segment's last control point belongs to it unless another
        segment claims it.
        """
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        out = np.full((len(dist), len(crs.coordinate_labels)), np.nan)
        placing = [x for x in self.geometry if axis_columns(x, crs)]
        masks = interval_masks(dist, [x.interval for x in placing])
        for segment, mask in zip(placing, masks, strict=True):
            axes = axis_columns(segment, crs)
            if errors := _axis_set_errors(segment, axes, crs):
                # A checked inventory cannot get here; an unchecked one says
                # so rather than filling a position from whichever spelling
                # the mapping held last.
                raise InvalidInventoryError(" ".join(errors))
            if not np.any(mask):
                continue
            rows = np.flatnonzero(mask)
            for name, index in axes.items():
                out[rows, index] = _placed(segment, name, dist[mask])
        return out

    def column_at(self, name: str, distances) -> np.ndarray | None:
        """
        Return one geometry column's values at the requested distances.

        None when no segment states the column. Coverage follows
        `OpticalPath.coordinates_at`, and values never bridge two segments:
        distance between them is uncovered, whatever either side holds.
        """
        stating = [x for x in self.geometry if name in x.coordinates]
        if not stating:
            return None
        dist = np.atleast_1d(np.asarray(distances, dtype=float))
        out = np.full(len(dist), np.nan)
        masks = interval_masks(dist, [x.interval for x in stating])
        for segment, mask in zip(stating, masks, strict=True):
            if np.any(mask):
                out[np.flatnonzero(mask)] = _placed(segment, name, dist[mask])
        return out

    def geometry_columns(self) -> tuple[str, ...]:
        """Return every column name this path's geometry segments state."""
        seen: dict[str, None] = {}
        for segment in self.geometry:
            seen.update(dict.fromkeys(segment.coordinates))
        return tuple(seen)

    def check(self, tolerance: float = 1e-9) -> Self:
        """
        Check track rules for this path.

        Checks that geometry and coupling stay within path bounds and do not
        overlap (partial coverage is legal), and that labels stay within
        bounds. Component tiling is inherent to the cumulative layout.
        """
        errors = []
        start, end = self.start_distance, self.end_distance
        geo_spans = [seg.interval for seg in self.geometry]
        coup_spans = [c.interval for c in self.coupling]
        label_spans = [a.interval for a in self.labels]
        for name, spans in (
            ("geometry", geo_spans),
            ("coupling", coup_spans),
            ("labels", label_spans),
        ):
            for lo, hi in spans:
                if lo < start - tolerance or hi > end + tolerance:
                    errors.append(
                        f"{name} interval ({lo}, {hi}) extends past path "
                        f"span ({start}, {end})."
                    )
        overlap = intervals_overlap(coup_spans)
        if overlap is not None:
            errors.append(
                f"Overlapping coupling intervals {overlap[0]} and "
                f"{overlap[1]}; coupling is a function track."
            )
        errors.extend(self._check_geometry_columns())
        errors.extend(self._check_label_groups())
        if errors:
            msg = "Optical path validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        return self

    def _check_geometry_columns(self) -> list[str]:
        """
        Check the geometry columns of this path against each other.

        Each column is its own function track, so two segments may overlap
        as long as they do not state the same column over the same distance.
        Overlapping segments do share a name, being two measurements of one
        stretch of fiber, or the bare ``geometry`` coordinate would have two
        of them for a channel. The rules needing the CRS -- which columns
        are axes -- are the inventory's, since only it knows the axes.
        """
        spans: dict[str, list[tuple[float, float]]] = {}
        units: dict[str, set[str]] = {}
        for segment in self.geometry:
            for name in segment.coordinates:
                spans.setdefault(name, []).append(segment.interval)
                if name in segment.units:
                    units.setdefault(name, set()).add(segment.units[name])
        groups = {x.group for x in self.labels if x.group}
        errors = [
            f"Geometry column {name!r} is a reserved name; a column becomes "
            "a coordinate and cannot shadow a structural coordinate or a "
            "typed track."
            for name in sorted(set(spans) & _RESERVED_COLUMN_NAMES)
        ]
        errors += [
            f"Geometry column {name!r} states a dotted name, which is how a "
            "field of a typed track is asked for; a column is asked for by a "
            "name of its own."
            for name in sorted(x for x in spans if "." in x)
        ]
        errors += [
            f"{name!r} is both a geometry column and a label group; "
            "one name is one coordinate."
            for name in sorted(set(spans) & groups)
        ]
        for first, second in itertools.combinations(self.geometry, 2):
            lo = max(first.interval[0], second.interval[0])
            hi = min(first.interval[1], second.interval[1])
            if first.name != second.name and lo < hi:
                errors.append(
                    f"Geometry segments {first.name!r} and {second.name!r} "
                    f"both cover ({lo}, {hi}); segments which overlap state "
                    "different columns of one stretch of fiber, so they "
                    "share its name."
                )
        for name in sorted(spans):
            if (overlap := intervals_overlap(spans[name])) is not None:
                errors.append(
                    f"Overlapping geometry intervals {overlap[0]} and "
                    f"{overlap[1]} for column {name!r}; a column is a "
                    "function track."
                )
            if len(stated := units.get(name, set())) > 1:
                errors.append(
                    f"Geometry column {name!r} is stated in {sorted(stated)}; "
                    "a column has one unit."
                )
        return errors

    def _check_label_groups(self) -> list[str]:
        """Check that each label group holds one kind of value."""
        groups: dict[str, list] = {}
        for label in self.labels:
            groups.setdefault(label.group, []).append(label)
        errors = []
        for group in sorted(set(groups) & RESERVED_GROUP_NAMES):
            errors.append(
                f"Label group {group!r} is a reserved name; a group "
                "becomes a coordinate and cannot shadow a structural "
                "coordinate, a typed track, or a coordinate label."
            )
        for group, items in groups.items():
            kinds = {value_kind(x.value) for x in items}
            if len(kinds) > 1:
                errors.append(
                    f"Label group {group!r} mixes {sorted(kinds)} values; "
                    "a group states membership or holds one kind of value."
                )
                continue
            if kinds == {"membership"}:  # membership groups may overlap
                continue
            overlap = intervals_overlap([x.interval for x in items])
            if overlap is not None:
                errors.append(
                    f"Overlapping intervals {overlap[0]} and {overlap[1]} in "
                    f"label group {group!r}; only membership groups, whose "
                    "labels carry no value, may overlap."
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
            new_coords = {
                name: tuple(np.interp(new_dist, dist, np.asarray(values, dtype=float)))
                for name, values in seg.coordinates.items()
            }
            # new(), not model_copy(): a copy skips the validators, and
            # would leave `coordinates` a plain mutable dict whose columns
            # nothing had checked against the new distance array.
            geometry.append(seg.new(distance=tuple(new_dist), coordinates=new_coords))
        outer = self.end_distance
        coupling = clip_intervals(self.coupling, lo, hi, outer)
        labels = clip_intervals(self.labels, lo, hi, outer)
        return self.model_copy(
            update={
                "start_distance": lo,
                "optical_components": tuple(components),
                "geometry": tuple(geometry),
                "coupling": tuple(coupling),
                "labels": tuple(labels),
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
            geometry.append(
                seg.new(
                    distance=tuple(flip(dist)[::-1]),
                    coordinates={
                        name: tuple(values[::-1])
                        for name, values in seg.coordinates.items()
                    },
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
        labels = sorted(
            (flip_item(a) for a in self.labels),
            key=lambda a: a.start_distance,
        )
        return self.model_copy(
            update={
                "optical_components": tuple(reversed(self.optical_components)),
                "geometry": tuple(geometry),
                "coupling": tuple(coupling),
                "labels": tuple(labels),
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
        labels = tuple(shift_item(a) for a in other.labels)
        return self.model_copy(
            update={
                "optical_components": (
                    *self.optical_components,
                    *other.optical_components,
                ),
                "geometry": (*self.geometry, *geometry),
                "coupling": (*self.coupling, *coupling),
                "labels": (*self.labels, *labels),
                "measurements": (*self.measurements, *other.measurements),
            }
        )


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
        errors += _containment_errors(
            self, self.channels, "Channel", lambda x: repr(x.code)
        )
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
        errors += _containment_errors(
            self, self.optical_paths, "Optical path", lambda x: repr(x.name)
        )
        errors += _containment_errors(
            self, self.acquisitions, "Acquisition", lambda x: repr(x.code)
        )
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
            errors += _containment_errors(
                self, items, kind.capitalize(), lambda x: repr(x.code)
            )
        if errors:
            msg = f"Network {self.code!r} validation failed:\n" + "\n".join(errors)
            raise InvalidInventoryError(msg)
        for array in self.fiber_arrays:
            array.check()
        for station in self.stations:
            station.check()
        return self


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


def _yaml_label(text: str) -> str:
    """
    Name YAML text in an error without quoting a whole document at it.

    A short document is worth showing; a long one would bury the reason
    it was rejected under itself.
    """
    short = len(text) <= 60 and "\n" not in text
    return f"{text!r}" if short else "the given YAML text"


class Inventory(NamespaceOwner, InventoryModel):
    """
    Top-level DASDAE inventory manifest.

    Stores document metadata, shareable resources keyed by resource_id, the
    inventory-wide coordinate reference system, and network containers.
    """

    # Annotated as a ClassVar so pydantic leaves it a plain class attribute;
    # an unannotated underscore name becomes a private attribute instead.
    _namespace_entry_point_group: ClassVar[str] = "dascore.inventory_namespace"

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
        label groups and the tracks which are actually described come
        from the paths themselves: an inventory with no coupling track has
        no coupling to select on.
        """
        axes = self.coordinate_reference_system.coordinate_labels
        # Both spellings of the same axes: the canonical storage names and
        # whatever this CRS declares they mean.
        out = dict.fromkeys(["distance", *("x", "y", "z")[: len(axes)], *axes])
        groups: dict[str, None] = {}
        tracks: dict[str, dict[str, None]] = {}
        shapes: dict[str, set[str]] = {}
        crs = self.coordinate_reference_system
        columns: dict[str, None] = {}
        for path in self._optical_paths():
            groups.update(dict.fromkeys(x.group for x in path.labels if x.group))
            # The axes are listed above whatever any path states; what a
            # path adds is the columns which are not positions.
            for segment in path.geometry:
                axes = axis_columns(segment, crs)
                columns.update(
                    dict.fromkeys(x for x in segment.coordinates if x not in axes)
                )
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
        out.update(columns)
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
                    errors.extend(self._check_geometry_axes(path, crs))
            for station in net.stations:
                if station.coordinates is not None:
                    check_width(len(station.coordinates), f"Station {station.code!r}")
                for channel in station.channels:
                    if channel.coordinates is None:
                        continue
                    what = f"Channel {station.code!r}.{channel.code!r}"
                    check_width(len(channel.coordinates), what)
        return errors

    @staticmethod
    def _check_geometry_axes(path, crs) -> list[str]:
        """
        Check one path's geometry against the CRS.

        Which columns are axes is the CRS's to say, so the rules needing it
        live here rather than on the path: that a segment states every axis
        or none, that it does not spell one twice, that it leaves the axes'
        units to the CRS, and that two segments do not place the same axis
        over one distance -- which the path cannot see, two spellings of an
        axis being two names to it.
        """
        errors = []
        spans: dict[int, list[tuple[float, float]]] = {}
        for segment in path.geometry:
            axes = axis_columns(segment, crs)
            errors += _axis_set_errors(segment, axes, crs)
            if on_axes := sorted(set(segment.units) & set(axes)):
                what = f"Geometry {segment.name!r}" if segment.name else "A geometry"
                errors.append(
                    f"{what} states units for the axis column(s) {on_axes}; "
                    "the CRS states the units of its own axes."
                )
            for index in set(axes.values()):
                spans.setdefault(index, []).append(segment.interval)
        for index in sorted(spans):
            if (overlap := intervals_overlap(spans[index])) is not None:
                errors.append(
                    f"Overlapping geometry intervals {overlap[0]} and "
                    f"{overlap[1]} for axis {crs.coordinate_labels[index]!r}; "
                    "an axis is a function track."
                )
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
        labels); pooled resources are addressed by their resource_id.
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
                                "labels": swap(path.labels),
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

    @classmethod
    def from_yaml(cls, text: str) -> Self:
        """
        Load an inventory from YAML text.

        A loaded document is checked before it is returned: reading one asks
        whether it is a valid inventory, and a document which fails should
        fail at its source rather than as a confusing error later. Building
        an inventory in memory stays unchecked until ``check`` is called, so
        it can be assembled a piece at a time.

        Parameters
        ----------
        text
            YAML text. Paths — a file or an authoring directory — load
            through [`dascore.inventory`](`dascore.inventory`), which is
            the one door every source goes through.
        """
        if not isinstance(text, str):
            # Otherwise the parser raises about the object's missing
            # read method, naming neither this function nor the path.
            msg = (
                f"from_yaml reads YAML text, got {type(text).__name__}. "
                "Load a path with dascore.inventory."
            )
            raise InvalidInventoryError(msg)
        # A document which does not parse is an invalid inventory, and says
        # so as one: a caller who asked for an inventory should not have to
        # know which parser was reaching for the text.
        data = parse_document(
            text,
            "yaml",
            label=_yaml_label(text),
            error=InvalidInventoryError,
        )
        return cls._from_mapping(data, _yaml_label(text))

    @classmethod
    def _from_mapping(cls, data, source: str) -> Self:
        """
        Build a checked inventory from one parsed document.

        Shared by every route into a whole inventory, so that a document
        which is not one is refused in the same words whichever parser
        read it.
        """
        if not isinstance(data, Mapping):
            msg = f"Could not parse an inventory mapping from {source}."
            raise InvalidInventoryError(msg)
        # `cls(**data)` would raise TypeError on a key which is not a
        # string -- `1: 2` is legal YAML -- naming Python's calling
        # convention rather than the document which broke the rule.
        if named := sorted(f"{x!r}" for x in data if not isinstance(x, str)):
            msg = f"{source} holds fields which are not named: {', '.join(named)}."
            raise InvalidInventoryError(msg)
        return cls(**data).check()


def inventory_to_yaml(inventory: Inventory, path: str | Path | None = None) -> str:
    """
    Serialize an inventory to YAML, optionally writing it to a path.

    Reached as ``inventory.io.to_yaml``.

    A field still holding its default is left out, so the document
    states what the inventory says rather than every field it has;
    what is written reloads equal to this inventory. A path whose
    directory is not there yet is made, as it is for every document
    DASCore writes.

    Parameters
    ----------
    inventory
        The inventory to serialize.
    path
        Where to write the text, or None to only return it.

    Examples
    --------
    >>> import dascore as dc
    >>> _, inventory = dc.examples.inventory_patch_pair()
    >>> text = inventory.io.to_yaml()
    >>> dc.inventory(text) == inventory
    True
    """
    # Everything defaulted is dropped, so the document records which
    # envelope it was written against even when that is the default.
    dumped = inventory.model_dump(mode="json", exclude_defaults=True)
    data = {"schema_version": inventory.schema_version} | dumped
    text = dump_document(data, "yaml")
    if path is not None:
        write_text_document(text, path)
    return text
