"""
DASCore annotations: labelled locations over the dimensions of patch data.

An annotation set describes *data* -- picks, events, noisy hours, vehicle
tracks -- in the frame of the patches it was made on. Facts about the fiber
itself belong to the inventory instead, in optical distance; see
[dascore.core.inventory](`dascore.core.inventory`).

A set holds three parts:

- ``annotations``: one row per located thing. For each declared dimension a
  row states a value (``<dim>``), a half-open range (``<dim>_min`` and
  ``<dim>_max``), or nothing, in which case it spans the whole dimension.
  Every row states at least one dimension, and a dimension holds one kind of
  coordinate: numbers, times or durations.
- ``features``: what annotations compose. Every annotation belongs to one
  feature, named by its ``feature_id``; a blank ``feature_id`` makes the row
  its own feature, and a ``feature_id`` naming no features row creates one.
  A feature is a group (blank ``geometry``), a ``path`` ordered by ``seq``
  within ``part``, or a ``polygon`` ordered by ``seq`` within ``(part,
  ring)``, ring 0 being the outer boundary. Features hold no coordinates.
- ``bases``: keyed curves (`Line`, `Moveout`) a path may be drawn from, so a
  path may exist as a curve alone.

Any other column is an extra carried untouched, except a column whose name
begins with an underscore: that is the author's own, and a set never holds it.
"""

from __future__ import annotations

import datetime
import json
import numbers

# Whole rather than by name: this module's own Path is a geometry.
import pathlib
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from contextlib import suppress
from typing import Annotated, Any, ClassVar, Literal, NamedTuple, Self, cast

import numpy as np
import pandas as pd
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    ValidationError,
    model_validator,
)
from rich.text import Text

from dascore.constants import dascore_styles, max_lens
from dascore.core.inventory import CreationInfo
from dascore.exceptions import InvalidInventoryError, ParameterError
from dascore.models import (
    DascoreBaseModel,
    DateTime64,
    FiniteFloat,
    FrozenDictType,
    PositiveFiniteFloat,
    UnitQuantity,
)
from dascore.utils.display import (
    NodeRepr,
    Repr,
    RichRepr,
    counts_to_text,
    get_header_text,
    mapping_to_text,
    model_to_line,
    range_texts,
    span_text,
    split_block,
    stated_fields,
)
from dascore.utils.documents import write_document
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import iterate, to_str, validate_acquisition_key
from dascore.utils.namespace import NamespaceOwner
from dascore.utils.tables import (
    PRIVATE_PREFIX,
    drop_private_columns,
    parquet_table,
    write_parquet,
    write_parquet_table,
)
from dascore.utils.time import to_datetime64, to_timedelta64

# Columns each table models; everything else is an extra.
ANNOTATION_COLUMNS = (
    "id",
    "name",
    "feature_id",
    "seq",
    "part",
    "ring",
    "acquisition_key",
    "data_id",
    "set",
)
FEATURE_COLUMNS = (
    "id",
    "name",
    "geometry",
    "basis",
    "acquisition_key",
    "data_id",
    "set",
)
RESERVED_COLUMNS = tuple(dict.fromkeys((*ANNOTATION_COLUMNS, *FEATURE_COLUMNS)))

# The order columns of path and polygon members.
ORDINAL_COLUMNS = ("seq", "part", "ring")

# Feature kinds; a blank geometry is a group.
FeatureKind = Literal["group", "path", "polygon"]
GEOMETRY_KINDS = ("group", "path", "polygon")
# Spellings read as a group on input.
_GROUP_SPELLINGS = ("", "group", "region")
# The fewest members each ordered kind needs per part or ring.
_LEAST = {"path": 2, "polygon": 3}

# The parts a stored set is spelled with.
ATTRS_STEM = "attrs"
ANNOTATION_STEM = "annotations"
FEATURE_STEM = "features"
BASES_STEM = "bases"
# CSV is the floor; parquet keeps types and needs pyarrow.
TABLE_SUFFIXES = (".csv", ".parquet")
TABLE_SUFFIX = TABLE_SUFFIXES[0]
OBJECT_SUFFIXES = (".json", ".yaml", ".yml")

# Where a parquet annotations table names its dimensions.
DIMS_KEY = "dascore:dims"

# Range suffixes, as everywhere else in DASCore.
_MIN, _MAX = "_min", "_max"

# The former range spelling, refused with a pointer to the new one.
_RETIRED_RANGE = ("_start", "_end")

# The resolution DASCore holds a time and a duration at.
_NS_TIME = np.dtype("datetime64[ns]")
_NS_SPAN = np.dtype("timedelta64[ns]")

# The dtype kinds a coordinate may be a number in.
_NUMBER_KINDS = "iuf"

# A moveout is physics, not geometry, so it names the dimensions it relates.
DISTANCE_DIM, TIME_DIM = "distance", "time"

# Spelled as PatchAttrs spells them, so a set and the data it describes
# state their provenance the same way.
AcquisitionKey = Annotated[str, AfterValidator(validate_acquisition_key)]


def _document(value):
    """Spell a bound or a vertex for a json document."""
    if isinstance(value, np.datetime64 | np.timedelta64):
        return to_str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


# A coordinate may be a time, which json has no type for. Written as the
# string DASCore writes every datetime as; `_coordinate` reads that
# spelling back, so a document holds the coordinates it was dumped from
# without anything downstream having to know which dimension is a time.
# One serializer rather than two stacked: a python-mode dump is what
# equality compares and what `new` rebuilds from, so it keeps the values.
def _serialize_coordinates(value, info):
    """Write a mapping of coordinates, as a document only in json mode."""
    if info.mode != "json":
        return dict(value)
    return {k: [_document(x) for x in values] for k, values in value.items()}


# Every spelling `to_str` gives a datetime64 from a whole date down. Numpy
# writes only the fields the value's unit carries, so a `datetime64[m]` is
# '2020-01-01T12:30' with no seconds to match, and requiring them read an
# ordinary pick time back as a string. A bare year or month is left out on
# purpose: '2020' is as readily a label as a time, and nothing tells them
# apart.
_DATETIME_TEXT = re.compile(r"^\d{4}-\d{2}-\d{2}(T\d{2}(:\d{2}(:\d{2}(\.\d+)?)?)?)?$")

# What numpy spells a duration with, which is what `to_str` writes and so
# what a stored curve over an offset dimension holds. Only the units a
# coordinate is held in are read: a month and a year are no fixed span,
# and numpy will not compare them against one.
_DurationUnit = Literal["ns", "us", "ms", "s", "m", "h", "D", "W"]
_DURATION_UNITS: dict[str, _DurationUnit] = {
    "nanoseconds": "ns",
    "microseconds": "us",
    "milliseconds": "ms",
    "seconds": "s",
    "minutes": "m",
    "hours": "h",
    "days": "D",
    "weeks": "W",
}
_DURATION_TEXT = re.compile(rf"^(-?\d+) ({'|'.join(_DURATION_UNITS)})$")


def _coordinate(value):
    """Read one coordinate, a time or duration written as text becoming one."""
    if not isinstance(value, str):
        return value
    if match := _DURATION_TEXT.match(value):
        # The spelling `to_str` gives a duration, which is how a curve over
        # an offset dimension is written down; read back, a basis holds the
        # coordinates it was dumped from rather than their text.
        count, unit = match.groups()
        return np.timedelta64(int(count), _DURATION_UNITS[unit])
    if not _DATETIME_TEXT.match(value):
        return value
    # Shaped like a date without being one: a label reading '2020-13-45'
    # is still the label it was written as.
    with suppress(ValueError, TypeError):
        return to_datetime64(value)
    return value


def _read_coordinates(value):
    """Read a mapping of coordinate sequences."""
    if not isinstance(value, Mapping):
        return value
    return {k: [_coordinate(x) for x in iterate(v)] for k, v in value.items()}


def _read_place(value):
    """Read a mapping of one coordinate per dimension, times at nanoseconds."""
    if not isinstance(value, Mapping):
        return value
    return {k: _nanoseconds(_scalar(_coordinate(v))) for k, v in value.items()}


def _nanoseconds(value):
    """Hold a numpy time or duration at nanoseconds; leave anything else."""
    if isinstance(value, np.datetime64):
        return value.astype(_NS_TIME)
    if isinstance(value, np.timedelta64):
        return value.astype(_NS_SPAN)
    return value


_freeze_map = AfterValidator(lambda x: FrozenDict(x))
_write_map = PlainSerializer(_serialize_coordinates, return_type=dict)
_read_map = BeforeValidator(_read_coordinates)

Bounds = Annotated[Mapping[str, tuple[Any, Any]], _read_map, _freeze_map, _write_map]

Vertices = Annotated[Mapping[str, tuple[Any, ...]], _read_map, _freeze_map, _write_map]


def _serialize_place(value, info):
    """Write a mapping of one coordinate per dimension, as `_serialize_coordinates`."""
    if info.mode != "json":
        return dict(value)
    return {k: _document(x) for k, x in value.items()}


Point = Annotated[
    Mapping[str, Any],
    BeforeValidator(_read_place),
    _freeze_map,
    PlainSerializer(_serialize_place, return_type=dict),
]


def _interpolate(start, end, fraction):
    """Walk from one coordinate to another, keeping whatever type it is."""
    if isinstance(start, np.datetime64):
        span = (end - start) / np.timedelta64(1, "ns")
        return start + (span * fraction).astype("timedelta64[ns]")
    return start + (end - start) * fraction


def _tag(name: str):
    """Return the serialization-only ``object_type`` field of a union member."""
    return Field(default=name, repr=False)


class _AnnotationModel(RichRepr, DascoreBaseModel):
    """Base for the immutable models an annotation set hands out."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def __rich__(self) -> Text:
        """One line naming the class and what it states."""
        return model_to_line(self)


# --- Curves ---------------------------------------------------------------


class AnnotationBasis(_AnnotationModel):
    """
    Base for the curves a path may be drawn from.

    A basis is not a geometry: it is a model a path's members came from, or
    the whole path where it has no members, sampled at any resolution.

    Every curve is stated in its dimensions' own coordinates -- a time is a
    time, a distance is a distance -- so it is anchored without a separate
    origin and its samples are annotation coordinates as they are. A curve
    parameterized in a dimension's raw numbers would put an apex at 1.6e18
    nanoseconds and a velocity in meters per nanosecond, which nobody can
    read, write or check.
    """

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions this curve is stated in."""
        raise NotImplementedError

    def vertices(self, count: int = 64) -> dict[str, np.ndarray]:
        """Return ``count`` points along the curve, keyed by dimension."""
        raise NotImplementedError

    @staticmethod
    def _fractions(count: int) -> np.ndarray:
        """Return where along the curve to sample, from one end to the other."""
        if count < 2:
            msg = f"A curve needs at least 2 points; got {count}."
            raise ParameterError(msg)
        return np.linspace(0.0, 1.0, count)


class Line(AnnotationBasis):
    """
    A straight line between two points.

    Stated as its endpoints rather than a slope, for two reasons: a slope
    cannot spell a line of constant time across distance -- an instant, a
    shot, a trigger -- which is an ordinary thing to annotate; and
    endpoints are what a person draws when they drag from one place to
    another.
    """

    object_type: Literal["Line"] = _tag("Line")
    start: Point = Field(description="Where the line begins, keyed by dimension.")
    end: Point = Field(description="Where the line ends, keyed by dimension.")

    @model_validator(mode="after")
    def _check_points(self) -> Self:
        """Both ends place the same dimensions, and are not the same place."""
        if not self.start:
            msg = "A line states no dimension, so it is nowhere."
            raise ValueError(msg)
        if set(self.start) != set(self.end):
            msg = (
                f"A line's ends place different dimensions: "
                f"{sorted(self.start)} and {sorted(self.end)}."
            )
            raise ValueError(msg)
        if all(self.start[x] == self.end[x] for x in self.start):
            msg = "A line begins and ends in the same place, so it has no length."
            raise ValueError(msg)
        return self

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions this line is stated in."""
        return tuple(self.start)

    def vertices(self, count: int = 64) -> dict[str, np.ndarray]:
        """Return ``count`` points evenly spaced from one end to the other."""
        fraction = self._fractions(count)
        return {
            dim: _interpolate(self.start[dim], self.end[dim], fraction)
            for dim in self.start
        }


class Moveout(AnnotationBasis):
    """
    The arrival time of a wavefront along the fiber, as a function of distance.

    Physics rather than geometry, so it is pinned to ``distance`` against
    ``time``. A source sitting ``standoff`` meters off the cable, abreast of
    fiber distance ``apex_distance``, arrives everywhere at

    ``time = apex_time + (hypot(standoff, distance - apex_distance)
    - standoff) / velocity``

    which is the hyperbola a point source makes. ``apex_time`` is therefore
    the earliest arrival and the curve's anchor. A source on the cable has
    no standoff, and the default of zero leaves the straight V of a wave
    running both ways at ``velocity``.
    """

    object_type: Literal["Moveout"] = _tag("Moveout")
    apex_distance: FiniteFloat = Field(
        description="Fiber distance the wavefront arrives earliest at, in meters."
    )
    apex_time: DateTime64 = Field(description="Time of that earliest arrival.")
    velocity: PositiveFiniteFloat = Field(
        description="Speed the wavefront moves along the fiber, in meters/second."
    )
    standoff: FiniteFloat = Field(
        default=0.0,
        ge=0,
        description=(
            "Perpendicular distance from the fiber to the source, in meters. "
            "Zero is a source on the cable, whose moveout is straight."
        ),
    )
    distance_min: FiniteFloat = Field(
        description="Fiber distance the curve is drawn from, in meters."
    )
    distance_max: FiniteFloat = Field(
        description="Fiber distance the curve is drawn to, in meters."
    )

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        """A curve with no span draws no vertices."""
        if not self.distance_max > self.distance_min:
            msg = (
                f"Moveout distance_max {self.distance_max} must exceed "
                f"distance_min {self.distance_min}."
            )
            raise ValueError(msg)
        return self

    @property
    def dims(self) -> tuple[str, ...]:
        """A moveout relates fiber distance to arrival time, and only those."""
        return (DISTANCE_DIM, TIME_DIM)

    def vertices(self, count: int = 64) -> dict[str, np.ndarray]:
        """Return ``count`` arrivals evenly spaced along the fiber."""
        fraction = self._fractions(count)
        distance = self.distance_min + fraction * (
            self.distance_max - self.distance_min
        )
        along = np.hypot(self.standoff, distance - self.apex_distance)
        seconds = (along - self.standoff) / self.velocity
        return {
            DISTANCE_DIM: distance,
            TIME_DIM: self.apex_time + to_timedelta64(seconds),
        }


Basis = Annotated[Line | Moveout, Field(discriminator="object_type")]


# --- Geometry -------------------------------------------------------------


class Region(_AnnotationModel):
    """
    Per-dimension bounds: where one annotation row is.

    Each dimension the row states maps to a half-open ``(start, end)`` pair;
    equal values are a point, and a dimension the mapping omits is
    unconstrained.
    """

    object_type: Literal["Region"] = _tag("Region")
    bounds: Bounds = Field(
        default_factory=dict, description="Half-open bounds, keyed by dimension."
    )

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions this region constrains."""
        return tuple(self.bounds)

    def is_point(self, dim: str) -> bool:
        """Whether this region is a point, rather than a span, along a dimension."""
        start, end = self.bounds[dim]
        return bool(start == end)


class Group(_AnnotationModel):
    """The unordered members of a group feature, one region each."""

    object_type: Literal["Group"] = _tag("Group")
    regions: tuple[Region, ...] = Field(
        min_length=1, description="The regions the members state, in table order."
    )


def _check_line(vertices: Mapping, least: int, what: str) -> None:
    """Refuse vertices naming no dimension, ragged, or too few."""
    if not vertices:
        msg = f"A {what} states no dimension, so it is nowhere."
        raise ValueError(msg)
    lengths = {len(x) for x in vertices.values()}
    if len(lengths) > 1:
        msg = (
            f"The vertices of this {what} differ in length ({sorted(lengths)}); "
            "every dimension states every point."
        )
        raise ValueError(msg)
    if (count := lengths.pop()) < least:
        msg = f"A {what} states {count} vertices; it is a shape with at least {least}."
        raise ValueError(msg)


class Path(_AnnotationModel):
    """
    An open sequence of vertices, e.g. a vehicle track.

    ``vertices`` holds one mapping of dimension to ordered coordinates per
    part; a single-part path holds one.
    """

    object_type: Literal["Path"] = _tag("Path")
    vertices: tuple[Vertices, ...] = Field(
        min_length=1, description="Ordered vertices keyed by dimension, per part."
    )
    basis: Basis | None = Field(
        default=None, description="The curve the path was drawn from."
    )

    @model_validator(mode="after")
    def _check_parts(self) -> Self:
        """Every part is a line in the same dimensions."""
        for part in self.vertices:
            _check_line(part, _LEAST["path"], "Path")
        if len({tuple(x) for x in self.vertices}) > 1:
            msg = "The parts of a Path are drawn in different dimensions."
            raise ValueError(msg)
        return self

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the vertices are stated in."""
        return tuple(self.vertices[0])


class Polygon(_AnnotationModel):
    """
    Closed rings of vertices bounding an area.

    ``vertices`` holds the parts, each a tuple of rings keyed by dimension:
    the first ring is the outer boundary, the rest are holes. Closure is
    implied, so a triangle states three points.
    """

    object_type: Literal["Polygon"] = _tag("Polygon")
    vertices: tuple[tuple[Vertices, ...], ...] = Field(
        min_length=1, description="Rings of vertices keyed by dimension, per part."
    )

    @model_validator(mode="after")
    def _check_parts(self) -> Self:
        """Every ring closes an area in the same dimensions."""
        for part in self.vertices:
            if not part:
                msg = "A Polygon part states no ring."
                raise ValueError(msg)
            for ring in part:
                _check_line(ring, _LEAST["polygon"], "Polygon")
        if len({tuple(x) for part in self.vertices for x in part}) > 1:
            msg = "The rings of a Polygon are drawn in different dimensions."
            raise ValueError(msg)
        return self

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the vertices are stated in."""
        return tuple(self.vertices[0][0])


Geometry = Annotated[
    Region | Group | Path | Polygon, Field(discriminator="object_type")
]

# Reads a basis, stated as the model or its document.
_BASIS_ADAPTER = TypeAdapter(Basis)


# --- Features -------------------------------------------------------------


class Feature(_AnnotationModel):
    """
    One feature of an [AnnotationSet](`dascore.core.annotations.AnnotationSet`).

    A view built on demand. A row with a blank ``feature_id`` is its own
    feature: its ``id`` is empty, its kind is a group, and its geometry is
    the row's [Region](`dascore.core.annotations.Region`).
    """

    id: str = Field(default="", description="The feature id; empty for a lone row.")
    name: str = Field(default="", description="A human-readable name.")
    kind: FeatureKind = Field(default="group", description="What the members compose.")
    geometry: Geometry = Field(description="Where the feature is.")
    basis: Basis | None = Field(
        default=None, description="The curve a path is drawn from, if any."
    )
    acquisition_key: AcquisitionKey = Field(
        default="",
        max_length=max_lens["acquisition_key"],
        description="Inventory identity of the annotated data, after fallback.",
    )
    data_id: str = Field(
        default="", description="Identity of the annotated data, after fallback."
    )
    set: str = Field(
        default="", description="The set this feature was read from, in a collection."
    )
    extra: FrozenDictType[str, Any] = Field(
        default_factory=dict, description="Columns the set does not model."
    )


# --- Set attributes -------------------------------------------------------


class AnnotationColumn(_AnnotationModel):
    """
    What a set says about one of its columns.

    Documenting a column never gates it: an undeclared column is carried
    as an extra just the same. A stated dtype is checked, so a column
    which says what it holds must hold it; a column read back from a CSV,
    which keeps no types, is given its stated dtype again.
    """

    description: str = Field(default="", description="What the column means.")
    units: UnitQuantity = Field(default=None, description="Units of the values.")
    dtype: str = Field(default="", description="Dtype the column must hold, if stated.")


class AnnotationSetAttrs(_AnnotationModel):
    """The attributes of an annotation set: what it describes and how."""

    dims: tuple[str, ...] = Field(
        description="The patch dimensions annotations are stated in."
    )
    creation_info: CreationInfo = Field(
        default_factory=CreationInfo,
        description=(
            "What produced these annotations, and when: a picker "
            "(author='phasenet', version='2.1', agency_id='INERIS') or a "
            "person (author='derrick'). Anything further -- a source file, a "
            "model checkpoint -- goes in its extra_fields."
        ),
    )
    acquisition_key: AcquisitionKey = Field(
        default="",
        max_length=max_lens["acquisition_key"],
        description=(
            "Inventory identity of the data these annotations were made on, "
            "spelled network.fiber_array.location.acquisition. The set-level "
            "default; a row may name its own."
        ),
    )
    data_id: str = Field(
        default="",
        description=(
            "Identity of the data the annotations were made on. The set-level "
            "default; a row may name its own."
        ),
    )
    annotation_columns: FrozenDictType[str, AnnotationColumn] = Field(
        default_factory=dict,
        description="Documentation for annotations columns, keyed by name.",
    )
    feature_columns: FrozenDictType[str, AnnotationColumn] = Field(
        default_factory=dict,
        description="Documentation for features columns, keyed by name.",
    )
    sets: FrozenDictType[str, AnnotationSetAttrs] = Field(
        default_factory=dict,
        description=(
            "The attributes of each set loaded together, keyed by the label "
            "the `set` column holds. What a child describes is its own table, "
            "not the merged one."
        ),
    )

    @model_validator(mode="after")
    def _check_dims(self) -> Self:
        """A set states which dimensions its annotations live in."""
        if not self.dims:
            msg = "An annotation set states at least one dimension."
            raise ValueError(msg)
        if len(set(self.dims)) != len(self.dims):
            msg = f"Annotation dimensions must be unique; got {list(self.dims)}."
            raise ValueError(msg)
        for dim in self.dims:
            for end in (_MIN, _MAX):
                stem = dim[: -len(end)] if dim.endswith(end) else None
                if stem and stem in self.dims:
                    msg = (
                        f"The dimension {dim!r} is spelled like the range column "
                        f"of {stem!r}, so one column would state both."
                    )
                    raise ValueError(msg)
        if overlap := sorted(set(self.dims) & set(RESERVED_COLUMNS)):
            msg = (
                f"The dimension(s) {', '.join(overlap)} name a reserved column; "
                f"a set may not dimension {', '.join(RESERVED_COLUMNS)}."
            )
            raise ValueError(msg)
        if private := sorted(x for x in self.dims if x.startswith(PRIVATE_PREFIX)):
            msg = (
                f"The dimension(s) {', '.join(private)} begin with an "
                "underscore, which names a column no set reads."
            )
            raise ValueError(msg)
        return self

    @model_validator(mode="after")
    def _check_sets(self) -> Self:
        """Sets loaded together are one collection, in its dimensions."""
        for name, child in self.sets.items():
            if child.sets:
                msg = (
                    f"The set {name!r} states sets of its own. Sets loaded "
                    "together are one collection, not a tree of them."
                )
                raise ValueError(msg)
            if extra := sorted(set(child.dims) - set(self.dims)):
                msg = (
                    f"The set {name!r} states the dimension(s) "
                    f"{', '.join(extra)}, which the sets loaded with it do not: "
                    f"they hold {list(self.dims)}."
                )
                raise ValueError(msg)
        return self


# --- The set --------------------------------------------------------------


class _Spelling(NamedTuple):
    """The columns one dimension is spelled with; either or both may exist."""

    dim: str
    point: str | None  # the bare value column
    low: str | None  # the range columns
    high: str | None


class AnnotationSet(NodeRepr, NamespaceOwner):
    """
    An immutable set of annotations, the features they compose, and bases.

    Parameters
    ----------
    annotations
        A dataframe, or anything one can be built from, of one row per
        located thing. Per dimension a row states ``<dim>``, a half-open
        ``<dim>_min``/``<dim>_max`` range, or nothing (spanning it). A
        ``feature_id`` names the feature a row belongs to; blank, the row is
        its own feature. ``seq``, ``part`` and ``ring`` order the members of
        a path or polygon.
    features
        A dataframe of one row per feature: a required ``id``, an optional
        ``geometry`` (blank for a group, ``path``, or ``polygon``), an
        optional ``basis`` key, and any other columns. Features hold no
        coordinates; a ``feature_id`` naming no row here creates a group.
    bases
        A mapping of key to curve, as a model or its document, e.g.
        ``{"m1": {"object_type": "Moveout", ...}}``. Only a path names one.
    dims
        Patch dimensions the annotations are stated in. Required unless
        ``attrs`` supplies them.
    attrs
        The set's attributes; ``dims`` and the keywords below override it.
    creation_info
        What produced the annotations, and when.
    acquisition_key
        Inventory identity of the annotated data.
    data_id
        Identity of the annotated data.
    annotation_columns
        Documentation for annotations columns, keyed by name.
    feature_columns
        Documentation for features columns, keyed by name.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> picks = pd.DataFrame(
    ...     {
    ...         "time": [1.0, 2.5],
    ...         "phase": ["P", "S"],
    ...         "feature_id": ["event_1", "event_1"],
    ...     }
    ... )
    >>> annotations = dc.AnnotationSet(picks, dims=("distance", "time"))
    >>> len(annotations)  # one feature, created from the feature_id
    1
    >>> feature = annotations["event_1"]
    >>> feature.kind, len(feature.geometry.regions)
    ('group', 2)
    """

    _namespace_entry_point_group: ClassVar[str] = "dascore.annotation_namespace"

    def __init__(
        self,
        annotations=None,
        features=None,
        bases: Mapping | None = None,
        dims: Sequence[str] | None = None,
        attrs: AnnotationSetAttrs | Mapping | None = None,
        creation_info: CreationInfo | Mapping | None = None,
        acquisition_key: str | None = None,
        data_id: str | None = None,
        annotation_columns: Mapping | None = None,
        feature_columns: Mapping | None = None,
    ):
        self._attrs = _build_attrs(
            attrs,
            dims=tuple(iterate(dims)) if dims is not None else None,
            creation_info=creation_info,
            acquisition_key=acquisition_key,
            data_id=data_id,
            annotation_columns=annotation_columns,
            feature_columns=feature_columns,
        )
        frame, self._spellings = _read_annotations(annotations, self._attrs)
        table = _read_features(features, self._attrs)
        self._bases = _read_bases(bases, self._attrs.dims)
        self._features = _add_implicit(frame, table, self._attrs.feature_columns)
        self._df = _check_members(frame, self._features, self._spellings, self._bases)

    # --- what the set is

    @property
    def attrs(self) -> AnnotationSetAttrs:
        """The set's attributes."""
        return self._attrs

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the annotations are stated in."""
        return self._attrs.dims

    # --- what the set holds

    @property
    def annotations(self) -> pd.DataFrame:
        """A copy of the annotations table, ``feature_id`` included."""
        return self._df.copy()

    @property
    def features(self) -> pd.DataFrame:
        """A copy of the features table, implicit features included."""
        return self._features.copy()

    @property
    def bases(self) -> FrozenDict:
        """The curves paths may be drawn from, keyed by name."""
        return self._bases

    def __len__(self) -> int:
        """The number of features, lone rows included."""
        return len(self._features) + int(self._lone().sum())

    def __iter__(self):
        """Iterate the features: the table's first, then each lone row."""
        for identity in self._features["id"]:
            yield self[identity]
        for row in _records(self._df[self._lone()]):
            yield self._lone_feature(row)

    def __getitem__(self, feature_id: str) -> Feature:
        """Return the feature with this id."""
        row = self._feature_row(feature_id)
        kind = cast(FeatureKind, _text(row.get("geometry")) or "group")
        return Feature(
            id=_text(row["id"]),
            name=_text(row.get("name")),
            kind=kind,
            geometry=self.geometry(feature_id),
            basis=self._basis(row),
            acquisition_key=self._fallback(row, "acquisition_key"),
            data_id=self._fallback(row, "data_id"),
            set=_text(row.get("set")),
            extra=_read_extra(row, FEATURE_COLUMNS),
        )

    def geometry(self, feature_id: str, count: int = 64) -> Group | Path | Polygon:
        """
        Build the geometry of one feature.

        Parameters
        ----------
        feature_id
            The id of a feature in the features table.
        count
            How many points to sample a path stated only by its basis at.
        """
        row = self._feature_row(feature_id)
        kind = _text(row.get("geometry")) or "group"
        members = self._df[self._df["feature_id"].map(_text) == _text(row["id"])]
        rows = _records(members)
        if kind == "group":
            bounds = (_read_bounds(x, self._spellings) for x in rows)
            return Group(regions=tuple(Region(bounds=x) for x in bounds))
        dims = _drawn_dims(rows, self._spellings)
        if kind == "path":
            basis = self._basis(row)
            if not rows:
                assert basis is not None  # a memberless path is refused without one
                drawn = basis.vertices(count)
                sampled = {k: [_scalar(x) for x in v] for k, v in drawn.items()}
                return Path(vertices=(sampled,), basis=basis)
            parts = _ordered_parts(rows, dims, self._spellings, ("part",))
            return Path(vertices=tuple(parts.values()), basis=basis)
        rings = _ordered_parts(rows, dims, self._spellings, ("part", "ring"))
        parts: dict[Any, list] = {}
        for (part, _), ring in rings.items():
            parts.setdefault(part, []).append(ring)
        return Polygon(vertices=tuple(tuple(x) for x in parts.values()))

    def _lone(self) -> pd.Series:
        """Which rows are their own feature."""
        return self._df["feature_id"].map(_text) == ""

    def _feature_row(self, feature_id) -> dict:
        """Return one features row by id, refusing an unknown one."""
        ids = self._features["id"].map(_text)
        found = self._features[ids == _text(feature_id)]
        if not len(found) or not _text(feature_id):
            msg = f"No feature has the id {feature_id!r}."
            raise KeyError(msg)
        return _records(found)[0]

    def _basis(self, row) -> Line | Moveout | None:
        """Return the curve a features row names, if any."""
        key = _text(row.get("basis"))
        return self._bases[key] if key else None

    def _lone_feature(self, row) -> Feature:
        """Build the feature a row with no feature_id is."""
        known = set(ANNOTATION_COLUMNS) | _spelled_columns(self._spellings)
        return Feature(
            name=_text(row.get("name")),
            geometry=Region(bounds=_read_bounds(row, self._spellings)),
            acquisition_key=self._fallback(row, "acquisition_key"),
            data_id=self._fallback(row, "data_id"),
            set=_text(row.get("set")),
            extra=_read_extra(row, known),
        )

    def _fallback(self, row, field: str) -> str:
        """Return a row's provenance, else its set's, else the collection's."""
        if stated := _text(row.get(field)):
            return stated
        child = self._attrs.sets.get(_text(row.get("set")))
        if child is not None and getattr(child, field):
            return getattr(child, field)
        return getattr(self._attrs, field)

    def __eq__(self, other) -> bool:
        """Two sets are equal when their attributes, tables and bases are."""
        if not isinstance(other, AnnotationSet):
            return NotImplemented
        return (
            self._attrs == other._attrs
            and _ordered_columns(self._df).equals(_ordered_columns(other._df))
            and _ordered_columns(self._features).equals(
                _ordered_columns(other._features)
            )
            and dict(self._bases) == dict(other._bases)
        )

    def _repr_node(self) -> Repr:
        """The banner, then what the set spans, holds, and says of itself."""
        blocks = [self._dims_text()]
        if contents := self._contents():
            blocks.append(mapping_to_text(contents, "Contents", style="dc_red"))
        attrs = stated_fields(self._attrs, skip=("dims",))
        if attrs:
            blocks.append(mapping_to_text(attrs, "Attributes"))
        return Repr(
            header=get_header_text("AnnotationSet \U0001f3f7"),
            body=tuple(split_block(x) for x in blocks),
        )

    def _dims_text(self) -> Text:
        """The extent each dimension is annotated over, and how it is spelled."""
        key_style = dascore_styles["keys"]
        base = Text("➤ ") + Text("Dimensions", style=dascore_styles["dc_blue"])
        base += Text(" (") + Text(", ".join(self.dims), style="bold") + Text(")")
        for dim in self.dims:
            spelling = self._spellings[dim]
            stated = {
                name: self._df[name].dropna()
                for name in (spelling.point, spelling.low, spelling.high)
                if name
            }
            columns = [x for x in stated.values() if len(x)]
            base += Text.assemble("\n    *", Text(dim, style="bold"), ": ")
            if not columns:
                base += Text("unstated", key_style)
                continue
            values = pd.concat(columns, ignore_index=True)
            low, high = values.min(), values.max()
            near, far = range_texts(low, high)
            # Appended one at a time so the label's style does not bleed.
            base += Text("min: ", key_style)
            base += near
            base += Text(" max: ", key_style)
            base += far
            if (span := span_text(low, high)) is not None:
                base += Text(" ") + span
            kinds = [
                kind
                for kind, name in (("value", spelling.point), ("range", spelling.low))
                if name and len(stated[name])
            ]
            base += Text(f" ({', '.join(kinds)})", key_style)
        return base

    def _contents(self) -> dict:
        """What the set holds, counted from its tables."""
        if not len(self._df) and not len(self._features) and not self._bases:
            return {}
        kinds = self._features["geometry"].map(lambda x: _text(x) or "group")
        counts = {
            "groups": int((kinds == "group").sum() + self._lone().sum()),
            "paths": int((kinds == "path").sum()),
            "polygons": int((kinds == "polygon").sum()),
        }
        contents = {
            "annotations": len(self._df),
            "features": Text(f"{len(self)} (") + counts_to_text(counts) + Text(")"),
            "bases": len(self._bases),
            "annotation columns": ", ".join(str(x) for x in self._df.columns),
        }
        if len(self._features):
            contents["feature columns"] = ", ".join(
                str(x) for x in self._features.columns
            )
        return contents


def _build_attrs(attrs, **overrides) -> AnnotationSetAttrs:
    """Build the set attributes from an attrs object and its overrides."""
    stated: dict[str, Any] = {}
    if attrs is not None:
        stated = (
            attrs.model_dump() if isinstance(attrs, AnnotationSetAttrs) else dict(attrs)
        )
    stated.update({k: v for k, v in overrides.items() if v is not None})
    if "dims" not in stated:
        msg = (
            "An annotation set states its dimensions; pass dims=... or attrs "
            "declaring them."
        )
        raise ParameterError(msg)
    return AnnotationSetAttrs(**stated)


def _declared_dtypes(columns: Mapping) -> frozenset[str]:
    """Return the columns whose dtype is stated, and so not decided here."""
    return frozenset(k for k, v in columns.items() if v.dtype)


def _read_annotations(data, attrs: AnnotationSetAttrs):
    """Return the checked annotations frame and each dimension's spelling."""
    frame = _coerce_frame(data, "annotations")
    declared = _declared_dtypes(attrs.annotation_columns)
    # Blanks first: a blank cell beside times written as text is unset.
    frame = _normalize_times(_normalize_blanks(frame, declared), attrs.dims)
    spellings = _read_spellings(frame, attrs.dims)
    text = [x for x in ANNOTATION_COLUMNS if x not in ORDINAL_COLUMNS]
    frame = _normalize_identities(frame, text, declared)
    _check_columns(frame, attrs, "annotations")
    _check_ranges(frame, spellings)
    _check_locations(frame, spellings)
    _check_set_labels(frame, attrs, "annotations")
    _check_ids(frame, "annotation", required=False)
    _check_keys(frame, "annotations")
    if "feature_id" not in frame.columns:
        frame = _assign(frame, {"feature_id": _blank_column(frame.index)})
    return frame, spellings


def _read_features(data, attrs: AnnotationSetAttrs) -> pd.DataFrame:
    """Return the checked features frame, geometry normalized."""
    frame = _coerce_frame(data, "features")
    declared = _declared_dtypes(attrs.feature_columns)
    frame = _normalize_times(_normalize_blanks(frame, declared))
    frame = _normalize_identities(frame, FEATURE_COLUMNS, declared)
    _check_columns(frame, attrs, "features")
    if len(frame) and "id" not in frame.columns:
        msg = "The features state no id column; annotations name a feature by id."
        raise ParameterError(msg)
    _check_ids(frame, "feature", required=True)
    _check_set_labels(frame, attrs, "features")
    _check_keys(frame, "features")
    changed = {"geometry": _read_geometry(frame)}
    if "id" not in frame.columns:
        changed["id"] = _blank_column(frame.index)
    return _assign(frame, changed)


def _read_geometry(frame: pd.DataFrame) -> pd.Series:
    """Return the geometry column, a group spelled blank."""
    cells = frame["geometry"] if "geometry" in frame.columns else [None] * len(frame)
    kinds = [_text(x) for x in cells]
    if unknown := sorted(set(kinds) - set(_GROUP_SPELLINGS) - set(_LEAST)):
        msg = (
            f"The geometry {', '.join(unknown)} is not one of "
            f"{', '.join(GEOMETRY_KINDS)}; a blank geometry is a group."
        )
        raise ParameterError(msg)
    read = [x if x in _LEAST else None for x in kinds]
    return pd.Series(read, index=frame.index, dtype=object)


def _read_bases(bases, dims) -> FrozenDict:
    """Return the bases as a frozen mapping of key to curve."""
    if bases is None:
        return FrozenDict()
    if not isinstance(bases, Mapping):
        msg = f"The bases are a mapping of key to curve; got {type(bases).__name__}."
        raise ParameterError(msg)
    out = {}
    for key, value in bases.items():
        if not _text(key):
            msg = "A basis is named by a nonblank key."
            raise ParameterError(msg)
        out[str(key)] = _read_basis(value, dims, str(key))
    return FrozenDict(out)


def _add_implicit(
    frame: pd.DataFrame, features: pd.DataFrame, columns: Mapping
) -> pd.DataFrame:
    """Append a group for every feature_id naming no features row."""
    ids = frame["feature_id"].map(_text)
    known = set(features["id"].map(_text))
    new = [x for x in dict.fromkeys(ids) if x and x not in known]
    if not new:
        return features
    rows = {"id": pd.Series(new, dtype=object)}
    if "set" in frame.columns:
        first = frame.groupby(ids.values, sort=False)["set"].first()
        rows["set"] = pd.Series([first[x] for x in new], dtype=object)
    out = pd.concat([features, pd.DataFrame(rows)], ignore_index=True, sort=False)
    # Concatenation fills text columns with NaN; blank is None here.
    text = {x: out[x] for x in out.columns if out[x].dtype == object}
    out = _assign(out, {k: v.where(v.notna(), None) for k, v in text.items()})
    try:
        _check_declared(out, columns)
    except ParameterError as error:
        msg = (
            f"The feature(s) {', '.join(new[:5])}, which feature_id implies, "
            f"leave a declared features column unstated: {error} Declare a "
            "dtype which holds a blank, such as Int64, or state the features."
        )
        raise ParameterError(msg) from error
    return out


def _check_members(frame, features, spellings, bases) -> pd.DataFrame:
    """
    Check each feature against its members; return the annotations frame
    with its order columns normalized.
    """
    ids = frame["feature_id"].map(_text)
    kinds = dict(
        zip(
            features["id"].map(_text),
            features["geometry"].map(lambda x: _text(x) or "group"),
            strict=True,
        )
    )
    ordered = ids.map(lambda x: kinds.get(x, "group") in _LEAST).to_numpy(bool)
    drawn = any(x in _LEAST for x in kinds.values())
    frame = _read_ordinals(frame, ordered, ids, drawn)
    members = frame.groupby(ids.values, sort=False).indices
    keys = features["basis"] if "basis" in features.columns else [None] * len(features)
    for identity, key in zip(features["id"].map(_text), keys, strict=True):
        kind, key = kinds[identity], _text(key)
        if key and key not in bases:
            msg = (
                f"The feature {identity!r} names the basis {key!r}, which is not "
                f"among the bases: {', '.join(sorted(bases)) or 'none'}."
            )
            raise ParameterError(msg)
        if key and kind != "path":
            msg = (
                f"The feature {identity!r} is a {kind} and names a basis; only a "
                "path is drawn from a curve."
            )
            raise ParameterError(msg)
        rows = members.get(identity, ())
        if not len(rows) and not key:
            msg = (
                f"The feature {identity!r} has no annotations and no basis, so "
                "nothing locates it."
            )
            raise ParameterError(msg)
        if kind in _LEAST and len(rows):
            basis = bases[key] if key else None
            _check_drawn(frame.iloc[rows], identity, kind, spellings, basis)
    return frame


def _read_ordinals(frame, ordered: np.ndarray, ids: pd.Series, drawn: bool):
    """
    Read seq, part and ring: whole numbers, only on ordered members, and
    held only where the set has a path or polygon.
    """
    present = [x for x in ORDINAL_COLUMNS if x in frame.columns]
    for name in present:
        stray = _stated_cells(frame[name]) & ~ordered
        if stray.any():
            rows = ", ".join(str(x) for x in frame.index[stray][:5])
            msg = (
                f"Row(s) {rows} state {name}, which orders the members of a path "
                "or polygon; their feature is neither."
            )
            raise ParameterError(msg)
    if not drawn:
        return frame.drop(columns=present)
    changed = {}
    for name in ORDINAL_COLUMNS:
        series = frame[name] if name in frame.columns else _blank_column(frame.index)
        stated = _stated_cells(series)
        values = read_ordinal(series)
        # Only a float can be fractional; integers stay integers, however big.
        whole = values % 1 == 0 if values.dtype.kind == "f" else True
        valid = (whole & (values >= 0)).fillna(False).to_numpy(bool)
        bad = stated & ~valid
        if bad.any():
            row = frame.index[bad][0]
            msg = (
                f"Row {row} states {name} {series[row]!r}; an ordinal is a "
                "non-negative whole number."
            )
            raise ParameterError(msg)
        values = values.astype("Int64")
        if name != "seq":
            values = values.where(stated | ~ordered, 0)
        changed[name] = values
    seq = changed["seq"]
    keys = [ids, changed["part"], changed["ring"]]
    blank = pd.Series(~_stated_cells(seq) & ordered, index=frame.index)
    groups = blank.groupby(keys, sort=False)
    if (groups.any() & ~groups.all()).any():
        msg = (
            "A path or polygon ring states seq on some members and not others; "
            "state it on all of them, or on none to take row order."
        )
        raise ParameterError(msg)
    order = blank.groupby(keys, sort=False).cumcount()
    changed["seq"] = seq.where(~blank, order)
    keep = pd.Series(ordered, index=frame.index)
    held = {k: v.where(keep, pd.NA) for k, v in changed.items()}
    return _assign(frame, held)


def _check_drawn(members, identity, kind, spellings, basis) -> None:
    """Refuse a path or polygon whose members do not draw one."""
    rows = _records(members)
    ranged = {
        dim
        for row in rows
        for dim, spelling in spellings.items()
        if spelling.low and _stated(row.get(spelling.low))
    }
    if ranged:
        msg = (
            f"The {kind} {identity!r} has a member stating a range of "
            f"{', '.join(sorted(ranged))}; a {kind} is drawn through values."
        )
        raise ParameterError(msg)
    drawn = {frozenset(_value_dims(row, spellings)) for row in rows}
    if len(drawn) > 1:
        msg = (
            f"The members of the {kind} {identity!r} state different dimensions; "
            "every vertex states the same ones."
        )
        raise ParameterError(msg)
    dims = _drawn_dims(rows, spellings)
    if kind == "polygon" and len(dims) < 2:
        msg = f"The polygon {identity!r} is drawn in {list(dims)}; an area needs two."
        raise ParameterError(msg)
    if basis is not None and set(basis.dims) != set(dims):
        msg = (
            f"The path {identity!r} is drawn in {sorted(dims)}, but its basis in "
            f"{sorted(basis.dims)}."
        )
        raise ParameterError(msg)
    if kind == "path" and (members["ring"] != 0).any():
        msg = f"The path {identity!r} states a ring; only a polygon has rings."
        raise ParameterError(msg)
    keys = ["part", "ring"]
    for (part, ring), group in members.groupby(keys, sort=True):
        where = f"part {part}" if kind == "path" else f"part {part} ring {ring}"
        if group["seq"].duplicated().any():
            msg = f"The {kind} {identity!r} repeats a seq in {where}."
            raise ParameterError(msg)
        points = [
            tuple(_scalar(row[spellings[d].point]) for d in dims)
            for row in _records(group.sort_values("seq", kind="stable"))
        ]
        distinct = kind == "polygon"
        if len(set(points) if distinct else points) < _LEAST[kind]:
            msg = (
                f"The {kind} {identity!r} states {len(points)} vertices in {where}; "
                f"it needs at least {_LEAST[kind]}{' distinct' if distinct else ''}."
            )
            raise ParameterError(msg)
        if kind == "polygon" and points[0] == points[-1]:
            msg = (
                f"The polygon {identity!r} repeats its first vertex last in "
                f"{where}; closure is implied."
            )
            raise ParameterError(msg)
    if kind == "polygon":
        outer = set(members.loc[members["ring"] == 0, "part"])
        if missing := sorted(set(members["part"]) - outer):
            msg = f"The polygon {identity!r} has part(s) {missing} with no ring 0."
            raise ParameterError(msg)


def _value_dims(row, spellings) -> list[str]:
    """The dimensions a row states a value of."""
    return [
        dim
        for dim, spelling in spellings.items()
        if spelling.point and _stated(row.get(spelling.point))
    ]


def _drawn_dims(rows, spellings) -> tuple[str, ...]:
    """The dimensions a path or polygon's members are drawn in."""
    return tuple(_value_dims(rows[0], spellings)) if rows else ()


def _ordered_parts(rows, dims, spellings, keys) -> dict:
    """Return vertices keyed by dimension, per ``keys`` group, in seq order."""
    ordered = sorted(rows, key=lambda x: tuple(int(x[k]) for k in (*keys, "seq")))
    out: dict[Any, dict[str, list]] = {}
    for row in ordered:
        group = tuple(int(row[k]) for k in keys)
        group = group[0] if len(group) == 1 else group
        vertex = out.setdefault(group, {dim: [] for dim in dims})
        for dim in dims:
            vertex[dim].append(_scalar(row[spellings[dim].point]))
    return out


def _records(frame: pd.DataFrame) -> list[dict]:
    """Return rows as mappings, each column keeping its own type."""
    columns = {str(name): list(frame[name]) for name in frame.columns}
    return [{k: v[i] for k, v in columns.items()} for i in range(len(frame))]


def _blank_column(index) -> pd.Series:
    """A column stating nothing, held as text is."""
    return pd.Series([None] * len(index), index=index, dtype=object)


def _spelled_columns(spellings) -> set[str]:
    """Every column a dimension is spelled by."""
    return {x for s in spellings.values() for x in (s.point, s.low, s.high) if x}


def _coerce_frame(data, what: str) -> pd.DataFrame:
    """Return the input as a dataframe, an empty one where nothing was given."""
    if data is None:
        return pd.DataFrame()
    if isinstance(data, pd.DataFrame):
        frame = data.reset_index(drop=True)
    else:
        try:
            frame = pd.DataFrame(data).reset_index(drop=True)
        except (ValueError, TypeError) as error:
            msg = f"Could not read the {what} as a dataframe: {error}."
            raise ParameterError(msg) from error
    # Pandas allows a name to repeat, and then getting that column hands
    # back a frame rather than a column; everything downstream reads one
    # column per name, so the repeat is refused where it can still be seen.
    if frame.columns.duplicated().any():
        repeated = sorted({str(x) for x in frame.columns[frame.columns.duplicated()]})
        msg = (
            f"The {what} name {', '.join(repeated)} more than once; one column "
            "states one thing."
        )
        raise ParameterError(msg)
    # A table names a column by a string, so any other name would come
    # back from a saved set as the string it was written as.
    if odd := [repr(x) for x in frame.columns if not isinstance(x, str)][:5]:
        msg = (
            f"The {what} name the column(s) {', '.join(odd)} by something other "
            "than a string, which is what a table names a column by."
        )
        raise ParameterError(msg)
    # Here rather than where a file is read, so a set holds what a stored
    # one holds: a private column is the author's own either way, and a set
    # which kept one from a frame would write a column it could not read
    # back.
    kept = drop_private_columns(frame)
    # A table writes rows by writing their cells, so rows with no cell to
    # write are rows a saved set comes back without. Refused rather than
    # counted here, where what went missing can still be named.
    if len(kept.index) and not len(kept.columns):
        msg = f"The {what} state rows and no column to hold them."
        if len(frame.columns):
            msg += " Every column they state is private, so none is theirs."
        raise ParameterError(msg)
    return _freeze_cells(kept)


def _freeze_cells(frame: pd.DataFrame) -> pd.DataFrame:
    """Hold nested cells immutably, so a shallow copy cannot reach them."""
    nested = list | tuple | set | dict | Mapping | np.ndarray
    changed = {}
    for name in frame.columns:
        series = frame[name]
        if series.dtype != object or not any(isinstance(x, nested) for x in series):
            continue
        cells = [_freeze(x) if isinstance(x, nested) else x for x in series]
        changed[name] = pd.Series(cells, index=series.index, dtype=object)
    return _assign(frame, changed)


def _read_spellings(frame: pd.DataFrame, dims) -> dict[str, _Spelling]:
    """
    Return the columns each dimension is spelled with.

    A dimension may have a value column, a range pair, both, or neither; a
    range is spelled by both of its columns or not at all.
    """
    columns = set(frame.columns)
    out = {}
    for dim in dims:
        point = dim if dim in columns else None
        start = f"{dim}{_MIN}" if f"{dim}{_MIN}" in columns else None
        end = f"{dim}{_MAX}" if f"{dim}{_MAX}" in columns else None
        if (start is None) != (end is None):
            stated = start or end
            missing = f"{dim}{_MAX}" if start is not None else f"{dim}{_MIN}"
            msg = f"{stated} states half a range; {missing} is not a column."
            raise ParameterError(msg)
        out[dim] = _Spelling(dim, point, start, end)
    return out


def _check_columns(frame: pd.DataFrame, attrs: AnnotationSetAttrs, table: str):
    """
    Refuse a column which nearly names something, and check stated dtypes.

    On annotations an undeclared ``<name>_min``/``<name>_max`` pair is a
    forgotten dimension, and ``geometry`` or ``basis`` belong to features.
    On features any dimension column is refused: features hold no
    coordinates.
    """
    spelled = {x for dim in attrs.dims for x in (dim, f"{dim}{_MIN}", f"{dim}{_MAX}")}
    if table == "features":
        if coords := sorted(spelled & set(frame.columns)):
            msg = (
                f"The features state the coordinate column(s) {', '.join(coords)}; "
                "a feature is located by its annotations, which hold coordinates."
            )
            raise ParameterError(msg)
        rows_only = ("feature_id", *ORDINAL_COLUMNS)
        if misplaced := sorted(set(rows_only) & set(frame.columns)):
            msg = (
                f"The features state {', '.join(misplaced)}, which an annotation "
                "states to name and order itself within a feature."
            )
            raise ParameterError(msg)
        _check_declared(frame, attrs.feature_columns)
        return
    if misplaced := sorted({"geometry", "basis"} & set(frame.columns)):
        msg = (
            f"The annotations state {', '.join(misplaced)}, which a feature "
            "states: put it on the features table, and name the feature with "
            "feature_id."
        )
        raise ParameterError(msg)
    extras = [str(x) for x in frame.columns if x not in set(RESERVED_COLUMNS) | spelled]
    low, high = _RETIRED_RANGE
    retired = {x[: -len(low)] for x in extras if x.endswith(low)}
    retired &= {x[: -len(high)] for x in extras if x.endswith(high)}
    if named := ", ".join(sorted(retired & set(attrs.dims))):
        msg = (
            f"The column(s) {named} state a range as {low}/{high}, which this "
            f"format now spells {_MIN}/{_MAX}, as every other range in DASCore "
            "is spelled. Rename the columns."
        )
        raise ParameterError(msg)
    stems = {x[: -len(_MIN)] for x in extras if x.endswith(_MIN)}
    stems &= {x[: -len(_MAX)] for x in extras if x.endswith(_MAX)}
    if stems:
        named = ", ".join(sorted(stems))
        msg = (
            f"The column(s) {named} are spelled as a range but name no declared "
            f"dimension. The set declares {list(attrs.dims)}."
        )
        raise ParameterError(msg)
    _check_declared(frame, attrs.annotation_columns)


def _check_declared(frame: pd.DataFrame, columns: Mapping) -> None:
    """Refuse a column which does not hold the dtype it declares."""
    for name, column in columns.items():
        if not column.dtype or name not in frame.columns:
            continue
        actual = frame[name].dtype
        # Through pandas: a column may hold an extension dtype numpy lacks.
        try:
            declared = pd.api.types.pandas_dtype(column.dtype)
        except TypeError as error:
            msg = f"The column {name!r} declares the dtype {column.dtype!r}: {error}."
            raise ParameterError(msg) from error
        if not _dtype_matches(declared, frame[name]):
            if declared.kind in "Mm":
                msg = (
                    f"The column {name!r} states dtype {column.dtype}, but a set "
                    f"holds every time at nanoseconds: state {actual} instead."
                )
                raise ParameterError(msg)
            msg = f"The column {name!r} states dtype {column.dtype} but holds {actual}."
            raise ParameterError(msg)


def _check_locations(frame: pd.DataFrame, spellings) -> None:
    """
    Refuse a row stating a dimension two ways or none at all, and a
    dimension holding two kinds of coordinate.
    """
    somewhere = np.zeros(len(frame), dtype=bool)
    for dim, spelling in spellings.items():
        value = _stated_mask(frame, spelling.point)
        ranged = _stated_mask(frame, spelling.low)
        if (both := value & ranged).any():
            row = frame.index[both][0]
            msg = (
                f"Row {row} states {dim} as a value and as a range; a row states "
                "one or the other."
            )
            raise ParameterError(msg)
        somewhere |= value | ranged
        names = (spelling.point, spelling.low, spelling.high)
        kinds = {
            _coordinate_kind(frame[x])
            for x in names
            if x and _stated_cells(frame[x]).any()
        }
        if len(kinds) > 1:
            msg = (
                f"The dimension {dim!r} holds {' and '.join(sorted(kinds))}; a "
                "dimension holds one kind of coordinate."
            )
            raise ParameterError(msg)
    if len(frame) and not somewhere.all():
        rows = ", ".join(str(x) for x in frame.index[~somewhere][:5])
        msg = (
            f"Row(s) {rows} state no dimension; an annotation states where it is "
            "along at least one."
        )
        raise ParameterError(msg)


def _stated_mask(frame: pd.DataFrame, column: str | None) -> np.ndarray:
    """Which cells of a column state anything; none where there is no column."""
    if column is None:
        return np.zeros(len(frame), dtype=bool)
    return _stated_cells(frame[column])


def _coordinate_kind(series: pd.Series) -> str:
    """Name the kind of coordinate a dimension column holds."""
    kind = getattr(series.dtype, "kind", "")
    return {"M": "times", "m": "durations"}.get(kind, "numbers")


def _check_ids(frame: pd.DataFrame, what: str, required: bool) -> None:
    """Refuse a repeated id, and a blank one where ids are required."""
    if "id" not in frame.columns:
        return
    ids = frame["id"].map(_text)
    if required and (ids == "").any():
        rows = ", ".join(str(x) for x in frame.index[ids == ""][:5])
        msg = f"Row(s) {rows} of the {what}s state no id; every {what} has one."
        raise ParameterError(msg)
    stated = ids[ids != ""]
    if stated.duplicated().any():
        repeated = sorted(set(stated[stated.duplicated()]))
        msg = f"The {what} id(s) {', '.join(repeated)} name more than one row."
        raise ParameterError(msg)


def _check_keys(frame: pd.DataFrame, table: str) -> None:
    """Refuse a row-level acquisition key which is not one."""
    if "acquisition_key" not in frame.columns:
        return
    for row, key in zip(frame.index, frame["acquisition_key"], strict=True):
        try:
            validate_acquisition_key(_text(key))
        except InvalidInventoryError as error:
            msg = f"Row {row} of the {table}: {error}"
            raise ParameterError(msg) from error


TEXT_DTYPES = frozenset({"object", "str", "string"})


def _dtype_matches(declared, series: pd.Series) -> bool:
    """
    Whether a column holds its declared dtype, however pandas spells text.

    Text has several spellings -- `object`, `str`, `string` -- and which
    one a column gets depends on the pandas version and on what wrote it,
    so a declaration of any of them is a declaration of text. An `object`
    column may hold anything, so it is text only if its cells are.
    """
    actual = series.dtype
    if declared.name == actual.name:
        return True
    if declared.name not in TEXT_DTYPES or actual.name not in TEXT_DTYPES:
        return False
    return actual.name != "object" or all(
        isinstance(x, str) for x in series if _stated(x)
    )


def _check_ranges(frame: pd.DataFrame, spellings) -> None:
    """
    Refuse a range which ends before it starts.

    Checked here rather than where a row is read: an impossible range is
    structural, so a set holding one does not load at all instead of
    raising later, on whichever operation happened to touch that row.
    """
    for dim, spelling in spellings.items():
        if spelling.low is None or spelling.high is None:
            continue
        start, end = frame[spelling.low], frame[spelling.high]
        stated = start.notna() & end.notna()
        half = start.notna() ^ end.notna()
        if half.any():
            first = frame.index[half][0]
            msg = (
                f"Row {first} states half a {dim} range; a range states both "
                f"{spelling.low} and {spelling.high}, and neither states an "
                "unconstrained dimension."
            )
            raise ParameterError(msg)
        if not stated.any():
            continue
        try:
            reversed_rows = stated & (end < start)
        except TypeError as error:
            msg = (
                f"The {dim} range cannot be compared, so nothing can say "
                f"whether it runs backwards: {error}."
            )
            raise ParameterError(msg) from error
        if reversed_rows.any():
            first = frame.index[reversed_rows][0]
            msg = (
                f"Row {first} states a {dim} range ({start[first]}, {end[first]}) "
                "which ends before it starts."
            )
            raise ParameterError(msg)


def _check_set_labels(frame: pd.DataFrame, attrs: AnnotationSetAttrs, table: str):
    """
    Refuse a row whose set label names none of the sets stated.

    Only checked where sets are stated: a set on its own may carry a `set`
    column meaning whatever it means.
    """
    if not attrs.sets or frame.empty:
        return
    stated = ", ".join(sorted(attrs.sets))
    if "set" not in frame.columns:
        msg = (
            f"This states the sets {stated} and its {table} have no set column, "
            "so no row says which of them it came from."
        )
        raise ParameterError(msg)
    labels = frame["set"].map(_text)
    if not labels.all():
        rows = ", ".join(str(x) for x in frame.index[labels == ""][:5])
        msg = (
            f"Row(s) {rows} of the {table} state no set, where the sets {stated} "
            "are stated. A row loaded with others says which of them it came from."
        )
        raise ParameterError(msg)
    if unknown := sorted(set(labels) - set(attrs.sets)):
        msg = (
            f"The set label(s) {', '.join(unknown)} name no set stated here, "
            f"which states {stated}. A label reaches back to what its set says "
            "about itself, so it names one of them."
        )
        raise ParameterError(msg)


def _read_bounds(row, spellings) -> dict[str, tuple[Any, Any]]:
    """Return the half-open bounds a row states, per dimension."""
    out = {}
    for dim, spelling in spellings.items():
        value = row.get(spelling.point) if spelling.point else None
        if _stated(value):
            out[dim] = (_scalar(value), _scalar(value))
            continue
        start = row.get(spelling.low) if spelling.low else None
        end = row.get(spelling.high) if spelling.high else None
        if _stated(start) and _stated(end):
            out[dim] = (_scalar(start), _scalar(end))
    return out


def _read_extra(row, known: Collection[str]) -> dict[str, Any]:
    """Return the stated cells of a row the set does not model."""
    return {
        str(k): _freeze(v) for k, v in row.items() if str(k) not in known and _stated(v)
    }


def _read_basis(value, dims, key: str) -> Line | Moveout:
    """Return the curve a basis entry states, as the model or its document."""
    if isinstance(value, Line | Moveout):
        basis = value
    else:
        try:
            basis = _BASIS_ADAPTER.validate_python(value)
        except ValidationError as error:
            named = ", ".join(sorted(x.__name__ for x in (Line, Moveout)))
            msg = f"Could not read the basis {key!r} as a {named}: {error}."
            raise ParameterError(msg) from error
    if foreign := sorted(set(basis.dims) - set(dims)):
        msg = (
            f"The basis {key!r} names the dimension(s) {', '.join(foreign)}, which "
            f"the set does not declare. It declares {list(dims)}."
        )
        raise ParameterError(msg)
    return basis


def _freeze(value):
    """
    Return a value nothing can change through the set which handed it out.

    A frame cell may hold a list or a dict, and copying a frame does not
    copy those; an annotation which handed one back would let its holder
    edit a set which says it is immutable.
    """
    if isinstance(value, Mapping):
        return FrozenDict({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list | set | tuple | np.ndarray):
        return tuple(_freeze(x) for x in value)
    return _scalar(value)


def read_dimension(series: pd.Series, where: str = "") -> pd.Series:
    """
    Read a dimension column as the numbers or times its cells state.

    Numbers are tried first because every datetime spelling this writes is
    an ISO string, which is not a number, while seconds from the epoch are
    a number a distance column would lose to a date. Times arriving at
    another resolution are held at nanoseconds, the resolution DASCore
    keeps them at, and a column holding neither numbers nor times is
    refused: a dimension is a coordinate, and the set's own store would
    refuse to read it back. A duration is a coordinate too, and is read
    where the cells hold one. `where` names the source in the refusal.
    """
    kind = getattr(series.dtype, "kind", "")
    if kind in "iuf":
        return series
    if kind == "M":
        return (
            series
            if series.dtype == np.dtype("datetime64[ns]")
            else to_datetime64(series)
        )
    if kind == "m":
        return (
            series
            if series.dtype == np.dtype("timedelta64[ns]")
            else to_timedelta64(series)
        )
    stated = _stated_cells(series)
    if not stated.any():
        return series
    # Through `_scalar`: python's own date and duration types are
    # coordinates a frame may plainly hold, and the readers below take
    # numpy's. Built rather than mapped, which would box a numpy scalar
    # back into the pandas type it came as.
    cells = pd.Series(
        [_scalar(x) for x in series[stated]], index=series.index[stated], dtype=object
    )
    if kind != "b" and not any(_is_bool(x) for x in cells):
        with suppress(TypeError, ValueError):
            # The kind it converted *to*, not merely that it converted:
            # `to_numeric` hands a complex column straight back, and a
            # complex number is no coordinate.
            if (read := pd.to_numeric(series)).dtype.kind in _NUMBER_KINDS:
                return read
        # A number already read as one cannot also be a time: reading the
        # column as times would take that number for an epoch and place
        # the row in 1970 rather than where it says.
        if not any(_is_number(x) for x in cells):
            readers = ((to_datetime64, _NS_TIME), (to_timedelta64, _NS_SPAN))
            # A duration is asked about first where the cells plainly are
            # durations: the time reader takes one as a count from the
            # epoch. Text keeps the other order, being the spelling a time
            # is written in and a duration is not.
            if all(isinstance(x, np.timedelta64 | datetime.timedelta) for x in cells):
                readers = tuple(reversed(readers))
            for read_as, dtype in readers:
                # AssertionError and NotImplementedError: these two state
                # that way what they cannot read; OverflowError: a year no
                # coordinate holds.
                with suppress(
                    TypeError,
                    ValueError,
                    AssertionError,
                    NotImplementedError,
                    OverflowError,
                ):
                    values = np.asarray(read_as(cells))
                    # Checked rather than trusted: these read text nobody
                    # can place -- a label, a name -- as NaT rather than
                    # refusing it, and taking that would delete the cell
                    # rather than say it is not a coordinate.
                    if not pd.isnull(values).any():
                        out = pd.Series(
                            np.array("NaT", dtype=dtype),
                            index=series.index,
                            dtype=dtype,
                        )
                        out[stated] = values
                        return out
        msg = (
            f"The column {series.name!r}{where} states neither numbers, times "
            "nor durations, so its values are not coordinates an annotation "
            "can be placed at."
        )
        raise ParameterError(msg)
    msg = (
        f"The column {series.name!r}{where} holds {cells.iloc[0]!r}, where a "
        "dimension holds numbers or times."
    )
    raise ParameterError(msg)


def read_ordinal(series: pd.Series, where: str = "") -> pd.Series:
    """Read an order column (seq, part or ring) as the numbers it states."""
    # Refused before `to_numeric`, which would count a truth value as 1 or 0.
    if getattr(series.dtype, "kind", "") == "b" or any(_is_bool(x) for x in series):
        error = "it holds truth values"
    else:
        try:
            order = pd.to_numeric(series)
        except (TypeError, ValueError) as exc:
            error = str(exc)
        else:
            # `to_numeric` hands a complex column straight back.
            if order.dtype.kind in _NUMBER_KINDS:
                return order
            error = f"it holds {order.dtype}"
    msg = (
        f"The column {series.name!r}{where} is not numeric: {error}. It "
        "states a member's place in an order, which is a number."
    )
    raise ParameterError(msg)


def _is_bool(value) -> bool:
    """Whether a cell is a truth value, numpy's included."""
    return isinstance(value, bool | np.bool_)


def _is_number(value) -> bool:
    """
    Whether a cell already holds a number, a boolean among them.

    A time and a duration are not numbers here, whatever python says of
    them: they are the two kinds this reads a column *into*.
    """
    if isinstance(value, np.datetime64 | np.timedelta64):
        return False
    return isinstance(value, numbers.Number | np.bool_)


def _normalize_identities(
    frame: pd.DataFrame, columns, declared: Collection[str] = ()
) -> pd.DataFrame:
    """
    Hold every identity as the text which names it.

    An id is a label, not a number: a table reads one back as text, so a
    set built from a frame and the same set reloaded would otherwise name
    one row two things.

    A whole number is named without a `.0` wherever one appears, since a
    blank beside it is all it takes for pandas to spell the column in
    floats -- and a feature's id and a row's feature_id are read from
    columns which need not each hold a blank. Naming them from what each
    column happens to hold would let one name `1` where another names
    `1.0`, and the row they both mean would be an orphan.
    """
    changed = {}
    for name in columns:
        if name not in frame.columns or name in declared:
            continue
        series = frame[name]
        if series.dtype == object and all(
            isinstance(x, str) for x in series if _stated(x)
        ):
            continue
        changed[name] = pd.Series(
            [_identity(x) for x in series], index=series.index, dtype=object
        )
    return _assign(frame, changed)


def _identity(value):
    """Return the text one identity is named by, an unstated one as nothing."""
    if not _stated(value):
        return None
    value = _scalar(value)
    # Beyond where a float counts by ones it no longer names one integer,
    # so its own text is the most that can be said of it.
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        return str(int(value))
    return str(value)


def _ordered_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a frame whose columns are in one order, whatever order it holds."""
    return frame[sorted(frame.columns, key=str)]


def _assign(frame: pd.DataFrame, changed: Mapping) -> pd.DataFrame:
    """Return the frame with these columns replaced, or itself where none are."""
    if not changed:
        return frame
    # Assigned by item rather than by keyword: a column need not be named
    # anything a keyword can spell.
    out = frame.copy()
    for name, series in changed.items():
        out[name] = series
    return out


def _normalize_times(frame: pd.DataFrame, dims: Sequence[str] = ()) -> pd.DataFrame:
    """
    Hold every time at nanoseconds, and read each dimension column.

    A column arriving at another resolution states the same times, but
    everything which reads one back -- a stored table, a coordinate, a
    curve -- states them at DASCore's, so a set which kept both spellings
    would differ from itself over nothing. A dimension column is read as
    the numbers or times it states, text included, for the same reason:
    the geometry a row builds reads that spelling back, so a frame which
    kept the text would disagree with the region built from it.
    """
    spelled = {x for dim in dims for x in (dim, f"{dim}{_MIN}", f"{dim}{_MAX}")}
    changed = {}
    for name in frame.columns:
        series = frame[name]
        kind = getattr(series.dtype, "kind", "")
        if name in spelled:
            read = read_dimension(series)
            if read is not series:
                changed[name] = read
        elif kind == "M" and series.dtype != np.dtype("datetime64[ns]"):
            changed[name] = to_datetime64(series)
        elif kind == "m" and series.dtype != _NS_SPAN:
            changed[name] = to_timedelta64(series)
    return frame.assign(**changed) if changed else frame


def _stated_cells(series: pd.Series) -> np.ndarray:
    """Return which cells of a column state anything, as a plain mask."""
    # Not `Series.map`: on a categorical column it returns a categorical,
    # which pandas 3 refuses to reduce with `any`.
    return np.array([bool(_stated(x)) for x in series], dtype=bool)


def _normalize_blanks(
    frame: pd.DataFrame, declared: Collection[str] = ()
) -> pd.DataFrame:
    """
    Read a cell holding the empty string as stating nothing, in one dtype.

    A table writes an unset cell and a cell holding the empty string the
    same way, and reads that back as unset, so a set says unset for both
    rather than holding a value which cannot survive being written down.

    Two dtypes are settled here for the same reason. A column no row
    states -- an all-blank value column is how a membership-only set is
    spelled -- arrives as whatever each reader inferred from nothing, and
    each format infers a different one. A category is a dtype only a frame
    has, and every table reads that column back as the text it holds. A
    column which declares its dtype is left alone: an author saying what a
    column holds outranks a canonical form.
    """
    changed = {}
    for name in frame.columns:
        series = frame[name]
        blank = np.array([isinstance(x, str) and not x for x in series], dtype=bool)
        if blank.any():
            series = series.where(~blank, None)
            changed[name] = series
        if name in declared:
            continue
        if isinstance(series.dtype, pd.CategoricalDtype):
            series = series.astype(object)
            changed[name] = series
        if series.dtype != object and not _stated_cells(series).any():
            changed[name] = pd.Series(
                [None] * len(frame), index=frame.index, dtype=object
            )
    return _assign(frame, changed)


def _table_suffix(format: str) -> str:
    """Return the suffix an encoding is named by, refusing an unknown one."""
    suffix = f".{str(format).lower().lstrip('.')}"
    if suffix not in TABLE_SUFFIXES:
        named = ", ".join(x.lstrip(".") for x in TABLE_SUFFIXES)
        msg = f"{format!r} is not a table encoding; a set is written as {named}."
        raise ParameterError(msg)
    return suffix


def _spell_table(frame: pd.DataFrame, suffix: str, dims: Sequence[str] | None = None):
    """
    Spell one table for the encoding its suffix names.

    Spelled before the directory is touched, so whichever encoding is
    asked for, a table which cannot be written raises with the stored set
    still whole.
    """
    if suffix == TABLE_SUFFIX:
        return _write_table(frame)
    # A parquet file has no comment line to declare its dimensions in, so
    # they go in the metadata its footer holds.
    metadata = None if dims is None else {DIMS_KEY: json.dumps(list(dims))}
    return parquet_table(frame, metadata)


def _write_spelled(payload, path) -> None:
    """Write what `_spell_table` spelled, whichever encoding it is."""
    if isinstance(payload, str):
        _write_text(payload, path)
    else:
        write_parquet_table(payload, path)


def _write_table(frame: pd.DataFrame, path=None) -> str:
    """Return a frame as CSV text, optionally writing it to a path."""
    spelled = pd.DataFrame({name: _writable(frame[name]) for name in frame.columns})
    text = spelled.to_csv(index=False)
    if path is not None:
        _write_text(text, path)
    return text


def _write_text(text: str, path) -> None:
    """Write table text exactly as it was spelled."""
    # newline="" so the line terminators pandas wrote are the ones which
    # land, rather than each one growing a carriage return on Windows.
    with open(path, "w", newline="", encoding="utf-8") as stream:
        stream.write(text)


def _writable(series: pd.Series) -> pd.Series:
    """
    Return one column as the text a table states it with.

    Built as an object column rather than mapped: pandas would infer a
    float column from an int beside an unset cell and write the int as
    `5.0`, which reads back as a float.
    """
    cells = [_writable_cell(x) for x in series]
    return pd.Series(cells, index=series.index, dtype=object)


def _json_default(value):
    """
    Spell a nested value json has no type of its own for.

    A hook which hands back what it was given is re-dispatched until json
    reports a circular reference, so anything this cannot spell goes in as
    its text: a bare `ValueError: Circular reference detected` names
    neither the cell nor the file it was being written to.
    """
    if isinstance(value, Mapping):
        return dict(value)
    spelled = _writable_cell(value)
    return spelled if spelled is not value else str(value)


def _writable_cell(value):
    """
    Spell one cell the way a table holds it.

    A value a CSV has no column shape for is written as its own document:
    a mapping as its JSON, a sequence as comma-separated text. An extra
    holding a nested object survives as that text rather than as the object.

    The same holds for an extra holding a time: only a declared dimension
    is known to hold times, so only it is read back as one, and an extra
    keeps the text it was written as.
    """
    if not _stated(value):
        return value
    # Through _scalar first: mapping a datetime column hands over pandas
    # Timestamps, which str() spells with a space where numpy uses a T,
    # and only the numpy spelling reads back as a time.
    value = _scalar(value)
    if isinstance(value, np.datetime64 | np.timedelta64):
        return to_str(value)
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return json.dumps(dict(value), default=_json_default)
    if isinstance(value, Iterable):
        return ", ".join(str(_writable_cell(x)) for x in value)
    return value


def _scalar(value):
    """
    Return a cell's value as the plainest thing which still says it.

    Numpy numbers become python ones, so a bound reads as the number it
    is; datetimes and timedeltas keep their numpy type, which is what
    makes an endpoint on a time dimension a time rather than an integer.
    """
    if isinstance(value, datetime.datetime | datetime.date):
        return to_datetime64(value)
    if isinstance(value, datetime.timedelta):
        # pd.Timedelta is one of these, and so is the stdlib's own.
        return np.timedelta64(value)
    if isinstance(value, np.generic) and value.dtype.kind not in "mM":
        return value.item()
    return value


def _text(value) -> str:
    """Return a cell as text, an unstated one as the empty string."""
    return str(value) if _stated(value) else ""


def _stated(value) -> bool:
    """Whether a cell states anything at all."""
    # Nested cells are frozen to tuples and FrozenDicts, which pandas
    # reads as one object rather than element by element.
    return value is not None and not bool(pd.isnull(value))


def annotation_set_to_csv(
    annotations: AnnotationSet, path: str | pathlib.Path | None = None
) -> str:
    """
    Return the annotations table as CSV text, optionally writing it to a path.

    Reached as ``annotation_set.io.to_csv``.

    A bare table holds only annotations, so a set with features beyond the
    groups its ``feature_id`` column implies, or with bases, is written with
    [save](`dascore.core.annotations.save_annotation_set`) instead. A
    duration dimension is refused: CSV has no spelling for one which reads
    back, where parquet has a type for it.

    The dimensions are not written. Reading the table back states them
    again, in the call or in a ``# dims: distance, time`` line written above
    the header by hand.

    Parameters
    ----------
    annotations
        The set to write.
    path
        Where to write the text, or None to only return it.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame(
    ...     {"phase": ["P"], "distance_min": [10.0], "distance_max": [80.0]}
    ... )
    >>> annotations = dc.AnnotationSet(frame, dims=("time", "distance"))
    >>> "phase" in annotations.io.to_csv()
    True
    """
    _refuse_bare(annotations)
    _refuse_unwritable_durations(annotations)
    return _write_table(annotations._df, path)


def annotation_set_to_parquet(
    annotations: AnnotationSet, path: str | pathlib.Path
) -> pathlib.Path:
    """
    Write the annotations table as one parquet file.

    Reached as ``annotation_set.io.to_parquet``.

    The parquet spelling of
    [to_csv](`dascore.core.annotations.annotation_set_to_csv`): columns keep
    their types, and the dimensions travel in the file's metadata. A column
    with no one type is written as JSON, which keeps each cell's value but
    not every python type -- a tuple comes back as a list. Needs pyarrow.

    Parameters
    ----------
    annotations
        The set to write.
    path
        Where to write the file.

    Returns
    -------
    The path written to, so a save reads straight back.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame(
    ...     {"phase": ["P"], "distance_min": [10.0], "distance_max": [80.0]}
    ... )
    >>> annotations = dc.AnnotationSet(frame, dims=("time", "distance"))
    >>> path = annotations.io.to_parquet("picks.parquet")  # doctest: +SKIP
    >>> dc.annotations(path) == annotations  # doctest: +SKIP
    True
    """
    _refuse_bare(annotations)
    dims = json.dumps(list(annotations.dims))
    write_parquet(annotations._df, path, {DIMS_KEY: dims})
    return pathlib.Path(path)


def _refuse_unwritable_durations(annotations: AnnotationSet) -> None:
    """Refuse to write a duration dimension as CSV, which nothing reads back."""
    spelled = _spelled_columns(annotations._spellings)
    frame = annotations._df
    named = sorted(
        str(name)
        for name in frame.columns
        if str(name) in spelled and getattr(frame[name].dtype, "kind", "") == "m"
    )
    if named:
        msg = (
            f"The dimension column(s) {', '.join(named)} hold durations, which "
            "a CSV has no spelling for: written as text, nothing reads them "
            "back as durations. Write the set as parquet, which keeps them."
        )
        raise ParameterError(msg)


def _refuse_bare(annotations: AnnotationSet) -> None:
    """Refuse to write a set a bare annotations table cannot rebuild."""
    attrs = annotations.attrs
    implied = _add_implicit(
        annotations._df, _read_features(None, attrs), attrs.feature_columns
    )
    features = _ordered_columns(annotations._features)
    if annotations._bases or not features.equals(_ordered_columns(implied)):
        msg = (
            "This set holds features or bases, which a bare annotations table "
            "has no row for. Save it as a directory with io.save."
        )
        raise ParameterError(msg)


def save_annotation_set(
    annotations: AnnotationSet, path: str | pathlib.Path, format: str = "csv"
) -> pathlib.Path:
    """
    Write the set to a directory, creating it if needed.

    Reached as ``annotation_set.io.save``. The directory holds
    ``attrs.json``, the ``annotations`` table, a ``features`` table when the
    set has features, and ``bases.json`` when it has bases; it reads back
    through [dascore.annotations](`dascore.annotations`). Tables are CSV by
    default or parquet when requested; duration dimensions require parquet.
    Saving replaces stale parts and alternate-format files.

    Parameters
    ----------
    annotations
        The set to write.
    path
        The directory to write into.
    format
        The encoding the tables are written in: ``csv`` or ``parquet``.

    Returns
    -------
    The directory written to, so a save reads straight back.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame(
    ...     {"phase": ["P"], "distance_min": [10.0], "distance_max": [80.0]}
    ... )
    >>> annotations = dc.AnnotationSet(frame, dims=("time", "distance"))
    >>> directory = annotations.io.save("picks")  # doctest: +SKIP
    >>> dc.annotations(directory) == annotations  # doctest: +SKIP
    True
    """
    # Everything is spelled before the directory is touched, so a refusal
    # leaves the stored set whole.
    suffix = _table_suffix(format)
    if suffix == TABLE_SUFFIX:
        _refuse_unwritable_durations(annotations)
    tables = {ANNOTATION_STEM: _spell_table(annotations._df, suffix, annotations.dims)}
    if len(annotations._features):
        tables[FEATURE_STEM] = _spell_table(annotations._features, suffix)
    documents = {
        ATTRS_STEM: annotations._attrs.model_dump(mode="json", exclude_defaults=True)
    }
    if annotations._bases:
        documents[BASES_STEM] = {
            k: v.model_dump(mode="json") for k, v in annotations._bases.items()
        }
    directory = pathlib.Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    json_suffix = OBJECT_SUFFIXES[0]
    writing = {
        *(directory / f"{x}{json_suffix}" for x in documents),
        *(directory / f"{x}{suffix}" for x in tables),
    }
    # The retired vertices table is claimed so saving over an old set clears it.
    claimed = {
        ATTRS_STEM: OBJECT_SUFFIXES,
        BASES_STEM: OBJECT_SUFFIXES,
        ANNOTATION_STEM: TABLE_SUFFIXES,
        FEATURE_STEM: TABLE_SUFFIXES,
        "vertices": TABLE_SUFFIXES,
    }
    superseded = [
        x
        for x in directory.iterdir()
        if x.suffix.casefold() in claimed.get(x.stem, ()) and x not in writing
    ]
    # Written before stale parts are cleared, so a failed write leaves the
    # old set readable rather than gone.
    for stem, document in documents.items():
        write_document(document, directory / f"{stem}{json_suffix}", "json")
    for stem, payload in tables.items():
        _write_spelled(payload, directory / f"{stem}{suffix}")
    for stale in superseded:
        # On a case-insensitive filesystem a shouted name is the file just
        # written, and unlinking it would take the part with it.
        if any(_one_file(stale, x) for x in writing):
            continue
        stale.unlink(missing_ok=True)
    return directory


def _one_file(one: pathlib.Path, other: pathlib.Path) -> bool:
    """Whether two names reach one file, as a case-insensitive store lets them."""
    try:
        return one.samefile(other)
    except OSError:
        # One of them is gone, so they are not the same file.
        return False
