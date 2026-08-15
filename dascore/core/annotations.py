"""
DASCore annotations: labelled geometry over the dimensions of patch data.

An annotation set describes *data* -- picks, events, noisy hours, vehicle
lines -- in the frame of the patches it was made on. Facts about the fiber
itself belong to the inventory instead, in optical distance; see
[dascore.core.inventory](`dascore.core.inventory`).

A set is dataframe-backed, one row per annotation. Columns name the
dimensions a row constrains: ``<dim>_start``/``<dim>_end`` state a
half-open range, a bare ``<dim>`` states a point, and a dimension no
column names is unconstrained. Paths and polygons keep their vertices in
a second, tidy frame keyed by annotation id, and every row keeps a
bounding region so table operations work whatever its geometry is.
"""

from __future__ import annotations

import datetime
from collections.abc import Iterable, Mapping, Sequence
from typing import Annotated, Any, ClassVar, Literal, NamedTuple

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
from typing_extensions import Self

from dascore.constants import max_lens
from dascore.core.inventory import CreationInfo
from dascore.exceptions import ParameterError
from dascore.models import (
    DascoreBaseModel,
    DateTime64,
    FiniteFloat,
    FrozenDictType,
    PositiveFiniteFloat,
    UnitQuantity,
)
from dascore.utils.intervals import normalize_value, value_kind
from dascore.utils.mapping import FrozenDict
from dascore.utils.misc import iterate, to_str, validate_acquisition_key
from dascore.utils.time import to_datetime64, to_timedelta64

# Columns any set may carry, whatever dimensions it declares.
RESERVED_COLUMNS = (
    "id",
    "group",
    "value",
    "tags",
    "parent",
    "geometry",
    "basis",
    "acquisition_key",
)

# The geometries a row may declare. A region is the default: it is what
# the dimension columns already state, so a set of boxes names nothing.
GEOMETRY_KINDS = ("region", "path", "polygon")

# Geometries whose shape lives in the vertices frame rather than in the
# dimension columns, and the fewest vertices each is a shape with.
_VERTEX_KINDS = {"path": 2, "polygon": 3}

# The vertices frame's own scaffolding; every other column is a dimension.
_VERTEX_COLUMNS = ("id", "seq")

# What a range column is spelled with.
_START, _END = "_start", "_end"

# A moveout is physics, not geometry, so it names the dimensions it relates.
DISTANCE_DIM, TIME_DIM = "distance", "time"


def _annotation_value(value):
    """Normalize an annotation value so its Python type survives validation."""
    return normalize_value(value, error=ParameterError)


# The value kind decides a group's shape, so it must be exact: a bool is
# membership and an int is a number, and 1 must not become True.
AnnotationValue = Annotated[
    str | bool | int | float, BeforeValidator(_annotation_value)
]


# Spelled as PatchAttrs spells them, so a set and the data it describes
# state their provenance the same way.
AcquisitionKey = Annotated[str, AfterValidator(validate_acquisition_key)]

# `iterate` rather than a bare tuple: PatchAttrs.history is `str |
# tuple[str, ...]`, so copying one straight off a patch may hand over a
# lone entry, which is a history of one rather than a history of letters.
History = Annotated[tuple[str, ...], BeforeValidator(lambda x: tuple(iterate(x)))]


def _document(value):
    """Spell a bound or a vertex for a json document."""
    if isinstance(value, np.datetime64 | np.timedelta64):
        return to_str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


# A coordinate may be a time, which json has no type for. Written as the
# string DASCore writes every datetime as; reading it back as a time again
# is the loader's job, since it knows what each dimension holds. One
# serializer rather than two stacked: a python-mode dump is what equality
# compares and what `new` rebuilds from, so it keeps the values themselves.
def _serialize_coordinates(value, info):
    """Write a mapping of coordinates, as a document only in json mode."""
    if info.mode != "json":
        return dict(value)
    return {k: [_document(x) for x in values] for k, values in value.items()}


_freeze_map = AfterValidator(lambda x: FrozenDict(x))
_write_map = PlainSerializer(_serialize_coordinates, return_type=dict)

Bounds = Annotated[Mapping[str, tuple[Any, Any]], _freeze_map, _write_map]

Vertices = Annotated[Mapping[str, tuple[Any, ...]], _freeze_map, _write_map]


def _serialize_place(value, info):
    """Write a mapping of one coordinate per dimension, as `_serialize_coordinates`."""
    if info.mode != "json":
        return dict(value)
    return {k: _document(x) for k, x in value.items()}


Point = Annotated[
    Mapping[str, Any],
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


class _AnnotationModel(DascoreBaseModel):
    """Base for the immutable models an annotation set hands out."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )


# --- Curves ---------------------------------------------------------------


class AnnotationBasis(_AnnotationModel):
    """
    Base for the curves a path's vertices may be regenerated from.

    A basis is not a geometry: it is the fit the vertices came from, kept
    so the curve can be redrawn at any resolution. Editing vertices drops
    it, since they no longer describe the curve.

    Every curve is stated in its dimensions' own coordinates -- a time is a
    time, a distance is a distance -- so it is anchored without a separate
    origin and its vertices drop straight into the vertices frame. A curve
    parameterized in a dimension's raw numbers would put an apex at 1.6e18
    nanoseconds and a velocity in metres per nanosecond, which nobody can
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
    ``time``. A source sitting ``standoff`` metres off the cable, abreast of
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
        description="Fiber distance the wavefront arrives earliest at, in metres."
    )
    apex_time: DateTime64 = Field(description="Time of that earliest arrival.")
    velocity: PositiveFiniteFloat = Field(
        description="Speed the wavefront moves along the fiber, in metres/second."
    )
    standoff: FiniteFloat = Field(
        default=0.0,
        ge=0,
        description=(
            "Perpendicular distance from the fiber to the source, in metres. "
            "Zero is a source on the cable, whose moveout is straight."
        ),
    )
    distance_start: FiniteFloat = Field(
        description="Fiber distance the curve is drawn from, in metres."
    )
    distance_end: FiniteFloat = Field(
        description="Fiber distance the curve is drawn to, in metres."
    )

    @model_validator(mode="after")
    def _check_span(self) -> Self:
        """A curve with no span draws no vertices."""
        if not self.distance_end > self.distance_start:
            msg = (
                f"Moveout distance_end {self.distance_end} must exceed "
                f"distance_start {self.distance_start}."
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
        distance = self.distance_start + fraction * (
            self.distance_end - self.distance_start
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
    Per-dimension bounds: the box an annotation occupies.

    Each dimension the annotation constrains maps to a half-open
    ``(start, end)`` pair; equal values are a point, and a dimension the
    mapping omits is unconstrained. Region subsumes a point, a span and a
    box, which differ only in how many dimensions they name.
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


class _VertexGeometry(_AnnotationModel):
    """Base for geometries whose shape is a sequence of vertices."""

    region: Region = Field(description="The bounding region of the vertices.")
    vertices: Vertices = Field(description="Ordered vertex values, keyed by dimension.")
    basis: Basis | None = Field(
        default=None, description="The curve these vertices were generated from."
    )

    # The fewest vertices this geometry is a shape with.
    _least: ClassVar[int] = 2

    @model_validator(mode="after")
    def _check_vertices(self) -> Self:
        """Vertices place every dimension, equally, and enough times."""
        if not self.vertices:
            msg = f"A {type(self).__name__} states no dimension, so it is nowhere."
            raise ValueError(msg)
        lengths = {len(x) for x in self.vertices.values()}
        if len(lengths) > 1:
            msg = (
                f"The vertices of this {type(self).__name__} differ in length "
                f"({sorted(lengths)}); every dimension states every point."
            )
            raise ValueError(msg)
        if (count := lengths.pop()) < self._least:
            msg = (
                f"A {type(self).__name__} states {count} vertices; it is a "
                f"shape with at least {self._least}."
            )
            raise ValueError(msg)
        return self

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the vertices are stated in."""
        return tuple(self.vertices)

    def __len__(self) -> int:
        """The number of vertices."""
        return len(next(iter(self.vertices.values())))


class Path(_VertexGeometry):
    """An open sequence of vertices, e.g. a pick or a vehicle track."""

    object_type: Literal["Path"] = _tag("Path")


class Polygon(_VertexGeometry):
    """
    A closed sequence of vertices bounding an area.

    Closure is implied rather than written: the last vertex is not the
    first repeated, so a triangle states three points.
    """

    object_type: Literal["Polygon"] = _tag("Polygon")
    _least: ClassVar[int] = 3


Geometry = Annotated[Region | Path | Polygon, Field(discriminator="object_type")]

# Reads a basis cell, which a set may state as the model or as its document.
_BASIS_ADAPTER = TypeAdapter(Basis)


# --- The annotation row view ----------------------------------------------


class Annotation(_AnnotationModel):
    """
    One annotation: a geometry, what it says, and who it belongs to.

    Instances are a view of a row of an
    [AnnotationSet](`dascore.core.annotations.AnnotationSet`), built on
    demand rather than held.
    """

    geometry: Geometry = Field(description="Where this annotation is.")
    id: str = Field(default="", description="Producer-supplied stable identifier.")
    group: str = Field(default="", description="Name of the annotated variable.")
    value: AnnotationValue = Field(
        default=True, description="Value of the variable over this geometry."
    )
    tags: tuple[str, ...] = Field(
        default=(), description="Free labels; an annotation may carry many."
    )
    parent: str = Field(
        default="", description="Id of the annotation this one belongs to."
    )
    acquisition_key: AcquisitionKey = Field(
        default="",
        max_length=max_lens["acquisition_key"],
        description=(
            "Inventory identity of the data this annotation was made on; the "
            "set's own where the row names none."
        ),
    )
    extra: FrozenDictType[str, Any] = Field(
        default_factory=dict, description="Columns the set does not model."
    )

    @property
    def region(self) -> Region:
        """The bounding region, whatever the geometry is."""
        geometry = self.geometry
        return geometry if isinstance(geometry, Region) else geometry.region


# --- Set attributes -------------------------------------------------------


class AnnotationColumn(_AnnotationModel):
    """
    What a set says about one of its columns.

    Documenting a column never gates it: an undeclared column is carried
    as an extra just the same. A stated dtype is checked, so a column
    which says what it holds must hold it.
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
    history: History = Field(
        default=(),
        description=(
            "Processing performed on the data the annotations were made on. "
            "Picks made on decimated or filtered data have coordinates which "
            "only mean anything against that lineage."
        ),
    )
    columns: FrozenDictType[str, AnnotationColumn] = Field(
        default_factory=dict, description="Documentation for columns, keyed by name."
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
            for end in (_START, _END):
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
        return self


# --- The set --------------------------------------------------------------


class _Spelling(NamedTuple):
    """How one dimension is spelled in the frame."""

    dim: str
    point: str | None  # the bare column, where the dimension is a point
    start: str | None  # the range columns, where it is a span
    end: str | None


class AnnotationSet:
    """
    An immutable, dataframe-backed set of annotations over patch dimensions.

    Parameters
    ----------
    data
        A dataframe, or anything a dataframe can be built from, holding one
        row per annotation.
    dims
        The patch dimensions the annotations are stated in. Required unless
        ``attrs`` states them.
    vertices
        A tidy frame of one row per vertex, with ``id``, ``seq`` and one
        column per dimension. Required by every path and polygon row.
    attrs
        The set's attributes; ``dims`` and the keywords below override it.
    creation_info
        What produced the annotations, and when.
    acquisition_key
        Inventory identity of the data the annotations were made on. The
        producer supplies it; ``AnnotationSet.from_patch`` will take it off
        a patch once the spool integration lands.
    history
        Processing performed on that data, in ``PatchAttrs.history``'s shape.
    columns
        Documentation for columns, keyed by name.

    Examples
    --------
    >>> import pandas as pd
    >>> import dascore as dc
    >>> frame = pd.DataFrame(
    ...     {"group": ["event"], "distance_start": [10.0], "distance_end": [80.0]}
    ... )
    >>> picks = dc.AnnotationSet(frame, dims=("time", "distance"))
    >>> len(picks), picks[0].group
    (1, 'event')
    >>> picks[0].region.bounds["distance"]
    (10.0, 80.0)
    """

    def __init__(
        self,
        data=None,
        dims: Sequence[str] | None = None,
        vertices=None,
        attrs: AnnotationSetAttrs | Mapping | None = None,
        creation_info: CreationInfo | Mapping | None = None,
        acquisition_key: str | None = None,
        history: Sequence[str] | str | None = None,
        columns: Mapping | None = None,
    ):
        self._attrs = _build_attrs(
            attrs, dims, creation_info, acquisition_key, history, columns
        )
        frame = _coerce_frame(data, "annotations")
        spellings = _read_spellings(frame, self._attrs.dims)
        _check_columns(frame, self._attrs)
        _check_ranges(frame, spellings)
        _check_values(frame)
        ids = _check_ids(frame)
        _check_basis(frame, self._attrs.dims)
        vertex_frame = _coerce_frame(vertices, "vertices")
        self._vertices = _check_vertices(vertex_frame, frame, ids, self._attrs.dims)
        self._df = _fill_vertex_bounds(frame, self._vertices, spellings)
        # Read again: filling a derived bounding region adds the range
        # columns a path row had none of, which are bounds like any other.
        self._spellings = _read_spellings(self._df, self._attrs.dims)

    # --- what the set is

    @property
    def attrs(self) -> AnnotationSetAttrs:
        """The set's attributes."""
        return self._attrs

    @property
    def dims(self) -> tuple[str, ...]:
        """The dimensions the annotations are stated in."""
        return self._attrs.dims

    def to_dataframe(self) -> pd.DataFrame:
        """Return the annotations as a dataframe."""
        return self._df.copy()

    def to_vertices(self) -> pd.DataFrame:
        """Return the vertices of every path and polygon as a tidy dataframe."""
        return self._vertices.copy()

    # --- what the set holds

    def __len__(self) -> int:
        """The number of annotations."""
        return len(self._df)

    def __iter__(self):
        """Iterate the annotations, building each on demand."""
        for position in range(len(self._df)):
            yield self[position]

    def __getitem__(self, position: int) -> Annotation:
        """Return one annotation by its position."""
        row = self._df.iloc[position]
        return Annotation(
            geometry=self._geometry(row),
            id=_text(row.get("id")),
            group=_text(row.get("group")),
            value=row["value"] if _stated(row.get("value")) else True,
            tags=_read_tags(row.get("tags")),
            parent=_text(row.get("parent")),
            # A set may span acquisitions, so a row naming one overrides
            # the set-level address rather than sitting beside it.
            acquisition_key=_text(row.get("acquisition_key"))
            or self._attrs.acquisition_key,
            extra=_read_extra(row, self._attrs.dims, self._spellings),
        )

    def __eq__(self, other) -> bool:
        """Two sets are equal when their attributes and frames are."""
        if not isinstance(other, AnnotationSet):
            return NotImplemented
        return (
            self._attrs == other._attrs
            and self._df.equals(other._df)
            and self._vertices.equals(other._vertices)
        )

    def __repr__(self) -> str:
        """Name the set by what it holds."""
        dims = ", ".join(self.dims)
        return f"AnnotationSet({len(self)} annotations, dims=({dims}))"

    __str__ = __repr__

    # --- internals

    def _geometry(self, row) -> Region | Path | Polygon:
        """Build the geometry a row states."""
        region = Region(bounds=_read_bounds(row, self._spellings))
        kind = _text(row.get("geometry")) or "region"
        if kind == "region":
            return region
        vertices = _row_vertices(self._vertices, row["id"], self.dims)
        model = Path if kind == "path" else Polygon
        return model(
            region=region,
            vertices=vertices,
            basis=_read_basis(row.get("basis"), self.dims),
        )


def _build_attrs(
    attrs, dims, creation_info, acquisition_key, history, columns
) -> AnnotationSetAttrs:
    """Build the set attributes from an attrs object and its overrides."""
    stated: dict[str, Any] = {}
    if attrs is not None:
        stated = (
            attrs.model_dump() if isinstance(attrs, AnnotationSetAttrs) else dict(attrs)
        )
    overrides = {
        "dims": tuple(iterate(dims)) if dims is not None else None,
        "creation_info": creation_info,
        "acquisition_key": acquisition_key,
        "history": history,
        "columns": columns,
    }
    stated.update({k: v for k, v in overrides.items() if v is not None})
    if "dims" not in stated:
        msg = (
            "An annotation set states its dimensions; pass dims=... or attrs "
            "declaring them."
        )
        raise ParameterError(msg)
    return AnnotationSetAttrs(**stated)


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
    return frame


def _read_spellings(frame: pd.DataFrame, dims) -> dict[str, _Spelling]:
    """
    Return how each dimension is spelled in the frame.

    A dimension is a point where a bare column names it, a span where a
    start/end pair does, and unconstrained where neither does. Spelling it
    both ways states one thing twice, which has no answer.
    """
    columns = set(frame.columns)
    out = {}
    for dim in dims:
        point = dim if dim in columns else None
        start = f"{dim}{_START}" if f"{dim}{_START}" in columns else None
        end = f"{dim}{_END}" if f"{dim}{_END}" in columns else None
        if point is not None and (start is not None or end is not None):
            msg = (
                f"The dimension {dim!r} is spelled both as a point ({dim}) and "
                f"as a range ({dim}{_START}/{dim}{_END}); it is one or the other."
            )
            raise ParameterError(msg)
        if (start is None) != (end is None):
            stated = start or end
            missing = f"{dim}{_END}" if start is not None else f"{dim}{_START}"
            msg = f"{stated} states half a range; {missing} is not a column."
            raise ParameterError(msg)
        out[dim] = _Spelling(dim, point, start, end)
    return out


def _check_columns(frame: pd.DataFrame, attrs: AnnotationSetAttrs) -> None:
    """
    Refuse a column which nearly names something, and check stated dtypes.

    An undeclared ``<name>_start``/``<name>_end`` pair is a dimension the
    set forgot to declare rather than two unrelated extras, and reading it
    as extras would quietly drop the constraint it states.
    """
    known = set(RESERVED_COLUMNS) | set(attrs.dims)
    known |= {f"{dim}{end}" for dim in attrs.dims for end in (_START, _END)}
    extras = [str(x) for x in frame.columns if x not in known]
    stems = {x[: -len(_START)] for x in extras if x.endswith(_START)}
    stems &= {x[: -len(_END)] for x in extras if x.endswith(_END)}
    if stems:
        named = ", ".join(sorted(stems))
        msg = (
            f"The column(s) {named} are spelled as a range but name no declared "
            f"dimension. The set declares {list(attrs.dims)}."
        )
        raise ParameterError(msg)
    for name, column in attrs.columns.items():
        if not column.dtype or name not in frame.columns:
            continue
        actual = frame[name].dtype
        # Resolved and compared through pandas rather than numpy: an
        # annotation column may hold an extension dtype -- `str`, which is
        # what pandas gives plain text, or `category`, or `Int64` -- and
        # `np.dtype` knows none of them.
        try:
            declared = pd.api.types.pandas_dtype(column.dtype)
        except TypeError as error:
            msg = f"The column {name!r} declares the dtype {column.dtype!r}: {error}."
            raise ParameterError(msg) from error
        # Compared by name rather than by identity: a column documented as
        # `category` says it is categorical, not which categories it holds,
        # and the two dtypes are otherwise unequal. Names still tell a
        # datetime64[ns] from a datetime64[us].
        if declared.name != actual.name:
            msg = f"The column {name!r} states dtype {column.dtype} but holds {actual}."
            raise ParameterError(msg)


def _check_ranges(frame: pd.DataFrame, spellings) -> None:
    """
    Refuse a range which ends before it starts.

    Checked here rather than where a row is read: an impossible range is
    structural, so a set holding one does not load at all instead of
    raising later, on whichever operation happened to touch that row.
    """
    for dim, spelling in spellings.items():
        if spelling.start is None or spelling.end is None:
            continue
        start, end = frame[spelling.start], frame[spelling.end]
        stated = start.notna() & end.notna()
        half = start.notna() ^ end.notna()
        if half.any():
            first = frame.index[half][0]
            msg = (
                f"Row {first} states half a {dim} range; a range states both "
                f"{spelling.start} and {spelling.end}, and neither states an "
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


def _check_values(frame: pd.DataFrame) -> None:
    """
    Refuse a group whose values are not all one kind.

    A group's kind decides its shape -- boolean groups state membership and
    may overlap, others are single valued -- so a group holding both is two
    variables sharing a name. Overlap itself is not checked here: it only
    means anything where the set is projected onto a coordinate.
    """
    if "value" not in frame.columns:
        return
    # Mapped to text first: pandas drops a null grouping key, which would
    # take every unnamed row out of the check that its group holds one kind.
    groups = (
        frame["group"].map(_text)
        if "group" in frame.columns
        else pd.Series([""] * len(frame))
    )
    for name, index in frame.groupby(groups.values, sort=True).groups.items():
        values = [x for x in frame.loc[index, "value"] if _stated(x)]
        kinds = {value_kind(_annotation_value(x)) for x in values}
        if len(kinds) > 1:
            msg = (
                f"The annotation group {str(name)!r} mixes {sorted(kinds)} values; "
                "a group holds one kind of value."
            )
            raise ParameterError(msg)


def _check_ids(frame: pd.DataFrame) -> pd.Series:
    """
    Check the identity columns and return the id of every row.

    Ids are the producer's: nothing is generated here, so an operation
    needing identity is available exactly where one was supplied.
    """
    if "id" not in frame.columns:
        ids = pd.Series([""] * len(frame), dtype=object)
    else:
        ids = frame["id"].map(_text)
    stated = ids[ids != ""]
    if stated.duplicated().any():
        repeated = sorted(set(stated[stated.duplicated()]))
        msg = f"The annotation id(s) {', '.join(repeated)} name more than one row."
        raise ParameterError(msg)
    kinds = frame["geometry"].map(_text) if "geometry" in frame.columns else None
    if kinds is not None:
        if unknown := sorted(set(kinds) - set(GEOMETRY_KINDS) - {""}):
            msg = (
                f"The geometry {', '.join(unknown)} is not one of "
                f"{', '.join(GEOMETRY_KINDS)}."
            )
            raise ParameterError(msg)
        needs_id = kinds.isin(list(_VERTEX_KINDS)) & (ids == "")
        if needs_id.any():
            rows = ", ".join(str(x) for x in frame.index[needs_id][:5])
            msg = (
                f"Row(s) {rows} state a path or polygon but no id; vertices are "
                "grouped by id, so one is required."
            )
            raise ParameterError(msg)
    if "basis" in frame.columns:
        has_basis = frame["basis"].map(_stated)
        # No geometry column means every row is a region, which is exactly
        # the case a basis does not belong to.
        is_vertex = (
            kinds.isin(list(_VERTEX_KINDS))
            if kinds is not None
            else pd.Series(False, index=frame.index)
        )
        stray = has_basis & ~is_vertex
        if stray.any():
            rows = ", ".join(str(x) for x in frame.index[stray][:5])
            msg = (
                f"Row(s) {rows} state a basis but no path or polygon; a basis "
                "is the curve vertices were generated from."
            )
            raise ParameterError(msg)
    if "parent" in frame.columns:
        parents = frame["parent"].map(_text)
        orphans = sorted(set(parents[parents != ""]) - set(stated))
        if orphans:
            msg = (
                f"The parent id(s) {', '.join(orphans)} name no annotation in this set."
            )
            raise ParameterError(msg)
    return ids


def _check_vertices(vertices, frame, ids, dims) -> pd.DataFrame:
    """Check that every vertex geometry has vertices, and only those do."""
    kinds = frame["geometry"].map(_text) if "geometry" in frame.columns else None
    wanted = (
        {}
        if kinds is None
        else dict(
            zip(
                ids[kinds.isin(list(_VERTEX_KINDS))],
                kinds[kinds.isin(list(_VERTEX_KINDS))],
                strict=True,
            )
        )
    )
    if vertices.empty:
        if wanted:
            named = ", ".join(sorted(wanted))
            msg = f"The path or polygon id(s) {named} state no vertices."
            raise ParameterError(msg)
        return pd.DataFrame(columns=[*_VERTEX_COLUMNS, *dims])
    missing = [x for x in _VERTEX_COLUMNS if x not in vertices.columns]
    if missing:
        msg = f"The vertices state no {', '.join(missing)} column."
        raise ParameterError(msg)
    vertex_dims = [str(x) for x in vertices.columns if x not in _VERTEX_COLUMNS]
    if unknown := sorted(set(vertex_dims) - set(dims)):
        msg = (
            f"The vertices state the column(s) {', '.join(unknown)}, which name "
            f"no declared dimension. The set declares {list(dims)}."
        )
        raise ParameterError(msg)
    if not vertex_dims:
        msg = "The vertices state no dimension column, so they place nothing."
        raise ParameterError(msg)
    if vertices["seq"].isnull().any():
        rows = ", ".join(str(x) for x in vertices.index[vertices["seq"].isnull()][:5])
        msg = f"Vertex row(s) {rows} state no seq, so they have no place in the order."
        raise ParameterError(msg)
    blank = vertices[vertex_dims].isnull().any(axis=1)
    if blank.any():
        rows = ", ".join(str(x) for x in vertices.index[blank][:5])
        msg = (
            f"Vertex row(s) {rows} leave a dimension empty; a vertex states "
            "every dimension its frame names."
        )
        raise ParameterError(msg)
    vertex_ids = vertices["id"].map(_text)
    if stray := sorted(set(vertex_ids) - set(wanted)):
        msg = (
            f"The vertex id(s) {', '.join(stray)} name no path or polygon in this set."
        )
        raise ParameterError(msg)
    for name, group in vertices.groupby(vertex_ids.values, sort=True):
        if group["seq"].duplicated().any():
            msg = f"The vertices of {str(name)!r} repeat a seq, so they have no order."
            raise ParameterError(msg)
        least = _VERTEX_KINDS[wanted[name]]
        if len(group) < least:
            msg = (
                f"The {wanted[name]} {str(name)!r} states {len(group)} vertices; "
                f"it is a shape with at least {least}."
            )
            raise ParameterError(msg)
    if bare := sorted(set(wanted) - set(vertex_ids)):
        msg = f"The path or polygon id(s) {', '.join(bare)} state no vertices."
        raise ParameterError(msg)
    return vertices.sort_values(["id", "seq"], kind="stable").reset_index(drop=True)


def _fill_vertex_bounds(frame, vertices, spellings) -> pd.DataFrame:
    """
    Give every path and polygon row the bounding region of its vertices.

    The vertices are the shape and the box is a cache of them, so a row
    which states one that disagrees is refused rather than quietly
    corrected. Rows stating none get theirs filled in, which is what lets
    a table operation treat every geometry alike.
    """
    if vertices.empty:
        return frame
    out = frame.copy()
    ids = out["id"].map(_text)
    for dim, spelling in spellings.items():
        if dim not in vertices.columns:
            continue
        bounds = vertices.groupby(vertices["id"].map(_text))[dim].agg(["min", "max"])
        if spelling.point is not None:
            _fill_point_bounds(out, ids, bounds, dim, spelling.point)
            continue
        for column, side in ((spelling.start, "min"), (spelling.end, "max")):
            if column is None:
                out[f"{dim}{_START if side == 'min' else _END}"] = pd.Series(
                    ids.map(bounds[side]), index=out.index
                )
                continue
            derived = ids.map(bounds[side])
            stated = out[column]
            clash = derived.notna() & stated.notna() & (derived != stated)
            if clash.any():
                rows = ", ".join(str(x) for x in out.index[clash][:5])
                msg = (
                    f"Row(s) {rows} state a {column} which disagrees with their "
                    "vertices; a bounding region is derived from them."
                )
                raise ParameterError(msg)
            out[column] = stated.where(derived.isna(), derived)
    return out


def _fill_point_bounds(out, ids, bounds, dim: str, column: str) -> None:
    """
    Give a point-spelled dimension the value its vertices sit at.

    A set spelling a dimension as a bare column says every annotation is a
    point along it, so a path which spans that dimension contradicts the
    set rather than merely bounding itself oddly.
    """
    spanning = bounds["min"] != bounds["max"]
    if spanning.any():
        named = ", ".join(str(x) for x in bounds.index[spanning][:5])
        msg = (
            f"The path or polygon id(s) {named} span {dim}, which this set "
            f"spells as a point ({column}); a spanning geometry needs "
            f"{dim}{_START}/{dim}{_END}."
        )
        raise ParameterError(msg)
    derived = ids.map(bounds["min"])
    stated = out[column]
    clash = derived.notna() & stated.notna() & (derived != stated)
    if clash.any():
        rows = ", ".join(str(x) for x in out.index[clash][:5])
        msg = (
            f"Row(s) {rows} state a {column} which disagrees with their "
            "vertices; a bounding region is derived from them."
        )
        raise ParameterError(msg)
    out[column] = stated.where(derived.isna(), derived)


def _read_bounds(row, spellings) -> dict[str, tuple[Any, Any]]:
    """Return the half-open bounds a row states, per dimension."""
    out = {}
    for dim, spelling in spellings.items():
        if spelling.point is not None:
            value = row.get(spelling.point)
            if _stated(value):
                out[dim] = (_scalar(value), _scalar(value))
            continue
        start = row.get(spelling.start) if spelling.start else None
        end = row.get(spelling.end) if spelling.end else None
        if _stated(start) and _stated(end):
            out[dim] = (_scalar(start), _scalar(end))
    return out


def _row_vertices(vertices, identity, dims) -> dict[str, tuple]:
    """Return one annotation's vertices, ordered by seq, keyed by dimension."""
    rows = vertices[vertices["id"].map(_text) == _text(identity)]
    stated = [x for x in dims if x in rows.columns]
    return {dim: tuple(_scalar(x) for x in rows[dim]) for dim in stated}


def _read_extra(row, dims, spellings) -> dict[str, Any]:
    """Return the columns a row states which the set does not model."""
    known = set(RESERVED_COLUMNS)
    for spelling in spellings.values():
        known |= {x for x in (spelling.point, spelling.start, spelling.end) if x}
    known |= set(dims)
    return {
        str(k): _freeze(v) for k, v in row.items() if str(k) not in known and _stated(v)
    }


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


def _check_basis(frame, dims) -> None:
    """Read every stated curve at load, so a bad one is not found later."""
    if "basis" not in frame.columns:
        return
    for value in frame["basis"]:
        _read_basis(value, dims)


def _read_basis(value, dims):
    """
    Return the curve a cell states, which may be the model or its document.

    A set carries a basis so the curve survives a round trip through a
    frame; the vertices remain the shape, and this is what drew them.
    """
    if not _stated(value):
        return None
    if isinstance(value, AnnotationBasis):
        basis = value
    else:
        try:
            basis = _BASIS_ADAPTER.validate_python(value)
        except ValidationError as error:
            msg = f"Could not read {value!r} as a curve: {error}."
            raise ParameterError(msg) from error
    if foreign := sorted(set(basis.dims) - set(dims)):
        msg = (
            f"The curve names the dimension(s) {', '.join(foreign)}, which the "
            f"set does not declare. It declares {list(dims)}."
        )
        raise ParameterError(msg)
    return basis


def _read_tags(value) -> tuple[str, ...]:
    """Return the tags a cell states, which may be text or a sequence."""
    if not _stated(value):
        return ()
    if isinstance(value, str):
        return tuple(x.strip() for x in value.split(",") if x.strip())
    if isinstance(value, Iterable):
        return tuple(str(x) for x in value)
    # A lone value is one tag; a cell holding a number is not a mistake,
    # it is a label which happens to be spelled as one.
    return (str(value),)


def _scalar(value):
    """
    Return a cell's value as the plainest thing which still says it.

    Numpy numbers become python ones, so a bound reads as the number it
    is; datetimes and timedeltas keep their numpy type, which is what
    makes an endpoint on a time dimension a time rather than an integer.
    """
    if isinstance(value, datetime.datetime | datetime.date):
        return to_datetime64(value)
    if isinstance(value, pd.Timedelta):
        return value.to_timedelta64()
    if isinstance(value, np.generic) and value.dtype.kind not in "mM":
        return value.item()
    return value


def _text(value) -> str:
    """Return a cell as text, an unstated one as the empty string."""
    return str(value) if _stated(value) else ""


def _stated(value) -> bool:
    """Whether a cell states anything at all."""
    if value is None:
        return False
    try:
        return not bool(pd.isnull(value))
    except (TypeError, ValueError):
        # An array-like cell; it is stated by being there.
        return True
