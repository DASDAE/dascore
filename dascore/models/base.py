"""Base classes for DASCore's pydantic models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from rich.text import Text

from dascore.compat import is_array_like
from dascore.exceptions import InvalidInventoryError
from dascore.models.registry import (
    TAG_FIELD,
    _lookup,
    check_tag_matches,
    get_model_tag,
    register_model,
)
from dascore.models.types import DateTime64, FrozenDictType
from dascore.utils.display import (
    RichRepr,
    Section,
    child_sections,
    model_to_line,
    render_text,
    section_indent,
)
from dascore.utils.misc import _all_null, all_close
from dascore.utils.time import to_datetime64


def sensible_model_equals(self: BaseModel | Mapping, other: object) -> bool:
    """Custom equality to not compare private attrs and handle numpy arrays."""
    d1 = self.model_dump() if isinstance(self, BaseModel) else self
    if isinstance(other, BaseModel):
        d2 = other.model_dump()
    elif isinstance(other, Mapping):
        d2 = other
    else:  # nothing else can carry the same fields
        return NotImplemented
    if not set(d1) == set(d2):  # different keys, not equal
        return False
    for name in set(x for x in d1 if not x.startswith("_")):
        # skip any private attributes.
        if not values_equal(d1[name], d2[name]):
            return False
    return True


def values_equal(val1, val2) -> bool:
    """Recursively compare dumped values; nulls are equal only to nulls."""
    if is_array_like(val1) or is_array_like(val2):
        arr1, arr2 = np.asarray(val1), np.asarray(val2)
        if arr1.shape != arr2.shape:
            return False
        if not np.array_equal(pd.isnull(arr1), pd.isnull(arr2)):
            return False
        return bool(all_close(arr1, arr2))
    if isinstance(val1, Mapping) and isinstance(val2, Mapping):
        if set(val1) != set(val2):
            return False
        return all(values_equal(val1[key], val2[key]) for key in val1)
    if isinstance(val1, list | tuple) and isinstance(val2, list | tuple):
        if len(val1) != len(val2):
            return False
        return all(values_equal(v1, v2) for v1, v2 in zip(val1, val2))
    return bool(val1 == val2 or (_all_null(val1) and _all_null(val2)))


def _hash_key(value):
    """Map a value onto one that hashes the way values_equal compares."""
    if value is None or isinstance(value, str | int):
        return value
    # Nulls count as equal to one another above, but every nan and NaT is a
    # fresh object and both hash by identity, so they collapse to one key.
    if isinstance(value, float):
        return None if value != value else value
    if isinstance(value, np.datetime64 | np.timedelta64):
        return None if np.isnat(value) else value
    # Mappings are compared without regard to order.
    if isinstance(value, Mapping):
        return frozenset((k, _hash_key(v)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(_hash_key(v) for v in value)
    return value


def sensible_model_hash(self: BaseModel) -> int:
    """Hash a model on its fields, agreeing with sensible_model_equals."""
    # Keyed by name and unordered, because equality compares the field names
    # it finds rather than the order they were declared in.
    fields = type(self).model_fields
    return hash(frozenset((x, _hash_key(getattr(self, x))) for x in fields))


class DascoreBaseModel(BaseModel):
    """A base model with sensible configurations."""

    _cache = {}

    model_config = ConfigDict(
        extra="ignore",  # TODO: change to raise, then let subclass overwrite
        validate_assignment=True,
        ignored_types=(cached_property,),
        frozen=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def __init_subclass__(cls, **kwargs):
        """Register every model so a document can name it."""
        super().__init_subclass__(**kwargs)
        register_model(cls)

    @model_serializer(mode="wrap")
    def _write_object_type(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        """
        Name this class in the document, in json-mode dumps only.

        A python-mode dump is not a document: it is what equality compares,
        what ``new`` reconstructs from and what the index ingests, none of
        which want a key that is not a field.

        Gated inside the serializer rather than with ``when_used="json"``,
        which reads better and silently breaks ``include``/``exclude``:
        pydantic skips the whole wrapper in python mode, and with it the
        field filtering the handler would have applied.
        """
        out = handler(self)
        if info.mode != "json" or TAG_FIELD in out:  # a union member's own
            return out
        # Its field is defaulted, so exclude_defaults drops it and leaves a
        # document the union cannot dispatch on. Put back what the model
        # says rather than what the registry calls the class: a subclass
        # out of tree has a different tag, and the Literal would refuse it.
        declared = getattr(self, TAG_FIELD, None)
        if (tag := declared or get_model_tag(type(self))) is not None:
            out[TAG_FIELD] = tag
        return out

    @model_validator(mode="before")
    @classmethod
    def _read_object_type(cls, data: Any) -> Any:
        """
        Consume a document's class tag, refusing one which names another class.

        The tag is never required: a document dispatches on it before it
        gets here, and a hand-written object or a nested one may simply not
        state it. What it may not do is disagree.

        A value which names no known class is left alone rather than
        consumed. ``PatchAttrs`` keeps extra fields, so this key may be a
        reader's own metadata, and eating it would lose that silently; a
        tag DASCore wrote always resolves.
        """
        if not isinstance(data, Mapping) or TAG_FIELD not in data:
            return data
        # A union member declares the tag as a real field and validates it
        # against its own Literal, which is stricter than this.
        if TAG_FIELD in cls.model_fields:
            return data
        tag = data[TAG_FIELD]
        if (declared := _lookup(tag)) is None:
            return data
        check_tag_matches(cls, declared, tag)
        out = dict(data)
        out.pop(TAG_FIELD)
        return out

    def new(self, **kwargs) -> Self:
        """Create new instance with some attributed updated."""
        out = self.model_dump(exclude_unset=True)
        out.update(kwargs)
        return self.__class__(**out)

    @classmethod
    def get_summary_df(cls):
        """Get dataframe of attributes and descriptions for display."""
        fields = cls.model_fields
        names_desc = {
            i: v.description
            for i, v in fields.items()
            if getattr(v, "description", False)
        }
        out = pd.Series(names_desc).to_frame(name="description")
        out.index.name = "attribute"
        return out

    __eq__ = sensible_model_equals
    # Defined together: pydantic would otherwise derive a hash straight from
    # the field values, which disagrees with how __eq__ treats nulls.
    __hash__ = sensible_model_hash


class InventoryModel(RichRepr, DascoreBaseModel):
    """
    Base class for immutable DASDAE inventory objects.

    Every inventory object carries two uniform attachment points for
    information the model does not otherwise represent: ``description``
    (free prose for humans, matching StationXML's Description element)
    and ``extra_fields`` (typed key-values, e.g. for round-tripping
    unmodeled metadata from external formats).

    Every field is immutable: collections are tuples and mappings are
    frozen, so instances are safe to hold by reference. They hash on their
    field values whenever those values are themselves hashable.
    """

    description: str = Field(default="", description="Free-text description.")
    extra_fields: FrozenDictType[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Extra metadata not represented by standardized fields.",
    )

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        validate_default=True,
        arbitrary_types_allowed=True,
    )

    def new(self, **kwargs) -> Self:
        """
        Create a new instance with some attributes updated.

        Dumps all fields (not just the set ones) so union discriminators and
        validator-normalized fields survive reconstruction.
        """
        out = self.model_dump()
        out.update(kwargs)
        return self.__class__(**out)

    def _repr_line(self) -> Text:
        """
        The one line which names this object and what it states.

        What a container puts on the line it gives this object, and the
        whole of the repr for one which holds nothing.
        """
        return model_to_line(self)

    def _repr_children(self) -> tuple[InventoryModel, ...]:
        """
        The objects this one holds, each of which prints itself.

        Empty by default: a model whose children are worth a count in its
        own line rather than a line each says nothing here, which is why
        a station shows ``channels: 3`` and not three channels.
        """
        return ()

    def _repr_section(self, depth: int = 0) -> Section:
        """
        The block a repr draws this object in.

        ``depth`` is how far into a containment tree it sits, which a
        terminal shows by indenting and a panel shows by nesting. The
        indentation is stated in the title, for the same reason a section
        body carries the newline which separated it from one: rendering
        has then only to concatenate.
        """
        title = section_indent(depth) + self._repr_line()
        return Section(title, child_sections(self._repr_children(), depth + 1), depth)

    def __rich__(self) -> Text:
        """The line naming this object, then whatever it holds."""
        return render_text(self._repr_section())


class TimeRangedModel(InventoryModel):
    """Base class for inventory objects with time-validity epochs.

    Validity intervals are half-open, ``[time_min, time_max)``; an unset
    (NaT) end time means the epoch is ongoing. All times are UTC.
    """

    time_min: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description="Start time for which this metadata item is valid (UTC).",
    )
    time_max: DateTime64 = Field(
        default=np.datetime64("NaT", "ns"),
        description=(
            "End time for which this metadata item is valid (UTC); NaT while ongoing."
        ),
    )

    @model_validator(mode="after")
    def _check_time_order(self):
        """A set end time must follow the start time."""
        start, end = self.time_min, self.time_max
        if not pd.isnull(start) and not pd.isnull(end) and end <= start:
            msg = f"time_max {end} must be after time_min {start}."
            raise InvalidInventoryError(msg)
        return self

    def is_effective_at(self, time) -> bool:
        """Return True if this epoch is valid at the supplied time (half-open)."""
        time = to_datetime64(time)
        if pd.isnull(time):
            return True
        start = self.time_min
        end = self.time_max
        after_start = pd.isnull(start) or start <= time
        before_end = pd.isnull(end) or time < end
        return bool(after_start and before_end)

    def overlaps(self, other: TimeRangedModel) -> bool:
        """
        Return True if two half-open validity intervals overlap.

        Unset (NaT) starts are unbounded past; unset ends are ongoing.
        """
        s1, e1, s2, e2 = (
            self.time_min,
            self.time_max,
            other.time_min,
            other.time_max,
        )
        first_starts_before = pd.isnull(e2) or pd.isnull(s1) or s1 < e2
        second_starts_before = pd.isnull(e1) or pd.isnull(s2) or s2 < e1
        return bool(first_starts_before and second_starts_before)
