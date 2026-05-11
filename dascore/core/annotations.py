"""
Annotations for DAS data.
"""
from typing import Self

from pydantic import Field, model_validator, field_validator

from dascore.utils.models import CommentableModel
from dascore.utils.misc import sanitize_range_param


class OpticalPathAnnotation(CommentableModel):
    """Named or categorized interval on an optical path."""

    distance: tuple[float | None, float | None] = Field(
        default=(None, None),
        description="Optical distance interval for this annotation (start, stop).",
    )
    label: str = Field(
        default="", description="Human-readable label for this path annotation."
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