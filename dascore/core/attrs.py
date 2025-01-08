"""Pydantic schemas used by DASCore."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import ConfigDict, Field, PlainValidator
from typing_extensions import Self

from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    max_lens,
)
from dascore.core.coords import CoordSummary
from dascore.utils.attrs import separate_coord_info
from dascore.utils.misc import (
    to_str,
)
from dascore.utils.models import (
    DascoreBaseModel,
    StrTupleStrSerialized,
    UnitQuantity,
)

str_validator = PlainValidator(to_str)
_coord_summary_suffixes = set(CoordSummary.model_fields)
_coord_required = {"min", "max"}


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    The default attributes are:
    ```{python}
    #| echo: false

    import dascore as dc
    from IPython.display import Markdown

    df_str = (
        dc.PatchAttrs.get_summary_df()
        .reset_index()
        .to_markdown(index=False, stralign="center")
    )
    Markdown(df_str)
    ```
    """

    model_config = ConfigDict(
        title="Patch Summary",
        extra="allow",
        frozen=True,
        arbitrary_types_allowed=True,
    )

    data_type: Annotated[Literal[VALID_DATA_TYPES], str_validator] = Field(
        description="Describes the quantity being measured.", default=""
    )
    data_category: Annotated[Literal[VALID_DATA_CATEGORIES], str_validator] = Field(
        description="Describes the type of data.",
        default="",
    )
    data_units: UnitQuantity | None = Field(
        default=None, description="The units of the data measurements"
    )
    instrument_id: str = Field(
        description="A unique id for the instrument which generated the data.",
        default="",
        max_length=max_lens["instrument_id"],
    )
    acquisition_id: str = Field(
        description="A unique identifier linking this data to an experiment.",
        default="",
        max_length=max_lens["experiment_id"],
    )
    tag: str = Field(
        default="", max_length=max_lens["tag"], description="A custom string field."
    )
    station: str = Field(
        default="", max_length=max_lens["station"], description="A station code."
    )
    network: str = Field(
        default="", max_length=max_lens["network"], description="A network code."
    )
    history: StrTupleStrSerialized = Field(
        default_factory=tuple,
        description="A list of processing performed on the patch.",
    )

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __len__(self):
        return len(self.model_dump())

    def get(self, item, default=None):
        """dict-like get method."""
        try:
            return self[item]
        except (AttributeError, KeyError):
            return default

    def items(self):
        """Yield (attribute, values) just like dict.items()."""
        yield from self.model_dump().items()

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        _, attr_info = separate_coord_info(kwargs)
        out = self.model_dump(exclude_unset=True)
        out.update(attr_info)
        return self.__class__(**out)

    def drop_private(self) -> Self:
        """Drop all private attributes."""
        contents = dict(self)
        out = {i: v for i, v in contents.items() if not i.startswith("_")}
        return self.__class__(**out)

    @classmethod
    def from_dict(cls, obj: Mapping | Self):
        if isinstance(obj, cls):
            return obj
        return cls(**obj)
