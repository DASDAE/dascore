"""Pydantic schemas used by DASCore."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, PlainValidator, model_validator
from typing_extensions import Self

from dascore.constants import (
    VALID_DATA_CATEGORIES,
    VALID_DATA_TYPES,
    max_lens,
)
from dascore.utils.attrs import _raise_if_coord_attr_updates
from dascore.utils.misc import (
    to_str,
)
from dascore.utils.models import DascoreBaseModel, UnitQuantity

str_validator = PlainValidator(to_str)


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    `PatchAttrs` stores non-structural metadata. Nested coordinate payloads in
    `coords` are rejected, while flat coord-like keys are treated like any
    other extra attrs. `dims` is ignored during normalization.

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
    history: str | tuple[str, ...] = Field(
        default_factory=tuple,
        description="A list of processing performed on the patch.",
    )

    @model_validator(mode="before")
    @classmethod
    def reject_coordinate_attributes(cls, data: Any) -> Any:
        """Reject nested coord payloads and ignore structural dims input."""
        if not isinstance(data, Mapping):
            return data
        data = dict(data)
        if "coords" in data and not isinstance(data["coords"], str):
            msg = (
                "PatchAttrs no longer accepts coordinate metadata. " "Received: coords."
            )
            raise ValueError(msg)
        data.pop("dims", None)
        return data

    def __getitem__(self, item):
        return getattr(self, item)

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

    @classmethod
    def from_dict(
        cls,
        attr_map: Mapping | PatchAttrs | None,
    ) -> Self:
        """
        Get a new instance of the PatchAttrs.

        Parameters
        ----------
        attr_map
            Anything convertible to a dict that contains attr info. `dims`
            entries are ignored during normalization.
        """
        if isinstance(attr_map, cls):
            return attr_map
        if attr_map is None:
            out = {}
        elif hasattr(attr_map, "model_dump"):
            out = attr_map.model_dump()
        else:
            out = attr_map
        if isinstance(out, Mapping):
            out = dict(out)
            out.pop("dims", None)
        return cls(**out)

    def update(self, **kwargs) -> Self:
        """Update an attribute in the model, return new model."""
        _raise_if_coord_attr_updates(kwargs)
        out = self.model_dump(exclude_unset=True)
        out.update(kwargs)
        return self.from_dict(out)

    def drop(self, *args):
        """Drop specific keys if they exist."""
        contents = dict(self)
        ok_to_keep = set(contents) - set(args)
        out = {i: v for i, v in contents.items() if i in ok_to_keep}
        return self.__class__(**out)

    def drop_private(self) -> Self:
        """Drop all private attributes."""
        contents = dict(self)
        out = {i: v for i, v in contents.items() if not i.startswith("_")}
        return self.__class__(**out)

    def flat_dump(self, exclude=None) -> dict:
        """Dump attrs to a flat dict."""
        return self.model_dump(exclude=exclude)
