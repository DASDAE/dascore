"""Pydantic schemas used by DASCore."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any

from pydantic import ConfigDict, Field, PlainValidator, computed_field, model_validator
from typing_extensions import Self

from dascore.constants import max_lens
from dascore.utils.attrs import _raise_if_coord_attr_updates
from dascore.utils.misc import (
    to_str,
)
from dascore.utils.models import DascoreBaseModel, UnitQuantity

str_validator = PlainValidator(to_str)
_DATA_SOURCE_ID_FIELDS = ("network", "fiber_array", "location", "acquisition")


def _parse_data_source_id(value: str) -> tuple[str, str, str, str]:
    """Parse and validate a dot-delimited data source id."""
    parts = tuple(value.split("."))
    if len(parts) != 4:
        msg = (
            "data_source_id must have exactly four dot-delimited components: "
            "network.fiber_array.location.acquisition."
        )
        raise ValueError(msg)
    network, fiber_array, location, acquisition = parts
    if not network or not fiber_array or not acquisition:
        msg = (
            "data_source_id network, fiber_array, and acquisition components "
            "must be non-empty."
        )
        raise ValueError(msg)
    return network, fiber_array, location, acquisition


def _validate_data_source_id_component(name: str, value) -> str:
    """Return a string component after validating delimiter safety."""
    value = to_str(value)
    if "." in value:
        msg = f"PatchAttrs {name!r} cannot contain '.'."
        raise ValueError(msg)
    return value


class PatchAttrs(DascoreBaseModel):
    """
    The expected attributes for a Patch.

    `PatchAttrs` stores non-structural metadata. Nested coordinate payloads in
    `coords` are rejected, while flat coord-like keys are treated like any
    other extra attrs. `dims` is ignored during normalization.

    Patch attrs are the metadata snapshot carried by one concrete patch. They
    are not a replacement for inventory records. Acquisition-derived defaults
    such as data type, data category, acquisition units, sample rate, and gauge
    length belong on [`Acquisition`](`dascore.core.inventory.Acquisition`) and
    can be copied onto a patch when needed. Patch attrs keep patch-local values
    and overrides such as `data_units`, `tag`, processing `history`, and the
    `data_source_id` used to resolve inventory context.

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

    data_units: UnitQuantity | None = Field(
        default=None, description="The units of the data measurements"
    )
    tag: str = Field(
        default="", max_length=max_lens["tag"], description="A custom string field."
    )
    network: str = Field(
        default="", max_length=max_lens["network"], description="A network code."
    )
    fiber_array: Annotated[str, str_validator] = Field(
        default="", description="A fiber array code."
    )
    location: Annotated[str, str_validator] = Field(
        default="", description="A location code within a fiber array."
    )
    acquisition: Annotated[str, str_validator] = Field(
        default="", description="An acquisition code."
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
        data_source_id = data.pop("data_source_id", "")
        if data_source_id:
            parsed = _parse_data_source_id(to_str(data_source_id))
            for name, value in zip(_DATA_SOURCE_ID_FIELDS, parsed, strict=True):
                old_value = data.get(name, "")
                if old_value and old_value != value:
                    msg = (
                        f"PatchAttrs {name!r} conflicts with data_source_id "
                        f"{data_source_id!r}."
                    )
                    raise ValueError(msg)
                data[name] = value
        for name in _DATA_SOURCE_ID_FIELDS:
            if name in data:
                data[name] = _validate_data_source_id_component(name, data[name])
        return data

    @computed_field
    @property
    def data_source_id(self) -> str:
        """Return network.fiber_array.location.acquisition when available."""
        if not (self.network and self.fiber_array and self.acquisition):
            return ""
        return ".".join(
            (self.network, self.fiber_array, self.location, self.acquisition)
        )

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
        out = self.model_dump(exclude={"data_source_id"}, exclude_unset=True)
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
