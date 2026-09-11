"""Processing functions dealing with units and unit conversions."""

from __future__ import annotations

from typing import Any

from pydantic import ConfigDict

import dascore as dc
from dascore.core.processor import PatchProcessor
from dascore.units import conversion_factors, get_factor_and_unit, units_match


def _replace_data_units(
    attrs: dc.PatchAttrs, data_units, preserve_existing_data_units: bool = False
):
    """Return attrs with updated data units; coordinate units live on coords."""
    out = attrs.model_dump(exclude_unset=True)
    if data_units not in (None, ""):
        out["data_units"] = data_units
    elif preserve_existing_data_units:
        out["data_units"] = attrs.data_units
    else:
        out["data_units"] = None
    return dc.PatchAttrs.from_dict(out)


class SetUnits(PatchProcessor):
    """
    Set the units of a patch's data or coordinates.

    Parameters
    ----------
    patch
        The input patch.
    data_units
        New units for the patch data. Accepts both unit and quantity strings.
    **kwargs
        Used to specify new units for any of the patch's coordinates.

    Warning
    -------
    Old units will be deleted without performing conversions. To *convert*
    units see [convert_units](`dascore.Patch.convert_units`).

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # set the data units
    >>> patch_with_units = patch.set_units("km/ms")
    >>>
    >>> # set the units of the distance coordinate
    >>> patch_feet = patch.set_units(distance='feet')
    >>>
    >>> # remove data units
    >>> patch_removed_units = patch_with_units.set_units(None)
    """

    data_units: Any = None

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    def derive(self, patch):
        """Return the coordinates and data units with the units set."""
        new_coords = patch.coords.set_units(**(self.model_extra or {}))
        # data_units=None means "clear them", which units_match reports as a change.
        if new_coords is patch.coords and units_match(
            patch.attrs.data_units, self.data_units
        ):
            return patch
        new_attrs = _replace_data_units(patch.attrs, self.data_units)
        return patch.new(attrs=new_attrs, coords=new_coords)


set_units = SetUnits.patch_function


class ConvertUnits(PatchProcessor):
    """
    Convert the patch data or coordinate units.

    Perform proper conversions from one unit to another, changing patch
    data and dimension labels to the new unit specified. If the data or
    coordinates whose units are to be converted are not set, the new units
    will simply be set without performing any conversions.

    See also [set_units](`dascore.Patch.set_units`) and
    [simplify_units](`dascore.Patch.simplify_units`)

    Parameters
    ----------
    patch
        The patch whose units should be converted.
    data_units
        If provided, new units for the patch data.
    **kwargs
        Used to specify the new units of the coordinates.

    Raises
    ------
    [UnitError](`dascore.exceptions.UnitError`) if any of the new units
    are not compatible with the old units.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Set initial units
    >>> patch_with_units = patch.set_units("m/s", distance="m", time="s")
    >>>
    >>> # Convert data units from m/s to km/s
    >>> converted_data = patch_with_units.convert_units(data_units="km/s")
    >>>
    >>> # Convert coordinate units
    >>> converted_coords = patch_with_units.convert_units(distance="km")
    """

    data_units: Any = None

    model_config = ConfigDict(extra="allow", frozen=True, arbitrary_types_allowed=True)

    def derive(self, patch):
        """Return the coordinates and data units converted."""
        coords = patch.coords.convert_units(**(self.model_extra or {}))
        data_units = self.data_units
        # Nothing to convert.
        if coords is patch.coords and (
            data_units is None or units_match(patch.attrs.data_units, data_units)
        ):
            return patch
        attrs = _replace_data_units(
            patch.attrs, data_units, preserve_existing_data_units=True
        )
        return patch.new(coords=coords, attrs=attrs)

    def plan(self, patch, out):
        """Return the affine factors which convert the data, if they change."""
        if self.data_units is None:
            return {}
        factors = conversion_factors(patch.attrs.data_units, self.data_units)
        if factors is None:
            return {}
        mult1, add, mult2 = factors
        return {"mult1": mult1, "add": add, "mult2": mult2}

    def kernel(self, data, *, mult1=None, add=None, mult2=None):
        """Return the data converted; the data if their units stand."""
        if mult1 is None:
            return data
        return (data * mult1 + add) * mult2


convert_units = ConvertUnits.patch_function


class SimplifyUnits(PatchProcessor):
    """
    Simplify the units contained by the patch to base metric units.

    All data and coordinate units will be converted to their
    base units and corresponding data/labels multiplied by a conversion factor.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>>
    >>> # Set complex units
    >>> complex_units = patch.set_units("km/h", distance="km", time="h")
    >>>
    >>> # Simplify to base units (m/s, m, s)
    >>> simplified = complex_units.simplify_units()
    """

    def derive(self, patch):
        """Return the coordinates and data units in base metric units."""
        attrs = patch.attrs
        _, d_units = get_factor_and_unit(attrs.get("data_units"), simplify=True)
        coords = patch.coords.simplify_units()
        if coords is patch.coords and units_match(attrs.get("data_units"), d_units):
            return patch
        new_attrs = _replace_data_units(attrs, d_units)
        return patch.new(coords=coords, attrs=new_attrs, dims=patch.dims)

    def plan(self, patch, out):
        """Return the factor which scales the data to base units."""
        factor, _ = get_factor_and_unit(patch.attrs.get("data_units"), simplify=True)
        return {"factor": factor}

    def kernel(self, data, *, factor):
        """Return the data scaled; the data themselves for a factor of one."""
        return data * factor if factor != 1 else data


simplify_units = SimplifyUnits.patch_function
