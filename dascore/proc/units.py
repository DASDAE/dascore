"""Processing functions dealing with units and unit conversions."""
from __future__ import annotations

import dascore as dc
from dascore.constants import PatchType
from dascore.units import Quantity, Unit, get_factor_and_unit
from dascore.units import convert_units as u_covert_units
from dascore.utils.patch import patch_function


def _update_attrs_coord_units(patch: dc.Patch, data_units, coords):
    """Update attributes with new units."""
    attrs = patch.attrs
    data_units = data_units or attrs.get("data_units")
    # set data units
    attrs = attrs.update(data_units=data_units, coords=coords.to_summary_dict())
    return attrs


@patch_function()
def set_units(
    patch: PatchType, data_units: str | Quantity | Unit | None = None, **kwargs
) -> PatchType:
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
    >>> # set the data units
    >>> patch_with_units = patch.set_units("km/ms")
    >>> # set the units of the distance coordinate
    >>> patch_feet = patch.set_units(distance='feet')
    """
    new_coords = patch.coords.set_units(**kwargs)
    new_attrs = _update_attrs_coord_units(patch, data_units, new_coords)
    return patch.new(attrs=new_attrs, coords=new_coords)


@patch_function()
def convert_units(
    patch: PatchType, data_units: str | Quantity | Unit | None = None, **kwargs
) -> PatchType:
    """
    Convert the patch data or coordinate units.

    Preform proper conversions from one unit to another, changing patch
    data and dimension labels to the new unit specified. If the data or
    coordinates whose units are to be converted are not set, the new units
    will simply be set without performing any conversions.

    See also [set_units](`dascore.Patch.set_units`) and
    [simplify_units](`dascore.Patch.simplify_units`)

    Parameters
    ----------
    data_units
        If provided, new units for the patch data.
    **kwargs
        Used to specify the new units of the coordinates.

    Raises
    ------
    [UnitError](`dascore.exceptions.UnitError`) if any of the new units
    are not compatible with the old units.
    """
    # convert data
    if data_units is not None:
        current_units = patch.attrs.data_units
        data = u_covert_units(patch.data, data_units, current_units)
        attrs = patch.attrs.update(data_units=data_units, coords={})
    else:
        data = patch.data
        attrs = None
    # then update coords and attrs
    coords = patch.coords.convert_units(**kwargs)
    return patch.new(data=data, coords=coords, attrs=attrs)


@patch_function()
def simplify_units(
    patch: PatchType,
) -> PatchType:
    """
    Simplify the units contained by the patch to base metric units.

    All data and coordinate units will be converted to their
    base units and corresponding data/labels multiplied by a conversion factor.
    """
    # get data and data units
    attrs = patch.attrs
    d_factor, d_units = get_factor_and_unit(attrs.get("data_units"), simplify=True)
    data = patch.data * d_factor if d_factor != 1 else patch.data
    # update coords and coord units in attrs
    coords = patch.coords.simplify_units()
    for name, coord in coords.coord_map.items():
        label = f"{name}_units"
        if label in attrs:
            attrs[label] = coord.units
    new_attrs = attrs.update(data_units=d_units, coords=coords.to_summary_dict())
    return patch.new(data=data, coords=coords, attrs=new_attrs, dims=patch.dims)
