"""
Utilities for working with coordinates on Patches.
"""

from dascore.constants import PatchType


def add_coords(patch: PatchType, **kwargs) -> PatchType:
    """
    Add non-dimensional coordinates to a patch.

    Parameters
    ----------
    patch
        The patch to which coordinates will be added.
    **kwargs
        Used to specify the name, dimension, and values of the new
        coordinates.

    Examples
    --------

    >>> import numpy as np
    >>> import dascore as dc
    >>> patch_1 = dc.get_example_patch()
    >>> coords = patch_1.coords
    >>> dist = coords['distance']
    >>> time = coords['time']
    >>> # Add a single coordinate associated with distance dimension
    >>> lat = np.arange(0, len(dist)) * .001 -109.857952
    >>> out_1 = patch_1.add_coords(latitude=('distance', lat))
    >>> # Add multiple coordinates associated with distance dimension
    >>> lon = np.arange(0, len(dist)) *.001 + 41.544654
    >>> out_2 = patch_1.add_coords(
    >>>     latitude=('distance', lat),
    >>>     longitude=('distance', lon),
    >>> )
    >>> # Add multi-dimensional coordinates
    >>> quality = np.ones(len(lat), len(lon))
    >>> out_3 = patch_1.add_coords(
    >>>     quality=(('latitude', 'longitude'), quality)
    >>> )
    """
    coords = {x: patch.coords[x] for x in patch.coords}
    for coord_key, (dimension, value) in kwargs.items():
        coords[coord_key] = (dimension, value)
    return patch.new(coords=coords, dims=patch.dims)
