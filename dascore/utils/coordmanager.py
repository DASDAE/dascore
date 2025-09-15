"""Utilities for working with coordinate managers."""

from __future__ import annotations

from _operator import and_, or_
from collections.abc import Sequence
from functools import reduce

import numpy as np

import dascore as dc
from dascore.exceptions import CoordMergeError
from dascore.utils.display import get_nice_text
from dascore.utils.models import ArrayLike


def merge_coord_managers(
    coord_managers: Sequence[dc.CoordManager],
    dim: str,
    snap_tolerance: float | None = None,
    drop_conflicting: bool = False,
) -> dc.CoordManager:
    """
    Merge coordinate managers along a specified dimension.

    Parameters
    ----------
    coord_managers
        A sequence of coord_managers to merge.
    dim
        The dimension along which to merge.
    snap_tolerance
        The tolerance for snapping CoordRanges together. E.G, allows
        coord ranges that have snap_tolerances differences from their
        start/end to be joined together. If they don't meet this requirement
        an [CoordMergeError](`dascore.exceptions.CoordMergeError`) is raised.
        If None, no checks are performed.
    drop_conflicting
        If True, drop conflicting (non-dimensional) coordinates, otherwise
        raise an exception if they occur.
    """

    def _get_dims(managers):
        """Ensure all managers have same dimensions."""
        dims = {x.dims for x in managers}
        if len(dims) != 1:
            msg = (
                "Can't merge coord managers, they don't all have the "
                "same dimensions!"
            )
            raise CoordMergeError(msg)
        return managers[0].dims

    def _drop_unshared_coordinates(managers):
        """Any coordinates not shared between managers should be dropped."""
        # gets [{(coord, dims, ...), (coord, dims, ...)}, ...] to ensure
        # both the coords name and their dimensions are common between managers
        coord_sets = [set(x._get_coord_dims_tuple()) for x in managers]
        common_coords = reduce(and_, coord_sets)
        all_coords = reduce(or_, coord_sets)
        if not (drop_coords := all_coords - common_coords):
            return managers
        coords_to_drop = [x[0] for x in drop_coords]
        return [x.drop_coords(*coords_to_drop)[0] for x in managers]

    def _get_non_merge_coords(managers, non_merger_names):
        """Ensure all non-merge coords are equal."""
        out = {}
        for coord_name in non_merger_names:
            first = managers[0].coord_map[coord_name]
            if all([first == x.coord_map[coord_name] for x in managers]):
                dims = managers[0].dim_map[coord_name]
                out[coord_name] = (dims, first)
                continue
            # Simply skip conflicting
            elif drop_conflicting:
                # These are non dimensional coords
                if not any(coord_name in x.dims for x in managers):
                    continue
            msg = (
                f"Non merging coordinates {coord_name} are not equal. "
                "Coordinate managers cannot be merged. Try using "
                "spool.chunk with conflict='drop'."
            )
            raise CoordMergeError(msg)
        return out

    def _snap_coords(coord_list):
        """Snap coordinates together."""
        if snap_tolerance is None:
            return coord_list  # skip snapping if no snap tolerance.
        for ind in range(1, len(coord_list)):
            c_coord = coord_list[ind - 1]
            n_coord = coord_list[ind]
            tolerance = snap_tolerance * c_coord.step
            assumed_start = c_coord.max() + c_coord.step
            diff = np.abs(assumed_start - n_coord.min())
            # snap is close enough, update coord.
            if diff > 0 and diff <= tolerance:
                coord_list[ind] = n_coord.update_limits(min=assumed_start)
            # snap is too far off, bail out.
            elif diff > tolerance:
                msg = (
                    f"Cannot merge. Snap tolerance: {get_nice_text(tolerance)}"
                    f" not met"
                )
                raise CoordMergeError(msg)
        return coord_list

    def _get_merged_coords(managers, coords_to_merge):
        """Get the merged coordinates."""
        out = {}
        for coord_name in coords_to_merge:
            merge_coords = [x.coord_map[dim] for x in managers]
            axis = managers[0].dim_map[coord_name].index(dim)
            if len(units := {x.units for x in merge_coords}) != 1:
                # TODO: we might try to convert all the units to a common
                # unit in the future.
                msg = (
                    f"Cannot merge coordinates {coord_name}, they dont all "
                    f"share the same units. Units found are: {set(units)}"
                )
                raise CoordMergeError(msg)
            snap_coords = _snap_coords(merge_coords)
            datas = [x.data for x in snap_coords]
            dims = managers[0].dim_map[dim]
            new_data = np.concatenate(datas, axis=axis)
            out[coord_name] = (dims, new_data)
        return out

    def _get_new_coords(managers) -> dict[str, tuple[tuple[str, ...], ArrayLike]]:
        """Merge relevant coordinates together."""
        # build up merged coords.
        coords_to_merge = managers[0].dim_to_coord_map[dim]
        coords_not_to_merge = set(managers[0].coord_map) - set(coords_to_merge)
        # non-merging coordinates should be identical.
        coords_dict = _get_non_merge_coords(managers, coords_not_to_merge)
        # merge coordinates
        coords_dict.update(_get_merged_coords(managers, coords_to_merge))
        return coords_dict

    dims = _get_dims(coord_managers)
    coord_managers = _drop_unshared_coordinates(coord_managers)
    sort_managers = sorted(coord_managers, key=lambda x: x.coord_map[dim].min())
    merged_coords = _get_new_coords(sort_managers)
    return dc.get_coord_manager(merged_coords, dims=dims)
