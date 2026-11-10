"""Utilities for working with coordinate managers."""

from __future__ import annotations

import warnings
from _operator import and_, or_
from collections.abc import Sequence
from functools import reduce

import numpy as np

import dascore as dc
from dascore.constants import CONFLICT
from dascore.exceptions import CoordMergeError
from dascore.utils.display import get_nice_text
from dascore.warnings import DASCoreWarning


def merge_coord_managers(
    coord_managers: Sequence[dc.CoordManager],
    dim: str,
    snap_tolerance: float | None = None,
    conflict: CONFLICT = "raise",
    dim_coord=None,
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
        The tolerance for snapping evenly sampled coords together. E.G, allows
        coord ranges that have snap_tolerances differences from their
        start/end to be joined together. If they don't meet this requirement
        an [CoordMergeError](`dascore.exceptions.CoordMergeError`) is raised.
        If None, no checks are performed.
    conflict
        What to do with a non-dimensional coordinate on another dimension
        which the managers hold differently: "raise" (default) raises a
        [CoordMergeError](`dascore.exceptions.CoordMergeError`),
        "keep_first" keeps the first manager's coordinate and "drop" leaves
        it out. A coordinate some manager lacks is dropped under "raise"
        and otherwise follows the same policy. The loosened policies warn,
        naming the coordinate; private coordinates (names starting with
        "_") are never named.
    dim_coord
        If provided, use this coordinate for `dim` instead of
        concatenating the members' values (which materializes them);
        callers which already built the merged dimension coordinate
        (e.g. via `concat_coords`) pass it here to avoid that cost.
    """

    def _get_dims(managers):
        """Ensure all managers have same dimensions."""
        dims = {x.dims for x in managers}
        if len(dims) != 1:
            msg = "Can't merge coord managers, they don't all have the same dimensions!"
            raise CoordMergeError(msg)
        return managers[0].dims

    # Public non-dimensional coordinates `conflict` decided, by outcome.
    decided: dict[str, list[str]] = {"kept": [], "dropped": []}

    def _resolve_conflict(name, keepable: bool) -> bool:
        """
        Decide a non-dimensional coordinate the managers disagree on.

        Returns True when the first manager's coordinate is kept. A
        coordinate on the merged dimension cannot be kept from one member
        (`keepable` False), so "keep_first" drops it like "drop".
        """
        if conflict == "raise":
            msg = (
                f"Non merging coordinates {name} are not equal. "
                "Coordinate managers cannot be merged. Try using "
                "spool.chunk with conflict='drop'."
            )
            raise CoordMergeError(msg)
        keep = conflict == "keep_first" and keepable
        if not name.startswith("_"):
            decided["kept" if keep else "dropped"].append(name)
        return keep

    def _drop_unshared_coordinates(managers):
        """
        Drop the coordinates not every manager holds on the same dims.

        Returns the managers and the first manager's coordinates which
        "keep_first" keeps although another manager lacks them.
        """
        # gets [{(coord, dims, ...), (coord, dims, ...)}, ...] to ensure
        # both the coords name and their dimensions are common between managers
        coord_sets = [set(x._get_coord_dims_tuple()) for x in managers]
        common_coords = reduce(and_, coord_sets)
        all_coords = reduce(or_, coord_sets)
        if not (drop_coords := all_coords - common_coords):
            return managers, {}
        first = managers[0]
        kept = {}
        # Under "raise" an unshared coordinate is dropped silently, as it
        # always was here; Spool.chunk polices members lacking a coordinate
        # before they reach this function.
        if conflict != "raise":
            for name in sorted({x[0] for x in drop_coords}):
                # A manager which renames a dimension is refused by _get_dims,
                # so an unshared coordinate is never a dimension here.
                keepable = name in first.coord_map and dim not in first.dim_map[name]
                if _resolve_conflict(name, keepable):
                    kept[name] = (first.dim_map[name], first.coord_map[name])
        coords_to_drop = [x[0] for x in drop_coords]
        return [x.drop_coords(*coords_to_drop)[0] for x in managers], kept

    def _get_non_merge_coords(managers, non_merger_names):
        """Ensure all non-merge coords are equal, or resolve them."""
        out = {}
        for coord_name in non_merger_names:
            first = managers[0].coord_map[coord_name]
            dims = managers[0].dim_map[coord_name]
            if all([first == x.coord_map[coord_name] for x in managers]):
                out[coord_name] = (dims, first)
            elif coord_name in managers[0].dims:
                # The policy covers non-dimensional coordinates only; members
                # whose other dimensions differ cannot be merged at all.
                msg = (
                    f"Non merging coordinates {coord_name} are not equal. "
                    f"{coord_name} is a dimension, so no conflict policy "
                    "can merge the coordinate managers."
                )
                raise CoordMergeError(msg)
            elif _resolve_conflict(coord_name, keepable=True):
                out[coord_name] = (dims, first)
        return out

    def _warn_decided():
        """Say which public coordinates the loosened policy decided."""
        parts = [f"{sorted(v)} {k}" for k, v in decided.items() if v]
        if not parts:
            return
        detail = (
            "the first patch's values are kept"
            if conflict == "keep_first"
            else "they are left out of the result"
        )
        msg = (
            f"Merging along {dim!r} patches whose coordinates differ or are "
            f"not all stated: {' and '.join(parts)}. With "
            f"conflict={conflict!r} {detail}."
        )
        warnings.warn(msg, DASCoreWarning, stacklevel=3)

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
            zero = np.asarray([0], dtype=np.asarray(diff).dtype)[0]
            # snap is close enough, update coord.
            if diff > zero and diff <= tolerance:
                coord_list[ind] = n_coord.update_limits(min=assumed_start)
            # snap is too far off, bail out.
            elif diff > tolerance:
                msg = (
                    f"Cannot merge. Snap tolerance: {get_nice_text(tolerance)} not met"
                )
                raise CoordMergeError(msg)
        return coord_list

    def _get_merged_coords(managers, coords_to_merge):
        """Get the merged coordinates."""
        out = {}
        for coord_name in coords_to_merge:
            if dim_coord is not None and coord_name == dim:
                out[coord_name] = (managers[0].dim_map[dim], dim_coord)
                continue
            merge_coords = [x.coord_map[coord_name] for x in managers]
            axis = managers[0].dim_map[coord_name].index(dim)
            if len(units := {x.units for x in merge_coords}) != 1:
                # TODO: we might try to convert all the units to a common
                # unit in the future.
                msg = (
                    f"Cannot merge coordinates {coord_name}, they don't all "
                    f"share the same units. Units found are: {set(units)}"
                )
                raise CoordMergeError(msg)
            # Only the dimension coordinate defines contiguity, so only it is
            # snapped; coords merely associated with dim just follow along.
            if coord_name == dim:
                merge_coords = _snap_coords(merge_coords)
            data = [x.values for x in merge_coords]
            dims = managers[0].dim_map[coord_name]
            new_data = np.concatenate(data, axis=axis)
            # raw value concatenation loses the coord's units; reattach
            # the (verified common) units so the merge stays unit-true
            common_units = next(iter(units))
            if common_units is not None:
                from dascore.core.coords import get_coord  # noqa: PLC0415

                coord = get_coord(data=new_data, units=common_units, snap=True)
                out[coord_name] = (dims, coord)
            else:
                out[coord_name] = (dims, new_data)
        return out

    dims = _get_dims(coord_managers)
    coord_managers, kept_unshared = _drop_unshared_coordinates(coord_managers)
    sort_managers = sorted(
        coord_managers,
        key=lambda x: x.coord_map[dim].min(),
        reverse=any(x.coord_map[dim].reverse_sorted for x in coord_managers),
    )
    to_merge = coord_managers[0].dim_to_coord_map[dim]
    others = set(coord_managers[0].coord_map) - set(to_merge)
    # non-merge coords resolve in input order; merged ones in `dim` order
    merged_coords = {
        **_get_non_merge_coords(coord_managers, others),
        **_get_merged_coords(sort_managers, to_merge),
        **kept_unshared,
    }
    _warn_decided()
    return dc.get_coord_manager(merged_coords, dims=dims)
