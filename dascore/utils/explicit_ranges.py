"""Validation of absolute windows shared by spool operations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from dascore.config import config_context
from dascore.exceptions import ParameterError
from dascore.units import Quantity, convert_units
from dascore.utils.misc import express_range_for_coord, suppress_warnings


@dataclass(frozen=True)
class ExplicitRanges:
    """Immutable requested windows, retaining their original row numbers."""

    rows: tuple[tuple[object, object], ...]


def looks_explicit(value) -> bool:
    """Flag shapes needing explicit validation before ordinary query parsing."""
    if isinstance(value, Quantity):  # an (n, 2) array with units
        return np.ndim(value.magnitude) == 2
    return (isinstance(value, np.ndarray) and value.ndim > 1) or (
        isinstance(value, list | tuple)
        and (not value or isinstance(value[0], list | tuple | np.ndarray))
    )


def explicit_ranges(value) -> ExplicitRanges | None:
    """
    Return validated window rows, or None for an existing input form.

    An open end is None; an infinity opens its own side, so a row holding
    one keeps its other bound on the far side. Bounds are left in the
    order given, which `_absolute_bounds` swaps where reversed.
    """
    if not looks_explicit(value):
        if isinstance(value, np.ndarray) and value.ndim == 1:
            msg = "Explicit ranges must have shape (n, 2), not a 1D array."
            raise ParameterError(msg)
        return None
    if isinstance(value, Quantity):  # each bound a quantity
        value = [[x * value.units for x in row] for row in value.magnitude]
    candidate = np.asarray(
        value, dtype=None if isinstance(value, np.ndarray) else object
    )
    if candidate.ndim != 2 or candidate.shape[1] != 2:
        msg = f"Explicit ranges must have shape (n, 2), got {candidate.shape}."
        raise ParameterError(msg)
    rows: list[tuple[object, object]] = []
    for index in range(len(candidate)):
        bounds, opens = [], []
        for bound in candidate[index]:
            bound = str(bound) if isinstance(bound, np.str_) else bound
            if bound is None or bound is Ellipsis:
                bounds.append(None)  # an open end
                continue
            magnitude: Any = bound.magnitude if isinstance(bound, Quantity) else bound
            if isinstance(magnitude, bool | np.bool_) or np.ndim(magnitude):
                msg = f"Explicit range row {index} has a non-scalar bound."
                raise ParameterError(msg)
            inf = isinstance(magnitude, float | np.floating) and np.isinf(magnitude)
            opens += [magnitude > 0] if inf else []
            bounds.append(None if inf or pd.isna(magnitude) else bound)  # NaN, NaT
        if len(opens) == 2 and opens[0] == opens[1]:  # one sign twice: nothing
            msg = f"Explicit range row {index} has two infinities of one sign."
            raise ParameterError(msg)
        if opens:  # +inf leaves the row's other bound low, -inf high
            closed = [x for x in bounds if x is not None] if len(set(opens)) < 2 else []
            bounds = [*closed, None] if set(opens) == {True} else [None, *closed]
            bounds = [*bounds, None][:2]
        rows.append((bounds[0], bounds[1]))
    return ExplicitRanges(tuple(rows))


def explicit_windows(values: Mapping) -> dict[str, ExplicitRanges]:
    """Return the explicit ranges in ``values``; all must have one row count."""
    out = {k: r for k, v in values.items() if (r := explicit_ranges(v)) is not None}
    counts = {name: len(ranges.rows) for name, ranges in out.items()}
    if len(set(counts.values())) > 1:
        msg = f"Explicit ranges must have the same number of rows, got {counts}."
        raise ParameterError(msg)
    return out


def file_source_coords(resolver, row, files=None):
    """Return a file source's full coordinate metadata without reading data."""
    # The public scanner retains arbitrary and associated coordinate arrays.
    # Import locally because dascore imports this module during initialization.
    import dascore as dc  # noqa: PLC0415

    path = str(row.get("source_path") or "")
    files = {} if files is None else files
    file_resolver = getattr(resolver, "file", resolver)
    resolved = file_resolver.resolve_path(path)
    key = (
        str(resolved),
        str(row.get("source_format") or ""),
        str(row.get("source_version") or ""),
    )
    if key not in files:
        # the decode the source's patches are read under, as its rows were
        snap, shift = file_resolver.decode(path)
        with config_context(max_shift=shift):
            payloads = dc.scan_payloads(
                resolved,
                file_format=key[1] or None,
                file_version=key[2] or None,
                progress=None,
                snap=snap,
            )
        by_key = {
            str(payload._source.key): payload
            for payload in payloads
            if payload._source is not None
        }
        files[key] = (payloads, by_key)
    payloads, by_key = files[key]
    source_key = str(row.get("source_patch_key") or "")
    payload = by_key.get(source_key)
    return None if payload is None else payload.coords


def _select_manager(coords, residuals):
    """Replay coordinate selectors on metadata, including shared dimensions."""
    for selectors, samples, relative in residuals:
        usable = {
            name: express_range_for_coord(value, coords.coord_map[name])
            for name, value in selectors.items()
            if name in coords.coord_map
        }
        if usable:
            coords, _ = coords.select(samples=samples, relative=relative, **usable)
    return coords


def _one_coordinate(name, coord):
    """Wrap one known dimension grid for projected metadata recovery."""
    # CoordManager imports spool/chunk planning during initialization.
    from dascore.core.coordmanager import get_coord_manager  # noqa: PLC0415

    return get_coord_manager({name: coord}, dims=(name,))


def _project_manager(coords, name):
    """Return only the requested coordinate after any full recovery."""
    if name is None:
        return coords
    coord = coords.coord_map.get(name)
    return None if coord is None else _one_coordinate(name, coord)


def _row_coordinate(row, name):
    """Rebuild an indexed regular grid before scanning a file payload."""
    from dascore.utils import patch_assembly as pa  # noqa: PLC0415

    unit = row.get(f"_{name}_units")
    unit = None if unit is None or pd.isnull(unit) else str(unit)
    if (source := pa.source_coord_from_row(row, name, unit)) is None:
        return pa.coord_from_row(row, name, unit)
    # A trimmed member's bounds need not sit on its file's samples.
    return source.select(pa._row_bounds(row, name))[0]


def _source_manager(resolver, row, files=None, projection=None):
    """Recover source metadata, projecting a regular grid when sufficient."""
    path = str(row.get("source_path") or "")
    live = resolver.live_entries().get(path)
    if live is not None:
        coords = live.coords
    elif path.startswith("plan://"):
        plans = getattr(resolver, "plan_entries", dict)()
        plan = next(
            (item for prefix, item in plans.items() if path.startswith(prefix)), None
        )
        assert plan is not None, "plan source has no resolver"
        return _planned_manager(plan, row, files, projection=projection)
    else:
        if not path or path.startswith("memorypatch://"):
            return None
        if projection is not None:
            coord = _row_coordinate(row, projection)
            dtype = row.get(f"_{projection}_coord_dtype")
            # Float envelopes omit the expression that rounded each label.
            # Only integer/tick grids can supply exact samples from the index.
            if (
                coord is not None
                and isinstance(dtype, str)
                and np.dtype(dtype).kind in "iuMm"
            ):
                return _one_coordinate(projection, coord)
        coords = file_source_coords(getattr(resolver, "loader", resolver), row, files)
    return None if coords is None else _project_manager(coords, projection)


def _planned_manager(plan, row, files=None, projection=None):
    """Recover a plan's coordinates with optional single-grid projection."""
    requested_projection = projection
    projected = projection == plan.dim and all(
        set(selectors) <= {projection} for selectors, _, _ in plan.parent_residuals
    )
    projection = projection if projected else None
    if projected and plan.merge_kwargs.get("fill_value") is not None:
        filled = _row_coordinate(row, projection)
        if filled is not None:
            return _one_coordinate(projection, filled)
    key = str(row.get("source_patch_key") or "")
    assert key.isdigit(), "plan source key is not an output ordinal"
    members = plan.member_rows[plan.member_rows["output_id"] == int(key)]
    rows, recovered = members.to_dict("records"), []
    for member in rows:
        coords = _source_manager(plan.loader, member, files, projection=projection)
        if coords is None:
            return None
        coords = _select_manager(coords, plan.parent_residuals)
        planned = (plan.dim, *plan.trim_dims)
        for dim in [x for x in planned if x in coords.coord_map]:
            native = coords.coord_map[dim].units
            unit = member.get(f"_{dim}_units") if dim == plan.dim else None
            if member.get("_modified"):
                low, high = member.get(f"{dim}_min"), member.get(f"{dim}_max")
                if native is not None and unit is not None and not pd.isnull(unit):
                    if str(native) != str(unit):
                        low, high = (
                            convert_units(x, to_units=native, from_units=unit)
                            for x in (low, high)
                        )
                coords, _ = coords.select(**{dim: (low, high)})
            if native is not None and unit is not None and not pd.isnull(unit):
                if str(native) != str(unit):
                    coords = coords.convert_units(**{dim: unit})
        recovered.append(coords)
    # Patch utilities import spool/chunk planning; defer until plans are read.
    from dascore.utils import patch_assembly as pa  # noqa: PLC0415

    if len(recovered) == 1:
        coords = recovered[0]
    else:
        from dascore.utils.patch import (  # noqa: PLC0415
            _get_merged_coord,
            _split_coord_merge_kwargs,
        )

        merge_dim = recovered[0].dim_map[plan.dim][0]
        _, kwargs = _split_coord_merge_kwargs(plan.merge_kwargs)
        # the merge which loads the output says what its policy settled
        coords = _get_merged_coord(
            pd.DataFrame(),
            merge_dim,
            recovered,
            warn=False,
            **kwargs,
        )
    coords = pa.on_stretch_grid(coords, plan.dim, rows[0])
    if (snap := plan.merge_kwargs.get("snap")) is not None:  # as the patch loads
        names = [x for x in snap[0] or coords.dims if x in coords.dims]
        with suppress_warnings(UserWarning):  # warned as it loads
            coords = coords.snap(*names, **snap[1])[0]
    if plan.merge_kwargs.get("fill_value") is None or plan.dim not in coords.dims:
        return _project_manager(coords, requested_projection)
    # A filled output takes its grid from the output, not its anchor.
    filled = _row_coordinate(row, plan.dim)
    current = coords.coord_map[plan.dim]
    if filled is None or filled == current:  # nothing filled, riders keep
        return _project_manager(coords, requested_projection)
    riding = [
        name
        for name, dims in coords.dim_map.items()
        if name != plan.dim and plan.dim in dims
    ]
    coords, _ = coords.drop_coords(*riding)
    coords = coords._update_grid(plan.dim, **{plan.dim: filled})
    return _project_manager(coords, requested_projection)


def known_coordinates(catalog, rows, name: str) -> dict:
    """Find exact coordinates, replaying shared-dimension residuals in metadata."""
    out = {}
    files = {}
    residuals = getattr(catalog, "residuals", ())
    dims_map = catalog.backend.coord_dims_map() if residuals else {}
    target_dims = set(str(dims_map.get(name, name)).split(","))
    shared = any(
        target_dims & set(str(dims_map.get(selector, selector)).split(","))
        for selectors, _, _ in residuals
        for selector in selectors
    )
    projection = None if shared else name
    for row in rows.to_dict("records"):
        coords = _source_manager(catalog.resolver, row, files, projection=projection)
        if coords is None:
            if shared:
                # A regular index cannot prove a shared-dimension residual
                # after the source coordinate metadata is lost.
                out[row["_patch_row"]] = None
            continue
        if shared:
            coords = _select_manager(coords, residuals)
        if name in coords.coord_map:
            out[row["_patch_row"]] = coords.coord_map[name]
    return out
