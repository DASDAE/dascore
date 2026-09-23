"""Validation of bounded absolute windows shared by spool operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from dascore.exceptions import ChunkError, ParameterError
from dascore.units import Quantity, convert_units
from dascore.utils.misc import express_range_for_coord


@dataclass(frozen=True)
class ExplicitRanges:
    """Immutable requested windows, retaining their original row numbers."""

    rows: tuple[tuple[object, object], ...]


def looks_explicit(value) -> bool:
    """Flag shapes needing explicit validation before ordinary query parsing."""
    return (isinstance(value, np.ndarray) and value.ndim != 1) or (
        isinstance(value, list | tuple)
        and (not value or isinstance(value[0], list | tuple | np.ndarray))
    )


def explicit_ranges(value) -> ExplicitRanges | None:
    """Return validated window rows, or None for an existing input form."""
    if not looks_explicit(value):
        if isinstance(value, np.ndarray) and value.ndim == 1:
            msg = "Explicit ranges must have shape (n, 2), not a 1D array."
            raise ParameterError(msg)
        return None
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return None
        candidate = value
    else:
        candidate = np.asarray(value, dtype=object)
    if candidate.ndim != 2 or candidate.shape[1] != 2:
        msg = f"Explicit ranges must have shape (n, 2), got {candidate.shape}."
        raise ParameterError(msg)
    rows: list[tuple[object, object]] = []
    for index in range(len(candidate)):
        low, high = candidate[index, 0], candidate[index, 1]
        bounds: list[object] = []
        for bound in (low, high):
            if isinstance(bound, np.str_):
                bound = str(bound)
            if bound is None or bound is Ellipsis:
                msg = f"Explicit range row {index} needs two closed bounds."
                raise ParameterError(msg)
            magnitude = bound.magnitude if isinstance(bound, Quantity) else bound
            if isinstance(magnitude, bool | np.bool_) or np.ndim(magnitude):
                msg = f"Explicit range row {index} has a non-scalar bound."
                raise ParameterError(msg)
            invalid = bool(pd.isna(magnitude)) or (
                isinstance(magnitude, (int, float, np.number))
                and not np.isfinite(magnitude)
            )
            if invalid:
                msg = f"Explicit range row {index} has a missing or nonfinite bound."
                raise ParameterError(msg)
            bounds.append(bound)
        if isinstance(bounds[0], Quantity) == isinstance(bounds[1], Quantity):
            if isinstance(bounds[0], Quantity):
                assert isinstance(bounds[1], Quantity)
                try:
                    comparable: list[Any] = [
                        bounds[0].magnitude,
                        cast(Any, bounds[1]).to(bounds[0].units).magnitude,
                    ]
                except Exception as exc:
                    msg = f"Explicit range row {index} has incompatible bound units."
                    raise ParameterError(msg) from exc
            else:
                comparable = bounds
            if any(isinstance(x, str) for x in comparable) and (
                all(isinstance(x, str) for x in comparable)
                or any(isinstance(x, (pd.Timestamp, np.datetime64)) for x in comparable)
            ):
                try:
                    comparable = [pd.Timestamp(cast(Any, x)) for x in comparable]
                except (TypeError, ValueError) as exc:
                    msg = (
                        f"Explicit range row {index} has string bounds which are "
                        "not parseable datetimes."
                    )
                    raise ParameterError(msg) from exc
            try:
                reversed_bounds = cast(Any, comparable[0]) > cast(Any, comparable[1])
            except (TypeError, ValueError) as exc:
                msg = f"Explicit range row {index} has incomparable bound types."
                raise ParameterError(msg) from exc
            if reversed_bounds:
                msg = (
                    f"Explicit range row {index} has its lower bound above "
                    "its upper bound."
                )
                raise ParameterError(msg)
        rows.append((bounds[0], bounds[1]))
    return ExplicitRanges(tuple(rows))


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
        payloads = dc.scan_payloads(
            resolved,
            file_format=key[1] or None,
            file_version=key[2] or None,
            progress=None,
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
    from dascore.utils.patch_assembly import coord_from_row  # noqa: PLC0415

    unit = row.get(f"_{name}_units")
    unit = None if unit is None or pd.isnull(unit) else str(unit)
    return coord_from_row(row, name, unit)


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
    recovered = []
    for member in members.to_dict("records"):
        coords = _source_manager(plan.loader, member, files, projection=projection)
        if coords is None:
            return None
        coords = _select_manager(coords, plan.parent_residuals)
        dim = plan.dim
        native = coords.coord_map[dim].units
        unit = member.get(f"_{dim}_units")
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
                coords = coords.convert_units(**{plan.dim: unit})
        recovered.append(coords)
    if not recovered:
        if projected:
            raise ChunkError("plan output has no members or reconstructable grid")
        anchor_id = plan._output_rows[int(key)].get("_anchor_patch_row")
        anchor = plan._anchor_rows.get(anchor_id)
        coords = (
            None
            if anchor is None
            else _source_manager(plan.loader, anchor, files, projection=projection)
        )
        if coords is not None:
            coords = _select_manager(coords, plan.parent_residuals)
    elif len(recovered) == 1:
        coords = recovered[0]
    else:
        # Patch utilities import spool/chunk planning; defer until plans are read.
        from dascore.utils.patch import _get_merged_coord  # noqa: PLC0415

        merge_dim = recovered[0].dim_map[plan.dim][0]
        coords = _get_merged_coord(
            pd.DataFrame(),
            merge_dim,
            recovered,
            snap_coords=plan.merge_kwargs.get("snap_coords", True),
            tolerance=plan.merge_kwargs.get("tolerance", 1.5),
        )
    if coords is None:
        return None
    if projected:
        target = row.get(f"_{plan.dim}_units")
        current = coords.coord_map[plan.dim].units
        if target is not None and not pd.isnull(target) and current is not None:
            if str(current) != str(target):
                coords = coords.convert_units(**{plan.dim: target})
    if plan.merge_kwargs.get("fill_value") is None or plan.dim not in coords.dims:
        return _project_manager(coords, requested_projection)
    # A filled output takes its grid from the output, not its anchor.
    filled = _row_coordinate(row, plan.dim)
    current = coords.coord_map[plan.dim]
    if filled is None or (
        filled.shape == current.shape
        and filled.min() == current.min()
        and filled.max() == current.max()
        and filled.step == current.step
        and str(filled.units) == str(current.units)
    ):
        # Float index envelopes can describe an identical integer grid.
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
