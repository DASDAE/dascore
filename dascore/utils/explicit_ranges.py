"""Validation of bounded absolute windows shared by spool operations."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd

from dascore.exceptions import ParameterError
from dascore.units import Quantity, convert_units
from dascore.utils.misc import express_range_for_coord


@dataclass(frozen=True)
class ExplicitRanges:
    """Immutable requested windows, retaining their original row numbers."""

    rows: tuple[tuple[object, object], ...]


def explicit_ranges(value) -> ExplicitRanges | None:
    """Return validated window rows, or None for an existing input form."""
    if isinstance(value, np.ndarray):
        if value.ndim == 1:
            msg = "Explicit ranges must have shape (n, 2), not a 1D array."
            raise ParameterError(msg)
        candidate = value
    elif isinstance(value, list | tuple) and (
        not value or isinstance(value[0], list | tuple | np.ndarray)
    ):
        candidate = np.asarray(value, dtype=object)
    else:
        return None
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
            value = bound.magnitude if isinstance(bound, Quantity) else bound
            if isinstance(value, bool | np.bool_) or np.ndim(value):
                msg = f"Explicit range row {index} has a non-scalar bound."
                raise ParameterError(msg)
            invalid = bool(pd.isna(value)) or (
                isinstance(value, (int, float, np.number)) and not np.isfinite(value)
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
    if not path or path.startswith(("memorypatch://", "plan://")):
        return None
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


def known_coordinates(catalog, rows, name: str) -> dict:
    """Find source coordinates from memory or metadata-only file scans."""
    live = catalog.resolver.live_entries()
    out = {}
    files = {}
    resolver = catalog.resolver
    for row in rows.to_dict("records"):
        patch_row = row["_patch_row"]
        path = str(row.get("source_path") or "")
        patch = live.get(path)
        if patch is not None:
            if name in patch.coords.coord_map:
                out[patch_row] = patch.coords.coord_map[name]
            continue
        if path.startswith("plan://"):
            coord = _planned_coordinate(catalog.resolver, row, name)
            if coord is not None:
                out[patch_row] = coord
            continue
        if not path or path.startswith("memorypatch://"):
            continue
        if not any(
            name in selectors for selectors, _, _ in getattr(catalog, "residuals", ())
        ):
            # patch_assembly imports chunk planning, which imports this module.
            from dascore.utils.patch_assembly import coord_from_row  # noqa: PLC0415

            unit = row.get(f"_{name}_units")
            unit = None if unit is None or pd.isnull(unit) else str(unit)
            coord = coord_from_row(row, name, unit)
            if coord is not None:
                out[patch_row] = coord
                continue
        coords = file_source_coords(resolver, row, files)
        if coords is not None and name in coords.coord_map:
            out[patch_row] = coords.coord_map[name]
    return out


def _planned_coordinate(resolver, row, name):
    """Recover a derived coordinate from its member metadata."""
    plan = resolver
    if not hasattr(plan, "member_rows"):
        plans = getattr(resolver, "plans", {})
        path = str(row.get("source_path") or "")
        plan = next(
            (item for prefix, item in plans.items() if path.startswith(prefix)), None
        )
    assert plan is not None and hasattr(plan, "member_rows"), (
        "plan source has no resolver"
    )
    if plan.merge_kwargs.get("fill_value") is not None:
        # patch_assembly imports chunk planning, which imports this module.
        from dascore.utils.patch_assembly import coord_from_row  # noqa: PLC0415

        unit = row.get(f"_{name}_units")
        unit = None if unit is None or pd.isnull(unit) else str(unit)
        filled_coord = coord_from_row(row, name, unit)
        if filled_coord is not None:
            return filled_coord
    key = str(row.get("source_patch_key") or "")
    assert key.isdigit(), "plan source key is not an output ordinal"
    members = plan.member_rows[plan.member_rows["output_id"] == int(key)]
    assert not members.empty, "plan output has no members or reconstructable grid"
    recovered = []
    for _, member_series in members.iterrows():
        member = member_series.to_dict()
        member["_patch_row"] = 0
        nested = known_coordinates(
            SimpleNamespace(resolver=plan.loader), pd.DataFrame([member]), name
        )
        coord = nested.get(0)
        if coord is None:
            return None
        for selectors, samples, relative in plan.parent_residuals:
            if name in selectors:
                value = express_range_for_coord(selectors[name], coord)
                coord, _ = coord.select(value, samples=samples, relative=relative)
        if member.get("_modified"):
            low, high = member.get(f"{name}_min"), member.get(f"{name}_max")
            source_unit = getattr(coord, "units", None)
            plan_unit = member.get(f"_{name}_units")
            if (
                source_unit is not None
                and plan_unit is not None
                and not pd.isnull(plan_unit)
                and str(source_unit) != str(plan_unit)
            ):
                low, high = (
                    convert_units(x, to_units=source_unit, from_units=plan_unit)
                    for x in (low, high)
                )
            coord, _ = coord.select((low, high))
        target = row.get(f"_{name}_units")
        if target is not None and not pd.isnull(target) and coord.units is not None:
            if str(coord.units) != str(target):
                coord = coord.convert_units(target)
        recovered.append(coord)
    if len(recovered) == 1:
        return recovered[0]
    # Patch utilities import spool/chunk planning; defer these until plans are read.
    from dascore.core.coordmanager import get_coord_manager  # noqa: PLC0415
    from dascore.utils.patch import _get_merged_coord  # noqa: PLC0415

    managers = [get_coord_manager({name: coord}, dims=(name,)) for coord in recovered]
    merged = _get_merged_coord(
        pd.DataFrame(),
        name,
        managers,
        snap_coords=plan.merge_kwargs.get("snap_coords", True),
        tolerance=plan.merge_kwargs.get("tolerance", 1.5),
    )
    return merged.coord_map[name]
