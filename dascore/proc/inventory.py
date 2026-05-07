"""Patch processing helpers for inventory-derived metadata."""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from dascore.constants import PatchType
from dascore.core.attrs import PatchAttrs
from dascore.core.inventory import (
    Acquisition,
    FiberArray,
    Geometry,
    Inventory,
    Network,
)
from dascore.exceptions import ParameterError
from dascore.utils.misc import iterate
from dascore.utils.patch import patch_function
from dascore.utils.time import is_datetime64

_BoundaryPolicy = Literal["raise", "warn", "ignore"]
_MissingPolicy = Literal["raise", "nan"]
_AttrMissingPolicy = Literal["raise", "ignore"]
_TrackName = Literal[
    "optical_component",
    "geometry",
    "coupling_condition",
    "annotation",
]
_TRACK_NAMES: tuple[_TrackName, ...] = (
    "optical_component",
    "geometry",
    "coupling_condition",
    "annotation",
)
_QUALIFIED_TRACKS = set(_TRACK_NAMES)
_ATTR_SOURCE_NAMES = ("fiber_array", "acquisition", "interrogator")
_XYZ_INDEX = {"x": 0, "y": 1, "z": 2}


@dataclass(frozen=True)
class _InventoryContext:
    """Resolved inventory objects for one patch data source."""

    fiber_array: FiberArray
    acquisition: Acquisition
    patch_time: object


def _validate_inventory(inventory) -> Inventory:
    """Return inventory after ensuring it is the expected model type."""
    if not isinstance(inventory, Inventory):
        msg = "inventory must be an Inventory instance."
        raise ParameterError(msg)
    return inventory


def _get_data_source_id(patch: PatchType, data_source_id: str | None) -> str:
    """Return an explicit or patch-stored data source id."""
    out = data_source_id or patch.attrs.data_source_id
    if not out:
        msg = "A data_source_id is required to resolve inventory context."
        raise ParameterError(msg)
    return out


def _get_patch_time(patch: PatchType):
    """Return patch start time when it is an absolute datetime coordinate."""
    if "time" not in patch.coords:
        return None
    time = patch.coords.min("time")
    if not is_datetime64(time) or np.isnat(time):
        return None
    return time


def _get_unique_match(items, predicate, description: str):
    """Return the unique item matching predicate or raise."""
    matches = tuple(item for item in items if predicate(item))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = f"No inventory {description} found."
        raise ParameterError(msg)
    msg = f"Multiple inventory {description} records found."
    raise ParameterError(msg)


def _resolve_inventory_context(
    patch: PatchType,
    inventory,
    data_source_id: str | None,
) -> _InventoryContext:
    """Resolve patch inventory context by traversing Network/FiberArray/Acquisition."""
    source_attrs, fiber_array = _resolve_fiber_array(patch, inventory, data_source_id)
    acquisition = _get_unique_match(
        fiber_array.acquisitions,
        lambda item: (
            item.location_code == source_attrs.location
            and item.code == source_attrs.acquisition
        ),
        (
            "acquisition with location "
            f"{source_attrs.location!r} and code {source_attrs.acquisition!r}"
        ),
    )
    return _InventoryContext(
        fiber_array=fiber_array,
        acquisition=acquisition,
        patch_time=_get_patch_time(patch),
    )


def _resolve_fiber_array(
    patch: PatchType,
    inventory,
    data_source_id: str | None,
) -> tuple[PatchAttrs, FiberArray]:
    """Resolve the network and fiber array from a data source id."""
    inventory = _validate_inventory(inventory)
    source_attrs = PatchAttrs(data_source_id=_get_data_source_id(patch, data_source_id))
    time = _get_patch_time(patch)
    networks = inventory.get_records(record_types=Network, time=time)
    network = _get_unique_match(
        networks,
        lambda item: item.code == source_attrs.network,
        f"network with code {source_attrs.network!r}",
    )
    fiber_array = _get_unique_match(
        network.fiber_arrays,
        lambda item: item.code == source_attrs.fiber_array,
        f"fiber array with code {source_attrs.fiber_array!r}",
    )
    return source_attrs, fiber_array


def _get_fiber_array(
    patch: PatchType,
    inventory,
    data_source_id: str | None,
) -> FiberArray:
    """Return the resolved fiber array view for a patch."""
    return _resolve_fiber_array(patch, inventory, data_source_id)[1]


def _get_acquisition(
    patch: PatchType,
    inventory: Inventory,
    data_source_id: str | None,
) -> Acquisition:
    """Return the resolved acquisition view for a patch."""
    return _resolve_inventory_context(patch, inventory, data_source_id).acquisition


def _get_integer_dim_array(patch: PatchType, dim: str) -> np.ndarray:
    """Return an integer dimension coordinate array."""
    patch.get_axis(dim)
    values = patch.get_array(dim)
    if not np.issubdtype(values.dtype, np.integer):
        msg = f"{dim!r} coordinate must contain integer values."
        raise ParameterError(msg)
    return values


def _validate_distance_update(patch: PatchType, dim: str) -> None:
    """Ensure an existing distance coordinate belongs to the requested dimension."""
    if "distance" not in patch.coords:
        return
    distance_dims = patch.coords.dim_map["distance"]
    if distance_dims == (dim,):
        return
    msg = (
        "Patch already has a distance coordinate associated with "
        f"{distance_dims}, not {(dim,)}."
    )
    raise ParameterError(msg)


def _distance_from_config(
    values: np.ndarray,
    config: Acquisition,
) -> np.ndarray:
    """Return optical distance from positional integer channel/index values."""
    interval = config.spatial_sampling_interval
    if interval is None:
        msg = "Acquisition.spatial_sampling_interval is required."
        raise ParameterError(msg)
    return config.first_channel_distance + values * interval


def _normalize_coord_requests(coords) -> tuple[str, ...]:
    """Normalize one or more requested inventory coordinates."""
    out = tuple(iterate(coords))
    if not out:
        msg = "At least one inventory coordinate must be requested."
        raise ParameterError(msg)
    if not all(isinstance(item, str) and item for item in out):
        msg = "Inventory coordinate requests must be non-empty strings."
        raise ParameterError(msg)
    return out


def _normalize_attr_requests(attrs) -> tuple[str, ...]:
    """Normalize one or more requested inventory attrs."""
    out = tuple(iterate(attrs))
    if not out:
        msg = "At least one inventory attr must be requested."
        raise ParameterError(msg)
    if not all(isinstance(item, str) and item for item in out):
        msg = "Inventory attr requests must be non-empty strings."
        raise ParameterError(msg)
    return out


def _validate_policy(value: str, valid: set[str], name: str) -> None:
    """Validate a string policy argument."""
    if value not in valid:
        valid_str = ", ".join(sorted(valid))
        msg = f"{name} must be one of {valid_str}, not {value!r}."
        raise ParameterError(msg)


def _get_optical_path(fiber_array: FiberArray, time=None):
    """Return the single optical path valid for a fiber array and time."""
    matches = tuple(
        path for path in fiber_array.optical_paths if path.is_effective_at(time)
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        msg = "FiberArray must resolve to an optical path."
        raise ParameterError(msg)
    msg = "FiberArray resolves to multiple optical paths."
    raise ParameterError(msg)


def _get_distance_values_and_dim(patch: PatchType) -> tuple[np.ndarray, str]:
    """Return distance values and their single associated dimension."""
    if "distance" not in patch.coords:
        msg = (
            "Patch must have a distance coordinate before adding inventory "
            "coordinates. For channel/index patches, call "
            "distance_from_inventory first."
        )
        raise ParameterError(msg)
    dims = patch.coords.dim_map["distance"]
    if len(dims) != 1:
        msg = "Patch distance coordinate must be associated with exactly one dimension."
        raise ParameterError(msg)
    distance = patch.get_array("distance")
    if distance.ndim != 1:
        msg = "Patch distance coordinate must be one-dimensional."
        raise ParameterError(msg)
    return distance.astype(float), dims[0]


def _check_distance_bounds(distance: np.ndarray, path) -> None:
    """Ensure distance values are within known optical path length."""
    if path.optical_length is None:
        return
    bad = (distance < 0.0) | (distance > float(path.optical_length))
    if np.any(bad):
        msg = "Patch distance values must fall within the optical path length."
        raise ParameterError(msg)


def _get_track_edges(path, track: _TrackName) -> tuple[tuple, np.ndarray]:
    """Return records and cumulative distance edges for an optical path track."""
    sequence = path._get_interval_sequence(track)
    records = sequence.records
    if not records:
        msg = f"Optical path has no {sequence.display_name} records."
        raise ParameterError(msg)
    lengths = np.asarray([record.optical_length for record in records], dtype=float)
    if np.any(np.isnan(lengths)):
        msg = f"Projecting {sequence.display_name} requires all records to have length."
        raise ParameterError(msg)
    total = float(np.sum(lengths))
    if path.optical_length is not None and not np.isclose(
        total,
        float(path.optical_length),
        atol=1e-9,
    ):
        msg = (
            f"{sequence.display_name} optical length {total} does not match "
            f"optical path length {path.optical_length}."
        )
        raise ParameterError(msg)
    return records, np.concatenate([[0.0], np.cumsum(lengths)])


def _get_interval_indices(
    path,
    track: _TrackName,
    distance: np.ndarray,
    on_boundary: _BoundaryPolicy,
    warned: list[bool],
) -> tuple[tuple, np.ndarray, np.ndarray]:
    """Return records, edges, and assigned interval index for distance values."""
    records, edges = _get_track_edges(path, track)
    interior = edges[1:-1]
    boundary = np.isin(distance, interior)
    if np.any(boundary):
        if on_boundary == "raise":
            msg = "Distance samples fall exactly on an inventory interval boundary."
            raise ParameterError(msg)
        if on_boundary == "warn" and not warned[0]:
            warnings.warn(
                "Distance samples fall exactly on an inventory interval boundary; "
                "assigning them to the interval on the right.",
                stacklevel=3,
            )
            warned[0] = True
    index = np.searchsorted(edges, distance, side="right") - 1
    index = np.clip(index, 0, len(records) - 1)
    return records, edges, index


def _project_field(
    path,
    distance: np.ndarray,
    track: _TrackName,
    field: str,
    on_boundary: _BoundaryPolicy,
    warned: list[bool],
) -> np.ndarray:
    """Project one inventory record field onto patch distance values."""
    if track == "annotation":
        return _project_annotation_field(path, distance, field)
    records, _, index = _get_interval_indices(
        path, track, distance, on_boundary, warned
    )
    values = [getattr(records[item], field, None) for item in index]
    return np.asarray(values, dtype=object)


def _project_annotation_field(path, distance: np.ndarray, field: str) -> np.ndarray:
    """Project overlapping annotation fields onto distance values."""
    out = []
    for value in distance:
        matches = []
        for annotation in path.annotations:
            start, stop = annotation.distance
            start = 0.0 if start is None else start
            stop = path.optical_length if stop is None else stop
            if stop is None:
                msg = "Open-ended annotation intervals require optical path length."
                raise ParameterError(msg)
            if start <= value < stop or (
                path.optical_length is not None and value == path.optical_length == stop
            ):
                matches.append(getattr(annotation, field, None))
        matches = tuple(matches)
        matches = tuple(item for item in matches if item not in ("", None))
        if len(matches) == 0:
            out.append("")
        elif len(matches) == 1:
            out.append(matches[0])
        else:
            out.append(matches)
    return np.asarray(out, dtype=object)


def _get_axis_index(geometry: Geometry, axis: str) -> int | None:
    """Return coordinate tuple index for a requested geometry axis."""
    crs = geometry.coordinate_reference_system
    if crs is not None and crs.axis_order:
        try:
            return crs.axis_order.index(axis)
        except ValueError:
            return None
    return _XYZ_INDEX.get(axis)


def _raise_or_nan(
    values: np.ndarray,
    mask: np.ndarray,
    message: str,
    on_missing: _MissingPolicy,
) -> None:
    """Raise for missing geometry data or leave NaNs in the output."""
    if not np.any(mask) or on_missing == "nan":
        return
    raise ParameterError(message)


def _project_geometry_axis(
    path,
    distance: np.ndarray,
    axis: str,
    on_boundary: _BoundaryPolicy,
    on_missing: _MissingPolicy,
    warned: list[bool],
) -> np.ndarray:
    """Project one named geometry axis onto patch distance values."""
    records, edges, index = _get_interval_indices(
        path, "geometry", distance, on_boundary, warned
    )
    out = np.full(distance.shape, np.nan, dtype=float)
    for geo_index, geometry in enumerate(records):
        mask = index == geo_index
        if not np.any(mask):
            continue
        geometry_type = geometry.geometry_type or "unknown"
        if geometry_type == "unknown":
            continue
        if geometry_type != "linear":
            msg = (
                "Inventory coordinate projection currently supports only linear "
                f"or unknown geometry, not {geometry.geometry_type!r}."
            )
            raise ParameterError(msg)
        coordinates = np.asarray(geometry.coordinates, dtype=float)
        axis_index = _get_axis_index(geometry, axis)
        missing = (
            coordinates.ndim != 2
            or coordinates.shape[0] == 0
            or axis_index is None
            or axis_index >= coordinates.shape[1]
        )
        if missing:
            msg = f"Geometry axis {axis!r} is not available for all intervals."
            _raise_or_nan(out, mask, msg, on_missing)
            continue
        if coordinates.shape[0] == 1 or geometry.optical_length == 0:
            out[mask] = coordinates[0, axis_index]
            continue
        start = edges[geo_index]
        local_fraction = (distance[mask] - start) / float(geometry.optical_length)
        axis_values = coordinates[[0, -1], axis_index]
        out[mask] = axis_values[0] + local_fraction * (axis_values[1] - axis_values[0])
    return out


def _track_has_field(path, track: _TrackName, field: str) -> bool:
    """Return True if a resolved path track exposes a field name."""
    if track == "annotation":
        records = path.annotations
    else:
        records = path._get_interval_sequence(track).records
    return any(field in record.__class__.model_fields for record in records)


def _resolve_metadata_track(path, field: str) -> _TrackName | None:
    """Return the unambiguous interval metadata track for a field, if any."""
    matches = tuple(
        track for track in _TRACK_NAMES if _track_has_field(path, track, field)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        suggestions = ", ".join(f"{track}.{field}" for track in matches)
        msg = f"Inventory coordinate {field!r} is ambiguous; use one of {suggestions}."
        raise ParameterError(msg)
    return None


def _project_inventory_coord(
    path,
    distance: np.ndarray,
    name: str,
    on_boundary: _BoundaryPolicy,
    on_missing: _MissingPolicy,
    warned: list[bool],
) -> tuple[str, np.ndarray]:
    """Project one requested inventory coordinate onto patch distance values."""
    if name == "label":
        return name, _project_annotation_field(path, distance, "label")
    if "." in name:
        track, field = name.split(".", 1)
        if track not in _QUALIFIED_TRACKS or not field:
            msg = f"Invalid inventory coordinate request {name!r}."
            raise ParameterError(msg)
        return name, _project_field(path, distance, track, field, on_boundary, warned)
    if (track := _resolve_metadata_track(path, name)) is not None:
        return name, _project_field(path, distance, track, name, on_boundary, warned)
    return name, _project_geometry_axis(
        path, distance, name, on_boundary, on_missing, warned
    )


def _get_attr_sources(
    fiber_array: FiberArray | None,
    acquisition: Acquisition | None,
) -> dict[str, object | None]:
    """Return supported inventory attr source objects."""
    interrogator = None if acquisition is None else acquisition.interrogator
    return {
        "fiber_array": fiber_array,
        "acquisition": acquisition,
        "interrogator": interrogator,
    }


def _requests_source(requests: tuple[str, ...], source_name: str) -> bool:
    """Return True if attr requests explicitly need one source."""
    return any(request.split(".", 1)[0] == source_name for request in requests)


def _handle_missing_attr(name: str, on_missing: _AttrMissingPolicy) -> None:
    """Raise or ignore one missing inventory attr request."""
    if on_missing == "ignore":
        return
    msg = f"Inventory attr {name!r} could not be resolved."
    raise ParameterError(msg)


def _get_qualified_inventory_attr(
    sources: dict[str, object | None],
    name: str,
    on_missing: _AttrMissingPolicy,
) -> tuple[str, object] | None:
    """Return one qualified inventory attr value."""
    source_name, field = name.split(".", 1)
    if source_name not in _ATTR_SOURCE_NAMES or not field:
        msg = f"Invalid inventory attr request {name!r}."
        raise ParameterError(msg)
    source = sources[source_name]
    if source is None or field not in source.__class__.model_fields:
        _handle_missing_attr(name, on_missing)
        return None
    return name.replace(".", "_"), getattr(source, field)


def _get_unqualified_inventory_attr(
    sources: dict[str, object | None],
    name: str,
    on_missing: _AttrMissingPolicy,
) -> tuple[str, object] | None:
    """Return one unqualified inventory attr value."""
    matches = tuple(
        source_name
        for source_name, source in sources.items()
        if source is not None and name in source.__class__.model_fields
    )
    if len(matches) == 1:
        return name, getattr(sources[matches[0]], name)
    if len(matches) > 1:
        suggestions = ", ".join(f"{source}.{name}" for source in matches)
        msg = f"Inventory attr {name!r} is ambiguous; use one of {suggestions}."
        raise ParameterError(msg)
    _handle_missing_attr(name, on_missing)
    return None


def _get_inventory_attr(
    sources: dict[str, object | None],
    name: str,
    on_missing: _AttrMissingPolicy,
) -> tuple[str, object] | None:
    """Return one requested inventory attr value."""
    if "." in name:
        return _get_qualified_inventory_attr(sources, name, on_missing)
    return _get_unqualified_inventory_attr(sources, name, on_missing)


@patch_function()
def distance_from_inventory(
    patch: PatchType,
    inventory: Inventory,
    dim: str = "channel",
    data_source_id: str | None = None,
) -> PatchType:
    """
    Add optical distance derived from an inventory acquisition.

    Parameters
    ----------
    inventory
        Inventory containing the acquisition.
    dim
        Positional integer patch dimension used as channel or index values.
    data_source_id
        Dot-delimited data source id. Defaults to ``patch.attrs.data_source_id``.

    See Also
    --------
    [Inventory tutorial](/tutorial/inventory.qmd)
        Examples of deriving distance and adding inventory metadata to patches.
    [Inventory design note](/notes/inventory_design.qmd)
        Rationale for the inventory model and standard deviations.
    """
    inventory = _validate_inventory(inventory)
    acquisition = _get_acquisition(patch, inventory, data_source_id)
    _validate_distance_update(patch, dim)
    values = _get_integer_dim_array(patch, dim)
    distance = _distance_from_config(values, acquisition)
    return patch.update_coords(distance=(dim, distance))


@patch_function()
def add_inventory_coords(
    patch: PatchType,
    inventory: Inventory,
    coords: str | Sequence[str] = ("label",),
    data_source_id: str | None = None,
    on_boundary: _BoundaryPolicy = "raise",
    on_missing: _MissingPolicy = "raise",
) -> PatchType:
    """
    Add coordinates derived from an inventory optical path.

    Parameters
    ----------
    inventory
        Inventory containing the fiber array and optical path.
    coords
        Inventory coordinates to add. ``"label"`` projects annotation labels.
        Geometry axes must be requested explicitly, such as ``"x"``, ``"y"``,
        ``"z"``, ``"latitude"``, ``"longitude"``, or ``"elevation"``.
    data_source_id
        Dot-delimited data source id. Defaults to ``patch.attrs.data_source_id``.
    on_boundary
        How to handle distance samples on interior inventory interval boundaries.
        Options are ``"raise"``, ``"warn"``, and ``"ignore"``.
    on_missing
        How to handle requested geometry axes which are unavailable on an
        interval. Options are ``"raise"`` and ``"nan"``.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch().update_attrs(
    ...     data_source_id="RD.RDAS..RAW",
    ... )
    >>> inventory = dc.get_example_inventory("random_das")
    >>> out = patch.add_inventory_coords(inventory, coords=("label", "x", "y", "z"))
    >>> {"label", "x", "y", "z"}.issubset(out.coords.coord_map)
    True
    >>> geo_axes = ("label", "latitude", "longitude", "elevation")
    >>> out = patch.add_inventory_coords(inventory, coords=geo_axes, on_missing="nan")
    >>> {"latitude", "longitude", "elevation"}.issubset(out.coords.coord_map)
    True

    See Also
    --------
    [Inventory tutorial](/tutorial/inventory.qmd)
        Examples of adding annotation labels, geometry axes, and interval metadata.
    [Inventory design note](/notes/inventory_design.qmd)
        Rationale for explicit geometry names and optical path tracks.
    """
    _validate_policy(on_boundary, {"raise", "warn", "ignore"}, "on_boundary")
    _validate_policy(on_missing, {"raise", "nan"}, "on_missing")
    coord_requests = _normalize_coord_requests(coords)
    fiber_array = _get_fiber_array(patch, inventory, data_source_id)
    path = _get_optical_path(fiber_array, _get_patch_time(patch))
    distance, dim = _get_distance_values_and_dim(patch)
    _check_distance_bounds(distance, path)
    warned = [False]
    new_coords = {}
    for request in coord_requests:
        name, values = _project_inventory_coord(
            path,
            distance,
            request,
            on_boundary,
            on_missing,
            warned,
        )
        new_coords[name] = (dim, values)
    return patch.update_coords(**new_coords)


@patch_function()
def add_inventory_attrs(
    patch: PatchType,
    inventory: Inventory,
    attrs: str | Sequence[str],
    data_source_id: str | None = None,
    on_missing: _AttrMissingPolicy = "raise",
) -> PatchType:
    """
    Add patch attrs derived from an inventory fiber array context.

    Parameters
    ----------
    inventory
        Inventory containing the fiber array, acquisition, and optionally linked
        interrogator.
    attrs
        Inventory attrs to add. Simple names must be unambiguous across the
        fiber array graph. Qualified names can use ``fiber_array.``,
        ``acquisition.``, or ``interrogator.`` prefixes.
    data_source_id
        Dot-delimited data source id. Defaults to ``patch.attrs.data_source_id``.
    on_missing
        How to handle missing requested attrs or linked objects. Options are
        ``"raise"`` and ``"ignore"``.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch().update_attrs(
    ...     data_source_id="RD.RDAS..RAW",
    ... )
    >>> inventory = dc.get_example_inventory("random_das")
    >>> out = patch.add_inventory_attrs(
    ...     inventory,
    ...     attrs=("tag", "acquisition_sample_rate", "interrogator.model"),
    ... )
    >>> out.attrs.tag
    'random'
    >>> out.attrs.acquisition_sample_rate
    250.0
    >>> out.attrs.interrogator_model
    'SyntheticInterrogator'

    See Also
    --------
    [Inventory tutorial](/tutorial/inventory.qmd)
        Examples of adding fiber array and interrogator attrs to patches.
    [Inventory design note](/notes/inventory_design.qmd)
        Rationale for fiber array and acquisition relationships.
    """
    _validate_policy(on_missing, {"raise", "ignore"}, "on_missing")
    attr_requests = _normalize_attr_requests(attrs)
    inventory = _validate_inventory(inventory)
    fiber_array = None
    acquisition = None
    has_source = bool(data_source_id or patch.attrs.data_source_id)
    needs_fiber_array = _requests_source(attr_requests, "fiber_array") or has_source
    if needs_fiber_array:
        fiber_array = _get_fiber_array(patch, inventory, data_source_id)
    if (
        _requests_source(attr_requests, "acquisition")
        or _requests_source(attr_requests, "interrogator")
        or has_source
    ):
        acquisition = _get_acquisition(patch, inventory, data_source_id)
    sources = _get_attr_sources(fiber_array, acquisition)
    new_attrs = {}
    for request in attr_requests:
        if resolved := _get_inventory_attr(sources, request, on_missing):
            name, value = resolved
            new_attrs[name] = value
    return patch.update_attrs(**new_attrs)
