"""Schema checks and metadata for PUNIU HDF5 files."""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import h5py
import numpy as np

import dascore as dc
from dascore.exceptions import InvalidFiberFileError
from dascore.io.utils import step_from_rate

_DATASET = "default"
_REQUIRED_ATTRS = frozenset(
    {
        "version",
        "format",
        "row_major_order",
        "column_major_order",
        "channel_count",
        "start_channel",
        "end_channel",
        "step",
        "sampling_rate",
        "epoch",
        "ns",
        "spatial_sampling_rate",
        "spatial_resolution",
    }
)
_ORDERS = (("time", "channel"), ("channel", "time"))
_NS_PER_S = 1_000_000_000


def _text(value) -> str:
    """Decode a scalar HDF5 string."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _scalar(value):
    """Return the one value an attribute holds, unwrapping 1-element arrays."""
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"expected a scalar, got {value!r}")
    item = array.reshape(()).item()
    return _text(item) if isinstance(item, bytes) else item


def _version(value) -> float | None:
    """Return the schema version as a number, or None if it is not one."""
    try:
        return float(_scalar(value))
    except (ValueError, TypeError):
        return None


def _is_puniu(resource) -> bool:
    """Check the dataset and attribute signature without reading samples."""
    dataset = resource.get(_DATASET)
    if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2:
        return False
    attrs = dataset.attrs
    return (
        _REQUIRED_ATTRS.issubset(attrs)
        and _version(attrs["version"]) == 1
        and _text(attrs["format"]) == "differential"
    )


def _integer(attrs, name) -> int:
    """Read an integer attribute, refusing fractional values."""
    value = _scalar(attrs[name])
    out = int(value)
    if float(value) != out:
        raise InvalidFiberFileError(f"{name} must be an integer")
    return out


def _positive(attrs, name) -> float:
    """Read a finite, positive float attribute."""
    value = float(_scalar(attrs[name]))
    if not np.isfinite(value) or value <= 0:
        raise InvalidFiberFileError(f"{name} must be finite and positive")
    return value


def _axis_order(attrs) -> bool:
    """Return True if the file stores (channel, time) rather than (time, channel)."""
    order = tuple(x.strip() for x in _text(attrs["row_major_order"]).split(","))
    if order not in _ORDERS:
        raise InvalidFiberFileError("row_major_order must be time/channel")
    reverse = tuple(x.strip() for x in _text(attrs["column_major_order"]).split(","))
    if reverse != order[::-1]:
        raise InvalidFiberFileError("row_major_order and column_major_order disagree")
    return order == ("channel", "time")


def _channel_spacing(attrs) -> Decimal:
    """Return the base channel spacing, rounded half-up to 0.1 m."""
    base = Decimal(str(_positive(attrs, "spatial_sampling_rate")))
    out = base.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    if out <= 0:
        raise InvalidFiberFileError("spatial_sampling_rate rounds to zero")
    return out


def _time_coord(attrs, count):
    """Build the time coordinate from epoch seconds, nanoseconds and rate."""
    rate = _positive(attrs, "sampling_rate")
    ns = _integer(attrs, "ns")
    if not 0 <= ns < _NS_PER_S:
        raise InvalidFiberFileError("ns must be within one second")
    start = _integer(attrs, "epoch") * _NS_PER_S + ns
    end = start + (count - 1) * _NS_PER_S / rate
    limit = np.iinfo(np.int64)
    if not limit.min < start <= end < limit.max:
        raise InvalidFiberFileError("Time range is outside datetime64[ns]")
    return dc.get_coord(
        start=np.datetime64(start, "ns"), step=step_from_rate(rate), shape=(count,)
    )


def _get_metadata(resource):
    """
    Return the coordinates, attrs, and whether storage is (channel, time).

    Only the dataset header is read.
    """
    try:
        dataset = resource[_DATASET]
        if not isinstance(dataset, h5py.Dataset):
            raise InvalidFiberFileError("Expected /default to be a dataset")
        attrs = dataset.attrs
        if dataset.ndim != 2 or dataset.dtype.kind not in "iuf":
            raise InvalidFiberFileError("Expected a numeric two-dimensional dataset")
        if _version(attrs["version"]) != 1 or _text(attrs["format"]) != "differential":
            raise InvalidFiberFileError("Expected differential HDF5 version 1")
        transposed = _axis_order(attrs)
        nt, nc = dataset.shape[::-1] if transposed else dataset.shape
        start = _integer(attrs, "start_channel")
        step = _integer(attrs, "step")
        end = _integer(attrs, "end_channel")
        if nt < 1 or nc < 1 or step <= 0 or _integer(attrs, "channel_count") != nc:
            raise InvalidFiberFileError("Invalid shape, channel count or step")
        # end_channel bounds the acquisition range; with step > 1 it need not
        # be a sampled channel itself.
        if not start + (nc - 1) * step <= end < start + nc * step:
            raise InvalidFiberFileError("Channel range contradicts shape and step")
        spacing = _channel_spacing(attrs)
        distance = dc.get_coord(
            start=float(spacing * start),
            step=float(spacing * step),
            shape=(nc,),
            units="m",
        )
        coords = {
            "time": _time_coord(attrs, nt),
            "distance": distance,
            "channel": ("distance", dc.get_coord(start=start, step=step, shape=nc)),
        }
        out_attrs = {
            "data_category": "DAS",
            # The acquisition reader treats spatial_resolution as gauge length.
            "gauge_length": _positive(attrs, "spatial_resolution"),
        }
    except (KeyError, ValueError, TypeError, OverflowError, InvalidOperation) as exc:
        raise InvalidFiberFileError(f"Invalid PUNIU metadata: {exc}") from exc
    cm = dc.get_coord_manager(coords, dims=("time", "distance"))
    return cm, out_attrs, transposed
