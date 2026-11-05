"""Schema validation, coordinates, and partial reads for PUNIU HDF5."""

from decimal import ROUND_HALF_UP, Decimal, InvalidOperation

import h5py
import numpy as np

import dascore as dc
from dascore.exceptions import InvalidFiberFileError, ParameterError

_DATASET = "default"
_REQUIRED_ATTRS = {
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


def _is_puniu(resource):
    """Check the dataset and attribute signature without reading samples."""
    ds = resource.get(_DATASET)
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 2:
        return False
    attrs = ds.attrs
    return (
        _REQUIRED_ATTRS.issubset(attrs)
        and _text(attrs["version"]) == "1"
        and _text(attrs["format"]) == "differential"
    )


def _text(value):
    """Decode a scalar HDF5 string."""
    return value.decode() if isinstance(value, bytes) else str(value)


def _integer(attrs, name):
    """Read an integer attribute without truncating fractional values."""
    value = attrs[name]
    out = int(value)
    if float(value) != out:
        raise InvalidFiberFileError(f"{name} must be an integer")
    return out


def _get_metadata(resource):
    """Build coordinates and acquisition attributes from the dataset header."""
    try:
        ds = resource[_DATASET]
        if not isinstance(ds, h5py.Dataset):
            raise InvalidFiberFileError("Expected /default to be a dataset")
        a = ds.attrs
        if ds.ndim != 2 or ds.dtype.kind not in "if":
            raise InvalidFiberFileError("Expected a numeric two-dimensional dataset")
        if _integer(a, "version") != 1 or _text(a["format"]) != "differential":
            raise InvalidFiberFileError("Expected differential HDF5 version 1")
        order = tuple(x.strip() for x in _text(a["row_major_order"]).split(","))
        if order not in (("time", "channel"), ("channel", "time")):
            raise InvalidFiberFileError(
                "Explicit time/channel row_major_order is required"
            )
        reverse = tuple(x.strip() for x in _text(a["column_major_order"]).split(","))
        if reverse != order[::-1]:
            raise InvalidFiberFileError("Contradictory axis declarations")
        nt = ds.shape[order.index("time")]
        nc = ds.shape[order.index("channel")]
        start = _integer(a, "start_channel")
        step = _integer(a, "step")
        end = _integer(a, "end_channel")
        if nt < 1 or nc < 1 or step <= 0 or _integer(a, "channel_count") != nc:
            raise InvalidFiberFileError("Invalid shape, channel count or step")
        # end_channel bounds the acquisition range;
        # it need not itself be a sampled channel when step > 1.
        if not start + (nc - 1) * step <= end < start + nc * step:
            raise InvalidFiberFileError("Channel range contradicts shape and step")
        fs = float(a["sampling_rate"])
        if not np.isfinite(fs) or fs <= 0:
            raise InvalidFiberFileError("sampling_rate must be finite and positive")
        period = 1e9 / fs
        dt = round(period)
        if dt < 1 or abs(period - dt) > 1e-6:
            raise InvalidFiberFileError(
                "Sampling interval not exactly representable in ns"
            )
        ns = _integer(a, "ns")
        epoch = _integer(a, "epoch")
        if not 0 <= ns < 1_000_000_000:
            raise InvalidFiberFileError("ns must be within one second")
        total = epoch * 1_000_000_000 + ns
        limit = np.iinfo(np.int64)
        if not limit.min < total <= total + (nt - 1) * dt <= limit.max:
            raise InvalidFiberFileError("Time range outside datetime64[ns]")
        base_spacing = float(a["spatial_sampling_rate"])
        if not np.isfinite(base_spacing) or base_spacing <= 0:
            raise InvalidFiberFileError(
                "spatial_sampling_rate must be finite and positive"
            )
        # The acquisition workflow rounds base spacing to 0.1 m before striding.
        dx = (
            float(
                Decimal(str(base_spacing)).quantize(
                    Decimal("0.1"), rounding=ROUND_HALF_UP
                )
            )
            * step
        )
        if not np.isfinite(dx) or dx <= 0:
            raise InvalidFiberFileError(
                "Rounded spatial sampling interval must be positive"
            )
        coords = dict(
            time=dc.get_coord(
                start=np.datetime64(total, "ns"),
                step=np.timedelta64(dt, "ns"),
                shape=nt,
            ),
            distance=dc.get_coord(start=0.0, step=dx, shape=nc, units="m"),
            channel=("distance", dc.get_coord(start=start, step=step, shape=nc)),
        )
        gauge_length = float(a["spatial_resolution"])
        if not np.isfinite(gauge_length) or gauge_length <= 0:
            raise InvalidFiberFileError(
                "spatial_resolution must be finite and positive"
            )
        attrs = dict(
            data_category="DAS",
            sampling_rate=fs,
            gauge_length=gauge_length,
            spatial_sampling_rate=base_spacing,
            channel_stride=step,
            recording_start_channel=start,
        )
        return dc.get_coord_manager(coords, dims=("time", "distance")), attrs, order
    except (KeyError, ValueError, TypeError, OverflowError, InvalidOperation) as exc:
        raise InvalidFiberFileError(f"Invalid Puniu metadata: {exc}") from exc


def _read_data(resource, coords, order, time=None, distance=None, channel=None):
    """Select coordinate intervals before reading the HDF5 hyperslab."""
    selected = coords
    for name, window in (("time", time), ("distance", distance), ("channel", channel)):
        if window is None:
            continue
        if not isinstance(window, tuple | list) or len(window) != 2:
            raise ParameterError(f"{name} must be a two-ended interval")
        selected, _ = selected.select(**{name: tuple(window)})
    if not selected.size:
        return selected, None
    slices = {}
    for name in coords.dims:
        coord = selected.get_coord(name)
        _, slices[name] = coords.get_coord(name).select((coord.min(), coord.max()))
    storage_dims = tuple("distance" if name == "channel" else name for name in order)
    data = resource[_DATASET][tuple(slices[name] for name in storage_dims)]
    if storage_dims != coords.dims:
        data = data.T
    return selected, data
