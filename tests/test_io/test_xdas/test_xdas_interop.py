"""Check the XDAS reader against files the XDAS package writes, if installed."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc

xd = pytest.importorskip("xdas")

START = np.datetime64("2025-04-05T12:13:14.987654321", "ns")
STEP = np.timedelta64(125, "us")


def _array(kind):
    """Return a small array with the named kind of time coordinate."""
    times = START + np.arange(23) * STEP
    time = xd.coordinates.InterpCoordinate.from_block(START, 23, STEP, dim="time")
    if kind == "dense":
        time = times + np.arange(23) ** 2 * np.timedelta64(1, "ns")
    elif kind == "gapped":
        shift = np.array([0, 0, 5000, 5000], dtype="timedelta64[us]")
        time = {
            "tie_indices": [0, 8, 9, 22],
            "tie_values": times[[0, 8, 9, 22]] + shift,
        }
    elif kind in ("sampled", "rational"):
        rate = {"sampling_interval": STEP}
        if kind == "rational":
            rate = {
                "sampling_numerator": np.timedelta64(1_000_000_000, "ns"),
                "sampling_denominator": 1024,
            }
        data = {"tie_values": [START, START + np.timedelta64(2, "s")]}
        data |= {"tie_lengths": [9, 14], **rate}
        time = xd.coordinates.SampledCoordinate(data, dim="time")
    coords = {
        "time": time,
        "distance": {"tie_indices": [0, 5], "tie_values": [100.0, 112.5]},
    }
    data = np.arange(23 * 6, dtype="float32").reshape(23, 6) / 7
    return xd.DataArray(data, coords=coords, dims=("time", "distance"), name="sig")


@pytest.mark.parametrize("kind", ["linear", "dense", "gapped", "sampled", "rational"])
def test_matches_xdas(tmp_path, kind):
    """Data and labels match what XDAS itself decodes from the file."""
    path = tmp_path / f"{kind}.nc"
    _array(kind).to_netcdf(path, virtual=False)
    expected = xd.DataArray.from_netcdf(path)
    spool = dc.read(path, snap=False)
    data = np.concatenate([x.data for x in spool])
    time = np.concatenate([x.get_array("time") for x in spool])
    np.testing.assert_array_equal(data, expected.values)
    np.testing.assert_allclose(
        spool[0].get_array("distance"), expected.coords["distance"].values
    )
    # Exact fractional grids floor to the nanosecond; XDAS rounds.
    diff = np.abs((time - expected.coords["time"].values).astype(np.int64))
    assert diff.max() <= (kind == "rational")
