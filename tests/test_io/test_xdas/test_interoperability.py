"""Independent fixtures written and decoded by the installed XDAS release.

Run against a newer upstream checkout by installing it into the test environment.
These tests require XDAS only for producing examples, never for DASCore reads.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.io.xdas.core import XdasV1

xd = pytest.importorskip("xdas", minversion="0.2.9")


@pytest.fixture(
    params=[
        "linear",
        "dense",
        "gapped",
        "sampled",
        "rational",
        "auxiliary",
        "single",
        "unnamed",
        "float32",
    ]
)
def xdas_array(request):
    """Build increasingly complex arrays through XDAS's public constructors."""
    kind = request.param
    size = 1 if kind == "single" else 23
    start = np.datetime64("2025-04-05T12:13:14.987654321", "ns")
    step = np.timedelta64(125, "us")
    times = start + np.arange(size) * step
    time = xd.coordinates.InterpCoordinate.from_block(start, size, step, dim="time")
    coords = {
        "time": time,
        "distance": {"tie_indices": [0, 5], "tie_values": [100.0, 112.5]},
    }
    if kind == "dense":
        coords["time"] = times + np.arange(size) ** 2 * np.timedelta64(1, "ns")
    elif kind == "gapped":
        coords["time"] = {
            "tie_indices": [0, 8, 9, 22],
            "tie_values": times[[0, 8, 9, 22]]
            + np.array([0, 0, 5000, 5000], dtype="timedelta64[us]"),
        }
    elif kind in ("sampled", "rational"):
        rate = (
            {
                "sampling_numerator": np.timedelta64(1_000_000_000, "ns"),
                "sampling_denominator": 1024,
            }
            if kind == "rational"
            else {"sampling_interval": step}
        )
        coords["time"] = xd.coordinates.SampledCoordinate(
            {
                "tie_values": [start, start + np.timedelta64(2, "s")],
                "tie_lengths": [9, 14],
                **rate,
            },
            dim="time",
        )
    elif kind == "float32":
        coords["distance"] = {
            "tie_indices": [0, 5],
            "tie_values": np.array([0.1, 1.0], dtype="float32"),
        }
    elif kind == "auxiliary":
        coords["latitude"] = ("distance", np.linspace(40, 41, 6))
        coords["longitude"] = (
            "distance",
            {"tie_indices": [0, 5], "tie_values": [10.0, 11.0]},
        )
        coords["station"] = "synthetic"
    data = np.arange(size * 6, dtype="float32").reshape(size, 6) / 7
    return xd.DataArray(
        data,
        coords=coords,
        dims=("time", "distance"),
        name=None if kind == "unnamed" else "signal",
        attrs={"tag": kind},
    )


def _assert_matches(patch, reference):
    """Compare data and every coordinate with XDAS's own decoder."""
    assert patch.dims == reference.dims
    np.testing.assert_array_equal(patch.data, np.asarray(reference.data))
    assert set(reference.coords) <= set(patch.coords.coord_map)
    for name, coord in reference.coords.items():
        expected = np.atleast_1d(coord.values)
        actual = patch.get_array(name)
        if np.issubdtype(expected.dtype, np.floating):
            np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-14)
        else:
            np.testing.assert_array_equal(actual, expected)


class TestXdasInteroperability:
    """XDAS itself supplies the files and expected labels, not our decoder."""

    def test_materialized(self, xdas_array, tmp_path):
        """An upstream-written file decodes the same way in both libraries."""
        path = tmp_path / "upstream.nc"
        xdas_array.to_netcdf(path, virtual=False)
        reference = xd.DataArray.from_netcdf(path)
        patch = dc.read(path)[0]
        _assert_matches(patch, reference)
        payload = XdasV1().scan(path, snap=False)[0]
        for dim in patch.dims:
            np.testing.assert_array_equal(
                payload["coords"].get_array(dim), patch.get_array(dim)
            )
        _assert_matches(dc.spool(path)[0], reference)

    def test_nested_collection(self, xdas_array, tmp_path):
        """Array leaves survive repeated names and differing time ranges."""
        path = tmp_path / "collection.nc"
        other = xdas_array.isel(time=slice(0, min(5, xdas_array.shape[0])))
        collection = xd.DataMapping(
            {"a": xdas_array, "nested": xd.DataMapping({"b": other}, name="events")},
            name="network",
        )
        collection.to_netcdf(path, virtual=False)
        patches = {p.attrs["_source_patch_key"]: p for p in dc.read(path)}
        name = xdas_array.name or "__values__"
        assert len(patches) == len(dc.spool(path)) == 2
        _assert_matches(patches[f"/network/a/{name}"], xdas_array)
        _assert_matches(patches[f"/network/nested/events/b/{name}"], other)

    def test_hdf5_virtual(self, xdas_array, tmp_path):
        """XDAS's HDF5 virtual layout reads data from an existing source file."""
        source = tmp_path / "source.nc"
        target = tmp_path / "virtual.nc"
        xdas_array.to_netcdf(source, virtual=False)
        xd.DataArray.from_netcdf(source).to_netcdf(target, virtual=True)
        _assert_matches(dc.read(target)[0], xdas_array)
