"""
Tests specific to DASvader format.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from h5py.h5r import Reference

import dascore as dc
from dascore.exceptions import DependencyError
from dascore.io.dasvader.utils import _dereference, _julia_ms_to_datetime64
from dascore.utils.downloader import fetch


class TestDASVader:
    """Test case for dasvader."""

    @pytest.fixture(scope="session")
    def legacy_das_vader_path(self):
        """Get the legacy DASVader file path."""
        return fetch("das_vader_1.jld2")

    @pytest.fixture(scope="class")
    def das_vader_strainrate_no_attrib_path(self, tmp_path_factory):
        """Create a DASVader file using `strainrate` data and no `atrib` ref."""
        path = (
            tmp_path_factory.mktemp("dasvader_strainrate") / "strainrate_no_atrib.jld2"
        )

        tp_dtype = np.dtype([("hi", "<f8"), ("lo", "<f8")])
        sr_dtype = np.dtype(
            [("ref", tp_dtype), ("step", tp_dtype), ("len", "<i8"), ("offset", "<i8")]
        )
        ddas_dtype = np.dtype(
            [
                ("strainrate", h5py.ref_dtype),
                ("time", sr_dtype),
                ("htime", h5py.ref_dtype),
                ("offset", sr_dtype),
            ]
        )

        with h5py.File(path, "w") as fi:
            ntime, ndistance = 4, 3
            strainrate = fi.create_dataset(
                "strainrate",
                data=np.arange(ntime * ndistance).reshape(ntime, ndistance),
            )
            htime = fi.create_dataset("htime", data=np.array([62_135_683_200_000]))

            ddas = np.zeros((), dtype=ddas_dtype)
            ddas["strainrate"] = strainrate.ref
            ddas["htime"] = htime.ref
            ddas["time"]["ref"]["hi"] = 0.0
            ddas["time"]["step"]["hi"] = 1.0
            ddas["time"]["len"] = ntime
            ddas["time"]["offset"] = 1
            ddas["offset"]["ref"]["hi"] = 0.0
            ddas["offset"]["step"]["hi"] = 1.0
            ddas["offset"]["len"] = ndistance
            ddas["offset"]["offset"] = 1
            fi.create_dataset("dDAS", data=ddas)
        return path

    def test_all_attrs_resolved(self, dasvader_modern_path):
        """Ensure all attributes are resolved (no hdf5 references)"""
        das_vader_patch = dc.read(dasvader_modern_path)[0]
        for _, value in das_vader_patch.attrs.items():
            assert not isinstance(value, Reference)

    def test_legacy_file_raises_clear_error(self, legacy_das_vader_path):
        """Skip on incompatible stacks, otherwise ensure the file still reads."""
        try:
            patch = dc.read(legacy_das_vader_path)[0]
        except DependencyError as exc:
            pytest.skip(str(exc))
        assert patch.attrs is not None

    def test_modern_file_scan_and_slice(self, dasvader_modern_path):
        """Named-reference DASVader files should still support scan and read."""
        scanned = dc.scan(dasvader_modern_path)
        assert len(scanned) == 1
        attrs = scanned[0]
        assert attrs.file_format == "DASVader"
        assert attrs.gauge_length == 10.0
        assert attrs.host_name == "test-host"
        patch = dc.read(
            dasvader_modern_path, time=(attrs.time_min + dc.to_timedelta64(1), ...)
        )[0]
        assert patch.dims == ("distance", "time")
        assert patch.attrs.gauge_length == 10.0

    def test_strainrate_without_atrib(self, das_vader_strainrate_no_attrib_path):
        """Ensure parsing works when data is `strainrate` and `atrib` is absent."""
        patch = dc.read(das_vader_strainrate_no_attrib_path)[0]
        assert patch.dims == ("time", "distance")
        assert "gauge_length" not in patch.attrs.model_dump()
        scanned = dc.scan(das_vader_strainrate_no_attrib_path)
        assert len(scanned) == 1

    def test_julia_ms_to_datetime64_unwraps_nested_void(self):
        """Ensure nested JLD2-style void values are unwrapped."""
        value = np.array(
            [[(62_135_683_200_000,)]],
            dtype=[("instant", [("periods", [("value", "<i8")])])],
        )[0, 0]
        out = _julia_ms_to_datetime64(value)
        assert out == np.datetime64("1970-01-01T00:00:00")

    def test_dereference_returns_non_reference(self):
        """Non-reference values should be returned unchanged."""
        value = np.float64(5_000.0)
        assert _dereference(None, value, "PulseRateFreq") == value
