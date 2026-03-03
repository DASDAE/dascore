"""
Tests specific to DASvader format.
"""

import h5py
import numpy as np
import pytest
from h5py.h5r import Reference

import dascore as dc


class TestDASVader:
    """Test case for dasvader."""

    @pytest.fixture(scope="session")
    def das_vader_patch(self):
        """Get the dasvader patch."""
        return dc.get_example_patch("das_vader_1.jld2")

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

    def test_all_attrs_resolved(self, das_vader_patch):
        """Ensure all attributes are resolved (no hdf5 references)"""
        for attr, value in das_vader_patch.attrs.items():
            assert not isinstance(value, Reference)

    def test_strainrate_without_atrib(self, das_vader_strainrate_no_attrib_path):
        """Ensure parsing works when data is `strainrate` and `atrib` is absent."""
        patch = dc.read(das_vader_strainrate_no_attrib_path)[0]
        assert patch.dims == ("time", "distance")
        assert "gauge_length" not in patch.attrs.model_dump()
        scanned = dc.scan(das_vader_strainrate_no_attrib_path)
        assert len(scanned) == 1
