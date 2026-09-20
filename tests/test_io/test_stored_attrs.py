"""
Reading files whose stored attrs are not the scalars a patch attr holds.

A reader gives the right scalar where the file says one thing; where it
says several, reading still works and the value is skipped with a warning.
Each case here is one a review reproduced against a released DASCore.
"""

from __future__ import annotations

import importlib.util
import pickle
import shutil
import warnings

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.utils.downloader import fetch


def _copy(source, directory, name):
    """Put a copy of a fetched file where a test may edit it."""
    path = directory / name
    shutil.copy2(source, path)
    return path


def _read_scan_and_index(path, tmp_path_factory):
    """Read, scan and index a file, returning the patch and the summary."""
    patch = dc.read(path)[0]
    summaries = dc.scan(path)
    assert len(summaries) == 1
    # A directory spool indexes it rather than silently skipping it.
    directory = tmp_path_factory.mktemp("indexed")
    shutil.copy2(path, directory / path.name)
    spool = dc.spool(directory).update()
    assert len(spool) == 1
    assert isinstance(spool[0], dc.Patch)
    return patch, summaries[0]


@pytest.fixture(scope="module")
def prodml_multi_facility(tmp_path_factory):
    """A ProdML file which names two facilities, as the schema allows."""
    path = _copy(fetch("prodml_2.1.h5"), tmp_path_factory.mktemp("prodml"), "two.h5")
    with h5py.File(path, "a") as h5:
        h5["Acquisition"].attrs["FacilityId"] = np.array([b"well_a", b"well_b"])
    return path


@pytest.fixture(scope="module")
def prodml_array_attr(tmp_path_factory):
    """A ProdML file whose acquisition id was stored as several values."""
    path = _copy(fetch("prodml_2.1.h5"), tmp_path_factory.mktemp("prodml"), "ids.h5")
    with h5py.File(path, "a") as h5:
        h5["Acquisition"].attrs["AcquisitionId"] = np.array([b"one", b"two"])
    return path


@pytest.fixture(scope="module")
def terra15_one_value(tmp_path_factory):
    """A Terra15 file spelling a scalar the way HDF5 usually does."""
    source = fetch("terra15_das_1_trimmed.hdf5")
    path = _copy(source, tmp_path_factory.mktemp("terra15"), "one.hdf5")
    with h5py.File(path, "a") as h5:
        h5.attrs["gauge_length"] = np.array([10.0])
        h5.attrs["serial_number"] = np.array([b"XYZ123"])
    return path


@pytest.fixture(scope="module")
def terra15_array_attr(tmp_path_factory):
    """A Terra15 file whose gauge length was stored as several values."""
    source = fetch("terra15_das_1_trimmed.hdf5")
    path = _copy(source, tmp_path_factory.mktemp("terra15"), "many.hdf5")
    with h5py.File(path, "a") as h5:
        h5.attrs["gauge_length"] = np.array([1.0, 2.0])
    return path


def _write_dashdf5(path, projects, epsg_code):
    """Write a minimal DASHDF5 file carrying the two named attrs."""
    with h5py.File(path, "w") as h5:
        h5.attrs["Conventions"] = np.array(
            ["CF-1.7", "DAS-HDF5-1.0"], dtype=h5py.string_dtype()
        )
        h5.attrs["project"] = np.array(projects, dtype=h5py.string_dtype())
        h5["channel"] = np.arange(4)
        h5["trace"] = np.arange(20)
        h5["t"] = np.arange(20) * 0.001 + 1e9
        for name in "xyz":
            h5[name] = np.arange(4, dtype=float)
            h5[name].attrs["units"] = "m"
        h5["das"] = np.zeros((4, 20), dtype="float32")
        h5["das"].attrs["long_name"] = "strain_rate"
        h5["crs"] = 0
        h5["crs"].attrs["epsg_code"] = epsg_code
    return path


@pytest.fixture(scope="module")
def dashdf5_one_value(tmp_path_factory):
    """A DASHDF5 file whose project and epsg code are length-1 arrays."""
    path = tmp_path_factory.mktemp("dashdf5") / "one_value.h5"
    return _write_dashdf5(path, ["survey"], np.array([4326]))


@pytest.fixture(scope="module")
def dashdf5_array_attr(tmp_path_factory):
    """A DASHDF5 file which names two projects."""
    path = tmp_path_factory.mktemp("dashdf5") / "two_projects.h5"
    return _write_dashdf5(path, ["one", "two"], np.array([4326]))


@pytest.fixture(scope="module")
def pickle_array_attr(tmp_path_factory, random_patch):
    """A pickle of a patch whose attrs hold an array."""
    path = tmp_path_factory.mktemp("pickle") / "legacy.pkl"
    # Put where validation is not: a pickle written by an older DASCore
    # holds whatever that version allowed, and unpickling never validates.
    patch = random_patch.update_attrs(gauge=1.0)
    patch.attrs.__pydantic_extra__["gauge"] = np.array([1.0, 2.0])
    with path.open("wb") as stream:
        pickle.dump(patch, stream)
    return path


class TestReadersGiveScalars:
    """A reader states one value where the file states one value."""

    def test_prodml_names_the_facilities(self, prodml_multi_facility):
        """Several facilities are one name, joined, not a tuple."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            patch = dc.read(prodml_multi_facility)[0]
        assert patch.attrs.facility_id == "well_a,well_b"

    def test_terra15_unwraps_one_value(self, terra15_one_value):
        """A length-1 array is the number or the text it holds."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            patch = dc.read(terra15_one_value)[0]
        assert patch.attrs.gauge_length == 10.0
        assert dict(patch.attrs)["interrogator.serial_number"] == "XYZ123"

    def test_dashdf5_unwraps_one_value(self, dashdf5_one_value):
        """`project` is a name and `epsg_code` a number, not arrays of one."""
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            patch = dc.read(dashdf5_one_value)[0]
        assert patch.attrs.project == "survey"
        assert patch.attrs.epsg_code == 4326


class TestStoredNonScalarAttrsStillRead:
    """A value holding more than one is skipped, and the file still reads."""

    cases = ("prodml_array_attr", "terra15_array_attr", "dashdf5_array_attr")

    @pytest.mark.parametrize("case", cases)
    def test_read_scan_and_index(self, case, request, tmp_path_factory):
        """Reading, scanning and indexing all work, each warning once."""
        path = request.getfixturevalue(case)
        with pytest.warns(UserWarning, match="Attrs hold scalars"):
            patch, summary = _read_scan_and_index(path, tmp_path_factory)
        assert isinstance(patch, dc.Patch)
        assert set(dict(summary.attrs)).issubset(set(dict(patch.attrs)))

    def test_pickle_of_an_unvalidated_patch(self, pickle_array_attr, tmp_path_factory):
        """A pickle keeps what it was given; reading it back does not."""
        with pytest.warns(UserWarning, match="Attrs hold scalars"):
            patch, summary = _read_scan_and_index(pickle_array_attr, tmp_path_factory)
        assert "gauge" not in dict(patch.attrs)
        assert "gauge" not in dict(summary.attrs)


class TestNetCDFAndXarray:
    """CF spells some attrs as arrays, so the conversion reads them too."""

    @pytest.fixture(scope="class")
    def netcdf_array_attr(self, tmp_path_factory, random_patch):
        """A netCDF file with an array-valued attribute beside the data."""
        if not importlib.util.find_spec("xarray"):
            pytest.skip("xarray not installed")
        xr = pytest.importorskip("xarray")
        if not importlib.util.find_spec("h5netcdf"):
            pytest.skip("xarray NetCDF-4 backend not installed")
        path = tmp_path_factory.mktemp("netcdf") / "cf.nc"
        dc.write(random_patch, path, "netcdf_cf")
        with xr.open_dataset(path, engine="h5netcdf") as dataset:
            loaded = dataset.load()
        name = next(iter(loaded.data_vars))
        loaded[name].attrs["valid_range"] = np.array([-1.0, 1.0])
        loaded.to_netcdf(path, engine="h5netcdf", mode="w")
        return path

    def test_read_scan_and_index(self, netcdf_array_attr, tmp_path_factory):
        """An array attribute costs the value, not the file."""
        with pytest.warns(UserWarning, match="Attrs hold scalars"):
            patch, summary = _read_scan_and_index(netcdf_array_attr, tmp_path_factory)
        assert "valid_range" not in dict(patch.attrs)
        assert "valid_range" not in dict(summary.attrs)

    def test_data_array_conversion(self, random_patch):
        """`valid_range` and `flag_values` are arrays by CF convention."""
        pytest.importorskip("xarray")
        # function-level: xarray is an optional dependency
        from dascore.xarray import (  # noqa: PLC0415
            patch_to_xarray,
            xarray_to_patch,
        )

        data_array = patch_to_xarray(random_patch)
        data_array.attrs["valid_range"] = [0.0, 1.0]
        data_array.attrs["epsg_code"] = np.array([4326])
        with pytest.warns(UserWarning, match="Attrs hold scalars"):
            patch = xarray_to_patch(data_array)
        assert "valid_range" not in dict(patch.attrs)
        assert patch.attrs.epsg_code == 4326
