"""Tests for writing standalone PRODML HDF5 files."""

from __future__ import annotations

from uuid import UUID

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import InvalidSpoolError, PatchError, UnitError
from dascore.io.prodml.core import ProdMLV2_0, ProdMLV2_1
from dascore.units import get_quantity_str
from dascore.utils.misc import suppress_warnings, unbyte


@pytest.fixture
def prodml_patch():
    """Return a small Patch which PRODML can represent without loss."""
    patch = dc.get_example_patch(
        shape=(4, 5),
        distance_min=4,
        distance_step=2,
        time_min=np.datetime64("2020-01-01", "us"),
        time_step=np.timedelta64(2_000, "us"),
        station="station-a",
    )
    data = np.arange(np.prod(patch.shape), dtype=np.int16).reshape(patch.shape)
    return patch.new(data=data).update_attrs(
        acquisition_id="27addb2c-e435-4cd5-81f1-080ed7cc5fd1",
        facility_id="facility-a",
        data_type="strain_rate",
        data_units="m/s",
    )


def _decode(value):
    """Decode one HDF5 string attribute."""
    return str(unbyte(value))


def _with_time(patch, values):
    """Replace the time coordinate, resizing data along time as needed."""
    values = np.asarray(values)
    axis = patch.dims.index("time")
    slices = [slice(None)] * patch.data.ndim
    slices[axis] = slice(0, len(values))
    data = patch.data[tuple(slices)]
    coord = dc.get_coord(data=values, units="s")
    return patch.new(data=data, coords=patch.coords.update(time=coord))


class TestProdMLWriteDispatch:
    """Writing should be exposed only by PRODML 2.1."""

    def test_supported_versions(self):
        """Version 2.1 implements write while version 2.0 remains read-only."""
        assert ProdMLV2_1().implements_write
        assert not ProdMLV2_0().implements_write

    @pytest.mark.parametrize(
        "api", ("dc_default", "dc_v21", "patch_default", "patch_v21")
    )
    def test_write_dispatch(self, api, prodml_patch, tmp_path):
        """Both write APIs should support default and explicit version 2.1."""
        path = tmp_path / f"{api}.h5"
        if api == "dc_default":
            dc.write(prodml_patch, path, "PRODML")
        elif api == "dc_v21":
            dc.write(prodml_patch, path, "PRODML", file_version="2.1")
        elif api == "patch_default":
            prodml_patch.io.write(path, "PRODML")
        else:
            prodml_patch.io.write(path, "PRODML", file_version="2.1")
        assert dc.get_format(path) == ("PRODML", "2.1")


class TestProdMLWriteLayout:
    """The writer should emit the documented standalone HDF5 layout."""

    def test_hdf_layout_and_metadata(self, prodml_patch, tmp_path):
        """Required groups, datasets, attributes, and UUIDs should be present."""
        path = dc.write(prodml_patch, tmp_path / "layout.h5", "PRODML")
        with h5py.File(path, "r") as file:
            assert set(file) == {"Acquisition"}
            acquisition = file["Acquisition"]
            assert set(acquisition) == {"Raw[0]"}
            raw = acquisition["Raw[0]"]
            assert set(raw) == {"RawData", "RawDataTime"}

            acquisition_attrs = {
                "AcquisitionId",
                "FacilityId",
                "MaximumFrequency",
                "MaximumFrequency.uom",
                "MeasurementStartTime",
                "MinimumFrequency",
                "MinimumFrequency.uom",
                "NumberOfLoci",
                "ServiceCompanyName",
                "SpatialSamplingInterval",
                "SpatialSamplingInterval.uom",
                "StartLocusIndex",
                "TriggeredMeasurement",
                "schemaVersion",
                "uuid",
            }
            raw_attrs = {
                "NumberOfLoci",
                "OutputDataRate",
                "OutputDataRate.uom",
                "RawDataUnit",
                "StartLocusIndex",
                "uuid",
            }
            data_attrs = {
                "Count",
                "Dimensions",
                "PartEndTime",
                "PartStartTime",
                "StartIndex",
            }
            time_attrs = {
                "Count",
                "EndTime",
                "PartEndTime",
                "PartStartTime",
                "StartIndex",
                "StartTime",
                "Uom",
            }
            assert acquisition_attrs <= set(acquisition.attrs)
            assert raw_attrs <= set(raw.attrs)
            assert data_attrs <= set(raw["RawData"].attrs)
            assert time_attrs <= set(raw["RawDataTime"].attrs)

            facility = acquisition.attrs["FacilityId"]
            assert isinstance(facility, np.ndarray)
            assert facility.shape == (1,)
            assert facility.dtype.kind == "S"
            assert _decode(facility[0]) == prodml_patch.attrs.facility_id
            assert _decode(acquisition.attrs["AcquisitionId"]) == (
                prodml_patch.attrs.acquisition_id
            )
            assert _decode(raw.attrs["RawDataUnit"]) == get_quantity_str(
                prodml_patch.attrs.data_units
            )

            raw_data = raw["RawData"]
            raw_time = raw["RawDataTime"]
            assert raw_data.shape == prodml_patch.shape
            assert raw_data.dtype == prodml_patch.data.dtype
            assert raw_data.attrs["Count"] == prodml_patch.data.size
            assert [_decode(x) for x in raw_data.attrs["Dimensions"]] == [
                "locus",
                "time",
            ]
            assert raw_time.dtype == np.dtype("int64")
            assert raw_time.attrs["Count"] == len(prodml_patch.get_coord("time"))
            assert _decode(raw_time.attrs["Uom"]) == "us"
            for attrs, names in (
                (acquisition.attrs, ("NumberOfLoci", "StartLocusIndex")),
                (raw.attrs, ("NumberOfLoci", "StartLocusIndex")),
                (raw_data.attrs, ("Count", "StartIndex")),
                (raw_time.attrs, ("Count", "StartIndex")),
            ):
                assert all(np.asarray(attrs[name]).dtype == np.dtype("int64") for name in names)

            uuid_values = (
                _decode(acquisition.attrs["AcquisitionId"]),
                _decode(file.attrs["uuid"]),
                _decode(acquisition.attrs["uuid"]),
                _decode(raw.attrs["uuid"]),
            )
            assert len(set(uuid_values)) == 4
            assert all(str(UUID(value)) == value for value in uuid_values)

    def test_replaces_existing_contents(self, prodml_patch, tmp_path):
        """Writing should replace rather than append to an existing HDF5 file."""
        path = tmp_path / "replace.h5"
        with h5py.File(path, "w") as file:
            file.create_group("sentinel")
            file.attrs["sentinel"] = "old"
        dc.write(prodml_patch, path, "PRODML")
        with h5py.File(path, "r") as file:
            assert set(file) == {"Acquisition"}
            assert "sentinel" not in file.attrs
            assert set(file["Acquisition"]) == {"Raw[0]"}


class TestProdMLWriteRoundTrip:
    """Representable Patch content should survive a write/read cycle."""

    @pytest.mark.parametrize("dims", (("distance", "time"), ("time", "distance")))
    def test_round_trip(self, dims, prodml_patch, tmp_path):
        """Data, dtype, coordinates, dimensions, and core attrs should round trip."""
        patch = prodml_patch.transpose(*dims)
        path = dc.write(patch, tmp_path / "round_trip.h5", "PRODML")
        out = dc.read(path, "PRODML", file_version="2.1")[0]

        with h5py.File(path, "r") as file:
            stored_dims = file["Acquisition/Raw[0]/RawData"].attrs["Dimensions"]
            assert [_decode(x) for x in stored_dims] == [
                "locus" if dim == "distance" else dim for dim in dims
            ]
        assert out.dims == patch.dims
        assert out.data.dtype == patch.data.dtype
        np.testing.assert_array_equal(out.data, patch.data)
        for name in patch.dims:
            np.testing.assert_array_equal(
                out.get_coord(name).values, patch.get_coord(name).values
            )
            assert out.get_coord(name).units == patch.get_coord(name).units
        assert out.attrs.acquisition_id == patch.attrs.acquisition_id
        assert out.attrs.data_units == patch.attrs.data_units

    def test_station_is_facility_fallback(self, prodml_patch, tmp_path):
        """Station should populate FacilityId when no explicit facility attr exists."""
        attrs = prodml_patch.attrs.drop("facility_id")
        patch = prodml_patch.new(attrs=attrs)
        path = dc.write(patch, tmp_path / "station.h5", "PRODML")
        with h5py.File(path, "r") as file:
            facility = file["Acquisition"].attrs["FacilityId"]
            assert _decode(facility[0]) == patch.attrs.station

    def test_missing_metadata_uses_defaults(self, prodml_patch, tmp_path):
        """Required metadata should receive deterministic compatibility defaults."""
        attrs = prodml_patch.attrs.drop("facility_id").update(
            acquisition_id="", station="", data_units=None
        )
        patch = prodml_patch.new(attrs=attrs)
        path = dc.write(patch, tmp_path / "defaults.h5", "PRODML")
        with h5py.File(path, "r") as file:
            acquisition = file["Acquisition"]
            assert _decode(acquisition.attrs["FacilityId"][0]) == "UNKNOWN"
            assert _decode(acquisition.attrs["ServiceCompanyName"]) == "UNKNOWN"
            assert _decode(acquisition["Raw[0]"].attrs["RawDataUnit"]) == "UNKNOWN"
            assert str(UUID(_decode(acquisition.attrs["AcquisitionId"])))
        assert dc.read(path)[0].attrs.data_units is None

    def test_acquisition_id_is_canonicalized(self, prodml_patch, tmp_path):
        """Compact UUID input should be stored in PRODML's canonical form."""
        canonical = prodml_patch.attrs.acquisition_id
        patch = prodml_patch.update_attrs(acquisition_id=canonical.replace("-", ""))
        path = dc.write(patch, tmp_path / "uuid.h5", "PRODML")
        with h5py.File(path, "r") as file:
            assert _decode(file["Acquisition"].attrs["AcquisitionId"]) == canonical

    def test_single_distance_sample(self, prodml_patch, tmp_path):
        """A one-locus Patch remains representable when its step is defined."""
        axis = prodml_patch.dims.index("distance")
        data = np.take(prodml_patch.data, [0], axis=axis)
        distance = dc.get_coord(start=4, stop=6, step=2, units="m")
        coords = prodml_patch.coords.update(distance=distance)
        patch = prodml_patch.new(data=data, coords=coords)
        out = dc.read(dc.write(patch, tmp_path / "one_locus.h5", "PRODML"))[0]
        np.testing.assert_array_equal(out.data, patch.data)
        np.testing.assert_array_equal(
            out.get_coord("distance").values, patch.get_coord("distance").values
        )


class TestProdMLWriteTimePrecision:
    """PRODML timestamp conversion should be explicit and predictable."""

    def test_exact_microseconds_do_not_warn(self, prodml_patch, tmp_path):
        """Exactly representable timestamps should be written silently."""
        with suppress_warnings(UserWarning, action="always", record=True) as record:
            dc.write(prodml_patch, tmp_path / "exact.h5", "PRODML")
        assert not record

    def test_sub_microseconds_round_and_warn_once(self, prodml_patch, tmp_path):
        """Sub-microsecond timestamps should round to nearest microsecond once."""
        base = np.datetime64("2020-01-01", "ns")
        time = (
            base
            + np.timedelta64(600, "ns")
            + np.arange(5) * np.timedelta64(2_000, "us")
        )
        patch = _with_time(prodml_patch, time)
        path = tmp_path / "rounded.h5"
        with pytest.warns(UserWarning, match="microsecond") as record:
            dc.write(patch, path, "PRODML")
        assert len(record) == 1

        expected = (time.astype("datetime64[ns]").astype(np.int64) + 500) // 1_000
        with h5py.File(path, "r") as file:
            stored = file["Acquisition/Raw[0]/RawDataTime"][:]
        np.testing.assert_array_equal(stored, expected)
        out_time = dc.read(path)[0].get_coord("time").values
        np.testing.assert_array_equal(
            out_time, expected.astype("datetime64[us]").astype("datetime64[ns]")
        )

    def test_pre_epoch_ties_round_away_from_epoch(self, prodml_patch, tmp_path):
        """Negative half-microseconds should round away from the Unix epoch."""
        base = np.datetime64("1969-12-31T23:59:59.000000500", "ns")
        time = base + np.arange(5) * np.timedelta64(2_000, "us")
        patch = _with_time(prodml_patch, time)
        path = tmp_path / "pre_epoch.h5"
        with pytest.warns(UserWarning, match="microsecond") as record:
            dc.write(patch, path, "PRODML")
        assert len(record) == 1

        expected = -1_000_000 + np.arange(5, dtype=np.int64) * 2_000
        with h5py.File(path, "r") as file:
            raw = file["Acquisition/Raw[0]"]
            np.testing.assert_array_equal(raw["RawDataTime"][:], expected)
            assert _decode(raw["RawData"].attrs["PartStartTime"]) == (
                "1969-12-31T23:59:59.000000+00:00"
            )


class TestProdMLWriteValidation:
    """Unrepresentable input should fail with a clear DASCore error."""

    def test_one_patch_spool_is_valid(self, prodml_patch, tmp_path):
        """A one-Patch spool is the supported cardinality."""
        path = dc.write(dc.spool([prodml_patch]), tmp_path / "one.h5", "PRODML")
        assert dc.get_format(path) == ("PRODML", "2.1")

    @pytest.mark.parametrize("count", (0, 2))
    def test_bad_spool_cardinality(self, count, prodml_patch, tmp_path):
        """Empty and multi-Patch spools cannot map to one Raw[0]."""
        patches = [
            prodml_patch,
            prodml_patch.update_attrs(tag="distinct-second-patch"),
        ][:count]
        spool = dc.spool(patches)
        with pytest.raises(InvalidSpoolError):
            dc.write(spool, tmp_path / f"spool_{count}.h5", "PRODML")

    @pytest.mark.parametrize("dtype", (bool, np.float16, np.complex64, "U4", object))
    def test_unsupported_dtype(self, dtype, prodml_patch, tmp_path):
        """Only standard integer and floating-point data types are supported."""
        patch = prodml_patch.new(data=np.ones(prodml_patch.shape, dtype=dtype))
        with pytest.raises(PatchError, match=r"dtype|data type"):
            dc.write(patch, tmp_path / "dtype.h5", "PRODML")

    def test_invalid_input_preserves_existing_file(self, prodml_patch, tmp_path):
        """Validation must finish before replacement clears an existing target."""
        path = tmp_path / "preserve.h5"
        with h5py.File(path, "w") as file:
            file.create_group("sentinel")
            file.attrs["sentinel"] = "old"
        patch = prodml_patch.new(data=np.ones(prodml_patch.shape, dtype=bool))
        with pytest.raises(PatchError):
            dc.write(patch, path, "PRODML")
        with h5py.File(path, "r") as file:
            assert set(file) == {"sentinel"}
            assert _decode(file.attrs["sentinel"]) == "old"

    def test_string64_metadata(self, prodml_patch, tmp_path):
        """PRODML String64 attributes cannot exceed their schema limit."""
        patch = prodml_patch.update_attrs(facility_id="x" * 65)
        with pytest.raises(PatchError, match="String64"):
            dc.write(patch, tmp_path / "long_string.h5", "PRODML")

    def test_wrong_dimensions(self, prodml_patch, tmp_path):
        """The dimensions must be exactly time and distance."""
        patch = prodml_patch.rename_coords(distance="channel")
        with pytest.raises(PatchError, match=r"time.*distance|dimensions"):
            dc.write(patch, tmp_path / "dims.h5", "PRODML")

    def test_one_time_sample(self, prodml_patch, tmp_path):
        """At least two time samples are needed to encode a sampling rate."""
        time = prodml_patch.get_coord("time").values[:1]
        patch = _with_time(prodml_patch, time)
        with pytest.raises(PatchError, match=r"two|time"):
            dc.write(patch, tmp_path / "one_time.h5", "PRODML")

    def test_nat_time(self, prodml_patch, tmp_path):
        """NaT cannot be represented by PRODML's integer epoch time array."""
        time = prodml_patch.get_coord("time").values.copy()
        time[2] = np.datetime64("NaT")
        patch = _with_time(prodml_patch, time)
        with pytest.raises(PatchError, match=r"NaT|time"):
            dc.write(patch, tmp_path / "nat.h5", "PRODML")

    def test_time_irregular_after_rounding(self, prodml_patch, tmp_path):
        """Uniform nanoseconds that round to irregular microseconds are invalid."""
        base = np.datetime64("2020-01-01", "ns")
        time = base + np.arange(5) * np.timedelta64(1_500, "ns")
        patch = _with_time(prodml_patch, time)
        with suppress_warnings(UserWarning):
            with pytest.raises(PatchError, match=r"even|regular|microsecond|time"):
                dc.write(patch, tmp_path / "irregular_time.h5", "PRODML")

    def test_distance_units_must_be_length(self, prodml_patch, tmp_path):
        """The implicit locus coordinate requires physical length units."""
        patch = prodml_patch.set_units(distance="s")
        with pytest.raises((PatchError, UnitError), match=r"length|distance|unit"):
            dc.write(patch, tmp_path / "distance_units.h5", "PRODML")

    def test_irregular_distance(self, prodml_patch, tmp_path):
        """Irregular distance cannot be reconstructed from PRODML attributes."""
        distance = dc.get_coord(data=[4.0, 6.0, 9.0, 10.0], units="m")
        patch = prodml_patch.update_coords(distance=distance)
        with pytest.raises(PatchError, match=r"even|regular|distance"):
            dc.write(patch, tmp_path / "irregular_distance.h5", "PRODML")

    def test_nonintegral_start_locus(self, prodml_patch, tmp_path):
        """Distance origin must be an integer multiple of sample spacing."""
        distance = dc.get_coord(data=0.5 + np.arange(4) * 2, units="m")
        patch = prodml_patch.update_coords(distance=distance)
        with pytest.raises(PatchError, match=r"locus|start|distance"):
            dc.write(patch, tmp_path / "start_locus.h5", "PRODML")
