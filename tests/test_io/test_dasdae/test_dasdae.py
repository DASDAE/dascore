"""Tests for DASDAE format."""

from __future__ import annotations

import json
import shutil
import warnings
from pathlib import Path

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.exceptions import InvalidFileHandlerError
from dascore.io.dasdae.core import DASDAEV1
from dascore.units import get_quantity
from dascore.utils.hdf5 import open_hdf5_file
from dascore.utils.misc import register_func
from dascore.utils.time import to_datetime64

# a list of fixture names for written DASDAE files
WRITTEN_FILES = []


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random(random_patch, tmp_path_factory):
    """Write the example patch to disk."""
    path = tmp_path_factory.mktemp("dasdae_file") / "test.hdf5"
    dc.write(random_patch, path, "dasdae", file_version="1")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random_indexed(written_dascore_v1_random, tmp_path_factory):
    """Copy the previous dasdae file and create an index."""
    new_path = tmp_path_factory.mktemp("dasdae_test_path") / "indexed_dasdae.h5"
    shutil.copy(written_dascore_v1_random, new_path)
    # index new path
    DASDAEV1().index(new_path)
    return new_path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_empty(tmp_path_factory):
    """Write an empty patch to the dascore format."""
    path = tmp_path_factory.mktemp("empty_patcc") / "empty.hdf5"
    patch = dc.Patch()
    dc.write(patch, path, "DASDAE", file_version="1")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_correlate(tmp_path_factory, random_patch):
    """Write a correlate patch to the dascore format."""
    path = tmp_path_factory.mktemp("correlate_patcc") / "correlate.hdf5"
    padded_pa = random_patch.pad(time="correlate")
    dft_pa = padded_pa.dft("time", real=True)
    cc_pa = dft_pa.correlate(distance=[0, 1, 2], samples=True)
    dc.write(cc_pa, path, "DASDAE", file_version="1")
    return path


@pytest.fixture(params=WRITTEN_FILES, scope="class")
def dasdae_v1_file_path(request):
    """Gatherer fixture to iterate through each written dasedae format."""
    return request.getfixturevalue(request.param)


class TestWriteDASDAE:
    """Ensure the format can be written."""

    def test_file_exists(self, dasdae_v1_file_path):
        """The file should *of course* exist."""
        assert Path(dasdae_v1_file_path).exists()

    def test_append(self, written_dascore_v1_random, tmp_path_factory, random_patch):
        """Ensure files can be appended to unindexed dasdae file."""
        # make a copy of the dasdae file.
        new_path = tmp_path_factory.mktemp("dasdae_append") / "tmp.h5"
        shutil.copy(written_dascore_v1_random, new_path)
        # ensure the patch exists in the copied spool.
        df_pre = dc.spool(new_path).get_contents()
        assert len(df_pre) == 1
        # append patch to dasdae file
        new_patch = random_patch.update_attrs(time_min="1990-01-01")
        dc.write(new_patch, new_path, "DASDAE")
        # ensure the file has grown in contents
        df = dc.spool(new_path).get_contents()
        assert len(df) == len(df_pre) + 1
        assert (df["time_min"] == to_datetime64("1990-01-01")).any()

    def test_append_with_index(
        self, written_dascore_v1_random_indexed, tmp_path_factory, random_patch
    ):
        """Ensure patches can be appended to indexed dasdae file."""
        # make a copy of the dasdae file.
        new_path = tmp_path_factory.mktemp("dasdae_append") / "tmp.h5"
        shutil.copy(written_dascore_v1_random_indexed, new_path)
        # ensure the patch exists in the copied spool.
        df_pre = dc.spool(new_path).get_contents()
        assert len(df_pre) == 1
        # append patch to dasdae file
        new_patch = random_patch.update_attrs(time_min="1990-01-01")
        dc.write(new_patch, new_path, "DASDAE")
        # ensure the file has grown in contents
        df = dc.spool(new_path).get_contents()
        assert len(df) == len(df_pre) + 1
        assert (df["time_min"] == to_datetime64("1990-01-01")).any()

    def test_write_again(self, written_dascore_v1_random, random_patch):
        """Ensure a patch can be written again to file (should overwrite old)."""
        random_patch.io.write(written_dascore_v1_random, "dasdae")
        read_patch = dc.spool(written_dascore_v1_random)[0]
        assert random_patch == read_patch

    def test_write_cc_patch(self, written_dascore_correlate):
        """Ensure cross correlated patches can be written and read."""
        sp_cc = dc.spool(written_dascore_correlate)
        assert isinstance(sp_cc[0], dc.Patch)


class TestReadDASDAE:
    """Test for reading a dasdae format."""

    def test_round_trip_random_patch(self, random_patch, tmp_path_factory):
        """Ensure the random patch can be round-tripped."""
        path = tmp_path_factory.mktemp("dasedae_round_trip") / "rt.h5"
        dc.write(random_patch, path, "DASDAE")
        out = dc.read(path)
        assert len(out) == 1
        assert out[0].equals(random_patch)

    def test_round_trip_empty_patch(self, written_dascore_v1_empty):
        """Ensure an empty patch can be deserialized."""
        spool = dc.read(written_dascore_v1_empty)
        assert len(spool) == 1
        spool[0].equals(dc.Patch())

    def test_datetimes(self, tmp_path_factory, random_patch):
        """Ensure the datetimes in the attrs come back as datetimes."""
        # create a patch with a custom dt attribute.
        path = tmp_path_factory.mktemp("dasdae_dt_saes") / "rt.h5"
        dt = np.datetime64("2010-09-12")
        patch = random_patch.update_attrs(custom_dt=dt)
        patch.io.write(path, "dasdae")
        patch_2 = dc.read(path)[0]
        # make sure custom tag with dt comes back from read.
        assert patch_2.attrs["custom_dt"] == dt
        # test coords are still dt64
        array = patch_2.coords.get_array("time")
        assert np.issubdtype(array.dtype, np.datetime64)
        # test attrs
        for name in ("time_min", "time_max"):
            assert isinstance(patch_2.attrs[name], np.datetime64)

    def test_read_file_no_wavegroup(self, generic_hdf5):
        """Ensure an h5 with no wavegroup returns empty patch."""
        parser = DASDAEV1()
        spool = parser.read(generic_hdf5)
        assert not len(spool)

    @pytest.mark.parametrize("method", ["read", "scan"])
    @pytest.mark.parametrize("mode", ["r", "a"])
    def test_open_handle(self, written_dascore_v1_random, method, mode):
        """Existing PyTables handles remain usable and owned by the caller."""
        with open_hdf5_file(written_dascore_v1_random, mode=mode) as handle:
            result = getattr(DASDAEV1(), method)(handle)
            assert len(result) == 1
            assert handle.isopen

    def test_file_spool_loads_distinct_attrs(self, tmp_path, random_patch):
        """Lazy loading should materialize the patch for each DASDAE row."""
        path = tmp_path / "multi_patch.h5"
        patches = [
            random_patch.update_attrs(tag="S100", label="L100"),
            random_patch.update_attrs(tag="S120", label="L120"),
        ]

        dc.write(dc.spool(patches), path, "DASDAE")
        spool = dc.spool(path)

        assert spool.get_contents()["tag"].to_list() == ["S100", "S120"]
        assert [x.attrs.tag for x in spool] == ["S100", "S120"]
        assert [x.attrs.label for x in spool] == ["L100", "L120"]


class TestSeparateMetadata:
    """Read version-1 files written with independent attrs and coordinates."""

    @pytest.fixture(params=["file", "group"])
    def separate_file(self, request, tmp_path):
        """Build the h5py layout used by the Galileo recordings."""
        path = tmp_path / "separate.h5"
        time = dc.to_datetime64("2026-08-06") + np.arange(6) * np.timedelta64(1, "ms")
        patch = dc.Patch(
            data=np.arange(18).reshape(6, 3),
            coords={"time": time, "distance": np.arange(3) * 2.0},
            dims=("time", "distance"),
            attrs={"time_units": "s", "distance_units": "m", "tag": "north"},
        )
        with h5py.File(path, "w") as h5:
            h5.attrs.update(__format__="DASDAE", __DASDAE_version__="1")
            waveforms = h5.create_group("waveforms")
            for tag in ("north", "south"):
                group = waveforms.create_group(tag)
                marked = h5 if request.param == "file" else group
                marked.attrs["__attrs_coords_separate__"] = True
                group.attrs.update(
                    _dims="time,distance",
                    _attrs_tag=tag,
                    _attrs_recorded=time[0].astype("int64"),
                    _attr_type_recorded="datetime64[ns]",
                    _attrs_duration=1000000,
                    _attr_type_duration="timedelta64[ns]",
                    _attrs_optional="",
                    _attr_type_optional="none",
                    _attrs_history=json.dumps(["recorded"]),
                    _attr_type_history="history_json",
                )
                group.create_dataset("data", data=patch.data)
                for name, coord in patch.coords.coord_map.items():
                    is_time = name == "time"
                    node = group.create_dataset(
                        f"_coord_{name}",
                        data=coord.values.astype("int64") if is_time else coord.values,
                    )
                    node.attrs.update(
                        is_datetime64=is_time,
                        is_timedelta64=False,
                        step=coord.step.astype("int64") if is_time else coord.step,
                        step_is_timedelta64=is_time,
                        units=str(coord.units),
                    )
                    group.attrs[f"_cdims_{name}"] = name
        return path, patch

    def test_load(self, separate_file):
        """Direct and lazy reads restore data, coordinate metadata and typed attrs."""
        path, expected = separate_file
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            direct = dc.read(path)[0]
            lazy = dc.spool(path)[0]
        assert not caught
        for patch in (direct, lazy):
            np.testing.assert_array_equal(patch.data, expected.data)
            assert patch.coords == expected.coords
            assert (
                patch.get_array("distance").dtype
                == expected.get_array("distance").dtype
            )
            assert patch.attrs.recorded == expected.attrs.time_min
            assert patch.attrs.duration == np.timedelta64(1, "ms")
            assert patch.attrs.optional is None
            assert patch.attrs.history == ("recorded",)

    def test_select(self, separate_file):
        """Metadata filtering and coordinate selection agree with an in-memory patch."""
        path, expected = separate_file
        limits = (expected.get_array("time")[1], expected.get_array("time")[3])
        spool = dc.spool(path).select(tag="south", time=limits)
        assert len(spool) == 1
        patch = spool[0]
        assert patch.attrs.tag == "south"
        assert patch.coords == expected.select(time=limits).coords
        np.testing.assert_array_equal(patch.data, expected.select(time=limits).data)

    def test_scan(self, separate_file, monkeypatch):
        """Scanning derives coordinate bounds without reading the waveform array."""
        path, expected = separate_file
        original = h5py.Dataset.__getitem__

        def checked_read(dataset, item):
            assert not dataset.name.endswith("/data")
            return original(dataset, item)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_read)
        contents = dc.scan_to_df(path)
        assert contents["tag"].to_list() == ["north", "south"]
        for name in ("time", "distance"):
            for field in ("min", "max", "step", "units"):
                key = f"{name}_{field}"
                values = contents[key]
                if field == "units":
                    values = values.map(get_quantity)
                assert (values == expected.attrs[key]).all()

    def test_filter_before_coords(self, separate_file, monkeypatch):
        """An attribute-filtered read does not load coordinates of rejected patches."""
        path, _ = separate_file
        original = h5py.Dataset.__getitem__

        def checked_read(dataset, item):
            assert "/south/" not in dataset.name
            return original(dataset, item)

        monkeypatch.setattr(h5py.Dataset, "__getitem__", checked_read)
        spool = dc.read(path, tag="north")
        assert len(spool) == 1
        assert spool[0].attrs.tag == "north"

    def test_single_sample_step(self, separate_file):
        """A one-sample time axis needs its stored interval, not an inferred one."""
        path, expected = separate_file
        with h5py.File(path, "a") as h5:
            for group in h5["waveforms"].values():
                node = group["_coord_time"]
                values, metadata = node[:1], dict(node.attrs)
                data = group["data"][:1]
                del group["_coord_time"], group["data"]
                group.create_dataset("_coord_time", data=values).attrs.update(metadata)
                group.create_dataset("data", data=data)
        for patch in (dc.read(path)[0], dc.spool(path)[0]):
            assert patch.shape == (1, 3)
            assert patch.get_coord("time").step == expected.get_coord("time").step

    def test_unitless(self, separate_file):
        """The newer writer omits the units attribute for unitless coordinates."""
        path, _ = separate_file
        with h5py.File(path, "a") as h5:
            for group in h5["waveforms"].values():
                del group["_coord_distance"].attrs["units"]
        for patch in (dc.read(path)[0], dc.spool(path)[0]):
            assert patch.get_coord("distance").units is None

    @pytest.mark.parametrize("method", ["read", "scan"])
    def test_marked_pytables_handle(self, separate_file, method):
        """Reject handles that cannot decode marked metadata without losing it."""
        path, _ = separate_file
        with open_hdf5_file(path) as handle:
            with pytest.raises(InvalidFileHandlerError, match="pass the file path"):
                getattr(DASDAEV1(), method)(handle)
            assert handle.isopen

    def test_mixed(self, separate_file):
        """Appending a legacy patch to a marked file preserves its metadata."""
        path, expected = separate_file
        dc.write(expected.update_attrs(tag="legacy"), path, "DASDAE")
        spool = dc.spool(path)
        assert len(spool) == 3
        assert {patch.attrs.tag for patch in spool} == {"north", "south", "legacy"}
        assert spool.select(tag="legacy")[0].equals(expected.update_attrs(tag="legacy"))

    def test_aux_coords(self, separate_file):
        """Associated timedelta coordinates keep their type and dimensions."""
        path, _ = separate_file
        offsets = np.arange(3).astype("timedelta64[ns]")
        with h5py.File(path, "a") as h5:
            for group in h5["waveforms"].values():
                offset = group.create_dataset(
                    "_coord_offset", data=offsets.astype("int64")
                )
                offset.attrs["is_timedelta64"] = True
                group.attrs["_cdims_offset"] = "distance"
        patch = dc.spool(path)[0]
        np.testing.assert_array_equal(patch.get_array("offset"), offsets)
        assert patch.coords.dim_map["offset"] == ("distance",)

    def test_index(self, separate_file):
        """Embedded indexes preserve distance bounds without enum warnings."""
        path, _ = separate_file
        before = dc.spool(path).select(tag="north", distance=(0, 2))[0]
        dc.spool(path).update()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            after = dc.spool(path).select(tag="north", distance=(0, 2))[0]
        assert not caught
        assert before.equals(after)


class TestScanDASDAE:
    """Tests for scanning the dasdae format."""

    def test_scan_returns_info(self, written_dascore_v1_random, random_patch):
        """Ensure scanning returns expected values."""
        info1 = dc.scan(written_dascore_v1_random)[0].model_dump()
        info2 = random_patch.attrs.model_dump()
        common_keys = set(info1) & set(info2) - {"history"}
        for key in common_keys:
            assert info1[key] == info2[key]

    # TODO we need to re-think indexing before this can work.
    @pytest.mark.xfail
    def test_indexed_vs_unindexed(
        self,
        written_dascore_v1_random,
        written_dascore_v1_random_indexed,
    ):
        """Whether the file is indexed or not the summary should be the same."""
        df1 = dc.scan_to_df(written_dascore_v1_random)
        df2 = dc.scan_to_df(written_dascore_v1_random_indexed)
        # common fields should be equal (except path)
        common = list((set(df1) & set(df2)) - {"path"})
        assert df1[common].equals(df2[common])


class TestRoundTrips:
    """Tests for round-tripping various patches/spools."""

    formatter = DASDAEV1()

    def test_write_patch_with_lat_lon(
        self, random_patch_with_lat_lon, tmp_path_factory
    ):
        """
        DASDAE should support writing patches with non-dimensional
        coords.
        """
        new_path = tmp_path_factory.mktemp("dasdae_append") / "tmp.h5"
        shape = random_patch_with_lat_lon.shape
        dims = random_patch_with_lat_lon.dims
        # add time deltas to ensure they are also serialized/deserialized.
        dist_shape = shape[dims.index("distance")]
        time_deltas = dc.to_timedelta64(random_state.random(dist_shape))
        patch = random_patch_with_lat_lon.update_coords(
            delta_times=("distance", time_deltas),
        )
        dc.write(patch, new_path, "DASDAE")
        spool = dc.read(new_path, file_format="DASDAE")
        assert len(spool) == 1
        new_patch = spool[0]
        assert patch.equals(new_patch)

    def test_roundtrip_empty_time_patch(self, tmp_path_factory, random_patch):
        """A patch with a dimension of length 0 should roundtrip."""
        path = tmp_path_factory.mktemp("round_trip_time_degenerate") / "out.h5"
        patch = random_patch
        # get degenerate patch
        time = patch.get_coord("time")
        time_max = time.max() + 3 * time.step
        empty_patch = patch.select(time=(time_max, ...))
        empty_patch.io.write(path, "dasdae")
        spool = self.formatter.read(path)
        new_patch = spool[0]
        assert empty_patch.equals(new_patch)

    def test_roundtrip_dim_1_patch(self, tmp_path_factory, random_patch):
        """A patch with length 1 time axis should roundtrip."""
        path = tmp_path_factory.mktemp("round_trip_dim_1") / "out.h5"
        patch = dc.get_example_patch(
            "random_das",
            time_step=0.999767552,
            shape=(100, 1),
            time_min="2023-06-13T15:38:00.49953408",
        )
        patch.io.write(path, "dasdae")

        spool = self.formatter.read(path)
        new_patch = spool[0]
        assert patch.equals(new_patch)

    def test_roundtrip_datetime_coord(self, tmp_path_factory, random_patch):
        """Ensure a patch with an attached datetime coord works."""
        path = tmp_path_factory.mktemp("roundtrip_datetme_coord") / "out.h5"
        dist = random_patch.get_coord("distance")
        dt = dc.to_datetime64(np.zeros_like(dist))
        dt[0] = dc.to_datetime64("2017-09-17")
        new = random_patch.update_coords(dt=("distance", dt))
        new.io.write(path, "dasdae")
        patch = dc.spool(path, file_format="DASDAE")[0]
        assert isinstance(patch, dc.Patch)

    def test_roundtrip_nullish_datetime_coord(self, tmp_path_factory, random_patch):
        """Ensure a patch with an attached datetime coord with nulls works."""
        path = tmp_path_factory.mktemp("roundtrip_datetime_coord") / "out.h5"
        dist = random_patch.get_coord("distance")
        dt = dc.to_datetime64(np.zeros_like(dist))
        dt[~dt.astype(bool)] = np.datetime64("nat")
        dt[0] = dc.to_datetime64("2017-09-17")
        dt[-4] = dc.to_datetime64("2020-01-03")
        new = random_patch.update_coords(dt=("distance", dt))
        new.io.write(path, "dasdae")
        patch = dc.spool(path, file_format="DASDAE")[0]
        assert isinstance(patch, dc.Patch)

    def test_roundtrip_coord_multiple_dims(
        self, tmp_path_factory, multi_dim_coords_patch
    ):
        """
        Ensure a patch with a non-dimensional coordinate that is associated
        with two dims can round-trip.
        """
        patch = multi_dim_coords_patch
        folder = tmp_path_factory.mktemp("dasdae_multi_dim_coord")
        path = folder / "multidimcoord.hdf"
        patch.io.write(path, "dasdae")

        # Ensure we can read it from a directory
        patch2 = dc.spool(folder).update()[0]
        # And from a single file
        patch3 = dc.spool(path)[0]
        # All of the patches should be equal.
        assert patch == patch2 == patch3

    # Frustratingly, it doesn't seem pytables can store NaN values using
    # create_array, even when specifying an Atom with dflt=np.nan. See
    # https://github.com/PyTables/PyTables/issues/423
    @pytest.mark.xfail(reason="Pytables issue 423")
    def test_roundtrip_len_1_non_coord(self, random_spool, tmp_path_factory):
        """Ensure we can round-trip Non-coords."""
        path = tmp_path_factory.mktemp("roundtrip_non_coord") / "out.h5"
        # create a spool that has all non coords
        spool = dc.spool([x.mean("time") for x in random_spool])
        in_patch = spool[0]
        in_patch.io.write(path, "dasdae")
        new_spool = dc.spool(path, file_format="DASDAE")
        out_patch = new_spool[0]
        assert in_patch == out_patch
