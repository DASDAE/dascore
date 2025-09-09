"""Tests for DASDAE format."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.io.dasdae.core import DASDAEV1
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
        """Ensure an emtpy patch can be deserialize."""
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
