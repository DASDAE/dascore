"""
Tests for DASDAE format.
"""
import shutil
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
from dascore.io.dasdae.core import DASDAEV1
from dascore.utils.misc import register_func
from dascore.utils.time import to_datetime64

# a list of fixture names for written DASDAE files
WRITTEN_FILES = []


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random(random_patch, tmp_path_factory):
    """write the example patch to disk."""
    path = tmp_path_factory.mktemp("dascore_file") / "test.hdf5"
    dc.write(random_patch, path, "dasdae", file_version="1")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random_indexed(written_dascore_v1_random, tmp_path_factory):
    """copy the previous dasdae file and create an index."""
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


@pytest.fixture(params=WRITTEN_FILES, scope="class")
def dasdae_v1_file_path(request):
    """Gatherer fixture to iterate through each written dasedae format."""
    return request.getfixturevalue(request.param)


class TestWrite:
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


class TestGetVersion:
    """Test for version gathering from files."""

    def test_version_tuple_returned(self, dasdae_v1_file_path):
        """Ensure the expected version str is returned."""
        # format_version_tuple = dc.get_format(written_dascore)
        formatter = DASDAEV1()
        ex_format, ex_version = formatter.name, formatter.version
        dasie_format_ver = DASDAEV1().get_format(dasdae_v1_file_path)
        format_ver = dc.get_format(dasdae_v1_file_path)
        assert dasie_format_ver == format_ver
        assert format_ver == (ex_format, ex_version)


class TestReadDasdae:
    """
    Test for reading a dasdae format.
    """

    def test_round_trip_random_patch(self, random_patch, tmp_path_factory):
        """Ensure the random patch can be round-tripped"""
        path = tmp_path_factory.mktemp("dasedae_round_trip") / "rt.h5"
        dc.write(random_patch, path, "DASDAE")
        out = dc.read(path)
        assert len(out) == 1
        assert out[0].equals(random_patch)

    def test_round_trip_empty_patch(self, written_dascore_v1_empty):
        """Ensure an emtpy patch can be deserialize."""
        stream = dc.read(written_dascore_v1_empty)
        assert len(stream) == 1
        stream[0].equals(dc.Patch())

    def test_datetimes(self, tmp_path_factory, random_patch):
        """Ensure the datetimes in the attrs come back as datetimes"""
        # create a patch with a custom dt attribute.
        path = tmp_path_factory.mktemp("dasdae_dt_saes") / "rt.h5"
        dt = np.datetime64("2010-09-12")
        patch = random_patch.update_attrs(custom_dt=dt)
        patch.io.write(path, "dasdae")
        patch_2 = dc.read(path)[0]
        # make sure custom tag with dt comes back from read.
        assert patch_2.attrs["custom_dt"] == dt
        # test coords are still dt64
        assert np.issubdtype(patch_2.coords["time"].dtype, np.datetime64)
        # test attrs
        for name in ("time_min", "time_max"):
            assert isinstance(patch_2.attrs[name], np.datetime64)


class TestScanDasDae:
    """Tests for scanning the dasdae format."""

    def test_scan_returns_info(self, written_dascore_v1_random, random_patch):
        """Ensure scanning returns expected values."""
        info1 = dc.scan(written_dascore_v1_random)[0].dict()
        info2 = dict(random_patch.attrs)
        common_keys = set(info1) & set(info2)
        for key in common_keys:
            assert info1[key] == info2[key]

    def test_scan_format_version(self, written_dascore_v1_random):
        """Ensure scanning returns expected values."""
        formatter = DASDAEV1()
        df = dc.scan_to_df(written_dascore_v1_random)
        assert all(df["file_version"] == formatter.version)
        assert all(df["file_format"] == formatter.name)

    def test_indexed_vs_unindexed(
        self,
        written_dascore_v1_random,
        written_dascore_v1_random_indexed,
    ):
        """Whether the file is indexed or not the summary should be the same."""
        df1 = dc.scan_to_df(written_dascore_v1_random)
        df2 = dc.scan_to_df(written_dascore_v1_random_indexed)
        assert df1.drop(columns="path").equals(df2.drop(columns="path"))
