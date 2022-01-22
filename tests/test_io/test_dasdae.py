"""
Tests for DASDAE format.
"""
from pathlib import Path

import pytest

import dascore as dc
from dascore.io.dasdae import __version__ as DASDAE_file_version
from dascore.io.dasdae.core import DASDAEIO
from dascore.utils.misc import register_func

# a list of fixture names for written DASDAE files
WRITTEN_FILES = []


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_random(random_patch, tmp_path_factory):
    """write the example patch to disk."""
    path = tmp_path_factory.mktemp("dascore_file") / "test.hdf5"
    dc.write(random_patch, path, "dasdae")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dasscore_empty(tmp_path_factory):
    """Write an empty patch to the dascore format."""
    path = tmp_path_factory.mktemp("empty_patcc") / "empty.hdf5"
    patch = dc.Patch()
    dc.write(patch, path, "DASDAE")
    return path


@pytest.fixture(params=WRITTEN_FILES, scope="class")
def dasdae_file_path(request):
    """Gatherer fixture to iterate through each written dasedae format."""
    return request.getfixturevalue(request.param)


class TestWrite:
    """Ensure the format can be written."""

    def test_file_exists(self, dasdae_file_path):
        """The file should *of course* exist."""
        assert Path(dasdae_file_path).exists()


class TestGetVersion:
    """Test for version gathering from files."""

    def test_version_tuple_returned(self, dasdae_file_path):
        """Ensure the expected version str is returned."""
        # format_version_tuple = dc.get_format(written_dascore)
        dasie_format_ver = DASDAEIO().get_format(dasdae_file_path)
        format_ver = dc.get_format(dasdae_file_path)
        expected = ("DASDAE", DASDAE_file_version)
        assert dasie_format_ver == format_ver
        assert format_ver == expected


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

    def test_round_trip_empty_patch(self, written_dasscore_empty):
        """Ensure an emtpy patch can be deserialize."""
        stream = dc.read(written_dasscore_empty)
        assert len(stream) == 1
        assert stream[0].equals(dc.Patch())
