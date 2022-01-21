"""
Tests for DASDAE format.
"""
from pathlib import Path

import pytest
import tables

import dascore as dc
from dascore.io.dasdae import __version__ as DASDAE_file_version


@pytest.fixture(scope="session")
def written_dascore(random_patch, tmp_path_factory):
    """write the example patch to disk."""
    path = tmp_path_factory.mktemp("dascore_file") / "test.hdf5"
    dc.write(random_patch, path, "dasdae")
    return path


class TestWrite:
    """Ensure the format can be written."""

    def test_file_exists(self, written_dascore):
        """The file should *of course* exist."""
        assert Path(written_dascore).exists()

    def test_format_and_version_written(self, written_dascore):
        """Ensure both version and format are in the file."""
        with tables.open_file(written_dascore) as hf:
            attr = hf.root._v_attrs
            assert attr["__DASDAE_version__"] == DASDAE_file_version
            assert attr["__format__"] == "DASDAE"
