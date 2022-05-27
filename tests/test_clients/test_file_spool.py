"""
Tests for FileSpool.
"""
from pathlib import Path
import pytest
import shutil

import pandas as pd
import dascore as dc
from dascore.utils.misc import register_func

FILE_SPOOLS = []


@pytest.fixture(scope="class")
def one_file_dir(tmp_path_factory, random_patch):
    """Create a directory with a single DAS file."""
    out = Path(tmp_path_factory.mktemp("one_file_file_spool"))
    random_patch.io.write(out / "file_1.hdf5", "dasdae")
    return out


@pytest.fixture(scope="class")
@register_func(FILE_SPOOLS)
def one_file_file_spool(one_file_dir):
    """Create a directory with a single DAS file."""
    return dc.FileSpool(one_file_dir).update()


@pytest.fixture(scope="class", params=FILE_SPOOLS)
def file_spool(request):
    """Meta fixture for getting all file spools."""
    return request.getfixturevalue(request.param)


@pytest.fixture(scope="class")
def basic_bank_directory(tmp_path_factory, terra15_das_example_path, random_patch):
    """Create a directory of DAS files for testing."""
    # first copy in a terra15 file
    dir_path = tmp_path_factory.mktemp("bank_basic")
    shutil.copy(terra15_das_example_path, dir_path)
    # save a random patch
    random_patch.io.write(dir_path / "random.hdf5", "dasdae")
    return dir_path


@pytest.fixture(scope="class")
@register_func(FILE_SPOOLS)
def basic_file_spool(basic_bank_directory):
    """Return a DAS bank on basic_bank_directory."""
    out = dc.FileSpool(basic_bank_directory).update()
    return out


class TestFileSpool:
    """Test that the file spool works."""

    def test_isinstance(self, file_spool):
        """Simply ensure expected type was returned."""
        assert isinstance(file_spool, dc.FileSpool)


class TestFileIndex:
    """Tests for returning summaries of all files in path."""

    @pytest.fixture(scope="class")
    def basic_index_df(self, basic_file_spool):
        """Return the index file of the basic bank."""
        spool = basic_file_spool
        return spool.get_contents()

    def test_index_exists(self, basic_file_spool):
        """An index should be returned."""
        assert basic_file_spool.index_path.exists()

    def test_index_len(self, basic_index_df, basic_bank_directory):
        """An index should be returned."""
        bank_paths = list(Path(basic_bank_directory).rglob("*hdf5"))
        assert isinstance(basic_index_df, pd.DataFrame)
        assert len(bank_paths) == len(basic_index_df)

    def test_index_columns(self, basic_index_df):
        """Ensure expected columns show up."""
