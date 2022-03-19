"""
Tests for directories of DAS files.
"""
import shutil

import pandas as pd
import pytest

from dascore.io.dasbank import DASBank


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
def basic_das_bank(basic_bank_directory):
    """Return a DAS bank on basic_bank_directory."""
    out = DASBank(basic_bank_directory)
    return out


class TestGetFileIndex:
    """Tests for returning summaries of all files in path."""

    def test_index(self, basic_das_bank):
        """An index should be returned."""
        df = basic_das_bank.contents_to_df(only_new=False)
        assert isinstance(df, pd.DataFrame)
