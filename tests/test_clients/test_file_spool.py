"""
Tests for FileSpool.
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.core.schema import PatchFileSummary
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
def basic_spool_directory(tmp_path_factory, terra15_das_example_path, random_patch):
    """Create a directory of DAS files for testing."""
    # first copy in a terra15 file
    dir_path = tmp_path_factory.mktemp("bank_basic")
    shutil.copy(terra15_das_example_path, dir_path)
    # save a random patch
    random_patch.io.write(dir_path / "random.hdf5", "dasdae")
    return dir_path


@pytest.fixture(scope="class")
@register_func(FILE_SPOOLS)
def basic_file_spool(basic_spool_directory):
    """Return a DAS bank on basic_bank_directory."""
    out = dc.FileSpool(basic_spool_directory)
    return out.update()


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

    def test_index_len(self, basic_index_df, basic_spool_directory):
        """An index should be returned."""
        bank_paths = list(Path(basic_spool_directory).rglob("*hdf5"))
        assert isinstance(basic_index_df, pd.DataFrame)
        assert len(bank_paths) == len(basic_index_df)

    def test_index_columns(self, basic_index_df):
        """Ensure expected columns show up in the index."""
        schema_fields = PatchFileSummary.__fields__
        assert set(basic_index_df).issuperset(schema_fields)


class TestSelect:
    """tests for subselecting data."""

    def test_subselect_trims_patches(self, basic_file_spool):
        """Ensure subslecting trims start/end times on df and output patches."""
        current = basic_file_spool.get_contents()
        new_min = current["time_min"].min() + np.timedelta64(2, "s")
        new_max = current["time_max"].max() - np.timedelta64(2, "s")
        spool = basic_file_spool.select(time=(new_min, new_max))
        # the limits of rows which were intersected should have been trimmed.
        df = spool.get_contents()
        assert (df["time_min"] >= new_min).all()
        assert (df["time_max"] <= new_max).all()
        # as well as the patches produced
        for patch in spool:
            assert patch.attrs["time_min"] >= new_min
            assert patch.attrs["time_max"] <= new_max


class TestExtractPatches:
    """Tests for extracting patches."""

    def test_patches_extracted(self, basic_file_spool):
        """Ensure the patches can be extracted."""
        index = basic_file_spool.get_contents()
        patches = [x for x in basic_file_spool]
        assert len(index) == len(patches)
        for patch in patches:
            assert isinstance(patch, dc.Patch)
