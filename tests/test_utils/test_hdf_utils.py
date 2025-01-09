"""Tests for hdf5 utils."""

from __future__ import annotations

from pathlib import Path

import h5py
import pandas as pd
import pytest

import dascore as dc
from dascore.utils.downloader import fetch
from dascore.utils.hdf5 import (
    HDFPatchIndexManager,
    extract_h5_attrs,
    h5_matches_structure,
)


@pytest.fixture(scope="class")
def h5_example_file():
    """Get an example file."""
    path = fetch("gdr_1.h5")
    with h5py.File(path, "r") as fi:
        yield fi
    fi.close()


class TestHDFPatchIndexManager:
    """Tests for the HDF5 index manager."""

    @pytest.fixture
    def index_manager(self, tmp_path_factory):
        """Create a new index."""
        path = Path(tmp_path_factory.mktemp("example")) / ".index"
        return HDFPatchIndexManager(path)

    @pytest.fixture
    def index_manager_with_content(self, index_manager, random_spool):
        """Add content to the index manager."""
        spool_df = dc.scan_to_df(random_spool)
        index_manager.write_update(spool_df)
        return index_manager

    def test_extra_columns(self, index_manager, random_spool):
        """
        Only the columns used for indexing should be kept, extras discarded.

        Here we include a column with types that can't be serialized. If the
        write_update works the test passes.
        """
        df = dc.scan_to_df(random_spool).assign(
            bad_cols=[[] for _ in range(len(random_spool))]
        )
        index_manager.write_update(df)

    def test_empty_tuple(self, index_manager, random_spool):
        """Empty dims should convert to empty string."""
        df = dc.scan_to_df(random_spool).assign(
            dims=[() for _ in range(len(random_spool))],
        )
        index_manager.write_update(df)

    def test_has_content(self, index_manager_with_content, tmp_path):
        """`has_index` should return True if data have been writen else False."""
        assert index_manager_with_content.has_index
        # create hdf5 file with no index
        path = tmp_path / "empty.h5"
        df = pd.DataFrame([1, 2, 3], columns=["first"])
        df.to_hdf(str(path), key="df")
        # assert it doesn't have an index
        assert not HDFPatchIndexManager(path).has_index


class TestH5MatchesStructure:
    """Tests for the h5 matches structure function."""

    def test_has_structure(self, h5_example_file):
        """Ensure the simple structure match works."""
        struct = (
            "DasMetadata",
            "DasMetadata/Cable/Fiber",
            "DasMetadata/Cable/Fiber.FiberComment",
        )
        assert h5_matches_structure(h5_example_file, struct)

    def test_missing_group(self, h5_example_file):
        """Ensure false is returned when a group is missing."""
        struct = ("DasBonkersData",)
        assert not h5_matches_structure(h5_example_file, struct)

    def test_missing_attribute(self, h5_example_file):
        """Ensure false is returned when an attribute is missing."""
        struct = ("DasMetadata/Cable.HasPolkaDots",)
        assert not h5_matches_structure(h5_example_file, struct)


class TestExtractH5Attrs:
    """Extract H5 attributes."""

    def test_extract_existing_attrs(self, h5_example_file):
        """Test extracting existing attributes."""
        acq = "DasMetadata/Interrogator/Acquisition"
        map_names = {
            "DasMetadata.Country": "country",
            f"{acq}.AcquisitionSampleRate": "sample_rate",
        }
        out = extract_h5_attrs(h5_example_file, map_names)
        assert "sample_rate" in out and "country" in out

    def test_bad_attr_raises(self, h5_example_file):
        """A bad attibute should raise a KeyError."""
        map_name = {
            "DasMetadata.CountryBumpkins": "country",
        }
        with pytest.raises(KeyError):
            extract_h5_attrs(h5_example_file, map_name)
