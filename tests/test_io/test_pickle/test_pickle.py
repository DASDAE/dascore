"""Tests for reading/writing pickles."""

from __future__ import annotations

import pickle
from io import BytesIO

import numpy as np
import pytest

import dascore as dc
from dascore.io import PatchSource
from dascore.io.pickle.core import PickleIO


@pytest.fixture(scope="session")
def pickle_patch_path(tmp_path_factory, random_patch):
    """Pickle a patch and return the path."""
    path = tmp_path_factory.mktemp("pickle_test") / "test.pkl"
    random_patch.io.write(path, "pickle")
    return path


class TestGetFormat:
    """Test detecting pickle format."""

    def test_detect_file(self, pickle_patch_path):
        """Simple test on output of is pickle."""
        parser = PickleIO()
        out = parser.get_format(pickle_patch_path)
        assert out
        assert out[0] == "PICKLE"

    def test_not_pickle(self, generic_hdf5):
        """Ensure non-pickle file returns false."""
        parser = PickleIO()
        assert not parser.get_format(generic_hdf5)

    def test_spool_from_pickle(self, pickle_patch_path, random_patch):
        """A file-backed pickle spool loads the serialized samples on demand."""
        spool = dc.spool(pickle_patch_path)
        assert len(spool) == 1
        assert len(spool.get_contents()) == 1
        assert spool[0] == random_patch
        assert next(iter(spool)) == random_patch

    def test_file_not_there(self):
        """Get format should return false if the file doesn't exist."""
        parser = PickleIO()
        assert not parser.get_format("surely_not_a_file_that_exists")

    def test_get_format_binary_file_too_small(self):
        """Ensure a file which is too small returns false."""
        bio = BytesIO()
        bio.write(b"one")
        fio = PickleIO()
        bio.seek(0)
        assert not fio.get_format(bio)

    def test_has_dascore_not_pickle(self):
        """Test a buffer which has dascore but isnt a DC pickle."""
        bio = BytesIO()
        bio.write(b"dascore.core Spool")
        fio = PickleIO()
        bio.seek(0)
        assert not fio.get_format(bio)

    def test_bytes_io_valid_pickle(self, random_patch):
        """Test a valid pickle in bytes io."""
        fio = PickleIO()
        bio = BytesIO()
        pickle.dump(random_patch, bio)
        bio.seek(0)
        fmt = fio.get_format(bio)
        assert "PICKLE" in fmt
        spool = fio.read(bio)
        assert spool[0] == random_patch


class TestScan:
    """Tests for scanning pickle files/."""

    comp_attrs = (
        "data_type",
        "data_units",
        "time_step",
        "time_min",
        "time_max",
        "distance_min",
        "distance_max",
        "distance_step",
        "tag",
        "acquisition_key",
    )

    @staticmethod
    def _get_summary_value(summary, attr):
        """Return a comparable summary value from attrs or coord summaries."""
        if attr.startswith("time_"):
            return getattr(
                summary.get_coord_summary("time"), attr.removeprefix("time_")
            )
        if attr.startswith("distance_"):
            return getattr(
                summary.get_coord_summary("distance"), attr.removeprefix("distance_")
            )
        return getattr(summary.attrs, attr)

    def test_scan_attrs_eq_read_attrs(self, pickle_patch_path):
        """Ensure read/scan produce the same attrs."""
        scan_list = dc.scan(pickle_patch_path)
        patch_summaries = [x.summary for x in dc.read(pickle_patch_path)]

        for scan_attrs, patch_summary in zip(scan_list, patch_summaries):
            scan_attrs = scan_attrs.summary
            for attr in self.comp_attrs:
                scan_attr = self._get_summary_value(scan_attrs, attr)
                patch_attr = self._get_summary_value(patch_summary, attr)
                assert scan_attr == patch_attr


class TestSerializedSourceKeys:
    """Existing pickle source keys remain valid for indexed reloads."""

    @pytest.mark.parametrize("legacy", [True, False])
    def test_native_keys_roundtrip(self, tmp_path, random_patch, legacy):
        """Both old attrs and PatchSource keys survive metadata and bounded reads."""
        patches = []
        keys = ["waveforms/first", "waveforms/second"]
        for index, key in enumerate(keys):
            patch = random_patch.new(data=random_patch.data + index)
            if legacy:
                patch = patch.new(attrs=patch.attrs.update(_source_patch_key=key))
            else:
                patch = patch.new(source=PatchSource(key=key))
            patches.append(patch)
        path = tmp_path / "keys.pkl"
        with path.open("wb") as stream:
            pickle.dump(dc.spool(patches), stream)
        summaries = dc.scan(path)
        assert [item.source_patch_key for item in summaries] == keys
        for index, key in enumerate(keys):
            loaded = dc.read(path, source_patch_key=key)[0]
            selected = dc.read(path, source_patch_key=key, time=(1, 4), samples=True)[0]
            expected = patches[index].select(time=(1, 4), samples=True)
            np.testing.assert_array_equal(selected.data, expected.data)
            assert (
                loaded.attrs.patch_id
                == selected.attrs.patch_id
                == summaries[index].attrs.patch_id
            )
            assert "_source_patch_key" not in loaded.attrs.model_dump()
            np.testing.assert_array_equal(
                dc.spool(path)[index].data, patches[index].data
            )
