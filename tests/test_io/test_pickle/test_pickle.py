"""Tests for reading/writing pickles."""

from __future__ import annotations

import gc
import pickle
import weakref
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
        """Legacy native keys survive; current provenance does not name new entries."""
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
        expected_keys = keys if legacy else ["", ""]
        assert [item.source_patch_key for item in summaries] == expected_keys
        read_keys = keys if legacy else ["0", "1"]
        for index, key in enumerate(read_keys):
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

    @pytest.mark.parametrize("legacy", [False, True])
    def test_duplicate_origins_use_positions(self, tmp_path, random_patch, legacy):
        """A patch and a derived copy remain separate pickle entries."""
        patch = random_patch.new(source=PatchSource(key="1"))
        if legacy:
            patch = patch.new(attrs=patch.attrs.update(_source_patch_key="1"))
        other = patch.new(data=patch.data * 2)
        path = tmp_path / "same-origin.pkl"
        dc.write(dc.spool([patch, other]), path, "pickle")
        loaded = dc.read(path)
        indexed = dc.spool(path)
        assert len(loaded) == len(indexed) == 2
        assert [p._source.key for p in loaded] == ["0", "1"]
        assert loaded[0].attrs.patch_id != loaded[1].attrs.patch_id
        for index, expected in enumerate([patch, other]):
            np.testing.assert_array_equal(loaded[index].data, expected.data)
            np.testing.assert_array_equal(indexed[index].data, expected.data)


class TestDecodeLifetime:
    """A pickle operation shares one decode and releases temporary samples."""

    def test_multi_patch_read_decodes_once(self, random_patch, monkeypatch):
        """The number of whole-file decodes does not grow with the patch count."""
        patches = [
            random_patch.new(data=random_patch.data + index) for index in range(4)
        ]
        stream = BytesIO(pickle.dumps(dc.spool(patches)))
        original = pickle.load
        calls = []

        def counted_load(resource):
            calls.append(True)
            return original(resource)

        monkeypatch.setattr(pickle, "load", counted_load)
        out = PickleIO().read(stream, time=(1, 4), samples=True)
        assert len(calls) == 1
        assert not stream.closed
        assert len(out) == len(patches)
        for loaded, patch in zip(out, patches, strict=True):
            np.testing.assert_array_equal(loaded.data, patch.data[:, 1:4])
        stream.seek(0)
        stream.truncate()
        pickle.dump(random_patch.new(data=random_patch.data * 3), stream)
        reloaded = PickleIO().read(stream)[0]
        assert len(calls) == 2
        np.testing.assert_array_equal(reloaded.data, random_patch.data * 3)

    def test_scan_releases_decoded_arrays(self, random_patch, monkeypatch):
        """Returned metadata and caller-owned streams keep no temporary arrays."""
        stream = BytesIO(pickle.dumps(random_patch))
        original = pickle.load
        refs = []

        def tracked_load(resource):
            patch = original(resource)
            refs.append(weakref.ref(patch.data))
            return patch

        monkeypatch.setattr(pickle, "load", tracked_load)
        metadata = PickleIO().get_metadata(stream)
        gc.collect()
        assert len(metadata) == 1
        assert metadata[0]._data is None
        assert not stream.closed
        assert refs and all(ref() is None for ref in refs)
