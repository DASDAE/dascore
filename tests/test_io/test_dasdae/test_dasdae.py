"""Tests for DASDAE format."""

from __future__ import annotations

import pickle
import shutil
from pathlib import Path
from typing import ClassVar, Literal

import h5py
import numpy as np
import pandas as pd
import pytest

import dascore as dc
from dascore.compat import random_state
from dascore.config import config_context
from dascore.core.coords import CoordString
from dascore.exceptions import InvalidFiberFileError
from dascore.io import dasdae as dasdae_mod
from dascore.io.core import BaseCodec
from dascore.io.dasdae._compat import translate_legacy_attrs
from dascore.io.dasdae.core import DASDAEV1
from dascore.io.dasdae.storage import DASDAEStorage
from dascore.io.dasdae.utils import (
    _decode_attr_value,
    _decode_legacy_attr_value,
    _encode_attr_value,
    _get_attrs,
    _get_contents_from_patch_groups_generic,
    _get_coords,
    _get_file_version,
    _get_scan_payload_from_group,
    _read_array,
    _save_array,
    _save_patch,
)
from dascore.io.hdf5 import Gzip
from dascore.utils.downloader import fetch
from dascore.utils.misc import register_func
from dascore.utils.time import to_datetime64

# a list of fixture names for written DASDAE files
WRITTEN_FILES = []


class _UnsupportedCodec(BaseCodec):
    """A non-HDF5 codec used to test DASDAE storage validation."""

    name: Literal["unsupported"] = "unsupported"


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random(random_patch, tmp_path_factory):
    """Write the example patch to disk."""
    path = tmp_path_factory.mktemp("dasdae_file") / "test.hdf5"
    dc.write(random_patch, path, "dasdae", file_version="1")
    return path


@pytest.fixture(scope="class")
@register_func(WRITTEN_FILES)
def written_dascore_v1_random_copy(written_dascore_v1_random, tmp_path_factory):
    """Copy the previous DASDAE file for compatibility-oriented tests."""
    new_path = tmp_path_factory.mktemp("dasdae_test_path") / "copied_dasdae.h5"
    shutil.copy(written_dascore_v1_random, new_path)
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

    def _assert_data_compression(self, path, *, compression, level=None):
        """Assert the stored DASDAE data array has the expected compression."""
        with h5py.File(path) as h5:
            group = next(iter(h5["waveforms"].values()))
            assert group["data"].compression == compression
            if level is not None:
                assert group["data"].compression_opts == level

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
        new_patch = random_patch.update_coords(time_min="1990-01-01")
        dc.write(new_patch, new_path, "DASDAE")
        # ensure the file has grown in contents
        df = dc.spool(new_path).get_contents()
        assert len(df) == len(df_pre) + 1
        assert (df["time_min"] == to_datetime64("1990-01-01")).any()

    def test_append_after_copy(
        self, written_dascore_v1_random_copy, tmp_path_factory, random_patch
    ):
        """Ensure append still works on a copied DASDAE file."""
        # make a copy of the dasdae file.
        new_path = tmp_path_factory.mktemp("dasdae_append") / "tmp.h5"
        shutil.copy(written_dascore_v1_random_copy, new_path)
        # ensure the patch exists in the copied spool.
        df_pre = dc.spool(new_path).get_contents()
        assert len(df_pre) == 1
        # append patch to dasdae file
        new_patch = random_patch.update_coords(time_min="1990-01-01")
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

    def test_write_compressed_with_storage(self, tmp_path_factory, random_patch):
        """DASDAE can write compressed arrays with a storage object."""
        path = tmp_path_factory.mktemp("dasdae_compressed") / "compressed.h5"
        storage = DASDAEStorage(codec=Gzip(level=5))
        dc.write(random_patch, path, "DASDAE", storage=storage)
        self._assert_data_compression(path, compression="gzip", level=5)
        assert dc.read(path)[0].equals(random_patch)

    def test_compressed_file_is_smaller(self, tmp_path_factory, random_patch):
        """Compression must actually shrink a compressible file on disk."""
        base = tmp_path_factory.mktemp("dasdae_size_check")
        # Constant data compresses extremely well, so any real compression
        # must produce a much smaller file than the uncompressed write.
        patch = random_patch.new(data=np.ones_like(random_patch.data))
        patch.io.write(base / "plain.h5", "DASDAE")
        patch.io.write(base / "compressed.h5", "DASDAE", storage="compressed")
        plain = (base / "plain.h5").stat().st_size
        compressed = (base / "compressed.h5").stat().st_size
        assert compressed < plain / 2

    def test_direct_write_coerces_storage(self, tmp_path_factory, random_patch):
        """Direct DASDAEV1 writes accept the same storage shorthand as dc.write."""
        path = tmp_path_factory.mktemp("dasdae_direct_storage") / "compressed.h5"
        with h5py.File(path, mode="w") as h5:
            DASDAEV1().write(random_patch, h5, storage="compressed")
        self._assert_data_compression(path, compression="gzip", level=5)
        assert dc.read(path)[0].equals(random_patch)

    def test_default_storage_skips_chunk_validation(
        self, tmp_path_factory, random_patch, monkeypatch
    ):
        """Unchunked writes do not scan dims only to no-op validation."""
        path = tmp_path_factory.mktemp("dasdae_no_chunk_validation") / "out.h5"

        def raise_if_called(*args, **kwargs):
            raise AssertionError("chunk validation should be skipped")

        monkeypatch.setattr(DASDAEStorage, "_validate_chunk_dims", raise_if_called)
        random_patch.io.write(path, "DASDAE")
        assert dc.read(path)[0].equals(random_patch)

    def test_compressed_selective_read(self, tmp_path_factory, random_patch):
        """Selecting from a compressed file must match selecting in memory."""
        path = tmp_path_factory.mktemp("dasdae_compressed_select") / "out.h5"
        random_patch.io.write(path, "DASDAE", storage="compressed")
        dist = random_patch.get_coord("distance").values
        select = {"distance": (dist[0], dist[len(dist) // 2])}
        from_disk = dc.spool(path).select(**select)[0]
        in_memory = random_patch.select(**select)
        assert from_disk.equals(in_memory)

    def test_write_compressed_empty_coord(self, tmp_path_factory, random_patch):
        """Compressed DASDAE writes preserve zero-length arrays."""
        path = tmp_path_factory.mktemp("dasdae_compressed_empty") / "out.h5"
        time = random_patch.get_coord("time")
        empty_patch = random_patch.select(time=(time.max() + 3 * time.step, ...))
        empty_patch.io.write(path, "dasdae", storage="compressed")
        new_patch = dc.read(path)[0]
        assert empty_patch.equals(new_patch)

    def test_compressed_scalar_array_falls_back(self, tmp_path_factory):
        """DASDAE compression skips scalar arrays HDF5 cannot chunk."""
        path = tmp_path_factory.mktemp("dasdae_compressed_scalar") / "out.h5"
        options = DASDAEStorage.from_preset("compressed")._dataset_options((), ())
        with h5py.File(path, mode="w") as h5:
            node = _save_array(np.array(1), "scalar", h5, options)
            assert node.shape == ()
            assert node.compression is None

    def test_write_dict_codec_coercion(self, tmp_path_factory, random_patch):
        """A dict/string codec shorthand is coerced by the write path."""
        path = tmp_path_factory.mktemp("dasdae_gzip") / "gzip.h5"
        random_patch.io.write(path, "DASDAE", storage={"codec": "gzip"})
        self._assert_data_compression(path, compression="gzip")
        assert dc.read(path)[0].equals(random_patch)

    def test_write_compressed_with_gzip(self, tmp_path_factory, random_patch):
        """DASDAE can use portable gzip HDF5 compression."""
        path = tmp_path_factory.mktemp("dasdae_gzip") / "gzip.h5"
        random_patch.io.write(path, "DASDAE", storage=DASDAEStorage(codec=Gzip()))
        self._assert_data_compression(path, compression="gzip")
        assert dc.read(path)[0].equals(random_patch)

    def test_chunked_unicode_array_roundtrips(self, tmp_path_factory):
        """Chunked/compressed string arrays preserve non-ASCII values."""
        path = tmp_path_factory.mktemp("dasdae_unicode_array") / "out.h5"
        data = np.array(["cafe", "café"], dtype="U8")
        storage = DASDAEStorage.from_preset("compressed")
        options = storage._dataset_options(("label",), data.shape) | {"chunks": (1,)}
        with h5py.File(path, mode="w") as h5:
            node = _save_array(data, "labels", h5, options)
            assert node[:].dtype.kind == "S"
            assert np.array_equal(_read_array(node), data)

    def test_unicode_patch_data_roundtrips(self, tmp_path_factory):
        """Compressed patches whose data is a unicode array round-trip exactly."""
        path = tmp_path_factory.mktemp("dasdae_unicode_data") / "out.h5"
        data = np.array([["café", "naïve"], ["日本", "ok"]], dtype="U8")
        patch = dc.Patch(
            data=data,
            coords={"distance": [0.0, 1.0], "time": [0.0, 1.0]},
            dims=("distance", "time"),
        )
        patch.io.write(path, "dasdae", storage="compressed")
        loaded = dc.read(path)[0]
        assert loaded.data.dtype.kind == "U"
        assert np.array_equal(loaded.data, data)
        # The selective-read branch decodes through the same attr-aware path.
        selected = dc.read(path, distance=(0.0, 0.0))[0]
        assert selected.data.dtype.kind == "U"
        assert np.array_equal(selected.data, data[:1])

    def test_datetime_patch_data_selective_read(self, tmp_path_factory):
        """Datetime data arrays keep their dtype through selective reads."""
        path = tmp_path_factory.mktemp("dasdae_datetime_data") / "out.h5"
        data = to_datetime64(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04"])
        patch = dc.Patch(
            data=data.reshape(2, 2),
            coords={"distance": [0.0, 1.0], "time": [0.0, 1.0]},
            dims=("distance", "time"),
        )
        patch.io.write(path, "dasdae", storage="compressed")
        selected = dc.read(path, distance=(0.0, 0.0))[0]
        assert np.issubdtype(selected.data.dtype, np.datetime64)
        assert np.array_equal(selected.data, data.reshape(2, 2)[:1])

    def test_write_with_chunks(self, tmp_path_factory, random_patch):
        """DASDAE storage can specify a per-dimension chunk layout."""
        path = tmp_path_factory.mktemp("dasdae_chunkshape") / "chunked.h5"
        storage = DASDAEStorage(codec=Gzip(), chunks={"distance": 10, "time": 10})
        random_patch.io.write(path, "DASDAE", storage=storage)
        with h5py.File(path) as h5:
            group = next(iter(h5["waveforms"].values()))
            assert group["data"].chunks == (10, 10)
            # Coordinate arrays share the layout, chunked by their dim name.
            assert group["_coord_distance"].chunks == (10,)
            assert group["_coord_distance"].compression == "gzip"
        assert dc.read(path)[0].equals(random_patch)

    def test_explicit_storage_none_writes_default(self, tmp_path_factory, random_patch):
        """An explicit storage=None is normalized to the default and round-trips."""
        path = tmp_path_factory.mktemp("dasdae_storage_none") / "out.h5"
        dc.write(random_patch, path, "DASDAE", storage=None)
        with h5py.File(path) as h5:
            group = next(iter(h5["waveforms"].values()))
            # Default storage is uncompressed.
            assert group["data"].compression is None
        assert dc.read(path)[0].equals(random_patch)

    def test_chunk_validation_does_not_load_patches(
        self, tmp_path_factory, random_patch, monkeypatch
    ):
        """Chunk-dim validation must use scan metadata, not load patch data."""
        src = tmp_path_factory.mktemp("dasdae_lazy_src") / "src.h5"
        out = tmp_path_factory.mktemp("dasdae_lazy_out") / "out.h5"
        dc.write(random_patch, src, "DASDAE")
        lazy_spool = dc.spool(src)

        calls = []
        original = dasdae_mod.core._read_patch

        def counting_read_patch(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        # Patch the name bound in core, which both read paths call through.
        monkeypatch.setattr(dasdae_mod.core, "_read_patch", counting_read_patch)
        dc.write(lazy_spool, out, "DASDAE", storage={"chunks": {"time": 100}})
        # One read per patch for the write itself; validation adds none.
        assert len(calls) == 1
        assert dc.read(out)[0].equals(random_patch)

    def test_typoed_chunk_dim_raises_on_write(self, tmp_path_factory, random_patch):
        """A chunk dim that isn't a real patch dim raises instead of no-op."""
        path = tmp_path_factory.mktemp("dasdae_chunk_typo") / "out.h5"
        with pytest.raises(ValueError, match="Unknown chunk dimension"):
            random_patch.io.write(path, "DASDAE", storage={"chunks": {"tim": 500}})
        # Validation happens before any patch data is written.
        with h5py.File(path) as h5:
            assert "waveforms" not in h5

    def test_chunks_without_codec(self, tmp_path_factory, random_patch):
        """Chunks apply even without a codec (chunked-uncompressed layout)."""
        path = tmp_path_factory.mktemp("dasdae_chunks_only") / "chunked.h5"
        random_patch.io.write(path, "DASDAE", storage={"chunks": {"time": 500}})
        with h5py.File(path) as h5:
            group = next(iter(h5["waveforms"].values()))
            # time is the second dim; chunk length clamps to the request.
            assert group["data"].chunks[1] == 500
            assert group["data"].compression is None
        assert dc.read(path)[0].equals(random_patch)


class TestDASDAEStorage:
    """Tests for DASDAE storage settings."""

    def test_default_is_uncompressed(self):
        """Default storage produces no dataset options and no chunking."""
        storage = DASDAEStorage()
        assert storage._dataset_options(("distance", "time"), (3, 4)) == {}
        assert storage._resolve_chunkshape(("distance", "time"), (3, 4)) is None

    def test_compressed_preset_uses_default_codec(self):
        """The compressed preset uses the DASDAE default codec."""
        storage = DASDAEStorage.from_preset("compressed")
        assert storage.codec == Gzip(level=5)
        options = storage._dataset_options(("time",), (10,))
        assert options["compression"] == "gzip"
        assert options["compression_opts"] == 5

    def test_codec_string_shorthand(self):
        """A bare codec name is coerced to the codec instance."""
        assert DASDAEStorage(codec="gzip").codec == Gzip()

    def test_codec_dict_shorthand(self):
        """A codec dict is dispatched by its discriminator name."""
        assert DASDAEStorage(codec={"name": "gzip", "level": 3}).codec == Gzip(level=3)

    def test_codec_dict_requires_name(self):
        """Codec dictionaries must include the registry discriminator."""
        with pytest.raises(ValueError, match="name"):
            DASDAEStorage(codec={"level": 3})

    def test_bad_codec_input_raises(self):
        """Unsupported codec input types raise a clear error."""
        with pytest.raises(ValueError, match="Cannot interpret codec"):
            DASDAEStorage(codec=object())

    def test_unsupported_codec_base_raises(self):
        """DASDAE rejects codecs it cannot store in HDF5 arrays."""
        with pytest.raises(ValueError, match="cannot store codec"):
            DASDAEStorage(codec=_UnsupportedCodec())

    def test_unknown_codec_name_raises(self):
        """An unknown codec name reports it is unregistered."""
        with pytest.raises(ValueError, match="Unknown codec"):
            DASDAEStorage(codec="lz4")

    def test_gzip_codec_maps_to_h5py_kwargs(self):
        """Gzip codec maps to the h5py gzip dataset kwargs."""
        options = DASDAEStorage(codec=Gzip(level=3))._dataset_options(("time",), (10,))
        assert options == {
            "compression": "gzip",
            "compression_opts": 3,
            "shuffle": True,
        }

    def test_level_zero_codec_disables_compression(self):
        """A level-0 codec writes uncompressed without erroring."""
        storage = DASDAEStorage(codec=Gzip(level=0))
        assert storage._dataset_options(("time",), (10,)) == {}

    def test_bare_base_codec_rejected_at_construction(self):
        """A codec base class with no registered name fails fast, not mid-write."""
        from dascore.io.hdf5 import HDF5Codec

        with pytest.raises(ValueError, match="name"):
            DASDAEStorage(codec=HDF5Codec(level=5))

    def test_new_preserves_codec(self):
        """Deriving a modified storage with .new() keeps the codec (GH #734)."""
        for storage in (
            DASDAEStorage.from_preset("compressed"),
            DASDAEStorage(codec="gzip"),
            DASDAEStorage(codec=Gzip(level=3)),
            DASDAEStorage(codec={"name": "gzip", "level": 4}),
        ):
            new = storage.new(chunks={"time": 100})
            assert new.codec == storage.codec
            assert new.chunks == {"time": 100}

    def test_chunkshape_resolves_by_dim_name(self):
        """Chunks map dim names to lengths, clamped to the array shape."""
        storage = DASDAEStorage(chunks={"time": 5})
        # time is the second dim here; distance uses its full length.
        assert storage._resolve_chunkshape(("distance", "time"), (3, 20)) == (3, 5)
        # request larger than the array is clamped down.
        assert storage._resolve_chunkshape(("time",), (4,)) == (4,)
        # scalar/mismatched arrays are left contiguous.
        assert storage._resolve_chunkshape((), ()) is None

    def test_negative_chunk_raises(self):
        """Chunk sizes must be positive."""
        with pytest.raises(ValueError, match="positive"):
            DASDAEStorage(chunks={"time": 0})

    def test_unknown_chunk_dim_raises(self):
        """A chunk dim that matches no real dimension is rejected."""
        storage = DASDAEStorage(chunks={"tim": 5})
        with pytest.raises(ValueError, match="Unknown chunk dimension"):
            storage._validate_chunk_dims(("distance", "time"))

    def test_known_chunk_dims_pass(self):
        """Valid chunk dims validate without error."""
        storage = DASDAEStorage(chunks={"time": 5})
        # Should not raise.
        storage._validate_chunk_dims(("distance", "time"))

    def test_validate_chunk_dims_no_chunks_returns(self):
        """Default storage has no chunk dims to validate."""
        storage = DASDAEStorage()
        storage._validate_chunk_dims(())

    def test_get_codecs(self):
        """DASDAE reports the registered HDF5 codecs it can store."""
        assert set(DASDAEStorage.get_codecs()) == {Gzip}


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
        """Ensure an empty patch can be deserialized."""
        spool = dc.read(written_dascore_v1_empty)
        assert len(spool) == 1
        spool[0].equals(dc.Patch())

    def test_reads_legacy_fixture(self):
        """Legacy DASDAE fixtures still need to remain readable."""
        path = fetch("example_dasdae_event_1.h5")
        with config_context(allow_dasdae_format_unpickle=True):
            spool = dc.read(path, file_format="DASDAE")
        assert len(spool) == 1
        assert spool[0].dims

    def test_append_to_legacy_file_keeps_new_attrs(
        self, random_patch, tmp_path_factory
    ):
        """New groups appended to a legacy file must not be legacy-stripped."""
        from dascore.io.dasdae.utils import _SEPARATE_ATTRS_KEY

        path = tmp_path_factory.mktemp("dasdae_legacy_append") / "mixed.h5"
        old_patch = random_patch.update_attrs(tag="old")
        old_patch.io.write(path, "dasdae")
        # simulate a legacy file: strip the root and group markers
        with h5py.File(path, "a") as h5:
            del h5.attrs[_SEPARATE_ATTRS_KEY]
            for group in h5["waveforms"].values():
                del group.attrs[_SEPARATE_ATTRS_KEY]
        # append a new patch carrying an attr shadowing its own coord envelope
        dim = random_patch.dims[0]
        new_patch = random_patch.update_attrs(tag="new", **{f"{dim}_step": 999})
        new_patch.io.write(path, "dasdae")
        with h5py.File(path, "r") as h5:
            assert not h5.attrs.get(_SEPARATE_ATTRS_KEY, False)  # file stays legacy
        patches = {p.attrs["tag"]: p for p in dc.read(path, file_format="DASDAE")}
        assert set(patches) == {"old", "new"}
        # the appended group round-trips exactly; the legacy one still reads
        assert patches["new"].attrs[f"{dim}_step"] == 999
        assert patches["old"].coords == old_patch.coords

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
        time_summary = patch_2.summary.get_coord_summary("time")
        for value in (time_summary.min, time_summary.max):
            assert isinstance(value, np.datetime64)

    def test_read_file_no_wavegroup(self, generic_hdf5):
        """Ensure an h5 with no wavegroup returns empty patch."""
        parser = DASDAEV1()
        spool = parser.read(generic_hdf5)
        assert not len(spool)

    def test_read_source_patch_id(self, tmp_path):
        """Reading with a source patch id should only load one patch."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=2)
        dc.write(spool, path, "DASDAE", file_version="1")
        scanned = dc.scan(path)
        target = scanned[1].source_patch_id
        out = dc.read(path, source_patch_id=target)
        assert len(out) == 1
        assert out[0].attrs["_source_patch_id"] == target
        assert out[0].summary.source_patch_id == out[0].attrs["_source_patch_id"]
        assert (
            out[0].summary.get_coord_summary("time").min
            == scanned[1].get_coord_summary("time").min
        )

    def test_read_multiple_source_patch_ids(self, tmp_path):
        """Reading with multiple source patch ids should return each match."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=3)
        dc.write(spool, path, "DASDAE", file_version="1")
        scanned = dc.scan(path)
        targets = [scanned[0].source_patch_id, scanned[2].source_patch_id]
        out = dc.read(path, source_patch_id=targets)
        assert len(out) == 2
        assert {patch.attrs["_source_patch_id"] for patch in out} == set(targets)
        assert {patch.summary.get_coord_summary("time").min for patch in out} == {
            scanned[0].get_coord_summary("time").min,
            scanned[2].get_coord_summary("time").min,
        }

    def test_read_ignores_multi_dim_coord_filters(
        self, tmp_path, multi_dim_coords_patch
    ):
        """Multi-dimensional coord kwargs should bypass coord selection safely."""
        path = tmp_path / "multi_dim_filter.h5"
        multi_dim_coords_patch.io.write(path, "dasdae")

        out = dc.read(path, quality=(0, 1))[0]

        assert out == multi_dim_coords_patch

    def test_file_spool_loads_distinct_attrs(self, tmp_path, random_patch):
        """Lazy loading should materialize the patch for each DASDAE row."""
        path = tmp_path / "multi_patch.h5"
        patches = [
            random_patch.update_attrs(tag="S100", label="L100"),
            random_patch.update_attrs(tag="S120", label="L120"),
        ]

        dc.write(dc.spool(patches), path, "DASDAE")
        spool = dc.spool(path)

        assert spool.get_contents()["tag"].to_list() == ["S100", "S120"]
        assert [x.attrs.tag for x in spool] == ["S100", "S120"]
        assert [x.attrs.label for x in spool] == ["L100", "L120"]

    def test_read_filters_patch_attrs_before_loading(self, tmp_path, random_patch):
        """DASDAE attr filters should skip non-matching patch groups."""
        path = tmp_path / "multi_patch.h5"
        patches = [
            random_patch.update_attrs(tag="S100"),
            random_patch.update_attrs(tag="S120"),
        ]
        dc.write(dc.spool(patches), path, "DASDAE")

        out = dc.read(path, tag="S120")

        assert len(out) == 1
        assert out[0].attrs.tag == "S120"

    def test_get_format_false(self, generic_hdf5):
        """A generic HDF5 file is not a DASDAE file."""
        parser = DASDAEV1()
        assert not parser.get_format(generic_hdf5)

    def test_read_empty_selection_returns_no_patches(
        self, tmp_path_factory, random_patch
    ):
        """Selections outside an empty patch should return no patches."""
        path = tmp_path_factory.mktemp("dasdae_read_empty_selection") / "out.h5"
        time = random_patch.get_coord("time")
        random_patch.io.write(path, "dasdae")
        empty_range_start = time.max() + 3 * time.step
        out = dc.read(path, time=(empty_range_start, ...))
        assert len(out) == 0


class TestScanDASDAE:
    """Tests for scanning the dasdae format."""

    def test_scan_returns_info(self, written_dascore_v1_random, random_patch):
        """Ensure scanning returns expected values."""
        info1 = dc.scan(written_dascore_v1_random)[0].attrs.model_dump()
        info2 = random_patch.attrs.model_dump()
        common_keys = set(info1) & set(info2) - {"history"}
        for key in common_keys:
            assert info1[key] == info2[key]

    def test_scan_has_source_patch_id(self, written_dascore_v1_random):
        """Scanned DASDAE patches should expose source patch ids."""
        patch = dc.scan(written_dascore_v1_random)[0]
        assert patch.source_patch_id

    def test_copied_fixture_matches_original(
        self,
        written_dascore_v1_random,
        written_dascore_v1_random_copy,
    ):
        """Copying a DASDAE file should not change scan output."""
        df1 = dc.scan_to_df(written_dascore_v1_random)
        df2 = dc.scan_to_df(written_dascore_v1_random_copy)
        # common fields should be equal (except path)
        common = list((set(df1) & set(df2)) - {"path"})
        assert df1[common].equals(df2[common])

    def test_get_patch_summary_has_file_metadata(self, random_spool):
        """The summary helper should stamp DASDAE metadata on each row."""
        out = DASDAEV1()._get_patch_summary(random_spool)
        assert set(out["file_format"]) == {"DASDAE"}
        assert set(out["file_version"]) == {"1"}
        assert out["source_patch_id"].notnull().all()


class TestLegacyFixtureCompatibility:
    """Tests for the retained legacy DASDAE fixture compatibility helpers."""

    def test_translate_legacy_attrs_coord_manager_like_coords(self):
        """Legacy coord managers should still flatten via to_summary_dict."""

        class CoordManagerLike:
            def to_summary_dict(self):
                return {"time": {"units": "s", "step": 1}}

        out = translate_legacy_attrs({"coords": CoordManagerLike()})
        assert out["time_units"] == "s"
        assert out["time_step"] == 1

    def test_translate_legacy_attrs_summary_like_coord(self):
        """Legacy coord summaries should normalize via to_summary/model_dump."""

        class SummaryLike:
            def to_summary(self):
                return dc.core.CoordSummary(min=0, max=1, step=2, units="m")

        out = translate_legacy_attrs(
            {
                "coords": {"distance": SummaryLike(), "time": object()},
                "dims": "distance,time",
                "d_time": 3,
            }
        )
        assert out["distance_units"] == "m"
        assert out["distance_step"] == 2
        assert out["time_step"] == 3

    def test_translate_legacy_attrs_ignores_non_mapping_coords(self):
        """Undecodable string coord payloads are ignored once opted in."""
        with config_context(allow_dasdae_format_unpickle=True):
            out = translate_legacy_attrs({"coords": "pickled-coords-placeholder"})
        assert "coords" not in out

    def test_translate_legacy_attrs_decodes_pickled_coord_payload(self):
        """Legacy pickled coord payloads should still restore coord metadata."""
        payload = pickle.dumps({"distance": {"min": 0, "max": 1, "units": "m"}})
        with config_context(allow_dasdae_format_unpickle=True):
            out = translate_legacy_attrs({"coords": payload.decode("latin1")})
        assert out["distance_units"] == "m"
        assert out["distance_min"] == 0
        assert out["distance_max"] == 1

    def test_translate_legacy_attrs_never_unpickles_without_opt_in(self):
        """The opt-in gate must fire before any pickle.loads call runs."""
        executed = []

        class _Payload:
            def __reduce__(self):
                return (executed.append, ("pickle ran",))

        payload = pickle.dumps(_Payload()).decode("latin1")
        with config_context(allow_dasdae_format_unpickle=False):
            with pytest.raises(InvalidFiberFileError, match="unpickle"):
                translate_legacy_attrs({"coords": payload})
        assert not executed, "pickle.loads ran before the security gate"

    def test_scan_preserves_legacy_coord_units_from_attr_payload(self):
        """Legacy attr coord units should backfill missing coord-node units."""
        with config_context(allow_dasdae_format_unpickle=True):
            summary = dc.scan(fetch("UoU_lf_urban.hdf5"))[0]
        assert str(summary.coords["distance"].units) == "1 m"

    def test_read_legacy_coord_payload_requires_opt_in(self):
        """Legacy pickled coord metadata should fail closed by default."""
        with config_context(allow_dasdae_format_unpickle=False):
            with pytest.raises(
                InvalidFiberFileError, match="allow_dasdae_format_unpickle=True"
            ):
                dc.read(fetch("UoU_lf_urban.hdf5"))

    def test_scan_prefers_exact_coord_over_legacy_step_metadata(self, tmp_path):
        """Exact coord scans should not invent steps from legacy summary metadata."""
        path = tmp_path / "legacy_coord_step.h5"
        payload = pickle.dumps(
            {
                "distance": {
                    "min": 0.0,
                    "max": 3.0,
                    "step": 1.0,
                    "units": "m",
                    "dtype": "float64",
                }
            }
        )
        with h5py.File(path, "w") as h5:
            h5.attrs["__format__"] = "DASDAE"
            h5.attrs["__DASDAE_version__"] = "1"
            waveforms = h5.create_group("waveforms")
            group = waveforms.create_group("patch")
            group.attrs["_dims"] = "distance"
            group.attrs["_attrs_coords"] = np.bytes_(payload)
            group.attrs["_cdims_distance"] = "distance"
            group.create_dataset("_coord_distance", data=np.array([0.0, 1.0, 3.0]))
            summary = _get_scan_payload_from_group(group)
        assert summary["coords"]["distance"].step is None

    def test_decode_legacy_attr_bytes_falls_back_to_text(self):
        """Undecodable legacy bytes should fall back to plain text."""
        assert _decode_legacy_attr_value(b"abc") == "abc"

    def test_decode_legacy_attr_pickled_bytes_fall_back_to_text(self):
        """Legacy pickled attrs should no longer be unpickled."""
        payload = pickle.dumps(("a", "b"))
        assert isinstance(_decode_legacy_attr_value(payload), str)

    def test_decode_legacy_attr_unboxes_scalar_arrays(self):
        """Scalar legacy arrays should be unpacked back to scalars."""
        assert _decode_legacy_attr_value(np.asarray(5)) == 5


class TestDASDAEInternalHelpers:
    """Direct tests for h5py-backed DASDAE helper branches."""

    def test_save_array_overwrites_existing_dataset(self, tmp_path):
        """Saving to an existing dataset name should replace it."""
        path = tmp_path / "overwrite_array.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms")
            _save_array(np.arange(2), "data", group=group)
            _save_array(np.arange(3), "data", group=group)
            assert np.array_equal(group["data"][:], np.arange(3))

    def test_save_patch_overwrites_existing_group(self, random_patch, tmp_path):
        """Saving a patch with the same name should replace the old group."""
        path = tmp_path / "overwrite_patch.h5"
        with h5py.File(path, "w") as h5:
            waveforms = h5.create_group("waveforms")
            _save_patch(random_patch, waveforms, "patch_0", DASDAEStorage())
            _save_patch(
                random_patch.update_attrs(tag="new"),
                waveforms,
                "patch_0",
                DASDAEStorage(),
            )
            attrs = _get_attrs(waveforms["patch_0"])
            assert attrs["tag"] == "new"

    def test_get_patch_summary_unpacks_scalar_attr_array(self, tmp_path):
        """Scalar encoded attrs should be unpacked in patch summaries."""
        path = tmp_path / "summary_scalar.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_attrs_station"] = np.asarray("A01", dtype=h5py.string_dtype())
            group.attrs["_cdims_time"] = "time"
            group.create_dataset("_coord_time", data=np.array([0, 1]))
            summary = _get_scan_payload_from_group(group)
        assert summary["attrs"].station == "A01"
        assert summary["dtype"] == ""

    def test_get_attrs_unpacks_scalar_attr_arrays(self, monkeypatch):
        """Scalar arrays returned by attr decoding should be unpacked."""

        class _Group:
            attrs: ClassVar[dict[str, str]] = {"_attrs_station": "unused"}

        monkeypatch.setattr(
            dasdae_mod.utils,
            "_decode_attr_value",
            lambda *_args, **_kwargs: np.asarray("A01"),
        )
        assert _get_attrs(_Group()) == {"station": "A01"}

    def test_get_patch_summary_unpacks_scalar_arrays_from_decoder(
        self, tmp_path, monkeypatch
    ):
        """Patch summaries should unpack scalar arrays returned by decoding."""
        path = tmp_path / "summary_scalar_decoder.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_attrs_station"] = "unused"
            group.attrs["_cdims_time"] = "time"
            group.create_dataset("_coord_time", data=np.array([0, 1]))
            monkeypatch.setattr(
                dasdae_mod.utils,
                "_decode_attr_value",
                lambda *_args, **_kwargs: np.asarray("A01"),
            )
            summary = _get_scan_payload_from_group(group)
        assert summary["attrs"].station == "A01"

    def test_get_patch_summary_preserves_empty_dims_and_shape(self, tmp_path):
        """Empty stored dims should remain empty tuples in summaries."""
        path = tmp_path / "summary_empty_dims.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = ""
            group.create_dataset("data", data=np.arange(6).reshape(2, 3))
            summary = _get_scan_payload_from_group(group)
        assert summary["dims"] == ()
        assert summary["shape"] == (2, 3)

    def test_get_contents_from_patch_groups_returns_empty_without_waveforms(
        self, tmp_path
    ):
        """Files without waveforms should scan as empty."""
        path = tmp_path / "empty_scan.h5"
        with h5py.File(path, "w") as h5:
            out = _get_contents_from_patch_groups_generic(h5)
            assert out == []

    def test_get_coords_range_like_node_skips_full_array_read(
        self, tmp_path, monkeypatch
    ):
        """Range-like coord nodes should reconstruct without materializing arrays."""
        path = tmp_path / "range_coord_fast_path.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_cdims_time"] = "time"
            node = group.create_dataset("_coord_time", data=np.array([10, 20, 30]))
            node.attrs["step"] = 10
            node.attrs["step_is_timedelta64"] = False

            def _forbid_full_read(*_args, **_kwargs):
                raise AssertionError("full coord reads should be skipped")

            monkeypatch.setattr(dasdae_mod.utils, "_read_array", _forbid_full_read)
            coords = _get_coords(group, ("time",), {})

        coord = coords.get_coord("time")
        assert coord.__class__.__name__ == "CoordRange"
        assert len(coord) == 3
        assert coord.start == 10
        assert coord.step == 10

    def test_get_coords_range_like_node_restores_timedelta_sample(
        self, tmp_path, monkeypatch
    ):
        """Range fast path should restore timedelta step/sample metadata."""
        path = tmp_path / "range_coord_timedelta_fast_path.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            group.attrs["_dims"] = "time"
            group.attrs["_cdims_time"] = "time"
            node = group.create_dataset(
                "_coord_time", data=np.array([1, 3, 5], dtype="int64")
            )
            node.attrs["is_timedelta64"] = True
            node.attrs["step"] = 2
            node.attrs["step_is_timedelta64"] = True

            def _forbid_full_read(*_args, **_kwargs):
                raise AssertionError("full coord reads should be skipped")

            monkeypatch.setattr(dasdae_mod.utils, "_read_array", _forbid_full_read)
            coords = _get_coords(group, ("time",), {})

        coord = coords.get_coord("time")
        assert coord.start == np.timedelta64(1, "ns")
        assert coord.step == np.timedelta64(2, "ns")

    def test_read_array_sample_restores_string_scalar(self, tmp_path):
        """Sample restoration should decode stored string scalars."""
        path = tmp_path / "string_coord_sample_path.h5"
        with h5py.File(path, "w") as h5:
            group = h5.create_group("waveforms").create_group("patch_0")
            node = group.create_dataset(
                "_coord_station", data=np.array([b"alpha", b"beta"], dtype="S5")
            )
            node.attrs["is_string"] = True
            node.attrs["original_string_dtype"] = "<U8"
            sample = dasdae_mod.utils._read_array_sample(node, 0)
        assert sample == "alpha"

    @pytest.mark.parametrize(
        ("key", "value", "expected_type", "expected_value"),
        [
            ("history", ("a", "b"), "history_json", ("a", "b")),
            (
                "value",
                np.timedelta64(5, "ns"),
                "timedelta64[ns]",
                np.timedelta64(5, "ns"),
            ),
        ],
    )
    def test_encode_decode_attr_value_round_trip(
        self, key, value, expected_type, expected_value
    ):
        """Canonical attr encoding should round-trip supported rich types."""
        encoded, attr_type = _encode_attr_value(key, value)
        decoded = _decode_attr_value(
            {f"_attr_type_{key}": attr_type},
            key,
            encoded,
        )
        assert attr_type == expected_type
        assert decoded == expected_value

    def test_encode_attr_value_empty_history(self):
        """Empty history should still use the dedicated JSON history branch."""
        encoded, attr_type = _encode_attr_value("history", [])
        assert attr_type == "history_json"
        assert encoded == "[]"

    def test_encode_attr_value_string_history_uses_single_entry_json(self):
        """String history should serialize as a one-entry JSON list."""
        encoded, attr_type = _encode_attr_value("history", "one step")
        assert attr_type == "history_json"
        assert encoded == '["one step"]'

    def test_encode_attr_value_generic_sequence_is_not_special_cased(self):
        """Non-history sequences should use default passthrough handling."""
        encoded, attr_type = _encode_attr_value("value", [1, 2])
        assert attr_type is None
        assert encoded == [1, 2]

    def test_decode_attr_value_unknown_type_returns_value(self):
        """Unknown attr types should pass values through unchanged."""
        value = "abc"
        assert (
            _decode_attr_value({"_attr_type_value": "mystery"}, "value", value) == value
        )

    def test_get_file_version_reads_dasdae_attr(self, tmp_path):
        """The DASDAE version helper should read the file-level version attr."""
        path = tmp_path / "versioned.h5"
        with h5py.File(path, "w") as h5:
            h5.attrs["__DASDAE_version__"] = "9"
            assert _get_file_version(h5) == "9"


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
        assert empty_patch.shape == new_patch.shape
        assert np.equal(empty_patch.data, new_patch.data).all()
        assert empty_patch.get_coord("distance") == new_patch.get_coord("distance")
        assert len(new_patch.get_coord("time")) == 0

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

    def test_roundtrip_coord_multiple_dims(
        self, tmp_path_factory, multi_dim_coords_patch
    ):
        """
        Ensure a patch with a non-dimensional coordinate that is associated
        with two dims can round-trip.
        """
        patch = multi_dim_coords_patch
        folder = tmp_path_factory.mktemp("dasdae_multi_dim_coord")
        path = folder / "multidimcoord.hdf"
        patch.io.write(path, "dasdae")

        # Ensure we can read it from a directory
        patch2 = dc.spool(folder).update()[0]
        # And from a single file
        patch3 = dc.spool(path)[0]
        # All of the patches should be equal.
        assert patch == patch2 == patch3

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

    def test_roundtrip_string_aux_coord(self, random_patch, tmp_path_factory):
        """Attached string coordinates should round-trip through DASDAE."""
        path = tmp_path_factory.mktemp("roundtrip_string_coord") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"sensor_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(sensor=("distance", labels))
        patch.io.write(path, "dasdae")
        out = dc.read(path, file_format="DASDAE")[0]
        coord = out.get_coord("sensor")
        assert isinstance(coord, CoordString)
        assert np.array_equal(coord.values, labels)

    def test_roundtrip_string_dim_coord(self, random_patch, tmp_path_factory):
        """String dimension coordinates should round-trip through DASDAE."""
        path = tmp_path_factory.mktemp("roundtrip_string_dim") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"ch_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(distance=labels)
        patch.io.write(path, "dasdae")
        out = dc.read(path, file_format="DASDAE")[0]
        coord = out.get_coord("distance")
        assert isinstance(coord, CoordString)
        assert np.array_equal(coord.values, labels)

    def test_scan_includes_string_coords(self, random_patch, tmp_path_factory):
        """String coordinates should appear in lossy scan summaries."""
        path = tmp_path_factory.mktemp("scan_string_coord") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"sensor_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(sensor=("distance", labels))
        patch.io.write(path, "dasdae")
        summary = dc.scan(path)[0]
        assert "sensor" in summary.coords
        assert summary.coords["sensor"].min == "sensor_000"
        assert summary.coords["sensor"].step is None

    def test_scan_to_df_includes_string_coord_columns(
        self, random_patch, tmp_path_factory
    ):
        """Flattened scan results should expose string coord summary fields."""
        path = tmp_path_factory.mktemp("scan_string_coord_df") / "out.h5"
        distance = random_patch.get_coord("distance")
        labels = np.array([f"sensor_{num:03d}" for num in range(len(distance))])
        patch = random_patch.update_coords(sensor=("distance", labels))
        patch.io.write(path, "dasdae")
        df = dc.scan_to_df(path)
        row = df.iloc[0]
        assert row["sensor_min"] == "sensor_000"
        assert row["sensor_max"] == labels[-1]
        assert pd.isnull(row["sensor_step"])


class TestStringArrayHelpers:
    """Tests for DASDAE string-array integration paths."""

    def test_non_string_object_array_not_converted_to_bytes(
        self, tmp_path, monkeypatch
    ):
        """Object arrays with non-string content should not be stringified."""
        path = tmp_path / "object_array.h5"
        data = np.array([1, 2], dtype=object)

        def _raise_if_called(data):
            msg = "non-string object arrays should not be string-converted"
            raise AssertionError(msg)

        monkeypatch.setattr(
            dasdae_mod.utils, "convert_strings_to_bytes", _raise_if_called
        )
        with h5py.File(path, mode="w") as h5:
            group = h5.create_group("waveforms")
            with pytest.raises(TypeError, match="Object dtype|object arrays"):
                _save_array(data, "obj", group=group)
