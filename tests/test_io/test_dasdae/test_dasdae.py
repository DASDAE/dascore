"""Tests for DASDAE format."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pytest
import tables

import dascore as dc
from dascore.compat import random_state
from dascore.io.core import BaseCodec
from dascore.io.dasdae.core import DASDAEV1
from dascore.io.dasdae.storage import DASDAEStorage
from dascore.io.dasdae.utils import _create_or_squash_array, _read_array, _save_array
from dascore.io.hdf5 import BloscZstd, Gzip
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
def written_dascore_v1_random_indexed(written_dascore_v1_random, tmp_path_factory):
    """Copy the previous dasdae file and create an index."""
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

    def _assert_data_filters(self, path, *, complib, complevel=None):
        """Assert the stored DASDAE data array has expected PyTables filters."""
        with tables.open_file(path) as h5:
            group = next(h5.iter_nodes("/waveforms"))
            assert group.data.filters.complib == complib
            if complevel is not None:
                assert group.data.filters.complevel == complevel

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
        storage = DASDAEStorage(codec=BloscZstd(level=5))
        dc.write(random_patch, path, "DASDAE", storage=storage)
        self._assert_data_filters(path, complib="blosc:zstd", complevel=5)
        assert dc.read(path)[0].equals(random_patch)

    def test_direct_write_coerces_storage(self, tmp_path_factory, random_patch):
        """Direct DASDAEV1 writes accept the same storage shorthand as dc.write."""
        path = tmp_path_factory.mktemp("dasdae_direct_storage") / "compressed.h5"
        with tables.open_file(path, mode="w") as h5:
            DASDAEV1().write(random_patch, h5, storage="compressed")
        self._assert_data_filters(path, complib="blosc:zstd", complevel=5)
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
        """DASDAE compression skips scalar arrays unsupported by CArray."""
        path = tmp_path_factory.mktemp("dasdae_compressed_scalar") / "out.h5"
        filters = DASDAEStorage.from_preset("compressed")._get_filters()
        with tables.open_file(path, mode="w") as h5:
            node = _create_or_squash_array(h5, h5.root, "scalar", np.array(1), filters)
            assert node.shape == ()
            assert node.filters.complevel == 0

    def test_write_dict_codec_coercion(self, tmp_path_factory, random_patch):
        """A dict/string codec shorthand is coerced by the write path."""
        path = tmp_path_factory.mktemp("dasdae_gzip") / "gzip.h5"
        random_patch.io.write(path, "DASDAE", storage={"codec": "gzip"})
        self._assert_data_filters(path, complib="zlib")
        assert dc.read(path)[0].equals(random_patch)

    def test_write_compressed_with_gzip(self, tmp_path_factory, random_patch):
        """DASDAE can use portable gzip HDF5 compression."""
        path = tmp_path_factory.mktemp("dasdae_gzip") / "gzip.h5"
        random_patch.io.write(path, "DASDAE", storage=DASDAEStorage(codec=Gzip()))
        self._assert_data_filters(path, complib="zlib")
        assert dc.read(path)[0].equals(random_patch)

    def test_chunked_unicode_array_roundtrips(self, tmp_path_factory):
        """Chunked/compressed string arrays preserve non-ASCII values."""
        path = tmp_path_factory.mktemp("dasdae_unicode_array") / "out.h5"
        data = np.array(["cafe", "café"], dtype="U8")
        filters = DASDAEStorage.from_preset("compressed")._get_filters()
        with tables.open_file(path, mode="w") as h5:
            _save_array(data, "labels", h5.root, h5, filters=filters, chunkshape=(1,))
            node = h5.root.labels
            assert node[:].dtype.kind == "S"
            assert np.array_equal(_read_array(node), data)

    def test_write_with_chunks(self, tmp_path_factory, random_patch):
        """DASDAE storage can specify a per-dimension chunk layout."""
        path = tmp_path_factory.mktemp("dasdae_chunkshape") / "chunked.h5"
        storage = DASDAEStorage(codec=BloscZstd(), chunks={"distance": 10, "time": 10})
        random_patch.io.write(path, "DASDAE", storage=storage)
        with tables.open_file(path) as h5:
            group = next(h5.iter_nodes("/waveforms"))
            assert group.data.chunkshape == (10, 10)
            assert group._coord_distance.chunkshape != (10, 10)
        assert dc.read(path)[0].equals(random_patch)

    def test_explicit_storage_none_writes_default(self, tmp_path_factory, random_patch):
        """An explicit storage=None is normalized to the default and round-trips."""
        path = tmp_path_factory.mktemp("dasdae_storage_none") / "out.h5"
        dc.write(random_patch, path, "DASDAE", storage=None)
        with tables.open_file(path) as h5:
            group = next(h5.iter_nodes("/waveforms"))
            # Default storage is uncompressed.
            assert group.data.filters.complevel == 0
        assert dc.read(path)[0].equals(random_patch)

    def test_typoed_chunk_dim_raises_on_write(self, tmp_path_factory, random_patch):
        """A chunk dim that isn't a real patch dim raises instead of no-op."""
        path = tmp_path_factory.mktemp("dasdae_chunk_typo") / "out.h5"
        with pytest.raises(ValueError, match="Unknown chunk dimension"):
            random_patch.io.write(path, "DASDAE", storage={"chunks": {"tim": 500}})
        # Validation happens before any patch data is written.
        with tables.open_file(path) as h5:
            assert "waveforms" not in h5.root

    def test_chunks_without_codec(self, tmp_path_factory, random_patch):
        """Chunks apply even without a codec (chunked-uncompressed layout)."""
        path = tmp_path_factory.mktemp("dasdae_chunks_only") / "chunked.h5"
        random_patch.io.write(path, "DASDAE", storage={"chunks": {"time": 500}})
        with tables.open_file(path) as h5:
            group = next(h5.iter_nodes("/waveforms"))
            # time is the second dim; chunk length clamps to the request.
            assert group.data.chunkshape[1] == 500
            assert group.data.filters.complevel == 0
        assert dc.read(path)[0].equals(random_patch)


class TestDASDAEStorage:
    """Tests for DASDAE storage settings."""

    def test_default_is_uncompressed(self):
        """Default storage produces no PyTables filters and no chunking."""
        storage = DASDAEStorage()
        assert storage._get_filters() is None
        assert storage._resolve_chunkshape(("distance", "time"), (3, 4)) is None

    def test_compressed_preset_uses_default_codec(self):
        """The compressed preset uses the DASDAE default codec."""
        storage = DASDAEStorage.from_preset("compressed")
        assert storage.codec == BloscZstd(level=5)
        filters = storage._get_filters()
        assert filters.complib == "blosc:zstd"
        assert filters.complevel == 5

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

    def test_gzip_codec_maps_to_pytables_zlib(self):
        """Gzip codec maps to the PyTables zlib filter."""
        filters = DASDAEStorage(codec=Gzip(level=3))._get_filters()
        assert filters.complib == "zlib"
        assert filters.complevel == 3

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
        assert set(DASDAEStorage.get_codecs()) == {BloscZstd, Gzip}


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
        for name in ("time_min", "time_max"):
            assert isinstance(patch_2.attrs[name], np.datetime64)

    def test_read_file_no_wavegroup(self, generic_hdf5):
        """Ensure an h5 with no wavegroup returns empty patch."""
        parser = DASDAEV1()
        spool = parser.read(generic_hdf5)
        assert not len(spool)


class TestScanDASDAE:
    """Tests for scanning the dasdae format."""

    def test_scan_returns_info(self, written_dascore_v1_random, random_patch):
        """Ensure scanning returns expected values."""
        info1 = dc.scan(written_dascore_v1_random)[0].model_dump()
        info2 = random_patch.attrs.model_dump()
        common_keys = set(info1) & set(info2) - {"history"}
        for key in common_keys:
            assert info1[key] == info2[key]

    # TODO we need to re-think indexing before this can work.
    @pytest.mark.xfail
    def test_indexed_vs_unindexed(
        self,
        written_dascore_v1_random,
        written_dascore_v1_random_indexed,
    ):
        """Whether the file is indexed or not the summary should be the same."""
        df1 = dc.scan_to_df(written_dascore_v1_random)
        df2 = dc.scan_to_df(written_dascore_v1_random_indexed)
        # common fields should be equal (except path)
        common = list((set(df1) & set(df2)) - {"path"})
        assert df1[common].equals(df2[common])


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
        assert empty_patch.equals(new_patch)

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
