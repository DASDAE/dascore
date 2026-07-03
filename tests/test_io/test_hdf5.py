"""Tests for HDF5 IO storage codecs."""

from __future__ import annotations

import pytest

from dascore.io.hdf5 import BloscZstd, Gzip


class TestHDF5Codecs:
    """Tests for reusable HDF5 codec objects."""

    def test_gzip_to_pytables_filter(self):
        """Gzip uses PyTables' zlib HDF5 filter."""
        filters = Gzip(level=3)._to_pytables_filters()
        assert filters.complib == "zlib"
        assert filters.complevel == 3
        assert filters.shuffle

    def test_blosc_zstd_to_pytables_filter(self):
        """BloscZstd uses the PyTables blosc:zstd HDF5 filter."""
        filters = BloscZstd(level=5)._to_pytables_filters()
        assert filters.complib == "blosc:zstd"
        assert filters.complevel == 5

    def test_level_zero_disables_compression(self):
        """A level of 0 produces no filters."""
        assert Gzip(level=0)._to_pytables_filters() is None
        assert BloscZstd(level=0)._to_pytables_filters() is None

    @pytest.mark.parametrize("codec_cls", [Gzip, BloscZstd])
    def test_bad_level_raises(self, codec_cls):
        """HDF5 compression levels must fit the PyTables range."""
        with pytest.raises(ValueError, match="level"):
            codec_cls(level=10)


class TestCodecSerialization:
    """The Literal name lets codecs round-trip through pydantic."""

    def test_name_included_in_dump(self):
        """Dumping a codec includes its discriminator name."""
        config = Gzip(level=2, shuffle=False).model_dump()
        assert config == {"name": "gzip", "level": 2, "shuffle": False}

    def test_roundtrip_through_validate(self):
        """A dumped codec reconstructs the same instance."""
        codec = BloscZstd(level=4)
        assert BloscZstd.model_validate(codec.model_dump()) == codec

    def test_extra_field_forbidden(self):
        """Codecs reject unknown fields to catch typos."""
        with pytest.raises(ValueError, match="Extra inputs"):
            Gzip(not_a_field=3)
