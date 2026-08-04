"""Tests for HDF5 IO storage codecs."""

from __future__ import annotations

import pytest

from dascore.io.hdf5 import Gzip


class TestHDF5Codecs:
    """Tests for reusable HDF5 codec objects."""

    def test_gzip_dataset_kwargs(self):
        """Gzip maps to the native h5py gzip filter kwargs."""
        kwargs = Gzip(level=3)._dataset_kwargs()
        assert kwargs["compression"] == "gzip"
        assert kwargs["compression_opts"] == 3
        assert kwargs["shuffle"]

    def test_shuffle_disabled(self):
        """The shuffle filter can be turned off."""
        assert not Gzip(shuffle=False)._dataset_kwargs()["shuffle"]

    def test_level_zero_disables_compression(self):
        """A level of 0 produces no dataset kwargs."""
        assert Gzip(level=0)._dataset_kwargs() == {}

    def test_bad_level_raises(self):
        """HDF5 compression levels must fit the 0-9 range."""
        with pytest.raises(ValueError, match="level"):
            Gzip(level=10)
        with pytest.raises(ValueError, match="level"):
            Gzip(level=-1)


class TestCodecSerialization:
    """The Literal name lets codecs round-trip through pydantic."""

    def test_name_included_in_dump(self):
        """Dumping a codec includes its discriminator name."""
        config = Gzip(level=2, shuffle=False).model_dump()
        assert config == {"name": "gzip", "level": 2, "shuffle": False}

    def test_roundtrip_through_validate(self):
        """A dumped codec reconstructs the same instance."""
        codec = Gzip(level=4)
        assert Gzip.model_validate(codec.model_dump()) == codec

    def test_extra_field_forbidden(self):
        """Codecs reject unknown fields to catch typos."""
        with pytest.raises(ValueError, match="Extra inputs"):
            Gzip(not_a_field=3)
