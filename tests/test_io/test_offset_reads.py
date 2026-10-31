"""Tests that every reader recording byte offsets loads from them its own array."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.utils.downloader import fetch
from tests.conftest import needs_chunk_iter

# The HDF5 example files of the formats whose readers record offsets.
OFFSET_FILES = (
    "ai4eps_1.h5",
    "ap_sensing_1.hdf5",
    "PoroTomo_iDAS_1.h5",
    "gdr_1.h5",
    "h5_simple_1.h5",
    "h5_simple_2.h5",
    "neubrex_das_1.h5",
    "optasense_odh4_1.h5",
    "opto_das_1.hdf5",
    "decimated_optodas.hdf5",
    "prodml_2.0.h5",
    "prodml_2.1.h5",
    "iDAS005_hdf5_example.626.h5",
    "silixa_h5_1.hdf5",
    "silixa_h5_ingv_1.h5",
    "uptech_as1000_1.hdf5",
)

# The formats some file above must record offsets for.
RECORDING = {
    "AI4EPS",
    "APSensing",
    "DASHDF5",
    "GDR_DAS",
    "H5Simple",
    "NeubrexDAS",
    "ODH4",
    "OptoDAS",
    "PRODML",
    "Silixa_H5",
    "Uptech_H5",
}


def _sources(name):
    """Return the sources of a file's patches which record offsets."""
    return [x._source for x in dc.read(fetch(name)) if x._source.offsets]


class TestOffsetReads:
    """An offset load is exactly what the reader loads."""

    @pytest.mark.parametrize("name", OFFSET_FILES)
    def test_equals_reader(self, name, stored_reads):
        """The whole array, and a window across a chunk boundary, equal the reader's."""
        for source in _sources(name):
            edge = source.chunk_rows
            for part in (slice(None), slice(edge - 3, edge + 3)):
                out = source[part].load()
                assert not stored_reads
                expected = source._without_layout()[part].load()
                assert out.dtype == expected.dtype and out.flags.c_contiguous
                np.testing.assert_array_equal(out, expected)
                stored_reads.clear()

    @needs_chunk_iter
    def test_formats_record_offsets(self):
        """Each format expected to record offsets does, so none passes vacuously."""
        found = {x.format for name in OFFSET_FILES for x in _sources(name)}
        assert found == RECORDING
