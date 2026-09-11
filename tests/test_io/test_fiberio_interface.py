"""Contract tests for assembling data-less reader metadata and array windows."""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.source import PatchSource
from dascore.exceptions import InvalidFiberIOError
from dascore.io import FiberIO
from dascore.io.utils import slice_dataset


class _ArrayReader(FiberIO):
    """A reader with distinguishable cells and two associated coordinates."""

    name = "_interface_contract_array"
    version = "1"

    def __init__(self):
        self.data = np.arange(48, dtype=np.int16).reshape(6, 8)
        self.calls = []
        self.snaps = []
        self.metadata = dc.Patch(
            coords={
                "distance": np.arange(6),
                "time": np.arange(8),
                "quality": ("time", np.arange(8) % 2),
                "receiver": ("distance", np.arange(6) % 2),
            },
            dims=("distance", "time"),
            dtype=self.data.dtype,
            attrs={"tag": "kept"},
            source=PatchSource(key="part"),
        )

    def get_version(self, resource) -> str | None:
        """Claim the in-memory sentinel."""
        return self.version if resource == "memory" else None

    def get_metadata(self, resource, *, snap=True) -> list[dc.Patch]:
        """Return metadata without touching samples."""
        self.snaps.append(snap)
        return [self.metadata]

    def read_array(self, resource, windows, key="") -> np.ndarray:
        """Record the requested bounding window and return its cells."""
        self.calls.append((windows, key))
        return slice_dataset(self.data, self.metadata.dims, windows)


class TestDerivedRead:
    """The shared reader owns selection, attachment, and contract validation."""

    @pytest.fixture
    def reader(self):
        """Fresh mutable state without registering another reader class."""
        return _ArrayReader()

    def test_whole_array(self, reader):
        """No selection retains every cell, coordinate, dtype and logical key."""
        out = reader.read("memory")[0]
        np.testing.assert_array_equal(out.data, reader.data)
        assert out.coords == reader.metadata.coords
        assert out.dtype == reader.metadata.dtype
        assert out._source == reader.metadata._source
        assert out.attrs.history == reader.metadata.attrs.history
        assert out.attrs.processing_id == reader.metadata.attrs.processing_id

    def test_noncontiguous_both_axes(self, reader):
        """Compose associated constraints on the original grid, then trim residuals."""
        out = reader.read(
            "memory", distance=(1, 4), time=(2, 6), quality=(1, 1), receiver=(1, 1)
        )[0]
        expected = reader.data[np.ix_([1, 3], [3, 5])]
        np.testing.assert_array_equal(out.data, expected)
        np.testing.assert_array_equal(out.get_coord("distance").values, [1, 3])
        np.testing.assert_array_equal(out.get_coord("time").values, [3, 5])
        assert reader.calls == [({"distance": (1, 4), "time": (3, 6)}, "part")]
        assert out.attrs.history == reader.metadata.attrs.history
        assert out.attrs.processing_id == reader.metadata.attrs.processing_id

    def test_sample_selection(self, reader):
        """Half-open sample bounds are passed through as exact array windows."""
        out = reader.read("memory", distance=(2, 5), time=(1, 7), samples=True)[0]
        np.testing.assert_array_equal(out.data, reader.data[2:5, 1:7])
        assert reader.calls == [({"distance": (2, 5), "time": (1, 7)}, "part")]

    @pytest.mark.parametrize(
        "selection",
        [{"time": (100, 200)}, {"tag": "missing"}, {"source_patch_key": "missing"}],
    )
    def test_rejected_metadata_never_reads(self, reader, selection):
        """Empty coordinates, rejected attrs, and unknown keys avoid sample I/O."""
        assert len(reader.read("memory", **selection)) == 0
        assert reader.calls == []

    @pytest.mark.parametrize(
        "options, expected",
        [
            ({}, True),
            ({"snap": False}, False),
            ({"snap_dims": False}, False),
            ({"snap": True, "snap_dims": False}, True),
        ],
    )
    def test_snap_forwarding(self, reader, options, expected):
        """Both spellings select the metadata mode; explicit snap takes precedence."""
        reader.read("memory", **options)
        assert reader.snaps == [expected]

    def test_null_attr_hints_do_not_filter(self, reader):
        """Index rows use NaN for absent attrs; those hints must not discard data."""
        out = reader.read("memory", tag=np.nan)[0]
        np.testing.assert_array_equal(out.data, reader.data)
        assert out.attrs.tag == "kept"

    def test_scan_never_reads_data(self, reader, monkeypatch):
        """Metadata and summaries do not allocate or access a sample array."""

        def fail(*args, **kwargs):
            raise AssertionError("sample access during metadata scan")

        monkeypatch.setattr(reader, "read_array", fail)
        monkeypatch.setattr(dc.Patch, "data", property(fail))
        metadata = reader.scan("memory", snap=False)[0]
        assert metadata._data is None
        assert metadata.summary.dtype == str(reader.data.dtype)
        assert reader.snaps == [False]

    @pytest.mark.parametrize("failure", ["shape", "dtype"])
    def test_bad_array_contract_raises(self, reader, monkeypatch, failure):
        """A reader cannot attach a different shape or silently cast its dtype."""
        data = (
            reader.data[:, :-1] if failure == "shape" else reader.data.astype("float32")
        )
        monkeypatch.setattr(reader, "read_array", lambda *a, **k: data)
        with pytest.raises(InvalidFiberIOError, match="metadata declared"):
            reader.read("memory")

    def test_empty_version_is_valid(self, reader):
        """An empty version, as used by Pickle, still identifies a format."""
        reader.version = ""
        assert reader.get_format("memory") == (reader.name, "")
        assert reader.get_format("other") is False
