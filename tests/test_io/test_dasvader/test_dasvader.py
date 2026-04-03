"""
Tests specific to DASvader format.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import pytest
from h5py.h5r import Reference

import dascore as dc
from dascore.exceptions import DependencyError, UnknownFiberFormatError
from dascore.io.dasvader.utils import _dereference, _julia_ms_to_datetime64
from dascore.utils.downloader import fetch


@dataclass(frozen=True)
class _ModernDASVaderValues:
    """Constants for the synthetic DASVader file used in these tests."""

    ntime: int = 8
    ndistance: int = 20
    time_step: float = 1.0
    distance_step: float = 0.5
    gauge_length: float = 10.0
    host_name: str = "test-host"
    pipeline_tracker: str = "tracker-1"
    pulse_rate_freq: float = 5000.0
    htime_ms: int = 62_135_683_200_000


MODERN_DASVADER = _ModernDASVaderValues()


def _write_modern_dasvader_file(
    path: Path, data_name: str = "data", include_attrib: bool = True
) -> Path:
    """Write a readable DASVader file with named datasets and references."""
    tp_dtype = np.dtype([("hi", "<f8"), ("lo", "<f8")])
    sr_dtype = np.dtype(
        [("ref", tp_dtype), ("step", tp_dtype), ("len", "<i8"), ("offset", "<i8")]
    )
    ddas_fields = [
        (data_name, h5py.ref_dtype),
        ("time", sr_dtype),
        ("htime", h5py.ref_dtype),
        ("offset", sr_dtype),
    ]
    if include_attrib:
        ddas_fields.append(("atrib", h5py.ref_dtype))
    ddas_dtype = np.dtype(ddas_fields)

    with h5py.File(path, "w") as fi:
        shape = (
            (MODERN_DASVADER.ndistance, MODERN_DASVADER.ntime)
            if data_name == "data"
            else (MODERN_DASVADER.ntime, MODERN_DASVADER.ndistance)
        )
        data = fi.create_dataset(
            data_name,
            data=np.arange(MODERN_DASVADER.ntime * MODERN_DASVADER.ndistance).reshape(
                shape
            ),
        )
        htime = fi.create_dataset("htime", data=np.array([MODERN_DASVADER.htime_ms]))

        ddas = np.zeros((), dtype=ddas_dtype)
        ddas[data_name] = data.ref
        ddas["htime"] = htime.ref
        ddas["time"]["ref"]["hi"] = 0.0
        ddas["time"]["step"]["hi"] = MODERN_DASVADER.time_step
        ddas["time"]["len"] = MODERN_DASVADER.ntime
        ddas["time"]["offset"] = 1
        ddas["offset"]["ref"]["hi"] = 0.0
        ddas["offset"]["step"]["hi"] = MODERN_DASVADER.distance_step
        ddas["offset"]["len"] = MODERN_DASVADER.ndistance
        ddas["offset"]["offset"] = 1

        if include_attrib:
            atrib_dtype = np.dtype(
                [
                    ("GaugeLength", h5py.ref_dtype),
                    ("Hostname", h5py.ref_dtype),
                    ("PipelineTracker", h5py.ref_dtype),
                    ("PulseRateFreq", "<f8"),
                ]
            )
            gauge = fi.create_dataset(
                "GaugeLength", data=np.array([MODERN_DASVADER.gauge_length])
            )
            host = fi.create_dataset(
                "Hostname",
                data=MODERN_DASVADER.host_name,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            tracker = fi.create_dataset(
                "PipelineTracker",
                data=MODERN_DASVADER.pipeline_tracker,
                dtype=h5py.string_dtype(encoding="utf-8"),
            )
            atrib = np.zeros((), dtype=atrib_dtype)
            atrib["GaugeLength"] = gauge.ref
            atrib["Hostname"] = host.ref
            atrib["PipelineTracker"] = tracker.ref
            atrib["PulseRateFreq"] = MODERN_DASVADER.pulse_rate_freq
            atrib_ds = fi.create_dataset("atrib", data=atrib)
            ddas["atrib"] = atrib_ds.ref

        fi.create_dataset("dDAS", data=ddas)
    return path


class TestDASVader:
    """Test case for dasvader."""

    @pytest.fixture(scope="session")
    def legacy_das_vader_path(self):
        """Get the legacy DASVader file path."""
        return fetch("das_vader_1.jld2")

    @pytest.fixture(scope="session")
    def dasvader_modern_path(self, tmp_path_factory):
        """Create a readable DASVader file with named datasets."""
        path = tmp_path_factory.mktemp("dasvader_modern") / "modern_named_refs.jld2"
        return _write_modern_dasvader_file(path)

    @pytest.fixture(scope="class")
    def das_vader_strainrate_no_attrib_path(self, tmp_path_factory):
        """Create a DASVader file using `strainrate` data and no `atrib` ref."""
        path = (
            tmp_path_factory.mktemp("dasvader_strainrate") / "strainrate_no_atrib.jld2"
        )

        tp_dtype = np.dtype([("hi", "<f8"), ("lo", "<f8")])
        sr_dtype = np.dtype(
            [("ref", tp_dtype), ("step", tp_dtype), ("len", "<i8"), ("offset", "<i8")]
        )
        ddas_dtype = np.dtype(
            [
                ("strainrate", h5py.ref_dtype),
                ("time", sr_dtype),
                ("htime", h5py.ref_dtype),
                ("offset", sr_dtype),
            ]
        )

        with h5py.File(path, "w") as fi:
            ntime, ndistance = 4, 3
            strainrate = fi.create_dataset(
                "strainrate",
                data=np.arange(ntime * ndistance).reshape(ntime, ndistance),
            )
            htime = fi.create_dataset("htime", data=np.array([62_135_683_200_000]))

            ddas = np.zeros((), dtype=ddas_dtype)
            ddas["strainrate"] = strainrate.ref
            ddas["htime"] = htime.ref
            ddas["time"]["ref"]["hi"] = 0.0
            ddas["time"]["step"]["hi"] = 1.0
            ddas["time"]["len"] = ntime
            ddas["time"]["offset"] = 1
            ddas["offset"]["ref"]["hi"] = 0.0
            ddas["offset"]["step"]["hi"] = 1.0
            ddas["offset"]["len"] = ndistance
            ddas["offset"]["offset"] = 1
            fi.create_dataset("dDAS", data=ddas)
        return path

    def test_all_attrs_resolved(self, dasvader_modern_path):
        """Ensure all attributes are resolved (no hdf5 references)"""
        das_vader_patch = dc.read(dasvader_modern_path)[0]
        for _, value in das_vader_patch.attrs.items():
            assert not isinstance(value, Reference)

    def test_read_and_scan_fallback_for_reference_dereference(
        self, monkeypatch, dasvader_modern_path
    ):
        """Ensure DASVader works when high-level reference deref fails."""
        # This intentionally patches h5py internals to simulate dereference
        # failures that are otherwise hard to trigger from the public API.
        # TODO: revisit if h5py internals change or a higher-level hook appears.
        original_getitem = h5py._hl.group.Group.__getitem__

        def _patched_getitem(group, key):
            if isinstance(key, Reference):
                raise KeyError("simulated token dereference failure")
            return original_getitem(group, key)

        monkeypatch.setattr(h5py._hl.group.Group, "__getitem__", _patched_getitem)

        patch = dc.read(dasvader_modern_path)[0]
        scanned = dc.scan(dasvader_modern_path)

        assert patch.dims
        assert len(scanned) == 1

    def test_legacy_file_raises_clear_error(self, legacy_das_vader_path):
        """Skip on incompatible stacks, otherwise ensure the file still reads."""
        try:
            patch = dc.read(legacy_das_vader_path)[0]
        except DependencyError as exc:
            pytest.skip(str(exc))
        assert patch.attrs is not None

    def test_legacy_file_scan_warns_and_skips(self, legacy_das_vader_path):
        """Scan should surface compatibility guidance as a warning, not an error."""
        with pytest.warns(UserWarning, match="legacy DASVader JLD2 file"):
            out = dc.scan(legacy_das_vader_path)
        assert out == []

    def test_non_dasvader_jld2_is_not_claimed(self, tmp_path):
        """Non-DASVader JLD2/HDF5 files should not be identified as DASVader."""
        path = tmp_path / "not_dasvader.jld2"
        with h5py.File(path, "w") as fi:
            fi.create_dataset("not_dDAS", data=np.arange(3))
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(path)

    def test_modern_file_scan_and_slice(self, dasvader_modern_path):
        """Named-reference DASVader files should still support scan and read."""
        scanned = dc.scan(dasvader_modern_path)
        assert len(scanned) == 1
        summary = scanned[0]
        assert summary.source_format == "DASVader"
        assert summary.attrs.gauge_length == MODERN_DASVADER.gauge_length
        assert summary.attrs.host_name == MODERN_DASVADER.host_name
        assert summary.attrs.pipeline_tracker == MODERN_DASVADER.pipeline_tracker
        assert summary.attrs.pulse_rate_frequency == MODERN_DASVADER.pulse_rate_freq
        time_summary = summary.get_coord_summary("time")
        patch = dc.read(
            dasvader_modern_path,
            time=(
                time_summary.min + dc.to_timedelta64(MODERN_DASVADER.time_step),
                ...,
            ),
        )[0]
        assert patch.dims == ("distance", "time")
        assert patch.attrs.gauge_length == MODERN_DASVADER.gauge_length

    def test_modern_file_out_of_range_read_is_empty(self, dasvader_modern_path):
        """Out-of-range selections should return an empty spool."""
        scanned = dc.scan(dasvader_modern_path)[0]
        distance_summary = scanned.get_coord_summary("distance")
        out = dc.read(
            dasvader_modern_path,
            distance=(distance_summary.max + MODERN_DASVADER.distance_step, ...),
        )
        assert len(out) == 0

    def test_strainrate_without_atrib(self, das_vader_strainrate_no_attrib_path):
        """Ensure parsing works when data is `strainrate` and `atrib` is absent."""
        patch = dc.read(das_vader_strainrate_no_attrib_path)[0]
        assert patch.dims == ("time", "distance")
        assert "gauge_length" not in patch.attrs.model_dump()
        scanned = dc.scan(das_vader_strainrate_no_attrib_path)
        assert len(scanned) == 1

    def test_julia_ms_to_datetime64_unwraps_nested_void(self):
        """
        Ensure nested JLD2-style void values are unwrapped.

        JLD2 stores Julia structs as nested compound values. When h5py reads a
        scalar field from such a compound dataset it can surface as nested
        ``np.void`` wrappers rather than a plain integer. This test uses the
        Julia ``DateTime -> Instant -> Periods -> value`` shape to verify the
        helper unwraps those layers before converting milliseconds.
        """
        value = np.array(
            [[(MODERN_DASVADER.htime_ms,)]],
            dtype=[("instant", [("periods", [("value", "<i8")])])],
        )[0, 0]
        out = _julia_ms_to_datetime64(value)
        assert out == np.datetime64("1970-01-01T00:00:00")

    def test_dereference_returns_non_reference(self):
        """Non-reference values should be returned unchanged."""
        value = np.float64(5_000.0)
        assert _dereference(None, value, "PulseRateFreq") == value
