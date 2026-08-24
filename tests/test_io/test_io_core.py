"""Test for basic IO and related functions."""

from __future__ import annotations

import copy
import io
import os
import shutil
import threading
from pathlib import Path
from typing import ClassVar, Literal, TypeVar

import h5py
import numpy as np
import pandas as pd
import pytest
import rich.progress as prog
from upath import UPath

import dascore as dc
from dascore.config import config_context
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import (
    CoordMonotonicArray,
    CoordRange,
    CoordSegmented,
    get_coord,
)
from dascore.exceptions import (
    DependencyError,
    InvalidFiberIOError,
    MissingOptionalDependencyError,
    MissingPatchError,
    PatchAttributeError,
    RemoteCacheError,
    UnknownFiberFormatError,
)
from dascore.io import core as io_core
from dascore.io.core import (
    STORED_PATCH_ID,
    FiberIO,
    _canonical_path,
    _FiberIOManager,
    _get_missing_install_name,
    _get_reloadable_source_path,
    _handle_missing_optionals,
    _reinit_manager_lock,
    _resolve_read_spool,
    _scan_result_to_summary,
    _select_patch_from_spool,
    _size_and_mtime,
    _source_stats,
    _validate_scan_payload,
    is_directory_format,
    make_scan_payload,
)
from dascore.io.dasdae.core import DASDAEV1
from dascore.io.utils import (
    build_patches,
    convert_attr_units,
    get_exact_coord,
    get_gridded_coord,
)
from dascore.utils.downloader import fetch
from dascore.utils.io import BinaryReader, BinaryWriter, IOResourceManager
from dascore.utils.misc import suppress_warnings
from dascore.utils.time import to_datetime64
from dascore.workflow.identity import source_patch_id

tvar = TypeVar("tvar", int, float, str, Path)


class _FiberFormatTestV1(FiberIO):
    """A test format v1."""

    name = "_TestFormatter"
    version = "1"


class _FiberFormatTestV2(FiberIO):
    """A test format v2."""

    name = "_TestFormatter"
    version = "2"


class _FiberImplementer(FiberIO):
    """A fiber io which implements all the methods (poorly)."""

    name = "_Implementer"
    version = "2"

    def read(self, resource, **kwargs):
        """Dummy read."""

    def write(self, spool, resource, **kwargs):
        """Dummy write."""

    def scan(self, resource: BinaryReader, *, snap: bool = True, **kwargs):
        """Dummy scan."""

    def get_format(self, resource, **kwargs):
        """Dummy get_format."""


class _FiberCaster(FiberIO):
    """A test class for casting inputs to certain types."""

    name = "_TestFormatter"
    version = "2"

    def read(self, resource: BinaryReader, **kwargs):
        """Just ensure read was cast to correct type."""
        assert isinstance(resource, io.BufferedReader)

    def write(self, spool, resource: BinaryWriter, **kwargs):
        """Ditto for write."""
        assert isinstance(resource, io.BufferedWriter)

    def get_format(self, resource: Path, **kwargs) -> tuple[str, str] | Literal[False]:
        """And get format."""
        assert isinstance(resource, Path)
        return False

    def scan(self, not_path: BinaryReader):
        """Ensure an off-name still works for type casting."""
        assert isinstance(not_path, io.BufferedReader)


class _FiberInheritsScan(FiberIO):
    """A FiberIO which implements only read, as the docs allow."""

    name = "_InheritsScan"
    version = "1"
    seen_snap: ClassVar[list] = []

    def read(self, resource, snap=True, **kwargs):
        """Record what the inherited scan forwarded."""
        self.seen_snap.append(snap)
        return dc.spool([dc.get_example_patch()])


class _FiberUnsupportedTypeHints(FiberIO):
    """A fiber io which implements typehints which have no casting meaning."""

    name = "_TypeHinterNotRight"
    version = "2"

    def read(self, resource: tvar, **kwargs):
        """Dummy read."""
        with open(resource) as fi:
            return fi.read()


class _FiberDirectory(FiberIO):
    """A FiberIO which accepts a directory."""

    name = "_directory_test_io"
    version = "0.1"
    input_type = "directory"

    def get_format(self, resource, **kwargs) -> tuple[str, str] | Literal[False]:
        """Only accept directories which have specific naming."""
        path = Path(resource)
        name = path.name
        if self.name in name:
            return self.name, self.version
        return False

    def scan(self, resource, snap=True, **kwargs):
        """Return a payload that records the forwarded snap mode."""
        patch = dc.get_example_patch().update_attrs(tag=str(snap))
        return [
            make_scan_payload(
                attrs=patch.attrs,
                coords=patch.coords,
                dims=patch.dims,
                shape=patch.shape,
                dtype=str(patch.dtype),
            )
        ]


class _ReadOnlySummaryFormatter(FiberIO):
    """A formatter that relies on FiberIO.scan falling back to read()."""

    name = "_read_only_summary_formatter"
    version = "1"

    def read(self, resource: Path, snap_dims=True, **kwargs) -> dc.BaseSpool:
        """Return a simple spool for default scan conversion."""
        patch = dc.get_example_patch().update_attrs(tag="fallback")
        values = patch.get_coord("time").values.copy()
        values[len(values) // 2] += np.timedelta64(1, "ms")
        time = dc.get_coord(data=values)
        if snap_dims:
            time = time.snap()
        return dc.spool([patch.update_coords(time=time)])

    def get_format(self, resource: Path, **kwargs) -> tuple[str, str] | Literal[False]:
        """Only accept the explicit fallback-scan test resource."""
        path = Path(resource)
        if path.suffix == ".h5" and path.name == "fallback_scan.h5":
            return self.name, self.version
        return False


class _MissingOptionalFormatter(FiberIO):
    """A formatter whose scan path requires an unavailable optional dependency."""

    name = "_missing_optional_formatter"
    version = "1"

    def scan(self, resource: Path, **kwargs):
        """Raise a stable missing-optional error for scan coverage tests."""
        msg = (
            "not_optional_pkg is not installed but is required for the requested "
            "functionality."
        )
        raise MissingOptionalDependencyError(msg)

    def get_format(self, resource: Path, **kwargs) -> tuple[str, str] | Literal[False]:
        """Only accept the explicit missing-optional test resource."""
        path = Path(resource)
        if path.suffix == ".opt" and path.name == "missing_optional.opt":
            return self.name, self.version
        return False


class _ScanBehaviorFormatter(FiberIO):
    """A reader whose scan misbehaves in whichever way its file name asks.

    Module level, because registration is permanent and keyed by
    (name, version): a second definition of the same pair inside a test
    function is ignored and this one keeps answering. It claims only its
    own suffix, so no other get_format in the process changes.

    The alternative -- patching `scan` on the live terra15 reader -- edits
    an object every other test in the session shares.
    """

    name = "_scan_behavior_formatter"
    version = "1"

    def get_format(self, resource: Path, **kwargs) -> tuple[str, str] | Literal[False]:
        """Claim only this test's sentinel files."""
        return (
            (self.name, self.version)
            if Path(resource).suffix == ".scanbehavior"
            else False
        )

    def scan(self, resource: Path, **kwargs):
        """Do what the file name says, rather than scanning it."""
        behavior = Path(resource).stem
        if behavior == "os_error":
            raise OSError("Simulated OS issue")
        if behavior == "remote_cache_error":
            raise RemoteCacheError("metadata cache blocked")
        if behavior == "patch_attrs":  # the pre-ScanPayload return type
            return [dc.PatchAttrs(tag="legacy")]
        if behavior == "missing_keys":
            return [{"unexpected": 1}]
        if behavior == "non_mapping":
            return ["not a payload"]
        if behavior == "summary_coords":  # coords collapsed to summaries
            patch = dc.get_example_patch()
            return [
                {
                    "attrs": patch.attrs,
                    "coords": patch.coords.to_summary_dict(),
                    "dims": patch.dims,
                    "shape": patch.shape,
                    "dtype": str(patch.data.dtype),
                }
            ]
        msg = f"no scan behavior called {behavior!r}"
        raise LookupError(msg)


def _misbehaving_scan_path(tmp_path, behavior: str) -> Path:
    """Return a path _ScanBehaviorFormatter will scan in the named way."""
    path = tmp_path / f"{behavior}.scanbehavior"
    path.write_text("placeholder")
    return path


class _DependencyErrorFormatter(FiberIO):
    """A formatter whose scan path hits a dependency/compatibility problem."""

    name = "_dependency_error_formatter"
    version = "1"

    def scan(self, resource: Path, **kwargs):
        """Raise a stable dependency error for scan coverage tests."""
        msg = "simulated stack incompatibility while scanning"
        raise DependencyError(msg)

    def get_format(self, resource: Path, **kwargs) -> tuple[str, str] | Literal[False]:
        """Only accept the explicit dependency-error test resource."""
        path = Path(resource)
        if path.suffix == ".dep" and path.name == "dependency_error.dep":
            return self.name, self.version
        return False


class TestGetGriddedCoord:
    """Tests for forcing a stored array onto the grid it restates."""

    def test_quantized_array_becomes_a_range(self):
        """Quantization past get_coord's tolerance still yields a grid."""
        values = np.linspace(4000.0, 4009.9, 100, dtype=np.float32).astype(np.float64)
        assert isinstance(get_coord(data=values), CoordMonotonicArray)

        coord = get_gridded_coord(values, units="m")

        assert isinstance(coord, CoordRange)
        assert coord.min() == values[0]
        assert coord.max() == values[-1]
        assert len(coord) == len(values)

    def test_grid_depends_only_on_span_and_count(self):
        """Two arrays stating the same span must agree on every sample.

        One lands inside get_coord's evenness tolerance and one outside, so
        anchoring on what get_coord infers rather than on the stored values
        would make them disagree and stop them merging.
        """
        n = 2_000
        even = np.linspace(850.0, 1049.9, n)
        jittered = even.copy()
        jittered[1:-1] += np.random.default_rng(1).normal(0, 5e-4, n - 2)
        jittered[0], jittered[-1] = even[0], even[-1]
        assert isinstance(get_coord(data=even), CoordRange)
        assert isinstance(get_coord(data=jittered), CoordMonotonicArray)

        first = get_gridded_coord(even, units="m")
        second = get_gridded_coord(jittered, units="m")

        np.testing.assert_array_equal(first.values, second.values)

    def test_narrow_integer_range_does_not_overflow(self):
        """An integer grid spanning most of its dtype keeps its own range.

        Recomputing the span in the stored dtype would wrap it negative.
        """
        values = np.array([-30000, 0, 30000], dtype=np.int16)

        coord = get_gridded_coord(values, units="m")

        assert isinstance(coord, CoordRange)
        np.testing.assert_array_equal(coord.values, values)

    def test_single_sample_states_no_step(self):
        """One sample states no spacing, so none should be invented."""
        coord = get_gridded_coord(np.array([42.0]), units="m")

        assert coord.step is None
        assert coord.min() == 42.0

    def test_non_monotonic_values_preserved(self):
        """Values that are not a grid are returned untouched."""
        values = np.array([0.0, 5.0, 2.0, 9.0])

        coord = get_gridded_coord(values, units="m")

        np.testing.assert_array_equal(coord.values, values)


class TestGetExactCoord:
    """Tests for constructing exact coordinates during scans."""

    def test_preserves_irregular_monotonic_values(self):
        """Irregular monotonic arrays should retain every stored value."""
        values = np.array([0.0, 1.0, 2.0, 5.0, 6.0])

        coord = get_exact_coord(values, units="m")

        np.testing.assert_array_equal(coord.values, values)
        assert coord.units == dc.get_quantity("m")

    def test_preserves_non_monotonic_values(self):
        """Non-monotonic arrays should use the exact generic fallback."""
        values = np.array([0.0, 2.0, 1.0])

        coord = get_exact_coord(values)

        np.testing.assert_array_equal(coord.values, values)

    def test_jittery_array_does_not_over_segment(self):
        """Sub-step jitter must not explode into a per-sample segmented coord."""
        rng = np.random.default_rng(0)
        n = 5_000
        values = np.maximum.accumulate(
            np.arange(n) * 1000 + rng.integers(-3, 4, size=n)
        ).astype("datetime64[ns]")

        coord = get_exact_coord(values)

        # Values are preserved exactly, but the degenerate segmented form is
        # avoided (it would hold roughly n / 2 short segments).
        np.testing.assert_array_equal(coord.values, values)
        assert not isinstance(coord, CoordSegmented)

    def test_large_non_monotonic_array_preserved(self):
        """A large non-monotonic array skips the segment guard and stays exact."""
        rng = np.random.default_rng(1)
        values = rng.permutation(2_000).astype(float)

        coord = get_exact_coord(values, units="m")

        np.testing.assert_array_equal(coord.values, values)

    def test_piecewise_uniform_array_stays_segmented(self):
        """Genuinely piecewise-uniform arrays keep their queryable seams."""
        values = np.concatenate([np.arange(0.0, 2_000.0), np.arange(3_000.0, 5_000.0)])

        coord = get_exact_coord(values, units="m")

        assert isinstance(coord, CoordSegmented)
        np.testing.assert_array_equal(coord.values, values)
        assert len(coord.get_discontinuities("gaps")) == 1


class TestMakeScanPayload:
    """Tests for the shared scan payload constructor."""

    def test_dims_shape_derived_from_coords(self):
        """Omitted dims/shape come from the coords."""
        patch = dc.get_example_patch()

        payload = make_scan_payload(attrs=patch.attrs, coords=patch.coords, dtype="f8")

        assert payload["dims"] == patch.coords.dims
        assert payload["shape"] == patch.coords.shape

    def test_explicit_values_win(self):
        """Explicit dims/shape are not overwritten by the coords."""
        patch = dc.get_example_patch()

        payload = make_scan_payload(
            attrs=patch.attrs, coords=patch.coords, dims=(), shape=(), dtype="f8"
        )

        assert payload["dims"] == ()
        assert payload["shape"] == ()

    def test_explicit_shape_with_derived_dims(self):
        """Shape may be given while dims still come from the coords."""
        patch = dc.get_example_patch()
        shape = tuple(x - 1 for x in patch.coords.shape)

        payload = make_scan_payload(
            attrs=patch.attrs, coords=patch.coords, shape=shape, dtype="f8"
        )

        assert payload["dims"] == patch.coords.dims
        assert payload["shape"] == shape

    def test_attrs_none(self):
        """No attrs yields default PatchAttrs rather than raising."""
        patch = dc.get_example_patch()

        payload = make_scan_payload(attrs=None, coords=patch.coords, dtype="f8")

        assert isinstance(payload["attrs"], dc.PatchAttrs)


class TestBuildPatches:
    """Tests for the shared read tail used by format readers."""

    @pytest.fixture
    def patch(self):
        """A patch whose pieces feed build_patches."""
        return dc.get_example_patch()

    def test_no_selection_builds_patch(self, patch):
        """With nothing to select the whole patch comes back."""
        out = build_patches(patch.coords, patch.data, patch.attrs)

        assert len(out) == 1
        assert out[0].shape == patch.shape

    def test_none_selections_skip_select(self, patch, monkeypatch):
        """All-None selections must not touch the data source."""

        def _boom(*args, **kwargs):
            raise AssertionError("select should not be called")

        monkeypatch.setattr(type(patch.coords), "select", _boom)

        out = build_patches(
            patch.coords,
            patch.data,
            patch.attrs,
            selection={"time": None, "distance": None},
        )

        assert len(out) == 1

    def test_selection_trims(self, patch):
        """A selection trims the returned patch."""
        time = patch.get_coord("time")
        stop = time.min() + (time.max() - time.min()) / 2

        out = build_patches(
            patch.coords, patch.data, patch.attrs, selection={"time": (None, stop)}
        )

        assert (
            out[0].shape[patch.dims.index("time")]
            < patch.shape[patch.dims.index("time")]
        )

    def test_emptied_selection_returns_empty_list(self, patch):
        """A selection which removes all data yields no patches."""
        time = patch.get_coord("time")
        after_end = time.max() + np.timedelta64(10, "s")

        out = build_patches(
            patch.coords,
            patch.data,
            patch.attrs,
            selection={"time": (after_end, None)},
        )

        assert out == []

    def test_empty_data_returns_empty_list(self, patch):
        """An already empty source yields no patches, even untrimmed."""
        coords = get_coord_manager(
            {
                "time": get_coord(
                    start=np.datetime64("2020-01-01"),
                    step=np.timedelta64(1, "s"),
                    shape=(0,),
                ),
                "distance": get_coord(start=0.0, step=1.0, shape=(3,)),
            },
            dims=("time", "distance"),
        )

        out = build_patches(coords, np.empty(coords.shape), patch.attrs)

        assert out == []

    def test_attr_cls_used(self, patch):
        """The format's attrs class is applied to a plain dict."""

        class _MyAttrs(dc.PatchAttrs):
            """Attrs with one format specific field."""

            my_field: float = np.nan

        out = build_patches(
            patch.coords, patch.data, {"my_field": 2.0}, attr_cls=_MyAttrs
        )

        assert isinstance(out[0].attrs, _MyAttrs)
        assert out[0].attrs.my_field == 2.0

    def test_attrs_none(self, patch):
        """No attrs yields a patch with default attrs."""
        out = build_patches(patch.coords, patch.data)

        assert isinstance(out[0].attrs, dc.PatchAttrs)


class TestScanResultToSummary:
    """Tests for converting scan metadata into summaries."""

    def test_scan_payload_dict_input_builds_summary(self):
        """Structured scan payloads should normalize to PatchSummary."""
        patch = dc.get_example_patch()
        payload = {
            "attrs": patch.attrs,
            "coords": patch.coords,
            "dims": patch.dims,
            "shape": patch.shape,
            "dtype": str(patch.data.dtype),
            "source_patch_key": "node-1",
        }
        out = _scan_result_to_summary(payload, source_path="some_path")
        assert isinstance(out, dc.PatchSummary)
        assert (
            out.get_coord_summary("time").fingerprint
            == patch.get_coord("time").fingerprint()
        )
        assert out.source_patch_key == "node-1"
        assert str(out.source_path) == "some_path"

    def test_scan_payload_missing_dtype_raises(self):
        """Structured scan payloads should require dtype metadata."""
        patch = dc.get_example_patch()
        payload = {
            "attrs": patch.attrs,
            "coords": patch.coords,
            "dims": patch.dims,
            "shape": patch.shape,
        }
        msg = r"requires a mapping with `coords`, `attrs`, and `dtype`"
        with pytest.raises(TypeError, match=msg):
            _scan_result_to_summary(payload, source_path="some_path")

    def test_make_scan_payload_uses_dtype_key(self):
        """The helper should emit the normalized dtype field."""
        patch = dc.get_example_patch()
        out = make_scan_payload(
            attrs=patch.attrs,
            coords=patch.coords,
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(patch.data.dtype),
        )
        assert out["dtype"] == str(patch.data.dtype)

    def test_invalid_dict_input_raises(self):
        """Untyped dict payloads should still be rejected."""
        msg = r"requires a mapping with `coords`, `attrs`, and `dtype`"
        with pytest.raises(TypeError, match=msg):
            _scan_result_to_summary({"tag": "x"})

    def test_invalid_non_mapping_input_raises(self):
        """Unsupported scan outputs should mention allowed input shapes."""
        msg = "only accepts PatchSummary or structured scan payload mappings"
        with pytest.raises(TypeError, match=msg):
            _scan_result_to_summary("bad scan output")

    def test_patch_attrs_input_raises(self):
        """PatchAttrs scan outputs should fail with a migration hint."""
        patch = dc.get_example_patch()
        msg = (
            "DASCore no longer accepts PatchAttrs from FiberIO.scan\\(\\).*"
            "docs/contributing/new_format.qmd"
        )
        with pytest.raises(ValueError, match=msg):
            _scan_result_to_summary(patch.attrs)

    def test_summary_source_patch_key_sets_private_attr(self):
        """Summary source ids should be copied onto private attrs."""
        summary = dc.PatchSummary(
            attrs=dc.PatchAttrs(tag="x"),
            source_patch_key="node-1",
        )
        assert summary.source_patch_key == "node-1"
        assert summary.attrs["_source_patch_key"] == "node-1"

    def test_private_attr_source_patch_key_sets_summary(self):
        """Private attr source ids should populate the summary field."""
        summary = dc.PatchSummary(
            attrs=dc.PatchAttrs(tag="x", _source_patch_key="node-2"),
        )
        assert summary.source_patch_key == "node-2"
        assert summary.attrs["_source_patch_key"] == "node-2"

    def test_summary_source_patch_key_wins_on_conflict(self):
        """Conflicting ids should resolve in favor of the summary field."""
        summary = dc.PatchSummary(
            attrs=dc.PatchAttrs(tag="x", _source_patch_key="attrs-id"),
            source_patch_key="summary-id",
        )
        assert summary.source_patch_key == "summary-id"
        assert summary.attrs["_source_patch_key"] == "summary-id"


class TestFormatManager:
    """Tests for the format manager."""

    @pytest.fixture(scope="class")
    def format_manager(self):
        """Deep copy manager to avoid changing state used by other objects."""
        manager = copy.deepcopy(FiberIO.manager)
        return manager

    def test_inherited_scan_delegates_to_read(self):
        """A FiberIO which implements only read still scans, snap and all."""
        fiber_io = _FiberInheritsScan()
        fiber_io.seen_snap.clear()
        assert len(fiber_io.scan("ignored")) == 1
        assert len(fiber_io.scan(resource="ignored", snap=False)) == 1
        assert fiber_io.seen_snap == [True, False]

    def test_get_fiberio_needs_a_registry(self):
        """get_fiberio promises a FiberIO; an empty registry cannot supply one."""
        manager = _FiberIOManager("dascore.fiber_io")
        manager.__dict__["_eps"] = pd.Series({}, dtype=object)
        with pytest.raises(AssertionError, match="no fiber_io"):
            manager.get_fiberio()

    def test_specific_format_and_version(self, format_manager):
        """
        Specifying a known format and version should return exactly one formatter.
        """
        out = list(format_manager.yield_fiberio("DASDAE", "1"))
        assert len(out) == 1
        assert isinstance(out[0], DASDAEV1)

    def test_get_all_formatters(self, format_manager):
        """Ensure getting all formatters through yield_fiberio works."""
        formatters = list(format_manager.yield_fiberio())
        assert len(formatters) >= len(format_manager._eps)

    def test_extension_priority(self, format_manager):
        """Ensure the extension priority is honored."""
        ext = "h5"
        ext_formatters = list(format_manager.yield_fiberio(extension=ext))
        all_formatters = list(format_manager.yield_fiberio())
        in_formatter = [ext in x.preferred_extensions for x in ext_formatters]
        format_array = np.array(in_formatter).astype(bool)
        # ensure all the start of the arrays are True.
        assert np.argmin(format_array) == np.sum(format_array)
        # ensure all formats are represented.
        assert len(format_array) == len(all_formatters)
        # ensure V2 of the Test formatter appears first
        v2_arg = np.argmax([isinstance(x, _FiberImplementer) for x in ext_formatters])
        v1_arg = np.argmax([isinstance(x, _FiberFormatTestV1) for x in ext_formatters])
        assert v2_arg < v1_arg

    def test_format_raises_unknown_format(self, format_manager):
        """Ensure we raise for unknown formats."""
        with pytest.raises(UnknownFiberFormatError, match="format"):
            list(format_manager.yield_fiberio(format="bob_2"))

    def test_format_raises_just_version(self, format_manager):
        """Providing only a version should also raise."""
        with pytest.raises(UnknownFiberFormatError, match="version"):
            list(format_manager.yield_fiberio(version="1"))

    def test_format_bad_version(self, format_manager):
        """Ensure providing a bad version but valid format raises."""
        with pytest.raises(UnknownFiberFormatError, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_format_no_version(self, format_manager):
        """Ensure providing a bad version but valid format raises."""
        with pytest.raises(UnknownFiberFormatError, match="known versions"):
            iterator = format_manager.yield_fiberio(format="DASDAE", version="-1")
            list(iterator)

    def test_format_multiple_versions(self, format_manager):
        """Ensure multiple versions are returned when only format is specified."""
        file_format = _FiberFormatTestV1.name
        out = list(format_manager.yield_fiberio(format=file_format))
        assert len(out) == 2

    def test_unique_values_extensions(self, format_manager):
        """Ensure unique FiberIO are returned for an extension."""
        out = list(format_manager.yield_fiberio(extension="h5"))
        name_ver = [(x.name, x.version) for x in out]
        assert len(name_ver) == len(set(name_ver))

    def test_unique_values_no_extensions(self, format_manager):
        """Ensure unique FiberIO are returned when nothing specified."""
        out = list(format_manager.yield_fiberio())
        name_ver = [(x.name, x.version) for x in out]
        assert len(name_ver) == len(set(name_ver))

    def test_known_formats_empty_entry_points(self, format_manager):
        """Known formats should tolerate an empty/non-string entry-point index."""
        format_manager.__dict__.pop("_eps", None)
        format_manager.__dict__.pop("known_formats", None)
        format_manager._eps = pd.Series(dtype=object)
        assert isinstance(format_manager.known_formats, frozenset)

    def test_load_plugins_empty_entry_points(self, format_manager):
        """Loading plugins should no-op when no entry points are present."""
        format_manager.__dict__.pop("_eps", None)
        format_manager.__dict__.pop("known_formats", None)
        format_manager._eps = pd.Series(dtype=object)
        format_manager.load_plugins()

    def test_load_entry_point_warns_on_broken_plugin(self, format_manager):
        """Broken plugin loaders should warn and return None."""

        def loader():
            raise ImportError("boom")

        with pytest.warns(UserWarning, match="Failed to load FiberIO plugin 'broken'"):
            out = format_manager._load_entry_point("broken", loader)

        assert out is None

    def test_load_entry_point_warns_on_runtime_plugin_error(self, format_manager):
        """Runtime plugin construction failures should warn and be skipped."""

        def loader():
            raise RuntimeError("boom")

        with pytest.warns(UserWarning, match="RuntimeError: boom"):
            out = format_manager._load_entry_point("broken", loader)

        assert out is None

    def test_prioritized_list_skips_unloaded_formats(self, format_manager):
        """Formats with no registered versions should not break prioritization."""
        manager = type(format_manager)("dascore.fiber_io")
        manager.__dict__.pop("known_formats", None)
        manager._eps = pd.Series(
            {
                "BROKEN__V1": lambda: (_ for _ in ()).throw(ImportError("boom")),
                "GOOD__V1": lambda: _ReadOnlySummaryFormatter,
            }
        )

        prioritized = manager._get_prioritized_list()

        registered = manager._format_version[_ReadOnlySummaryFormatter.name.upper()]
        assert "1" in registered
        assert isinstance(registered["1"], _ReadOnlySummaryFormatter)
        assert isinstance(prioritized, tuple)


class TestBrokenEntryPoint:
    """Tests for graceful handling of unloadable FiberIO entry points."""

    @pytest.fixture()
    def broken_ep_manager(self):
        """Return a manager with an entry point that cannot be loaded."""
        manager = copy.deepcopy(FiberIO.manager)

        def bad_loader():
            raise ModuleNotFoundError("No module named 'dascore.io.not_real'")

        eps = manager._eps.copy()
        eps["NOT_REAL_FORMAT__V1"] = bad_loader
        manager.__dict__["_eps"] = eps
        manager.__dict__.pop("known_formats", None)
        # clear the method cache so load_plugins runs again for this instance.
        manager.__dict__.pop("_cache", None)
        # the copy inherits the original's "everything is loaded" state,
        # which the swapped-in entry points invalidate.
        manager._all_loaded = False
        return manager

    def test_load_plugins_warns_and_skips(self, broken_ep_manager):
        """A stale/broken entry point should warn, not raise."""
        with pytest.warns(UserWarning, match="Failed to load FiberIO"):
            broken_ep_manager.load_plugins()
        assert "NOT_REAL_FORMAT" not in broken_ep_manager.unloaded_formats

    def test_other_formats_still_usable(self, broken_ep_manager):
        """Remaining FiberIOs should work after a plugin fails to load."""
        with pytest.warns(UserWarning, match="Failed to load FiberIO"):
            broken_ep_manager.load_plugins()
        out = list(broken_ep_manager.yield_fiberio("DASDAE", "1"))
        assert len(out) == 1
        # The prioritized list (used by scan/read) must also not choke.
        assert len(list(broken_ep_manager.yield_fiberio()))


class TestGetFormatErrors:
    """Errors which must not be mistaken for a format mismatch."""

    def test_remote_cache_error_propagates(self, monkeypatch, tmp_path):
        """A remote fetch failure is a real error, not a wrong-format signal.

        The loop over FiberIOs swallows exceptions so a reader which does
        not recognize a file can be skipped; a cache failure has to escape
        that handler instead of being reported as an unknown format.
        """
        path = tmp_path / "unfetchable.h5"
        path.write_bytes(b"not really an h5 file")

        def _raise(*args, **kwargs):
            raise RemoteCacheError("cannot fetch this resource")

        monkeypatch.setattr(IOResourceManager, "get_resource", _raise)
        with pytest.raises(RemoteCacheError, match="cannot fetch this resource"):
            dc.get_format(path)


class TestFormatManagerConcurrency:
    """Concurrent plugin loading must never expose a partial registry."""

    def _make_manager(self, eps):
        """Return a manager whose entry points are the provided loaders."""
        manager = _FiberIOManager("dascore.fiber_io")
        manager.__dict__["_eps"] = pd.Series(eps)
        return manager

    @pytest.mark.concurrency
    def test_multi_version_format_never_partial(self, run_in_threads):
        """Every thread sees all versions, even mid-load."""
        entered, release, calls = threading.Event(), threading.Event(), []

        def slow_loader():
            """Stall inside the first version's loader."""
            calls.append("v1")
            entered.set()
            release.wait()
            return _FiberFormatTestV1

        manager = self._make_manager(
            {
                "_TESTFORMATTER__V1": slow_loader,
                "_TESTFORMATTER__V2": lambda: _FiberFormatTestV2,
            }
        )
        # Free the stalled loader once it has registered nothing but v1;
        # the other threads are queued behind the manager lock by then.
        releaser = threading.Thread(target=lambda: (entered.wait(), release.set()))
        releaser.start()
        results = run_in_threads(
            lambda _: tuple(
                x.version for x in manager.yield_fiberio(format="_TestFormatter")
            ),
            4,
        )
        releaser.join()
        # Newest version first, and the loader ran exactly once.
        assert all(x == ("2", "1") for x in results)
        assert calls == ["v1"]

    @pytest.mark.concurrency
    def test_concurrent_full_load_runs_each_loader_once(self, run_in_threads):
        """Loading all formats from several threads loads each entry point once."""
        calls = []

        def make_loader(fiber_io):
            """Return a loader which records that it ran."""

            def loader():
                calls.append(fiber_io.version)
                return fiber_io

            return loader

        manager = self._make_manager(
            {
                "_TESTFORMATTER__V1": make_loader(_FiberFormatTestV1),
                "_TESTFORMATTER__V2": make_loader(_FiberFormatTestV2),
            }
        )
        results = run_in_threads(lambda _: len(list(manager.yield_fiberio())))
        assert set(results) == {2}
        assert sorted(calls) == ["1", "2"]

    def test_snapshots_are_immutable(self):
        """Cached lookups hand back immutable snapshots."""
        manager = self._make_manager({"_TESTFORMATTER__V1": lambda: _FiberFormatTestV1})
        assert isinstance(manager.known_formats, frozenset)
        assert isinstance(manager._get_prioritized_list(), tuple)
        assert isinstance(manager._get_fiber_io_by_input_type("file"), frozenset)

    def test_copy_gets_own_lock(self):
        """A copied manager must not share the original's lock."""
        manager = self._make_manager({"_TESTFORMATTER__V1": lambda: _FiberFormatTestV1})
        copied = copy.deepcopy(manager)
        assert copied._lock is not manager._lock
        assert list(copied.yield_fiberio(format="_TestFormatter"))

    def test_fork_handler_replaces_held_lock(self):
        """A lock held at fork time is replaced so the child cannot deadlock."""
        manager = FiberIO.manager
        old_lock = manager._lock
        try:
            with old_lock:
                _reinit_manager_lock()
                new_lock = manager._lock
                # The replacement is free even while the old lock is held.
                assert new_lock.acquire(blocking=False)
                new_lock.release()
            assert new_lock is not old_lock
        finally:
            manager._lock = old_lock


class TestFormatter:
    """Tests for adding file supports through Formatter."""

    # the methods a formatter can implement.

    class FormatterWithName(FiberIO):
        """A formatter with a file name."""

        name = "_test_format"

    def test_empty_formatter_raises(self):
        """An empty formatter can't exist; it at least needs a name."""
        with pytest.raises(InvalidFiberIOError):

            class EmptyFormatter(FiberIO):
                """formatter with no name."""

    def test_empty_formatter_undefined_methods(self, random_patch):
        """
        Ensure a Not Implemented error is raised for un-implemented methods
        of FormatterWithName.
        """
        instance = self.FormatterWithName()
        with pytest.raises(NotImplementedError):
            instance.read("empty_path")
        with pytest.raises(NotImplementedError):
            instance.write(random_patch, "empty_path")
        with pytest.raises(NotImplementedError):
            instance.get_format("empty_path")
        with pytest.raises(NotImplementedError):
            instance.scan("bad_path")

    def test_doesnt_implements(self):
        """Tests for implements_x methods."""
        # this test fiber io don't implement anything
        fio = _FiberFormatTestV1()
        assert not fio.implements_scan
        assert not fio.implements_get_format
        assert not fio.implements_read
        assert not fio.implements_write

    def test_implements(self):
        """Tests for implements_x methods."""
        # this test fiber implements all the things
        fio = _FiberImplementer()
        assert fio.implements_scan
        assert fio.implements_get_format
        assert fio.implements_read
        assert fio.implements_write


class TestGetFormat:
    """Tests to ensure formats can be retrieved."""

    @pytest.fixture(scope="class")
    def empty_h5_path(self, tmpdir_factory):
        """Create an empty HDF5 file."""
        path = tmpdir_factory.mktemp("empty") / "empty.h5"
        with h5py.File(path, "w"):
            pass
        return path

    def test_empty_hdf5_no_format(self, empty_h5_path):
        """Ensure the empty hdf5 doesn't have a format."""
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(empty_h5_path)

    def test_not_known(self, dummy_text_file):
        """Ensure a non-path/str object raises."""
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(dummy_text_file)

    def test_missing_file(self):
        """Ensure a missing file raises."""
        with pytest.raises(FileNotFoundError):
            dc.get_format("bad/file")

    def test_fiberio_directory(self, tmp_path_factory):
        """Ensure a directory can be recognized as a FiberIO."""
        fiber_io = _FiberDirectory()
        path = tmp_path_factory.mktemp(fiber_io.name)
        assert fiber_io.get_format(path)
        (name, version) = dc.get_format(path)
        assert fiber_io.name == name
        assert fiber_io.version == version

    def test_manager_get_format_invokes_fiberio_get_format(self, monkeypatch, tmp_path):
        """Manager format detection should execute FiberIO get_format loop bodies."""
        path = tmp_path / "format_loop.h5"
        path.write_text("placeholder")
        fiber_io = _ReadOnlySummaryFormatter()
        seen = {}

        def _yield_fiberio(*_args, **_kwargs):
            yield fiber_io

        def _get_format(resource, **_kwargs):
            seen["resource"] = resource
            return (fiber_io.name, fiber_io.version)

        monkeypatch.setattr(FiberIO.manager, "yield_fiberio", _yield_fiberio)
        monkeypatch.setattr(fiber_io, "get_format", _get_format)
        fiber_io.get_format._required_type = Path

        assert FiberIO.manager._get_format(path=path) == (
            fiber_io.name,
            fiber_io.version,
        )
        assert seen["resource"] == path


class TestFileUri:
    """Local file:// URIs should behave like plain local paths."""

    @pytest.fixture(scope="class")
    def dasdae_path(self, tmp_path_factory):
        """Write an example patch to a local dasdae file."""
        path = tmp_path_factory.mktemp("file_uri") / "patch.h5"
        dc.get_example_patch().io.write(path, "dasdae")
        return path

    @pytest.fixture(scope="class")
    def file_uri(self, dasdae_path):
        """Return the file:// URI form of the local dasdae file."""
        return dasdae_path.resolve().as_uri()

    def test_get_format(self, file_uri, dasdae_path):
        """get_format should agree for path and file:// URI."""
        assert dc.get_format(file_uri) == dc.get_format(dasdae_path)

    def test_scan(self, file_uri, dasdae_path):
        """Scan should succeed for a file:// URI."""
        assert len(dc.scan(file_uri)) == len(dc.scan(dasdae_path))

    def test_read(self, file_uri, dasdae_path):
        """Read via file:// URI should equal read via plain path."""
        assert dc.read(file_uri)[0] == dc.read(dasdae_path)[0]


class TestScan:
    """Tests for scanning fiber files."""

    @pytest.fixture(scope="class")
    def nested_directory_with_patches(self, tmpdir_factory, random_patch):
        """Return a nested directory with patch files interlaced."""
        out = Path(tmpdir_factory.mktemp("nested_random_patch"))
        path_1 = out / "patch_1.h5"
        path_2 = out / "subdir" / "patch_2.h5"
        path_3 = out / "subdir" / "suber_dir" / "patch_3.h5"
        random_patch.io.write(path_1, "dasdae")
        random_patch.io.write(path_2, "dasdae")
        random_patch.io.write(path_3, "dasdae")
        return out

    @pytest.fixture(scope="class")
    def two_files(self, tmp_path_factory, random_patch):
        """Two patch files, for the tests which only scan them."""
        path = tmp_path_factory.mktemp("two_files")
        paths = (path / "patch_1.h5", path / "patch_2.h5")
        for each in paths:
            random_patch.io.write(each, "dasdae")
        return paths

    @pytest.mark.parametrize("func", [dc.scan, dc.scan_to_df, dc.scan_payloads])
    def test_scan_accepts_a_collection(self, func, two_files):
        """A collection of resources scans as the sum of its members."""
        path_1, path_2 = two_files
        expected = len(func(path_1)) + len(func(path_2))
        assert len(func([path_1, path_2])) == expected
        # A set is a collection too, and the dispatcher does not index.
        assert len(func({path_1, path_2})) == expected

    def test_scan_accepts_a_collection_of_patches(self, random_patch):
        """Patches can be scanned directly, one summary each."""
        assert len(dc.scan([random_patch, random_patch])) == 2

    @pytest.mark.parametrize("func", [dc.scan, dc.scan_to_df, dc.scan_payloads])
    def test_scan_accepts_a_one_shot_iterable(self, func, two_files):
        """A generator input scans every element, not silently nothing (#818)."""
        path_1, path_2 = two_files
        expected = len(func([path_1, path_2]))
        assert expected == 2
        assert len(func(p for p in [path_1, path_2])) == expected
        assert len(func(iter([path_1, path_2]))) == expected

    def test_scan_accepts_a_generator_of_patches(self, random_patch):
        """A generator of patches yields one summary each (#818)."""
        assert len(dc.scan(p for p in [random_patch, random_patch])) == 2

    def test_scan_no_good_files(self, tmp_path):
        """Scan with no fiber files should return []."""
        dummy_file = tmp_path / "data.txt"
        dummy_file.touch()
        out = dc.scan(dummy_file)
        assert not len(out)
        assert out == []

    def test_scan_directory(self, tmp_path):
        """Trying to scan an empty directory should return empty list."""
        out = dc.scan(tmp_path)
        assert len(out) == 0

    def test_scan_payloads_directory_forwards_snap(self, tmp_path):
        """Exact payload scans should forward snap to directory formatters."""
        path = tmp_path / _FiberDirectory.name
        path.mkdir()

        out = dc.scan_payloads(path, snap=False)

        assert len(out) == 1
        assert out[0]["attrs"].tag == "False"

    def test_scan_bad_files(self, tmp_path):
        """Trying to scan a directory should raise a nice error."""
        new = tmp_path / "myfile.txt"
        with pytest.raises(FileNotFoundError):
            _ = dc.scan(new)

    def test_scan_patch(self, random_patch):
        """Scan should also work on a patch."""
        out = dc.scan_to_df(random_patch)
        summary = random_patch.summary
        assert len(out) == 1
        ser = out.iloc[0]
        time_summary = summary.get_coord_summary("time")
        assert to_datetime64(ser["time_min"]) == to_datetime64(time_summary.min)
        assert to_datetime64(ser["time_max"]) == to_datetime64(time_summary.max)

    def test_scan_patch_returns_summary(self, random_patch):
        """Direct patch scan should normalize to a PatchSummary."""
        out = dc.scan(random_patch)
        assert len(out) == 1
        scanned = out[0]
        assert isinstance(scanned, dc.PatchSummary)
        assert scanned.dtype == str(random_patch.dtype)
        assert not scanned.source_patch_key

    def test_scan_payloads_patch_returns_full_coords(self, random_patch):
        """Direct patch payload scans should retain full coordinate values."""
        out = dc.scan_payloads(random_patch)

        assert len(out) == 1
        payload = out[0]
        assert isinstance(payload["coords"], dc.CoordManager)
        assert payload["coords"] == random_patch.coords
        assert payload["source_path"] == ""
        assert payload["source_format"] == ""
        assert payload["source_version"] == ""

    def test_scan_payloads_spool_returns_each_patch(self, random_patch):
        """Spool inputs should produce one raw payload per patch."""
        spool = dc.spool(
            [random_patch.update_attrs(tag="one"), random_patch.update_attrs(tag="two")]
        )

        out = dc.scan_payloads(spool)

        assert [payload["attrs"].tag for payload in out] == ["one", "two"]
        assert all(isinstance(payload["coords"], dc.CoordManager) for payload in out)

    def test_scan_multi_patch_includes_source_patch_key(self, tmp_path):
        """Multi-patch scan rows should include a stable source patch id."""
        path = tmp_path / "multi_patch.h5"
        spool = dc.examples.get_example_spool("random_das", length=2)
        dc.write(spool, path, "DASDAE", file_version="1")
        out = dc.scan_to_df(path)
        assert "source_patch_key" in out.columns
        assert out["source_patch_key"].astype(bool).all()

    def test_scan_nested_directory(self, nested_directory_with_patches):
        """Ensure scan picks up files in nested directories."""
        out = dc.scan(nested_directory_with_patches)
        assert len(out) == 3

    def test_scan_single_file(self, terra15_v6_path):
        """Ensure scan works on a single file."""
        out = dc.scan(terra15_v6_path)
        assert len(out) == 1

    def test_scan_missing_optional_dependency_raises(self, tmp_path):
        """Scan should raise if optional deps are missing and nothing else loads."""
        path = tmp_path / "missing_optional.opt"
        path.write_text("placeholder")

        msg = "found files that can be read if additional packages"
        with pytest.raises(MissingOptionalDependencyError, match=msg) as exc_info:
            dc.scan(path)
        # The message should say how to install the missing package.
        assert "pip install not_optional_pkg" in str(exc_info.value)

    def test_scan_missing_optional_dependency_warns_with_other_outputs(self, tmp_path):
        """Scan should warn if optional deps are missing but other files load."""
        missing_path = tmp_path / "missing_optional.opt"
        missing_path.write_text("placeholder")
        readable_path = tmp_path / "fallback_scan.h5"
        readable_path.write_text("placeholder")

        msg = "found files that can be read if additional packages"
        with pytest.warns(UserWarning, match=msg):
            out = dc.scan([missing_path, readable_path])

        assert len(out) == 1
        assert out[0].source_format == _ReadOnlySummaryFormatter.name

    def test_scan_dependency_error_warns_and_skips(self, tmp_path):
        """
        A scan-time dependency/compatibility problem warns and keeps scanning.

        Scan is best-effort across many resources, so such problems must
        surface as warnings on the affected file rather than aborting the
        whole scan. (The DASVader legacy file used to exercise this branch,
        but whether it does depends on the installed HDF5 stack.)
        """
        dep_path = tmp_path / "dependency_error.dep"
        dep_path.write_text("placeholder")
        readable_path = tmp_path / "fallback_scan.h5"
        readable_path.write_text("placeholder")

        with pytest.warns(UserWarning, match="simulated stack incompatibility"):
            out = dc.scan([dep_path, readable_path])

        assert len(out) == 1
        assert out[0].source_format == _ReadOnlySummaryFormatter.name

    def test_local_upath_file_interfaces(self, terra15_v6_path):
        """Ensure core file IO accepts local UPath inputs."""
        path = UPath(terra15_v6_path)
        file_format, file_version = dc.get_format(path)
        assert file_format
        assert file_version
        assert len(dc.scan(path)) == 1
        assert len(dc.read(path)) == 1

    def test_updated_after_warns_when_remote_mtime_missing(self, monkeypatch):
        """Timestamp filtering should warn and continue on unsupported backends."""
        fiber_io = _FiberFormatTestV1()
        resource = UPath("memory://dascore/fiberio/mtime.txt")
        resource.write_text("x")
        path_type = type(resource)
        original_stat = path_type.stat

        def _stat(self, *args, **kwargs):
            raise OSError("no mtime")

        monkeypatch.setattr(path_type, "stat", _stat)
        with pytest.warns(UserWarning, match="does not expose reliable mtime"):
            assert fiber_io._updated_after(resource, 1) is True
        monkeypatch.setattr(path_type, "stat", original_stat)

    def test_local_stat_failure_returns_false(self, monkeypatch, tmp_path):
        """Local stat failures should conservatively skip timestamp-matched scans.

        Remote backends sometimes cannot provide mtimes at all, so DASCore warns
        and continues scanning in that case. For local files, a failed ``stat``
        usually means the path disappeared or became unreadable, so
        ``_updated_after`` should return ``False`` instead of treating that as an
        implicit update.
        """
        fiber_io = _FiberFormatTestV1()
        path = tmp_path / "mtime.txt"
        path.write_text("x")

        def _stat(_self, *args, **kwargs):
            raise OSError("no stat")

        monkeypatch.setattr(Path, "stat", _stat)
        with suppress_warnings(action="always", record=True) as record:
            assert fiber_io._updated_after(path, 1) is False
        assert not record


class TestReloadableSourcePath:
    """Tests for reloading source path extraction."""

    def test_io_resource_manager_source(self, tmp_path):
        """IOResourceManager candidates should resolve to their source path."""
        path = tmp_path / "example.txt"
        path.write_text("x")
        manager = IOResourceManager(path)
        out = _get_reloadable_source_path(manager)
        assert isinstance(out, UPath)
        assert out == UPath(path)

    def test_can_raise(self):
        """
        Scan, when called from a FiberIO, should be able to raise if
        type coercion fails.
        """
        fio = _FiberImplementer()
        bad_input = _FiberFormatTestV1()
        with pytest.raises(NotImplementedError):
            fio.scan(bad_input)

    def test_bad_checksum(self, tmp_path):
        """Test for when format is identified but can't read part of file #346"""
        path = _misbehaving_scan_path(tmp_path, "os_error")
        # Ensure scanning doesn't raise and warns
        msg = "Failed to scan"
        with pytest.warns(UserWarning, match=msg):
            scan = dc.scan(path)
        assert not len(scan)

    def test_remote_cache_error_is_not_swallowed(self, tmp_path):
        """Remote cache policy errors during scan should propagate to callers."""
        path = _misbehaving_scan_path(tmp_path, "remote_cache_error")
        with pytest.raises(RemoteCacheError, match="metadata cache blocked"):
            dc.scan(path)

    def test_scan_legacy_patch_attrs_raises(self, tmp_path):
        """FiberIO returning PatchAttrs should now fail loudly."""
        path = _misbehaving_scan_path(tmp_path, "patch_attrs")
        with pytest.raises(ValueError, match=r"PatchAttrs from FiberIO\.scan"):
            dc.scan(path)

    def test_scan_payloads_legacy_patch_attrs_raises(self, tmp_path):
        """Raw payload scans should reject legacy summary-only results."""
        path = _misbehaving_scan_path(tmp_path, "patch_attrs")
        with pytest.raises(ValueError, match="no longer accepts PatchAttrs"):
            dc.scan_payloads(path)

    def test_scan_payloads_missing_keys_raises(self, tmp_path):
        """Raw payload scans should validate all required payload keys."""
        path = _misbehaving_scan_path(tmp_path, "missing_keys")
        with pytest.raises(TypeError, match="missing required keys"):
            dc.scan_payloads(path)

    def test_scan_payloads_non_mapping_raises(self, tmp_path):
        """Raw payload scans should reject unsupported result types."""
        path = _misbehaving_scan_path(tmp_path, "non_mapping")
        with pytest.raises(TypeError, match="must return ScanPayload mappings"):
            dc.scan_payloads(path)

    def test_scan_payloads_requires_coord_manager(self, tmp_path):
        """Raw payload scans should reject collapsed coordinate summaries."""
        path = _misbehaving_scan_path(tmp_path, "summary_coords")
        with pytest.raises(TypeError, match="must be a CoordManager"):
            dc.scan_payloads(path)

    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("attrs", "not-attrs"),
            ("attrs", {"tag": ["not-a-string"]}),
            ("coords", object()),
            ("dims", "time"),
            ("dims", ("time", "")),
            ("dims", ("time", "time")),
            ("shape", [1, 2]),
            ("shape", (1, -1)),
            ("dtype", np.dtype("float64")),
            ("dtype", "not-a-dtype"),
            ("source_patch_key", 1),
            ("source_path", 1),
            ("source_format", Path("format")),
            ("source_version", None),
        ],
    )
    def test_scan_payload_field_validation(self, key, value):
        """Every declared payload field should enforce its public type."""
        patch = dc.get_example_patch()
        payload = make_scan_payload(
            attrs=patch.attrs,
            coords=patch.coords,
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(patch.dtype),
        )
        payload[key] = value

        with pytest.raises(TypeError, match=key):
            _validate_scan_payload(payload)

    @pytest.mark.parametrize("key", ["dims", "shape"])
    def test_scan_payload_coord_metadata_must_match(self, key):
        """Strict payload metadata must agree with the full coord manager."""
        patch = dc.get_example_patch()
        payload = make_scan_payload(
            attrs=patch.attrs,
            coords=patch.coords,
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(patch.dtype),
        )
        if key == "dims":
            payload[key] = tuple(reversed(patch.dims))
        else:
            payload[key] = (patch.shape[0] + 1, *patch.shape[1:])

        with pytest.raises(ValueError, match=rf"`{key}` must exactly match"):
            _validate_scan_payload(payload, require_coord_manager=True)

    def test_scan_payloads_normalizes_attrs(self, monkeypatch, terra15_v6_path):
        """The public payload API should always return PatchAttrs."""
        fname, ver = FiberIO.manager._get_format(path=terra15_v6_path)
        fiber_io = FiberIO.manager.get_fiberio(format=fname, version=ver)
        patch = dc.get_example_patch()
        payload = make_scan_payload(
            attrs=patch.attrs,
            coords=patch.coords,
            dims=patch.dims,
            shape=patch.shape,
            dtype=str(patch.dtype),
        )
        payload["attrs"] = patch.attrs.model_dump()
        monkeypatch.setattr(fiber_io, "scan", lambda *args, **kwargs: [payload])

        out = dc.scan_payloads(terra15_v6_path)

        assert isinstance(out[0]["attrs"], dc.PatchAttrs)

    def test_default_fiberio_scan_uses_reloadable_source_path(self, tmp_path):
        """Default FiberIO.scan should return structured scan payloads."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()

        out = fio.scan(path)

        assert len(out) == 1
        assert isinstance(out[0], dict)
        assert "source_path" not in out[0]
        assert "source_format" not in out[0]
        assert "source_version" not in out[0]
        assert not out[0]["source_patch_key"]

    def test_default_fiberio_scan_forwards_snap_dims(self, tmp_path):
        """Default scans should forward exact-coordinate mode to read()."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()

        exact = fio.scan(path, snap=False)[0]["coords"].get_coord("time")
        snapped = fio.scan(path, snap=True)[0]["coords"].get_coord("time")
        read_exact = fio.read(path, snap_dims=False)[0].get_coord("time")

        np.testing.assert_array_equal(exact.values, read_exact.values)
        assert not np.array_equal(exact.values, snapped.values)

    def test_default_fiberio_scan_forwards_snap(self, monkeypatch, tmp_path):
        """Default scans should also support a read() snap parameter."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()
        seen = {}

        def read(resource, snap=True):
            seen["snap"] = snap
            return dc.spool([dc.get_example_patch()])

        monkeypatch.setattr(fio, "read", read)

        fio.scan(path, snap=False)

        assert seen["snap"] is False

    def test_default_fiberio_scan_forwards_read_kwargs(self, monkeypatch, tmp_path):
        """Default scans should preserve reader filters and override snap mode."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()
        seen = {}

        def read(resource, snap_dims=True, **kwargs):
            seen.update(kwargs)
            seen["snap_dims"] = snap_dims
            return dc.spool([dc.get_example_patch()])

        monkeypatch.setattr(fio, "read", read)

        fio.scan(path, snap=False, snap_dims=True, time=(1, 2), custom="value")

        assert seen == {
            "snap_dims": False,
            "time": (1, 2),
            "custom": "value",
        }

    def test_dc_scan_adds_source_metadata_to_raw_fiberio_scan(self, tmp_path):
        """dc.scan should add path/format/version on top of raw formatter scan."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")

        raw = _ReadOnlySummaryFormatter().scan(path)
        assert len(raw) == 1
        assert "source_path" not in raw[0]
        assert "source_format" not in raw[0]
        assert "source_version" not in raw[0]

        out = dc.scan(path)
        assert len(out) == 1
        assert isinstance(out[0], dc.PatchSummary)
        assert str(out[0].source_path) == str(path)
        assert out[0].source_format == _ReadOnlySummaryFormatter.name
        assert out[0].source_version == _ReadOnlySummaryFormatter.version
        assert "path" not in out[0].attrs.model_dump()
        assert "file_format" not in out[0].attrs.model_dump()
        assert "file_version" not in out[0].attrs.model_dump()

    def test_scan_payloads_adds_provenance_and_forwards_snap(
        self, monkeypatch, tmp_path
    ):
        """Raw public scans should attach provenance and pass snap to FiberIO."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fiber_io = FiberIO.manager.get_fiberio(
            format=_ReadOnlySummaryFormatter.name,
            version=_ReadOnlySummaryFormatter.version,
        )
        original_scan = fiber_io.scan
        seen = {}

        def _scan(resource, snap=True, **kwargs):
            seen["snap"] = snap
            return original_scan(resource, snap=snap, **kwargs)

        monkeypatch.setattr(fiber_io, "scan", _scan)

        out = dc.scan_payloads(path, snap=False)

        assert seen["snap"] is False
        assert len(out) == 1
        assert isinstance(out[0]["coords"], dc.CoordManager)
        assert str(out[0]["source_path"]) == str(path)
        assert out[0]["source_format"] == fiber_io.name
        assert out[0]["source_version"] == fiber_io.version

    def test_default_fiberio_scan_multi_patch_does_not_set_source_patch_key(
        self, tmp_path, monkeypatch
    ):
        """Default scan should not invent source ids for multi-patch readers."""
        path = tmp_path / "fallback_scan.h5"
        path.write_text("placeholder")
        fio = _ReadOnlySummaryFormatter()

        def read_two_patches(resource: Path, **kwargs) -> dc.BaseSpool:
            patches = [
                dc.get_example_patch().update_attrs(tag="first"),
                dc.get_example_patch().update_attrs(tag="second"),
            ]
            return dc.spool(patches)

        monkeypatch.setattr(fio, "read", read_two_patches)
        out = fio.scan(path)

        assert len(out) == 2
        assert not any(summary["source_patch_key"] for summary in out)

    @pytest.mark.concurrency
    def test_keyboard_interrupt(self, monkeypatch):
        """Ensure a keyboard interrupt works when progress bar is going"""

        class Progress(prog.Progress):
            """A dummy class for progress that just raises interrupt."""

            def track(self, *args, **kwargs):
                """Track progress."""
                raise KeyboardInterrupt("test interrupt")

        # Switch off debug to force progress bar, then make contents to scan.
        contents = list(dc.examples.get_example_spool(length=22))

        with config_context(debug=False):
            with pytest.raises(KeyboardInterrupt, match="test interrupt"):
                dc.scan(contents, progress=Progress())


class TestScanToDF:
    """Tests for scanning to dataframes."""

    def test_input_dataframe(self, random_spool):
        """Ensure a dataframe returns a dataframe."""
        df = random_spool.get_contents()
        out = dc.scan_to_df(df)
        assert out is df

    def test_spool_dataframe(self, random_directory_spool):
        """Ensure scan_to_df just gets the dataframe from the spool."""
        expected = random_directory_spool.get_contents()
        out = dc.scan_to_df(random_directory_spool)
        assert out.equals(expected)


class TestCastType:
    """Test suite to ensure types are intelligently cast to type hints."""

    def test_read(self, dummy_text_file):
        """Ensure write casts type."""
        io = _FiberCaster()
        # this passes if it doesnt raise.
        io.read(dummy_text_file)

    def test_write(self, tmp_path, random_spool):
        """Ensure write casts type."""
        path = tmp_path / "write_fiber_cast.txt"
        io = _FiberCaster()
        # this passes if it doesnt raise.
        io.write(random_spool, path)

    def test_non_standard_name(self, dummy_text_file):
        """Ensure non-standard names still work."""
        io = _FiberCaster()
        io.scan(dummy_text_file)

    def test_unsupported_typehints(self, dummy_text_file):
        """Ensure FiberIO with non-"special" type hints still works."""
        fiberio = _FiberUnsupportedTypeHints()
        out = fiberio.read(dummy_text_file)
        assert out == Path(dummy_text_file).read_text()

    def test_unsupported_type(self, dummy_text_file):
        """Ensure FiberIO from above works with dascore.read."""
        name = _FiberUnsupportedTypeHints.name
        version = _FiberUnsupportedTypeHints.version
        out = dc.read(dummy_text_file, name, version)
        assert out == Path(dummy_text_file).read_text()

    def test_handle_closed_when_method_raises(self, dummy_text_file, monkeypatch):
        """A handle opened by the caster must close when the method raises."""

        class _Recorder:
            closed = False

            def close(self):
                self.closed = True

        recorder = _Recorder()
        monkeypatch.setattr(
            "dascore.io.core.get_handle_from_resource",
            lambda resource, required_type: recorder,
        )

        class _Exploder(FiberIO):
            name = "_ExploderIO"
            version = "1"

            def read(self, resource: BinaryReader, **kwargs):
                raise ValueError("mid-read failure")

        with pytest.raises(ValueError, match="mid-read failure"):
            _Exploder().read(dummy_text_file)
        assert recorder.closed

    def test_handle_aborted_when_write_raises(self, dummy_text_file, monkeypatch):
        """A failed write must discard its handle, not commit a partial file."""

        class _Recorder:
            aborted = False
            closed = False

            def abort(self):
                self.aborted = True

            def close(self):
                self.closed = True

        recorder = _Recorder()
        monkeypatch.setattr(
            "dascore.io.core.get_handle_from_resource",
            lambda resource, required_type: recorder,
        )

        class _BadWriter(FiberIO):
            name = "_BadWriterIO"
            version = "1"

            def write(self, patch, resource: BinaryWriter, **kwargs):
                raise ValueError("mid-write failure")

        with pytest.raises(ValueError, match="mid-write failure"):
            _BadWriter().write(None, dummy_text_file)
        assert recorder.aborted
        assert not recorder.closed


class TestGetSupportedIOTable:
    """A test for creating the supported io table."""

    def test_get_supported_io_table(self):
        """Test the get_supported_io_table function."""
        # call the function to get the result
        result_df = FiberIO.get_supported_io_table()

        # assert that the result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)

        # assert that the length of the DataFrame is not 0
        assert len(result_df) > 0


class TestMissingInstallName:
    """Tests for guessing the package to install from a dependency error."""

    def test_name_sources(self):
        """The name comes from the attr, the module, or the legacy message."""
        errors = [
            MissingOptionalDependencyError("blah", install_name="protobuf"),
            MissingOptionalDependencyError("blah", name="google.protobuf"),
            MissingOptionalDependencyError("protobuf is not installed but..."),
        ]
        assert [_get_missing_install_name(x) for x in errors] == ["protobuf"] * 3

    def test_unidentifiable_package(self):
        """Arbitrary messages should not be mistaken for package names."""
        error = MissingOptionalDependencyError("Optional dependency foo is missing")
        assert _get_missing_install_name(error) == ""
        assert _get_missing_install_name(MissingOptionalDependencyError()) == ""
        # Subclasses which skip the init still have an install name.
        assert MissingOptionalDependencyError.install_name is None

    def test_message_omits_unknown_packages(self):
        """Only identified packages belong in the install command."""
        with pytest.raises(MissingOptionalDependencyError) as exc_info:
            _handle_missing_optionals(0, {"": 2, "segyio": 1, "protobuf": 3})
        msg = str(exc_info.value)
        assert "unknown (2 files)" in msg
        assert "pip install protobuf segyio" in msg
        # Nothing should be recommended when no package was identified.
        with pytest.raises(MissingOptionalDependencyError) as exc_info:
            _handle_missing_optionals(0, {"": 2})
        assert "pip install" not in str(exc_info.value)


class TestIOCoreCoverageEdges:
    """Remaining io.core resolution/robustness branches."""

    def test_directory_format_ignores_unknown_format(self, monkeypatch, tmp_path):
        """An unsupported directory is not itself a scan unit."""

        def _raise_unknown_format(_path):
            raise UnknownFiberFormatError

        monkeypatch.setattr("dascore.io.core.get_format", _raise_unknown_format)
        assert not is_directory_format(tmp_path)

    def test_directory_format_propagates_unexpected_error(self, monkeypatch, tmp_path):
        """Unexpected format-detection failures remain visible to callers."""

        def _raise_unexpected_error(_path):
            raise RuntimeError("format detection failed")

        monkeypatch.setattr("dascore.io.core.get_format", _raise_unexpected_error)
        with pytest.raises(RuntimeError, match="format detection failed"):
            is_directory_format(tmp_path)

    def test_numeric_singleton_without_identity_not_trusted(self):
        """A positional ID cannot resolve an anonymous trimmed singleton."""
        spool = dc.spool([dc.get_example_patch()])
        with pytest.raises(PatchAttributeError, match="uniquely resolved"):
            _resolve_read_spool(spool, source_patch_key="1")

    def test_non_unique_patch_resolution_raises(self):
        """An unresolvable source id in a multi-patch read raises clearly."""
        spool = dc.spool([dc.get_example_patch(tag="a"), dc.get_example_patch(tag="b")])
        with pytest.raises(PatchAttributeError, match="uniquely resolved"):
            _select_patch_from_spool(spool, source_patch_key="neither-id-nor-index")

    @pytest.mark.parametrize("source_patch_key", ["", "node-1"])
    def test_empty_read_raises_missing_patch(self, source_patch_key):
        """A read trimmed to nothing raises MissingPatchError, not IndexError.

        MissingPatchError subclasses IndexError so spool iteration can skip
        these (see #583); the guard must fire before any identity matching,
        with or without a requested source id.
        """
        with pytest.raises(MissingPatchError, match="No patch remained"):
            _select_patch_from_spool(dc.spool([]), source_patch_key=source_patch_key)

    def test_single_patch_resolved_by_name(self):
        """A one-patch read resolves when the id matches the patch name."""
        patch = dc.get_example_patch()
        spool = dc.spool([patch])
        resolved = _select_patch_from_spool(
            spool, source_patch_key=str(patch.get_patch_name())
        )
        assert resolved == patch

    def test_corrupt_file_format_detection_is_robust(self, tmp_path):
        """A reader raising during format detection is caught, not propagated."""
        # valid HDF5 magic followed by garbage: an HDF5 reader raises while
        # probing, which format detection must swallow before giving up.
        bad = tmp_path / "bad.h5"
        bad.write_bytes(b"\x89HDF\r\n\x1a\n" + b"\x00" * 256)
        with pytest.raises(UnknownFiberFormatError):
            dc.get_format(bad)


class TestConvertAttrUnits:
    """Readers spend a file's unit declaration at the parse boundary."""

    def test_converts_and_drops_units(self):
        """The companion attr is consumed, the value converted."""
        attrs = {"gauge_length": 100.0, "gauge_length_units": "cm"}
        out = convert_attr_units(attrs, "gauge_length", "m")
        assert out["gauge_length"] == 1.0
        assert "gauge_length_units" not in out

    def test_documented_units(self):
        """A format whose units live in the key name states them itself."""
        attrs = {"pulse_width": 10.0}
        out = convert_attr_units(attrs, "pulse_width", "s", from_units="ns")
        assert out["pulse_width"] == pytest.approx(1e-8)

    def test_declared_units_win(self):
        """A file which states its units is believed over the default."""
        attrs = {"pulse_width": 10.0, "pulse_width_units": "us"}
        out = convert_attr_units(attrs, "pulse_width", "s", from_units="ns")
        assert out["pulse_width"] == pytest.approx(1e-5)

    def test_missing_units_keeps_value(self):
        """With no declaration the format's documented default stands."""
        assert convert_attr_units({"gauge_length": 5.0}, "gauge_length", "m") == {
            "gauge_length": 5.0
        }

    def test_missing_value_is_noop(self):
        """A file which omits the measure gets no measure."""
        assert (
            convert_attr_units({"gauge_length_units": "m"}, "gauge_length", "m") == {}
        )

    @pytest.mark.parametrize(
        "attrs",
        [
            {"gauge_length": 5.0, "gauge_length_units": "not-a-unit"},
            {"gauge_length": 5.0, "gauge_length_units": "s"},
            {"gauge_length": "ten", "gauge_length_units": "m"},
        ],
    )
    def test_unusable_conversion_drops_value(self, attrs):
        """
        A value whose stated units cannot be used has an unknown scale.

        Keeping the number would pass it off as canonical: a gauge length
        of "5 s" would be read downstream as 5 meters. Vendor headers do
        carry junk, so this warns rather than refusing the file.
        """
        with pytest.warns(UserWarning, match="Dropping gauge_length"):
            out = convert_attr_units(attrs, "gauge_length", "m")
        assert "gauge_length" not in out


class TestSummaryRoundTrip:
    """What a summary keeps when it is rebuilt from itself."""

    @pytest.fixture
    def remote_summary(self, random_patch):
        """A summary naming a source on a filesystem which is not local."""
        return random_patch.summary.new(
            source_path=UPath("memory://archive/one.h5"),
            source_format="DASDAE",
            source_version="1",
        )

    def test_a_remote_path_survives(self, remote_summary):
        """
        A remote path does not survive `model_dump` as a path.

        It comes back as its parts, and a summary which cannot read those
        back drops the path -- and then the format and version with it,
        because a summary with nothing to reload from states no reload
        metadata. Every `new` on a remote-backed summary went that way.
        """
        rebuilt = remote_summary.new(attrs=remote_summary.attrs.update(tag="x"))
        assert str(rebuilt.source_path) == "memory://archive/one.h5"
        assert rebuilt.source_format == "DASDAE"
        assert rebuilt.source_version == "1"
        assert rebuilt.attrs.tag == "x"

    @pytest.mark.parametrize("path", ["one.h5", "/tmp/one.h5"])
    def test_a_local_path_survives_too(self, random_patch, path):
        """Local paths dump as themselves; this is the control."""
        summary = random_patch.summary.new(source_path=path, source_format="DASDAE")
        rebuilt = summary.new(dtype="float32")
        assert str(rebuilt.source_path) == str(summary.source_path)
        assert rebuilt.source_format == "DASDAE"

    def test_a_mapping_naming_no_filesystem(self, random_patch):
        """The parts of a local path, which is the form with no protocol."""
        dumped = {"path": "/tmp/one.h5", "protocol": "", "storage_options": {}}
        summary = random_patch.summary.new(
            source_path=dumped, source_format="DASDAE", source_version="1"
        )
        # Compared as paths, not as text: windows spells this one with
        # backslashes, and what matters is that it is the same path.
        assert summary.source_path == UPath("/tmp/one.h5")
        assert summary.source_format == "DASDAE"

    def test_a_mapping_naming_no_such_filesystem(self, random_patch):
        """A protocol nothing implements is not a path either."""
        dumped = {"path": "/one.h5", "protocol": "nosuchfs", "storage_options": {}}
        summary = random_patch.summary.new(source_path=dumped, source_format="DASDAE")
        assert str(summary.source_path) == ""
        assert summary.source_format == ""

    def test_a_mapping_which_is_not_a_path(self, random_patch):
        """Something else shaped like one is not one, and is dropped."""
        summary = random_patch.summary.new(
            source_path={"not": "a path"}, source_format="DASDAE"
        )
        assert str(summary.source_path) == ""
        assert summary.source_format == ""


class TestSourceIds:
    """The id a patch gets from the file it was read out of."""

    @pytest.fixture
    def dasdae_path(self, random_patch, tmp_path):
        """A written file, and the patch which was written to it."""
        path = tmp_path / "one.h5"
        random_patch.io.write(path, "dasdae")
        return path

    @pytest.fixture
    def terra15_path(self):
        """A file written by something other than DASCore."""
        return fetch("terra15_das_1_trimmed.hdf5")

    def test_reading_twice_is_the_same_data(self, terra15_path):
        """Which is the whole point of deriving the id."""
        first = dc.read(terra15_path)[0]
        second = dc.read(terra15_path)[0]
        assert first.attrs.patch_id
        assert first.attrs.patch_id == second.attrs.patch_id

    def test_the_id_names_the_source(self, terra15_path):
        """Every field the index keeps, and nothing else."""
        patch = dc.read(terra15_path)[0]
        stat = Path(terra15_path).stat()
        fmt, version = dc.get_format(terra15_path)
        expected = source_patch_id(
            fmt,
            version,
            str(terra15_path),
            patch.attrs.get("_source_patch_key", "") or 0,
            stat.st_size,
            stat.st_mtime_ns,
        )
        assert patch.attrs.patch_id == expected

    def test_a_rewritten_file_is_new_data(self, terra15_path, tmp_path):
        """Data written over a path does not inherit the id it replaced."""
        path = tmp_path / "rewritten.hdf5"
        shutil.copy(terra15_path, path)
        before = dc.read(path)[0].attrs.patch_id
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        assert dc.read(path)[0].attrs.patch_id != before

    def test_a_stored_id_beats_a_derived_one(self, dasdae_path, tmp_path):
        """A DASDAE file carries its ids, so they survive being moved."""
        patch = dc.read(dasdae_path)[0]
        moved = tmp_path / "moved.h5"
        patch.io.write(moved, "dasdae")
        assert dc.read(moved)[0].attrs.patch_id == patch.attrs.patch_id

    def test_the_marker_does_not_survive(self, dasdae_path):
        """The stored id is consumed, not left lying on the attrs."""
        patch = dc.read(dasdae_path)[0]
        assert STORED_PATCH_ID not in dict(patch.attrs)

    def test_an_open_file_is_the_file_it_was_opened_on(self, terra15_path):
        """Reading by handle is reading the same data as reading by name."""
        name, version = dc.get_format(terra15_path)
        by_name = dc.read(terra15_path)[0]
        with Path(terra15_path).open("rb") as fid:
            by_handle = dc.read(fid, name, version)[0]
        assert by_handle.attrs.patch_id == by_name.attrs.patch_id

    def test_a_manager_names_what_it_was_built_around(self, terra15_path):
        """A manager is a way of holding a source, not a source of its own."""
        by_name = dc.read(terra15_path)[0]
        with IOResourceManager(terra15_path) as man:
            assert dc.read(man)[0].attrs.patch_id == by_name.attrs.patch_id

    def test_a_source_with_no_path_keeps_its_own_id(self, terra15_path):
        """Two streams must not derive one id out of having no path."""
        name, version = dc.get_format(terra15_path)
        data = Path(terra15_path).read_bytes()
        streams = (io.BytesIO(data), io.BytesIO(data))
        ids = {dc.read(x, name, version)[0].attrs.patch_id for x in streams}
        assert all(ids) and len(ids) == 2

    def test_a_key_naming_several_patches_names_none(self, idless_multi_patch):
        """Or every patch asked for at once would answer to one id."""
        spool = dc.read(idless_multi_patch)
        keys = [x.attrs.get("_source_patch_key", "") for x in spool]
        ids = {
            x.attrs.patch_id for x in dc.read(idless_multi_patch, source_patch_key=keys)
        }
        assert len(ids) == len(keys)

    def test_the_readers_spelling_of_its_format(self, terra15_path):
        """Two spellings resolve to one reader, so they name one datum."""
        name, version = dc.get_format(terra15_path)
        spelled = dc.read(terra15_path, name.lower(), version)[0]
        assert spelled.attrs.patch_id == dc.read(terra15_path)[0].attrs.patch_id

    def test_a_hidden_member_is_not_part_of_a_directory(self, tmp_path):
        """Including one under a hidden directory, which is hidden too."""
        (tmp_path / "member.h5").write_bytes(b"data")
        before = _source_stats(tmp_path)
        (tmp_path / ".cache").mkdir()
        (tmp_path / ".cache" / "member.h5").write_bytes(b"much more data")
        assert _source_stats(tmp_path) == before

    def test_one_file_spelled_two_ways(self, terra15_path, monkeypatch):
        """
        A relative and an absolute spelling name one datum.

        The relative one is made by moving to the file's own directory:
        `relpath` refuses to answer across windows drives, and where the
        test data is cached is not this test's business.
        """
        absolute = dc.read(terra15_path)[0].attrs.patch_id
        monkeypatch.chdir(Path(terra15_path).parent)
        assert dc.read(Path(terra15_path).name)[0].attrs.patch_id == absolute

    def test_a_path_which_cannot_be_canonicalized(self, monkeypatch):
        """
        A spelling nothing can resolve is still the spelling given.

        Forced rather than found: which strings a filesystem refuses is
        the filesystem's business, and differs by platform and python.
        """

        def _refuse(_):
            raise OSError("no")

        monkeypatch.setattr(io_core, "coerce_to_local_path", _refuse)
        assert _canonical_path("one.h5") == "one.h5"

    def test_scanning_with_the_ids_disabled(self, terra15_path):
        """The config which turns the ids off turns scanning off too."""
        with config_context(patch_provenance="disabled"):
            assert dc.scan(terra15_path)[0].attrs.patch_id == ""

    def test_a_source_which_will_not_answer(self):
        """Nothing said is better than fields which pretend to be equal."""
        assert _source_stats(Path("no/such/file.h5")) == (None, None)

    def test_a_stat_which_counts_in_seconds(self):
        """Not every filesystem answers in nanoseconds."""

        class _Stat:
            st_size = 12
            st_mtime = 1.5

        assert _size_and_mtime(_Stat()) == (12, 1_500_000_000)

    def test_a_directory_format_covers_its_members(self, tmp_path):
        """A member rewritten in place is not the data which was there."""
        directory = fetch("dispersion_event.h5").parent
        stats = _source_stats(directory)
        assert all(x is not None for x in stats)
        # The directory's own stat says nothing about a rewritten member.
        assert stats != _size_and_mtime(Path(directory).stat())

    @pytest.fixture
    def idless_multi_patch(self, tmp_path):
        """
        A multi-patch file whose patches carry no ids of their own.

        Built with the ids turned off, which is what a file written
        before they existed holds, so the reader has to derive them.
        """
        path = tmp_path / "multi.h5"
        with config_context(patch_provenance="disabled"):
            first = dc.get_example_patch().update_attrs(tag="first")
            second = dc.get_example_patch().update_attrs(tag="second")
            assert not first.attrs.patch_id
            dc.write(dc.spool([first, second]), path, "dasdae")
        return path

    def test_each_patch_of_a_file_is_its_own_data(self, idless_multi_patch):
        """Or every patch of a file would answer to one id."""
        ids = {patch.attrs.patch_id for patch in dc.read(idless_multi_patch)}
        assert len(ids) == 2

    def test_one_patch_read_by_key(self, idless_multi_patch):
        """Asking for one patch gives the id reading them all would."""
        spool = dc.read(idless_multi_patch)
        wanted = spool[1]
        key = wanted.attrs.get("_source_patch_key", "")
        alone = dc.read(idless_multi_patch, source_patch_key=key)[0]
        assert alone.attrs.patch_id == wanted.attrs.patch_id

    def test_disabled_mints_nothing(self, terra15_path):
        """The config which turns the ids off turns this off too."""
        with config_context(patch_provenance="disabled"):
            assert dc.read(terra15_path)[0].attrs.patch_id == ""
