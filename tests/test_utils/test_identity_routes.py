"""The same samples reached different ways carry the same data_id."""

from __future__ import annotations

import shutil

import numpy as np
import pytest

import dascore as dc
from dascore.io.core import scan_payloads, selected_read_attrs
from dascore.utils.downloader import fetch
from dascore.warnings import DASCoreWarning


def _third(patch, dim="time"):
    """Return a range covering the first third of a coordinate."""
    coord = patch.get_coord(dim)
    start = coord.min()
    return start, start + (coord.max() - start) / 3


@pytest.fixture(scope="module")
def terra15_dir(tmp_path_factory):
    """A directory holding one single-patch, non-DASDAE file."""
    path = tmp_path_factory.mktemp("terra15_routes")
    shutil.copy(fetch("terra15_das_1_trimmed.hdf5"), path / "terra15.hdf5")
    return path


@pytest.fixture(scope="module")
def terra15_path(terra15_dir):
    """The file inside the directory spool."""
    return terra15_dir / "terra15.hdf5"


@pytest.fixture(scope="module")
def terra15_patch(terra15_path):
    """The whole file, read directly."""
    return dc.read(terra15_path)[0]


@pytest.fixture(scope="module")
def dasdae_dir(tmp_path_factory):
    """A directory holding a DASDAE file written from a processed patch."""
    path = tmp_path_factory.mktemp("dasdae_routes")
    patch = dc.get_example_patch().pass_filter(time=(..., 10))
    dc.write(patch, path / "processed.h5", "dasdae")
    return path


@pytest.fixture(scope="module")
def dasdae_path(dasdae_dir):
    """The DASDAE file inside the directory spool."""
    return dasdae_dir / "processed.h5"


class TestReadRouteParity:
    """A read bound, a select, and a spool selection name one array."""

    def test_value_routes_agree(self, terra15_dir, terra15_path, terra15_patch):
        """Every route to the same samples gives one data_id."""
        window = _third(terra15_patch)
        ids = {
            "read_bound": dc.read(terra15_path, time=window)[0],
            "select": terra15_patch.select(time=window),
            "directory_spool": dc.spool(terra15_dir).update().select(time=window)[0],
            "memory_spool": dc.spool([terra15_patch]).select(time=window)[0],
        }
        shapes = {name: patch.shape for name, patch in ids.items()}
        assert len(set(shapes.values())) == 1, shapes
        found = {name: patch.attrs.data_id for name, patch in ids.items()}
        assert len(set(found.values())) == 1, found

    def test_sample_routes_agree(self, terra15_dir, terra15_path, terra15_patch):
        """The same samples selected by index give one data_id."""
        window = (10, 100)
        kwargs = {"time": window, "samples": True}
        ids = {
            "read_bound": dc.read(terra15_path, **kwargs)[0],
            "select": terra15_patch.select(**kwargs),
            "directory_spool": dc.spool(terra15_dir).update().select(**kwargs)[0],
            "memory_spool": dc.spool([terra15_patch]).select(**kwargs)[0],
        }
        shapes = {name: patch.shape for name, patch in ids.items()}
        assert len(set(shapes.values())) == 1, shapes
        found = {name: patch.attrs.data_id for name, patch in ids.items()}
        assert len(set(found.values())) == 1, found

    def test_value_and_sample_routes_agree(self, terra15_path, terra15_patch):
        """Naming the same samples by value or by index gives one data_id."""
        coord = terra15_patch.get_coord("time")
        window = (coord[10], coord[99])
        by_value = dc.read(terra15_path, time=window)[0]
        by_sample = dc.read(terra15_path, time=(10, 100), samples=True)[0]
        assert by_value.shape == by_sample.shape
        assert by_value.attrs.data_id == by_sample.attrs.data_id

    def test_dasdae_routes_agree(self, dasdae_dir, dasdae_path):
        """A stored derived id windows the same way through every route."""
        patch = dc.read(dasdae_path)[0]
        window = _third(patch)
        ids = {
            "read_bound": dc.read(dasdae_path, time=window)[0],
            "select": patch.select(time=window),
            "directory_spool": dc.spool(dasdae_dir).update().select(time=window)[0],
            "memory_spool": dc.spool([patch]).select(time=window)[0],
        }
        found = {name: value.attrs.data_id for name, value in ids.items()}
        assert len(set(found.values())) == 1, found

    def test_trimmed_differs_from_whole(self, terra15_path, terra15_patch):
        """A trimmed read is not the whole file, and still came from it."""
        window = _third(terra15_patch)
        trimmed = dc.read(terra15_path, time=window)[0]
        assert trimmed.attrs.data_id != terra15_patch.attrs.data_id
        assert trimmed.attrs.origin_id == terra15_patch.attrs.origin_id

    def test_whole_read_is_its_source(self, terra15_patch):
        """A patch nothing has been done to is its source."""
        assert terra15_patch.attrs.data_id == terra15_patch._source.data_id

    def test_stored_patch_is_its_source(self, dasdae_path):
        """A stored derived id is the identity its source windows."""
        patch = dc.read(dasdae_path)[0]
        assert patch.attrs.data_id == patch._source.data_id


class TestChunkRouteParity:
    """Chunking a file-backed spool names the same arrays either way."""

    def test_chunk_routes_agree(self, terra15_dir, terra15_patch):
        """The directory and memory spools chunk to the same data_ids."""
        directory = dc.spool(terra15_dir).update().chunk(time=0.05)
        memory = dc.spool([terra15_patch]).chunk(time=0.05)
        first = [patch.attrs.data_id for patch in directory]
        second = [patch.attrs.data_id for patch in memory]
        assert len(first) == len(second) > 1
        assert len(set(first)) == len(first), first
        assert first == second


class TestWindowComposition:
    """Windows compose, so how a selection was reached does not matter."""

    def test_two_selects_equal_one(self, terra15_patch):
        """Selecting twice lands on the same id as selecting once."""
        kwargs = {"samples": True}
        twice = terra15_patch.select(time=(0, 100), **kwargs).select(
            time=(10, 20), **kwargs
        )
        once = terra15_patch.select(time=(10, 20), **kwargs)
        assert twice.shape == once.shape
        assert twice.attrs.data_id == once.attrs.data_id

    def test_full_select_is_the_patch(self, terra15_patch):
        """A selection which removes nothing returns the patch itself."""
        coord = terra15_patch.get_coord("time")
        out = terra15_patch.select(time=(coord.min(), coord.max()))
        assert out is terra15_patch

    def test_prior_operation_breaks_the_window(self, terra15_patch):
        """A patch which is no longer its source derives as usual."""
        window = (10, 100)
        first = terra15_patch.set_units("m").select(time=window, samples=True)
        second = terra15_patch.set_units("s").select(time=window, samples=True)
        plain = terra15_patch.select(time=window, samples=True)
        assert first.attrs.data_id != second.attrs.data_id
        assert first.attrs.data_id != plain.attrs.data_id

    def test_isel_names_the_same_window(self, terra15_patch):
        """Isel narrows the source too, so it names what select names."""
        by_isel = terra15_patch.isel(time=slice(10, 20))
        by_select = terra15_patch.select(time=(10, 20), samples=True)
        assert by_isel.shape == by_select.shape
        assert by_isel.attrs.data_id == by_select.attrs.data_id

    def test_stepped_selection_derives(self, terra15_patch):
        """A selection the source cannot load derives, deterministically."""
        distance = terra15_patch.get_array("distance")
        first = terra15_patch.select(distance=distance[::2])
        second = terra15_patch.select(distance=distance[::2])
        assert first.attrs.data_id == second.attrs.data_id
        contiguous = terra15_patch.select(distance=(0, 10), samples=True)
        assert first.attrs.data_id != contiguous.attrs.data_id

    def test_fancy_read_bound_is_not_the_whole_file(self, terra15_path):
        """A read whose selection is not contiguous never states the file's id."""
        patch = dc.read(terra15_path)[0]
        distance = patch.get_array("distance")
        out = dc.read(terra15_path, distance=distance[::2])[0]
        assert out.attrs.data_id != patch.attrs.data_id
        assert out.attrs.data_id == patch.select(distance=distance[::2]).attrs.data_id

    def test_dasdae_round_trip_windows_stored_id(self, tmp_path):
        """A round trip, then a select, windows the stored id."""
        patch = dc.get_example_patch().pass_filter(time=(..., 10))
        path = tmp_path / "round_trip.h5"
        dc.write(patch, path, "dasdae")
        back = dc.read(path)[0]
        assert back.attrs.data_id == patch.attrs.data_id
        # The source names the stored id, which the window then builds on.
        assert back._source.origin_id == patch.attrs.data_id
        selected = back.select(time=(0, 100), samples=True)
        axis = back.dims.index("time")
        index = tuple(
            slice(0, 100) if x == axis else slice(None) for x in range(len(back.dims))
        )
        assert selected.attrs.data_id == back._source[index].data_id
        assert selected.attrs.data_id != back.attrs.data_id


class TestRefusedBound:
    """A bound with no faithful spelling still reads; only its id is unknown."""

    def test_random_id_rather_than_the_whole_file(self, terra15_patch):
        """A refused bound never leaves the trimmed patch stating the file's id."""
        attrs = terra15_patch.attrs
        queries = {"time": (0, object())}
        with pytest.warns(DASCoreWarning, match="No id could be derived"):
            first = selected_read_attrs(attrs, None, None, queries)
        with pytest.warns(DASCoreWarning, match="No id could be derived"):
            second = selected_read_attrs(attrs, None, None, queries)
        assert first.data_id not in (attrs.data_id, second.data_id)
        assert first.origin_id == second.origin_id == attrs.origin_id


class TestDecodeOptions:
    """A decode option which changes the coordinates changes the id."""

    def test_snap_false_differs(self, terra15_path):
        """snap=False is an operation on the file, not the file itself."""
        default = dc.read(terra15_path)[0]
        unsnapped = dc.read(terra15_path, snap=False)[0]
        assert unsnapped.coords != default.coords
        assert unsnapped.attrs.data_id != default.attrs.data_id
        assert unsnapped.attrs.origin_id == default.attrs.origin_id

    def test_snap_false_is_deterministic(self, terra15_path):
        """The same decode options name the same array."""
        first = dc.read(terra15_path, snap=False)[0]
        second = dc.read(terra15_path, snap=False)[0]
        assert first.attrs.data_id == second.attrs.data_id

    def test_scan_agrees_with_read(self, terra15_path):
        """Scanning and reading derive ids at one place."""
        scanned = dc.scan(terra15_path)[0]
        assert scanned.attrs.data_id == dc.read(terra15_path)[0].attrs.data_id

    def test_scan_agrees_with_read_unsnapped(self, terra15_path):
        """The agreement holds under non-default decode options."""
        scanned = scan_payloads(terra15_path, snap=False)[0]
        read = dc.read(terra15_path, snap=False)[0]
        assert scanned.attrs.data_id == read.attrs.data_id
        assert scanned.attrs.origin_id == read.attrs.origin_id

    def test_snapped_window_builds_on_the_decoded_id(self, terra15_path):
        """A window of an unsnapped read builds on the unsnapped id."""
        whole = dc.read(terra15_path, snap=False)[0]
        trimmed = dc.read(terra15_path, snap=False, time=(0, 100), samples=True)[0]
        assert trimmed.attrs.data_id == whole._source[0:100].data_id


class TestMemoryPatchesUnchanged:
    """A patch with no loadable source derives as it always did."""

    def test_example_patch_select_derives(self):
        """Nothing about an in-memory selection moves."""
        patch = dc.get_example_patch()
        first = patch.select(time=(0, 100), samples=True)
        second = patch.select(time=(0, 100), samples=True)
        assert first.attrs.data_id == second.attrs.data_id
        assert first.attrs.data_id != patch.attrs.data_id

    def test_no_op_select_keeps_id(self):
        """A selection which changes nothing changes no id."""
        patch = dc.get_example_patch()
        assert patch.select(time=...) is patch

    def test_constant_source_is_not_windowed(self):
        """A patch built in memory names no window of anything."""
        patch = dc.get_example_patch()
        assert patch._source is None
        out = patch.select(distance=(0, 5), samples=True)
        assert out.attrs.data_id != patch.attrs.data_id
        assert np.allclose(out.data, patch.data[0:5])
