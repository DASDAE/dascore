"""The same samples reached different ways carry the same data_id."""

from __future__ import annotations

import shutil
from decimal import Decimal

import numpy as np
import pytest

import dascore as dc
import dascore.utils.patch_assembly as assembly
from dascore.config import config_context
from dascore.core.coordmanager import get_coord_manager
from dascore.io.core import scan_payloads
from dascore.utils.downloader import fetch
from dascore.utils.identity import narrowed_data_id, read_operation_id
from dascore.utils.misc import suppress_warnings
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


@pytest.fixture(scope="module")
def aux_path(tmp_path_factory):
    """A DASDAE file holding a coordinate attached to no dimension."""
    path = tmp_path_factory.mktemp("aux_routes") / "aux.h5"
    patch = dc.get_example_patch().update_coords(aux=((), np.arange(5.0)))
    dc.write(patch, path, "dasdae")
    return path


@pytest.fixture(scope="module")
def aux_patch(aux_path):
    """The whole file, read back."""
    return dc.read(aux_path)[0]


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
        shapes = {name: value.shape for name, value in ids.items()}
        assert len(set(shapes.values())) == 1, shapes
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

    def test_joined_float_grid(self, tmp_path):
        """Files cut from one float grid join to one data_id either way (#1718)."""
        patch = dc.get_example_patch()
        grid = dc.get_coord(start=0.0, step=0.1, shape=(patch.shape[0],))
        whole = patch.update_coords(distance=grid)
        paths = [tmp_path / f"p{num}.h5" for num in range(2)]
        for path, cut in zip(paths, [(0, 150), (150, None)], strict=True):
            whole.select(distance=cut, samples=True).io.write(path, "DASDAE")
        directory = dc.spool(tmp_path).update().chunk(distance=None)
        memory = dc.spool([dc.read(x)[0] for x in paths]).chunk(distance=None)
        (first,), (second,) = directory, memory
        assert first.equals(second)
        assert first.attrs.data_id == second.attrs.data_id
        # and the directory chunk names it without loading (`Spool.__eq__`)
        assert [x.data_id for x in directory._loads()] == [first.attrs.data_id]


class TestChunkLayout:
    """
    A chunk output is named by what was done (#1643): its members, the
    samples each gives, and where they go, not how the plan asked for them.
    """

    @staticmethod
    def _patch(start, size=6):
        """A float time coordinate, which a read widens its bounds on."""
        time = np.arange(float(size)) + start
        data = np.arange(float(size))
        return dc.Patch(data=data, coords={"time": time}, dims=("time",))

    def test_a_window_ignores_the_read_pad(self, monkeypatch):
        """However wide the assembler reads, a window names its samples."""
        spool = dc.spool([self._patch(0)])
        first = spool.chunk(time=3)[0]
        monkeypatch.setattr(assembly, "_READ_PAD", 0.3)
        again = spool.chunk(time=3)[0]
        assert again == first
        assert again.attrs.data_id == first.attrs.data_id

    def test_windows_of_one_member_differ(self):
        """Each window is its own samples, and none is the whole member."""
        patch = self._patch(0)
        ids = [x.attrs.data_id for x in dc.spool([patch]).chunk(time=3)]
        assert len(set(ids)) == len(ids) == 2
        assert patch.attrs.data_id not in ids

    def test_a_setting_which_changes_nothing(self):
        """Two max_shifts which merge the same members alike give one id."""
        members = [self._patch(0), self._patch(6.25)]
        with config_context(max_shift=0.3):
            first = dc.spool(members).chunk(time=None)
        with config_context(max_shift=0.45):
            second = dc.spool(members).chunk(time=None)
        assert len(first) == len(second) == 1
        assert first[0] == second[0]
        assert first[0].attrs.data_id == second[0].attrs.data_id

    def test_a_setting_which_changes_the_members(self):
        """A max_shift too small to join the members gives other outputs."""
        members = [self._patch(0), self._patch(6.25)]
        with config_context(max_shift=0.3):
            merged = dc.spool(members).chunk(time=None)
        apart = dc.spool(members).chunk(time=None)
        assert (len(merged), len(apart)) == (1, 2)
        ids = {x.attrs.data_id for x in [*merged, *apart]}
        assert len(ids) == 3

    def test_the_same_plan_twice(self):
        """Re-chunking names each output as it did before."""
        spool = dc.spool([self._patch(0), self._patch(6)])
        first = [x.attrs.data_id for x in spool.chunk(time=4)]
        second = [x.attrs.data_id for x in spool.chunk(time=4)]
        again = [x.attrs.data_id for x in spool.chunk(time=None).chunk(time=4)]
        assert first == second == again
        assert len(set(first)) == len(first)

    def test_labels_moved_by_a_little(self):
        """A grid which moves the labels at all names another array."""
        members = [self._patch(0), self._patch(6.00001)]
        snapped = dc.spool(members).chunk(time=None)[0]
        spread = dc.spool(members).chunk(time=None, max_shift=1e-8, tolerance=1.5)[0]
        labels = [x.get_coord("time").values for x in (snapped, spread)]
        assert not np.array_equal(*labels)
        assert snapped.attrs.data_id != spread.attrs.data_id

    def test_a_grid_named_by_its_labels(self):
        """A window's grid is named by its labels, not the stretch it is cut from."""
        members = [self._patch(0), self._patch(6.25), self._patch(12.25)]
        two = dc.spool(members[:2]).chunk(time=4, max_shift=0.3)[1]
        three = dc.spool(members).chunk(time=4, max_shift=0.3)[1]
        assert two == three
        assert two.attrs.data_id == three.attrs.data_id

    def test_a_regrid_is_named_by_its_labels(self):
        """Stretches of other lengths which label a slice alike name it alike."""
        labels = dc.get_coord(data=np.arange(4) * 0.1 + 0.4 + 1e-9)
        coords = get_coord_manager({"time": labels}, dims=("time",))
        attrs = dc.PatchAttrs(data_id="a" * 32)
        placed, ids = [], []
        for count in (12, 14):  # a float slice remembers its parent's length
            row = {"_time_stretch": (0.0, 0.1, 0, count, 0, 4, 4, "float64")}
            out_coords, out_attrs = assembly._regridded(coords, attrs, "time", row)
            placed.append(out_coords.coord_map["time"].values)
            ids.append(out_attrs.data_id)
        assert np.array_equal(*placed)
        assert ids[0] == ids[1] != attrs.data_id

    def test_a_file_window_is_the_read_window(self, terra15_dir, terra15_path):
        """A chunk of a file names what reading that window names."""
        out = dc.spool(terra15_dir).update().chunk(time=0.05)[1]
        time = out.get_coord("time")
        read = dc.read(terra15_path, time=(time.min(), time.max()))[0]
        assert read == out
        assert out.attrs.data_id == read.attrs.data_id


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
        """A window of a patch which is no longer its source builds on its id."""
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

    @staticmethod
    def _refused_bound(patch, dim="distance"):
        """Positions the reader can load and the encoder cannot spell."""
        # Every other one, so the source cannot load the result either.
        values = patch.get_array(dim)[:6:2]
        return np.array([Decimal(float(x)) for x in values], dtype=object)

    def test_random_id_rather_than_the_whole_file(self, terra15_path, terra15_patch):
        """A refused bound never leaves the trimmed patch stating the file's id."""
        bound = self._refused_bound(terra15_patch)
        with pytest.warns(DASCoreWarning, match="No id could be derived"):
            first = dc.read(terra15_path, distance=bound)[0]
        with suppress_warnings(DASCoreWarning, message="No id", action="error"):
            second = dc.read(terra15_path, distance=bound)[0]
        assert first.shape != terra15_patch.shape
        # Nothing names the array, so the two reads are two arrays.
        assert first.attrs.data_id not in ("", terra15_patch.attrs.data_id)
        assert first.attrs.data_id != second.attrs.data_id
        assert first.attrs.origin_id == terra15_patch.attrs.origin_id


class TestUnattachedCoords:
    """A coordinate riding no dimension is not part of any window."""

    def test_select_derives(self, aux_patch):
        """Trimming it changes the patch, so it changes the id."""
        out = aux_patch.select(aux=(1, 2))
        assert out.get_coord("aux").shape != aux_patch.get_coord("aux").shape
        assert out.attrs.data_id != aux_patch.attrs.data_id
        assert out.attrs.data_id == aux_patch.select(aux=(1, 2)).attrs.data_id
        assert out.attrs.data_id != aux_patch.select(aux=(1, 3)).attrs.data_id

    def test_read_bound_derives(self, aux_path, aux_patch):
        """A read trimmed the same way names what the select names."""
        out = dc.read(aux_path, aux=(1, 2))[0]
        assert out.attrs.data_id != aux_patch.attrs.data_id
        assert out.attrs.data_id == aux_patch.select(aux=(1, 2)).attrs.data_id

    def test_with_a_dimensional_slice(self, aux_patch):
        """A window says nothing about it, so the whole call is derived."""
        window = {"time": (0, 100), "samples": True}
        plain = aux_patch.select(**window)
        first = aux_patch.select(aux=(1, 2), **window)
        second = aux_patch.select(aux=(2, 3), **window)
        assert first.attrs.data_id not in (plain.attrs.data_id, second.attrs.data_id)
        assert (
            first.attrs.data_id == aux_patch.select(aux=(1, 2), **window).attrs.data_id
        )


class TestWindowGuards:
    """A window is named only where the source can say what it holds."""

    @staticmethod
    def _narrowed(patch):
        """The source of the patch's first ten samples along its first dim."""
        index = tuple(
            slice(0, 10) if x == 0 else slice(None) for x in range(patch.ndim)
        )
        return patch._source.narrow(index)

    def test_without_the_coordinates_there_is_no_window(self, terra15_patch):
        """A caller which does not say what moved gets no shortcut."""
        after = self._narrowed(terra15_patch)
        args = (terra15_patch.attrs, terra15_patch._source, after)
        coords = terra15_patch.coords
        assert narrowed_data_id(*args, coords, coords) == after.data_id
        assert narrowed_data_id(*args) is None

    def test_relaid_dims_are_not_a_window(self, terra15_patch):
        """A source window says which samples, not which way round they lie."""
        after = self._narrowed(terra15_patch)
        coords = terra15_patch.coords
        flipped = coords.transpose(*terra15_patch.dims[::-1])
        found = narrowed_data_id(
            terra15_patch.attrs, terra15_patch._source, after, coords, flipped
        )
        assert found is None


class TestDisabledProvenance:
    """A read which trimmed carries no id rather than a stale one."""

    def test_a_trimmed_read_clears_stored_ids(self, dasdae_path):
        """A stored id names the whole patch on disk, never a window of it."""
        with dc.config_context(patch_provenance="disabled"):
            whole = dc.read(dasdae_path)[0]
            trimmed = dc.read(dasdae_path, time=(0, 100), samples=True)[0]
        # The file states them, so a whole read still reports what it read.
        assert whole.attrs.data_id and whole.attrs.origin_id
        assert trimmed.shape != whole.shape
        assert trimmed.attrs.data_id == trimmed.attrs.origin_id == ""

    def test_a_read_with_no_stored_ids(self, terra15_path):
        """Nothing was derived while it was disabled, so nothing is claimed."""
        with dc.config_context(patch_provenance="disabled"):
            out = dc.read(terra15_path, time=(0, 100), samples=True)[0]
        assert out.attrs.data_id == out.attrs.origin_id == ""


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

    def test_the_dims_snapped_are_a_set(self, terra15_path):
        """One decode, however the dimensions it names were spelled."""
        one = read_operation_id("time")
        assert one == read_operation_id(("time",)) == read_operation_id(["time"])
        both = read_operation_id(("time", "distance"))
        assert both == read_operation_id(("distance", "time"))
        assert both != one
        # The default decodes the stored thing itself and derives nothing.
        assert read_operation_id(True) is None
        assert read_operation_id(False) not in (None, one, both)

    def test_snapped_window_builds_on_the_decoded_id(self, terra15_path):
        """A window of an unsnapped read builds on the unsnapped id."""
        whole = dc.read(terra15_path, snap=False)[0]
        trimmed = dc.read(terra15_path, snap=False, time=(0, 100), samples=True)[0]
        assert trimmed.attrs.data_id == whole._source[0:100].data_id


class TestWindowsOfMemoryPatches:
    """
    A window is its parent's data_id and the samples it keeps, however it
    was asked for, so a patch in memory and the file holding it name their
    windows alike.
    """

    start, stop = 100, 200  # the samples along time every route keeps

    @pytest.fixture(scope="class")
    @classmethod
    def patch(cls):
        """A patch built in memory, which has no source to window."""
        out = dc.get_example_patch()
        assert out._source is None
        return out

    @pytest.fixture(scope="class")
    @classmethod
    def patch_dir(cls, patch, tmp_path_factory):
        """A directory holding the patch, written by DASCore."""
        path = tmp_path_factory.mktemp("memory_routes")
        dc.write(patch, path / "patch.h5", "dasdae")
        return path

    @pytest.fixture(scope="class")
    @classmethod
    def bounds(cls, patch):
        """The first and last time kept."""
        time = patch.get_coord("time")
        return time[cls.start], time[cls.stop - 1]

    @pytest.fixture(scope="class")
    @classmethod
    def routes(cls, patch, patch_dir, bounds):
        """The same samples of the patch, taken every way there is."""
        time, size = patch.get_coord("time"), cls.stop - cls.start
        offsets = tuple(x - time.min() for x in bounds)
        samples = {"time": (cls.start, cls.stop), "samples": True}
        return {
            "select": patch.select(time=bounds),
            "samples": patch.select(**samples),
            "relative": patch.select(time=offsets, relative=True),
            "isel": patch.isel(time=slice(cls.start, cls.stop)),
            "sel": patch.sel(time=slice(*bounds)),
            "chunk_member": dc.spool([patch]).chunk(time=time.step * size)[1],
            "memory_spool": dc.spool([patch]).select(time=bounds)[0],
            "stored_window": dc.read(patch_dir / "patch.h5", time=bounds)[0],
            "file_spool": dc.spool(patch_dir).update().select(time=bounds)[0],
        }

    def test_every_route_names_one_window(self, routes):
        """Each route keeps the same samples, so each gives one data_id."""
        first = routes["select"]
        assert all(x == first for x in routes.values())
        found = {name: x.attrs.data_id for name, x in routes.items()}
        assert len(set(found.values())) == 1, found

    def test_bounds_between_samples(self, patch, bounds, routes):
        """Bounds anywhere short of the next sample keep the same samples."""
        nudge = patch.get_coord("time").step / 3
        out = patch.select(time=(bounds[0] - nudge, bounds[1] + nudge))
        assert out == routes["select"]
        assert out.attrs.data_id == routes["select"].attrs.data_id

    def test_other_samples_differ(self, patch, routes):
        """A window keeping other samples, or a stepped one, is another array."""
        start, stop = self.start, self.stop
        others = [
            patch.select(time=(start + 1, stop), samples=True),
            patch.select(time=(start, stop + 1), samples=True),
            patch.select(time=(start, stop), distance=(0, 10), samples=True),
            patch.isel(time=slice(start, stop, 2)),
        ]
        found = {x.attrs.data_id for x in others}
        assert len(found) == len(others)
        assert not found & {routes["select"].attrs.data_id, patch.attrs.data_id}

    def test_spools_of_the_file_and_patch(self, patch, patch_dir, bounds):
        """A selection of the file equals the same selection of the patch."""
        files = dc.spool(patch_dir).update().select(time=bounds)
        assert files == dc.spool([patch]).select(time=bounds)
        # by what each loads: other data under the same metadata differs
        other = patch.new(data=patch.data * 2)
        assert files != dc.spool([other]).select(time=bounds)

    def test_every_sample_kept_is_the_patch(self, patch):
        """Keeping every sample names the patch; select hands it back."""
        time = patch.get_coord("time")
        assert patch.select(time=(time.min(), time.max())) is patch
        whole = patch.isel(time=slice(None))
        assert whole.attrs.data_id == patch.attrs.data_id

    def test_a_window_of_a_processed_patch(self, patch, bounds):
        """It builds on the processed patch's id, and leaves history alone."""
        processed = patch.pass_filter(time=(..., 10))
        out = processed.select(time=bounds)
        by_index = processed.isel(time=slice(self.start, self.stop))
        assert out.attrs.data_id == by_index.attrs.data_id
        assert out.attrs.data_id != patch.select(time=bounds).attrs.data_id
        assert out.attrs.history == by_index.attrs.history
        assert out.attrs.history == processed.attrs.history

    def test_no_op_select_keeps_id(self, patch):
        """A selection which changes nothing changes no id."""
        assert patch.select(time=...) is patch


class TestPinnedSources:
    """A pinned patch is no longer the array its source loads."""

    def test_the_source_is_kept(self, terra15_patch):
        """Pinning states an id; it does not detach the file behind it."""
        pinned = terra15_patch.pin_id()
        assert terra15_patch._source is not None
        assert pinned._source is terra15_patch._source

    def test_a_window_builds_on_the_pinned_id(self, terra15_patch):
        """So a window of it is a window of the pinned id, however taken."""
        window = {"time": (0, 100), "samples": True}
        windowed = terra15_patch.select(**window)
        assert windowed.attrs.data_id == terra15_patch._source[0:100].data_id
        pinned = terra15_patch.pin_id()
        out = pinned.select(**window)
        assert out.attrs.data_id not in (windowed.attrs.data_id, pinned.attrs.data_id)
        assert out.attrs.data_id == pinned.isel(time=slice(0, 100)).attrs.data_id

    def test_dasdae_round_trip(self, dasdae_path, tmp_path):
        """A pinned id is stored and read back like any other."""
        pinned = dc.read(dasdae_path)[0].pin_id()
        path = tmp_path / "pinned.h5"
        dc.write(pinned, path, "dasdae")
        out = dc.read(path)[0]
        assert out.attrs.data_id == pinned.attrs.data_id
        assert out.attrs.origin_id == pinned.attrs.origin_id
