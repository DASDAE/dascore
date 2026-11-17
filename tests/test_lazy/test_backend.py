"""Tests for moving a patch or a spool to another array backend."""

from __future__ import annotations

import os
import sys
import warnings
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.core.processor import register_kernel
from dascore.core.source import ArraySource
from dascore.exceptions import (
    LazyOperationError,
    NumpyFallbackError,
    ParameterError,
)
from dascore.io import core as io_core
from dascore.lazy import execute
from dascore.lazy.execute import chunk_boxes
from dascore.lazy.recipe import Recipe
from dascore.utils import array_api as array_api_module
from dascore.utils import patch_assembly as assembly
from dascore.utils.array_api import (
    lazy_read_backend,
    parse_backend,
    reading_lazily,
    to_backend,
    to_numpy,
)
from dascore.warnings import NumpyFallbackWarning

STRICT = "array_api_strict"


@pytest.fixture(scope="module")
def strict():
    """An array library which does only what the array API says."""
    return pytest.importorskip(STRICT)


def _in_four_files(patch, directory):
    """Write a patch as four files which abut along time; return their spool."""
    size = patch.shape[patch.get_axis("time")]
    edges = [0, size // 4, size // 2, 3 * size // 4, size]
    for index, (start, stop) in enumerate(pairwise(edges)):
        part = patch.select(time=(start, stop), samples=True)
        dc.write(part, directory / f"part_{index}.h5", "dasdae")
    return dc.spool(directory).update()


@pytest.fixture(scope="module")
def whole():
    """The patch the four files of `spool` hold between them."""
    return dc.get_example_patch("random_das", shape=(20, 2000))


@pytest.fixture(scope="module")
def spool(whole, tmp_path_factory):
    """The spool of a directory of four files which abut along time."""
    return _in_four_files(whole, tmp_path_factory.mktemp("lazy_backend"))


@pytest.fixture(scope="module")
def undescribed(patch, tmp_path_factory):
    """A spool of four files whose patches its index cannot stand in for."""
    # The example file's attrs are of an older layout, which is read from
    # each file and not rebuilt from the index's rows.
    return _in_four_files(patch, tmp_path_factory.mktemp("lazy_undescribed"))


@pytest.fixture()
def reads(reads, monkeypatch):
    """Every array read from a file, by whichever route it is read."""
    # The fixture this adds to sees a source loading its own array, which
    # is how a lazy array and a recipe read. A patch read by its format
    # takes the other route, through the loader `FiberIO.read` is handed.
    hook = io_core._paired_hook

    def paired(fiber_io, name):
        found = hook(fiber_io, name)
        if name != "_prepare_read":
            return found

        def prepare(*args, **kwargs):
            patches, load = found(*args, **kwargs)

            def recorded(requests):
                for data in load(requests):
                    reads.append(data)
                    yield data

            return patches, recorded

        return prepare

    monkeypatch.setattr(io_core, "_paired_hook", paired)
    return reads


def _refuse_numpy(monkeypatch):
    """Fail the test if DASCore takes an array of another library to NumPy."""

    def refuse(*args, **kwargs):
        """Raise: nothing should be converted."""
        raise AssertionError("the data were converted to NumPy")

    # Under every name a module of DASCore's has imported it by.
    for name, module in list(sys.modules.items()):
        if name.startswith("dascore") and getattr(module, "to_numpy", None) is to_numpy:
            monkeypatch.setattr(module, "to_numpy", refuse)
    assert array_api_module.to_numpy is refuse
    # And by NumPy taking the strict library's arrays, as `np.asarray` does.
    if (strict := sys.modules.get(STRICT)) is not None:
        monkeypatch.setattr(type(strict.asarray(0)), "__array__", refuse)


class _Doubles(dc.PatchProcessor):
    """An operation written for NumPy alone, which a chunk can be given."""

    name = None

    def chunk_contract(self, dtype, plan):
        """Return that each sample is made from its own."""
        from dascore.lazy.contract import Pointwise  # noqa: PLC0415

        return Pointwise()

    def numpy_kernel(self, data):
        """Return the data doubled."""
        return data * 2


class TestBackendNames:
    """A backend is an array library by name, or a lazy one which names it."""

    @pytest.mark.parametrize(
        "name, expected",
        (
            ("numpy", (False, "numpy")),
            ("JAX", (False, "jax")),
            ("lazy", (True, "numpy")),
            ("LazyArray", (True, "numpy")),
            ("lazyarray[ jax ]", (True, "jax")),
            ("LazyArray[array_api_strict]", (True, STRICT)),
            ("lazy[numpy]", (True, "numpy")),
        ),
    )
    def test_spellings(self, name, expected):
        """The class or its alias, in any case, with the library in brackets."""
        assert parse_backend(name) == expected

    @pytest.mark.parametrize(
        "name", ("", "lazy[", "LazyArray[a][b]", "nu mpy", "a-b", "a.b", "lazy[a.b]")
    )
    def test_not_a_backend(self, name):
        """What names nothing is refused, and says what would."""
        with pytest.raises(ParameterError, match="is not a backend"):
            parse_backend(name)

    def test_to_backend(self, strict):
        """An array is put on the library named, and left where it already is."""
        data = np.arange(6.0).reshape(2, 3)
        there = to_backend(data, STRICT)
        assert type(there) is type(strict.asarray(data))
        assert to_backend(there, STRICT) is there
        assert np.array_equal(to_backend(there, "numpy"), data)


class TestFallbackSetting:
    """One setting decides a conversion through NumPy."""

    def test_in_memory_operation(self, patch, strict):
        """An operation with no kernel for the library obeys the setting."""
        there = patch.set_backend(STRICT)
        expected = patch.detrend("time")
        with pytest.warns(NumpyFallbackWarning, match="detrend has no"):
            out = there.detrend("time")
        assert out.backend == STRICT
        np.testing.assert_allclose(to_numpy(out.data), expected.data)
        with config_context(numpy_fallback="raise"):
            with pytest.raises(NumpyFallbackError, match="refuses to convert"):
                there.detrend("time")
        with config_context(numpy_fallback="ignore"), warnings.catch_warnings():
            warnings.simplefilter("error")
            quiet = there.detrend("time")
        assert quiet.backend == STRICT


class TestPatchSetBackend:
    """A patch is moved to a backend by name, and stays the same patch."""

    def test_lazy_reads_nothing(self, patch, reads):
        """A patch from a file becomes a promise to read it."""
        lazy = patch.set_backend("lazy")
        assert lazy.backend == "lazy" and lazy.shape == patch.shape
        assert lazy.attrs == patch.attrs and lazy.coords == patch.coords
        assert not reads
        loaded = lazy.set_backend("numpy")
        assert len(reads) == 1 and loaded.equals(patch)
        assert loaded.attrs.data_id == patch.attrs.data_id

    def test_nothing_to_do(self, patch):
        """A patch already on the backend is handed back as it is."""
        assert patch.set_backend("numpy") is patch
        lazy = patch.set_backend("LazyArray")
        assert lazy.set_backend("lazy[numpy]") is lazy

    def test_made_in_memory(self):
        """A patch with nothing to read later cannot be made lazy."""
        made = dc.get_example_patch()
        with pytest.raises(ParameterError, match="nothing to read later"):
            made.set_backend("lazy")
        # Already in memory, it is where NumPy was asked to put it.
        assert made.set_backend("numpy") is made

    def test_changed_in_memory(self, patch):
        """Nor can one whose data are no longer what its file holds."""
        with pytest.raises(ParameterError, match="nothing to read later"):
            patch.abs().set_backend("lazy")

    def test_there_and_back(self, patch, reads):
        """Read into memory, a patch still names the array it was read from."""
        loaded = patch.set_backend("lazy").set_backend("numpy")
        again = loaded.set_backend("lazy")
        assert again.backend == "lazy" and again.attrs == patch.attrs
        reads.clear()
        assert again.load().equals(patch) and len(reads) == 1

    def test_relative_path_is_pinned(self, patch, tmp_path, monkeypatch):
        """A lazy patch reads the file it was named beside, wherever it is read."""
        dc.write(patch, tmp_path / "here.h5", "dasdae")
        monkeypatch.chdir(tmp_path)
        made = dc.read("here.h5")[0].set_backend("lazy")
        with reading_lazily("numpy"):
            read = dc.read("here.h5")[0]
        handed = dc.spool("here.h5").set_backend("lazy")[0]
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        for lazy in (made, read, handed):
            assert lazy.backend == "lazy" and lazy.load().equals(patch)

    def test_there_and_back_from_elsewhere(self, patch, tmp_path, monkeypatch):
        """Made lazy from a relative path, it reads the same file after a move."""
        dc.write(patch, tmp_path / "here.h5", "dasdae")
        monkeypatch.chdir(tmp_path)
        with reading_lazily("numpy"):
            read = dc.read("here.h5")[0]
        made = dc.read("here.h5")[0].set_backend("lazy")
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        # A file of the same name, holding other data, where the process is.
        dc.write(patch.abs(), elsewhere / "here.h5", "dasdae")
        monkeypatch.chdir(elsewhere)
        for lazy in (read, made):
            again = lazy.set_backend("numpy").set_backend("lazy")
            assert again.load().equals(patch)

    def test_library_which_narrows_the_data(self, patch, monkeypatch):
        """A library which would hold the values in fewer bits is refused."""
        strict = pytest.importorskip("array_api_strict")

        def narrowing(name):
            """Return a namespace which keeps 64-bit numbers in 32."""
            return SimpleNamespace(
                asarray=lambda x: strict.asarray(np.asarray(x, np.float32))
            )

        monkeypatch.setattr(array_api_module, "import_module", narrowing)
        with pytest.raises(ParameterError, match="holds float64 data as float32"):
            patch.set_backend("narrowing")

    def test_structured_data(self, monkeypatch):
        """A record dtype is checked as it is, and not rebuilt from its name."""
        dtype = np.dtype([("x", "<i4"), ("y", "<f8")])
        source = ArraySource(path="/a/record.h5", format="DASDAE", key="")
        source = source.describe((6, 4), dtype)
        monkeypatch.setattr(
            Recipe, "load", lambda self: np.zeros(self.shape, self.dtype)
        )
        coords = {"x": np.arange(6), "y": np.arange(4)}
        lazy = dc.Patch(data=Recipe.from_source(source), coords=coords, dims=("x", "y"))
        part = lazy.select(x=(1, 4), samples=True)
        assert part.backend == "lazy" and len(part.data.chain)
        assert part.load().dtype == dtype

    def test_structured_data_to_another_library(self):
        """A record dtype survives the check that a library kept the dtype."""
        dask_array = pytest.importorskip("dask.array")
        dtype = np.dtype([("x", "<i4"), ("y", "<f8")])
        coords = {"x": np.arange(6), "y": np.arange(4)}
        patch = dc.Patch(data=np.zeros((6, 4), dtype), coords=coords, dims=("x", "y"))
        there = patch.set_backend("dask")
        assert isinstance(there.data, dask_array.Array)
        assert there.dtype == dtype and to_backend(there.data, "numpy").dtype == dtype

    def test_as_a_numpy_array(self, patch, strict):
        """`np.asarray` of a lazy array on another library is a NumPy array."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]")
        out = np.asarray(lazy.data)
        assert type(out) is np.ndarray and np.array_equal(out, patch.data)

    def test_an_empty_box(self, patch, strict):
        """A box of no samples is on the library the others are."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]").abs()
        empty = execute.compute(lazy.data, (slice(0, 0), slice(0, 5)))
        assert empty.shape == (0, 5)
        assert type(empty) is type(strict.asarray(np.zeros(1)))

    def test_another_library(self, patch, strict):
        """An array library by name holds the data in memory."""
        there = patch.set_backend(STRICT)
        assert there.backend == STRICT and there.attrs == patch.attrs
        assert np.array_equal(to_numpy(there.data), patch.data)
        back = there.set_backend("numpy")
        assert back.backend == "numpy" and back.equals(patch)

    def test_lazy_on_another_library(self, patch, strict, reads):
        """Chunks are computed on the library in the brackets."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]")
        assert lazy.backend == "lazy" and lazy.data.backend == STRICT
        out = lazy.abs()
        assert out.backend == "lazy" and not reads
        loaded = out.load()
        assert loaded.backend == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), np.abs(patch.data))

    def test_asked_for_on_numpy_is_computed_there(self, patch, strict, monkeypatch):
        """Moved back to NumPy, a lazy patch computes there, converting nothing."""
        out = patch.set_backend(f"LazyArray[{STRICT}]").abs()
        _refuse_numpy(monkeypatch)
        loaded = out.set_backend("numpy")
        assert loaded.backend == "numpy"
        np.testing.assert_array_equal(loaded.data, np.abs(patch.data))

    def test_library_which_cannot_be_imported(self, patch):
        """A library which is not installed is refused when it is named."""
        for name in ("not_an_array_library", "LazyArray[not_an_array_library]"):
            with pytest.raises(ParameterError, match="No array library called"):
                patch.set_backend(name)

    def test_registered_kernel_runs_each_chunk(self, patch, strict, monkeypatch):
        """A kernel registered for the library is the one a lazy chunk runs."""
        handed = []

        class Halves(dc.PatchProcessor):
            """An operation written for NumPy alone."""

            name = None

            def chunk_contract(self, dtype, plan):
                """Return that each sample is made from its own."""
                return _Doubles().chunk_contract(dtype, plan)

            def numpy_kernel(self, data):
                """Return the data doubled."""
                return data * 2

        @register_kernel(Halves, STRICT)
        def halves_there(self, data):
            """Return the data doubled, on the library they are on."""
            handed.append(type(data))
            return data * 2

        lazy = patch.set_backend(f"LazyArray[{STRICT}]")
        # Small enough blocks that the patch is computed in several boxes.
        block = patch.data.nbytes // 6
        with config_context(lazy_block_size=block), warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = Halves()(lazy)
            assert len(chunk_boxes(out.data)) > 3
            # Nothing is taken through NumPy on the way: not a chunk for
            # the kernel, and not the boxes to put them together.
            _refuse_numpy(monkeypatch)
            loaded = out.load()
        assert len(handed) > 3 and set(handed) == {type(strict.asarray(np.zeros(1)))}
        assert loaded.backend == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)

    def test_boxes_joined_along_both_axes(self, patch, strict):
        """Boxes cut along more than one axis go back where they came from."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]").abs()
        boxes = [
            (slice(a, b), slice(c, d))
            for a, b in ((0, 200), (200, 601))
            for c, d in ((0, 300), (300, 700), (700, 1001))
        ]
        pieces = [execute.compute(lazy.data, box) for box in boxes]
        joined = execute._joined(pieces, boxes)
        np.testing.assert_array_equal(to_numpy(joined), np.abs(patch.data))

    @pytest.mark.parametrize("policy", (None, "raise", "warn", "ignore"))
    def test_no_kernel_for_the_library(self, patch, strict, policy):
        """Without one, converting each chunk is the setting's to allow."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]")
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            if policy in (None, "raise"):
                with pytest.raises(LazyOperationError, match="refuses to convert"):
                    _Doubles()(lazy)
                return
            if policy == "warn":
                match = f"_Doubles has no {STRICT} implementation; each chunk"
                with pytest.warns(NumpyFallbackWarning, match=match):
                    out = _Doubles()(lazy)
            else:
                out = _Doubles()(lazy)
            # Said once, when it was chained, and not again for each chunk.
            loaded = out.load()
        assert loaded.backend == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)


class TestPendingStepsOnAnotherLibrary:
    """Steps already pending are asked for a kernel on the new library."""

    @pytest.mark.parametrize("target", (f"LazyArray[{STRICT}]", STRICT))
    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_step_with_no_kernel(self, patch, strict, policy, target):
        """A step chained on NumPy is not converted on another library unsaid."""
        pending = _Doubles()(patch.set_backend("lazy"))
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            if policy is None:
                with pytest.raises(LazyOperationError, match="refuses to convert"):
                    pending.set_backend(target)
                return
            match = f"_Doubles has no {STRICT} implementation"
            with pytest.warns(NumpyFallbackWarning, match=match):
                moved = pending.set_backend(target)
            loaded = moved.load()
        assert loaded.backend == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)

    def test_every_pending_step_is_asked(self, patch, strict):
        """Not the first alone: a later step with no kernel is found too."""
        pending = _Doubles()(patch.set_backend("lazy").abs())
        with pytest.raises(LazyOperationError, match="_Doubles has no"):
            pending.set_backend(f"LazyArray[{STRICT}]")

    def test_conversion_a_kernel_makes_itself(self, patch):
        """A kernel which converts on its own is heard when it is computed."""
        dask_array = pytest.importorskip("dask.array")
        # Normalize with a window converts dask data through NumPy itself,
        # which nothing could ask about when it was chained in a window.
        lazy = patch.set_backend("LazyArray[dask]")
        window = lazy.window(time=100, samples=True)
        out = window.normalize("time", window=11, samples=True)
        assert dask_array is not None and out.backend == "lazy"
        with pytest.warns(NumpyFallbackWarning, match="normalize has no dask"):
            out.load()
        with config_context(numpy_fallback="raise"):
            with pytest.raises(NumpyFallbackError, match="normalize has no dask"):
                out.load()

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_operation_in_a_window(self, patch, strict, policy):
        """A window is asked about the operation it runs, when it is chained."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]")
        window = lazy.window(time=100, samples=True)
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="detrend has no"):
                    window.detrend("time")
                return
            with pytest.warns(NumpyFallbackWarning, match="detrend has no"):
                out = window.detrend("time")
        assert out.backend == "lazy"
        # Said once, when it was chained: not again for each box, though
        # the setting refuses by the time the boxes are computed.
        block = patch.data.nbytes // 6
        settings = {"lazy_block_size": block, "numpy_fallback": "raise"}
        with config_context(**settings), warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(chunk_boxes(out.data)) > 1
            loaded = out.load()
        expected = patch.window(time=100, samples=True).detrend("time")
        assert loaded.backend == STRICT
        np.testing.assert_allclose(to_numpy(loaded.data), expected.data, atol=1e-9)
        # One with a kernel for the library is chained with nothing said.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert lazy.window(time=100, samples=True).abs().backend == "lazy"

    def test_steps_with_kernels(self, patch, strict):
        """Steps which run on any library move with nothing said."""
        pending = patch.set_backend("lazy").abs()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            moved = pending.set_backend(f"LazyArray[{STRICT}]")
            # And back to NumPy, where every step has a kernel.
            assert moved.set_backend("lazy").load().equals(patch.abs())
        assert moved.load().backend == STRICT


class TestOperationWithNoContract:
    """An operation which cannot run a chunk at a time obeys the setting."""

    @pytest.mark.parametrize(
        "call",
        (lambda x: x + 1, lambda x: np.add(x, 1), lambda x: dc.spool([x, x]).stack()),
        ids=("arithmetic", "numpy_function", "stack"),
    )
    def test_refused_whatever_the_setting(self, patch, call, reads):
        """What takes the values themselves never reads them unasked."""
        lazy = patch.set_backend("lazy")
        with config_context(numpy_fallback="ignore"):
            with pytest.raises(LazyOperationError):
                call(lazy)
        assert not reads

    def test_on_another_library(self, patch, strict):
        """Read onto the library, an operation with no kernel for it says so too."""
        lazy = patch.set_backend(f"LazyArray[{STRICT}]")
        with config_context(numpy_fallback="warn"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = lazy.detrend("time")
        said = [str(x.message) for x in caught if x.category is NumpyFallbackWarning]
        # Once that the data were read, and once that they were converted.
        assert len(said) == 2 and "read into memory" in said[0]
        assert f"detrend has no {STRICT} implementation" in said[1]
        assert out.backend == STRICT
        np.testing.assert_allclose(
            to_numpy(out.data), patch.detrend("time").data, atol=1e-9
        )

    def test_nothing_to_do_is_still_nothing(self, patch):
        """An operation which changes nothing leaves the lazy patch as it was."""
        lazy = patch.set_backend("lazy")
        assert patch.fill_gaps(time=None) is patch
        with config_context(numpy_fallback="ignore"):
            out = lazy.fill_gaps(time=None)
        # Lazy still, with no step recorded and its own id.
        assert out.backend == "lazy" and out.attrs == lazy.attrs

    def test_read_into_memory(self, patch, reads):
        """Allowed, the data are read and the operation run in memory."""
        lazy = patch.set_backend("lazy")
        with config_context(numpy_fallback="warn"):
            with pytest.warns(NumpyFallbackWarning, match="read into memory"):
                out = lazy.detrend("time")
        assert out.backend == "numpy" and len(reads) == 1
        np.testing.assert_allclose(out.data, patch.detrend("time").data)


class TestLazyRead:
    """Inside `reading_lazily` a read returns a promise to read."""

    def test_reads_nothing(self, path, patch, reads):
        """The patch names its stored array and holds none of it."""
        with reading_lazily("numpy"):
            assert lazy_read_backend() == "numpy"
            lazy = dc.read(path)[0]
        assert lazy_read_backend() is None
        assert lazy.backend == "lazy" and not reads
        assert lazy.attrs == patch.attrs and lazy.data.backend == "numpy"
        assert lazy.load().equals(patch) and len(reads) == 1

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_format_which_reads_its_own_way(self, path, policy, monkeypatch):
        """A format which replaces `read` is not read with nothing said."""
        expected = dc.read(path)[0]
        # As one which reads its files by a method of its own does.
        manager = io_core.FiberIO.manager
        name, version = dc.get_format(path)
        fiber_io = type(manager.get_fiberio(format=name, version=version))

        def own_read(self, *args, **kwargs):
            """Read as the base class does, from a method of the format's."""
            return io_core.FiberIO.read(self, *args, **kwargs)

        monkeypatch.setattr(fiber_io, "read", own_read)
        spool = dc.spool(path).set_backend("lazy")
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            if policy is None:
                with pytest.raises(LazyOperationError, match="a method of its own"):
                    spool[0]
                return
            with pytest.warns(NumpyFallbackWarning, match="a method of its own"):
                out = spool[0]
        assert out.backend == "numpy" and out.equals(expected)

    def test_window_of_the_file(self, path, patch, reads):
        """A selection of neighbouring samples is a window to read later."""
        with reading_lazily("numpy"):
            lazy = dc.read(path, time=(10, 60), samples=True)[0]
        expected = dc.read(path, time=(10, 60), samples=True)[0]
        reads.clear()
        assert lazy.backend == "lazy" and lazy.attrs == expected.attrs
        loaded = lazy.load()
        assert loaded.equals(expected)
        # Only the window is read, not the whole array.
        assert [x.shape for x in reads] == [expected.shape]

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_samples_which_are_apart(self, path, policy):
        """A selection which is no window is read, if the setting allows."""
        wanted = np.array([3, 40, 41, 300])
        expected = dc.read(path, time=wanted, samples=True)[0]
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            with reading_lazily("numpy"):
                if policy is None:
                    with pytest.raises(LazyOperationError, match="not next to each"):
                        dc.read(path, time=wanted, samples=True)
                    return
                with pytest.warns(NumpyFallbackWarning, match="read into memory"):
                    out = dc.read(path, time=wanted, samples=True)[0]
        # In memory; the spool which asked puts it where its chunks compute.
        assert out.backend == "numpy" and out.equals(expected)


class TestSpoolSetBackend:
    """A spool hands its patches out on the backend it was set to."""

    def test_nothing_is_read_until_asked(self, spool, reads):
        """Indexing and iterating hand out promises."""
        lazy = spool.set_backend("lazy")
        assert not reads and len(lazy) == len(spool)
        first = lazy[0]
        every = list(lazy)
        assert not reads
        assert first.backend == "lazy" and {x.backend for x in every} == {"lazy"}
        for made, expected in zip(every, spool, strict=True):
            assert made.attrs == expected.attrs and made.load().equals(expected)

    def test_the_callers_reads_are_their_own(self, spool, path):
        """A read made while a lazy spool is iterated fills memory as usual."""
        for patch in spool.set_backend("lazy"):
            assert patch.backend == "lazy"
            assert dc.read(path)[0].backend == "numpy"
            assert lazy_read_backend() is None

    @pytest.mark.skipif(
        sys.platform == "emscripten", reason="emscripten has no threads"
    )
    def test_prefetching_iterator(self, spool):
        """The iterator which loads ahead hands out the same."""
        backends = [x.backend for x in spool.set_backend("lazy").iterate()]
        assert backends == ["lazy"] * len(spool)

    def test_follows_the_spool(self, spool):
        """A spool made from one keeps its backend."""
        lazy = spool.set_backend("lazy")
        assert lazy.sort("time")[0].backend == "lazy"
        assert lazy[1:3][0].backend == "lazy"
        narrowed = lazy.select(distance=(0, 50), samples=True)
        assert narrowed[0].backend == "lazy"
        expected = spool.select(distance=(0, 50), samples=True)[0]
        assert narrowed[0].load().equals(expected)
        # And is set again without touching the first.
        assert lazy.set_backend("numpy")[0].backend == "numpy"
        assert lazy[0].backend == "lazy" and spool[0].backend == "numpy"

    def test_added_spools_keep_their_backend(self, spool, reads):
        """A sum of spools on one backend is on it; of two, it is refused."""
        lazy = spool.set_backend("lazy")
        both = lazy[:2] + lazy[2:]
        assert len(both) == len(spool)
        assert {x.backend for x in both} == {"lazy"} and not reads
        with pytest.raises(ParameterError, match="different backends"):
            lazy[:2] + spool[2:].set_backend("numpy")
        # A spool with none set takes the other's, so spools gathered onto
        # an empty one keep theirs.
        gathered = sum([lazy[:2], spool[2:]], dc.spool([]))
        assert len(gathered) == len(spool)
        assert {x.backend for x in gathered} == {"lazy"} and not reads
        # And a spool is not the same spool on another backend.
        assert lazy != spool and lazy == spool.set_backend("LazyArray")

    def test_update_of_one_file_keeps_it(self, path):
        """A spool of one file, looked at again, is on the same backend."""
        lazy = dc.spool(path).set_backend("lazy")
        assert lazy.update()[0].backend == "lazy"

    def test_load_reads_everything(self, spool, strict):
        """`load` gives a spool of patches in memory."""
        loaded = spool.set_backend("lazy").load()
        assert [x.backend for x in loaded] == ["numpy"] * len(spool)
        assert all(x.equals(y) for x, y in zip(loaded, spool, strict=True))
        # On the library a lazy backend computes on.
        there = spool.set_backend(f"LazyArray[{STRICT}]").load()
        assert [x.backend for x in there] == [STRICT] * len(spool)

    def test_load_reads_what_could_not_be_lazy(self, undescribed):
        """Asked to read everything, nothing is refused for not being lazy."""
        expected = undescribed.chunk(time=None).load()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            loaded = undescribed.set_backend("lazy").chunk(time=None).load()
        assert len(loaded) == 1 and loaded[0].equals(expected[0])

    def test_another_library(self, spool, strict):
        """An array library by name, lazily or in memory."""
        there = spool.set_backend(STRICT)[0]
        assert there.backend == STRICT
        lazy = spool.set_backend(f"lazy[{STRICT}]")[0]
        assert lazy.backend == "lazy" and lazy.load().backend == STRICT
        np.testing.assert_array_equal(to_numpy(lazy.load().data), spool[0].data)

    def test_refused_when_set(self, spool):
        """A name which is no backend, or no library, is refused at once."""
        with pytest.raises(ParameterError, match="is not a backend"):
            spool.set_backend("lazy[")
        with pytest.raises(ParameterError, match="No array library called"):
            spool.set_backend("not_an_array_library")

    def test_patches_held_in_memory(self, strict):
        """A patch with nothing to read later is put where chunks compute."""
        held = dc.get_example_spool("random_das")
        assert held.set_backend("lazy")[0].backend == "numpy"
        assert held.set_backend(f"lazy[{STRICT}]")[0].backend == STRICT

    def test_patches_held_in_memory_merge(self):
        """Merging what is in memory reads nothing, so nothing is refused."""
        held = dc.get_example_spool("random_das")
        expected = held.chunk(time=None)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            merged = held.set_backend("lazy").chunk(time=None)[0]
        assert merged.backend == "numpy" and merged.equals(expected)

    def test_lazy_patches_held_are_moved_and_asked(self, patch, strict):
        """A patch held lazy computes on the library the spool names."""
        held = dc.spool([patch.set_backend("lazy").abs()])
        out = held.set_backend(f"LazyArray[{STRICT}]")[0]
        assert out.backend == "lazy" and out.data.backend == STRICT
        assert out.load().backend == STRICT
        # And a step it has pending is asked for its kernel there.
        pending = dc.spool([_Doubles()(patch.set_backend("lazy"))])
        with pytest.raises(LazyOperationError, match="_Doubles has no"):
            pending.set_backend(f"LazyArray[{STRICT}]")[0]

    @pytest.mark.parametrize("backend", (None, "lazy"), ids=("unset", "lazy"))
    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_lazy_patches_held_merge(self, spool, policy, backend, reads):
        """Merging patches a spool holds lazy reads them, which is said."""
        # Members the merge has to load: each is an operation on a file.
        held = dc.spool([x.abs() for x in spool.set_backend("lazy")])
        expected = dc.spool([x.abs() for x in spool]).chunk(time=None)[0]
        reads.clear()
        # Whatever backend the spool which holds them is on, or none.
        held = held if backend is None else held.set_backend(backend)
        merged = held.chunk(time=None)
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="cannot stay lazy"):
                    merged[0]
                assert not reads
                return
            with pytest.warns(NumpyFallbackWarning, match="cannot stay lazy"):
                out = merged[0]
        assert out.backend == "numpy" and len(reads) == len(spool)
        np.testing.assert_array_equal(out.data, expected.data)

    def test_lazy_patches_concatenated(self, spool, reads):
        """Lazy patches joined end to end stay lazy, reading nothing."""
        expected = spool.concatenate(time=None)[0]
        reads.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = spool.set_backend("lazy").concatenate(time=None)[0]
        assert out.backend == "lazy" and not reads
        assert out.attrs == expected.attrs and out.coords == expected.coords
        # Each member read once when it is loaded, and no more.
        assert out.load().equals(expected) and len(reads) == len(spool)

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_concatenated_with_operations_pending(self, spool, policy, reads):
        """Lazy patches with operations pending are joined only with leave."""
        held = dc.spool([x.abs() for x in spool.set_backend("lazy")])
        expected = dc.spool([x.abs() for x in spool]).concatenate(time=None)[0]
        reads.clear()
        joined = held.concatenate(time=None)
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="concatenation cannot"):
                    joined[0]
                assert not reads
                return
            with pytest.warns(NumpyFallbackWarning, match="concatenation cannot"):
                out = joined[0]
        assert out.backend == "numpy"
        np.testing.assert_array_equal(out.data, expected.data)

    def test_concatenated_along_a_new_dimension(self, spool):
        """A new dimension cannot be given to lazy data, so it asks."""
        lazy = spool.set_backend("lazy")
        with pytest.raises(LazyOperationError, match="new axis"):
            lazy.concatenate(run=None)[0]

    def test_concatenated_on_another_library(self, spool, strict, monkeypatch):
        """Patches on one library are joined by it, not through NumPy."""
        expected = spool.concatenate(time=None)[0]
        held = dc.spool(list(spool.set_backend(STRICT)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            _refuse_numpy(monkeypatch)
            out = held.concatenate(time=None)[0]
        assert out.backend == STRICT
        monkeypatch.undo()
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)

    @pytest.mark.parametrize("call", ("chunk", "concatenate"))
    def test_numpy_patches_join_another_library(self, spool, strict, call, monkeypatch):
        """Patches in NumPy beside another library's are joined on that library."""
        held = dc.spool([spool[0].set_backend(STRICT), *list(spool)[1:]])
        expected = getattr(spool, call)(time=None)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            _refuse_numpy(monkeypatch)
            out = getattr(held, call)(time=None)[0]
        monkeypatch.undo()
        assert out.backend == STRICT
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)

    def test_numpy_patches_beside_dask_stay_dask(self, spool):
        """Beside dask, NumPy patches join the dask graph and nothing is computed."""
        dask_array = pytest.importorskip("dask.array")
        callbacks = pytest.importorskip("dask.callbacks")
        first = spool[0]
        lazy_first = first.new(data=dask_array.from_array(first.data, chunks=-1))
        held = dc.spool([lazy_first, *list(spool)[1:]])
        expected = spool.concatenate(time=None)[0]
        computed = []
        counter = callbacks.Callback(posttask=lambda *x: computed.append(1))
        with counter, warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = held.concatenate(time=None)[0]
        assert out.backend == "dask" and not computed
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)

    def test_two_other_libraries_are_joined_through_numpy(self, spool, strict):
        """Patches on two libraries other than NumPy meet in NumPy, said."""
        dask_array = pytest.importorskip("dask.array")
        first = spool[0]
        on_dask = first.new(data=dask_array.from_array(first.data, chunks=-1))
        held = dc.spool([on_dask, *[x.set_backend(STRICT) for x in list(spool)[1:]]])
        expected = spool.concatenate(time=None)[0]
        with pytest.warns(NumpyFallbackWarning, match="cannot be joined on one"):
            out = held.concatenate(time=None)[0]
        assert out.backend == "numpy"
        np.testing.assert_array_equal(out.data, expected.data)

    def test_fill_onto_a_grid_on_another_library_is_said(self, whole, tmp_path):
        """Placing members of another library onto a grid converts them, said."""
        pytest.importorskip("dask.array")
        size = whole.shape[whole.get_axis("time")]
        parts = [
            whole.select(time=(0, size // 4), samples=True),
            whole.select(time=(size // 2, size), samples=True),
        ]
        held = dc.spool([x.set_backend("dask") for x in parts])
        time = whole.get_coord("time")
        # A part of the filled stretch, which is placed onto its grid.
        window = (time[size // 10], time[3 * size // 4])
        with pytest.warns(NumpyFallbackWarning, match="Filling onto a grid"):
            held.fill_gaps(time=None).select(time=window)[0]

    def test_new_dimension_on_another_library(self, spool, strict, monkeypatch):
        """A new dimension is added by the library the data are on."""
        expected = spool.concatenate(run=None)[0]
        held = dc.spool(list(spool.set_backend(STRICT)))
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            _refuse_numpy(monkeypatch)
            out = held.concatenate(run=None)[0]
        monkeypatch.undo()
        assert out.backend == STRICT and out.shape == expected.shape
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)

    @pytest.mark.parametrize("call", ("chunk", "concatenate"))
    def test_mixed_dtypes_on_another_library(self, whole, strict, call):
        """Members of different dtypes on one library promote as NumPy's do."""
        size = whole.shape[whole.get_axis("time")]
        first = whole.select(time=(0, size // 2), samples=True)
        second = whole.select(time=(size // 2, size), samples=True)
        parts = [
            first.new(data=(first.data * 10).astype(np.int32)),
            second.new(data=second.data.astype(np.float32)),
        ]
        expected = getattr(dc.spool(parts), call)(time=None)[0]
        held = dc.spool([x.set_backend(STRICT) for x in parts])
        out = getattr(held, call)(time=None)[0]
        assert out.backend == STRICT
        values = to_numpy(out.data)
        assert values.dtype == expected.dtype
        np.testing.assert_array_equal(values, expected.data)

    def test_lazy_patches_with_units_concatenated(self, whole, tmp_path, reads):
        """Members which state their units are joined lazily too."""
        spool = _in_four_files(whole.set_units("m/s"), tmp_path)
        expected = spool.concatenate(time=None)[0]
        reads.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = spool.set_backend("lazy").concatenate(time=None)[0]
        assert out.backend == "lazy" and not reads
        assert out.attrs == expected.attrs and out.load().equals(expected)

    def test_private_dimension_is_not_dropped_lazily(self, patch):
        """Dropping a private dimension is refused, as it was."""
        lazy = patch.set_backend("lazy")
        renamed = lazy.rename_coords(time="_time")
        with pytest.raises(LazyOperationError, match="drop_private_coords"):
            renamed.drop_private_coords()

    def test_recipes_of_different_dtypes_are_read(self, whole, tmp_path):
        """Recipes of different dtypes are read: a lazy join could skip a cast."""
        size = whole.shape[whole.get_axis("time")]
        first = whole.select(time=(0, size // 2), samples=True)
        second = whole.select(time=(size // 2, size), samples=True)
        dc.write(
            first.new(data=first.data.astype(np.float32)), tmp_path / "a.h5", "dasdae"
        )
        dc.write(second, tmp_path / "b.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        expected = spool.concatenate(time=None)[0]
        lazy = spool.set_backend("lazy").concatenate(time=None)
        with pytest.raises(LazyOperationError, match="different dtypes"):
            lazy[0]
        with config_context(numpy_fallback="warn"):
            with pytest.warns(NumpyFallbackWarning, match="different dtypes"):
                out = lazy[0]
        assert out.dtype == expected.dtype
        np.testing.assert_array_equal(out.data, expected.data)

    def test_library_which_narrows_in_a_join(self, monkeypatch):
        """A library which would cast to fewer bits than asked is refused."""
        from dascore.lazy import array as lazy_array  # noqa: PLC0415

        strict = pytest.importorskip("array_api_strict")
        parts = [
            strict.asarray(np.ones(3, np.int32)),
            strict.asarray(np.ones(3, np.float32)),
        ]
        # As a library which keeps 64-bit floats in 32 bits does.
        monkeypatch.setattr(
            lazy_array,
            "asarray_like",
            lambda x, like: strict.asarray(np.asarray(x, np.float32)),
        )
        with pytest.raises(ParameterError, match="holds float64 data as float32"):
            lazy_array.join(parts, axis=0)

    def test_wrapper_numpy_takes_through_its_array_method(self):
        """Something NumPy takes only through `__array__` is converted, said."""
        from dascore.lazy.array import join  # noqa: PLC0415

        class Wrapper:
            """An array-like which NumPy reads through `__array__`."""

            def __init__(self, data):
                self.data = data
                self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim

            def __array__(self, dtype=None, copy=None):
                return self.data

        parts = [Wrapper(np.ones(2)), Wrapper(np.zeros(2))]
        with pytest.warns(NumpyFallbackWarning, match="cannot be joined on one"):
            out = join(parts, axis=0)
        np.testing.assert_array_equal(out, [1, 1, 0, 0])

    def test_summary_of_patches_on_another_library(self, spool, strict):
        """A spool of patches on another library can still say what it holds."""
        there = spool.set_backend(STRICT).load()
        contents = there.get_contents()
        assert len(contents) == len(spool) == len(there)
        assert set(contents["dtype"]) == {"float64"}

    def test_promise_of_another_kind(self):
        """A dask patch a spool holds is a promise already, and left one."""
        dask_array = pytest.importorskip("dask.array")
        patch = dc.get_example_patch()
        there = patch.new(data=dask_array.from_array(patch.data, chunks=(100, 500)))
        out = dc.spool([there]).set_backend("lazy")[0]
        assert out.backend == "dask"
        np.testing.assert_array_equal(to_numpy(out.data), patch.data)


class TestLazyMerge:
    """A spool which merges files hands out one patch which reads none."""

    def test_merge_reads_nothing(self, spool, whole, reads):
        """The merged patch names a window of each file."""
        expected = spool.chunk(time=None)[0]
        reads.clear()
        merged = spool.set_backend("lazy").chunk(time=None)
        assert len(merged) == 1
        lazy = merged[0]
        assert lazy.backend == "lazy" and not reads
        assert lazy.shape == expected.shape and lazy.attrs == expected.attrs
        assert lazy.coords == expected.coords
        loaded = lazy.load()
        assert loaded.equals(expected) and len(reads) == len(spool)
        np.testing.assert_array_equal(loaded.data, whole.transpose(*loaded.dims).data)

    def test_selection_reads_the_files_it_touches(self, spool, reads):
        """A part of the merged patch opens only the files which hold it."""
        lazy = spool.set_backend("lazy").chunk(time=None)[0]
        size = lazy.shape[lazy.get_axis("time")]
        # Inside the second of the four files.
        window = (size // 4 + 5, size // 2 - 5)
        part = lazy.select(time=window, samples=True)
        assert part.backend == "lazy" and not reads
        loaded = part.load()
        assert len(reads) == 1 and "part_1" in reads[0].path
        expected = spool.chunk(time=None)[0].select(time=window, samples=True)
        assert loaded.equals(expected)

    def test_operations_wait(self, spool, reads):
        """What is done to the merged patch is done when it is loaded."""
        lazy = spool.set_backend("lazy").chunk(time=None)[0]
        out = lazy.abs()
        assert out.backend == "lazy" and not reads
        assert out.load().equals(spool.chunk(time=None)[0].abs())

    def test_members_the_index_cannot_describe(self, undescribed, reads):
        """A merge of members read as patches is lazy still, joined by recipe."""
        expected = undescribed.chunk(time=None)[0]
        reads.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            merged = undescribed.set_backend("lazy").chunk(time=None)[0]
        assert merged.backend == "lazy" and not reads
        assert merged.attrs == expected.attrs and merged.load().equals(expected)

    def test_units_spelled_another_way_are_restated(self, tmp_path):
        """The same quantity under another name is converted still, as it was."""
        base = dc.get_example_patch("random_das", shape=(4, 20), time_step=0.01)
        start = np.datetime64("2020-01-01T00:00:00")
        for index, units in enumerate(("m/s", "mm/ms", "m/s")):
            moved = base.update_coords(
                time_min=start + index * np.timedelta64(200, "ms")
            )
            data = np.arange(moved.data.size).reshape(moved.shape) + index
            part = moved.new(data=data.astype(np.int32)).set_units(units)
            dc.write(part, tmp_path / f"part_{index}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        # A selection which trims the members, so each is loaded and merged.
        window = (start + np.timedelta64(50, "ms"), start + np.timedelta64(450, "ms"))
        merged = spool.select(time=window).chunk(time=None, conflict="drop")[0]
        # Restated in the plan's units, which a conversion makes floats of.
        assert merged.attrs.data_units == dc.get_quantity("m/s")
        assert merged.dtype == np.float64

    def test_files_which_state_data_units(self, whole, tmp_path):
        """Members already in the plan's units are not asked to convert."""
        spool = _in_four_files(whole.set_units("m/s"), tmp_path)
        # Shorter than a file, so each patch is a part of one member, and
        # all of them, so one is the merge.
        for size in (0.3, None):
            eager = list(spool.chunk(time=size))
            lazy = list(spool.set_backend("lazy").chunk(time=size))
            assert len(lazy) == len(eager) and {x.backend for x in lazy} == {"lazy"}
            for made, expected in zip(lazy, eager, strict=True):
                assert made.attrs == expected.attrs
                assert made.load().equals(expected)

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_fill_of_a_gap(self, whole, tmp_path, policy):
        """A fill is written into the data, so it is the setting's to allow."""
        spool = _in_four_files(whole, tmp_path)
        (tmp_path / "part_2.h5").unlink()
        # Single precision before the gap, double after it, so the fill
        # is in a wider dtype than the first members are stored in.
        for name in ("part_0.h5", "part_1.h5"):
            part = dc.read(tmp_path / name)[0]
            single = part.new(data=part.data.astype(np.float32))
            dc.write(single, tmp_path / name, "dasdae")
        spool = dc.spool(tmp_path).update()
        expected = spool.fill_gaps(time=None)[0]
        lazy = spool.set_backend("lazy").fill_gaps(time=None)
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="fill cannot stay lazy"):
                    lazy[0]
                return
            with pytest.warns(NumpyFallbackWarning, match="fill cannot stay lazy"):
                filled = lazy[0]
        assert filled.backend == "numpy" and filled.attrs == expected.attrs
        assert filled.coords == expected.coords and np.isnan(filled.data).any()
        np.testing.assert_array_equal(filled.data, expected.data)

    def test_no_gap_to_fill(self, spool, reads):
        """With no gap there is nothing to write, so nothing is read or refused."""
        expected = spool.fill_gaps(time=None)[0]
        reads.clear()
        filled = spool.set_backend("lazy").fill_gaps(time=None)[0]
        assert filled.backend == "lazy" and not reads
        assert filled.attrs == expected.attrs and filled.load().equals(expected)

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_members_in_other_units(self, whole, tmp_path, policy):
        """Members which are rescaled as they merge are read, if allowed."""
        size = whole.shape[whole.get_axis("time")]
        first = whole.select(time=(0, size // 2), samples=True).set_units("m/s")
        second = whole.select(time=(size // 2, size), samples=True)
        dc.write(first, tmp_path / "first.h5", "dasdae")
        dc.write(second.set_units("mm/s"), tmp_path / "second.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        expected = spool.chunk(time=None, conflict="keep_first")[0]
        lazy = spool.set_backend("lazy").chunk(time=None, conflict="keep_first")
        with config_context(numpy_fallback=policy):
            if policy is None:
                # Rescaling is arithmetic, which a lazy member cannot run.
                with pytest.raises(LazyOperationError, match="convert_units"):
                    lazy[0]
                return
            with pytest.warns(NumpyFallbackWarning, match="convert_units"):
                merged = lazy[0]
        # Rescaled, as the eager merge rescales it.
        assert merged.attrs.data_units == expected.attrs.data_units
        np.testing.assert_allclose(merged.data, expected.data)
        assert not np.allclose(merged.data, whole.transpose(*merged.dims).data)

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_dtypes_which_promote_in_order(
        self, whole, tmp_path, policy, monkeypatch, reads
    ):
        """Where promoting in order differs from at once, the members are read."""
        size = whole.shape[whole.get_axis("time")]
        edges = [0, size // 3, 2 * size // 3, size]
        dtypes = (np.int8, np.uint8, np.float16)
        for index, dtype in enumerate(dtypes):
            part = whole.select(time=(edges[index], edges[index + 1]), samples=True)
            part = part.new(data=(part.data * 10).astype(dtype))
            dc.write(part, tmp_path / f"part_{index}.h5", "dasdae")
        spool = dc.spool(tmp_path).update()
        expected = spool.chunk(time=None)[0]
        # From the index the merge casts as it reads, and stays lazy.
        indexed = spool.set_backend("lazy").chunk(time=None)[0]
        assert indexed.backend == "lazy" and indexed.dtype == expected.dtype
        np.testing.assert_array_equal(indexed.load().data, expected.data)
        # Members the index cannot describe are merged as patches, which a
        # join of recipes cannot cast in order.
        monkeypatch.setattr(
            assembly.PatchAssembler, "_member_meta_from_index", lambda *args: None
        )
        lazy = spool.set_backend("lazy").chunk(time=None)
        reads.clear()
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="different dtypes"):
                    lazy[0]
                # Refused before a member is read.
                assert not reads
                return
            with pytest.warns(NumpyFallbackWarning, match="different dtypes"):
                merged = lazy[0]
        # The dtype a member at a time gives, as the eager merge gives it.
        assert merged.dtype == expected.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(merged.data, expected.data)

    def test_dtypes_which_promote_in_order_on_a_library(self, whole):
        """Members on another library are cast there, in the buffer's order."""
        # dask, which holds half precision; the strict library does not.
        pytest.importorskip("dask.array")
        size = whole.shape[whole.get_axis("time")]
        edges = [0, size // 3, 2 * size // 3, size]
        parts = []
        for index, dtype in enumerate((np.int8, np.uint8, np.float16)):
            part = whole.select(time=(edges[index], edges[index + 1]), samples=True)
            parts.append(part.new(data=(part.data * 10).astype(dtype)))
        expected = dc.spool(parts).chunk(time=None)[0]
        held = dc.spool([x.set_backend("dask") for x in parts])
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            merged = held.chunk(time=None)[0]
        assert merged.backend == "dask"
        assert to_numpy(merged.data).dtype == expected.dtype == np.dtype(np.float32)
        np.testing.assert_array_equal(to_numpy(merged.data), expected.data)

    def test_members_in_files_and_in_memory(self, spool, whole):
        """One member held in memory does not excuse reading the others."""
        size = whole.shape[whole.get_axis("time")]
        # The last quarter held in memory, changed so the index cannot stand
        # for it, beside the three files before it.
        held = whole.select(time=(3 * size // 4, size), samples=True) * 1
        mixed = (spool[:3] + dc.spool([held])).set_backend("lazy")
        with pytest.raises(LazyOperationError, match="cannot stay lazy"):
            mixed.chunk(time=None)[0]

    def test_rounding_part_way_along_the_promotion(self, whole):
        """A member rounds where the NumPy buffer rounds it, on any library."""
        pytest.importorskip("dask.array")
        if np.finfo(np.longdouble).nmant <= np.finfo(np.float64).nmant:
            pytest.skip("long double is no wider than double here")
        size = whole.shape[whole.get_axis("time")]
        edges = [0, size // 3, 2 * size // 3, size]
        # int64 then uint64 promote to float64, which rounds 2**53 + 1,
        # before the long double member, which would hold it exactly.
        parts = []
        for index, dtype in enumerate((np.int64, np.uint64, np.longdouble)):
            part = whole.select(time=(edges[index], edges[index + 1]), samples=True)
            data = np.full(part.shape, 2**53 + 1).astype(dtype)
            parts.append(part.new(data=data))
        expected = dc.spool(parts).chunk(time=None)[0]
        merged = dc.spool([x.set_backend("dask") for x in parts]).chunk(time=None)[0]
        assert merged.backend == "dask"
        np.testing.assert_array_equal(to_numpy(merged.data), expected.data)

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_member_in_memory_before_lazy_ones(self, spool, whole, policy, reads):
        """Lazy members after one in memory are not copied into it unasked."""
        size = whole.shape[whole.get_axis("time")]
        # The first quarter held in memory, changed so the index cannot stand
        # for it, before the three files after it.
        held = whole.select(time=(0, size // 4), samples=True) * 1
        expected = (dc.spool([held]) + spool[1:]).chunk(time=None)[0]
        mixed = (dc.spool([held]) + spool[1:]).set_backend("lazy")
        reads.clear()
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="cannot stay lazy"):
                    mixed.chunk(time=None)[0]
                assert not reads
                return
            with pytest.warns(NumpyFallbackWarning, match="cannot stay lazy"):
                merged = mixed.chunk(time=None)[0]
        assert merged.backend == "numpy"
        np.testing.assert_array_equal(merged.data, expected.data)

    def test_file_changed_since_it_was_indexed(self, whole, tmp_path, reads):
        """A file the index no longer describes is read as it is now, lazily."""
        spool = _in_four_files(whole, tmp_path)
        expected = spool.chunk(time=None)[0]
        # Written again: the same samples, at another time.
        again = dc.read(tmp_path / "part_2.h5")[0]
        dc.write(again, tmp_path / "part_2.h5", "dasdae")
        os.utime(tmp_path / "part_2.h5", ns=(1, 1))
        reads.clear()
        merged = spool.set_backend("lazy").chunk(time=None)[0]
        assert merged.backend == "lazy" and not reads
        np.testing.assert_array_equal(merged.load().data, expected.data)

    def test_merge_of_unknown_size(self, spool, monkeypatch, reads):
        """Members merged as loaded patches stay lazy, joined by recipe."""
        expected = spool.chunk(time=None)[0]
        # As for members whose merged length the index's rows do not state.
        monkeypatch.setattr(assembly, "_estimate_merge_samples", lambda *args: None)
        reads.clear()
        merged = spool.set_backend("lazy").chunk(time=None)[0]
        assert merged.backend == "lazy" and not reads
        assert merged.attrs == expected.attrs
        np.testing.assert_array_equal(merged.load().data, expected.data)
