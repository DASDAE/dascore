"""Tests for moving a patch or a spool to another array backend."""

from __future__ import annotations

import warnings
from itertools import pairwise

import numpy as np
import pytest

import dascore as dc
from dascore.config import config_context
from dascore.core.processor import register_kernel
from dascore.exceptions import (
    LazyOperationError,
    NumpyFallbackError,
    ParameterError,
)
from dascore.utils import array_api as array_api_module
from dascore.utils.array_api import (
    fallback_policy,
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

    @pytest.mark.parametrize("name", ("", "lazy[", "LazyArray[a][b]", "nu mpy", "a-b"))
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

    def test_library_which_cannot_be_imported(self):
        """A library which is not installed is said to be the trouble."""
        with pytest.raises(ParameterError, match="No array library called"):
            to_backend(np.zeros(2), "not_an_array_library")


class TestFallbackSetting:
    """One setting decides a conversion through NumPy."""

    def test_default(self):
        """Unset, it refuses for lazy data and warns for any other."""
        assert fallback_policy() == "warn" and fallback_policy(lazy=True) == "raise"

    @pytest.mark.parametrize("policy", ("raise", "warn", "ignore"))
    def test_set(self, policy):
        """Set, it is the same for both."""
        with config_context(numpy_fallback=policy):
            assert fallback_policy() == fallback_policy(lazy=True) == policy

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
        # A window of a stored array is still one to read later.
        assert made.set_backend("numpy") is made

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
        # Asked for on NumPy, it is computed there and not converted.
        assert out.set_backend("numpy").backend == "numpy"

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
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = Halves()(lazy)
            # Nothing is taken through NumPy on the way.
            refuse = pytest.fail
            monkeypatch.setattr(
                "dascore.core.processor.to_numpy",
                lambda *args: refuse("the data were converted to NumPy"),
            )
            loaded = out.load()
        assert handed and set(handed) == {type(strict.asarray(np.zeros(1)))}
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)

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
    @pytest.mark.parametrize("policy", (None, "warn", "ignore"))
    def test_step_with_no_kernel(self, patch, strict, policy, target):
        """A step chained on NumPy is not converted for unsaid elsewhere."""
        pending = _Doubles()(patch.set_backend("lazy"))
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            if policy is None:
                with pytest.raises(LazyOperationError, match="refuses to convert"):
                    pending.set_backend(target)
                return
            if policy == "warn":
                match = f"_Doubles has no {STRICT} implementation"
                with pytest.warns(NumpyFallbackWarning, match=match):
                    moved = pending.set_backend(target)
            else:
                moved = pending.set_backend(target)
            loaded = moved.load()
        assert loaded.backend == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)

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

    def test_refused_by_default(self, patch):
        """Nothing is read for an operation a lazy patch cannot run."""
        lazy = patch.set_backend("lazy")
        with pytest.raises(LazyOperationError):
            lazy.detrend("time")

    @pytest.mark.parametrize("policy", ("warn", "ignore"))
    def test_read_into_memory(self, patch, policy, reads):
        """Allowed, the data are read and the operation run in memory."""
        lazy = patch.set_backend("lazy")
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            if policy == "warn":
                with pytest.warns(NumpyFallbackWarning, match="read into memory"):
                    out = lazy.detrend("time")
            else:
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
        assert lazy.attrs == patch.attrs
        assert lazy.load().equals(patch) and len(reads) == 1

    def test_none_is_an_ordinary_read(self, path):
        """Given no library, reads fill memory as usual."""
        with reading_lazily(None):
            assert dc.read(path)[0].backend == "numpy"

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

    @pytest.mark.parametrize("policy", (None, "warn", "ignore"))
    def test_samples_which_are_apart(self, path, policy, strict):
        """A selection which is no window is read, if the setting allows."""
        wanted = np.array([3, 40, 41, 300])
        expected = dc.read(path, time=wanted, samples=True)[0]
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            with reading_lazily(STRICT):
                if policy is None:
                    with pytest.raises(LazyOperationError, match="not next to each"):
                        dc.read(path, time=wanted, samples=True)
                    return
                if policy == "warn":
                    with pytest.warns(NumpyFallbackWarning, match="read into memory"):
                        out = dc.read(path, time=wanted, samples=True)[0]
                else:
                    out = dc.read(path, time=wanted, samples=True)[0]
        # In memory, on the library the lazy read computes on.
        assert out.backend == STRICT and out.coords == expected.coords
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)


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

    def test_load_reads_everything(self, spool):
        """`load` gives a spool of patches in memory."""
        loaded = spool.set_backend("lazy").load()
        assert [x.backend for x in loaded] == ["numpy"] * len(spool)
        assert all(x.equals(y) for x, y in zip(loaded, spool, strict=True))

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

    @pytest.mark.parametrize("policy", (None, "warn", "ignore"))
    def test_members_the_index_cannot_describe(self, undescribed, policy, reads):
        """A merge which must read its members is the setting's to allow."""
        expected = undescribed.chunk(time=None)[0]
        reads.clear()
        lazy = undescribed.set_backend("lazy").chunk(time=None)
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            if policy is None:
                with pytest.raises(LazyOperationError, match="cannot stay lazy"):
                    lazy[0]
                # Refused before a file is read.
                assert not reads
                return
            if policy == "warn":
                with pytest.warns(NumpyFallbackWarning, match="could not stay lazy"):
                    merged = lazy[0]
            else:
                merged = lazy[0]
        # Read after all, whole, by the merge which reads its members.
        assert merged.backend == "numpy" and merged.equals(expected)

    def test_setting_restored(self):
        """Nothing here leaves reads lazy behind it."""
        assert array_api_module._LAZY_READ.get() is None
