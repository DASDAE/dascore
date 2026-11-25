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
from dascore import backends as backends_module
from dascore.backends import (
    ArrayAPIBackend,
    ArrayBackend,
    JaxBackend,
    LazyBackend,
    NumpyBackend,
    get_backend,
    read_backend,
    reading_onto,
)
from dascore.config import config_context
from dascore.core import processor as processor_module
from dascore.core.processor import register_kernel
from dascore.core.source import ArraySource
from dascore.exceptions import (
    IncompatiblePatchError,
    InvalidFiberIOError,
    LazyOperationError,
    NumpyFallbackError,
    ParameterError,
    PatchError,
)
from dascore.io import core as io_core
from dascore.lazy import execute
from dascore.lazy.execute import chunk_boxes
from dascore.lazy.recipe import Recipe
from dascore.proc.filter import PassFilter
from dascore.proc.window import Windowed
from dascore.utils import array_api as array_api_module
from dascore.utils import patch_assembly as assembly
from dascore.utils.array_api import backend_name, to_numpy
from dascore.warnings import NumpyFallbackWarning

STRICT = "array_api_strict"
MISSING = "library_not_installed"


class _Strict(ArrayAPIBackend):
    """The strict library, as a package which ships no backend registers it."""

    name = STRICT
    array_type = "array_api_strict._array_object.Array"


class _Missing(ArrayAPIBackend):
    """A registered backend whose library is not installed."""

    name = MISSING
    array_type = "library_not_installed.Array"


class _Dask(ArrayAPIBackend):
    """Dask arrays made from ones in memory."""

    name = "dask"
    array_type = "dask.array.Array"
    module = "dask.array"


class _Wrapped:
    """An array type of a backend's own, of no array library."""

    def __init__(self, data):
        self.shape, self.dtype, self.ndim = data.shape, data.dtype, data.ndim
        self._data = data

    def __array__(self, dtype=None, copy=None):
        """Return the values, which costs something."""
        return self._data

    def __getitem__(self, index):
        """Return NumPy's values, as a chunk of a lazy array is cut from it."""
        return self._data[index]


@pytest.fixture(autouse=True)
def _registry():
    """Forget, after each test, the backends it registered."""
    before = dict(backends_module._BACKENDS)
    yield
    backends_module._BACKENDS.clear()
    backends_module._BACKENDS.update(before)


@pytest.fixture()
def filter_kernels():
    """Forget, after a test, the kernels it registered for the pass filter."""
    before = dict(PassFilter.__dict__.get("_kernels", {}))
    yield
    PassFilter._kernels = before


def _on(patch, backend):
    """Return a patch a spool holds, as the spool set to a backend hands it out."""
    return dc.spool([patch]).set_backend(backend)[0]


def _lazy(patch, backend="lazy"):
    """Return the patch of a file, as a spool of it set to a lazy backend reads it."""
    return dc.spool(patch._source.path).set_backend(backend)[0]


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
    """A backend is an instance, or a string which names a registered one."""

    @pytest.mark.parametrize(
        "name, expected",
        (
            ("numpy", NumpyBackend()),
            ("JAX", JaxBackend()),
            ("lazy", LazyBackend()),
            ("LazyArray", LazyBackend()),
            ("lazyarray[ jax ]", LazyBackend(compute=JaxBackend())),
            ("LazyArray[array_api_strict]", LazyBackend(compute=_Strict())),
            ("lazy[numpy]", LazyBackend(compute=NumpyBackend())),
        ),
    )
    def test_spellings(self, name, expected):
        """A name or an alias, in any case, with what it takes in brackets."""
        assert get_backend(name) == expected

    @pytest.mark.parametrize(
        "name",
        (
            "",
            "lazy[",
            "nu mpy",
            "a-b",
            "a.b",
            "lazy[a.b]",
            "LazyArray[a][b]",
            "not_an_array_library",
        ),
    )
    def test_not_a_backend(self, name):
        """What names no registered backend is refused, and says which are."""
        with pytest.raises(ParameterError, match=r"is not a backend.*numpy"):
            get_backend(name)

    @pytest.mark.parametrize("name", ("numpy[jax]", "jax[gpu]"))
    def test_brackets_a_backend_does_not_take(self, name):
        """Only a backend which says so takes anything in brackets."""
        with pytest.raises(ParameterError, match="nothing in square brackets"):
            get_backend(name)

    def test_lazy_computes_on_an_array_library(self):
        """A lazy backend computing on a lazy one would never read."""
        with pytest.raises(ParameterError, match="not on another lazy backend"):
            get_backend("lazy[lazy]")
        with pytest.raises(ParameterError, match="not on another lazy backend"):
            LazyBackend(compute=LazyBackend())

    def test_backend_which_reads_for_itself(self, path, patch, strict):
        """A backend overriding `from_recipe` makes the array a read hands out."""
        handed = []

        class Reading(_Strict):
            """The strict library, reading each recipe itself."""

            def from_recipe(self, recipe):
                """Read the recipe, and say that this was asked."""
                handed.append(recipe)
                return super().from_recipe(recipe)

        # A subclass which states no name is not registered in its parent's place.
        assert type(get_backend(STRICT)) is _Strict
        out = dc.spool(path).set_backend(Reading())[0]
        assert len(handed) == 1 and backend_name(out.data) == STRICT
        assert np.array_equal(to_numpy(out.data), patch.data)
        # What it cannot make of one stored window is read as usual, with
        # nothing to refuse: it never promised not to read.
        wanted = np.array([3, 40, 41, 300])
        expected = dc.read(path, time=wanted, samples=True)[0]
        with warnings.catch_warnings(), reading_onto(Reading()):
            warnings.simplefilter("error")
            apart = dc.read(path, time=wanted, samples=True)[0]
        assert len(handed) == 1 and apart.equals(expected)

    def test_backend_which_reads_an_array_of_its_own_kind(self, path, patch):
        """What a backend makes of a stored array is handed out as it made it."""

        class Wrapping(NumpyBackend):
            """NumPy's, reading each recipe into a wrapper which is not indexed."""

            def from_recipe(self, recipe):
                """Return the recipe's array, wrapped."""
                return _Wrapped(recipe.load())

        out = dc.spool(path).set_backend(Wrapping())[0]
        assert type(out.data) is _Wrapped and out.load().equals(patch)

    def test_backend_which_reads_another_dtype_is_refused(self, spool):
        """A read which is not the array its file holds is refused, merged or not."""

        class Narrowing(NumpyBackend):
            """NumPy's, reading each recipe into fewer bits."""

            def from_recipe(self, recipe):
                """Return the recipe's array in another dtype."""
                return recipe.load().astype(np.float32)

        narrow = spool.set_backend(Narrowing())
        for read in (lambda: narrow[0], lambda: narrow.chunk(time=None)[0]):
            with pytest.raises(InvalidFiberIOError, match="float32"):
                read()

    def test_backend_which_reads_for_itself_and_a_format_which_does(
        self, path, patch, monkeypatch
    ):
        """A format with its own `read` reads, and only a lazy backend objects."""

        class Reading(NumpyBackend):
            """NumPy, reading each recipe itself."""

            def from_recipe(self, recipe):
                """Fail: a format which reads its own way hands out no recipe."""
                raise AssertionError("the format's own read was not used")

        manager = io_core.FiberIO.manager
        name, version = dc.get_format(path)
        fiber_io = type(manager.get_fiberio(format=name, version=version))

        def own_read(self, *args, **kwargs):
            """Read as the base class does, from a method of the format's."""
            return io_core.FiberIO.read(self, *args, **kwargs)

        monkeypatch.setattr(fiber_io, "read", own_read)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = dc.spool(path).set_backend(Reading())[0]
        assert out.equals(patch)

    def test_an_instance_is_handed_back(self):
        """An instance is the backend it is, options and all."""
        backend = LazyBackend(compute=_Strict(device="somewhere"))
        assert get_backend(backend) is backend
        assert backend != LazyBackend(compute=_Strict())

    def test_string_and_repr(self):
        """The string names the backend, and the repr shows its options."""
        backend = LazyBackend(compute=_Strict())
        assert str(backend) == f"lazy[{STRICT}]" and str(LazyBackend()) == "lazy"
        assert get_backend(str(backend)) == backend
        assert repr(_Strict(device="gpu")) == "_Strict(device='gpu')"

    def test_an_option_which_is_not_one(self):
        """A misspelled option is refused, not dropped."""
        with pytest.raises(ValueError, match="devise"):
            _Strict(devise="gpu")

    def test_from_array(self, strict):
        """An array is put on the library, and left where it already is."""
        data = np.arange(6.0).reshape(2, 3)
        there = _Strict().from_array(data)
        assert type(there) is type(strict.asarray(data))
        assert _Strict().from_array(there) is there
        assert np.array_equal(NumpyBackend().from_array(there), data)
        assert NumpyBackend().from_array(data) is data

    def test_device(self, strict, monkeypatch):
        """Arrays are put on the device given, and moved there from another."""
        device = strict.Device("device1")
        backend = _Strict(device=device)
        there = backend.from_array(np.arange(3.0))
        assert there.device == device and backend.from_array(there) is there
        # Loaded, a lazy backend's arrays are those of the one it computes on.
        assert LazyBackend(compute=backend)._native == backend._native == STRICT
        elsewhere = _Strict().from_array(np.arange(3.0))
        assert elsewhere.device != device
        # Moved by its own library, with nothing taken through NumPy.
        _refuse_numpy(monkeypatch)
        moved = backend.from_array(elsewhere)
        assert moved.device == device and bool(strict.all(moved == there))

    def test_options_reach_the_arrays_a_spool_hands_out(self, spool, strict):
        """An instance's options are the spool's, in memory and lazily."""
        device = strict.Device("device1")
        backend = _Strict(device=device)
        assert spool.set_backend(backend)[0].data.device == device
        lazy = spool.set_backend(LazyBackend(compute=backend))[0]
        assert lazy.data._backend == backend
        assert lazy.abs().load().data.device == device

    def test_pending_steps_are_not_asked_again_for_a_device(self, patch, strict):
        """The same library on another device runs the kernels it ran."""
        with config_context(numpy_fallback="ignore"):
            pending = _Doubles()(_lazy(patch, f"lazy[{STRICT}]"))
        device = strict.Device("device1")
        # Under the default setting, which would refuse were it asked.
        moved = _on(pending, LazyBackend(compute=_Strict(device=device)))
        assert moved.data._backend == _Strict(device=device)

    def test_plugin_registers_the_backend(self, monkeypatch):
        """A name nothing registered loads the entry points named for it."""
        loaded = [True, False]

        def load(asked):
            """Register a backend, as importing a plugin's module does."""
            assert asked == "plugged_backend"

            class Plugged(ArrayAPIBackend):
                """A backend a plugin supplies."""

                name = "plugged_backend"
                array_type = "plugged.Array"

            return loaded.pop(0)

        monkeypatch.setattr(backends_module, "load_backend_plugins", load)
        assert str(get_backend("plugged_backend")) == "plugged_backend"
        # One whose module failed part way leaves no backend half made.
        backends_module._BACKENDS.pop("plugged_backend")
        with pytest.raises(ParameterError, match="is not a backend"):
            get_backend("plugged_backend")
        assert "plugged_backend" not in backends_module._BACKENDS


class _Fast(NumpyBackend):
    """Another namespace over NumPy's arrays, as a package of kernels registers."""

    name = "fast_numpy"


class _Tripled(dc.PatchProcessor):
    """An operation the namespace has a kernel of its own for."""

    name = None

    def chunk_contract(self, dtype, plan):
        """Return that each sample is made from its own."""
        return _Doubles().chunk_contract(dtype, plan)

    def numpy_kernel(self, data):
        """Return the data doubled, as NumPy's own kernel does."""
        return data * 2


@register_kernel(_Tripled, "fast_numpy")
def _tripled_fast(self, data):
    """Return the data tripled, so the namespace's kernel shows."""
    return data * 3


def _triples(patch) -> bool:
    """Whether the namespace's kernel, and not NumPy's, runs on a patch."""
    made, data = to_numpy(_Tripled()(patch).data), to_numpy(patch.data)
    assert np.array_equal(made, data * 3) or np.array_equal(made, data * 2)
    return bool(np.array_equal(made, data * 3))


class TestNamespace:
    """A backend's namespace chooses the kernels; its array type, the arrays."""

    @pytest.fixture()
    def fast(self, spool):
        """A patch a spool set to the namespace handed out, in memory."""
        return spool.set_backend("fast_numpy")[0]

    def test_arrays_are_its_array_type_s(self, spool, fast):
        """Under another namespace the arrays are the ones its parent makes."""
        assert _Fast.array_type is np.ndarray
        assert type(fast.data) is np.ndarray and fast.equals(spool[0])

    def test_namespace_needs_an_array_type(self):
        """A backend which states a namespace says which class its arrays are."""
        with pytest.raises(TypeError, match="no array_type"):

            class Unfinished(ArrayAPIBackend):
                """A namespace over nothing."""

                name = "unfinished"

    def test_namespace_is_in_lower_case(self):
        """A namespace is spelled one way, since kernels are matched exactly."""
        with pytest.raises(TypeError, match="lower-case letters"):

            class Shouting(NumpyBackend):
                """A namespace a kernel under "fast" would never match."""

                name = "Fast"

        # Nor one a string could not name.
        with pytest.raises(TypeError, match="lower-case letters"):

            class Hyphenated(NumpyBackend):
                """A name `get_backend` could not read."""

                name = "fast-numpy"

    def test_namespace_which_is_taken(self):
        """Another class does not take a namespace; the same one made again may."""
        with pytest.raises(TypeError, match=r"'numpy' is taken by .*NumpyBackend"):

            class Impostor(_Strict):
                """Another library's arrays, under NumPy's namespace."""

                name = "numpy"

        assert type(get_backend("numpy")) is NumpyBackend

        def made_again():
            """Make a class, as running a notebook cell twice makes it twice."""

            class Twice(NumpyBackend):
                """A backend defined more than once."""

                name = "made_twice"

            return Twice

        first, second = made_again(), made_again()
        assert first is not second and type(get_backend("made_twice")) is second

    @pytest.mark.parametrize("where", ("numpy.ndaray", "numpy.core.nothing.Array"))
    def test_array_type_which_names_no_class(self, spool, where):
        """A misspelled array type is refused when a spool is set, not ignored."""

        class Misspelled(NumpyBackend):
            """NumPy's arrays, told by a class which does not exist."""

            name = "misspelled"
            array_type = where

        for backend in ("misspelled", "lazy[misspelled]"):
            with pytest.raises(ParameterError, match="not an instance of its array"):
                spool.set_backend(backend)

    def test_lazy_chunks_are_read_by_the_backend(self, spool):
        """A backend which reads for itself reads each chunk a lazy array needs."""
        read = []

        class Reading(NumpyBackend):
            """NumPy's arrays, reading each recipe itself."""

            def from_recipe(self, recipe):
                """Read the recipe, and say that this was asked."""
                read.append(recipe.shape)
                return super().from_recipe(recipe)

        lazy = spool.set_backend(LazyBackend(compute=Reading()))[0].abs()
        assert not read
        with config_context(lazy_block_size=lazy.data.size * 8 // 5):
            boxes = chunk_boxes(lazy.data)
            loaded = lazy.load()
        assert len(boxes) > 1 and len(read) == len(boxes)
        assert loaded.equals(spool[0].abs())

    def test_a_namespace_of_any_name(self, spool, strict):
        """The namespace is any name; the array type says what the arrays are."""

        class Named(_Strict):
            """Someone's own kernels, on another library's arrays."""

            name = "any_name_at_all"

        patch = spool.set_backend("any_name_at_all")[0]
        assert backend_name(patch.data) == STRICT and patch._backend == Named()
        # With no kernel of its own it runs what the arrays would anyway.
        with pytest.warns(NumpyFallbackWarning, match=f"no {STRICT} impl"):
            out = _Doubles()(patch)
        assert backend_name(out.data) == STRICT

    def test_arrays_told_by_their_class(self, spool):
        """A backend's arrays are the instances of its array type, and no others."""

        class Wrapping(ArrayBackend):
            """A backend which makes its own arrays, and has its own kernels."""

            name = "wrapping"
            array_type = _Wrapped

            def from_array(self, array):
                """Return the array wrapped, or as it is if it is already."""
                return array if self._owns(array) else _Wrapped(np.asarray(array))

        ran = []

        @register_kernel(_Tripled, "wrapping")
        def _tripled_wrapped(self, data):
            """Say that the namespace's kernel ran, and on what."""
            ran.append(type(data))
            return _Wrapped(np.asarray(data) * 3)

        patch = spool.set_backend("wrapping")[0]
        assert type(patch.data) is _Wrapped and _triples(patch)
        assert ran == [_Wrapped]
        # A patch of the backend holds its arrays: other data are made them.
        plain = np.asarray(patch.data)
        assert type(patch.new(data=plain).data) is _Wrapped
        # One of no backend holds what it is given, and runs what that runs.
        assert not _triples(patch.set_backend(None).new(data=plain))
        assert ran == [_Wrapped]
        # A lazy load is one of its arrays however many boxes it took.
        lazy = spool.set_backend("lazy[wrapping]")[0].abs()
        size = plain.nbytes
        for block, several in ((size * 4, False), (size // 5, True)):
            with config_context(lazy_block_size=block):
                assert (len(chunk_boxes(lazy.data)) > 1) is several
                loaded = lazy.load()
            assert type(loaded.data) is _Wrapped
            assert np.array_equal(np.asarray(loaded.data), np.abs(plain))

    @pytest.mark.parametrize(
        "where, owned",
        (("numpy.ndarray", True), ("not_imported.Array", False), ("numpy.no", False)),
    )
    def test_array_type_by_name(self, where, owned):
        """A string says where the class is, and imports nothing for it."""

        class Named(NumpyBackend):
            """NumPy's arrays, told by a class named in a string."""

            array_type = where

        assert Named()._owns(np.ones(2)) is owned
        assert "not_imported" not in sys.modules

    def test_its_kernel_runs(self, spool, fast):
        """An operation with a kernel for the namespace runs it."""
        assert _triples(fast)
        # And a patch of no namespace runs what its array does.
        assert not _triples(spool[0])

    def test_no_kernel_runs_what_the_array_would(self, fast):
        """Without one the array's own kernel runs, and nothing is said of it."""
        with config_context(numpy_fallback="raise"), warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _Doubles()(fast)
            np.testing.assert_array_equal(out.data, fast.data * 2)
            # Nor of an operation which has only an array API kernel.
            assert fast.abs().equals(dc.spool([fast]).set_backend("numpy")[0].abs())

    def test_a_class_which_wrote_its_own_kernel_means_it(self):
        """A namespace kernel registered against a parent does not answer."""

        class Own(_Tripled):
            """A subclass with a kernel of its own."""

            name = None

            def numpy_kernel(self, data):
                """Return the data times five."""
                return data * 5

        assert Own.kernel_for("numpy", "fast_numpy") is Own.__dict__["numpy_kernel"]
        assert _Tripled.kernel_for("numpy", "fast_numpy") is _tripled_fast
        assert _Tripled.kernel_for("numpy") is _Tripled.__dict__["numpy_kernel"]

    def test_plugins_load_for_the_array_and_the_namespace(self, monkeypatch):
        """Kernels a package has for the arrays load, as do the namespace's."""
        asked = []
        monkeypatch.setattr(processor_module, "load_backend_plugins", asked.append)
        _Tripled.kernel_for(STRICT, "fast_numpy")
        assert asked == [STRICT, "fast_numpy"]

    def test_operations_pass_it_on(self, fast):
        """What an operation makes is of the namespace its input was."""
        made = (
            fast.abs(),
            fast.update_attrs(station="here"),
            fast.select(time=(0, 10), samples=True),
            fast.new(data=fast.data + 1),
            fast.drop_data().to_patch(fast.data),
            _Tripled()(fast),
        )
        assert all(_triples(x) for x in made)

    def test_other_data_are_made_its_arrays(self, fast, strict):
        """A patch of a backend holds its arrays, whatever it is given."""
        there = fast.new(data=strict.asarray(fast.data))
        assert type(there.data) is np.ndarray and there.backend == _Fast()
        assert _triples(there)
        # Of no backend, the patch holds what it is given and runs what that runs.
        loose = fast.set_backend(None).new(data=strict.asarray(fast.data))
        assert backend_name(loose.data) == STRICT and loose.backend is None
        with pytest.warns(NumpyFallbackWarning, match=f"no {STRICT} impl"):
            assert not _triples(loose)

    def test_two_patches(self, fast, spool, strict):
        """Patches of one kind of array combine, on the first one's backend."""
        plain = spool[0]
        assert (fast + plain).backend == _Fast() and _triples(fast + plain)
        assert (plain + fast).backend is None and not _triples(plain + fast)
        # Of another library they do not, whichever comes first.
        there = plain.set_backend(STRICT)
        for one, two in ((fast, there), (there, fast)):
            with pytest.raises(
                IncompatiblePatchError, match="arrays of one array library"
            ):
                one + two
        # But a NumPy patch of no backend is taken in by the other's library.
        for out in (there + plain, plain + there):
            assert backend_name(out.data) == STRICT
        assert (there + plain).backend == _Strict() and (plain + there).backend is None

    def test_patches_a_spool_puts_together(self, spool):
        """Merged or concatenated, they are of the first one's backend."""
        held = dc.spool([x.set_backend("fast_numpy") for x in spool])
        for call in ("chunk", "concatenate"):
            out = getattr(held, call)(time=None)[0]
            assert out.backend == _Fast() and _triples(out)
        # Unless the spool was set to one, which is what it hands out.
        assert held.set_backend("numpy").chunk(time=None)[0].backend == NumpyBackend()

    def test_another_backend_replaces_it(self, fast):
        """A spool set to another backend hands the patch out with its kernels."""
        plain = _on(fast, "numpy")
        assert plain.equals(fast) and not _triples(plain)
        # One already of the namespace is handed out as it is.
        assert _on(fast, "fast_numpy") is fast

    def test_lazy_chunks_run_its_kernels(self, spool, fast):
        """A lazy backend computing on the namespace runs its kernels."""
        lazy = spool.set_backend("lazy[fast_numpy]")[0]
        assert backend_name(lazy.data) == "lazy"
        block = fast.data.nbytes // 6
        # Under the default setting, which refuses a conversion on lazy data.
        with config_context(lazy_block_size=block), warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _Doubles()(_Tripled()(lazy))
            assert len(chunk_boxes(out.data)) > 3
            loaded = out.load()
        np.testing.assert_array_equal(loaded.data, fast.data * 6)
        # In memory it is still of the namespace it was computed on.
        assert type(loaded.data) is np.ndarray and _triples(loaded)

    def test_merged_by_the_spool_is_of_it(self, spool):
        """What a spool set to the namespace merges is of the namespace too."""
        for name in ("fast_numpy", "lazy[fast_numpy]"):
            assert _triples(spool.set_backend(name).chunk(time=None)[0].load())

    def test_operation_which_reads_lazy_data_keeps_it(self, spool, fast):
        """Read into memory for an operation, the patch is still of the namespace."""
        lazy = spool.set_backend("lazy[fast_numpy]")[0]
        with config_context(numpy_fallback="ignore"):
            out = lazy.detrend("time")
        assert type(out.data) is np.ndarray and _triples(out)
        np.testing.assert_allclose(out.data, fast.detrend("time").data)

    def test_over_arrays_which_compute_later(self):
        """A namespace over arrays which compute later is planned as they are."""
        dask_array = pytest.importorskip("dask.array")

        class Flavour(_Dask):
            """Dask arrays, under a namespace of its own."""

            name = "dask_flavour"

        data = np.ones((12, 400), np.uint8)
        coords = {"distance": np.arange(12.0), "time": np.arange(400.0)}
        patch = dc.Patch(data, coords, ("distance", "time"))
        patch = patch.new(data=dask_array.from_array(data, chunks=(12, 100)))
        there = dc.spool([patch]).set_backend("dask_flavour")[0]
        assert there._backend == Flavour()
        # Windows a detrend widens only where one holds NaN: stated
        # before anything is computed, as for the library's own arrays.
        out = there.window(time=8, samples=True, fill=np.nan).detrend("time")
        blocks = [x.compute().dtype for x in out.data.blocks]
        assert set(blocks) == {np.dtype(out.dtype)} == {np.dtype(np.float64)}

    def test_operation_run_in_a_window(self, spool, fast):
        """An operation a window runs is of the backend its patch is of."""
        windowed = Windowed(_Tripled(), time=100, samples=True)
        np.testing.assert_allclose(windowed(fast).data, fast.data * 3)
        np.testing.assert_allclose(windowed(spool[0]).data, fast.data * 2)
        # And in each chunk of a lazy patch computing on the backend.
        lazy = windowed(spool.set_backend("lazy[fast_numpy]")[0])
        with config_context(lazy_block_size=fast.data.nbytes // 6):
            assert len(chunk_boxes(lazy.data)) > 3
            np.testing.assert_allclose(lazy.load().data, fast.data * 3)

    def test_kernel_asked_for_with_no_name_is_the_array_s(self, spool, fast):
        """Inside a backend's kernel, the array's own kernel is still to be had."""
        inside = []

        class Wraps(dc.PatchProcessor):
            """An operation whose named kernel is the array's own, and more."""

            name = None

            def numpy_kernel(self, data):
                """Return the data doubled."""
                return data * 2

        @register_kernel(Wraps, "fast_numpy")
        def _wraps_fast(self, data):
            """Run the array's own kernel, and an operation on another patch."""
            own = type(self).kernel_for("numpy")
            # A patch of no backend runs its array's kernel in here too.
            inside.append(_triples(spool[0]))
            return own(self, data) + 1

        np.testing.assert_array_equal(Wraps()(fast).data, fast.data * 2 + 1)
        assert inside == [False]

    def test_spool_set_to_a_backend_puts_kinds_on_it(self, spool, strict):
        """Members of different kinds are made the spool's backend's, then joined."""
        held = dc.spool([spool[0].set_backend(STRICT), *list(spool)[1:]])
        expected = spool.chunk(time=None)[0]
        for call in ("chunk", "concatenate"):
            out = getattr(held.set_backend(STRICT), call)(time=None)[0]
            assert out.backend == _Strict()
            np.testing.assert_array_equal(to_numpy(out.data), expected.data)
        # Files and a member in memory, of another kind than the files' are.
        mixed = (spool[1:] + dc.spool([spool[0].set_backend(STRICT)])).set_backend(
            STRICT
        )
        out = mixed.chunk(time=None)[0]
        assert out.backend == _Strict() and out.shape == expected.shape

    def test_patch_set_to_a_backend_is_another_patch(self, spool):
        """A spool holding a patch and the same one of a backend holds two."""
        plain = spool[0]
        there = plain.set_backend("numpy")
        both = dc.spool([plain, there])
        assert len(both) == 2
        assert [x.backend for x in both] == [None, NumpyBackend()]

    def test_stacked_and_copied_patches_are_of_it(self, fast):
        """A stack of patches, and a patch made of one, are of its backend."""
        assert dc.Patch(fast).backend == _Fast()
        stacked = dc.spool([fast, fast.new(data=fast.data * 2)]).stack(dim_vary="time")
        # Both were added, and the sum is of the backend they were of.
        np.testing.assert_array_equal(stacked.data, fast.data * 3)
        assert stacked.backend == _Fast() and _triples(stacked)

    def test_stack_of_another_library_is_refused(self, spool, strict):
        """Nor are patches of different array libraries stacked."""
        plain = spool[0]
        there = plain.new(data=strict.asarray(plain.data))
        with pytest.raises(IncompatiblePatchError, match="arrays of one array lib"):
            dc.spool([plain, there]).stack(dim_vary="time")
        # One which is passed over, being of another kind, is not asked.
        other = there.update_attrs(station="another")
        out = dc.spool([plain, other]).stack(dim_vary="time", check_behavior="ignore")
        np.testing.assert_array_equal(out.data, plain.data)

    def test_lazy_patches_of_one_kind_join_on_the_first_s(self, spool, reads):
        """Lazy patches on unequal backends of one kind are joined, unread."""
        lazy = list(spool.set_backend("lazy"))
        fast = list(spool.set_backend("lazy[fast_numpy]"))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = dc.spool([fast[0], *lazy[1:]]).concatenate(time=None)[0]
        assert backend_name(out.data) == "lazy" and not reads
        assert out.backend == LazyBackend(compute=_Fast()) and _triples(out.load())

    def test_lazy_steps_are_handed_its_arrays(self, spool):
        """Between two steps of a lazy patch the data are the backend's own."""

        class Wrapping(ArrayBackend):
            """A backend which makes its own arrays, and has a kernel for them."""

            name = "wrapping_steps"
            array_type = _Wrapped

            def from_array(self, array):
                """Return the array wrapped, or as it is if it is already."""
                return array if self._owns(array) else _Wrapped(np.asarray(array))

        @register_kernel(_Tripled, "wrapping_steps")
        def _tripled_wrapped(self, data):
            """Read the wrapper's own storage, as a kernel written for it does."""
            return _Wrapped(data._data * 3)

        lazy = spool.set_backend("lazy[wrapping_steps]")[0]
        expected = np.abs(spool[0].data) * 3
        # NumPy's `abs` makes an ndarray of the wrapper; the next step, as
        # the same two operations in memory, is handed a wrapper again.
        for patch in (_Tripled()(lazy.abs()).load(), _Tripled()(lazy.load().abs())):
            assert type(patch.data) is _Wrapped
            np.testing.assert_array_equal(np.asarray(patch.data), expected)

    def test_lazy_read_which_is_not_the_stored_array(self, spool):
        """A backend which reads another array than is stored is refused at load."""

        class Narrowing(NumpyBackend):
            """NumPy's arrays, reading each recipe into fewer bits."""

            def from_recipe(self, recipe):
                """Return the recipe's array in another dtype."""
                return recipe.load().astype(np.float32)

        lazy = spool.set_backend(LazyBackend(compute=Narrowing()))[0].abs()
        with pytest.raises(PatchError, match="must make the array the recipe"):
            lazy.load()

        class Short(NumpyBackend):
            """NumPy's arrays, reading each recipe a sample short."""

            def from_recipe(self, recipe):
                """Return all but the last sample of the recipe's array."""
                return recipe.load()[:, :-1]

        lazy = spool.set_backend(LazyBackend(compute=Short()))[0].abs()
        with pytest.raises(PatchError, match="must make the array the recipe"):
            lazy.load()

    def test_merge_of_kinds_is_refused_before_anything_is_read(
        self, spool, strict, reads
    ):
        """Members of different libraries are refused, lazy ones unread."""
        lazy = list(spool.set_backend("lazy"))
        first = spool[0].new(data=spool[0].data.astype(np.float32))
        mixed = dc.spool([first.set_backend(STRICT), *lazy[1:]])
        reads.clear()
        with config_context(numpy_fallback="ignore"):
            with pytest.raises(
                IncompatiblePatchError, match="arrays of one array library"
            ):
                mixed.chunk(time=None)[0]
        assert not reads

    def test_promise_of_another_kind_loads_to_numpy(self, spool):
        """A dask patch of a backend which makes dask arrays loads to NumPy's."""
        pytest.importorskip("dask.array")
        there = spool[0].set_backend("dask")
        assert there.backend == _Dask()
        loaded = there.load()
        assert type(loaded.data) is np.ndarray and loaded.backend is None
        assert loaded.load() is loaded and loaded.equals(spool[0])

    def test_array_of_no_library_in_arithmetic(self, spool, strict):
        """An array-like of no library, with no backend, is NumPy's there too."""
        plain = spool[0]
        held = plain.new(data=_Wrapped(np.asarray(plain.data)))
        there = plain.set_backend(STRICT)
        out = there + held
        assert backend_name(out.data) == STRICT
        np.testing.assert_array_equal(to_numpy(out.data), plain.data * 2)

    def test_window_and_impulse_window_on_another_library(
        self, spool, strict, filter_kernels
    ):
        """An operation a window runs is of the backend on its library too."""
        ran = []

        class Named(_Strict):
            """The strict library's arrays, with kernels of its own."""

            name = "strict_named"

        @register_kernel(_Tripled, "strict_named")
        def _tripled_there(self, data):
            """Triple the data where they are, and say that this ran."""
            ran.append(type(data))
            return data * 3

        @register_kernel(PassFilter, "strict_named")
        def _filter_there(self, data, **plan):
            """Say that this ran, and do what NumPy's does."""
            ran.append("filter")
            kernel = PassFilter.kernel_for("numpy")
            return strict.asarray(kernel(self, to_numpy(data), **plan))

        patch = spool[0].set_backend("strict_named")
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = Windowed(_Tripled(), time=100, samples=True)(patch)
            assert ran and set(ran) == {type(patch.data)}
            np.testing.assert_allclose(to_numpy(out.data), spool[0].data * 3)
            ran.clear()
            filtered = patch.impulse_window(time=1e-3).pass_filter(time=(1, 20))
        assert "filter" in ran and backend_name(filtered.data) == STRICT

    def test_impulse_window_on_its_own_arrays(self, spool, fast, filter_kernels):
        """An impulse window runs the backend's kernel on NumPy's arrays too."""
        ran = []

        @register_kernel(PassFilter, "fast_numpy")
        def _filter_fast(self, data, **plan):
            """Say that this ran, and do what NumPy's does."""
            ran.append(1)
            return PassFilter.kernel_for("numpy")(self, data, **plan)

        expected = spool[0].impulse_window(time=1e-3).pass_filter(time=(1, 20))
        assert not ran
        out = fast.impulse_window(time=1e-3).pass_filter(time=(1, 20))
        assert ran and out.equals(expected)

    def test_lazy_steps_under_a_name_on_another_library(self, spool, strict):
        """A step chained on a lazy patch is asked about under the backend's name."""

        class Named(_Strict):
            """The strict library's arrays, with a kernel of its own."""

            name = "strict_lazy_named"

        @register_kernel(_Tripled, "strict_lazy_named")
        def _tripled_there(self, data):
            """Triple the data where they are."""
            return data * 3

        lazy = spool.set_backend("lazy[strict_lazy_named]")[0]
        # Under the default setting, which refuses a step with no kernel for
        # the library: the backend has one, under its name.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            out = _Tripled()(lazy).load()
        np.testing.assert_array_equal(to_numpy(out.data), spool[0].data * 3)

    def test_operation_with_no_contract_runs_its_kernel(self, spool, fast):
        """Read into memory for an operation, lazy data run the backend's kernel."""

        class Whole(dc.PatchProcessor):
            """An operation which states no contract, so reads lazy data whole."""

            name = None

            def numpy_kernel(self, data):
                """Return the data doubled."""
                return data * 2

        @register_kernel(Whole, "fast_numpy")
        def _whole_fast(self, data):
            """Return the data tripled, so the backend's kernel shows."""
            return data * 3

        lazy = spool.set_backend("lazy[fast_numpy]")[0]
        with config_context(numpy_fallback="ignore"):
            out = Whole()(lazy)
        np.testing.assert_array_equal(out.data, fast.data * 3)

    def test_patch_reconcile_makes_is_of_it(self, fast):
        """A patch an operation's `reconcile` makes is of its input's backend."""

        class Rebuilt(dc.PatchProcessor):
            """An operation whose reconcile hands back a patch of its own making."""

            name = None

            def numpy_kernel(self, data):
                """Return the data doubled."""
                return data * 2

            def reconcile(self, data, out, meta):
                """Return a new patch, made without the one which came in."""
                return dc.Patch(data=data, coords=out.coords, attrs=out.attrs)

        out = Rebuilt()(fast)
        assert out.backend == _Fast() and _triples(out)

    def test_first_backend_wins_whichever_way_they_are_put_together(self, spool):
        """Merged, concatenated or stacked, patches are of the first one's backend."""
        plain = [x.set_backend("numpy") for x in spool]
        fast = [x.set_backend("fast_numpy") for x in spool]
        for first, rest, expected in (
            (fast, plain, _Fast()),
            (plain, fast, NumpyBackend()),
        ):
            held = dc.spool([first[0], *rest[1:]])
            assert held.chunk(time=None)[0].backend == expected
            assert held.concatenate(time=None)[0].backend == expected
            stacked = dc.spool([first[0], rest[0].new(data=rest[0].data * 2)]).stack(
                dim_vary="time"
            )
            np.testing.assert_array_equal(stacked.data, first[0].data * 3)
            assert stacked.backend == expected

    def test_kinds_are_told_before_a_fallback(self, spool, strict):
        """Two patches of other libraries are refused, not met in NumPy."""
        dask_array = pytest.importorskip("dask.array")
        callbacks = pytest.importorskip("dask.callbacks")
        plain = spool[0]
        there = plain.new(data=strict.asarray(plain.data))
        waiting = plain.new(data=dask_array.from_array(plain.data, chunks=-1))
        computed = []
        with callbacks.Callback(posttask=lambda *x: computed.append(1)):
            with pytest.raises(
                IncompatiblePatchError, match="arrays of one array library"
            ):
                there + waiting
        assert not computed
        # Nor a patch of a backend with another library's of none.
        with pytest.raises(IncompatiblePatchError, match="arrays of one array library"):
            plain.set_backend("fast_numpy") + there

    def test_spool_load_keeps_it(self, spool):
        """Loaded by the spool, lazily or not, the patches are of the namespace."""
        for name in ("fast_numpy", "lazy[fast_numpy]"):
            assert all(_triples(x) for x in spool.set_backend(name).load())
        # And held in memory under a lazy backend, of the one it computes on.
        assert _triples(dc.spool([spool[0]]).set_backend("lazy[fast_numpy]")[0])


class TestFallbackSetting:
    """One setting decides a conversion through NumPy."""

    def test_in_memory_operation(self, patch, strict):
        """An operation with no kernel for the library obeys the setting."""
        there = _on(patch, STRICT)
        expected = patch.detrend("time")
        with pytest.warns(NumpyFallbackWarning, match="detrend has no"):
            out = there.detrend("time")
        assert backend_name(out.data) == STRICT
        np.testing.assert_allclose(to_numpy(out.data), expected.data)
        with config_context(numpy_fallback="raise"):
            with pytest.raises(NumpyFallbackError, match="refuses to convert"):
                there.detrend("time")
        with config_context(numpy_fallback="ignore"), warnings.catch_warnings():
            warnings.simplefilter("error")
            quiet = there.detrend("time")
        assert backend_name(quiet.data) == STRICT


class TestPatchOnABackend:
    """A spool set to a backend hands out the same patch, on that backend."""

    def test_lazy_reads_nothing(self, patch, reads):
        """A patch from a file is handed out as a promise to read it."""
        lazy = _lazy(patch)
        assert backend_name(lazy.data) == "lazy" and lazy.shape == patch.shape
        assert lazy.attrs == patch.attrs and lazy.coords == patch.coords
        assert not reads
        loaded = lazy.load()
        assert len(reads) == 1 and loaded.equals(patch)
        assert loaded.attrs.data_id == patch.attrs.data_id

    def test_nothing_to_do(self, patch):
        """A patch already on the backend is handed out as it is."""
        there = _on(patch, "numpy")
        # Its array is NumPy's already, so only whose the patch is changes.
        assert there.data is patch.data and there.backend == NumpyBackend()
        assert _on(there, "numpy") is there
        lazy = _lazy(patch, "LazyArray")
        assert _on(lazy, "lazy[numpy]") is lazy

    @pytest.mark.parametrize("name", ("numpy", STRICT, "lazy", f"lazy[{STRICT}]"))
    def test_set_on_the_patch_is_what_a_spool_hands_out(self, patch, strict, name):
        """A patch set to a backend is the one a spool set to it hands out."""
        for held in (patch, _lazy(patch), _lazy(patch).abs()):
            mine, theirs = held.set_backend(name), _on(held, name)
            assert mine.backend == theirs.backend
            assert type(mine.data) is type(theirs.data)
            assert mine.load().equals(theirs.load())

    def test_backend_of_a_patch(self, patch, strict):
        """A patch says which backend it is of; None, if it was given none."""
        assert patch.backend is None and patch.abs().backend is None
        there = patch.set_backend(_Strict())
        assert there.backend == _Strict() == there.abs().backend
        # A lazy one is of the lazy backend, and of what it computes on once
        # it is loaded; in memory already, it has nothing to read later.
        lazy = _lazy(patch, f"lazy[{STRICT}]")
        assert lazy.backend == LazyBackend(compute=_Strict()) == lazy.abs().backend
        assert (
            lazy.load().backend == _Strict() == patch.set_backend(lazy.backend).backend
        )
        assert lazy.set_backend("numpy").backend == NumpyBackend()
        # None leaves the data as they are, of no backend.
        loose = there.set_backend(None)
        assert loose.backend is None and loose.data is there.data
        with pytest.raises(AttributeError):
            there.backend = NumpyBackend()

    def test_data_given_to_a_patch_are_made_its_backend_s(self, patch, strict):
        """A patch of a backend holds that backend's arrays, whatever it is given."""
        there = patch.set_backend(STRICT)
        for made in (
            there.new(data=patch.data),
            there.update(data=patch.data * 2),
            there.drop_data().to_patch(patch.data),
        ):
            assert backend_name(made.data) == STRICT and made.backend == _Strict()
        # One of no backend holds what it is given.
        assert backend_name(patch.new(data=there.data).data) == STRICT

    def test_backend_which_cannot_be_used(self, patch):
        """A patch is not set to a backend which is not one, or has no library."""
        with pytest.raises(ParameterError, match="is not a backend"):
            patch.set_backend("not_a_backend")
        with pytest.raises(ParameterError, match="No array library called"):
            patch.set_backend(MISSING)

        class Misspelled(NumpyBackend):
            """NumPy's arrays, told by a class which does not exist."""

            name = "misspelled_on_a_patch"
            array_type = "numpy.ndaray"

        with pytest.raises(ParameterError, match="not an instance of its array"):
            patch.set_backend("misspelled_on_a_patch")

    def test_dtype_a_library_does_not_keep(self, monkeypatch):
        """A dtype the library holds as another is refused, with no samples too."""
        strict = pytest.importorskip("array_api_strict")

        def narrowing(name):
            """Return a namespace which keeps 64-bit numbers in 32."""
            return SimpleNamespace(
                asarray=lambda x: strict.asarray(np.asarray(x, np.float32))
            )

        monkeypatch.setattr(backends_module, "import_module", narrowing)
        # It can be set to, since what it is first handed it keeps.
        _Strict()._check()
        with pytest.raises(ParameterError, match="holds float64 data as float32"):
            _Strict().from_array(np.empty(0, np.float64))

    def test_relative_path_is_pinned(self, patch, tmp_path, monkeypatch):
        """A lazy patch reads the file it was named beside, wherever it is read."""
        dc.write(patch, tmp_path / "here.h5", "dasdae")
        monkeypatch.chdir(tmp_path)
        with reading_onto(LazyBackend()):
            read = dc.read("here.h5")[0]
        handed = dc.spool("here.h5").set_backend("lazy")[0]
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        for lazy in (read, handed):
            assert backend_name(lazy.data) == "lazy" and lazy.load().equals(patch)

    def test_array_numpy_handles_is_left_alone(self, patch, strict):
        """An array-like of no array library is not made an ndarray of."""
        held = patch.new(data=_Wrapped(np.asarray(patch.data)))
        assert type(held.data) is _Wrapped
        # Reader's choice, NumPy and lazy on NumPy all leave it as it is.
        assert dc.spool([held])[0] is held
        assert _on(held, "numpy") is held and _on(held, "lazy").data is held.data
        # Another library was asked for, so its values are what is wanted.
        there = _on(held, STRICT)
        assert backend_name(there.data) == STRICT
        assert np.array_equal(to_numpy(there.data), patch.data)

    def test_big_endian_data(self, tmp_path):
        """Data stored big-endian read lazily as they read into memory (#1644)."""
        data = np.arange(12.0).reshape(3, 4).astype(">f8")
        coords = {"distance": np.arange(3.0), "time": np.arange(4.0)}
        path = tmp_path / "big_endian.h5"
        dc.write(dc.Patch(data=data, coords=coords, dims=tuple(coords)), path, "dasdae")
        eager = dc.read(path)[0]
        with reading_onto(LazyBackend()):
            read = dc.read(path)[0]
        handed = dc.spool(path).set_backend("lazy")[0]
        for lazy in (read, handed):
            assert lazy.dtype == eager.dtype == np.dtype(">f8")
            assert lazy.attrs.data_id == eager.attrs.data_id
            assert lazy.load().equals(eager)

    def test_library_which_narrows_the_data(self, patch, monkeypatch):
        """A library which would hold the values in fewer bits is refused."""
        strict = pytest.importorskip("array_api_strict")

        def narrowing(name):
            """Return a namespace which keeps 64-bit numbers in 32."""
            return SimpleNamespace(
                asarray=lambda x: strict.asarray(np.asarray(x, np.float32))
            )

        monkeypatch.setattr(backends_module, "import_module", narrowing)
        with pytest.raises(ParameterError, match="holds float64 data as float32"):
            _on(patch, STRICT)

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
        assert backend_name(part.data) == "lazy" and len(part.data.chain)
        assert part.load().dtype == dtype

    def test_structured_data_to_another_library(self):
        """A record dtype survives the check that a library kept the dtype."""
        dask_array = pytest.importorskip("dask.array")
        dtype = np.dtype([("x", "<i4"), ("y", "<f8")])
        coords = {"x": np.arange(6), "y": np.arange(4)}
        patch = dc.Patch(data=np.zeros((6, 4), dtype), coords=coords, dims=("x", "y"))
        there = _on(patch, "dask")
        assert isinstance(there.data, dask_array.Array)
        assert there.dtype == dtype and _on(there, "numpy").dtype == dtype

    def test_as_a_numpy_array(self, patch, strict):
        """`np.asarray` of a lazy array on another library is a NumPy array."""
        lazy = _lazy(patch, f"LazyArray[{STRICT}]")
        out = np.asarray(lazy.data)
        assert type(out) is np.ndarray and np.array_equal(out, patch.data)

    def test_an_empty_box(self, patch, strict):
        """A box of no samples is on the library the others are."""
        lazy = _lazy(patch, f"LazyArray[{STRICT}]").abs()
        empty = execute.compute(lazy.data, (slice(0, 0), slice(0, 5)))
        assert empty.shape == (0, 5)
        assert type(empty) is type(strict.asarray(np.zeros(1)))

    def test_another_library(self, patch, strict):
        """An array library by name holds the data in memory."""
        there = _on(patch, STRICT)
        assert backend_name(there.data) == STRICT and there.attrs == patch.attrs
        assert np.array_equal(to_numpy(there.data), patch.data)
        back = _on(there, "numpy")
        assert backend_name(back.data) == "numpy" and back.equals(patch)

    def test_lazy_on_another_library(self, patch, strict, reads):
        """Chunks are computed on the library in the brackets."""
        lazy = _lazy(patch, f"LazyArray[{STRICT}]")
        assert backend_name(lazy.data) == "lazy" and lazy.data._backend == _Strict()
        out = lazy.abs()
        assert backend_name(out.data) == "lazy" and not reads
        loaded = out.load()
        assert backend_name(loaded.data) == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), np.abs(patch.data))

    def test_asked_for_on_numpy_is_computed_there(self, patch, strict, monkeypatch):
        """Moved back to NumPy, a lazy patch computes there, converting nothing."""
        out = _lazy(patch, f"LazyArray[{STRICT}]").abs()
        _refuse_numpy(monkeypatch)
        loaded = _on(out, "numpy")
        assert backend_name(loaded.data) == "numpy"
        np.testing.assert_array_equal(loaded.data, np.abs(patch.data))

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

        lazy = _lazy(patch, f"LazyArray[{STRICT}]")
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
        assert backend_name(loaded.data) == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)

    def test_boxes_joined_along_both_axes(self, patch, strict):
        """Boxes cut along more than one axis go back where they came from."""
        lazy = _lazy(patch, f"LazyArray[{STRICT}]").abs()
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
        lazy = _lazy(patch, f"LazyArray[{STRICT}]")
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
        assert backend_name(loaded.data) == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)


class TestPendingStepsOnAnotherLibrary:
    """Steps already pending are asked for a kernel on the new library."""

    @pytest.mark.parametrize("target", (f"LazyArray[{STRICT}]", STRICT))
    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_step_with_no_kernel(self, patch, strict, policy, target):
        """A step chained on NumPy is not converted on another library unsaid."""
        pending = _Doubles()(_lazy(patch))
        with config_context(numpy_fallback=policy), warnings.catch_warnings():
            warnings.simplefilter("error")
            if policy is None:
                with pytest.raises(LazyOperationError, match="refuses to convert"):
                    _on(pending, target)
                return
            match = f"_Doubles has no {STRICT} implementation"
            with pytest.warns(NumpyFallbackWarning, match=match):
                moved = _on(pending, target)
            loaded = moved.load()
        assert backend_name(loaded.data) == STRICT
        np.testing.assert_array_equal(to_numpy(loaded.data), patch.data * 2)

    def test_every_pending_step_is_asked(self, patch, strict):
        """Not the first alone: a later step with no kernel is found too."""
        pending = _Doubles()(_lazy(patch).abs())
        with pytest.raises(LazyOperationError, match="_Doubles has no"):
            _on(pending, f"LazyArray[{STRICT}]")

    def test_conversion_a_kernel_makes_itself(self, patch):
        """A kernel which converts on its own is heard when it is computed."""
        dask_array = pytest.importorskip("dask.array")
        # Normalize with a window converts dask data through NumPy itself,
        # which nothing could ask about when it was chained in a window.
        lazy = _lazy(patch, "LazyArray[dask]")
        window = lazy.window(time=100, samples=True)
        out = window.normalize("time", window=11, samples=True)
        assert dask_array is not None and backend_name(out.data) == "lazy"
        with pytest.warns(NumpyFallbackWarning, match="normalize has no dask"):
            out.load()
        with config_context(numpy_fallback="raise"):
            with pytest.raises(NumpyFallbackError, match="normalize has no dask"):
                out.load()

    @pytest.mark.parametrize("policy", (None, "warn"))
    def test_operation_in_a_window(self, patch, strict, policy):
        """A window is asked about the operation it runs, when it is chained."""
        lazy = _lazy(patch, f"LazyArray[{STRICT}]")
        window = lazy.window(time=100, samples=True)
        with config_context(numpy_fallback=policy):
            if policy is None:
                with pytest.raises(LazyOperationError, match="detrend has no"):
                    window.detrend("time")
                return
            with pytest.warns(NumpyFallbackWarning, match="detrend has no"):
                out = window.detrend("time")
        assert backend_name(out.data) == "lazy"
        # Said once, when it was chained: not again for each box, though
        # the setting refuses by the time the boxes are computed.
        block = patch.data.nbytes // 6
        settings = {"lazy_block_size": block, "numpy_fallback": "raise"}
        with config_context(**settings), warnings.catch_warnings():
            warnings.simplefilter("error")
            assert len(chunk_boxes(out.data)) > 1
            loaded = out.load()
        expected = patch.window(time=100, samples=True).detrend("time")
        assert backend_name(loaded.data) == STRICT
        np.testing.assert_allclose(to_numpy(loaded.data), expected.data, atol=1e-9)
        # One with a kernel for the library is chained with nothing said.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert (
                backend_name(lazy.window(time=100, samples=True).abs().data) == "lazy"
            )

    def test_steps_with_kernels(self, patch, strict):
        """Steps which run on any library move with nothing said."""
        pending = _lazy(patch).abs()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            moved = _on(pending, f"LazyArray[{STRICT}]")
            # And back to NumPy, where every step has a kernel.
            assert _on(moved, "lazy").load().equals(patch.abs())
        assert backend_name(moved.load().data) == STRICT


class TestOperationWithNoContract:
    """An operation which cannot run a chunk at a time obeys the setting."""

    @pytest.mark.parametrize(
        "call",
        (lambda x: x + 1, lambda x: np.add(x, 1), lambda x: dc.spool([x, x]).stack()),
        ids=("arithmetic", "numpy_function", "stack"),
    )
    def test_refused_whatever_the_setting(self, patch, call, reads):
        """What takes the values themselves never reads them unasked."""
        lazy = _lazy(patch)
        with config_context(numpy_fallback="ignore"):
            with pytest.raises(LazyOperationError):
                call(lazy)
        assert not reads

    def test_on_another_library(self, patch, strict):
        """Read onto the library, an operation with no kernel for it says so too."""
        lazy = _lazy(patch, f"LazyArray[{STRICT}]")
        with config_context(numpy_fallback="warn"):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out = lazy.detrend("time")
        said = [str(x.message) for x in caught if x.category is NumpyFallbackWarning]
        # Once that the data were read, and once that they were converted.
        assert len(said) == 2 and "read into memory" in said[0]
        assert f"detrend has no {STRICT} implementation" in said[1]
        assert backend_name(out.data) == STRICT
        np.testing.assert_allclose(
            to_numpy(out.data), patch.detrend("time").data, atol=1e-9
        )

    def test_nothing_to_do_is_still_nothing(self, patch):
        """An operation which changes nothing leaves the lazy patch as it was."""
        lazy = _lazy(patch)
        assert patch.fill_gaps(time=None) is patch
        with config_context(numpy_fallback="ignore"):
            out = lazy.fill_gaps(time=None)
        # Lazy still, with no step recorded and its own id.
        assert backend_name(out.data) == "lazy" and out.attrs == lazy.attrs

    def test_read_into_memory(self, patch, reads):
        """Allowed, the data are read and the operation run in memory."""
        lazy = _lazy(patch)
        with config_context(numpy_fallback="warn"):
            with pytest.warns(NumpyFallbackWarning, match="read into memory"):
                out = lazy.detrend("time")
        assert backend_name(out.data) == "numpy" and len(reads) == 1
        np.testing.assert_allclose(out.data, patch.detrend("time").data)


class TestLazyRead:
    """Inside `reading_onto` a lazy backend, a read returns a promise to read."""

    def test_reads_nothing(self, path, patch, reads):
        """The patch names its stored array and holds none of it."""
        with reading_onto(LazyBackend()):
            assert read_backend() == LazyBackend()
            lazy = dc.read(path)[0]
        assert read_backend() is None
        assert backend_name(lazy.data) == "lazy" and not reads
        assert lazy.attrs == patch.attrs and lazy.data._backend == NumpyBackend()
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
        assert backend_name(out.data) == "numpy" and out.equals(expected)

    def test_window_of_the_file(self, path, patch, reads):
        """A selection of neighbouring samples is a window to read later."""
        with reading_onto(LazyBackend()):
            lazy = dc.read(path, time=(10, 60), samples=True)[0]
        expected = dc.read(path, time=(10, 60), samples=True)[0]
        reads.clear()
        assert backend_name(lazy.data) == "lazy" and lazy.attrs == expected.attrs
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
            with reading_onto(LazyBackend()):
                if policy is None:
                    with pytest.raises(LazyOperationError, match="not next to each"):
                        dc.read(path, time=wanted, samples=True)
                    return
                with pytest.warns(NumpyFallbackWarning, match="read into memory"):
                    out = dc.read(path, time=wanted, samples=True)[0]
        # In memory; the spool which asked puts it where its chunks compute.
        assert backend_name(out.data) == "numpy" and out.equals(expected)


class TestSpoolSetBackend:
    """A spool hands its patches out on the backend it was set to."""

    def test_nothing_is_read_until_asked(self, spool, reads):
        """Indexing and iterating hand out promises."""
        lazy = spool.set_backend("lazy")
        assert not reads and len(lazy) == len(spool)
        first = lazy[0]
        every = list(lazy)
        assert not reads
        assert backend_name(first.data) == "lazy" and {
            backend_name(x.data) for x in every
        } == {"lazy"}
        for made, expected in zip(every, spool, strict=True):
            assert made.attrs == expected.attrs and made.load().equals(expected)

    def test_the_callers_reads_are_their_own(self, spool, path):
        """A read made while a lazy spool is iterated fills memory as usual."""
        for patch in spool.set_backend("lazy"):
            assert backend_name(patch.data) == "lazy"
            assert backend_name(dc.read(path)[0].data) == "numpy"
            assert read_backend() is None

    @pytest.mark.skipif(
        sys.platform == "emscripten", reason="emscripten has no threads"
    )
    def test_prefetching_iterator(self, spool):
        """The iterator which loads ahead hands out the same."""
        backends = [backend_name(x.data) for x in spool.set_backend("lazy").iterate()]
        assert backends == ["lazy"] * len(spool)

    def test_follows_the_spool(self, spool):
        """A spool made from one keeps its backend."""
        lazy = spool.set_backend("lazy")
        assert backend_name(lazy.sort("time")[0].data) == "lazy"
        assert backend_name(lazy[1:3][0].data) == "lazy"
        narrowed = lazy.select(distance=(0, 50), samples=True)
        assert backend_name(narrowed[0].data) == "lazy"
        expected = spool.select(distance=(0, 50), samples=True)[0]
        assert narrowed[0].load().equals(expected)
        # And is set again without touching the first.
        assert backend_name(lazy.set_backend("numpy")[0].data) == "numpy"
        assert (
            backend_name(lazy[0].data) == "lazy"
            and backend_name(spool[0].data) == "numpy"
        )

    def test_added_spools_keep_their_backend(self, spool, strict, reads):
        """A sum of spools on one backend is on it; of two, it is refused."""
        lazy = spool.set_backend("lazy")
        both = lazy[:2] + lazy[2:]
        assert len(both) == len(spool)
        assert {backend_name(x.data) for x in both} == {"lazy"} and not reads
        # Backends which make one kind of array add, on the first one's.
        assert lazy[:2] + spool[2:].set_backend(LazyBackend()) == both
        assert (lazy[:2] + spool[2:].set_backend("numpy")).backend == lazy.backend
        assert (spool[:2].set_backend("numpy") + lazy[2:]).backend == NumpyBackend()
        # Those which make different kinds do not.
        with pytest.raises(ParameterError, match="arrays of different array libraries"):
            lazy[:2] + spool[2:].set_backend(LazyBackend(compute=_Strict()))
        # A spool with none set takes the other's, so spools gathered onto
        # an empty one keep theirs.
        gathered = sum([lazy[:2], spool[2:]], dc.spool([]))
        assert len(gathered) == len(spool)
        assert {backend_name(x.data) for x in gathered} == {"lazy"} and not reads
        # And a spool is not the same spool on another backend.
        assert lazy != spool and lazy == spool.set_backend("LazyArray")
        # And None hands the patches out as their readers return them again.
        assert lazy.set_backend(None) == spool
        assert spool.backend is None and lazy.backend == LazyBackend()
        assert (
            lazy[:2].backend == lazy.select(time=(None, None)).backend == lazy.backend
        )

    def test_update_of_one_file_keeps_it(self, path):
        """A spool of one file, looked at again, is on the same backend."""
        lazy = dc.spool(path).set_backend("lazy")
        assert backend_name(lazy.update()[0].data) == "lazy"

    def test_load_reads_everything(self, spool, strict):
        """`load` gives a spool of patches in memory."""
        loaded = spool.set_backend("lazy").load()
        assert [backend_name(x.data) for x in loaded] == ["numpy"] * len(spool)
        assert all(x.equals(y) for x, y in zip(loaded, spool, strict=True))
        # On the library a lazy backend computes on.
        there = spool.set_backend(f"LazyArray[{STRICT}]").load()
        assert [backend_name(x.data) for x in there] == [STRICT] * len(spool)

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
        assert backend_name(there.data) == STRICT
        assert there._source == spool[0]._source
        lazy = spool.set_backend(f"lazy[{STRICT}]")[0]
        assert (
            backend_name(lazy.data) == "lazy"
            and backend_name(lazy.load().data) == STRICT
        )
        np.testing.assert_array_equal(to_numpy(lazy.load().data), spool[0].data)

    def test_refused_when_set(self, spool):
        """A name which is no backend, or no library, is refused at once."""
        with pytest.raises(ParameterError, match="is not a backend"):
            spool.set_backend("lazy[")
        for name in (MISSING, f"LazyArray[{MISSING}]"):
            with pytest.raises(ParameterError, match="No array library called"):
                spool.set_backend(name)

    def test_patches_held_in_memory(self, strict):
        """A patch with nothing to read later is put where chunks compute."""
        held = dc.get_example_spool("random_das")
        there = held.set_backend("lazy")[0]
        assert there.data is held[0].data and there.backend == NumpyBackend()
        assert backend_name(held.set_backend(f"lazy[{STRICT}]")[0].data) == STRICT

    def test_patches_held_in_memory_merge(self):
        """Merging what is in memory reads nothing, so nothing is refused."""
        held = dc.get_example_spool("random_das")
        expected = held.chunk(time=None)[0]
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            merged = held.set_backend("lazy").chunk(time=None)[0]
        assert backend_name(merged.data) == "numpy" and merged.equals(expected)

    @pytest.mark.wasm_skip
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
        assert backend_name(out.data) == "numpy" and len(reads) == len(spool)
        np.testing.assert_array_equal(out.data, expected.data)

    def test_lazy_patches_concatenated(self, spool, reads):
        """Lazy patches joined end to end stay lazy, reading nothing."""
        expected = spool.concatenate(time=None)[0]
        reads.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            out = spool.set_backend("lazy").concatenate(time=None)[0]
        assert backend_name(out.data) == "lazy" and not reads
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
        assert backend_name(out.data) == "numpy"
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
        assert backend_name(out.data) == STRICT
        monkeypatch.undo()
        np.testing.assert_array_equal(to_numpy(out.data), expected.data)

    @pytest.mark.parametrize("call", ("chunk", "concatenate"))
    def test_patches_of_another_library_are_not_joined(self, spool, strict, call):
        """Patches of different array libraries are refused, not put on one."""
        held = dc.spool([_on(spool[0], STRICT), *list(spool)[1:]])
        with pytest.raises(IncompatiblePatchError, match="arrays of one array library"):
            getattr(held, call)(time=None)[0]

    def test_two_other_libraries_are_not_joined(self, spool, strict):
        """Nor patches on two libraries other than NumPy, and nothing is read."""
        dask_array = pytest.importorskip("dask.array")
        callbacks = pytest.importorskip("dask.callbacks")
        first = spool[0]
        on_dask = first.new(data=dask_array.from_array(first.data, chunks=-1))
        held = dc.spool([on_dask, *[_on(x, STRICT) for x in list(spool)[1:]]])
        computed = []
        with callbacks.Callback(posttask=lambda *x: computed.append(1)):
            with pytest.raises(IncompatiblePatchError, match="dask"):
                held.concatenate(time=None)[0]
        assert not computed

    def test_fill_onto_a_grid_on_another_library_is_said(self, whole, tmp_path):
        """Placing members of another library onto a grid converts them, said."""
        pytest.importorskip("dask.array")
        size = whole.shape[whole.get_axis("time")]
        parts = [
            whole.select(time=(0, size // 4), samples=True),
            whole.select(time=(size // 2, size), samples=True),
        ]
        held = dc.spool([_on(x, "dask") for x in parts])
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
        assert backend_name(out.data) == STRICT and out.shape == expected.shape
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
        held = dc.spool([_on(x, STRICT) for x in parts])
        out = getattr(held, call)(time=None)[0]
        assert backend_name(out.data) == STRICT
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
        assert backend_name(out.data) == "lazy" and not reads
        assert out.attrs == expected.attrs and out.load().equals(expected)

    def test_private_dimension_is_not_dropped_lazily(self, patch):
        """Dropping a private dimension is refused, as it was."""
        lazy = _lazy(patch)
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
        with pytest.warns(NumpyFallbackWarning, match="cannot be joined as they"):
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
        assert backend_name(out.data) == "dask"
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
        assert backend_name(lazy.data) == "lazy" and not reads
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
        assert backend_name(part.data) == "lazy" and not reads
        loaded = part.load()
        assert len(reads) == 1 and "part_1" in reads[0].path
        expected = spool.chunk(time=None)[0].select(time=window, samples=True)
        assert loaded.equals(expected)

    def test_operations_wait(self, spool, reads):
        """What is done to the merged patch is done when it is loaded."""
        lazy = spool.set_backend("lazy").chunk(time=None)[0]
        out = lazy.abs()
        assert backend_name(out.data) == "lazy" and not reads
        assert out.load().equals(spool.chunk(time=None)[0].abs())

    def test_members_the_index_cannot_describe(self, undescribed, reads):
        """A merge of members read as patches is lazy still, joined by recipe."""
        expected = undescribed.chunk(time=None)[0]
        reads.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            merged = undescribed.set_backend("lazy").chunk(time=None)[0]
        assert backend_name(merged.data) == "lazy" and not reads
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

    @pytest.mark.wasm_skip
    def test_files_which_state_data_units(self, whole, tmp_path):
        """Members already in the plan's units are not asked to convert."""
        spool = _in_four_files(whole.set_units("m/s"), tmp_path)
        # Shorter than a file, so each patch is a part of one member, and
        # all of them, so one is the merge.
        for size in (0.3, None):
            eager = list(spool.chunk(time=size))
            lazy = list(spool.set_backend("lazy").chunk(time=size))
            assert len(lazy) == len(eager) and {backend_name(x.data) for x in lazy} == {
                "lazy"
            }
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
        assert backend_name(filled.data) == "numpy" and filled.attrs == expected.attrs
        assert filled.coords == expected.coords and np.isnan(filled.data).any()
        np.testing.assert_array_equal(filled.data, expected.data)

    def test_no_gap_to_fill(self, spool, reads):
        """With no gap there is nothing to write, so nothing is read or refused."""
        expected = spool.fill_gaps(time=None)[0]
        reads.clear()
        filled = spool.set_backend("lazy").fill_gaps(time=None)[0]
        assert backend_name(filled.data) == "lazy" and not reads
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
        assert backend_name(indexed.data) == "lazy" and indexed.dtype == expected.dtype
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
        held = dc.spool([_on(x, "dask") for x in parts])
        with warnings.catch_warnings():
            warnings.simplefilter("error", NumpyFallbackWarning)
            merged = held.chunk(time=None)[0]
        assert backend_name(merged.data) == "dask"
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
        merged = dc.spool([_on(x, "dask") for x in parts]).chunk(time=None)[0]
        assert backend_name(merged.data) == "dask"
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
        assert backend_name(merged.data) == "numpy"
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
        assert backend_name(merged.data) == "lazy" and not reads
        np.testing.assert_array_equal(merged.load().data, expected.data)

    def test_merge_of_unknown_size(self, spool, monkeypatch, reads):
        """Members merged as loaded patches stay lazy, joined by recipe."""
        expected = spool.chunk(time=None)[0]
        # As for members whose merged length the index's rows do not state.
        monkeypatch.setattr(assembly, "_estimate_merge_samples", lambda *args: None)
        reads.clear()
        merged = spool.set_backend("lazy").chunk(time=None)[0]
        assert backend_name(merged.data) == "lazy" and not reads
        assert merged.attrs == expected.attrs
        np.testing.assert_array_equal(merged.load().data, expected.data)
