"""
dascore's classes, and registered stand-ins, run the tests an implementation
of an interface runs.
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.core.coordmanager import get_coord_manager
from dascore.core.coords import get_coord
from dascore.testing import (
    BaseCoordManagerTests,
    BaseLazyArrayTests,
    BaseNumericCoordTests,
    BaseSpoolTests,
    BaseStringCoordTests,
)
from dascore.testing.interfaces import _check
from tests._stand_ins import (
    _LazyArray,
    _Manager,
    _NumericCoord,
    _Spool,
    _StringCoord,
)

# The base tests run in other packages' suites, under a pytest which may
# refuse what this one warns of.
pytestmark = pytest.mark.filterwarnings(
    "error:Class-scoped fixtures? defined as instance methods?"
)


class TestNumericCoord(BaseNumericCoordTests):
    """dascore's numeric coordinates."""

    @pytest.fixture
    def make_coord(self):
        """Dascore's builder."""
        return get_coord


class TestStringCoord(BaseStringCoordTests):
    """dascore's coordinates of text."""

    @pytest.fixture
    def make_coord(self):
        """Dascore's builder."""
        return get_coord


class TestCoordManager(BaseCoordManagerTests):
    """dascore's coordinate managers."""

    @pytest.fixture
    def make_manager(self):
        """Dascore's builder."""
        return get_coord_manager


class TestLazyArray(BaseLazyArrayTests):
    """dascore's lazy arrays, as a lazy read of a file gives them."""

    @pytest.fixture
    def make_lazy_array(self, tmp_path):
        """Write the array to a file and read it lazily."""

        def make(array):
            coords = {f"dim_{i}": np.arange(x) for i, x in enumerate(array.shape)}
            patch = dc.Patch(data=array, coords=coords, dims=tuple(coords))
            path = dc.write(patch, tmp_path / "array.h5", "DASDAE")
            return dc.spool(path).set_backend("lazy")[0].data

        return make


class TestSpool(BaseSpoolTests):
    """dascore's spools."""

    @pytest.fixture
    def make_spool(self):
        """Dascore's builder."""
        return dc.spool


class TestRegisteredNumericCoord(BaseNumericCoordTests):
    """A registered stand-in for a numeric coordinate."""

    @pytest.fixture
    def make_coord(self):
        """Dascore's coordinate, wrapped."""
        return lambda **kwargs: _NumericCoord(get_coord(**kwargs))


class TestRegisteredStringCoord(BaseStringCoordTests):
    """A registered stand-in for a coordinate of text."""

    @pytest.fixture
    def make_coord(self):
        """Dascore's coordinate, wrapped."""
        return lambda **kwargs: _StringCoord(get_coord(**kwargs))


class TestRegisteredCoordManager(BaseCoordManagerTests):
    """A registered stand-in for a coordinate manager."""

    @pytest.fixture
    def make_manager(self):
        """Dascore's manager, wrapped."""
        return lambda *args, **kwargs: _Manager(get_coord_manager(*args, **kwargs))


class TestRegisteredLazyArray(BaseLazyArrayTests):
    """A registered stand-in for a lazy array, indexed to select."""

    @pytest.fixture
    def make_lazy_array(self):
        """The stand-in holding the array."""
        return _LazyArray


class TestRegisteredSpool(BaseSpoolTests):
    """A registered stand-in for a spool."""

    @pytest.fixture
    def make_spool(self):
        """Dascore's spool, wrapped."""
        return lambda source: _Spool(dc.spool(source))


class TestCheck:
    """The comparison the tests make, a pure helper, fails what differs."""

    @pytest.fixture(scope="class")
    @classmethod
    def coord(cls):
        """An evenly sampled coordinate."""
        return get_coord(start=0.0, stop=10.0, step=1.0, units="m")

    def test_another_id(self, coord):
        """A coordinate with other labels under the same name fails."""
        with pytest.raises(AssertionError, match="data_id"):
            _check(lambda x: x, coord.update(min=1.0), coord)

    def test_another_kind(self, coord):
        """A text coordinate where dascore gives a number fails."""
        text = get_coord(data=np.array(["a", "b"]))
        with pytest.raises(AssertionError, match="BaseNumericCoord"):
            _check(lambda x: x, text, coord)

    def test_no_error(self, coord):
        """Where dascore raises, giving a result fails."""
        text = get_coord(data=np.array(["a", "b"]))
        with pytest.raises(pytest.fail.Exception):
            _check(lambda x: x.set_units("m"), coord, text)

    @pytest.mark.parametrize(
        ("theirs", "ours"),
        [
            (np.float64(0.0), np.zeros(3)),
            (np.zeros(3, dtype=np.float32), np.zeros(3)),
            (np.nan, None),
            (np.datetime64("NaT"), np.nan),
            ("float64", np.dtype("float64")),
            ([1, 2], (1, 2)),
            (np.array([3]), np.int64(3)),
            (1, True),
        ],
    )
    def test_alike_values(self, theirs, ours):
        """A value of another type, kind, shape or dtype, equal or not, fails."""
        with pytest.raises(AssertionError):
            _check(lambda x: x, theirs, ours)

    def test_another_backend(self, random_patch):
        """A spool whose patches are on another backend fails."""
        spool = dc.spool([random_patch])
        with pytest.raises(AssertionError, match="backend"):
            _check(lambda x: x, spool, spool.set_backend("numpy"))

    def test_other_patches(self, random_patch):
        """A spool of other patches fails."""
        spool = dc.spool([random_patch])
        other = dc.spool([random_patch.update_attrs(tag="other")])
        with pytest.raises(AssertionError, match=r"\[0\].attrs\['tag'\]"):
            _check(lambda x: x, other, spool)
