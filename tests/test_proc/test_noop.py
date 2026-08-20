"""A patch function asked to do nothing hands the patch back."""

from __future__ import annotations

import numpy as np
import pytest

from dascore.units import get_quantity


@pytest.fixture(scope="module")
def unit_patch(random_patch):
    """A patch whose data and coordinates all carry units."""
    return random_patch.set_units("m/s", distance="m", time="s")


# (patch fixture, method, args, kwargs) for a call which cannot change anything.
NO_OPS = (
    ("unit_patch", "convert_units", (), {"distance": "m"}),
    # The same units spelled differently are still those units.
    ("unit_patch", "convert_units", (), {"distance": "meter"}),
    ("unit_patch", "convert_units", (), {"data_units": "m/s"}),
    # Not `unit_patch`: `set_units` clears data units it is not given,
    # which is a change and rightly not short circuited.
    ("random_patch", "set_units", (), {"distance": "m"}),
    ("unit_patch", "simplify_units", (), {}),
    ("random_patch_with_lat_lon", "drop_coords", ("not_a_coord",), {}),
    ("random_patch", "drop_private_coords", (), {}),
    ("random_patch", "dropna", ("time",), {}),
    ("random_patch", "fillna", (0,), {}),
    ("random_patch", "real", (), {}),
    ("random_patch", "conj", (), {}),
)


class TestNothingToDo:
    """The operations which can tell they have no work."""

    @pytest.mark.parametrize(("fixture", "method", "args", "kwargs"), NO_OPS)
    def test_returns_the_input(self, request, fixture, method, args, kwargs):
        """The patch itself comes back, so nothing is copied or recorded."""
        patch = request.getfixturevalue(fixture)
        assert getattr(patch, method)(*args, **kwargs) is patch


class TestTheGuardDoesNotOverFire:
    """Calls which look like no-ops and are not."""

    def test_equal_quantities_still_convert(self, unit_patch):
        """1 m and 100 cm are equal quantities but not the same units."""
        out = unit_patch.convert_units(distance="100 cm")
        assert out is not unit_patch
        assert get_quantity(out.get_coord("distance").units) == get_quantity("100 cm")

    def test_set_units_still_drops_data_units(self, unit_patch):
        """`set_units` clears the data units when it is not given any."""
        assert unit_patch.set_units(distance="m").attrs.data_units is None

    def test_object_array_is_still_conjugated(self, random_patch):
        """
        An object dtype says nothing about what it holds.

        The elements can be complex, and conjugating them is real work,
        so the dtype check must not claim there is none.
        """
        patch = random_patch.new(data=np.full(random_patch.shape, 1 + 2j, dtype=object))
        out = patch.conj()
        assert out is not patch
        assert out.data[0, 0] == 1 - 2j
