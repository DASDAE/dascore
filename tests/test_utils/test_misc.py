"""
Misc. tests for misfit utilities.
"""
import numpy as np

from dascore.utils.misc import MethodNameSpace, get_slice


class ParentClass:
    """A test parent class."""

    @property
    def namespace(self):
        """Your run-o-the-mill namespace"""
        return MNS(self)


class MNS(MethodNameSpace):
    """method name space subclass."""

    def func1(self, expected_type):
        """First func"""
        return isinstance(self, expected_type)


class TestNamespaceClass:
    """Tests for namespace class."""

    def test_parent_self_passed_to_namespace(self):
        """Ensure the parent of namespace gets passed to self."""
        pc = ParentClass()
        assert pc.namespace.func1(ParentClass)

    def test_assign_adhoc_method(self):
        """Ensure methods added after class definition still work."""

        def new_method(self, expected_type):
            return isinstance(self, expected_type)

        MNS.new_method = new_method
        pc = ParentClass()
        assert pc.namespace.new_method(ParentClass)


class TestGetSlice:
    """Ensure getting slices of arrays for indexing works."""

    ar = np.arange(100)

    def test_two_intervals(self):
        """test get slice for two intervals"""
        array_slice = get_slice(self.ar, cond=(1, 10))
        expected = slice(1, 11, None)
        assert array_slice == expected

    def test_right_side(self):
        """test for only right interval"""
        array_slice = get_slice(self.ar, cond=(None, 10))
        expected = slice(None, 11, None)
        assert array_slice == expected

    def test_left_side(self):
        """Ensure left side interval works."""
        array_slice = get_slice(self.ar, cond=(1, None))
        expected = slice(1, None, None)
        assert array_slice == expected

    def test_no_bounds(self):
        """Empty slice should be returned when no bounds specified."""
        array_slice = get_slice(self.ar, cond=(None, None))
        expected = slice(None, None, None)
        assert array_slice == expected

    def test_out_of_bounds(self):
        """When out of bounds, non should be returned."""
        array_slice = get_slice(self.ar, cond=(-100, 1_000))
        expected = slice(None, None, None)
        assert array_slice == expected

    def test_cond_is_none(self):
        """Ensure None is a valid input, returns empty slice"""
        array_slice = get_slice(self.ar, cond=None)
        expected = slice(None, None, None)
        assert array_slice == expected

    def test_slice_end_with_zeros(self):
        """
        Ensure we get a slice without None at the end if arrays are zeroed at end.
        """
        ar = np.arange(100)
        ar[-20:] = 0
        sliced = get_slice(ar, (None, ar.max()))
        assert sliced.stop is not None
        assert sliced.stop == ar.max()

    def test_slice_middle_with_zeros(self):
        """
        Ensure we get a slice without None at the end if arrays are zeroed at end.
        """
        ar = np.arange(100)
        ar[-20:] = 0
        sliced = get_slice(ar, (None, ar.max() - 10))
        assert sliced.stop is not None
        assert ar[sliced].max() == (ar.max() - 10)
