"""
Misc. tests for misfit utilities.
"""
import os
import time
from pathlib import Path

import numpy as np
import pytest

from dascore.exceptions import ParameterError
from dascore.utils.misc import (
    MethodNameSpace,
    check_evenly_sampled,
    get_slice,
    iter_files,
    iterate,
)


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


class TestCheckEvenlySampled:
    """Tests for uniform sampling check."""

    dt = np.datetime64("2021-01-01")
    td = np.timedelta64(1, "s")

    def test_check_uniform(self):
        """Ensure no exception is raised for uniform sampled arrays."""
        arrays = dict(
            a=[1, 2, 3],
            b=np.arange(0, 10),
            c=[self.dt + self.td, self.dt + 2 * self.td, self.dt + 3 * self.td],
            d=np.linspace(0, 1, 50),
        )
        for _, array in arrays.items():
            check_evenly_sampled(array)

    def test_check_non_uniform(self):
        """Ensure no exception is raised for uniform sampled arrays."""

        arrays = dict(
            a=[1, 2, 3, 5],
            c=[self.dt + self.td, self.dt + 4 * self.td, self.dt + 3 * self.td],
        )
        for _, array in arrays.items():
            with pytest.raises(ParameterError):
                check_evenly_sampled(array)


class TestIterFiles:
    """Tests for iterating directories of files."""

    sub = {"D": {"C": ".mseed"}, "F": ".json", "G": {"H": ".txt"}}
    file_paths = {"A": ".txt", "B": sub}

    # --- helper functions
    def setup_test_directory(self, some_dict: dict, path: Path):
        """Build the test directory."""
        for path in self.get_file_paths(some_dict, path):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as fi:
                fi.write("useful text")

    def get_file_paths(self, some_dict, path):
        """Return expected paths to files."""
        for i, v in some_dict.items():
            if isinstance(v, dict):
                yield from self.get_file_paths(v, path / i)
            else:
                yield path / (i + v)

    # --- fixtures
    @pytest.fixture(scope="class")
    def simple_dir(self, tmp_path_factory):
        """Return a simple directory for iterating."""
        path = Path(tmp_path_factory.mktemp("iterfiles"))
        self.setup_test_directory(self.file_paths, path)
        return path

    @pytest.fixture(scope="class")
    def dir_with_hidden_dir(self, tmp_path_factory):
        """Create a directory with a hidden directory inside."""
        path = Path(tmp_path_factory.mktemp("iterfiles_hidden"))
        struct = dict(self.file_paths)
        # add hidden directory with files in it.
        struct[".Hidden"] = {"Another": {"hidden_by_parent": ".txt"}}
        self.setup_test_directory(struct, path)
        return path

    def test_basic(self, simple_dir):
        """test basic usage of iterfiles."""
        files = set(self.get_file_paths(self.file_paths, simple_dir))
        out = set((Path(x) for x in iter_files(simple_dir)))
        assert files == out

    def test_one_subdir(self, simple_dir):
        """Test with one sub directory."""
        subdirs = simple_dir / "B" / "D"
        out = set(iter_files(subdirs))
        assert len(out) == 1

    def test_multiple_subdirs(self, simple_dir):
        """Test with multiple sub directories."""
        path1 = simple_dir / "B" / "D"
        path2 = simple_dir / "B" / "G"
        out = {Path(x) for x in iter_files([path1, path2])}
        files = self.get_file_paths(self.file_paths, simple_dir)
        expected = {
            x
            for x in files
            if str(x).startswith(str(path1)) or str(x).startswith(str(path2))
        }
        assert out == expected

    def test_extention(self, simple_dir):
        """Test filtering based on extention."""
        out = set(iter_files(simple_dir, ext=".txt"))
        for val in out:
            assert val.endswith(".txt")

    def test_mtime(self, simple_dir):
        """Test filtering based on modified time"""
        files = list(self.get_file_paths(self.file_paths, simple_dir))
        # set the first file mtime in future
        now = time.time()
        first_file = files[0]
        os.utime(first_file, (now + 10, now + 10))
        # get output make sure it only returned first file
        out = list(iter_files(simple_dir, mtime=now + 5))
        assert len(out) == 1
        assert Path(out[0]) == first_file

    def test_skips_files_in_hidden_directory(self, dir_with_hidden_dir):
        """Hidden directory files should be skipped."""
        out1 = list(iter_files(dir_with_hidden_dir))
        has_hidden_by_parent = ["hidden_by_parent" in x for x in out1]
        assert not any(has_hidden_by_parent)
        # But if skip_hidden is False it should be there
        out2 = list(iter_files(dir_with_hidden_dir, skip_hidden=False))
        has_hidden_by_parent = ["hidden_by_parent" in x for x in out2]
        assert sum(has_hidden_by_parent) == 1


class TestIterate:
    """Test case for iterate."""

    def test_none(self):
        """None should return an empty tuple"""
        assert iterate(None) == tuple()

    def test_object(self):
        """A single object should be returned in a tuple"""
        assert iterate(1) == (1,)

    def test_str(self):
        """A single string object should be returned as a tuple"""
        assert iterate("hey") == ("hey",)
