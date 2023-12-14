"""Misc. tests for misfit utilities."""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

from dascore.exceptions import MissingOptionalDependency
from dascore.utils.misc import (
    MethodNameSpace,
    cached_method,
    get_stencil_coefs,
    iter_files,
    iterate,
    maybe_get_items,
    optional_import,
    separate_coord_info,
)


class ParentClass:
    """A test parent class."""

    @property
    def namespace(self):
        """Your run-o-the-mill namespace."""
        return MNS(self)


class MNS(MethodNameSpace):
    """method name space subclass."""

    def func1(self, expected_type):
        """First func."""
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


class TestIterFiles:
    """Tests for iterating directories of files."""

    sub = {"D": {"C": ".mseed"}, "F": ".json", "G": {"H": ".txt"}}  # noqa
    file_paths = {"A": ".txt", "B": sub}  # noqa

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
        """Test basic usage of iterfiles."""
        files = set(self.get_file_paths(self.file_paths, simple_dir))
        out = {Path(x) for x in iter_files(simple_dir)}
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
        """Test filtering based on modified time."""
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

    def test_pass_file(self, dummy_text_file):
        """Just pass a single file and ensure it gets returned."""
        out = list(iter_files(dummy_text_file))
        assert len(out) == 1
        assert out[0] == dummy_text_file


class TestIterate:
    """Test case for iterate."""

    def test_none(self):
        """None should return an empty tuple."""
        assert iterate(None) == tuple()

    def test_object(self):
        """A single object should be returned in a tuple."""
        assert iterate(1) == (1,)

    def test_str(self):
        """A single string object should be returned as a tuple."""
        assert iterate("hey") == ("hey",)


class TestOptionalImport:
    """Ensure the optional import works."""

    def test_import_installed_module(self):
        """Test to ensure an installed module imports."""
        import dascore as dc

        mod = optional_import("dascore")
        assert mod is dc
        sub_mod = optional_import("dascore.core")
        assert sub_mod is dc.core

    def test_missing_module_raises(self):
        """Ensure a module which is missing raises the appropriate Error."""
        with pytest.raises(MissingOptionalDependency, match="boblib4"):
            optional_import("boblib4")


class TestGetStencilCoefficients:
    """Tests for stencil coefficients."""

    def test_3_point_1st_derivative(self):
        """3 point 1st derivative."""
        out = get_stencil_coefs(1, 1)
        expected = np.array([-1 / 2, 0, 1 / 2])
        assert np.allclose(out, expected)

    def test_5_point_1st_derivative(self):
        """5 point 1st derivative."""
        out = get_stencil_coefs(2, 1)
        expected = np.array([1, -8, 0, 8, -1]) / 12.0
        assert np.allclose(out, expected)

    def test_3_point_2nd_derivative(self):
        """3 point 2nd derivative."""
        out = get_stencil_coefs(1, 2)
        expected = np.array([1, -2, 1])
        assert np.allclose(out, expected)


class TestCachedMethod:
    """Ensure cached methods caches method calls (duh)."""

    class _JohnnyCached:
        """Tests class for caching."""

        @cached_method
        def no_args(self):
            """No argument method."""
            return {"output", "defined"}

        @cached_method
        def multiargs(self, a, b):
            """Multiple arguments for cache testing."""
            return a + b

    def test_no_args_kwargs(self):
        """Ensure objects cache without args or kwargs."""
        john = self._JohnnyCached()
        first = john.no_args()
        assert first is john.no_args()

    def test_positional(self):
        """Ensure positional arguments work."""
        john = self._JohnnyCached()
        out = john.multiargs(1, 2)
        assert john.multiargs(1.0, 2.0) == 3.0
        assert out == 3

    def test_kwargs(self):
        """Ensure keywords also work."""
        john = self._JohnnyCached()
        assert john.multiargs(1, b=1) == 2
        assert john.multiargs(a=2, b=3) == 5


class TestSeparateCoordInfo:
    """Tests for separating coord info from attr dict."""

    def test_empty(self):
        """Empty args should return emtpy dicts."""
        out1, out2 = separate_coord_info(None)
        assert out1 == out2 == {}

    def test_meets_reqs(self):
        """Simple case for filtering out required attrs."""
        input_dict = {"coords": {"time": {"min": 10}}}
        coords, attrs = separate_coord_info(input_dict)
        assert coords == input_dict["coords"]


class TestMaybeGetItems:
    """Tests for maybe_get_attrs."""

    def test_missed_itme(self):
        """Ensure it still works when a key is missing."""
        data = {"bob": 1, "bill": 2}
        expected = {"bob": "sue", "lary": "who"}
        out = maybe_get_items(data, attr_map=expected)
        assert "sue" in out
        assert "who" not in out
