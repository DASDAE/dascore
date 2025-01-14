"""
Tests for file system utilities.
"""

import os
import time
from pathlib import Path

import fsspec
import pytest
from upath import UPath

from dascore.utils.fs import get_ls_details, iter_path_contents
from dascore.utils.misc import register_func

UPATHS = []


@pytest.fixture(scope="class")
def complex_folder(tmp_path_factory):
    """Make a temp path with several sub folders and such."""
    path = tmp_path_factory.mktemp("complex_fs_folder")

    csv_1 = path / "csv_1.csv"
    with csv_1.open("w") as f:
        f.write("Name,Age,Occupation,City")
        f.write("Alice,30,Engineer,New York")

    text1 = path / "text_1.txt"
    with text1.open("w") as f:
        f.write("Ground control to major tom!")

    f1 = path / "folder_1"
    f1.mkdir(exist_ok=True, parents=True)
    text2 = f1 / "text_2.txt"
    with text2.open("w") as f:
        f.write("Planet Earth is blue and there's nothing I can do")

    f2 = path / "folder_2"
    f2.mkdir(exist_ok=True, parents=True)
    csv_2 = f2 / "csv_2.csv"
    with csv_2.open("w") as f:
        f.write("Name,Age,Occupation,City")
        f.write("Bob,35,Engineer,Salt Lake City")
    return path


@pytest.fixture(scope="class")
@register_func(UPATHS)
def upath_local(complex_folder):
    """Get a UPath object from a local tmpdir."""
    return UPath(complex_folder)


@pytest.fixture(scope="class")
@register_func(UPATHS)
def upath_github(request):
    """Get a UPath object from DASCore's github test data."""
    if not request.config.getoption("--network"):
        pytest.skip("Network tests not selected.")
    fs = fsspec.filesystem("github", repo="test_data", org="dasdae")
    return UPath(fs)


@pytest.fixture(scope="class", params=UPATHS)
def fspath(request):
    """Meta fixture for fspaths of different types."""
    name = request.param
    return request.getfixturevalue(name)


class TestIterContents:
    """Tests for iterating over the contents of directories."""

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
        return UPath(path)

    @pytest.fixture(scope="class")
    def dir_with_hidden_dir(self, tmp_path_factory):
        """Create a directory with a hidden directory inside."""
        path = Path(tmp_path_factory.mktemp("iterfiles_hidden"))
        struct = dict(self.file_paths)
        # add hidden directory with files in it.
        struct[".Hidden"] = {"Another": {"hidden_by_parent": ".txt"}}
        self.setup_test_directory(struct, path)
        return UPath(path)

    def test_basic(self, simple_dir):
        """Test basic usage of iterfiles."""
        files = set(self.get_file_paths(self.file_paths, simple_dir))
        out = {x for x in iter_path_contents(simple_dir)}
        assert files == out

    def test_one_subdir(self, simple_dir):
        """Test with one sub directory."""
        subdir = simple_dir / "B" / "D"
        fspath = UPath(subdir)
        out = set(iter_path_contents(fspath))
        assert len(out) == 1

    def test_multiple_subdirs(self, simple_dir):
        """Test with multiple sub directories."""
        path1 = UPath(simple_dir / "B" / "D")
        path2 = UPath(simple_dir / "B" / "G")

        out = set(iter_path_contents(path1)) | set(iter_path_contents(path2))
        files = self.get_file_paths(self.file_paths, simple_dir)
        expected = {
            x for x in files if str(path1.path) in str(x) or str(path2.path) in str(x)
        }
        assert out == expected

    def test_extension(self, simple_dir):
        """Test filtering based on extention."""
        out = set(iter_path_contents(simple_dir, ext=".txt"))
        for val in out:
            assert val.name.endswith(".txt")

    def test_mtime(self, simple_dir):
        """Test filtering based on modified time."""
        files = list(self.get_file_paths(self.file_paths, simple_dir))
        # set the first file mtime in future
        now = time.time()
        first_file = files[0]
        os.utime(first_file, (now + 10, now + 10))
        # get output make sure it only returned first file
        out = list(iter_path_contents(simple_dir, timestamp=now + 5))
        assert len(out) == 1
        assert out[0] == first_file

    def test_skips_files_in_hidden_directory(self, dir_with_hidden_dir):
        """Hidden directory files should be skipped."""
        out1 = list(iter_path_contents(dir_with_hidden_dir, skip_hidden=True))
        has_hidden_by_parent = ["hidden_by_parent" in str(x) for x in out1]
        assert not any(has_hidden_by_parent)
        # But if skip_hidden is False it should be there
        out2 = list(iter_path_contents(dir_with_hidden_dir, skip_hidden=False))
        has_hidden_by_parent = ["hidden_by_parent" in str(x) for x in out2]
        assert sum(has_hidden_by_parent) == 1

    def test_no_directories(self, simple_dir):
        """Ensure no directories are included when include_directories=False."""
        out = list(iter_path_contents(simple_dir, include_directories=False))
        has_dirs = [x.is_dir() for x in out]
        assert not any(has_dirs)

    def test_include_directories(self, simple_dir):
        """Ensure we can get directories back."""
        out = list(iter_path_contents(simple_dir, include_directories=True))
        returned_dirs = [x for x in out if x.is_dir()]
        assert len(returned_dirs)
        # Directory names
        dir_names = {x.name for x in returned_dirs}
        expected_names = {"B", "G", "D"}
        assert expected_names.issubset(dir_names)

    def test_skip_signal_directory(self, simple_dir):
        """Ensure a skip signal can be sent to stop parsing on directory."""
        out = []
        iterator = iter_path_contents(simple_dir, include_directories=True)
        for upath in iterator:
            if upath.name == "B":
                iterator.send("skip")
            out.append(upath)
        names = {x.stem for x in out}
        # Anything after B should have been skipped
        assert {"C", "D", "E", "F"}.isdisjoint(names)

    def test_max_depth_no_dirs(self, simple_dir):
        """Ensure maxdepth=1 only returns top level contents."""
        out = {x for x in iter_path_contents(simple_dir, maxdepth=1)}
        expected = {x for x in simple_dir.glob("*") if not x.is_dir()}
        assert out == expected

    def test_max_depth_dirs(self, simple_dir):
        """Ensure directories can also be returned."""
        myiter = iter_path_contents(simple_dir, maxdepth=1, include_directories=True)
        out = {x for x in myiter}
        expected = {x for x in simple_dir.glob("*")}
        assert out == expected


class TestGetLsDetails:
    """Tests for the get_ls details function."""

    def test_get_ls_details_basic(self, upath_local):
        """Happy path for getting file info."""
        ls_details = get_ls_details(upath_local)
        for detail_dict in ls_details:
            assert detail_dict
            assert isinstance(detail_dict, dict)

    def test_raises(self):
        """Ensure errors can be propagated."""
        match = "No such file or directory"
        bad_path = UPath("_probably_not_a_file.txt")
        with pytest.raises(OSError, match=match):
            get_ls_details(bad_path, on_error="raise")

    def test_callable(self):
        """Ensure a callable can be used to handle details."""
        my_counter = 0

        def increment(*args, **kwargs):
            nonlocal my_counter
            my_counter += 1

        bad_file = UPath("_probably_not_a_file.txt")
        get_ls_details(bad_file, on_error=increment)
        assert my_counter == 1

#
# class TestGetPath:
#     """Tests for getting a path from various objects."""
#
#     def test_pathlib(self):
#         """Ensure a pathlib object works with uri generator."""
#         my_path = Path(__file__)
#         path = get_uri(my_path)
#         assert isinstance(path, str)
#         assert path == f"file://{my_path!s}"
#
#     def test_str(self):
#         """Ensure a string simply returns itself."""
#         my_path = str(Path(__file__))
#         path = get_uri(my_path)
#         assert isinstance(path, str)
#         assert path == f"file://{my_path!s}"
#
#     def test_fs_spec(self, tmp_path):
#         """Ensure a fs spec object returns a path string."""
#         fs = fsspec.open(Path(tmp_path))
#         out = get_uri(fs)
#         assert out == f"file://{tmp_path}"
#
#     def test_open_file(self, tmp_path):
#         """Ensure an open file can be used."""
#         path = tmp_path / "file.txt"
#         with open(path, "wb") as f:
#             uri = get_uri(f)
#             assert uri == f"file://{path}"
#
#     def test_h5(self, tmp_path):
#         """Ensure a h5 file returns a path."""
#         path = tmp_path / "file.h5"
#         with h5py.File(path, "w") as f:
#             uri = get_uri(f)
#             assert uri == f"file://{path}"
#
#     def test_idempotent(self, tmp_path):
#         """Ensure the protocol doesn't keep getting appended."""
#         my_path = Path(__file__)
#         path = get_uri(my_path)
#         path2 = get_uri(path)
#         path3 = get_uri(path2)
#         assert path == path2 == path3
