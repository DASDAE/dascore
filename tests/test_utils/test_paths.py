"""Tests for path classification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

from dascore.exceptions import InvalidSpoolError
from dascore.utils.paths import (
    coerce_to_local_path,
    coerce_to_upath,
    directory_writable,
    get_path_protocol,
    is_example_uri,
    is_local_path,
    is_pathlike,
    requires_local_directory,
)


class TestDirectoryWritable:
    """directory_writable probes without leaking exceptions."""

    def test_writable_directory(self, tmp_path):
        """A normal writable directory returns True."""
        assert directory_writable(tmp_path) is True

    def test_unwritable_returns_false(self, tmp_path):
        """A probe that can't create its parent returns False, not OSError."""
        # a path *under a file* makes mkdir raise NotADirectoryError (OSError)
        a_file = tmp_path / "a_file"
        a_file.write_text("x")
        assert directory_writable(a_file / "sub") is False

    def test_existing_legacy_probe_is_preserved(self, tmp_path):
        """The writability probe never truncates a predictable old sentinel."""
        sentinel = tmp_path / "._dascore_write_test_delete_me"
        sentinel.write_text("keep me")
        assert directory_writable(tmp_path) is True
        assert sentinel.read_text() == "keep me"


class TestIsPathlike:
    """Tests for ``is_pathlike``."""

    def test_is_pathlike(self, tmp_path):
        """Recognized path-like values should return True."""
        assert is_pathlike("a.txt")
        assert is_pathlike(tmp_path)
        assert is_pathlike(UPath(tmp_path))
        assert not is_pathlike(object())


class TestCoerceToUPath:
    """Tests for ``coerce_to_upath``."""

    def test_coerce_path(self, tmp_path):
        """Path-like inputs should coerce to UPath."""
        out = coerce_to_upath(tmp_path)
        assert isinstance(out, UPath)


class TestGetPathProtocol:
    """Tests for ``get_path_protocol``."""

    def test_get_path_protocol(self, tmp_path):
        """Protocols should normalize for local and remote paths."""
        assert get_path_protocol(tmp_path) == "file"
        assert get_path_protocol("local.txt") == "file"
        assert get_path_protocol("memory://dascore/test.txt") == "memory"
        assert get_path_protocol(object()) is None


class TestIsLocalPath:
    """Tests for ``is_local_path``."""

    def test_is_local_path(self, tmp_path):
        """Local and remote path classification should be correct."""
        assert is_local_path(tmp_path)
        assert not is_local_path("memory://dascore/test.txt")
        assert not is_local_path(object())


class TestCoerceToLocalPath:
    """Tests for ``coerce_to_local_path``."""

    def test_plain_path_passthrough(self, tmp_path):
        """A plain Path should be returned unchanged."""
        assert coerce_to_local_path(tmp_path) == tmp_path

    def test_plain_string(self, tmp_path):
        """A plain string should become an equivalent Path."""
        target = tmp_path / "b.h5"
        assert coerce_to_local_path(str(target)) == target

    def test_file_uri_round_trips(self, tmp_path):
        """A file:// URI must strip the scheme back to the original path."""
        # Use as_uri()/tmp_path so the assertion is OS-agnostic (Windows file
        # URIs carry a drive letter, e.g. file:///C:/...).
        target = tmp_path / "b.h5"
        out = coerce_to_local_path(target.as_uri())
        assert out == target
        assert "://" not in str(out)

    def test_local_scheme_stripped(self, tmp_path):
        """The local:// scheme must also be stripped (same code path as file://).

        Asserted drive-agnostically since Windows drive handling in the URL can
        vary; the invariant is that no scheme survives and the name is kept.
        """
        out = coerce_to_local_path(f"local://{(tmp_path / 'x.h5').as_posix()}")
        assert "://" not in str(out)
        assert Path(out).name == "x.h5"


class TestRequiresLocalDirectory:
    """Tests for ``requires_local_directory``."""

    def test_requires_local_directory(self):
        """Remote directories should be rejected by policy helpers."""
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            requires_local_directory(
                UPath("memory://dascore/testdir"),
                label="Directory spool",
            )


class TestIsExampleUri:
    """The examples:// scheme is matched exactly."""

    def test_example_uri(self):
        """A name with the scheme is an example uri."""
        assert is_example_uri("examples://terra15_das_1_trimmed.hdf5")

    def test_similar_name_is_not_matched(self):
        """A real file whose name merely starts with examples is not."""
        assert not is_example_uri("examples_notes.h5")
        assert not is_example_uri("example://not_the_scheme.h5")

    def test_path_is_not_matched(self, tmp_path):
        """Ordinary paths are not example uris."""
        assert not is_example_uri(tmp_path / "data.h5")
