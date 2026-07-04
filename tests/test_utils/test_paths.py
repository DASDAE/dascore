"""Tests for path classification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

from dascore.exceptions import InvalidSpoolError
from dascore.utils.paths import (
    coerce_to_local_path,
    coerce_to_upath,
    get_path_protocol,
    is_local_path,
    is_pathlike,
    requires_local_directory,
)


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

    def test_plain_string(self):
        """A plain string should become an equivalent Path."""
        assert coerce_to_local_path("/tmp/a/b.h5") == Path("/tmp/a/b.h5")

    def test_file_uri_scheme_stripped(self):
        """A file:// URI must strip the scheme to a real filesystem path."""
        out = coerce_to_local_path("file:///tmp/a/b.h5")
        assert out == Path("/tmp/a/b.h5")
        assert "://" not in str(out)

    def test_local_uri_scheme_stripped(self):
        """A local:// URI must also strip to a real filesystem path."""
        assert coerce_to_local_path("local:///tmp/x.h5") == Path("/tmp/x.h5")


class TestRequiresLocalDirectory:
    """Tests for ``requires_local_directory``."""

    def test_requires_local_directory(self):
        """Remote directories should be rejected by policy helpers."""
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            requires_local_directory(
                UPath("memory://dascore/testdir"),
                label="Directory spool",
            )
