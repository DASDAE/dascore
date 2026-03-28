"""Tests for path classification helpers."""

from __future__ import annotations

from pathlib import Path

import pytest
from upath import UPath

from dascore.exceptions import InvalidSpoolError
from dascore.utils.paths import coerce_path, get_path_protocol, is_local_path, is_pathlike, requires_local_directory


class TestPathHelpers:
    """Tests for path helper utilities."""

    def test_is_pathlike(self, tmp_path):
        """Recognized path-like values should return True."""
        assert is_pathlike("a.txt")
        assert is_pathlike(tmp_path)
        assert is_pathlike(UPath(tmp_path))
        assert not is_pathlike(object())

    def test_coerce_path(self, tmp_path):
        """Path-like inputs should coerce to UPath."""
        out = coerce_path(tmp_path)
        assert isinstance(out, UPath)

    def test_get_path_protocol(self, tmp_path):
        """Protocols should normalize for local and remote paths."""
        assert get_path_protocol(tmp_path) == "file"
        assert get_path_protocol("local.txt") == "file"
        assert get_path_protocol("memory://dascore/test.txt") == "memory"
        assert get_path_protocol(object()) is None

    def test_is_local_path(self, tmp_path):
        """Local and remote path classification should be correct."""
        assert is_local_path(tmp_path)
        assert not is_local_path("memory://dascore/test.txt")

    def test_requires_local_directory(self):
        """Remote directories should be rejected by policy helpers."""
        with pytest.raises(InvalidSpoolError, match="local filesystem"):
            requires_local_directory(UPath("memory://dascore/testdir"), label="Directory spool")
