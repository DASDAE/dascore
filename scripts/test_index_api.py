"""Tests for the api indexing helpers."""

from __future__ import annotations

from _index_api import _get_base_address, _is_environment_path


class TestIsEnvironmentPath:
    """Tests for detecting environment paths nested in the project."""

    def test_venv_path(self):
        """A .venv created in the repo should be excluded."""
        assert _is_environment_path("/repo/.venv/lib/python3.12/foo.py")

    def test_site_packages_path(self):
        """Any site-packages path should be excluded."""
        assert _is_environment_path("/repo/env/lib/site-packages/scipy/x.py")

    def test_project_path(self):
        """Regular project paths should not be excluded."""
        assert not _is_environment_path("/repo/dascore/core/patch.py")


class TestGetBaseAddress:
    """Tests for converting paths to addresses."""

    def test_environment_path_returns_empty(self):
        """Environment paths should not get a base address."""
        path = "/repo/.venv/lib/python3.12/site-packages/scipy/signal.py"
        assert _get_base_address(path, "/repo") == ""

    def test_project_path_returns_address(self):
        """Project paths should convert to dotted addresses."""
        path = "/repo/dascore/core/patch.py"
        assert _get_base_address(path, "/repo") == "dascore.core.patch"
