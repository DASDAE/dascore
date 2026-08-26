"""Tests for the api indexing helpers."""

from __future__ import annotations

import pytest
from _index_api import (
    _get_base_address,
    _is_environment_path,
    assert_documenting_this_checkout,
    parse_project,
)

import dascore as dc


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


class TestAliases:
    """Tests that a second name for one object doesn't claim its page."""

    def test_class_documented_under_defined_name(self):
        """BaseSpool is an alias of Spool; only Spool gets documented."""
        data = parse_project(dc)
        assert data[str(id(dc.Spool))]["name"] == "Spool"

    def test_non_alias_still_documented(self):
        """A name matching its object's own name is not treated as an alias."""
        data = parse_project(dc)
        assert data[str(id(dc.Patch))]["name"] == "Patch"


class TestAssertDocumentingThisCheckout:
    """Tests for catching a build which imported another checkout."""

    def test_this_checkout_passes(self):
        """The dascore in this checkout is the one the docs describe."""
        assert_documenting_this_checkout(dc)

    def test_other_checkout_raises(self, tmp_path):
        """A dascore installed elsewhere is named in the error."""
        with pytest.raises(RuntimeError, match="PYTHONPATH"):
            assert_documenting_this_checkout(dc, tmp_path)
