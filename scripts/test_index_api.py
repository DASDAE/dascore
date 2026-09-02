"""Tests for the api indexing helpers."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace

import pytest
from _index_api import (
    _get_base_address,
    _is_environment_path,
    _yield_get_submodules,
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

    def test_nested_environment_raises(self, tmp_path):
        """A .venv in the checkout holds a copy, not the checkout's own."""
        installed = tmp_path / ".venv" / "lib" / "site-packages" / "dascore"
        installed.mkdir(parents=True)
        (installed / "__init__.py").write_text("")
        module = SimpleNamespace(
            __name__="dascore", __file__=str(installed / "__init__.py")
        )

        with pytest.raises(RuntimeError, match="PYTHONPATH"):
            assert_documenting_this_checkout(module, tmp_path)


class TestOptionalSubmodules:
    """A submodule whose optional dependency is absent is skipped, not fatal."""

    @pytest.fixture
    def package(self, tmp_path, monkeypatch):
        """A package with one importable and one dependency-less submodule."""
        pkg = tmp_path / "optpkg"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("")
        (pkg / "plain.py").write_text("VALUE = 1\n")
        (pkg / "needs_dep.py").write_text("import surely_not_installed_dep\n")
        (pkg / "broken.py").write_text("import optpkg.no_such_submodule\n")
        monkeypatch.syspath_prepend(str(tmp_path))
        # one package name per test: a cached import would hide the new files
        for name in [k for k in sys.modules if k.split(".")[0] == "optpkg"]:
            monkeypatch.delitem(sys.modules, name)
        return importlib.import_module("optpkg"), tmp_path

    def test_missing_dependency_warns_and_skips(self, package):
        """A third-party import failure skips the submodule with a warning."""
        module, base = package
        (base / "optpkg" / "broken.py").unlink()
        with pytest.warns(UserWarning, match="surely_not_installed_dep"):
            found = dict(_yield_get_submodules(module, base))
        assert set(found) == {"optpkg.plain"}

    def test_project_import_failure_still_raises(self, package):
        """A missing module inside the project itself is a bug, not a skip."""
        module, base = package
        (base / "optpkg" / "needs_dep.py").unlink()
        with pytest.raises(ModuleNotFoundError, match="no_such_submodule"):
            dict(_yield_get_submodules(module, base))
