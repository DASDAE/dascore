"""Benchmarks for importing dascore using pytest-codspeed."""

from __future__ import annotations

import importlib
import sys
import warnings

import pint
import pytest


def _dascore_module_names():
    """Get the names of all currently imported dascore modules."""
    return [x for x in sys.modules if x == "dascore" or x.startswith("dascore.")]


class TestImportBenchmarks:
    """
    Benchmarks for importing dascore.

    Only dascore's own modules are removed from the module cache, so these
    measure the cost of executing dascore's module bodies rather than the
    cost of importing third party dependencies (which python only pays once
    per interpreter). See tests/test_imports.py for the guards which keep
    slow optional dependencies out of the import chain entirely.
    """

    @pytest.fixture()
    def restore_dascore_modules(self):
        """Put the original dascore modules back after re-importing them."""
        # Import here so third party dependencies are warm before timing,
        # even when this file is the only one collected.
        importlib.import_module("dascore")
        saved = {x: sys.modules[x] for x in _dascore_module_names()}
        # Re-importing dascore makes (and installs) a new pint registry.
        registry = pint.get_application_registry().get()
        # dascore adds a warning filter on import; catch_warnings undoes that.
        with warnings.catch_warnings():
            yield
        for name in _dascore_module_names():
            del sys.modules[name]
        sys.modules.update(saved)
        pint.set_application_registry(registry)

    @pytest.mark.benchmark
    def test_import_dascore(self, restore_dascore_modules):
        """Time importing the top-level dascore module."""
        for name in _dascore_module_names():
            del sys.modules[name]
        importlib.import_module("dascore")
