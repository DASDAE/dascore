"""
Tests for dascore's import behavior.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_snippet(code: str) -> None:
    """Run a python snippet in a subprocess."""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=30,
        capture_output=True,
        text=True,
    )


class TestLazyImports:
    """Ensure expensive optional machinery is not imported eagerly."""

    @pytest.mark.concurrency
    def test_matplotlib_not_imported(self):
        """Importing dascore should not import matplotlib (it is slow)."""
        code = "import dascore, sys; assert 'matplotlib' not in sys.modules"
        _run_snippet(code)

    @pytest.mark.concurrency
    def test_scipy_signal_not_imported(self):
        """Importing dascore should not import scipy.signal (it is slow)."""
        code = "import dascore, sys; assert 'scipy.signal' not in sys.modules"
        _run_snippet(code)

    @pytest.mark.concurrency
    def test_lazy_import_doesnt_import_scipy_signal_until_use(self):
        """The lazy proxy should resolve scipy.signal only on first use."""
        code = (
            "import sys; "
            "from dascore.utils.imports import lazy_import; "
            "hann = lazy_import('scipy.signal.windows', 'hann'); "
            "assert 'scipy.signal' not in sys.modules; "
            "assert hann.__name__ == 'hann'; "
            "assert 'scipy.signal' in sys.modules"
        )
        _run_snippet(code)

    def test_lazy_import_proxy_forwards_calls_and_attrs(self):
        """The lazy proxy should behave like the resolved target object."""
        from dascore.utils.imports import lazy_import

        sqrt = lazy_import("math", "sqrt")
        assert sqrt(4) == 2
        assert sqrt.__name__ == "sqrt"

    @pytest.mark.concurrency
    def test_viz_module_lazy_loads(self):
        """Accessing dascore.viz should still work via lazy (PEP 562) import."""
        code = (
            "import dascore; "
            "assert callable(dascore.viz.waterfall); "
            "import sys; assert 'matplotlib' in sys.modules"
        )
        _run_snippet(code)

    def test_viz_module_lazy_loads_in_process(self):
        """Accessing dascore.viz should use the package attribute hook."""
        import dascore

        assert callable(dascore.__getattr__("viz").waterfall)

    @pytest.mark.concurrency
    def test_viz_from_import_still_works(self):
        """The package attribute hook should preserve from-import behavior."""
        code = "from dascore import viz; assert callable(viz.waterfall)"
        _run_snippet(code)

    @pytest.mark.concurrency
    def test_missing_attribute_raises(self):
        """Unknown attributes on the package should still raise AttributeError."""
        code = (
            "import dascore\n"
            "try:\n"
            "    dascore.not_a_real_attribute\n"
            "except AttributeError:\n"
            "    pass\n"
            "else:\n"
            "    raise AssertionError('AttributeError not raised')\n"
        )
        _run_snippet(code)

    def test_missing_attribute_raises_in_process(self):
        """Unknown package attributes should raise in the parent process too."""
        import dascore

        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            dascore.__getattr__("not_a_real_attribute")
