"""
Tests for dascore's import behavior.
"""

from __future__ import annotations

import subprocess
import sys


class TestLazyImports:
    """Ensure expensive optional machinery is not imported eagerly."""

    def test_matplotlib_not_imported(self):
        """Importing dascore should not import matplotlib (it is slow)."""
        code = "import dascore, sys; assert 'matplotlib' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_scipy_signal_not_imported(self):
        """Importing dascore should not import scipy.signal (it is slow)."""
        code = "import dascore, sys; assert 'scipy.signal' not in sys.modules"
        subprocess.run([sys.executable, "-c", code], check=True)

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
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_viz_module_lazy_loads(self):
        """Accessing dascore.viz should still work via lazy (PEP 562) import."""
        code = (
            "import dascore; "
            "assert callable(dascore.viz.waterfall); "
            "import sys; assert 'matplotlib' in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_viz_from_import_still_works(self):
        """The package attribute hook should preserve from-import behavior."""
        code = "from dascore import viz; assert callable(viz.waterfall)"
        subprocess.run([sys.executable, "-c", code], check=True)

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
        subprocess.run([sys.executable, "-c", code], check=True)
