"""
Tests for dascore's import behavior.
"""

from __future__ import annotations

import subprocess
import sys
from textwrap import dedent

import pytest

import dascore
from dascore.utils.imports import lazy_import


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
    def test_nothing_expensive_is_imported_eagerly(self):
        """Walk one clean interpreter from a bare import through to viz.

        Every check needs a process which has not yet imported what the
        check before it pulls in, so they run in order inside one
        subprocess rather than one process each.
        """
        code = dedent("""
            import sys
            import dascore

            for name in ("matplotlib", "scipy.signal", "numba"):
                assert name not in sys.modules, name + " imported by dascore"

            from dascore.utils.imports import lazy_import

            hann = lazy_import("scipy.signal.windows", "hann")
            assert "scipy.signal" not in sys.modules, "lazy import resolved early"
            assert hann.__name__ == "hann", "lazy proxy resolved to the wrong thing"
            assert "scipy.signal" in sys.modules, "use did not resolve the proxy"

            try:
                dascore.not_a_real_attribute
            except AttributeError:
                pass
            else:
                raise AssertionError("AttributeError not raised")

            from dascore import viz

            assert callable(viz.waterfall), "from-import of viz did not work"
            assert callable(dascore.viz.waterfall), "viz attribute hook did not work"
            assert "matplotlib" in sys.modules, "viz left matplotlib unimported"
        """)
        _run_snippet(code)

    @pytest.mark.concurrency
    def test_jit_kernels_import_numba(self):
        """The jit kernel modules pull numba in when they are imported.

        Its own subprocess, and its own importorskip: folded into the test
        above it would report as passed on a job without numba installed,
        where what it says is nothing at all.
        """
        pytest.importorskip("numba")
        code = (
            "import sys, dascore; "
            "assert 'numba' not in sys.modules; "
            "import dascore.transform._kurtosis_kernels; "
            "assert 'numba' in sys.modules"
        )
        _run_snippet(code)

    def test_lazy_import_proxy_forwards_calls_and_attrs(self):
        """The lazy proxy should behave like the resolved target object."""
        sqrt = lazy_import("math", "sqrt")
        assert sqrt(4) == 2
        assert sqrt.__name__ == "sqrt"

    def test_viz_module_lazy_loads_in_process(self):
        """Accessing dascore.viz should use the package attribute hook."""
        assert callable(dascore.__getattr__("viz").waterfall)

    def test_missing_attribute_raises_in_process(self):
        """Unknown package attributes should raise in the parent process too."""
        with pytest.raises(AttributeError, match="not_a_real_attribute"):
            dascore.__getattr__("not_a_real_attribute")
