"""Tests for chunk contracts and the hook which states them."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import numpy as np
import pytest

import dascore as dc
import dascore.lazy
from dascore.exceptions import IncompatiblePatchError, LazyOperationError
from dascore.lazy.contract import (
    ChunkContract,
    Halo,
    Pointwise,
    Unchanged,
    Window,
)


class TestContracts:
    """The contracts themselves."""

    @pytest.mark.parametrize("kind", (Pointwise, Unchanged))
    def test_is_a_contract(self, kind):
        """Each is a contract, equal to another of its kind and frozen."""
        found = kind()
        assert isinstance(found, ChunkContract)
        assert found == kind()
        with pytest.raises(dataclasses.FrozenInstanceError):
            found.extra = 1

    def test_kinds_differ(self):
        """A no-op is not pointwise, nor the other way round."""
        assert Pointwise() != Unchanged()


class TestNeeds:
    """Which box of the input each contract reads for a box of the output."""

    box = (slice(10, 20), slice(2, 4))

    def test_halo_stops_at_the_ends(self):
        """Nothing is read from beyond the array."""
        halo = Halo({0: (50, 50), 1: (1, 1)})
        assert halo.needs(self.box, (25, 4)) == (slice(0, 25), slice(1, 4))

    @pytest.mark.parametrize(
        "index, start",
        ((slice(None), 0), (slice(5, None), 5), (slice(-30, -1), 70), (slice(7, 9), 7)),
    )
    def test_window_moves(self, index, start):
        """The box is moved by where the window starts, however it is spelled."""
        window = Window((index, slice(None)))
        found = window.needs((slice(0, 2), slice(2, 4)), (100, 5))
        assert found == (slice(start, start + 2), slice(2, 4))


class TestApply:
    """Computing a box of the output from the input it needs."""

    def test_trims_what_was_grown(self):
        """The kernel's result is cut back to the box wanted."""
        data = np.arange(14.0).reshape(7, 2)
        box, needed = (slice(5, 8), slice(0, 2)), (slice(3, 10), slice(0, 2))
        out = Halo({0: (2, 2)}).apply(lambda x: x * 2, data, box, needed)
        assert np.array_equal(out, data[2:5] * 2)


class TestHook:
    """What `PatchProcessor.chunk_contract` says by default."""

    def test_default_is_none(self):
        """An operation states no contract unless it writes one."""
        found = dc.PatchProcessor().chunk_contract(np.dtype("float64"), {})
        assert found is None

    def test_error_is_incompatible_patch(self):
        """The refusal is caught as any other incompatible patch."""
        assert issubclass(LazyOperationError, IncompatiblePatchError)


def _dascore_imports(nodes) -> list[str]:
    """The modules under dascore which statements among `nodes` import."""
    found = []
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            found.append("." if node.level else node.module or "")
        elif isinstance(node, ast.Import):
            found.extend(x.name for x in node.names)
    return [x for x in found if x == "." or x.split(".")[0] == "dascore"]


class TestLeaf:
    """
    The contract module can be imported while dascore is still importing.

    `dascore.core.processor` and the processors import it, so neither it
    nor the package around it may import anything else from DASCore. Read
    from the source, since `import dascore` runs first in any interpreter.
    """

    @staticmethod
    def _tree(name):
        """The parsed source of a file of the package."""
        path = Path(dascore.lazy.__file__).parent / name
        return ast.parse(path.read_text())

    def test_contract_imports_nothing_from_dascore(self):
        """Not at the top, and not inside a function either."""
        assert not _dascore_imports(ast.walk(self._tree("contract.py")))

    def test_package_imports_nothing_eagerly(self):
        """Only a module-level `__getattr__` may import, and only when asked."""
        tree = self._tree("__init__.py")
        hooks = [
            x
            for x in tree.body
            if isinstance(x, ast.FunctionDef) and x.name == "__getattr__"
        ]
        asked = {id(y) for x in hooks for y in ast.walk(x)}
        eager = [x for x in ast.walk(tree) if id(x) not in asked]
        assert not _dascore_imports(eager)
