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
from dascore.lazy.contract import ChunkContract, Pointwise, Unchanged


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
