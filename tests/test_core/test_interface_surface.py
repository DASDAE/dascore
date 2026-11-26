"""
dascore reads only the members its interfaces state.

An AST pass over dascore's modules follows, within each function, the
expressions which hold a coordinate, a manager, a lazy array or a spool,
and collects each attribute read on one outside that class's own module.
Another package's object meets dascore's code only through what its
interface states, so a read of anything else breaks an implementation:
the guard fails first, here.

An expression holds such an object when it is

- a parameter annotated with an interface, or, unannotated, named `coord`,
  `coords` or `cm` (a manager) or `spool`;
- a name an `if isinstance(name, <interface>)` narrows, inside that `if`;
- `x.coords` (a manager), `x.coord_map[...]`, a manager indexed, or a call
  of `get_coord` (a coordinate), a call of `spool`, or a spool's `select`,
  `chunk`, `update`, `concatenate` or `set_backend` (a spool);
- a local name assigned from one of these.

A name annotated or narrowed with a concrete class holds dascore's own
object, so reads on it are not followed. A coordinate's reads are checked
against the members of both kinds together.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

import dascore
from dascore.core import interfaces

ROOT = Path(dascore.__file__).parent
INTERFACES = {
    "BaseCoord": "coord",
    "BaseNumericCoord": "coord",
    "BaseStringCoord": "coord",
    "BaseCoordManager": "manager",
    "BaseLazyArray": "lazy",
    "BaseSpool": "spool",
    "SpoolType": "spool",
}
OWN = {"NumericCoord", "CoordString", "CoordManager", "LazyArray", "Spool"}
PARAMETERS = {"coord": "coord", "coords": "manager", "cm": "manager", "spool": "spool"}
SPOOL_METHODS = {"select", "chunk", "update", "concatenate", "set_backend"}
# Each kind's own module, whose code runs on dascore's objects only, and
# the interface whose members its reads must be.
HOMES = {
    "coord": ("core/coords.py", "BaseNumericCoord"),
    "manager": ("core/coordmanager.py", "BaseCoordManager"),
    "lazy": ("lazy/array.py", "BaseLazyArray"),
    "spool": ("core/spool.py", "BaseSpool"),
}
# Reads on what a route takes for one of the objects, but is not.
NOT_THE_OBJECT = {
    "coord": {
        # a netCDF variable (`io/netcdf/utils.py`)
        "attrs",
        "name",
        # an index's coordinate record (`io/index/planned.py`)
        "coord_name",
        "is_relative",
        "value_kind",
        # an xarray coordinate (`xarray/patch.py`)
        "dims",
    },
    # The `.coords` of a summary, a query or an xarray object, a mapping.
    "manager": {"items"},
}


def _classes(node) -> set[str] | None:
    """The class names a bare annotation or isinstance target names."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        node = ast.parse(node.value, mode="eval").body
    if isinstance(node, ast.Name):
        return {node.id}
    if isinstance(node, ast.Attribute):
        return {node.attr}
    if isinstance(node, ast.Constant) and node.value is None:
        return set()
    parts = []
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        parts = [node.left, node.right]
    elif isinstance(node, ast.Tuple):
        parts = node.elts
    found = [_classes(x) for x in parts]
    return None if not found or None in found else set().union(*found)


def _kind_of(node) -> str | None:
    """The kind a class expression names: one of HOMES, 'own' or None."""
    names = _classes(node)
    kinds = {"own" if x in OWN else INTERFACES.get(x) for x in names or ()}
    return kinds.pop() if len(kinds) == 1 else None


def _on_module(node) -> bool:
    """Whether an expression is a path into dascore's modules, as `dc.core`."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name) and node.id in ("dascore", "dc")


class _Reads(ast.NodeVisitor):
    """The attributes read on each kind, function by function."""

    def __init__(self, path: str):
        self.path = path
        self.names: dict[str, str] = {}
        self.found: dict[str, set[str]] = defaultdict(set)

    def kind(self, node) -> str | None:
        """The kind of object an expression holds, if any."""
        if isinstance(node, ast.Name):
            return self.names.get(node.id)
        if isinstance(node, ast.Attribute) and node.attr == "coords":
            return None if _on_module(node.value) else "manager"
        if isinstance(node, ast.Subscript):
            value = node.value
            held = isinstance(value, ast.Attribute) and value.attr == "coord_map"
            return "coord" if held or self.kind(value) == "manager" else None
        if isinstance(node, ast.Call) and isinstance(func := node.func, ast.Name):
            return {"get_coord": "coord", "spool": "spool"}.get(func.id)
        if isinstance(node, ast.Call) and isinstance(func := node.func, ast.Attribute):
            if func.attr == "get_coord":
                return "coord"
            if func.attr == "spool" and _on_module(func.value):
                return "spool"
            if func.attr in SPOOL_METHODS and self.kind(func.value) == "spool":
                return "spool"
        return None

    def bind(self, name: str, kind: str | None) -> None:
        """Say what a name holds from here on."""
        if kind in HOMES:
            self.names[name] = kind
        else:
            self.names.pop(name, None)

    def visit_FunctionDef(self, node):
        outer, self.names = self.names, dict(self.names)
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            stated = arg.annotation
            kind = PARAMETERS.get(arg.arg) if stated is None else _kind_of(stated)
            self.bind(arg.arg, kind)
        self.generic_visit(node)
        self.names = outer

    def visit_If(self, node):
        self.visit(node.test)
        test = node.test
        both = isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And)
        narrowed = {}
        for call in test.values if both else [test]:
            is_check = (
                isinstance(call, ast.Call)
                and getattr(call.func, "id", None) == "isinstance"
                and isinstance(call.args[0], ast.Name)
            )
            if is_check:
                name = call.args[0].id
                narrowed[name] = self.names.get(name)
                self.bind(name, _kind_of(call.args[1]))
        for child in node.body:
            self.visit(child)
        # Narrowed only inside the `if`; what the body assigned holds on.
        for name, kind in narrowed.items():
            self.bind(name, kind)
        for child in node.orelse:
            self.visit(child)

    def visit_Assign(self, node):
        self.generic_visit(node)
        kind = self.kind(node.value)
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.bind(target.id, kind)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            stated = _kind_of(node.annotation)
            self.bind(node.target.id, stated or self.kind(node.value))

    def visit_Attribute(self, node):
        kind = self.kind(node.value)
        if kind is not None and self.path != HOMES[kind][0]:
            self.found[kind].add(node.attr)
        self.generic_visit(node)


def attribute_reads(source: str, path: str = "") -> dict[str, set[str]]:
    """Each kind's attributes read in a module's source."""
    reads = _Reads(path)
    reads.visit(ast.parse(source))
    return dict(reads.found)


def members(name: str) -> set[str]:
    """The members an interface states, with its bases' and object's."""
    tree = ast.parse(Path(interfaces.__file__).read_text())
    classes = {x.name: x for x in tree.body if isinstance(x, ast.ClassDef)}
    out, todo = set(dir(object)), [name]
    while todo:
        node = classes[todo.pop()]
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef):
                out.add(child.name)
            elif isinstance(child, ast.AnnAssign):
                out.add(ast.unparse(child.target))
        todo += [x.id for x in node.bases if getattr(x, "id", None) in classes]
    return out


@pytest.fixture(scope="module")
def reads():
    """Each kind's attributes read across dascore."""
    found = defaultdict(set)
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("docs/"):
            continue
        for kind, names in attribute_reads(path.read_text(), relative).items():
            found[kind] |= names
    return found


class TestSurface:
    """dascore reads what the interfaces state."""

    @pytest.mark.parametrize("kind", sorted(HOMES))
    def test_reads_are_members(self, reads, kind):
        """Every attribute read on an object is a member of its interface."""
        stated = members(HOMES[kind][1])
        assert sorted(reads[kind] - stated - NOT_THE_OBJECT.get(kind, set())) == []

    @pytest.mark.parametrize("kind", sorted(NOT_THE_OBJECT))
    def test_exceptions_are_read(self, reads, kind):
        """Each read the guard sets aside is still made, and is no member."""
        stated = members(HOMES[kind][1])
        assert NOT_THE_OBJECT[kind] <= reads[kind] - stated

    def test_finds_reads(self, reads):
        """The pass sees the reads dascore makes on each kind."""
        assert {"runs", "_grid", "data_id"} <= reads["coord"]
        assert {"coord_map", "dims", "_update_grid"} <= reads["manager"]
        assert {"get_gaps", "split"} <= reads["spool"]


class TestRoutes:
    """Each route the pass follows, on a small source."""

    def test_parameters_and_narrowing(self):
        """Annotated, named and narrowed names are followed; dascore's not."""
        source = """
def f(coord, spool, a: BaseLazyArray, b: "LazyArray", c: BaseSpool | None, d):
    coord.one, spool.two, a.three, b.four, c.five
    if isinstance(d, BaseCoordManager):
        d.six
    d.seven
    if isinstance(coord, NumericCoord):
        coord.eight
"""
        assert attribute_reads(source) == {
            "coord": {"one"},
            "spool": {"two", "five"},
            "lazy": {"three"},
            "manager": {"six"},
        }

    def test_expressions(self):
        """Managers, coordinates and spools reached from expressions."""
        source = """
def f(patch, path):
    patch.coords.one
    patch.coords["time"].two
    patch.coords.coord_map["time"].three
    coord = patch.get_coord("time")
    coord.four
    out = dc.spool(path).select(time=None)
    out.five
    out = None
    out.six
    dc.core.coords.seven
    held: BaseCoord = get_coord(data=[1])
    held.eight
    if path:
        late = patch.coords
    late.nine
"""
        assert attribute_reads(source) == {
            "manager": {"one", "coord_map", "nine"},
            "coord": {"two", "three", "four", "eight"},
            "spool": {"five", "select"},
        }

    def test_own_module(self):
        """Reads in a class's own module are its own."""
        source = "def f(coord, spool):\n    coord.one, spool.two\n"
        assert attribute_reads(source, "core/coords.py") == {"spool": {"two"}}

    def test_members(self):
        """An interface's members include its annotations and its bases'."""
        assert {"runs", "values", "_grid", "select", "__eq__"} <= members(
            "BaseNumericCoord"
        )
        assert "runs" not in members("BaseStringCoord")
