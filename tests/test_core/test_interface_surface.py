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
  of `get_coord` (a coordinate), or a call of `spool` (a spool);
- what a method of one returns, as its interface annotates it, and the
  first of a tuple it returns: `coord, indexer = coord.select(...)`;
- a loop's or comprehension's coordinate, from `coord_map.items()` or
  `.values()`, a manager or a coordinate's `segments`;
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
# Each kind's own module, skipped as its code runs on dascore's objects
# only, and the interface whose members reads elsewhere must be.
HOMES = {
    "coord": ("core/coords.py", "BaseNumericCoord"),
    "manager": ("core/coordmanager.py", "BaseCoordManager"),
    "lazy": ("lazy/array.py", "BaseLazyArray"),
    "spool": ("core/spool.py", "BaseSpool"),
}
# Reads, by module, the guard sets aside: on what the pass mistakes for one
# of the objects, or with a default another package's object need not give.
SET_ASIDE = {
    # The flag `map` sets on all but one of the spools a split gives, so one
    # progress bar shows.
    ("spool", "utils/misc.py"): {"_no_progress"},
    # A parameter named `coord` holding a netCDF variable, an index's
    # coordinate record or an xarray coordinate.
    ("coord", "io/netcdf/utils.py"): {"attrs", "name"},
    ("coord", "io/index/planned.py"): {"coord_name", "is_relative", "value_kind"},
    ("coord", "xarray/patch.py"): {"attrs", "dims"},
    # The `.coords` of a summary, a query, a plan or an xarray object, a mapping.
    **{
        ("manager", path): {"items"}
        for path in (
            "core/summary.py",
            "io/index/catalog.py",
            "io/index/ingest.py",
            "io/index/query.py",
            "io/netcdf/utils.py",
            "utils/chunk_plan.py",
            "xarray/patch.py",
        )
    },
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
            held = self.kind(func.value)
            return RESULTS[held].get(func.attr) if held else None
        return None

    def first(self, node) -> str | None:
        """The kind of the first item of the tuple a method call returns."""
        call = isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        held = self.kind(node.func.value) if call else None
        return FIRSTS[held].get(node.func.attr) if held else None

    def element(self, node) -> tuple[int | None, str | None]:
        """Which item of each element of an iterable is a coordinate, if any."""
        call = isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        if call and node.func.attr in ("items", "values"):
            base = node.func.value
            maps = isinstance(base, ast.Attribute) and base.attr == "coord_map"
            if maps:
                return (1 if node.func.attr == "items" else None), "coord"
        if self.kind(node) == "manager":
            return 1, "coord"
        held = isinstance(node, ast.Attribute) and node.attr == "segments"
        return None, ("coord" if held and self.kind(node.value) == "coord" else None)

    def bind_target(self, target, kind: str | None, item: int | None = None):
        """Bind a loop or assignment target, or one item of a tuple target."""
        if isinstance(target, ast.Tuple) and item is not None:
            for number, part in enumerate(target.elts):
                self.bind_target(part, kind if number == item else None)
        elif isinstance(target, ast.Name):
            self.bind(target.id, kind if item is None else None)

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
        for target in node.targets:
            if isinstance(target, ast.Tuple):
                self.bind_target(target, self.first(node.value), 0)
            else:
                self.bind_target(target, self.kind(node.value))

    def visit_For(self, node):
        self.visit(node.iter)
        self.bind_target(node.target, *reversed(self.element(node.iter)))
        for child in (*node.body, *node.orelse):
            self.visit(child)

    def _visit_comprehension(self, node):
        outer = dict(self.names)
        for loop in node.generators:
            self.visit(loop.iter)
            self.bind_target(loop.target, *reversed(self.element(loop.iter)))
            for condition in loop.ifs:
                self.visit(condition)
        for field in ("elt", "key", "value"):
            if hasattr(node, field):
                self.visit(getattr(node, field))
        self.names = outer

    def visit(self, node):
        comprehensions = ast.ListComp | ast.SetComp | ast.GeneratorExp | ast.DictComp
        if isinstance(node, comprehensions):
            return self._visit_comprehension(node)
        return super().visit(node)

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.value is not None:
            stated = _kind_of(node.annotation)
            self.bind(node.target.id, stated or self.kind(node.value))

    def read(self, node, name: str) -> None:
        """Record a read of an attribute on an expression, if it holds one."""
        kind = self.kind(node)
        if kind is not None and self.path != HOMES[kind][0]:
            self.found[kind].add(name)

    def visit_Attribute(self, node):
        self.read(node.value, node.attr)
        self.generic_visit(node)

    def visit_Call(self, node):
        # `getattr(x, "name", default)` reads as `x.name` does.
        args = node.args
        named = len(args) > 1 and isinstance(args[1], ast.Constant)
        if getattr(node.func, "id", None) == "getattr" and named:
            self.read(args[0], args[1].value)
        self.generic_visit(node)


def attribute_reads(source: str, path: str = "") -> dict[str, set[str]]:
    """Each kind's attributes read in a module's source."""
    reads = _Reads(path)
    reads.visit(ast.parse(source))
    return dict(reads.found)


def _statements(name: str):
    """The methods and annotations an interface states, with its bases'."""
    tree = ast.parse(Path(interfaces.__file__).read_text())
    classes = {x.name: x for x in tree.body if isinstance(x, ast.ClassDef)}
    todo = [name]
    while todo:
        node = classes[todo.pop()]
        for child in ast.walk(node):
            if isinstance(child, ast.FunctionDef | ast.AnnAssign):
                yield child
        todo += [x.id for x in node.bases if getattr(x, "id", None) in classes]


def members(name: str) -> set[str]:
    """The members an interface states, with its bases' and object's."""
    names = (
        x.name if isinstance(x, ast.FunctionDef) else ast.unparse(x.target)
        for x in _statements(name)
    )
    return set(dir(object)) | set(names)


def _results(kind: str, first: bool) -> dict[str, str]:
    """The kind each method of a kind's interface returns, or leads a tuple of."""
    out = {}
    for node in _statements(HOMES[kind][1]):
        returns = getattr(node, "returns", None)
        if returns is None:
            continue
        is_tuple = isinstance(returns, ast.Subscript) and ast.unparse(returns.value)
        if first and is_tuple == "tuple":
            returns = returns.slice.elts[0]
        elif first or is_tuple:
            continue
        found = kind if ast.unparse(returns) == "Self" else _kind_of(returns)
        if found in HOMES:
            out[node.name] = found
    return out


RESULTS = {x: _results(x, first=False) for x in HOMES}
FIRSTS = {x: _results(x, first=True) for x in HOMES}


@pytest.fixture(scope="module")
def reads():
    """Each module's attributes read on each kind, by (kind, module)."""
    found = {}
    for path in sorted(ROOT.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        if relative.startswith("docs/"):
            continue
        for kind, names in attribute_reads(path.read_text(), relative).items():
            found[kind, relative] = names
    return found


def _read_on(reads, kind: str) -> set[str]:
    """Every attribute read on a kind, less those set aside."""
    out = set()
    for key, names in reads.items():
        if key[0] == kind:
            out |= names - SET_ASIDE.get(key, set())
    return out


class TestSurface:
    """dascore reads what the interfaces state."""

    @pytest.mark.parametrize("kind", sorted(HOMES))
    def test_reads_are_members(self, reads, kind):
        """Every attribute read on an object is a member of its interface."""
        assert sorted(_read_on(reads, kind) - members(HOMES[kind][1])) == []

    @pytest.mark.parametrize("key", sorted(SET_ASIDE))
    def test_exceptions_are_read(self, reads, key):
        """Each read the guard sets aside is still made there, and is no member."""
        stated = members(HOMES[key[0]][1])
        assert SET_ASIDE[key] <= reads.get(key, set()) - stated

    def test_finds_reads(self, reads):
        """The pass sees the reads dascore makes on each kind."""
        assert {"runs", "_grid", "data_id"} <= _read_on(reads, "coord")
        assert {"coord_map", "dims", "_update_grid"} <= _read_on(reads, "manager")
        assert {"get_gaps", "split"} <= _read_on(reads, "spool")


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

    def test_results_loops_and_getattr(self):
        """Method results, loop items and getattr reads are followed."""
        source = """
def f(cm: BaseCoordManager, spool):
    new, _ = cm.select(time=None)
    new.one
    cm.update(time=None).two
    cm.get_coord("time").three
    for name, coord in cm.coord_map.items():
        coord.four
    for coord in cm.coord_map.values():
        coord.five
    for name, coord in cm:
        coord.six
    [x.seven for x in cm.get_coord("time").segments]
    getattr(spool.select(time=None), "eight", None)
    x.nine
"""
        assert attribute_reads(source) == {
            "manager": {"select", "update", "get_coord", "coord_map", "one", "two"},
            "coord": {"three", "four", "five", "six", "segments", "seven"},
            "spool": {"select", "eight"},
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
