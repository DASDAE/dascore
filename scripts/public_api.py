"""
Record DASCore's public API and detect changes to it.

The baseline lists every public name with how it is called: a signature for
functions and methods, or `class`, `module`, `attribute`, `constant` and the
like for everything else. It covers `dascore.__all__`, the public names of
`PUBLIC_MODULES`, and the public members of `Patch` and `Spool`, namespaces
included.

    freeze    write the current surface to the baseline
    check     report names which were added, removed or changed

Signatures leave out annotations and render defaults by value or type name,
so the baseline reads the same on every Python version and platform CI runs.
"""

from __future__ import annotations

import argparse
import enum
import functools
import inspect
import sys
from collections.abc import Iterator
from importlib import import_module
from pathlib import Path, PurePath
from types import ModuleType

from _api_urls import _echo, load_baseline, write_baseline

BASELINE_PATH = Path(__file__).absolute().parent / "_baselines" / "public_api.tsv"

# Modules whose public names, beyond `dascore.__all__`, are pinned.
PUBLIC_MODULES = (
    "dascore.config",
    "dascore.examples",
    "dascore.exceptions",
    "dascore.io",
    "dascore.units",
    "dascore.utils.time",
    "dascore.warnings",
    "dascore.xarray",
)

# Classes whose public members are pinned, keyed as written here.
PUBLIC_CLASSES = ("dascore.Patch", "dascore.Spool")

_HEADER = (
    "# DASCore's public API: one name and how it is called per line.\n"
    "# Regenerate with `python scripts/public_api.py freeze`, and justify\n"
    "# each added, removed or changed entry in the pull request.\n"
)
_FIX = (
    "Run `python scripts/public_api.py freeze` and justify each change "
    "in the pull request."
)
_PRIMITIVES = (bool, int, float, complex, str, bytes, type(None), type(...))
_KINDS = ("added", "removed", "changed")


def _default_text(value) -> str:
    """Render a default the same way on every platform and version."""
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if type(value).__module__ == "numpy" and hasattr(value, "dtype"):
        # A numpy scalar: its str is stable where its repr changed in 2.0.
        return f"{type(value).__name__}({value})"
    if isinstance(value, _PRIMITIVES):
        return repr(value)
    if isinstance(value, tuple | list | set | frozenset):
        items = [_default_text(x) for x in value]
        if isinstance(value, set | frozenset):
            items = sorted(items)
        inner = ", ".join(items)
        if isinstance(value, tuple):
            return f"({inner}{',' if len(items) == 1 else ''})"
        return f"[{inner}]" if isinstance(value, list) else f"{{{inner}}}"
    if isinstance(value, dict):
        inner = ", ".join(
            f"{_default_text(k)}: {_default_text(v)}" for k, v in value.items()
        )
        return f"{{{inner}}}"
    if isinstance(value, PurePath):
        # Path defaults are often machine specific (a cache directory).
        return "<Path>"
    if inspect.isclass(value) or inspect.isroutine(value):
        return getattr(value, "__qualname__", type(value).__name__)
    # Anything else (a sentinel, a quantity) by type, never by a repr
    # which may embed a memory address.
    return f"<{type(value).__name__}>"


class _Default:
    """A default whose repr is its stable text."""

    def __init__(self, text: str):
        self.text = text

    def __repr__(self):
        return self.text


def render_signature(signature: inspect.Signature) -> str:
    """Return a signature without annotations and with stable defaults."""
    empty = inspect.Parameter.empty
    parameters = [
        x.replace(
            annotation=empty,
            default=empty if x.default is empty else _Default(_default_text(x.default)),
        )
        for x in signature.parameters.values()
    ]
    return str(signature.replace(parameters=parameters, return_annotation=empty))


def describe(obj) -> str:
    """Say how an object is used: a signature, or what kind of thing it is."""
    if isinstance(obj, ModuleType):
        return "module"
    if inspect.isclass(obj):
        return "class"
    if callable(obj):
        try:
            return render_signature(inspect.signature(obj))
        except (TypeError, ValueError):
            return "callable"
    return "constant"


def describe_member(cls, name: str) -> str:
    """Describe a class member without running its descriptor."""
    static = inspect.getattr_static(cls, name)
    if isinstance(static, property | functools.cached_property):
        return "attribute"
    if isinstance(static, classmethod | staticmethod):
        # Bound, so a classmethod reads as it is called: without `cls`.
        return describe(getattr(cls, name))
    if inspect.isfunction(static) or inspect.isclass(static):
        return describe(static)
    if callable(static):
        # A descriptor such as a patch ufunc; its type says how it is used.
        return f"<{type(static).__name__}>"
    return "constant"


def _is_defined_in(obj, module: ModuleType) -> bool:
    """
    True for what a module defines, or a package gathers from DASCore.

    A plain module's imports are not its API, while a package's
    `__init__` exists to re-export. Values are kept unless their type
    comes from a third-party library.
    """
    if isinstance(obj, ModuleType):
        return False
    origin = getattr(obj, "__module__", None) if callable(obj) else None
    if origin is None:
        return type(obj).__module__.split(".")[0] in {"builtins", "dascore"}
    if hasattr(module, "__path__"):
        return origin.split(".")[0] == "dascore"
    return origin == module.__name__


def module_names(module: ModuleType) -> list[str]:
    """Return a module's public names: its `__all__`, or what it defines."""
    names = getattr(module, "__all__", None)
    if names is not None:
        return sorted(names)
    return sorted(
        name
        for name, obj in vars(module).items()
        if not name.startswith("_") and _is_defined_in(obj, module)
    )


def _public_members(cls) -> Iterator[str]:
    """Yield the public member names of a class."""
    yield from (x for x in dir(cls) if not x.startswith("_"))


def _namespace_entries(cls, key: str) -> Iterator[tuple[str, str]]:
    """Yield the methods of each namespace registered on a class."""
    for name, namespace in sorted(cls.get_registered_namespaces().items()):
        # Plugins, tests and doc examples register namespaces of their own.
        if namespace.__module__.split(".")[0] != "dascore":
            continue
        yield f"{key}.{name}", "namespace"
        for member in _public_members(namespace):
            yield f"{key}.{name}.{member}", describe_member(namespace, member)


def current_api() -> dict[str, str]:
    """Return the public API, name to description."""
    dascore = import_module("dascore")
    out = {f"dascore.{x}": describe(getattr(dascore, x)) for x in dascore.__all__}
    for module_name in PUBLIC_MODULES:
        module = import_module(module_name)
        for name in module_names(module):
            out[f"{module_name}.{name}"] = describe(getattr(module, name))
    for key in PUBLIC_CLASSES:
        cls = getattr(dascore, key.split(".")[-1])
        namespaces = dict(_namespace_entries(cls, key))
        out.update(namespaces)
        for name in _public_members(cls):
            if f"{key}.{name}" not in namespaces:
                out[f"{key}.{name}"] = describe_member(cls, name)
    return out


def compare(current: dict[str, str], baseline: dict[str, str]) -> dict:
    """Return the names added, removed and changed relative to the baseline."""
    changed = {
        k: (v, current[k]) for k, v in baseline.items() if current.get(k, v) != v
    }
    return {
        "added": {k: current[k] for k in sorted(set(current) - set(baseline))},
        "removed": sorted(set(baseline) - set(current)),
        "changed": dict(sorted(changed.items())),
    }


def format_difference(difference: dict) -> str:
    """Describe a comparison, every entry included, and how to accept it."""
    lines = [
        "The public API differs from scripts/_baselines/public_api.tsv: "
        + ", ".join(f"{len(difference[x])} {x}" for x in _KINDS)
    ]
    lines += [f"  added: {k}\t{v}" for k, v in difference["added"].items()]
    lines += [f"  removed: {x}" for x in difference["removed"]]
    lines += [
        f"  changed: {k}\n      was {old}\n      now {new}"
        for k, (old, new) in difference["changed"].items()
    ]
    return "\n".join([*lines, _FIX])


def check(path: Path = BASELINE_PATH) -> str | None:
    """Return a description of how the API moved off the baseline, if it did."""
    difference = compare(current_api(), load_baseline(path))
    if not any(difference.values()):
        return None
    return format_difference(difference)


def _get_parser() -> argparse.ArgumentParser:
    """Build the command line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze", help="write the current surface to the baseline")
    sub.add_parser("check", help="fail if the surface left the baseline")
    return parser


def main(args=None, path: Path = BASELINE_PATH) -> int:
    """Run one sub command."""
    parsed = _get_parser().parse_args(args)
    if parsed.command == "freeze":
        api = current_api()
        write_baseline(api, path, header=_HEADER)
        _echo(f"Froze {len(api)} public names to {path}")
        return 0
    problem = check(path)
    _echo(problem or "The public API matches the baseline.")
    return 1 if problem else 0


if __name__ == "__main__":
    sys.exit(main())
