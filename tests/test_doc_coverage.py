"""
Tests that every public object in DASCore is represented in the documentation.

The API reference in great-docs.yml is curated: entries are grouped by task
rather than mirroring the package layout. Curation reads better than a generated
dump, but it means new public API is invisible in the docs until someone lists
it, and nothing about adding a function makes that omission noticeable. These
tests compare DASCore's public surface against everything the docs cover and
fail when an object is represented nowhere.

An object counts as covered when it is listed in the reference, rendered on the
utilities page, shown in the supported formats table, or named below as
deliberately undocumented.

Scope: the functions and classes a module defines or re-exports, and the public
methods and properties of documented classes. Constants and type aliases are
not checked, since they are rendered inline in signatures rather than given
pages of their own.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import pkgutil
from pathlib import Path

import pytest

import dascore as dc
from dascore.io.core import FiberIO
from dascore.utils.docs import iter_package_modules, iter_public

# Not every job that collects this suite installs the docs/test extras; the
# free-threaded and WASM workflows install pytest alone.
yaml = pytest.importorskip("yaml")

CONFIG_PATH = Path(dc.__file__).parents[1] / "great-docs.yml"

# The package whose public objects are rendered onto a single page rather than
# given a reference entry each. Must match utilities/index.qmd.
SINGLE_PAGE_PACKAGE = "dascore.utils"

# Modules whose contents are deliberately absent from the API reference. A
# trailing ".*" exempts the submodules but not the module itself, so public API
# added to dascore/io/__init__.py is still checked while reader internals
# several levels down are not.
UNREFERENCED_MODULES = {
    "dascore.compat": "Shims over numpy/scipy, not DASCore's own API.",
    "dascore.constants": "Type aliases and constants, rendered inline in signatures.",
    "dascore.io.*": (
        "Format reader internals. IO meant for users is re-exported at "
        "dascore.io and listed in the reference; the readers themselves appear "
        "in the supported formats table on about/supported_formats.qmd."
    ),
}

# Individual objects deliberately absent, where the module they live in is
# otherwise documented.
UNREFERENCED_OBJECTS = {
    "dascore.core.coords.ensure_consistent_dtype": "Coord construction helper.",
    "dascore.core.coords.get_compatible_values": "Coord construction helper.",
    "dascore.core.summary.normalize_source_patch_id": "Summary internal.",
    "dascore.core.inventory.InventoryNames": "Names the inventory machinery fills in.",
    "dascore.core.inventory.interval_masks": "Inventory coverage helper.",
    "dascore.proc.inventory.get_attr_values": "Enrichment internal, not re-exported.",
    "dascore.proc.inventory.resolve_contexts": "Enrichment internal, not re-exported.",
    "dascore.proc.inventory.PlacedRow": "Inventory placement internal.",
    "dascore.proc.inventory.RowEpochs": "Inventory placement internal.",
    "dascore.proc.inventory.channel_placements": "Inventory placement internal.",
    "dascore.proc.inventory.resolve_channel_pieces": "Inventory placement internal.",
    "dascore.proc.inventory.resolve_row_epochs": "Inventory placement internal.",
    "dascore.proc.inventory.resolve_split_pieces": "Inventory placement internal.",
    "dascore.core.inventory_loader.load_directory": "Reached through dc.inventory().",
}

# Class members which are mechanism rather than API: pydantic validators run
# during model construction and are never called by hand. Keyed by the class's
# path in great-docs.yml, and inherited by subclasses.
UNREFERENCED_MEMBERS = {
    "PatchAttrs.reject_coordinate_attributes": "Pydantic validator.",
    "proc.BaseCoord.check_time_units": "Pydantic validator.",
}

MISSING_MSG = """
{count} public object(s) appear nowhere in the documentation:

{listing}

Every public object needs one of:
  * an entry in the `reference:` section of great-docs.yml (the usual fix),
  * an entry in that file's `exclude:` list, if discovery should skip it,
  * a home in {package}, whose public objects are all rendered
    automatically onto the utilities page,
  * an entry in UNREFERENCED_MODULES or UNREFERENCED_OBJECTS in this file,
    with a short reason, if it is public but intentionally undocumented.
"""


def _load_config() -> dict:
    """Return the parsed great-docs configuration."""
    return yaml.safe_load(CONFIG_PATH.read_text())


def _walk_modules(package) -> list[str]:
    """Return importable module names in a package, skipping private paths."""
    out = [package.__name__]
    prefix = f"{package.__name__}."
    for info in pkgutil.walk_packages(package.__path__, prefix=prefix):
        if not any(part.startswith("_") for part in info.name.split(".")):
            out.append(info.name)
    return out


def _resolve(dotted: str):
    """
    Getattr-walk a dascore-relative dotted path, returning None if unreachable.

    Reference entries must be reachable this way, not merely importable: griffe
    resolves them with getattr, and one unreachable entry drops the whole
    reference to static analysis.
    """
    obj = dc
    for part in dotted.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _key(obj):
    """Return a hashable identity for an object, or None if it has none."""
    obj = inspect.unwrap(obj)
    try:
        hash(obj)
    except TypeError:
        return None
    return obj


def _iter_public_safe(module_name: str) -> list:
    """
    Return a module's public objects, tolerating a missing optional dependency.

    Only a package which is not installed at all is tolerated. An ImportError
    naming a dascore module, or naming an installed one (a misspelled `from
    numpy import ...`), is a broken import; swallowing it would drop every
    object in that module from the comparison and quietly weaken the test.
    """
    try:
        return list(iter_public(module_name))
    except ImportError as e:
        name = e.name or ""
        if name.startswith("dascore") or _is_installed(name):
            raise
        return []


def _is_installed(module_name: str) -> bool:
    """Return True if the named package can be found in the environment."""
    if not module_name:
        return True
    try:
        return importlib.util.find_spec(module_name.split(".")[0]) is not None
    except (ImportError, ValueError):
        return True


def _reexports(module_name: str) -> list:
    """
    Return public objects a package exposes but does not define.

    Re-exporting is how a package states what it considers public, so these are
    checked even when the module the object is defined in is exempt.
    """
    module = importlib.import_module(module_name)
    out = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name)
        unwrapped = inspect.unwrap(obj)
        if not (inspect.isfunction(unwrapped) or inspect.isclass(unwrapped)):
            continue
        if getattr(unwrapped, "__module__", "") != module_name:
            out.append((name, obj))
    return out


def _public_surface() -> dict:
    """Map each public object to the dotted path of the module defining it."""
    out = {}
    for module_name in _walk_modules(dc):
        for name, obj in _iter_public_safe(module_name):
            out.setdefault(inspect.unwrap(obj), f"{module_name}.{name}")
    # Anything a package with an exempt subtree re-exports is public by
    # intent, so record it under the package rather than where it is defined.
    for prefix in UNREFERENCED_MODULES:
        if not prefix.endswith(".*"):
            continue
        root = prefix[:-2]
        for name, obj in _reexports(root):
            out[inspect.unwrap(obj)] = f"{root}.{name}"
    return out


def _reference_entries(config) -> list[tuple[str, list[str] | None]]:
    """
    Return (dotted path, member names) for every reference entry.

    Members are None for an entry which does not name them, which is how
    great-docs is told to document them all; an entry naming an empty list
    documents none of them, and is checked like any other explicit list.
    """
    out = []
    for group in config.get("reference", []):
        for item in group.get("contents", []):
            if isinstance(item, str):
                out.append((item, None))
            else:
                out.append((item["name"], item.get("members")))
    return out


def _documented(config) -> set:
    """Return every object the documentation covers."""
    out = set()
    for dotted, members in _reference_entries(config):
        obj = _resolve(dotted)
        if obj is None:
            continue
        out.add(_key(obj))
        for member in members or []:
            sub = getattr(obj, member, None)
            if sub is not None:
                out.add(_key(sub))
    # Walked the way the page renders it, so the two cannot disagree about
    # which modules are covered.
    for module_name in iter_package_modules(SINGLE_PAGE_PACKAGE):
        out.update(_key(obj) for _, obj in _iter_public_safe(module_name))
    for dotted in config.get("exclude", []):
        out.add(_key(_resolve(dotted)))
    out.discard(None)
    return out


def _accounted_members(config) -> dict:
    """Map each documented class to the member names accounted for on it."""
    out: dict = {}
    for dotted, members in _reference_entries(config):
        obj = _resolve(dotted)
        if not inspect.isclass(obj):
            continue
        exempt = {
            name.rsplit(".", 1)[1]
            for name in UNREFERENCED_MEMBERS
            if name.rsplit(".", 1)[0] == dotted
        }
        out.setdefault(obj, set()).update(members or [], exempt)
    return out


def _formats_table_entries() -> set:
    """Return the reader classes the supported formats page lists."""
    FiberIO.get_supported_io_table()  # loads the plugins the table is built from
    return {
        type(fiberio)
        for versions in FiberIO.manager._format_version.values()
        for fiberio in versions.values()
    }


def _in_formats_table(obj, readers: set) -> bool:
    """
    Return True for a reader the supported formats page actually lists.

    Subclassing FiberIO is not enough, and neither is sharing a registered
    reader's name and version: the table lists the reader the registry holds,
    so an unregistered one is documented nowhere.
    """
    return obj in readers


def _member_origin(attr) -> str:
    """
    Return the module a class member comes from.

    A property has no __module__ of its own, so ask the getter it wraps;
    without this every new property would look like inherited third-party
    machinery and be skipped.
    """
    if isinstance(attr, property):
        attr = attr.fget
    return getattr(attr, "__module__", "") or ""


def _is_exempt(dotted: str) -> bool:
    """Return True if policy says this object need not be documented."""
    if dotted in UNREFERENCED_OBJECTS:
        return True
    module = dotted.rsplit(".", 1)[0]
    for prefix in UNREFERENCED_MODULES:
        if prefix.endswith(".*"):
            if module.startswith(prefix[:-1]):
                return True
        elif module == prefix or module.startswith(f"{prefix}."):
            return True
    return False


@pytest.fixture(scope="module")
def config():
    """The parsed great-docs configuration."""
    if not CONFIG_PATH.exists():
        pytest.skip("great-docs.yml is only present in a source checkout")
    return _load_config()


@pytest.fixture(scope="module")
def documented(config):
    """Every object covered by the documentation."""
    return _documented(config)


@pytest.fixture(scope="module")
def surface():
    """Every public object in DASCore, mapped to where it is defined."""
    return _public_surface()


class TestDocCoverage:
    """Ensure the curated documentation keeps pace with the public API."""

    def test_every_public_object_is_documented(self, documented, surface):
        """Public objects must be represented somewhere in the docs."""
        table = _formats_table_entries()
        missing = {
            dotted
            for obj, dotted in surface.items()
            if obj not in documented
            and not _in_formats_table(obj, table)
            and not _is_exempt(dotted)
        }
        if missing:
            listing = "\n".join(f"  {name}" for name in sorted(missing))
            msg = MISSING_MSG.format(
                count=len(missing), listing=listing, package=SINGLE_PAGE_PACKAGE
            )
            pytest.fail(msg)

    def test_reference_entries_resolve(self, config):
        """
        Every reference entry must be reachable by getattr from dascore.

        The doc build fails on a deleted object, but slowly and with a less
        direct message; an entry that is importable yet not attribute-reachable
        does something worse, silently degrading the whole reference to static
        analysis so templated docstrings render as literal placeholders.
        """
        unresolved = [
            dotted
            for dotted, _ in _reference_entries(config)
            if _resolve(dotted) is None
        ]
        assert not unresolved, (
            f"great-docs.yml lists objects which no longer exist or are not "
            f"reachable as attributes of dascore: {sorted(unresolved)}"
        )

    def test_listed_members_exist(self, config):
        """Members named in the reference must exist on their class."""
        missing = []
        for dotted, members in _reference_entries(config):
            obj = _resolve(dotted)
            if obj is None:
                continue
            missing += [
                f"{dotted}.{m}" for m in members or [] if getattr(obj, m, None) is None
            ]
        assert not missing, (
            f"great-docs.yml names members which no longer exist: {sorted(missing)}"
        )

    def test_public_members_are_documented(self, config, documented):
        """
        Public members of documented classes must themselves be covered.

        A class listed with an explicit `members` list shows only those members,
        so a new method is invisible unless it is added there. A member is also
        covered when it is documented in its own right (patch functions are
        attached to Patch as methods), when an ancestor's entry already accounts
        for the name (Spool.chunk overrides the documented BaseSpool.chunk), or
        when its type is documented, which is how the generated arithmetic
        methods on Patch are covered: they are all instances of PatchUFunc,
        whose page describes what every one of them does.
        """
        accounted = _accounted_members(config)
        missing = []
        for dotted, members in _reference_entries(config):
            obj = _resolve(dotted)
            if not inspect.isclass(obj) or members is None:
                continue
            names = set().union(*(accounted.get(a, set()) for a in obj.__mro__))
            for name in dir(obj):
                if name.startswith("_") or name in names:
                    continue
                attr = inspect.unwrap(inspect.getattr_static(obj, name, None))
                if not _member_origin(attr).startswith("dascore"):
                    continue  # inherited from pydantic, etc.
                if _key(attr) in documented or type(attr) in documented:
                    continue
                missing.append(f"{dotted}.{name}")
        assert not missing, (
            f"Public members are missing from the `members` lists in "
            f"great-docs.yml: {sorted(missing)}"
        )

    def test_exclude_list_is_current(self, config):
        """Excluded names must still exist, so the list does not go stale."""
        stale = [d for d in config.get("exclude", []) if _resolve(d) is None]
        assert not stale, (
            f"great-docs.yml excludes objects which no longer exist: {sorted(stale)}"
        )
