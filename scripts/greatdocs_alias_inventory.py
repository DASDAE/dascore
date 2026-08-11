"""
Quarto pre-render hook: add alias entries to the great-docs link inventory.

DASCore exposes most functionality through aliases: processing functions are
attached to Patch as methods (``Patch.taper = dascore.proc.taper``), objects
are re-exported at several import paths, and narrative docs link to modules.
great-docs only indexes the canonical location of each documented object, so
links written against an alias path (e.g. ``[taper](`dascore.Patch.taper`)``)
would not resolve.

This script runs inside the great-docs build directory (wired via the
``pre_render`` key of great-docs.yml) after the API reference and
``objects.json`` are generated but before Quarto renders the site. It imports
dascore, resolves every inventory entry to its runtime object, and then adds
an inventory alias for every public access path that reaches a documented
object. Module links are pointed at their section of the API index page.

The heavy lifting (link resolution in rendered HTML) stays in great-docs;
this only feeds it a richer inventory.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

OBJECTS_JSON = Path("objects.json")

# Modules whose links should land on a section of the API index.
MODULE_SECTIONS = {
    "dascore.core": "reference/index.html#core-classes",
    "dascore.examples": "reference/index.html#example-data",
    "dascore.exceptions": "reference/index.html#exceptions",
    "dascore.io": "reference/index.html#io-interfaces",
    "dascore.proc": "reference/index.html#patch-processing",
    "dascore.proc.aggregate": "reference/index.html#patch-processing",
    "dascore.transform": "reference/index.html#transforms",
    "dascore.units": "reference/index.html#units-and-time",
    "dascore.utils": "reference/index.html#utilities",
    "dascore.viz": "reference/index.html#visualization",
    "dascore.constants": "reference/index.html",
}


def _ensure_dascore():
    """Import dascore, re-execing with the project venv python if needed.

    Quarto runs pre-render scripts with whatever python it finds first, which
    is not necessarily the environment dascore is installed in.
    """
    try:
        import dascore  # noqa: F401

        return
    except ImportError:
        pass
    project_dir = Path(os.environ.get("QUARTO_PROJECT_DIR", "."))
    for candidate in (
        project_dir.parent / ".venv" / "bin" / "python",
        project_dir.parent / ".venv" / "Scripts" / "python.exe",
    ):
        if candidate.is_file() and str(candidate) != sys.executable:
            os.execv(str(candidate), [str(candidate), *sys.argv])
    raise SystemExit(
        "greatdocs_alias_inventory: could not import dascore; install it in "
        "the python environment quarto uses for pre-render scripts."
    )


def _unwrap(obj):
    seen = set()
    while getattr(obj, "__wrapped__", None) is not None and id(obj) not in seen:
        seen.add(id(obj))
        obj = obj.__wrapped__
    return obj


def _resolve(name: str):
    """Resolve a dotted inventory name to a runtime object, or None."""
    import dascore

    parts = name.split(".")
    if parts[0] != "dascore":
        return None
    obj = dascore
    for part in parts[1:]:
        try:
            obj = getattr(obj, part)
        except AttributeError:
            return None
    return obj


def main() -> None:
    _ensure_dascore()
    import dascore as dc

    data = json.loads(OBJECTS_JSON.read_text())
    items = data["items"]
    known = {it["name"] for it in items}

    # Map runtime object id -> best inventory item (prefer non-module roles,
    # e.g. proc.taper the function page over proc.taper the module).
    by_id: dict[int, dict] = {}
    for it in items:
        obj = _resolve(it["name"])
        if obj is None:
            continue
        current = by_id.get(id(_unwrap(obj)))
        if current is None or (current["role"] == "module" and it["role"] != "module"):
            by_id[id(_unwrap(obj))] = it

    new_items = []

    def add_alias(name: str, target: dict) -> None:
        if name in known:
            return
        known.add(name)
        new_items.append(
            {
                "name": name,
                "domain": "py",
                "role": target["role"],
                "priority": "2",
                "uri": target["uri"],
                "dispname": target.get("dispname", "-"),
            }
        )

    # 1. Canonical module.qualname alias for every documented object (e.g.
    #    dascore.utils.patch.get_patch_names for BaseSpool.get_patch_names).
    for target in by_id.values():
        obj = _resolve(target["name"])
        module = getattr(obj, "__module__", None)
        qualname = getattr(obj, "__qualname__", None)
        if module and qualname and "<" not in qualname:
            add_alias(f"{module}.{qualname}", target)

    # 2. Aliases for public attributes of the documented core classes
    #    (covers the dynamically attached Patch/Spool methods).
    from dascore.core.coords import BaseCoord

    for cls in (dc.Patch, dc.BaseSpool, dc.PatchAttrs, dc.CoordManager, BaseCoord):
        prefixes = [f"dascore.{cls.__name__}", f"{cls.__module__}.{cls.__qualname__}"]
        import dascore.core

        if getattr(dascore.core, cls.__name__, None) is cls:
            prefixes.append(f"dascore.core.{cls.__name__}")
        for attr in dir(cls):
            if attr.startswith("_"):
                continue
            try:
                target = by_id.get(id(_unwrap(getattr(cls, attr))))
            except Exception:
                continue
            if target is None:
                continue
            for prefix in prefixes:
                add_alias(f"{prefix}.{attr}", target)

    # 3. Aliases for public re-exports one level below dascore and
    #    dascore.core (e.g. dascore.core.Patch, dascore.io.write).
    import dascore.core

    for module in (dc, dc.core):
        for attr in dir(module):
            if attr.startswith("_"):
                continue
            try:
                target = by_id.get(id(_unwrap(getattr(module, attr))))
            except Exception:
                continue
            if target is not None:
                add_alias(f"{module.__name__}.{attr}", target)

    # 4. Module links -> API index sections.
    for name, uri in MODULE_SECTIONS.items():
        add_alias(name, {"role": "module", "uri": uri, "dispname": "-"})

    if new_items:
        data["items"] = items + new_items
        data["count"] = len(data["items"])
        OBJECTS_JSON.write_text(json.dumps(data, indent=1))
    print(f"greatdocs_alias_inventory: added {len(new_items)} alias entries")


if __name__ == "__main__":
    main()
