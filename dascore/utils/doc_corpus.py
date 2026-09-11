"""Prepare local Markdown from installed documentation and public APIs."""

from __future__ import annotations

import inspect
import os
import re
from collections import defaultdict
from functools import cached_property
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit, urlunsplit

import yaml
from pydantic import BaseModel

import dascore as dc
from dascore.utils.array import PatchUFunc

PACKAGE_PATH = Path(dc.__file__).resolve().parent
DOC_PATH = PACKAGE_PATH / "docs"
# Import the analysis API, without crawling file-format or external plugins.
_API_PACKAGES = ("core", "proc", "transform", "viz")
_API_MODULES = (
    "dascore",
    "dascore.io",
    "dascore.io.core",
    "dascore.config",
    "dascore.examples",
    "dascore.units",
    "dascore.utils.docs",
    "dascore.utils.downloader",
    "dascore.utils.time",
)
_EXCLUDED = {"api", "filters", "_static", "_site", "site_libs", "lite", "lite_contents"}
_LINK = re.compile(r"(!?\[[^\n]*?\])\(([^\s)]+)\)")
_INCLUDE = re.compile(r"\{\{<\s*include\s+([^>]+?)\s*>\}\}")


class DocumentationError(ValueError):
    """An installed documentation source or lookup could not be resolved."""


def source_paths() -> list[Path]:
    """Return authored documents, pruning assets and generated directories."""
    out = []
    for parent, directories, files in DOC_PATH.walk():
        directories[:] = [
            name
            for name in directories
            if not (
                name.startswith(".") or name.endswith("_files") or name in _EXCLUDED
            )
        ]
        for name in files:
            path = parent / name
            if path.suffix in {".qmd", ".md"}:
                out.append(path)
    return sorted(out)


def _unwrap(obj):
    """Inspect the callable behind a method or decorator without binding it."""
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        obj = obj.fget
    if isinstance(obj, cached_property):
        obj = obj.func
    # Read wrapper metadata statically, without invoking arbitrary descriptors.
    wrapped = inspect.getattr_static(obj, "__wrapped__", None)
    if inspect.isfunction(wrapped):
        obj = wrapped
    if isinstance(obj, PatchUFunc):
        return obj
    if not (
        inspect.ismodule(obj)
        or inspect.isclass(obj)
        or inspect.isfunction(obj)
        or inspect.ismethod(obj)
    ):
        return None
    return inspect.unwrap(obj)


class _APIDocumentCollector:
    """Collect public API records and aliases without evaluating operations."""

    def __init__(self):
        self.records = {}
        self.seen = set()
        self.omitted = []

    def collect(self):
        """Discover owned APIs and return their records and unavailable modules."""
        self._collect_modules()
        self._collect_namespaces()
        self._finalize_aliases()
        return self.records, self.omitted

    @staticmethod
    def _module_names():
        """Find public analysis modules without crawling external plugins."""
        modules = set(_API_MODULES)
        for package in _API_PACKAGES:
            for path in (PACKAGE_PATH / package).rglob("*.py"):
                relative = path.relative_to(PACKAGE_PATH)
                if any(p.startswith("_") for p in relative.parts[:-1]):
                    continue
                if path.stem.startswith("_") and path.stem != "__init__":
                    continue
                parts = relative.with_suffix("").parts
                modules.add(
                    "dascore."
                    + ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
                )
        return sorted(modules)

    def _collect_modules(self):
        """Inspect public module members and report missing optional dependencies."""
        for name in self._module_names():
            try:
                module = import_module(name)
            except ModuleNotFoundError as exc:
                if (exc.name or "").startswith("dascore"):
                    raise
                self.omitted.append(f"{name}: missing {exc.name}")
                continue
            self._add_object(module, name)
            for member_name, member in sorted(vars(module).items()):
                if not member_name.startswith("_"):
                    self._add_object(member, f"{name}.{member_name}")

    def _collect_namespaces(self):
        """Add only namespace entry points owned by the installed distribution."""
        hosts = {
            "patch": dc.Patch,
            "spool": dc.Spool,
            "inventory": dc.Inventory,
            "annotation": dc.AnnotationSet,
        }
        for entry in metadata.distribution("dascore").entry_points:
            for kind, host in hosts.items():
                if entry.group == f"dascore.{kind}_namespace":
                    namespace = entry.load()
                    for prefix in (
                        f"dascore.{host.__name__}",
                        f"{host.__module__}.{host.__name__}",
                    ):
                        self._add_object(namespace, f"{prefix}.{entry.name}")

    def _add_object(self, original, alias, owner=None):
        """Record an owned object and visit public members of each class alias."""
        obj = _unwrap(original)
        module = (
            obj.__name__
            if isinstance(obj, ModuleType)
            else getattr(obj, "__module__", "")
        )
        if not module.startswith("dascore"):
            return
        if not (
            inspect.ismodule(obj)
            or inspect.isclass(obj)
            or inspect.isfunction(obj)
            or inspect.ismethod(obj)
            or isinstance(original, PatchUFunc)
        ):
            return
        if isinstance(original, PatchUFunc):
            assert owner is not None
            module = owner.__module__
            key = f"{module}.{owner.__qualname__}.{alias.rsplit('.', 1)[-1]}"
        else:
            key = (
                f"module:{module}"
                if inspect.ismodule(obj)
                else f"{module}.{obj.__qualname__}"
            )
        if key not in self.records:
            body = self._object_body(original, obj, module, owner)
            self.records[key] = self._new_record(key, body)
        self.records[key]["aliases"].append(alias)
        visit = (id(obj), alias)
        if not inspect.isclass(obj) or visit in self.seen:
            return
        self.seen.add(visit)
        self._add_class_members(obj, alias, key)

    @staticmethod
    def _new_record(key, body):
        """Give object and model-field documents the same record structure."""
        return dict(
            id=key,
            title=key,
            kind="api",
            aliases=[],
            keywords=[],
            path="api/" + key.removeprefix("module:").replace(".", "/") + ".md",
            body=body,
        )

    @staticmethod
    def _signature(original, obj):
        """Read signatures without binding descriptors or evaluating annotations."""
        if isinstance(original, (property, cached_property)):
            return ""
        try:
            target = obj.__call__ if isinstance(original, PatchUFunc) else obj
            return str(inspect.signature(target, eval_str=False))
        except (TypeError, ValueError):
            return ""

    def _object_body(self, original, obj, module, owner):
        """Describe the raw docstring, signature, and Python calling convention."""
        signature = self._signature(original, obj)
        prefix = f"Defined in: `{module}`\n\n"
        if isinstance(original, PatchUFunc):
            prefix += "This Patch operation wraps the NumPy ufunc documented below.\n\n"
        if isinstance(original, (property, cached_property)):
            prefix += "Property: access this as an attribute, without calling it.\n\n"
        if signature:
            prefix += f"```python\n{obj.__name__}{signature}\n```\n\n"
        if owner is not None and signature:
            prefix += (
                "Class method signatures are unbound; "
                "Python supplies the first class parameter.\n\n"
                if isinstance(original, classmethod)
                else "Method signatures are unbound. For an instance method, "
                "its instance supplies the first parameter. "
                "Direct and static calls require all shown parameters.\n\n"
            )
        return prefix + (inspect.getdoc(obj) or "No docstring is available.")

    def _add_class_members(self, obj, alias, key):
        """Inspect public members of a class without evaluating properties."""
        for name, member in inspect.getmembers_static(obj):
            if not name.startswith("_") and not inspect.isclass(member):
                self._add_object(member, f"{alias}.{name}", obj)
        if issubclass(obj, BaseModel):
            self._add_model_fields(obj, alias, key)

    def _add_model_fields(self, obj, alias, key):
        """Document model fields using their declared types and descriptions."""
        for name, field in obj.model_fields.items():
            field_key = f"{key}.{name}"
            body = (
                f"Defined in: `{obj.__module__}`\n\nType: `{field.annotation}`\n\n"
                f"{field.description or 'No field description.'}"
            )
            record = self.records.setdefault(
                field_key, self._new_record(field_key, body)
            )
            record["aliases"].append(f"{alias}.{name}")

    def _finalize_aliases(self):
        """Add short names and prefer exported objects over colliding modules."""
        object_aliases = {
            alias
            for record in self.records.values()
            if not record["id"].startswith("module:")
            for alias in (*record["aliases"], record["id"])
        }
        for record in self.records.values():
            aliases = set(record["aliases"]) | {record["id"]}
            # Modules remain addressable by their explicit module: identifier.
            if record["id"].startswith("module:"):
                aliases -= object_aliases
            aliases |= {x.removeprefix("dascore.") for x in aliases}
            record["aliases"] = sorted(aliases)


def _api_documents():
    """Collect owned objects and aliases from the supported public modules."""
    return _APIDocumentCollector().collect()


def _read_readme():
    """Read the canonical README from the checkout or installed wheel metadata."""
    path = PACKAGE_PATH.parent / "readme.md"
    if path.is_file() and (PACKAGE_PATH.parent / "pyproject.toml").is_file():
        return path.read_text(encoding="utf-8")
    text = metadata.distribution("dascore").read_text("METADATA") or ""
    return text.partition("\n\n")[2]


def _read_source(path, stack=()):
    """Expand packaged includes, including the README shipped in metadata."""
    if path in stack:
        raise DocumentationError(f"Circular documentation include: {path}")
    text = path.read_text(encoding="utf-8")

    def include(match):
        target = (path.parent / match[1].strip().strip("\"'")).resolve()
        if target == PACKAGE_PATH.parent / "readme.md":
            return _read_readme()
        if not target.is_relative_to(DOC_PATH):
            raise DocumentationError(
                f"Documentation include escapes its source directory: {target}"
            )
        return _read_source(target, (*stack, path))

    return _INCLUDE.sub(include, text)


def _frontmatter(text):
    """Separate YAML metadata from a document's authored body."""
    if not text.startswith("---\n"):
        return {}, text
    parts = text.split("\n---", 2)
    if len(parts) < 2:
        raise DocumentationError("Unterminated documentation frontmatter")
    front = yaml.safe_load(parts[0][4:]) or {}
    return front, "\n---".join(parts[1:]).lstrip("\n")


def _image_ref():
    """Pin hosted images to the installed commit or release when known."""
    version = dc.__version__
    if commit := re.search(r"\+g([0-9a-f]+)", version):
        return commit[1]
    public = version.split("+", 1)[0]
    return "dev" if ".dev" in public or public == "0.0.0" else f"v{public}"


def _markdown(text, path, aliases, authored):
    """Normalize executable fences and links while keeping example code intact."""

    def link(match):
        label, target = match.groups()
        url = urlsplit(target)
        reference = unquote(url.path).strip("`")
        if reference in aliases and len(aliases[reference]) == 1:
            dest = aliases[reference][0]
            relative = os.path.relpath(dest, Path(path).parent).replace(os.sep, "/")
            destination = urlunsplit(("", "", relative, url.query, url.fragment))
            return f"{label}({destination})"
        if unquote(url.path).startswith("`"):
            # Unsupported optional APIs have an explicit route to the site index.
            return f"{label}(https://dascore.org/api/dascore.html)"
        if url.scheme or url.netloc or not url.path:
            return match[0]
        source = Path(os.path.normpath(str(Path(path).parent / url.path)))
        if url.path.startswith("/"):
            source = Path(url.path.lstrip("/"))
        if source.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg"}:
            hosted = urlunsplit(
                (
                    "https",
                    "raw.githubusercontent.com",
                    f"/DASDAE/dascore/{_image_ref()}/dascore/docs/{source.as_posix()}",
                    url.query,
                    url.fragment,
                )
            )
            return f"{label}({hosted})"
        if (
            source.suffix in {".qmd", ".md"}
            and source.with_suffix("").as_posix() in authored
        ):
            destination = source.with_suffix(".md").as_posix()
            relative = os.path.relpath(destination, Path(path).parent).replace(
                os.sep, "/"
            )
            return f"{label}({urlunsplit(('', '', relative, url.query, url.fragment))})"
        if url.path.endswith(".html") or source.suffix == ".qmd":
            external = urlunsplit(
                (
                    "https",
                    "dascore.org",
                    source.with_suffix(".html").as_posix(),
                    url.query,
                    url.fragment,
                )
            )
            return f"{label}({external})"
        return match[0]

    lines, fence = [], ""
    for line in text.splitlines():
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            if not fence:
                fence = marker[1]
                line = re.sub(r"^(\s*`{3,})\{(\w+)[^}]*\}", r"\1\2", line)
            elif stripped.startswith(fence) and not stripped[len(fence) :].strip():
                fence = ""
            lines.append(line)
        elif fence:
            lines.append(line)
        elif stripped.startswith(":::"):
            if callout := re.search(r"callout-(\w+)", stripped):
                lines.append(f"**{callout[1].capitalize()}:**")
        else:
            lines.append(_LINK.sub(link, line))
    return "\n".join(lines) + "\n"


def _disambiguate_paths(records):
    """Keep distinct Python names distinct on case-insensitive filesystems."""
    groups = defaultdict(list)
    for record in records.values():
        groups[record["path"].casefold()].append(record)
    for group in groups.values():
        if len(group) > 1:
            for number, record in enumerate(sorted(group, key=lambda x: x["id"]), 1):
                path = Path(record["path"])
                # A hyphen cannot collide with another Python identifier.
                record["path"] = path.with_name(f"{path.stem}-{number}.md").as_posix()


def write_corpus(destination: Path) -> dict:
    """Write the installed API and authored Markdown corpus and return its index."""
    records, omitted = _api_documents()
    for source in source_paths():
        relative = source.relative_to(DOC_PATH)
        front, body = _frontmatter(_read_source(source))
        key = relative.with_suffix("").as_posix()
        keywords = front.get("keywords", [])
        if isinstance(keywords, str):
            keywords = [keywords]
        records[key] = dict(
            id=key,
            title=front.get("title", key),
            kind="document",
            aliases=[
                key,
                relative.as_posix(),
                f"docs/{relative.as_posix()}",
                f"dascore/docs/{relative.as_posix()}",
            ],
            keywords=keywords,
            path=relative.with_suffix(".md").as_posix(),
            body=body,
        )
    _disambiguate_paths(records)
    aliases = {}
    for record in records.values():
        for alias in record["aliases"]:
            aliases.setdefault(alias, set()).add(record["path"])
    aliases = {key: sorted(value) for key, value in sorted(aliases.items())}
    authored = {key for key, record in records.items() if record["kind"] == "document"}
    for record in records.values():
        body = _markdown(record.pop("body"), record["path"], aliases, authored)
        output = destination / record["path"]
        output.parent.mkdir(parents=True, exist_ok=True)
        header = yaml.safe_dump(
            {
                "version": dc.__version__,
                **{key: record[key] for key in ("id", "title", "kind", "keywords")},
            },
            sort_keys=False,
        )
        output.write_text(
            f"---\n{header}---\n\n# {record['title']}\n\n{body}", encoding="utf-8"
        )
    return {
        "documents": list(records.values()),
        "aliases": aliases,
        "assets": [],
        "omitted": omitted,
    }
