"""Prepare local Markdown from installed documentation and public APIs."""

from __future__ import annotations

import inspect
import os
import re
from collections import defaultdict
from importlib import import_module, metadata
from pathlib import Path
from types import ModuleType
from urllib.parse import unquote, urlsplit, urlunsplit

import yaml
from pydantic import BaseModel

import dascore as dc

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
_EXCLUDED = {"api", "filters", "_site", "site_libs", "lite", "lite_contents"}
_LINK = re.compile(r"(!?\[[^\n]*?\])\(([^\s)]+)\)")
_INCLUDE = re.compile(r"\{\{<\s*include\s+([^>]+?)\s*>\}\}")


class DocumentationError(ValueError):
    """An installed documentation source or lookup could not be resolved."""


def source_paths() -> list[Path]:
    """Return authored documents and assets, pruning generated directories."""
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
            if path.suffix in {".qmd", ".md", ".png", ".svg", ".jpg", ".jpeg"}:
                out.append(path)
    return sorted(out)


def _unwrap(obj):
    """Inspect the callable behind a method or decorator without binding it."""
    if isinstance(obj, (classmethod, staticmethod)):
        obj = obj.__func__
    if isinstance(obj, property):
        obj = obj.fget
    if not (
        inspect.ismodule(obj)
        or inspect.isclass(obj)
        or inspect.isfunction(obj)
        or inspect.ismethod(obj)
    ):
        return None
    return inspect.unwrap(obj)


def _api_documents():
    """Collect owned objects and aliases from the supported public modules."""
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
                "dascore." + ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
            )
    records, seen, omitted = {}, set(), []

    def add(obj, alias, owner=None):
        is_property = isinstance(obj, property)
        obj = _unwrap(obj)
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
        ):
            return
        key = module if inspect.ismodule(obj) else f"{module}.{obj.__qualname__}"
        if key not in records:
            try:
                signature = (
                    "" if is_property else str(inspect.signature(obj, eval_str=False))
                )
            except (TypeError, ValueError):
                signature = ""
            body = inspect.getdoc(obj) or "No docstring is available."
            prefix = f"Defined in: `{module}`\n\n"
            if is_property:
                prefix += (
                    "Property: access this as an attribute, without calling it.\n\n"
                )
            if signature:
                prefix += f"```python\n{obj.__name__}{signature}\n```\n\n"
            if owner is not None and signature:
                prefix += (
                    "Method signatures are unbound; "
                    "the instance supplies the first parameter.\n\n"
                )
            records[key] = dict(
                id=key,
                title=key,
                kind="api",
                aliases=[],
                keywords=[],
                path="api/" + key.replace(".", "/") + ".md",
                body=prefix + body,
            )
        records[key]["aliases"].append(alias)
        # Visit each class under each public alias, without evaluating properties.
        visit = (id(obj), alias)
        if not inspect.isclass(obj) or visit in seen:
            return
        seen.add(visit)
        for name, member in inspect.getmembers_static(obj):
            if not name.startswith("_") and not inspect.isclass(member):
                add(member, f"{alias}.{name}", obj)
        if issubclass(obj, BaseModel):
            for name, field in obj.model_fields.items():
                field_key = f"{key}.{name}"
                record = records.setdefault(
                    field_key,
                    dict(
                        id=field_key,
                        title=field_key,
                        kind="api",
                        aliases=[],
                        keywords=[],
                        path="api/" + field_key.replace(".", "/") + ".md",
                        body=(
                            f"Defined in: `{module}`\n\nType: `{field.annotation}`\n\n"
                            f"{field.description or 'No field description.'}"
                        ),
                    ),
                )
                record["aliases"].append(f"{alias}.{name}")

    for name in sorted(modules):
        try:
            module = import_module(name)
        except ModuleNotFoundError as exc:
            if (exc.name or "").startswith("dascore"):
                raise
            omitted.append(f"{name}: missing {exc.name}")
            continue
        add(module, name)
        for member_name, member in sorted(vars(module).items()):
            if not member_name.startswith("_"):
                add(member, f"{name}.{member_name}")
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
                    add(namespace, f"{prefix}.{entry.name}")
    for record in records.values():
        aliases = set(record["aliases"]) | {record["id"]}
        aliases |= {x.removeprefix("dascore.") for x in aliases}
        record["aliases"] = sorted(aliases)
    return records, omitted


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


def _markdown(text, path, aliases, authored):
    """Normalize executable fences and links while keeping example code intact."""

    def link(match):
        label, target = match.groups()
        reference = unquote(target).strip("`")
        if reference in aliases and len(aliases[reference]) == 1:
            dest = aliases[reference][0]
            relative = os.path.relpath(dest, Path(path).parent).replace(os.sep, "/")
            return f"{label}({relative})"
        if target.startswith("`"):
            # Unsupported optional APIs have an explicit route to the site index.
            return f"{label}(https://dascore.org/api/dascore.html)"
        url = urlsplit(target)
        if url.scheme or url.netloc or not url.path:
            return match[0]
        source = Path(os.path.normpath(str(Path(path).parent / url.path)))
        if url.path.startswith("/"):
            source = Path(url.path.lstrip("/"))
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
    assets = []
    for source in source_paths():
        relative = source.relative_to(DOC_PATH)
        if source.suffix not in {".qmd", ".md"}:
            output = destination / relative
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(source.read_bytes())
            assets.append(relative.as_posix())
            continue
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
        "assets": assets,
        "omitted": omitted,
    }
