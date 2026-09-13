"""Maintain an installed version's local documentation cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory

from filelock import FileLock

import dascore as dc
from dascore.utils.doc_corpus import (
    PACKAGE_PATH,
    DocumentationError,
    _read_readme,
    source_paths,
    write_corpus,
)

_FORMAT_VERSION = 1


def _source_identity():
    """Fingerprint source content, including editable changes under one version."""
    digest = hashlib.sha256(str(PACKAGE_PATH).encode())
    paths = list(PACKAGE_PATH.glob("*.py"))
    for child in PACKAGE_PATH.iterdir():
        if child.is_dir() and child.name != "docs":
            paths.extend(child.rglob("*.py"))
    paths.extend(source_paths())
    digest.update(_read_readme().encode())
    for entry in sorted(
        metadata.distribution("dascore").entry_points,
        key=lambda x: (x.group, x.name, x.value),
    ):
        digest.update(f"{entry.group}:{entry.name}:{entry.value}".encode())
    for path in sorted(paths):
        digest.update(str(path).encode())
        digest.update(path.read_bytes())
    return {
        "version": dc.__version__,
        "format": _FORMAT_VERSION,
        "source": digest.hexdigest(),
    }


def _load_manifest(root, identity):
    """Return a complete matching manifest, or None when rebuilding is needed."""
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if manifest["identity"] != identity:
            return None
        paths = [record["path"] for record in manifest["documents"]] + manifest[
            "assets"
        ]
        if not all((root / path).is_file() for path in paths):
            return None
        return manifest
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return None


@contextmanager
def documentation_cache(rebuild: bool = False):
    """
    Prepare the local Markdown corpus and hold its process lock while reading.

    The cache root is `get_config().docs_cache_dir / dascore.__version__`.
    Yield the corpus path and manifest. A failed rebuild retains the preceding
    complete corpus; source changes in editable installs invalidate the cache.
    """
    root = dc.get_config().docs_cache_dir / dc.__version__
    root.parent.mkdir(parents=True, exist_ok=True)
    backup = root.with_name(root.name + ".previous")
    with FileLock(str(root.with_name(root.name + ".lock")), timeout=60):
        # Recover a process interrupted between moving the old and new corpus.
        if backup.exists() and not root.exists():
            os.replace(backup, root)
        # No other builder for this version can own staging files under the lock.
        for abandoned in root.parent.glob(f".{root.name}-*"):
            if abandoned.is_dir():
                shutil.rmtree(abandoned)
        identity = _source_identity()
        manifest = None if rebuild else _load_manifest(root, identity)
        if manifest is None:
            with TemporaryDirectory(prefix=f".{root.name}-", dir=root.parent) as temp:
                staging = Path(temp)
                manifest = write_corpus(staging)
                if identity != _source_identity():
                    raise DocumentationError(
                        "DASCore sources changed during the documentation build; "
                        "retry the command."
                    )
                manifest["identity"] = identity
                (staging / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )
                if backup.exists():
                    shutil.rmtree(backup)
                if root.exists():
                    os.replace(root, backup)
                try:
                    os.replace(staging, root)
                except OSError:
                    if backup.exists():
                        os.replace(backup, root)
                    raise
        if backup.exists():
            shutil.rmtree(backup)
        yield root, manifest


def read_document(root: Path, manifest: dict, target: str) -> str:
    """Read an exact documented name or raise an actionable lookup error."""
    parts = target.split(".")
    hosts = {
        "Patch": "patch",
        "Spool": "spool",
        "BaseSpool": "spool",
        "Inventory": "inventory",
        "AnnotationSet": "annotation",
    }
    for i, part in enumerate(parts[:-1]):
        if part in hosts:
            group = f"dascore.{hosts[part]}_namespace"
            providers = [
                e for e in metadata.entry_points(group=group) if e.name == parts[i + 1]
            ]
            if len(providers) > 1:
                names = ", ".join(e.value for e in providers)
                raise DocumentationError(
                    f"Ambiguous namespace {parts[i + 1]!r}: {names}"
                )
    paths = manifest["aliases"].get(target, [])
    if not paths:
        raise DocumentationError(
            f"No local documentation for {target!r}. Use a public API name such as "
            "Patch.select or a page identifier such as tutorial/patch."
        )
    if len(paths) != 1:
        choices = ", ".join(
            record["id"] for record in manifest["documents"] if record["path"] in paths
        )
        raise DocumentationError(
            f"Ambiguous documentation name {target!r}; use one of: {choices}"
        )
    return (root / paths[0]).read_text(encoding="utf-8")
