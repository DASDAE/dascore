"""Maintain an installed version's local documentation cache."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from contextlib import contextmanager, suppress
from importlib import metadata
from pathlib import Path
from tempfile import mkdtemp

import dascore as dc
from dascore.utils.doc_corpus import (
    PACKAGE_PATH,
    SKILL_PATH,
    DocumentationError,
    source_paths,
    write_corpus,
)

_FORMAT_VERSION = 1
# Idle time after which a corpus or staging directory is presumed abandoned; long
# enough that agents keep a reported corpus directory for a working session.
_STALE_SECONDS = 24 * 3600


def _source_identity():
    """Fingerprint source content, including editable changes under one version."""
    digest = hashlib.sha256(str(PACKAGE_PATH).encode())
    paths = list(PACKAGE_PATH.glob("*.py"))
    for child in PACKAGE_PATH.iterdir():
        if child.is_dir() and child.name != "docs":
            paths.extend(child.rglob("*.py"))
    paths.extend(source_paths())
    paths.extend(SKILL_PATH.rglob("*.md"))
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


def _publish(root, identity, rebuild):
    """Build a corpus in private staging and rename it into place as `root`."""
    staging = Path(mkdtemp(prefix=".", dir=root.parent))
    try:
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
        # Vacate a damaged or rebuilt corpus; the sweep deletes it later.
        if root.exists() and (rebuild or _load_manifest(root, identity) is None):
            with suppress(FileNotFoundError):  # a concurrent rebuild moved it
                os.replace(root, staging.with_name(staging.name + "-old"))
        try:
            os.replace(staging, root)
        except OSError:
            # Another process published the same corpus first; use theirs.
            if (manifest := _load_manifest(root, identity)) is None:
                raise
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _sweep(parent):
    """Delete this version's corpora, staging, and old-layout files left unused."""
    cutoff = time.time() - _STALE_SECONDS
    for path in parent.iterdir():
        # Another process may delete or (on Windows) hold the same entry open.
        with suppress(FileNotFoundError, PermissionError):
            if path.lstat().st_mtime > cutoff:
                continue
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink()


@contextmanager
def documentation_cache(rebuild: bool = False):
    """
    Prepare the local Markdown corpus and yield its absolute path and manifest.

    Each source identity has its own corpus under
    `get_config().docs_cache_dir / dascore.__version__`. A corpus is built in
    private staging and published by renaming, so concurrent processes need no
    lock: a build that fails before publishing leaves the previous corpus in place,
    and the loser of a simultaneous first build uses the winner's corpus. Each use
    refreshes the corpus's modification time; corpora unused for a day are deleted.
    """
    parent = (dc.get_config().docs_cache_dir / dc.__version__).resolve()
    parent.mkdir(parents=True, exist_ok=True)
    identity = _source_identity()
    root = parent / identity["source"][:16]
    manifest = None if rebuild else _load_manifest(root, identity)
    if manifest is None:
        manifest = _publish(root, identity, rebuild)
    os.utime(root)
    _sweep(parent)
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
