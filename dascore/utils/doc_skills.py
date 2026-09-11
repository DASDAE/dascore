"""Expose selected canonical recipes through one skill catalog."""

from __future__ import annotations

import re

import yaml

from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import (
    DOC_PATH,
    PACKAGE_PATH,
    DocumentationError,
    _frontmatter,
    source_paths,
)


def get_skills() -> dict[str, dict[str, str]]:
    """Read skill membership and each recipe's authored title and description."""
    catalog = yaml.safe_load((DOC_PATH / "skills.yml").read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        raise DocumentationError("The skill catalog must map names to document IDs.")
    pages = {
        path.relative_to(DOC_PATH).with_suffix("").as_posix(): path
        for path in source_paths()
    }
    out = {}
    for name, target in catalog.items():
        if not isinstance(name, str) or not re.fullmatch(
            r"[a-z0-9]+(?:-[a-z0-9]+)*", name
        ):
            raise DocumentationError(f"Invalid skill name: {name!r}.")
        if not isinstance(target, str) or target not in pages:
            raise DocumentationError(
                f"Skill {name!r} refers to an unknown document: {target!r}."
            )
        front, _body = _frontmatter(pages[target].read_text(encoding="utf-8"))
        if not all(
            isinstance(front.get(key), str) and front[key].strip()
            for key in ("title", "description")
        ):
            raise DocumentationError(
                f"Skill {name!r} needs a title and description in {target!r}."
            )
        out[name] = {
            "target": target,
            "title": front["title"],
            "description": front["description"],
        }
    return out


def read_skill(name: str) -> str:
    """Read a catalogued recipe from the current installation's Markdown corpus."""
    skills = get_skills()
    if name not in skills:
        raise DocumentationError(
            f"Unknown skill {name!r}. Available skills: {', '.join(skills)}."
        )
    with documentation_cache() as (root, manifest):
        return read_document(root, manifest, skills[name]["target"])


def read_bootstrap() -> str:
    """Return the native agent entry point verbatim for explicit installation."""
    return (PACKAGE_PATH / "skills" / "dascore" / "SKILL.md").read_text(
        encoding="utf-8"
    )
