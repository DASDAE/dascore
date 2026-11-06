"""Expose selected canonical documents through one skill catalog."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import (
    DOC_PATH,
    SKILL_PATH,
    DocumentationError,
    _read_source,
    source_paths,
)

# Skills that must work where the project's DASCore has no command line (an
# older release, or no environment yet), so they are installed whole.
_STANDALONE = frozenset({"dascore", "migrate-to-v1", "start-task", "test-beta"})

# What any other installed skill says in place of its procedure, which would
# go stale.
_INSTALLED_BODY = (
    "\nThe procedure for this skill ships with DASCore and changes between "
    "versions. Run `<python> -m dascore skill {name}` with the project's "
    "Python, selected as the `dascore` skill describes, and follow what it "
    "prints.\n\n"
    "If the command reports an unknown skill, the project's DASCore predates "
    "it. Say so rather than working from memory.\n"
)


def get_skills() -> dict[str, dict[str, str]]:
    """Read skill membership, each page's title, and its skill's description."""
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
        front, _body = _read_source(pages[target])
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
    """Read a catalogued page from the current installation's Markdown corpus."""
    skills = get_skills()
    if name not in skills:
        raise DocumentationError(
            f"Unknown skill {name!r}. Available skills: {', '.join(skills)}."
        )
    with documentation_cache() as (root, manifest):
        return read_document(root, manifest, skills[name]["target"])


def install_skills(directory: Path | str) -> list[Path]:
    """
    Write an entry for every shipped skill into an agent's skill directory.

    A skill that must work without the project's DASCore is copied whole.
    Every other entry keeps the skill's name and description and tells the
    agent to read the current procedure from the project's DASCore, so it
    cannot go stale.
    """
    out = []
    for source in sorted(SKILL_PATH.glob("*/SKILL.md")):
        name = source.parent.name
        text = source.read_text(encoding="utf-8")
        if name not in _STANDALONE:
            front = text.split("\n---\n", 1)[0]
            text = f"{front}\n---\n{_INSTALLED_BODY.format(name=name)}"
        target = Path(directory).expanduser() / name / source.name
        # A directory linked to the package holds the skill files themselves.
        if not (target.exists() and target.samefile(source)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        out.append(target)
    return out


def read_bootstrap() -> str:
    """Return the native agent entry point verbatim for explicit installation."""
    return (SKILL_PATH / "dascore" / "SKILL.md").read_text(encoding="utf-8")
