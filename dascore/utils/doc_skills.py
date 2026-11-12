"""Serve the shipped skills, each through the docs page that includes it."""

from __future__ import annotations

import re
from pathlib import Path

from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import (
    DOC_PATH,
    SKILL_PATH,
    DocumentationError,
    _frontmatter,
    _read_source,
    source_paths,
)

# Skills that must work where the project's DASCore has no command line (an
# older release, or no environment yet), so they are installed whole.
_STANDALONE = frozenset(
    {"dascore", "dascore-migrate-to-v1", "dascore-start-task", "dascore-test-beta"}
)

# What any other installed skill says in place of its procedure, which would
# go stale.
_INSTALLED_BODY = (
    "\nThe procedure for this skill ships with DASCore and changes between "
    "versions. Run `<python> -m dascore skill {name}` with the project's "
    "Python, selected as the `dascore` skill describes, and follow what it "
    "prints.\n\n"
    "If the command is missing or reports an unknown skill, the project's "
    "DASCore predates this skill. Say so rather than working from memory.\n"
)


# A page serves a skill by including its file.
_SKILL_INCLUDE = re.compile(
    r"\{\{<\s*include\s+\S*\.agents/skills/([^/\s]+)/SKILL\.md\s*>\}\}"
)


def _read_skill_file(name: str) -> tuple[dict, str]:
    """Return a shipped skill file's metadata and body, its title from its heading."""
    path = SKILL_PATH / name / "SKILL.md"
    if not path.is_file():
        raise DocumentationError(f"No skill file for {name!r}.")
    front, body = _read_source(path)
    heading = re.search(r"^# (.+)$", body, flags=re.MULTILINE)
    return {**front, "title": heading[1].strip() if heading else None}, body


def _skill_pages() -> dict[str, str]:
    """
    Map each skill to the docs page serving it.

    A page serves the skill whose file it includes, or the one its front
    matter names under `skill`, for a page holding a skill's procedure in its
    own text.
    """
    out = {}
    for path in source_paths():
        text = path.read_text(encoding="utf-8")
        claimed = set(_SKILL_INCLUDE.findall(text))
        if named := _frontmatter(text)[0].get("skill"):
            claimed.add(named)
        doc_id = path.relative_to(DOC_PATH).with_suffix("").as_posix()
        for name in claimed:
            if out.setdefault(name, doc_id) != doc_id:
                raise DocumentationError(
                    f"Skill {name!r} is served by both {out[name]!r} and {doc_id!r}."
                )
    return out


def get_skills() -> dict[str, dict[str, str | None]]:
    """
    Read each shipped skill's page, title, and description.

    Every folder of the shipped skills is a skill, described by its file. Its
    title comes from the page serving it, or, for a skill no page serves,
    from the file's first heading; that file is then served as written.
    """
    names = sorted(x.parent.name for x in SKILL_PATH.glob("*/SKILL.md"))
    pages = _skill_pages()
    if unknown := set(pages) - set(names):
        raise DocumentationError(f"Pages serve unknown skills: {sorted(unknown)}.")
    out = {}
    for name in names:
        if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
            raise DocumentationError(f"Invalid skill name: {name!r}.")
        front, _body = _read_skill_file(name)
        target = pages.get(name)
        if target is not None:
            page = (DOC_PATH / f"{target}.qmd").read_text(encoding="utf-8")
            front["title"] = _frontmatter(page)[0].get("title")
        if not all(
            isinstance(front.get(key), str) and front[key].strip()
            for key in ("title", "description")
        ):
            source = f"{target}.qmd" if target else f"{name}/SKILL.md"
            raise DocumentationError(
                f"Skill {name!r} needs a title and description in {source!r}."
            )
        out[name] = {
            "target": target,
            "title": front["title"],
            "description": front["description"],
        }
    return out


def read_skill(name: str) -> str:
    """Read a skill's page from the Markdown corpus, or a page-less skill file."""
    skills = get_skills()
    if name not in skills:
        raise DocumentationError(
            f"Unknown skill {name!r}. Available skills: {', '.join(skills)}."
        )
    target = skills[name]["target"]
    if target is None:
        return _read_skill_file(name)[1]
    with documentation_cache() as (root, manifest):
        return read_document(root, manifest, target)


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
        # Replace a link rather than write through it: it may point into a
        # package, whose procedure the short entry would then overwrite.
        for link in (target.parent, target):
            if link.is_symlink():
                link.unlink()
        # The same holds when the directory is this package's own skills.
        if not (target.exists() and target.samefile(source)):
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        out.append(target)
    return out


def read_bootstrap() -> str:
    """Return the native agent entry point verbatim for explicit installation."""
    return (SKILL_PATH / "dascore" / "SKILL.md").read_text(encoding="utf-8")
