"""
Test which enforces that DASCore keeps no changelog.

Release notes are assembled from the pull requests merged since the last tag,
each of which describes its own user-facing and breaking changes. The site's
changelog page is generated at build time from the GitHub releases, so no
changelog source belongs in the repository and this test fails if one appears.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CHECKER_PATH = _REPO_ROOT / ".github" / "scripts" / "check_pr_changelog.py"
_CONFIG_PATH = _REPO_ROOT / "great-docs.yml"

# The directories the docs are written in, read from the configuration so a new
# section is searched too, plus `docs` (assets the sections draw on) and the
# repository root, which is where a hand-written changelog would most likely
# land.
_SECTION_RE = re.compile(r"^\s+dir:\s*(\S+)\s*$", re.MULTILINE)

# Run wherever the docs are, skip where they are not. The sdist grafts tests but
# ships only docs/LICENSE, so a directory existing is not enough to tell the two
# apart; the landing page is present iff the real docs tree is.
_DOCS_PRESENT = (_REPO_ROOT / "index.qmd").is_file()

_POLICY = (
    "DASCore does not keep a changelog. Describe user-facing and breaking "
    "changes in the pull request that makes them, under the headings in "
    ".github/pull_request_template.md; they are collected into the release "
    "notes when a version is tagged. See .agents/agents.md. The published "
    "dascore.org/changelog.html page is generated from those releases by "
    "great-docs, so writing one here would collide with it."
)


def _doc_directories() -> list[Path]:
    """Return the repository root and every directory a section is built from."""
    config = _CONFIG_PATH.read_text(encoding="utf-8")
    dirs = {"docs", *_SECTION_RE.findall(config)}
    return [_REPO_ROOT, *(_REPO_ROOT / name for name in sorted(dirs))]


def _changelog_sources() -> list[Path]:
    """Return any file in the docs which is a changelog of its own."""
    out = []
    for directory in _doc_directories():
        # The root is searched shallowly (its subdirectories are the sections,
        # searched in their own right); a section, all the way down.
        paths = directory.glob("*") if directory == _REPO_ROOT else directory.rglob("*")
        for path in paths:
            name = path.name.lower()
            if name.startswith(("changelog", "change_log")):
                out.append(path)
    return sorted(out)


@pytest.mark.skipif(
    not _DOCS_PRESENT,
    reason="the docs are not present (e.g. running from an sdist)",
)
class TestNoChangelogSource:
    """The changelog must come from the releases, not from a file."""

    def test_no_changelog_page(self):
        """No changelog source file may exist in the docs or repo root."""
        found = _changelog_sources()
        assert not found, f"{[str(p) for p in found]} should not exist. {_POLICY}"

    def test_release_notes_are_configured(self):
        """
        The generated page needs the repository it draws releases from.

        Dropping `repo` from the configuration would silently take the
        changelog page with it, leaving dascore.org/changelog.html a 404.
        """
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        match = re.search(r'^repo:\s*"?(\S+?)"?\s*$', text, re.MULTILINE)
        repo = match.group(1) if match else ""
        assert "github.com" in repo, (
            f"great-docs.yml must set `repo:` to the GitHub repository the "
            f"changelog page is generated from, not {repo!r}. {_POLICY}"
        )


def _load_checker():
    """Import the PR body checker, which lives outside the package."""
    spec = importlib.util.spec_from_file_location("check_pr_changelog", _CHECKER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    """The changelog-section validator run by CI."""
    if not _CHECKER_PATH.is_file():
        pytest.skip("the checker script is not present")
    return _load_checker()


def _body(section: str) -> str:
    """Build a PR body whose Changelog section holds the given text."""
    return f"## Description\n\nSomething.\n\n## Changelog\n\n{section}\n"


class TestChangelogSectionChecker:
    """The CI check must accept conforming bodies and reject the rest."""

    @pytest.mark.parametrize(
        "section",
        [
            "- added: a new thing.",
            "- changed **breaking**: an old thing moved.",
            "- fixed: a wrong thing.\n- security: a scary thing.",
            "none",
            "None",
            "- none",
            "- None",
            "* none",
            "- deprecated: an aging thing.\n- removed: a gone thing.",
            "- Added: a capitalized category is a harmless slip.",
            "- Changed **breaking**: capitalized, with a marker.",
        ],
    )
    def test_accepts_valid_sections(self, checker, section):
        """Categories, the breaking marker, and 'none' are all accepted."""
        assert checker.validate(_body(section)) == []

    @pytest.mark.parametrize(
        "section",
        [
            "- a bullet with no category.",
            "- breaking: not a category.",
            "- added a new thing.",
            "- added:",
            "",
            "- none\n- added: a real change, so 'none' is a lie.",
            "- none of this is user facing.",
        ],
    )
    def test_rejects_invalid_sections(self, checker, section):
        """Malformed or empty sections are reported rather than passed."""
        assert checker.validate(_body(section))

    @pytest.mark.parametrize(
        "section",
        [
            "- changed: **breaking** the marker is in the text.",
            "- added: a thing, which is **breaking** for callers.",
            "- fixed: **BREAKING** shouting does not help either.",
        ],
    )
    def test_rejects_a_misplaced_breaking_marker(self, checker, section):
        """A marker in the text would pass while going unrecorded."""
        problems = checker.validate(_body(section))
        assert problems and "read as prose" in problems[0]

    def test_rejects_missing_section(self, checker):
        """A body with no Changelog heading fails."""
        problems = checker.validate("## Description\n\nSomething.\n")
        assert problems and "no '## Changelog' section" in problems[0]

    def test_rejects_empty_body(self, checker):
        """An empty body fails rather than passing vacuously."""
        assert checker.validate("")
        assert checker.validate(None)

    def test_instructions_do_not_count_as_entries(self, checker):
        """The template's commented guidance must not satisfy the check."""
        commented = "<!--\n- added: an example from the template.\n-->"
        assert checker.validate(_body(commented))

    @pytest.mark.parametrize("heading", ["## Checklist", "  ## Checklist"])
    def test_section_ends_at_the_next_heading(self, checker, heading):
        """Entries belonging to a later section are not read as changelog ones.

        The heading may be indented: several merged pull requests carry an
        indented template, and reading past it would take the checklist's
        boxes for malformed entries.
        """
        body = _body("none") + f"\n{heading}\n\n- [ ] did a thing.\n"
        assert checker.validate(body) == []
