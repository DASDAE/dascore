"""
Test which enforces that DASCore keeps no changelog.

Release notes are assembled from the pull requests merged since the last tag,
each of which describes its own user-facing and breaking changes. The changelog
page survives only as a stub preserving its published URL, so this test pins its
contents exactly and fails if anything is added to it.
"""

from __future__ import annotations

# E501: the expected text below is a byte-for-byte copy of a markdown page whose
# prose is deliberately not hard-wrapped; rewrapping it would break the match.
# ruff: noqa: E501
import importlib.util
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DOC_PATH = _REPO_ROOT / "docs"
_CHANGELOG_PATH = _DOC_PATH / "changelog.qmd"
_CHECKER_PATH = _REPO_ROOT / ".github" / "scripts" / "check_pr_changelog.py"

# Run wherever the docs are, skip where they are not. The sdist grafts tests but
# ships only docs/LICENSE, so the directory existing is not enough to tell the
# two apart; index.qmd is present iff the real docs tree is. Deliberately not
# keyed on changelog.qmd, or deleting the page would skip rather than fail.
_DOCS_PRESENT = (_DOC_PATH / "index.qmd").is_file()

_EXPECTED = """# Changelog

DASCore's changelog is the [releases page](https://github.com/DASDAE/dascore/releases), which tracks changes from one version to another.

This page remains only so that existing links to it keep working; it is not part of the site navigation and is never updated per pull request. Each pull request describes its own user-facing and breaking changes, and those descriptions are collected into the release notes when a version is tagged. See [publish a new release](contributing/publish_a_new_release.qmd) for that workflow.
"""

_POLICY = (
    "DASCore does not keep a changelog. Describe user-facing and breaking "
    "changes in the pull request that makes them, under the headings in "
    ".github/pull_request_template.md; they are collected into the release "
    "notes when a version is tagged. See .agents/agents.md. The page itself "
    "must stay exactly as pinned here, since it is published as "
    "dascore.org/changelog.html."
)


@pytest.mark.skipif(
    not _DOCS_PRESENT,
    reason="the docs are not present (e.g. running from an sdist)",
)
class TestChangelogIsAStub:
    """The changelog page must stay a pointer to the releases page."""

    def test_contents_are_unchanged(self):
        """The page must match the expected stub exactly."""
        assert _CHANGELOG_PATH.exists(), f"{_CHANGELOG_PATH} is missing. {_POLICY}"
        contents = _CHANGELOG_PATH.read_text(encoding="utf-8")
        assert contents.strip() == _EXPECTED.strip(), _POLICY


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
            "- deprecated: an aging thing.\n- removed: a gone thing.",
        ],
    )
    def test_accepts_valid_sections(self, checker, section):
        """Categories, the breaking marker, and 'none' are all accepted."""
        assert checker.validate(_body(section)) == []

    @pytest.mark.parametrize(
        "section",
        [
            "- a bullet with no category.",
            "- Added: capitalized category.",
            "- breaking: not a category.",
            "- added a new thing.",
            "- added:",
            "",
        ],
    )
    def test_rejects_invalid_sections(self, checker, section):
        """Malformed or empty sections are reported rather than passed."""
        assert checker.validate(_body(section))

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

    def test_section_ends_at_the_next_heading(self, checker):
        """Entries belonging to a later section are not read as changelog ones."""
        body = _body("none") + "\n## Checklist\n\n- [ ] did a thing.\n"
        assert checker.validate(body) == []
