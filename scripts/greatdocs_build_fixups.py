"""
Quarto pre-render hook: patch great-docs-generated pages before rendering.

great-docs generates two pages whose content needs small post-processing that
the tool does not (yet) do itself. This script runs inside the great-docs build
directory (wired via the ``pre_render`` key of great-docs.yml), after the pages
are generated but before Quarto renders them.

1. ``index.qmd`` — great-docs builds the landing page from the repo readme and
   bumps every ``#`` heading one level (``#`` -> ``##``) so the page nests under
   its title. That regex also rewrites ``#`` comment lines *inside* fenced code
   cells, so ``# import ...`` renders as ``## import ...``. We undo the bump for
   ``##``-or-deeper lines that sit inside code fences.

2. ``changelog.qmd`` — each release section already gets a
   ``## {version} {.changelog-version}`` heading from great-docs, but the GitHub
   release body usually *also* starts with a ``# {version}`` heading. That
   duplicate renders as a stray paragraph or a repeated heading. We drop the
   leading body heading when its text matches the section's version.

Both fixes are written to be safe to run anywhere: the index fix only touches
``##``-or-deeper lines (repo-source comments are single ``#`` and are left
alone), and missing files are skipped.
"""

from __future__ import annotations

import re
from pathlib import Path

FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
# A code-comment line that great-docs over-bumped (two or more leading #).
BUMPED_COMMENT_RE = re.compile(r"^(#{2,})(\s)")
HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*?)\s*$")
CHANGELOG_VERSION_RE = re.compile(r"^#{1,6}\s+(.*?)\s*\{\.changelog-version\}\s*$")
DATE_LINE_RE = re.compile(r"^\s*\*.*\[GitHub\]")


def fix_index(path: Path) -> int:
    """De-bump ``#`` comment lines that great-docs doubled inside code fences."""
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").split("\n")
    in_fence = False
    marker = ""
    changed = 0
    for i, line in enumerate(lines):
        m = FENCE_RE.match(line)
        if m:
            mk = m.group(1)[0]
            if not in_fence:
                in_fence, marker = True, mk
            elif mk == marker:
                in_fence, marker = False, ""
            continue
        if in_fence:
            bm = BUMPED_COMMENT_RE.match(line)
            if bm:
                lines[i] = line[1:]  # drop one leading '#'
                changed += 1
    if changed:
        path.write_text("\n".join(lines), encoding="utf-8")
    return changed


def _fix_release_body(version: str, body: list[str]) -> tuple[list[str], int, int]:
    """Clean one release body: drop a duplicate version heading and dedent.

    Some GitHub release bodies are uniformly indented (~2 spaces). great-docs'
    ``.strip()`` un-indents only the *first* line of the body, so the rest stay
    indented and pandoc renders their ``### ...`` headings as literal text. We
    split off the date line, drop a leading heading that repeats the version,
    then strip the body's base indent from every line. The base indent is taken
    from the note lines *after* the anomalous first line, and we remove at most
    that many leading spaces per line so relative nesting (e.g. list-item
    continuations) is preserved.
    """
    dropped = deindented = 0
    # Keep the leading date line (and blanks before it) outside the dedent.
    head: list[str] = []
    idx = 0
    while idx < len(body):
        head.append(body[idx])
        if DATE_LINE_RE.match(body[idx]):
            idx += 1
            break
        idx += 1
    note = body[idx:]

    # Drop a leading heading that just repeats the version.
    j = 0
    while j < len(note) and note[j].strip() == "":
        j += 1
    if j < len(note):
        hm = HEADING_RE.match(note[j])
        if hm and hm.group(2).strip() == version:
            note = note[:j] + note[j + 1 :]
            dropped = 1

    # Base indent = the most common leading indent among non-blank lines. Using
    # the mode (not the min) is robust to the anomalous 0-indent lines these
    # bodies contain: the generator un-indents the first line, and GitHub's
    # auto-generated "**Full Changelog**" footer also sits at column 0.
    indents = [len(ln) - len(ln.lstrip(" ")) for ln in note if ln.strip()]
    base = 0
    if indents:
        counts: dict[int, int] = {}
        for ind in indents:
            counts[ind] = counts.get(ind, 0) + 1
        # Highest count wins; ties resolve to the smaller indent.
        base = min(counts, key=lambda k: (-counts[k], k))
    if base:
        strip_re = re.compile(rf"^ {{1,{base}}}")
        new_note = [strip_re.sub("", ln) for ln in note]
        if new_note != note:
            note = new_note
            deindented = 1
    return head + note, dropped, deindented


def fix_changelog(path: Path) -> int:
    """Drop duplicate version headings and dedent indented release bodies."""
    if not path.exists():
        return 0
    lines = path.read_text(encoding="utf-8").split("\n")

    # Find the frontmatter/preamble, then split into per-release blocks at each
    # `{.changelog-version}` heading.
    out: list[str] = []
    version: str | None = None
    block: list[str] = []
    dropped = deindented = 0

    def flush() -> None:
        nonlocal dropped, deindented
        if version is None:
            out.extend(block)
            return
        fixed, d, di = _fix_release_body(version, block)
        dropped += d
        deindented += di
        out.extend(fixed)

    for line in lines:
        vm = CHANGELOG_VERSION_RE.match(line)
        if vm:
            flush()
            out.append(line)
            version = vm.group(1).strip()
            block = []
        elif version is None:
            out.append(line)
        else:
            block.append(line)
    flush()

    if dropped or deindented:
        # Collapse runs of blank lines left behind to a single blank line.
        collapsed: list[str] = []
        for ln in out:
            if ln.strip() == "" and collapsed and collapsed[-1].strip() == "":
                continue
            collapsed.append(ln)
        path.write_text("\n".join(collapsed), encoding="utf-8")
    return dropped + deindented


def main() -> int:
    """Patch the generated index and changelog pages in place."""
    cwd = Path.cwd()
    n_index = fix_index(cwd / "index.qmd")
    n_changelog = fix_changelog(cwd / "changelog.qmd")
    print(
        f"greatdocs_build_fixups: de-bumped {n_index} code-comment line(s) in "
        f"index.qmd, dropped {n_changelog} duplicate changelog heading(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
