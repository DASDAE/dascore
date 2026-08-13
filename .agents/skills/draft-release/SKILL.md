---
name: draft-release
description: Draft the next release version and changelog by fetching tags, computing the next semantic v* tag, collecting merged PRs into the release branch (usually dev) since last release via gh, and printing the proposed version and categorized changelog.
---

# draft-release

Draft the next release version and changelog from merged PRs.

Input: `release_type`, one of `major`, `minor`, `patch`/`bugfix` (equivalent). Default `bugfix`.

## Workflow

1. Ask for network permission (`git fetch`, `gh`), then `git fetch --tags origin`.
   Not `--all`: unrelated remotes can fail or clobber tags, aborting the fetch.
   To read tags without touching local state, `git ls-remote --tags origin`.

2. Next version. Among tags matching `^v[0-9]+\.[0-9]+\.[0-9]+$` (base `v0.0.0`
   if none), take the latest and bump per `release_type`: `X+1.0.0`, `X.Y+1.0`,
   or `X.Y.Z+1`. Pre-releases break this: if the target branch carries a newer
   `aN`/`bN`/`rcN` tag, use *it* as step 3's lower bound (or the notes repeat what
   it published) and continue that series (`v0.2.0b1` → `v0.2.0b2`). Ask before
   finalizing to a stable tag instead.

3. Scope PRs to `last_release_tag..origin/<target>`, where `<target>` is the
   branch being released; ask if not given. It is **not** always `master` — work
   lands on `dev`, so scoping to `master` silently omits everything not merged
   down. Compare `git rev-list --count` for both first. Read them with `gh`,
   keeping number, title, merge date, labels, and body; consult diffs when
   the body is thin.

4. Drop reverted pairs. When a PR in scope reverts another in scope, omit both
   and note them as `Reverted (no net change)` so the missing numbers are
   explained.

5. Collect entries. Each PR body has a `## Changelog` section, already written
   and categorized by its author per "Changelog entries" in
   `docs/contributing/general_guidelines.qmd` and enforced by
   `.github/scripts/check_pr_changelog.py`. Read those bullets rather than
   re-deriving them: each becomes one entry under the section its category
   names, and every `**breaking**` entry is also listed under `Breaking Changes`.
   Preserve the author's wording. Never add the marker yourself — it means
   "breaks the last *released* version", so a drastic-looking change may break
   nothing any user has.

   Reuse `extract_section` and `validate` from that script rather than writing
   a second parser: a body may indent the heading that ends the section, and
   `none` may be bulleted, so a hand-rolled one reads a later section's
   checklist as malformed entries.

   A literal `none` is trustworthy; an empty or absent section means the PR
   predates the policy, so classify it yourself from title, body, and diff.
   Labels here are topical (`proc`, `spool`, `IO`) rather than semantic: `Added`
   for a new capability, `Fixed` for corrected behavior, `Changed` for anything
   else observable including performance. Check the tag before marking anything
   breaking — an API introduced after it cannot break anyone.

   See "Draft the release notes" in `docs/contributing/publish_a_new_release.qmd`
   for the sections to use, and their order.

6. Print the new tag and the changelog. Split a PR's unrelated changes into
   separate entries, omit PRs with no user-facing effect rather than letting them
   fall through to `Fixed`, and sort each section by PR number ascending.

## Output Format

One section per entry category, in this order: `Breaking Changes`, `Added`,
`Changed`, `Deprecated`, `Removed`, `Fixed`, `Security`.

```text
Next Version: vX.Y.Z

## Breaking Changes
- Short summary (#125)

## Added
- Short summary (#126)

## Changed
- Short summary (#125)

## Fixed
- Short summary (#127)

Reverted (no net change): #128 reverted by #129
```

Each entry ends with its pull request in parentheses, which GitHub renders as a
link; do not write the full URL. A `**breaking**` entry appears twice, as #125
does. Omit empty sections and the `Reverted` line when unused. If no PRs are in
scope, print the version and say so.
