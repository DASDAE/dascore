# DASCore

- Keep changes focused; follow nearby code and preserve unrelated work.
- For code changes, use an isolated worktree and environment. See `dascore/.agents/skills/start-task/SKILL.md` when setting up a task. Target PRs at `dev`.
- Skills ship with the package in `dascore/.agents/skills/`; read the one that matches the work, such as `add-format`, `make-namespace`, or `draft-release`.
- Read `.agents/agents.local.md` if present. For code, tests, or docs, read the relevant section of `.agents/conventions.md`.
- Before changing coordinate behavior, read `dascore/docs/notes/coordinates.qmd`. Changes conflicting with that contract need a separate note PR first.
- Run affected tests and `pre-commit run --files <changed-files>`. Repeat checks after auto-fixes. Broaden testing for shared behavior; CI runs the full suite. Report checks run and any gaps.
- Update docs for behavior or API changes. Put release notes in the PR's required `## Changelog` section, following `dascore/docs/contributing/general_guidelines.qmd`. Do not add changelog files or edit the generated changelog stub.
- Hand off with what changed, validation results, and remaining risks.
