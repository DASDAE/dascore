---
name: start-task
description: Instructions for working on a task in this repo.
---

# start-task

Instructions for starting work on a task in this repo in an isolated git worktree.

## Inputs

- `task_name`: A short human-readable task name.

## Workflow

1. Load and follow canonical agent instructions from `.agents/agents.md`.
2. Determine the repository root and repo name.
3. Normalize `task_name` into a safe slug.
   - Lowercase.
   - Replace spaces and underscores with `-`.
   - Remove characters that are not letters, numbers, `-`, or `/`.
   - Collapse repeated `-`.
   - Trim leading/trailing `-`.
4. Use branch name `{slug}` unless the user explicitly requested another branch naming convention.
5. Create the worktree at `.agents/worktrees/{slug}`.
6. Preflight checks before creating anything:
   - Confirm the repo root exists and is a git repository.
   - If `.agents/worktrees/{slug}` already exists, reuse it if it is attached to the intended branch; otherwise stop and report the conflict.
   - If branch `{slug}` already exists, create the worktree from that branch instead of failing.
7. Create the worktree:
   - If the branch does not exist: create it from the current default working branch.
   - If the branch exists: attach the worktree to that branch.
8. Change into the new worktree directory.
9. Activate the matching mamba environment with `mamba activate {repo_name}`.
10. Run one lightweight sanity check before starting work:
   - `python -c "import {repo_name}"`
   - If the package import name differs from the repo name, use the package name used by the repo.
11. Report:
   - worktree path
   - branch name
   - activated environment
   - whether the sanity check passed

## Notes

- Use `.agents/worktrees/` consistently. 
- Do not modify or delete existing worktrees unless the user explicitly asks.
- If the repo already has unrelated uncommitted changes in the main worktree, leave them alone; creating a separate worktree is the isolation mechanism.
