---
name: start-task
description: DASCore development. Set up an isolated task worktree and environment when starting work on DASCore itself.
---

Read the repository's `AGENTS.md` and local instructions if present.

1. Find the repository root and choose a task slug containing only lowercase letters, digits, and hyphens. Reject an empty slug. Use it as the branch name unless the user supplies one.
2. Resolve the worktree path to `<repo-root>/worktrees/<slug>`. If it exists, reuse it only when attached to the intended branch. If that branch is checked out elsewhere, report its path rather than moving or deleting it. Leave unrelated changes alone.
3. For an existing branch, attach the worktree to it. For a new branch, use `origin/dev` as the base; refresh it when available. If it cannot be refreshed, report the local base used. If no dev ref exists, ask which base to use. Do not silently branch from the current task branch.
4. In the worktree, use a dedicated mamba environment named for the slug or a worktree-local uv environment. Install `.[dev]` with that environment's Python and install pre-commit hooks once. Keep editable installs isolated between tasks.
5. Verify `python -c "import dascore; print(dascore.__file__)"` resolves to this worktree. Report path, branch/base, environment, and sanity-check result.
