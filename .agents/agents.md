# DASCore Agent Guide

This file gives AI/code agents a practical checklist for contributing safely to DASCore.

## User local specific instructions
Also load .agents/agents.local.md if present.

## Scope and priorities

1. Keep changes minimal, targeted, and test-backed.
2. Preserve DASCore conventions over personal preferences.
3. Prefer consistency with existing code/tests/docs in this repo.

## Development workflow

1. Work on a feature/fix branch, not `master`.
2. Create task worktrees under the repository root at `worktrees/{slug}`. Do not create task worktrees under `.agents/worktrees`, even if the current shell starts there.
3. Keep commits focused (one logical change per commit where possible).
4. Open pull requests against `dev`, the integration branch, not `master`. `dev` is merged to `master` at release time.

## Environment setup

Use an environment named after the current worktree slug. If it is unavailable, create it with mamba or uv before running checks. This keeps editable installs isolated between worktrees.

Typical setup:

```bash
git pull origin master --tags
pip install -e ".[dev]"
pre-commit install -f
```

## Linting and formatting

- Run pre-commit hooks before finalizing changes.

```bash
pre-commit run --all
```

Tip: running twice can apply auto-fixes on first pass.

## Testing requirements

Run targeted tests for changed behavior, then broader tests as needed:

```bash
pytest tests/path/to/affected_test.py
pytest tests
```

For coverage checks:

```bash
pytest tests --cov dascore --cov-report term-missing
```

For doctests:

```bash
pytest dascore --doctest-modules
```

Unless otherwise specified, a job is not finished until the tests pass.

## Test authoring conventions

- Put tests under `tests/` mirroring package structure.
- Group tests in classes.
- Place fixtures as close as practical to usage (class, module, then `conftest.py`).
- Write tests that focus on boundaries, not implementation details.
- Keep test names short; put extra detail in the docstring when needed.


## Code conventions

- Import stdlib modules and required dependencies at module top, never inside functions or methods (tests included). Function-level imports are reserved for optional dependencies (use `dascore.utils.misc.optional_import`) and genuine circular-import avoidance; when used, a brief comment should say which.
- For dataframes, use snake_case column names and access via getitem, not getattr.
- Prefer non-inplace dataframe operations unless inplace is explicitly required.
- Add type hints for public functions/methods.
- Use NumPy-style docstrings for public APIs. Strive for short, informative example sections.
- Add a short explanatory docstring for private objects.
- Add a short explanatory comment for private helpers when intent is not obvious.
- Keep comments meaningful; do not restate obvious code.
- Always use `dascore.utils.misc.suppress_warnings` to suppress warnings.

## Documentation changes

If behavior or API changes, update docs in the same PR.

- Documentation source lives in `docs/` (`.qmd` files).
- API docs are generated from docstrings.
- Build docs workflow:
- Don't add newlines to markdown prose; let editors wrap.

```bash
python scripts/build_api_docs.py
quarto render docs
```

Important: if changing site structure, edit `scripts/_templates/_quarto.yml` (not `docs/_quarto.yml`, which is generated/overwritten).

### Changelog

DASCore does not maintain a changelog in the repository. Do not create one — not `CHANGELOG.md`, not a `changelog.d/` fragment directory — and do not add "unreleased changes" sections to any page. Release notes are written at release time from the pull requests merged since the last tag (see the `draft-release` skill), so a checked-in changelog is a second copy of that information which conflicts on every PR and drifts from what shipped. `docs/changelog.qmd` still exists, but only as a stub preserving a published URL: it points at the releases page and must never accumulate entries again. This is enforced, not merely conventional — `tests/test_changelog.py` fails if the page gains a list item, a section, or much prose.

Put the user-facing summary in the pull request description instead, under its `User-facing changes` and `Breaking changes` headings. Write for the person reading the release notes: what changed for them and what they should do about it, a sentence or two per item, with the reasoning and implementation detail left to the diff.

## Quality bar for agent changes

Before handing off:

1. Code compiles/runs for changed paths.
2. Relevant tests pass locally.
3. Lint/format checks pass.
4. Docs updated for user-visible behavior changes.
5. No unrelated refactors bundled with bug fixes.

## When uncertain

- Prefer existing patterns in nearby DASCore modules/tests.
- Call out assumptions explicitly in PR notes.
- Choose the simpler behavior-preserving implementation first.
