# DASCore Agent Guide

This file gives AI/code agents a practical checklist for contributing safely to DASCore.

## Scope and priorities

1. Keep changes minimal, targeted, and test-backed.
2. Preserve DASCore conventions over personal preferences.
3. Prefer consistency with existing code/tests/docs in this repo.

## Development workflow

1. Work on a feature/fix branch, not `master`.
2. Keep commits focused (one logical change per commit where possible).
3. Use pull requests to merge to `master`.

## Environment setup

Follow `docs/contributing/dev_install.qmd`.

Typical setup:

```bash
git pull origin master --tags
pip install -e ".[dev]"
pre-commit install -f
```

## Linting and formatting

- Run pre-commit hooks before finalizing changes.
- Project lint/format is driven by pre-commit and Ruff config in `pyproject.toml`.

```bash
pre-commit run --all
```

Tip: running twice can apply auto-fixes on first pass.

## Testing requirements

Follow `docs/contributing/testing.qmd`.

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

## Test authoring conventions

- Put tests under `tests/` mirroring package structure.
- Group tests in classes.
- Place fixtures as close as practical to usage (class, module, then `conftest.py`).

## Code conventions


- Prefer `pathlib.Path` over raw path strings (except performance-sensitive bulk file workflows).
- Use snake_case dataframe column names when possible.
- Use `df["col"]` (getitem), not `df.col` (getattr).
- Prefer non-inplace dataframe operations unless inplace is explicitly required.
- Add type hints for public functions/methods.
- Use NumPy-style docstrings for public APIs.
- Keep comments meaningful; do not restate obvious code.

## Documentation changes

If behavior or API changes, update docs in the same PR.

- Documentation source lives in `docs/` (`.qmd` files).
- API docs are generated from docstrings.
- Build docs workflow (see `docs/contributing/documentation.qmd`):

```bash
python scripts/build_api_docs.py
quarto render docs
```

Important: if changing site structure, edit `scripts/_templates/_quarto.yml` (not `docs/_quarto.yml`, which is generated/overwritten).

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
