# Development Guidelines

## Style

- Every public function should have:
  - a full numpy style docstring, complete with a parameters, return, and examples section.
    - Don't include type hints in the docstring; they are already on the function signature.
  - Type hints for each parameter.
  
- Every private function should have a line or two explaining what it does.
  - More than this is overkill.
  
- Comments should be used liberally, while avoiding redundant use. Generally, every 3-7 lines of code should have a comment.

- Always run `pre-commit run --all-files` before completing a task.

## Testing

- Tests should be grouped into classes named after the function/method they are testing.

- The testing structure should mirror the project structure (eg tests for derzug/util/misc.py for in tests/util/misc.py)

- Write meaningful tests that focus on boundaries, not implementation details.

- Test names should not be more than 5 or 7 words. Add more detail in the docstrings.

- Ensure any setup for tests that requires more than 3 lines is put into a fixture. Put these fixtures as close to the tests as possible, moving them further away (eg from class to module, module to conftest) when more tests need the fixture. Stay DRY.

- Preserve the split between headless and interactive Qt test runs:
  - plain `pytest` should remain headless-safe via `QT_QPA_PLATFORM=offscreen`
  - `pytest -m show` should remain interactive, run only the `show` tests, and keep windows open until the developer closes them
