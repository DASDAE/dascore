# Import Speedup Plan

## The issue

`import dascore` currently takes ~1.2–2.1s (measured with `python -X importtime` on this machine). The root cause is that `dascore/core/patch.py` binds processing, transform, and viz functions onto the `Patch` class at class-definition time:

```python
select = dascore.proc.select
dft = transform.dft
# ... ~80 more bindings
```

Because these bindings execute when `dascore.core.patch` is imported (which `dascore/__init__.py` does unconditionally), importing the package transitively imports the entire processing stack before any user code runs. The dominant costs from the importtime profile:

| Import | Approx. cost | Pulled in by |
| --- | --- | --- |
| `scipy.signal` (+ `scipy.stats`) | ~280 ms | `dascore.transform.taup`, `dascore.compat`, filters |
| `matplotlib.pyplot` | ~230 ms | `dascore.viz` (also `proc/filter.py`, `proc/basic.py`, `transform/kurtosis.py`) |
| `numba` | ~270 ms (incl. children) | `dascore.transform.taup` via `utils.jit` |
| `pandas` | ~260 ms | `dascore.constants` (unavoidable, hard dependency) |
| `dascore.utils.array`, `utils.moving`, etc. | ~100+ ms | `dascore.proc.aggregate` |

This matters most for DASCore's core audience: HPC/batch workflows that launch many short-lived Python processes, and CLI-style scripts where a 2-second import dwarfs the actual work.

## Goals

1. Cut cold `import dascore` time substantially (target: under ~500 ms, ideally pandas/pydantic-bound).
2. No public API changes: `patch.select(...)`, `patch.viz.waterfall()`, `dc.spool(...)`, tab completion, and docs generation must all keep working.
3. Keep static-analysis friendliness (IDE autocompletion, type checkers).
4. Add a regression guard so import time doesn't silently creep back up.

## Plan

### Phase 1: Defer matplotlib by making `dascore.viz` lazy — DONE

Correction to the original analysis: the apparent module-level matplotlib
imports in `proc/filter.py`, `proc/basic.py`, and `transform/kurtosis.py` were
doctest examples inside docstrings, not real imports. Matplotlib entered the
eager import chain through exactly one line: `import dascore.viz` in
`dascore/core/patch.py`. The `Patch.viz` namespace already resolves lazily via
the `dascore.patch_namespace` entry point (`utils/namespace.py` +
`utils/plugins.py`), so the eager import was unnecessary.

Changes made:

- Removed `import dascore.viz` from `dascore/core/patch.py`.
- Added a PEP 562 module `__getattr__` to `dascore/__init__.py` so
  `dascore.viz` still works as a (now lazy) package attribute.
- Added `tests/test_imports.py` asserting matplotlib is absent from
  `sys.modules` after `import dascore`, and that `dascore.viz` still resolves.

Result: cold import dropped from ~1.25s to ~1.0s on the dev machine;
matplotlib (~250 ms) is now only imported on first viz use. Full test suite
(6075 passed) and doctests pass.

### Phase 2: Lazy method binding on `Patch` (the big win)

Replace direct bindings like `select = dascore.proc.select` with a small descriptor that resolves on first access:

```python
class _LazyPatchMethod:
    """Descriptor binding a patch function from a module on first access."""
    def __init__(self, module: str, name: str | None = None): ...
    def __set_name__(self, owner, name): ...
    def __get__(self, instance, owner):
        func = getattr(import_module(self.module), self.name)
        setattr(owner, self.name, func)  # cache: replace descriptor with real func
        return func.__get__(instance, owner) if instance is not None else func
```

Then `core/patch.py` no longer imports `dascore.proc` / `dascore.transform` at module scope. Details to handle:

- `PatchUFunc` instances (`Patch.add`, `Patch.exp`, ...) — cheap (numpy only); can stay eager via `utils.array` if that module's own imports are trimmed, otherwise same treatment.
- Dunder operators (`__add__`, etc.) call `dascore.utils.array.apply_ufunc`; keep those as thin lazy calls.
- `dascore/proc/__init__.py` and `dascore/transform/__init__.py` should themselves use PEP 562 `__getattr__` so importing the subpackage doesn't import every submodule (e.g. `transform.taup` pulling numba + scipy.signal).
- `dascore/__init__.py` itself re-exports `read/scan/write/get_format` etc. Audit whether `io.core` can stay eager (it imports pydantic/pandas, already needed) — likely fine.

### Phase 3: Preserve static typing and introspection

Lazy binding breaks "go to definition" and stub-less type checking. Mitigations:

- Generate a `Patch.pyi`-style stub (or a `TYPE_CHECKING` block in `patch.py` with the original `name = module.func` assignments) so IDEs and mypy see the real signatures. A `TYPE_CHECKING` block is simplest and keeps everything in one file:

  ```python
  if TYPE_CHECKING:
      select = dascore.proc.select
      ...
  ```

- Confirm `scripts/build_api_docs.py` still discovers all methods (it may need to force-resolve the descriptors; add a helper that materializes all lazy attributes).
- Confirm `dir(Patch)` / rich repr / pickling of patches are unaffected (descriptors live on the class, instances are unchanged).

### Phase 4: Regression guard + measurement

- Extend `tests/test_imports.py` (added in Phase 1 for matplotlib) to also assert `numba` and `scipy.signal` are absent from `sys.modules` after `import dascore` once Phase 2 lands. The `sys.modules` check is more stable than a wall-clock threshold.
- Record before/after numbers in the PR description using `python -X importtime`.

## Risks / open questions

- **Entry-point namespaces**: `Patch` and `BaseSpool` support third-party namespaces via `dascore.patch_namespace` entry points (`utils/namespace.py`). Lazy loading must not change when those load (they already load on attribute access — good precedent).
- **`warnings.filterwarnings` in `__init__`** relies on module import order in a few tests; verify warning-related tests still pass.
- **Doctest collection** (`pytest dascore --doctest-modules`) imports every module anyway, so doctests are unaffected; but ensure descriptor docstrings don't shadow function docstrings before resolution (resolve in `__getattr__`/`__get__` for `Patch.__doc__` access patterns used by the docs build).
- **Pickled spools/patches** referencing bound methods: patches don't pickle bound methods today (data/coords/attrs only), but double-check `spool.map` with process pools.

## Validation checklist

1. `pytest tests` passes.
2. `pytest dascore --doctest-modules` passes.
3. `python scripts/build_api_docs.py && quarto render docs` produces identical API pages.
4. `python -X importtime -c "import dascore"`: matplotlib, numba, and scipy.signal absent; total time recorded.
5. Tab completion on a `Patch` instance in IPython shows the same method set as before.
