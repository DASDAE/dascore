"""The DASCore command-line entry point."""

from __future__ import annotations

import sys

from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.doc_corpus import DocumentationError
from dascore.utils.misc import optional_import


def main(argv: list[str] | None = None) -> int:
    """Run the Typer CLI and return its exit status."""
    try:
        optional_import("typer", required_for="the DASCore command-line interface")
        # Command definitions depend on the optional Typer package.
        app = optional_import("dascore._cli").app

        return app(args=argv, prog_name="dascore")
    except MissingOptionalDependencyError as exc:
        sys.stderr.write(f"dascore: {exc}\n")
        return 1
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code
    except (DocumentationError, MissingOptionalDependencyError, OSError) as exc:
        sys.stderr.write(f"dascore: {exc}\n")
        return 1
