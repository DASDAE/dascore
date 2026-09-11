"""The DASCore command-line interface."""

from __future__ import annotations

import sys

from dascore.exceptions import MissingOptionalDependencyError
from dascore.utils.misc import optional_import


def main(argv: list[str] | None = None) -> int:
    """Run the optional DASCore CLI and return its exit status."""
    try:
        optional_import("typer", required_for="the DASCore command-line interface")
    except MissingOptionalDependencyError as exc:
        sys.stderr.write(
            f"dascore: {exc}\nInstall CLI support with:\n"
            f'  "{sys.executable}" -m pip install "dascore[agents]"\n'
        )
        return 1
    # The command definitions require the optional Typer dependency.
    app = optional_import("dascore._cli").app

    try:
        return app(args=argv, prog_name="dascore")
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code
