"""Typer command definitions, loaded only when the CLI is invoked."""

from __future__ import annotations

import io
import sys

import typer

import dascore as dc
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_skills import get_skills, read_bootstrap, read_skill

app = typer.Typer(
    help="The DASCore command-line interface.",
    context_settings={"help_option_names": ["-h", "--help"]},
    add_completion=False,
    pretty_exceptions_enable=False,
)


def _show_version(value: bool) -> None:
    """Handle the eager version option before command validation."""
    if value:
        typer.echo(f"DASCore {dc.__version__}")
        raise typer.Exit()


@app.callback()
def cli(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_show_version,
        is_eager=True,
        help="Show the DASCore version.",
    ),
) -> None:
    """Configure the DASCore command group."""
    # Redirected Windows streams can otherwise reject Unicode documentation.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")


@app.command()
def doc(
    target: str | None = typer.Argument(
        None, help="Public API name or documentation page identifier"
    ),
    rebuild: bool = typer.Option(
        False, "--rebuild", help="Regenerate the Markdown corpus"
    ),
) -> None:
    """Build or read the installed documentation."""
    with documentation_cache(rebuild=rebuild) as (root, manifest):
        if target:
            sys.stdout.write(read_document(root, manifest, target))
        else:
            sys.stdout.write(
                f"DASCore: {dc.__version__}\nPython: {sys.executable}\n"
                f"Package: {dc.__file__}\nDocumentation: {root}\n"
                f"Documents: {len(manifest['documents'])}\n"
            )
            for omitted in manifest["omitted"]:
                sys.stdout.write(f"Unavailable: {omitted}\n")


@app.command()
def skills() -> None:
    """List the recipes available as skills."""
    for name, info in get_skills().items():
        typer.echo(f"{name}: {info['description']}")


@app.command()
def skill(
    name: str | None = typer.Argument(None, help="Name from the skills catalog"),
    bootstrap: bool = typer.Option(
        False, "--bootstrap", help="Print the native agent bootstrap"
    ),
) -> None:
    """Read a skill or print the bootstrap for explicit agent installation."""
    if bootstrap and name is not None:
        raise typer.BadParameter("Use a skill name or --bootstrap, not both.")
    if bootstrap:
        sys.stdout.write(read_bootstrap())
    elif name is not None:
        sys.stdout.write(read_skill(name))
    else:
        raise typer.BadParameter("Provide a skill name or use --bootstrap.")
