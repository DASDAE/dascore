"""Typer command definitions, loaded only when the CLI is invoked."""

from __future__ import annotations

import io
import sys

import typer

import dascore as dc
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_search import search_documents

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


@app.command("doc-search")
def doc_search(
    query: str = typer.Argument(
        "", help="Search words, a quoted phrase, or Boolean query"
    ),
    tag: str | None = typer.Option(None, "--tag", help="Filter by an exact keyword"),
    limit: int = typer.Option(5, "--limit", help="Maximum results (1-100)"),
) -> None:
    """Search installed documentation and keyword tags."""
    results = search_documents(query, tag=tag, limit=limit)
    if not results:
        typer.echo("No matching documentation.")
    for number, result in enumerate(results, 1):
        typer.echo(f"{number}. {result['title']} [{result['kind']}]")
        if result["keywords"]:
            typer.echo(f"   Keywords: {', '.join(result['keywords'])}")
        typer.echo(f"   {result['excerpt']}")
        typer.echo(f"   Read: dascore doc {result['id']}\n")
