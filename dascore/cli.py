"""The DASCore command-line entry point."""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import dascore as dc
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import DocumentationError
from dascore.utils.doc_skills import (
    get_skills,
    install_skills,
    read_bootstrap,
    read_skill,
)


def _doc(args: argparse.Namespace) -> None:
    """Build or read the installed documentation."""
    with documentation_cache(rebuild=args.rebuild) as (root, manifest):
        if args.target:
            sys.stdout.write(read_document(root, manifest, args.target))
        else:
            sys.stdout.write(
                f"DASCore: {dc.__version__}\nPython: {sys.executable}\n"
                f"Package: {dc.__file__}\nDocumentation: {root}\n"
                f"Documents: {len(manifest['documents'])}\n"
            )
            for omitted in manifest["omitted"]:
                sys.stdout.write(f"Unavailable: {omitted}\n")


def _skills(args: argparse.Namespace) -> None:
    """List the skills, or install all of them into an agent's skills directory."""
    if args.install is not None:
        for path in install_skills(args.install):
            sys.stdout.write(f"{path}\n")
        return
    for name, info in get_skills().items():
        sys.stdout.write(f"{name}: {info['description']}\n")


def _skill(args: argparse.Namespace) -> None:
    """Read a skill or print the bootstrap for explicit agent installation."""
    if args.bootstrap and args.name is not None:
        args.error("Use a skill name or --bootstrap, not both.")
    if args.bootstrap:
        sys.stdout.write(read_bootstrap())
    elif args.name is not None:
        sys.stdout.write(read_skill(args.name))
    else:
        args.error("Provide a skill name or use --bootstrap.")


def _parser() -> argparse.ArgumentParser:
    """Declare the commands and their arguments."""
    parser = argparse.ArgumentParser(
        prog="dascore", description="The DASCore command-line interface."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"DASCore {dc.__version__}",
        help="Show the DASCore version.",
    )
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    doc = commands.add_parser("doc", help=_doc.__doc__, description=_doc.__doc__)
    doc.add_argument(
        "target", nargs="?", help="Public API name or documentation page identifier"
    )
    doc.add_argument(
        "--rebuild", action="store_true", help="Regenerate the Markdown corpus"
    )
    doc.set_defaults(run=_doc)
    skills = commands.add_parser(
        "skills", help=_skills.__doc__, description=_skills.__doc__
    )
    skills.add_argument(
        "--install",
        type=Path,
        metavar="DIRECTORY",
        help="Copy every skill into this agent skills directory",
    )
    skills.set_defaults(run=_skills)
    skill = commands.add_parser(
        "skill", help=_skill.__doc__, description=_skill.__doc__
    )
    skill.add_argument("name", nargs="?", help="Name from the skills catalog")
    skill.add_argument(
        "--bootstrap", action="store_true", help="Print the native agent bootstrap"
    )
    skill.set_defaults(run=_skill, error=skill.error)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit status."""
    # Redirected Windows streams can otherwise reject Unicode documentation.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        args = _parser().parse_args(argv)
        args.run(args)
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code
    except (DocumentationError, OSError) as exc:
        sys.stderr.write(f"dascore: {exc}\n")
        return 1
    return 0
