"""Command-line access to the installed DASCore documentation."""

from __future__ import annotations

import argparse
import sys

import dascore as dc
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import DocumentationError


def main(argv: list[str] | None = None) -> int:
    """
    Run the DASCore command line and return its exit status.

    Use `dascore doc` to prepare the installed version's local documentation,
    or `dascore doc Patch.select` to display a documented API.
    """
    parser = argparse.ArgumentParser(prog="dascore", description=__doc__)
    parser.add_argument(
        "--version", action="version", version=f"DASCore {dc.__version__}"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    doc = commands.add_parser("doc", help="Build or read the installed documentation")
    doc.add_argument(
        "target", nargs="?", help="Public API name or documentation page identifier"
    )
    doc.add_argument(
        "--rebuild", action="store_true", help="Regenerate the Markdown corpus"
    )
    args = parser.parse_args(argv)
    try:
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
    except (DocumentationError, OSError) as exc:
        sys.stderr.write(f"dascore: {exc}\n")
        return 1
    return 0
