"""The DASCore command-line entry point."""

from __future__ import annotations

import argparse
import io
import os
import sys
from pathlib import Path

import dascore as dc
from dascore.core.inventory import (
    Acquisition,
    Cable,
    DistanceMap,
    FiberArray,
    FiberSegment,
    Interrogator,
    Inventory,
    Network,
    OpticalPath,
)
from dascore.core.inventory_loader import load_directory
from dascore.exceptions import InvalidInventoryError
from dascore.utils.doc_cache import documentation_cache, read_document
from dascore.utils.doc_corpus import DocumentationError
from dascore.utils.doc_skills import (
    get_skills,
    install_skills,
    read_bootstrap,
    read_skill,
)
from dascore.utils.documents import write_document

# Stands in for every descriptive value the user should replace.
_PLACEHOLDER = "TODO"


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


def _skeleton() -> dict[str, tuple]:
    """Map each skeleton file to the object it holds and the fields it leaves out."""
    span = (0.0, 1000.0)
    path = OpticalPath(
        optical_components=(
            FiberSegment(distance_min=0.0, distance_max=1000.0, container="cable"),
        )
    )
    acquisition = Acquisition(
        code="DAS",
        location_code="00",
        data_category="DAS",
        data_type="strain_rate",
        interrogator="interrogator",
        gauge_length=10.0,
        spatial_interval=1.0,
        sample_rate=1000.0,
        distance_map=DistanceMap(acquisition_distance=span, optical_distance=span),
    )
    collections = {"networks", "resources", "fiber_arrays", "stations"}
    return {
        "inventory.yaml": (Inventory(), collections),
        "resources/interrogator.yaml": (
            Interrogator(model=_PLACEHOLDER),
            {"resource_id"},
        ),
        "resources/cable.yaml": (Cable(model=_PLACEHOLDER), {"resource_id"}),
        "networks/XX.yaml": (Network(code="XX"), collections),
        "fiber_arrays/XX.STA/attrs.yaml": (
            FiberArray(code="STA"),
            {"acquisitions", "optical_paths"},
        ),
        "fiber_arrays/XX.STA/path.00/attrs.yaml": (path, set()),
        "acquisitions/XX.STA.00.DAS.yaml": (acquisition, set()),
    }


def _inventory_init(args: argparse.Namespace) -> None:
    """Write a minimal inventory directory to fill in."""
    if args.directory.exists() and any(args.directory.iterdir()):
        args.error(f"{args.directory} exists and is not empty.")
    for name, (model, leave_out) in _skeleton().items():
        # Unbounded times dump as NaT, which a person filling in should not meet.
        leave_out |= {"time_min", "time_max"}
        data = model.model_dump(mode="json", exclude_defaults=True, exclude=leave_out)
        sys.stdout.write(f"{write_document(data, args.directory / name, 'yaml')}\n")
    sys.stdout.write(
        f"Replace {_PLACEHOLDER!r}, the codes XX, STA, 00 and DAS, and every "
        "number with your deployment's values.\n"
    )


def _inventory_check(args: argparse.Namespace) -> None:
    """Load an inventory directory and report what is wrong with it."""
    if not args.directory.is_dir():
        args.error(f"{args.directory} is not a directory.")
    inventory = load_directory(args.directory)
    arrays = [x for net in inventory.networks for x in net.fiber_arrays]
    counts = {
        "networks": len(inventory.networks),
        "fiber arrays": len(arrays),
        "acquisitions": sum(len(x.acquisitions) for x in arrays),
        "optical paths": sum(len(x.optical_paths) for x in arrays),
        "resources": len(inventory.resources),
    }
    summary = ", ".join(f"{value} {key}" for key, value in counts.items())
    sys.stdout.write(f"OK: {summary}\n")


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
        help="Write an entry for every skill into this agent skills directory",
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
    inventory = commands.add_parser(
        "inventory", help="Start or check an inventory directory."
    )
    actions = inventory.add_subparsers(required=True, metavar="ACTION")
    for name, run in (("init", _inventory_init), ("check", _inventory_check)):
        action = actions.add_parser(name, help=run.__doc__, description=run.__doc__)
        action.add_argument("directory", type=Path, help="The inventory directory")
        action.set_defaults(run=run, error=action.error)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return its exit status."""
    # Redirected Windows streams can otherwise reject Unicode documentation.
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        args = _parser().parse_args(argv)
        args.run(args)
        # Flush here so a closed pipe is handled below, not at interpreter exit.
        sys.stdout.flush()
    except SystemExit as exc:
        assert isinstance(exc.code, int)
        return exc.code
    except BrokenPipeError:
        # The reader, such as `head`, stopped early. Point stdout at devnull so
        # the interpreter's own exit flush stays quiet as well.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
        return 1
    except (DocumentationError, InvalidInventoryError, OSError) as exc:
        sys.stderr.write(f"dascore: {exc}\n")
        return 1
    return 0
