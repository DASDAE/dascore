"""
Python script to run benchmarks for dascore.

It works by using airspeed velocity to run code on current commit,
collecting the results, then running code on reference branch (eg master)
the resetting the repo to current commit.

A few notes:
- Before running this script make sure all your changes are committed. The
    script will raise an exception otherwise.
- asv can usually be run in a more intelligent way, but it doesn't yet
    support projects without setup.py files, so this is a bit of a hack
    until that gets fixed.
"""
import os
from contextlib import contextmanager
from pathlib import Path
from subprocess import run

from rich.console import Console

console = Console()

# install(show_locals=True)

BASE_PATH = Path(__file__).absolute().parent.parent
REFERENCE_BRANCH = "master"


@contextmanager
def cd(path):
    """Change directory temporarily"""
    current = os.getcwd()
    os.chdir(path)
    try:
        yield
    except Exception:
        os.chdir(current)
        raise
    else:
        os.chdir(current)


def run_asv(name):
    """Run asv, rename output"""
    expected_output_file = BASE_PATH / '.asv' / "results" / "benchmarks.json"
    if expected_output_file.exists():
        expected_output_file.unlink()
    # check for machine file, if it doesnt exist just use defaults.
    machine_file_path = Path.home() / ".asv-machine.json"
    if not machine_file_path.exists():
        run("asv machine --yes", check=True, shell=True)
    assert machine_file_path.exists()
    # tell the user what is going on
    console.print()
    console.rule(f"[bold red]Running benchmarks for {name}")
    console.print()
    # run benchmarks
    run("asv run -E existing -e", check=True, shell=True)
    assert expected_output_file.exists()
    expected_output_file.rename(expected_output_file.parent / f"{name}.json")
    return


def git_branch_name():
    """Return the current branch name."""
    kwargs = dict(check=True, shell=True, capture_output=True, text=True)
    out = run("git branch --show-current", **kwargs)
    return out.stdout.strip()


@contextmanager
def git_checkout(new_branch):
    """Checkout branch/commit within context manager."""

    def _checkout(name):
        """Checkout something in git."""
        run(f"git checkout {name}", check=True, shell=True)
        return

    current = git_branch_name()
    _checkout(new_branch)
    try:
        yield
    except Exception:
        _checkout(current)
        raise
    else:
        _checkout(current)


if __name__ == "__main__":
    with cd(BASE_PATH):
        current_name = git_branch_name()
        # first run benchmarks of reference.
        # with git_checkout(REFERENCE_BRANCH):
        #     run_asv(REFERENCE_BRANCH)
        # now run benchmarks on current branch.
        run_asv(current_name)
