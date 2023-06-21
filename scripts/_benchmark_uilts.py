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
    expected_output_file = BASE_PATH / ".asv" / "results" / "benchmarks.json"
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
    hash = git_hash()
    run(f"asv dev -e --set-commit-hash {hash}", check=True, shell=True)
    assert expected_output_file.exists()
    # expected_output_file.rename(expected_output_file.parent / f"{name}.json")
    return


def compare_asv(hash1, hash2):
    """Compare two commit hash in something vaguely approximating a table."""
    kwargs = dict(check=True, shell=True, capture_output=True, text=True)
    out = run(f"asv compare {hash1} {hash2}", **kwargs)
    return out.stdout.strip()


def git_branch_name():
    """Return the current branch name."""
    kwargs = dict(check=True, shell=True, capture_output=True, text=True)
    out = run("git branch --show-current", **kwargs)
    return out.stdout.strip()


def git_hash(branch_name=None):
    """Return the current git hash name."""
    branch = branch_name or git_branch_name()
    kwargs = dict(check=True, shell=True, capture_output=True, text=True)
    out = run(f"git rev-parse --short --verify {branch}", **kwargs)
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
