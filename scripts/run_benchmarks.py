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
from __future__ import annotations
from _benchmark_uilts import (
    BASE_PATH,
    REFERENCE_BRANCH,
    cd,
    git_branch_name,
    git_checkout,
    run_asv,
)

if __name__ == "__main__":
    with cd(BASE_PATH):
        current_name = git_branch_name()
        # first run benchmarks of reference.
        with git_checkout(REFERENCE_BRANCH):
            run_asv(REFERENCE_BRANCH)
        # now run benchmarks on current branch.
        run_asv(current_name)
