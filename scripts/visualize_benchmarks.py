"""Load the benchmark files and create simple table."""
from __future__ import annotations

from _benchmark_uilts import REFERENCE_BRANCH, compare_asv, console, git_hash

if __name__ == "__main__":
    commit1 = git_hash(REFERENCE_BRANCH)
    commit2 = git_hash()
    out = compare_asv(commit1, commit2)
    console.print(out)
