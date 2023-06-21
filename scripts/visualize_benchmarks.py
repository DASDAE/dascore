"""
Load the benchmark files and create simple table.
"""
import json

import pandas as pd

import os
from contextlib import contextmanager
from pathlib import Path
from subprocess import run

from rich.console import Console

console = Console()

# install(show_locals=True)

BASE_PATH = Path(__file__).absolute().parent.parent
RESULTS_PATH = BASE_PATH / ".asv" / "results"


def read_benchmark_files():
    out = {}
    for result_path in RESULTS_PATH.glob("*.json"):
        result = result_path.read_text()
        out[result_path.name.replace(".json", '')] = json.loads(result)
    return out





if __name__ == "__main__":
    results = read_benchmark_files()
    breakpoint()

    pass
