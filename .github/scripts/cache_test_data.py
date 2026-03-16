"""Populate the pooch cache with every file in dascore's data registry."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dascore.utils.downloader import fetch, get_registry_df  # noqa: E402


def main() -> None:
    """Fetch every registered test-data file into the local pooch cache."""
    registry = get_registry_df()
    total = len(registry)
    print(f"Priming DASCore test-data cache with {total} files")  # noqa
    for index, name in enumerate(registry["name"], start=1):
        path = Path(fetch(name))
        print(f"[{index}/{total}] {name} -> {path}")  # noqa
    print("Finished priming DASCore test-data cache")  # noqa


if __name__ == "__main__":
    main()
