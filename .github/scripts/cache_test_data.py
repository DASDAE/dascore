"""Populate the pooch cache with every file in dascore's data registry."""

from __future__ import annotations

from pathlib import Path

from dascore.utils.downloader import fetch, get_registry_df


def main() -> None:
    """Fetch every registered test-data file into the local pooch cache."""
    registry = get_registry_df()
    total = len(registry)
    print(f"Priming DASCore test-data cache with {total} files")
    for index, name in enumerate(registry["name"], start=1):
        path = Path(fetch(name))
        print(f"[{index}/{total}] {name} -> {path}")
    print("Finished priming DASCore test-data cache")


if __name__ == "__main__":
    main()
