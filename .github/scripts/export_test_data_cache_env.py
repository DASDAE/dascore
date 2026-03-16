"""Export DASCore test-data cache metadata for GitHub Actions."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dascore.utils.downloader import get_test_data_cache_info  # noqa: E402


def main() -> None:
    """Print cache metadata as KEY=VALUE lines for GitHub Actions env files."""
    runner_os = os.environ["RUNNER_OS"]
    cache_number = os.environ["INPUT_CACHE_NUMBER"]
    info = get_test_data_cache_info()

    print(f"DATA_REGISTRY_HASH={info.registry_hash}")  # noqa: T201
    print(f"DATA_CACHE_PATH={info.cache_path}")  # noqa: T201
    print(f"DATA_VERSION={info.data_version}")  # noqa: T201
    print(  # noqa: T201
        f"DATA_CACHE_KEY={info.get_key(runner_os=runner_os, cache_number=cache_number)}"
    )


if __name__ == "__main__":
    main()
