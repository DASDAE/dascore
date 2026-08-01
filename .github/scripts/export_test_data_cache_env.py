"""Export DASCore test-data cache metadata for GitHub Actions."""

from __future__ import annotations

import os
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dascore.constants import DATA_VERSION  # noqa: E402
from dascore.utils.downloader import REGISTRY_PATH, get_fetcher  # noqa: E402


def get_restore_prefix(runner_os: str, cache_number: str) -> str:
    """
    Return the cache-key prefix used to fall back to an older cache.

    A registry change makes the exact key miss. Restoring the previous cache
    under this prefix leaves pooch fetching only the added files rather than
    the whole registry again.

    ``cache_number`` is part of the prefix, not just the full key, so bumping
    it still resets the cache; a prefix which stopped at the data version
    would fall back onto the cache the bump meant to drop.
    """
    return f"data-{runner_os}-{DATA_VERSION}-{cache_number}-"


def get_key(runner_os: str, cache_number: str, registry_hash: str) -> str:
    """Return the cache key for the given OS, cache number, and registry."""
    return f"{get_restore_prefix(runner_os, cache_number)}{registry_hash}"


def main() -> None:
    """Print cache metadata as KEY=VALUE lines for GitHub Actions env files."""
    runner_os = os.environ["RUNNER_OS"]
    cache_number = os.environ["INPUT_CACHE_NUMBER"]
    registry_hash = sha256(REGISTRY_PATH.read_bytes()).hexdigest()
    # pooch stores files in <root>/<DATA_VERSION>; cache the whole root, since
    # DATA_VERSION is already part of the key.
    cache_path = Path(get_fetcher().path).parent

    print(f"DATA_REGISTRY_HASH={registry_hash}")  # noqa: T201
    print(f"DATA_CACHE_PATH={cache_path}")  # noqa: T201
    print(f"DATA_VERSION={DATA_VERSION}")  # noqa: T201
    print(f"DATA_CACHE_KEY={get_key(runner_os, cache_number, registry_hash)}")  # noqa: T201
    print(f"DATA_CACHE_RESTORE_PREFIX={get_restore_prefix(runner_os, cache_number)}")  # noqa: T201


if __name__ == "__main__":
    main()
