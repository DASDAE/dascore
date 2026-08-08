"""Populate the CI test-data cache from a checkout of DASDAE/test_data.

Copies every file listed in dascore's data registry from a local checkout of
https://github.com/DASDAE/test_data into the layout pooch expects
(.test_data_cache/<DATA_VERSION>/<name>), verifying each sha256 against the
registry. Registry entries hosted elsewhere are skipped with a warning (old
release tags may still contain them; a unit test forbids new ones) — pooch
just lazy-fetches those at test time.

Uses only the standard library so it can run on the runners' system python.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = ROOT / "dascore" / "data_registry.txt"
CONSTANTS_PATH = ROOT / "dascore" / "constants.py"
REPO_PATH = ROOT / ".test_data_repo"
CACHE_PATH = ROOT / ".test_data_cache"
URL_REGEX = re.compile(
    r"^https?://github\.com/dasdae/test_data/raw/master/(?P<subpath>.+)$",
    re.IGNORECASE,
)


def get_data_version() -> str:
    """Parse DATA_VERSION from dascore/constants.py without importing dascore."""
    match = re.search(
        r'^DATA_VERSION\s*=\s*"([^"]+)"', CONSTANTS_PATH.read_text(), re.MULTILINE
    )
    if match is None:
        sys.exit(f"DATA_VERSION not found in {CONSTANTS_PATH}")
    return match.group(1)


def sha256sum(path: Path) -> str:
    """Return the sha256 hex digest of a file, reading in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Copy registry files from the repo checkout into the cache layout."""
    dest_dir = CACHE_PATH / get_data_version()
    dest_dir.mkdir(parents=True, exist_ok=True)
    errors = []
    copied = 0
    skipped = 0
    total_bytes = 0
    for line in REGISTRY_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, expected_hash, url = line.split()
        match = URL_REGEX.match(url)
        if match is None:
            print(f"Skipping {name}: not hosted in the DASDAE/test_data repo")  # noqa: T201
            skipped += 1
            continue
        # Registry URLs may percent-encode characters (e.g. + as %2B); the
        # files in the repo use the decoded names.
        source = (REPO_PATH / unquote(match["subpath"])).resolve()
        dest = (dest_dir / name).resolve()
        if not source.is_relative_to(REPO_PATH) or dest.parent != dest_dir.resolve():
            errors.append(f"{name}: escapes its intended directory")
            continue
        if not source.exists():
            errors.append(f"{name}: {source} missing from the test_data checkout")
            continue
        if (digest := sha256sum(source)) != expected_hash:
            errors.append(
                f"{name}: sha256 mismatch (registry {expected_hash}, repo {digest})"
            )
            continue
        shutil.copy2(source, dest)
        copied += 1
        total_bytes += source.stat().st_size
    print(  # noqa: T201
        f"Copied {copied} files ({total_bytes / 1_000_000:.0f} MB) to {dest_dir}"
        f" ({skipped} skipped)"
    )
    if errors:
        print("Failed to prime test data cache:", file=sys.stderr)  # noqa: T201
        for error in errors:
            print(f"  {error}", file=sys.stderr)  # noqa: T201
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
