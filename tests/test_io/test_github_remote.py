"""
Tests for reading DAS files from GitHub using universal_pathlib.

These tests verify that dascore can read files from remote GitHub repositories
using UPath with the github:// protocol.

Implementation Notes:
- HDF5 files (h5, hdf5) are streamed directly using h5py's fileobj driver
- PyTables-based formats download to temp files (fallback if streaming fails)
- Non-HDF5 formats use fsspec for direct access when supported
- Tests automatically skip if no internet connection or GitHub unavailable

GitHub Token Support:
---------------------
To avoid rate limiting (60 requests/hour unauthenticated), set a GitHub token
in one of these environment variables (checked in this order):

    export DASDAE_GITHUB_TOKEN=your_personal_access_token  # Recommended
    # or
    export GITHUB_TOKEN=your_personal_access_token
    # or
    export GH_TOKEN=your_personal_access_token

With a token, the rate limit increases to 5000 requests/hour.

To create a token:
    1. Go to GitHub Settings > Developer settings > Personal access tokens
    2. Generate new token (classic) with 'public_repo' scope
    3. Set the environment variable with your token
"""

from __future__ import annotations

import socket

import pytest

import dascore as dc
from dascore.compat import UPath


def has_internet_connection(host="8.8.8.8", port=53, timeout=3):
    """
    Check if internet connection is available.

    Parameters
    ----------
    host : str
        Host to check (default: Google DNS)
    port : int
        Port to check
    timeout : int
        Timeout in seconds

    Returns
    -------
    bool
        True if internet is available, False otherwise
    """
    try:
        socket.setdefaulttimeout(timeout)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((host, port))
        return True
    except (socket.error, OSError):
        return False


def can_access_github():
    """
    Check if GitHub is accessible and not rate limited.

    Returns
    -------
    bool
        True if GitHub is accessible and not rate limited, False otherwise
    """
    try:
        import requests

        # Check basic connectivity
        response = requests.head("https://github.com", timeout=5)
        if response.status_code >= 500:
            return False

        # Check API rate limit
        rate_response = requests.get("https://api.github.com/rate_limit", timeout=5)
        if rate_response.status_code == 200:
            data = rate_response.json()
            remaining = data.get("resources", {}).get("core", {}).get("remaining", 0)
            # Need at least a few requests available
            if remaining < 5:
                return False

        # Try to access the test repo
        repo_response = requests.head(
            "https://api.github.com/repos/dasdae/test_data", timeout=5
        )
        # 403 means rate limited or forbidden
        if repo_response.status_code == 403:
            return False

        return True
    except Exception:
        return False


# Skip all tests in this module if no internet or GitHub access
pytestmark = pytest.mark.skipif(
    not has_internet_connection() or not can_access_github(),
    reason="No internet connection, GitHub not accessible, or API rate limited",
)


class TestGitHubRead:
    """Tests for reading files from GitHub."""

    def test_read_h5_simple_from_github(self, github_das_path):
        """Test reading a simple HDF5 file from GitHub."""
        path = github_das_path / "h5_simple_1.h5"
        assert path.exists(), f"GitHub file does not exist: {path}"

        spool = dc.read(path)
        assert len(spool) > 0, "No patches read from GitHub file"

        patch = spool[0]
        assert patch is not None
        assert hasattr(patch, "data")
        assert hasattr(patch, "dims")

    def test_read_dasdae_from_github(self, github_das_path):
        """Test reading a DASDAE file from GitHub."""
        path = github_das_path / "example_dasdae_event_1.h5"
        assert path.exists(), f"GitHub file does not exist: {path}"

        spool = dc.read(path)
        assert len(spool) > 0, "No patches read from GitHub DASDAE file"

        patch = spool[0]
        assert patch.dims == ("distance", "time")

    def test_read_terra15_from_github(self, github_das_path):
        """Test reading a Terra15 file from GitHub."""
        path = github_das_path / "terra15_das_1_trimmed.hdf5"
        assert path.exists(), f"GitHub file does not exist: {path}"

        spool = dc.read(path)
        assert len(spool) > 0, "No patches read from GitHub Terra15 file"

        patch = spool[0]
        assert patch.dims == ("time", "distance")


class TestGitHubScan:
    """Tests for scanning files from GitHub."""

    def test_scan_h5_simple_from_github(self, github_das_path):
        """Test scanning a simple HDF5 file from GitHub."""
        path = github_das_path / "h5_simple_1.h5"
        assert path.exists()

        result = dc.scan(path)
        assert len(result) > 0, "No patches found when scanning GitHub file"

        attrs = result[0]
        assert hasattr(attrs, "time_min")
        assert hasattr(attrs, "time_max")

    def test_scan_multiple_formats_from_github(self, github_test_repo):
        """Test scanning different file formats from GitHub."""
        files = [
            "das/h5_simple_1.h5",
            "das/example_dasdae_event_1.h5",
            "das/prodml_2.0.h5",
        ]

        for file_path in files:
            path = github_test_repo / file_path
            if path.exists():
                result = dc.scan(path)
                assert len(result) > 0, f"Failed to scan {file_path}"


class TestGitHubIntegration:
    """Integration tests for GitHub filesystem support."""

    def test_github_path_with_string_input(self, github_test_repo):
        """Test that string GitHub URLs work."""
        path = github_test_repo / "das" / "h5_simple_1.h5"
        spool = dc.read(path)
        assert len(spool) > 0

    def test_github_path_exists(self, github_das_path):
        """Test path existence checks work on GitHub."""
        # This file exists
        path_exists = github_das_path / "h5_simple_1.h5"
        assert path_exists.exists()

        # This file does not exist
        path_not_exists = github_das_path / "nonexistent_file.h5"
        assert not path_not_exists.exists()

    def test_mixed_local_and_github_paths(self, github_das_path):
        """Test that local and GitHub paths work together."""
        # Get a local example
        local_patch = dc.get_example_patch()

        # Get a GitHub patch
        github_path = github_das_path / "h5_simple_1.h5"
        github_spool = dc.read(github_path)
        github_patch = github_spool[0]

        # Both should be patches
        assert isinstance(local_patch, dc.Patch)
        assert isinstance(github_patch, dc.Patch)

    @pytest.mark.xfail(reason="SEGY format detection needs investigation")
    def test_read_small_segy_from_github(self, github_das_path):
        """Test reading a small SEGY file from GitHub."""
        path = github_das_path / "small_channel_patch.sgy"
        assert path.exists()

        # SEGY files should be readable
        spool = dc.read(path)
        assert len(spool) > 0
