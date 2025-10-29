"""Fixtures for IO tests, especially remote file testing."""

from __future__ import annotations

import os

import pytest

from dascore.compat import UPath


@pytest.fixture(scope="session")
def github_token():
    """
    Get GitHub token from environment variables.

    Checks DASDAE_GITHUB_TOKEN, GITHUB_TOKEN, and GH_TOKEN environment variables
    in that order.

    Returns
    -------
    str or None
        The GitHub token if found, None otherwise
    """
    return (
        os.getenv("DASDAE_GITHUB_TOKEN")
        or os.getenv("GITHUB_TOKEN")
        or os.getenv("GH_TOKEN")
    )


@pytest.fixture(scope="session")
def github_test_repo(github_token):
    """
    Create a UPath for the test data repository with token if available.

    Parameters
    ----------
    github_token : str or None
        GitHub token from the github_token fixture

    Returns
    -------
    UPath
        UPath configured with token if available
    """
    repo_url = "github://dasdae:test_data@"
    if github_token:
        return UPath(repo_url, token=github_token)
    return UPath(repo_url)


@pytest.fixture(scope="session")
def github_das_path(github_test_repo):
    """
    Get the path to the DAS data directory in the test repository.

    Parameters
    ----------
    github_test_repo : UPath
        Base path to test repository

    Returns
    -------
    UPath
        Path to das/ directory
    """
    return github_test_repo / "das"
