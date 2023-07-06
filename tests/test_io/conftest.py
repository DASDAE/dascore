"""Io specific fixtures."""

import pytest

from dascore.utils.downloader import fetch, get_registry_df


@pytest.fixture(scope="class", params=get_registry_df()["name"])
def data_file_path(request):
    """A fixture of all data files. Will download if needed."""
    return fetch(request.param)
