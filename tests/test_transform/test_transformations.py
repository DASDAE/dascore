"""General tests for transformations."""

from __future__ import annotations

import pytest


def test_deprecated(random_patch):
    """Ensure the tran patch namespace is deprecated."""
    with pytest.warns(DeprecationWarning):
        _ = random_patch.tran
