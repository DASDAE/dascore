"""Configuration for all visualization tests."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="function", autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


@pytest.fixture(autouse=True)
def shown(monkeypatch) -> list:
    """Record what `show=True` shows, rather than trying to show it.

    Agg makes the backend headless; this is what keeps `plt.show()` from
    warning that the canvas is non-interactive, and it gives a test asking
    whether `show=True` reached the call something to assert.
    """
    calls = []
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: calls.append(kwargs))
    return calls
