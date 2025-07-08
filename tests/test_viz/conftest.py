"""Configuration for all vizualization tests."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="function", autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")
