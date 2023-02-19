"""Configuration for all vizualization tests."""
import matplotlib.pyplot as plt
import pytest


@pytest.fixture(scope="function", autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")
