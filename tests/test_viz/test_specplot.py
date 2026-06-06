"""Tests for spectrum plotting."""

import matplotlib.pyplot as plt
import pytest

from dascore.exceptions import CoordError


def test_specplot_requires_fourier_dimension(random_patch):
    """Specplot raises if the patch has no Fourier-transformed coordinate."""
    with pytest.raises(CoordError, match="Fourier-transformed coordinate"):
        random_patch.viz.specplot()


def test_specplot_returns_axes(random_patch):
    """Specplot returns a matplotlib Axes instance."""
    patch = random_patch.dft("time").abs()

    ax = patch.viz.specplot()

    assert isinstance(ax, plt.Axes)


def test_specplot_uses_existing_axes(random_patch):
    """Specplot plots onto a provided axes."""
    patch = random_patch.dft("time").abs()
    _, ax = plt.subplots()

    out = patch.viz.specplot(ax=ax)

    assert out is ax


def test_specplot_relabels_frequency_axis(random_patch):
    """The ft_time axis label is rewritten as Frequency."""
    patch = random_patch.dft("time").abs()

    ax = patch.viz.specplot()

    labels = {ax.get_xlabel(), ax.get_ylabel()}
    assert any("Frequency" in label for label in labels)
    assert not any("Ft_time" in label for label in labels)


def test_specplot_relabels_wavenumber_axis(random_patch):
    """The ft_distance axis label is rewritten as Wavenumber."""
    patch = random_patch.dft("distance").abs()

    ax = patch.viz.specplot()

    labels = {ax.get_xlabel(), ax.get_ylabel()}
    assert any("Wavenumber" in label for label in labels)
    assert not any("Ft_distance" in label for label in labels)


def test_specplot_log_time_axis(random_patch):
    """The ft_time axis uses log scaling when log=True."""
    patch = random_patch.dft("time").abs()

    ax = patch.viz.specplot(log=True)

    assert ax.get_xscale() == "log" or ax.get_yscale() == "log"


def test_specplot_log_distance_axis_uses_symlog(random_patch):
    """The ft_distance axis uses symlog scaling when log=True."""
    patch = random_patch.dft("distance").abs()

    ax = patch.viz.specplot(log=True)

    assert ax.get_xscale() == "symlog" or ax.get_yscale() == "symlog"


def test_show(random_patch, monkeypatch):
    """Ensure show path is callable."""
    monkeypatch.setattr(plt, "show", lambda: None)
    random_patch.dft("distance").abs().viz.specplot(show=True)
