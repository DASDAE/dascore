"""Tests for spectra plotting."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes


class TestSpecPlot:
    """Tests for spectra plot visualization."""

    def test_runs_with_time_spectrum(self, random_patch):
        """Ensure specplot runs on a time-frequency patch."""
        patch = random_patch.spectra(dim="time")

        ax = patch.viz.specplot(show=False)

        assert isinstance(ax, Axes)

    def test_runs_with_distance_spectrum(self, random_patch):
        """Ensure specplot runs on a distance-frequency patch."""
        patch = random_patch.spectra(dim="distance")

        ax = patch.viz.specplot(show=False)

        assert isinstance(ax, Axes)

    def test_uses_provided_axis(self, random_patch):
        """Ensure specplot uses the supplied axis."""
        patch = random_patch.spectra(dim="time")
        _, ax = plt.subplots()

        out = patch.viz.specplot(ax=ax, show=False)

        assert out is ax

    def test_time_frequency_label_is_formatted(self, random_patch):
        """Ensure ft_time label is replaced by Frequency."""
        patch = random_patch.spectra(dim="time")

        ax = patch.viz.specplot(show=False)

        labels = {ax.get_xlabel(), ax.get_ylabel()}

        assert any("Frequency" in label for label in labels)

    def test_distance_frequency_label_is_formatted(self, random_patch):
        """Ensure ft_distance label is replaced by Wavenumber."""
        patch = random_patch.spectra(dim="distance")

        ax = patch.viz.specplot(show=False)

        labels = {ax.get_xlabel(), ax.get_ylabel()}

        assert any("Wavenumber" in label for label in labels)

    def test_log_sets_fft_axis_to_log_scale(self, random_patch):
        """Ensure log=True sets the Fourier axis to log scale."""
        patch = random_patch.spectra(dim="time")

        ax = patch.viz.specplot(log=True, show=False)

        fft_dim = next(dim for dim in patch.dims if dim.startswith("ft_"))
        scale = ax.get_yscale() if patch.dims.index(fft_dim) == 0 else ax.get_xscale()

        assert scale == "log"

    def test_log_lower_limit_uses_frequency_step(self, random_patch):
        """Ensure log=True sets lower FFT-axis limit to coordinate step."""
        patch = random_patch.spectra(dim="time")

        ax = patch.viz.specplot(log=True, show=False)

        fft_dim = next(dim for dim in patch.dims if dim.startswith("ft_"))
        fft_axis = patch.dims.index(fft_dim)

        lim = ax.get_ylim() if fft_axis == 0 else ax.get_xlim()

        assert lim[0] == patch.get_coord(fft_dim).step

    def test_raises_without_fourier_dimension(self, random_patch):
        """Ensure a patch without Fourier coordinates raises."""
        with pytest.raises(Exception, match="Fourier-transformed coordinate"):
            random_patch.viz.specplot(show=False)

    def test_passes_waterfall_kwargs(self, random_patch):
        """Ensure plotting kwargs are accepted and forwarded."""
        patch = random_patch.spectra(dim="time")

        ax = patch.viz.specplot(
            cmap="viridis",
            scale=(0.1, 0.9),
            scale_type="relative",
            interpolation="nearest",
            show=False,
        )

        assert isinstance(ax, Axes)

    def test_show(self, random_patch, monkeypatch):
        """Ensure show path is callable."""
        monkeypatch.setattr(plt, "show", lambda: None)
        patch = random_patch.spectra(dim="time")

        patch.viz.specplot(show=True)
