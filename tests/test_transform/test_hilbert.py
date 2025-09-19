"""
Tests for signal processing functions (hilbert, envelope, phase_weighted_stack).
"""

from __future__ import annotations

import numpy as np
import pytest

import dascore as dc
from dascore.examples import ricker_moveout
from dascore.exceptions import ParameterError


class TestHilbert:
    """Tests for the Hilbert transform function."""

    def test_hilbert_basic(self, random_patch):
        """Test basic Hilbert transform functionality."""
        result = random_patch.hilbert(dim="time")

        # Result should be complex
        assert np.iscomplexobj(result.data)

        # Real part should equal original data
        assert np.allclose(result.data.real, random_patch.data)

        # Shape should be preserved
        assert result.shape == random_patch.shape

        # Coordinates should be preserved
        assert result.coords.equals(random_patch.coords)

    def test_hilbert_distance_dim(self, random_patch):
        """Test Hilbert transform along distance dimension."""
        result = random_patch.hilbert(dim="distance")

        # Result should be complex
        assert np.iscomplexobj(result.data)

        # Real part should equal original data
        assert np.allclose(result.data.real, random_patch.data)

        # Shape should be preserved
        assert result.shape == random_patch.shape

    def test_hilbert_sine_wave(self):
        """Test Hilbert transform on a known sine wave."""
        # Create a sine wave patch
        freq = 100
        patch = dc.get_example_patch("sin_wav", sample_rate=100, frequency=freq)
        time = dc.to_float(patch.get_array("time"))

        # Apply Hilbert transform
        result = patch.hilbert(dim="time")

        # For a sine wave, the imaginary part should be approximately -cos
        expected_imag = -np.cos(2 * np.pi * freq * time)

        # Allow some tolerance due to edge effects
        central_slice = slice(50, -50)  # Avoid edges
        assert np.allclose(
            result.data[0, central_slice].imag, expected_imag[central_slice], atol=0.1
        )


class TestEnvelope:
    """Tests for the envelope function."""

    @pytest.fixture(autouse=True)
    def modulated_patch_and_envelope(self):
        """Return a modulated patch"""
        # Create AM signal: A(t) * cos(w*t) where A(t) is the envelope
        t = np.linspace(0, 2, 1000)
        carrier_freq = 50  # Hz
        mod_freq = 2  # Hz

        # Envelope function (slowly varying amplitude)
        true_envelope = 1 + 0.5 * np.sin(2 * np.pi * mod_freq * t)

        # AM signal
        signal = true_envelope * np.cos(2 * np.pi * carrier_freq * t)

        # Create patch
        patch = dc.Patch(
            data=signal[np.newaxis, :],
            coords={"distance": np.array([0]), "time": t},
            dims=("distance", "time"),
        )

        # Calculate envelope
        result = patch.envelope(dim="time")
        return result, true_envelope

    def test_envelope_basic(self, random_patch):
        """Test basic envelope functionality."""
        result = random_patch.envelope(dim="time")

        # Result should be real and positive
        assert np.isrealobj(result.data)
        assert np.all(result.data >= 0)

        # Shape should be preserved
        assert result.shape == random_patch.shape

        # Coordinates should be preserved
        assert result.coords.equals(random_patch.coords)

    def test_envelope_amplitude_modulated_signal(self, modulated_patch_and_envelope):
        """Test envelope on an amplitude modulated signal."""
        patch, envelope = modulated_patch_and_envelope
        result = patch.envelope("time")
        # Should recover the original envelope (approximately)
        # Allow tolerance for edge effects and numerical precision
        central_slice = slice(100, -100)
        recovered_envelope = result.data[0, central_slice]
        expected_envelope = envelope[central_slice]

        # Check that correlation is high (envelope shape is preserved)
        correlation = np.corrcoef(recovered_envelope, expected_envelope)[0, 1]
        assert correlation > 0.95

    def test_envelope_vs_hilbert(self, random_patch):
        """Test that envelope equals abs(hilbert(signal))."""
        hilbert_result = random_patch.hilbert(dim="time")
        envelope_result = random_patch.envelope(dim="time")

        # Envelope should equal magnitude of analytic signal
        expected_envelope = np.abs(hilbert_result.data)
        assert np.allclose(envelope_result.data, expected_envelope)


class TestPhaseWeightedStack:
    """Tests for the phase weighted stacking function."""

    def test_phase_weighted_stack_basic(self, random_patch):
        """Test basic phase weighted stacking functionality."""
        result = random_patch.phase_weighted_stack(
            transform_dim="time", stack_dim="distance"
        )

        # Result should be real
        assert np.isrealobj(result.data)

        # Stacked dimension should have length 1
        distance_axis = random_patch.get_axis("distance")
        assert result.shape[distance_axis] == 1

        # Other dimensions should be preserved
        expected_shape = list(random_patch.shape)
        expected_shape[distance_axis] = 1
        assert result.shape == tuple(expected_shape)

    def test_phase_weighted_stack_time_dim(self, random_patch):
        """Test phase weighted stacking along time dimension."""
        result = random_patch.phase_weighted_stack(
            stack_dim="time", transform_dim="distance"
        )
        # Time dimension should be reduced to 1
        time_axis = random_patch.get_axis("time")
        assert result.shape[time_axis] == 1

    def test_infer_stack_dim(self):
        """Test that the stack dim can be inferred."""
        two_d_patch = dc.get_example_patch("nd_patch", dim_count=2)
        three_d_patch = dc.get_example_patch("nd_patch", dim_count=3)

        # Ensure transform dim can be inferred on two-d patch
        out_2d = two_d_patch.phase_weighted_stack(two_d_patch.dims[0])
        assert isinstance(out_2d, dc.Patch)

        # But raises on 3D patch
        with pytest.raises(ParameterError, match="can't infer transform dim"):
            dim = three_d_patch.dims[1]
            three_d_patch.phase_weighted_stack(dim)

        # But it should work fine if specified
        out_3d = three_d_patch.phase_weighted_stack(
            stack_dim=three_d_patch.dims[0],
            transform_dim=three_d_patch.dims[1],
        )
        assert isinstance(out_3d, dc.Patch)

    def test_doc_example(self):
        """Ensure the docstring examples work."""
        ricker_patch = ricker_moveout(velocity=0, peak_time=0.75)
        noise_level = ricker_patch.data.max() * 0.2
        rng = np.random.default_rng(42)
        noise = rng.normal(size=ricker_patch.data.shape) * noise_level
        patch = ricker_patch + noise

        # Make normal stack to phase weighted stack
        stack = patch.mean("distance").squeeze().data
        pws = patch.phase_weighted_stack("distance").squeeze().data
        assert stack.shape == pws.shape

    def test_phase_weighted_stack_coordinates(self, random_patch):
        """Test that coordinates are properly updated."""
        result = random_patch.phase_weighted_stack(stack_dim="distance")

        # Distance coordinate should have length 1
        assert result.shape[random_patch.get_axis("distance")] == 1

        # Time coordinate should be unchanged (use array_equal for datetime)
        assert np.array_equal(result.get_array("time"), random_patch.get_array("time"))

    def test_dim_reduce_squeeze(self, random_patch):
        """Ensure dim reduce squeeze collapses stack dimension."""
        out = random_patch.phase_weighted_stack("distance", dim_reduce="squeeze")
        assert out.ndim == 1
        assert "distance" not in out.dims

    def test_one_dim_raises(self, random_patch):
        """Test that one dimension raises error."""
        patch = random_patch.mean("distance").squeeze()
        msg = "Patch has one dimension"
        with pytest.raises(ParameterError, match=msg):
            patch.phase_weighted_stack("time")


class TestHilbertIntegration:
    """Integration tests for Hilbert functions."""

    def test_hilbert_envelope_consistency(self, random_patch):
        """Test that envelope(patch) == abs(hilbert(patch))."""
        hilbert_result = random_patch.hilbert(dim="time")
        envelope_result = random_patch.envelope(dim="time")

        expected_envelope = np.abs(hilbert_result.data)
        assert np.allclose(envelope_result.data, expected_envelope)

    def test_chain_operations(self, random_patch):
        """Test chaining signal processing operations."""
        # Test that envelope(hilbert(x)) == envelope(x) for real signals
        # Since envelope uses hilbert internally, this should work

        # Direct envelope calculation
        direct_envelope = random_patch.envelope("time")

        # Manual calculation: get hilbert transform, then take magnitude
        hilbert_result = random_patch.hilbert(dim="time")
        manual_envelope = random_patch.new(data=np.abs(hilbert_result.data))

        # Should be equivalent
        assert np.allclose(direct_envelope.data, manual_envelope.data)
