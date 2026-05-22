"""Tests for adaptive spectral filtering."""

from __future__ import annotations

import builtins
from typing import Any

import numpy as np
import pytest

import dascore as dc
from dascore.exceptions import (
    CoordError,
    MissingOptionalDependencyError,
    ParameterError,
    PatchCoordinateError,
)
from dascore.proc import adaptive_spectral_filter as adaptive_spectral_filter_func
from dascore.proc.adaptive_spectral_filter import (
    _adaptive_spectral_filter_scipy,
    _get_engine,
    _validate_window_and_overlap,
)
from dascore.utils.signal import _triangular_taper


def _patch(
    shape: tuple[int, ...],
    dims: tuple[str, ...],
    *,
    dtype=np.float32,
    time_step=np.timedelta64(4, "ms"),
    distance_step=1.0,
) -> dc.Patch:
    """Return a deterministic patch for adaptive spectral tests."""
    rng = np.random.default_rng(20260508)
    data = rng.normal(size=shape).astype(dtype)
    coords = {}
    for dim, length in zip(dims, shape, strict=True):
        if dim == "time":
            coords[dim] = np.datetime64("2020-01-01") + np.arange(length) * time_step
        elif dim == "distance":
            coords[dim] = np.arange(length, dtype=float) * distance_step
        else:
            coords[dim] = np.arange(length, dtype=float)
    return dc.Patch(data=data, coords=coords, dims=dims)


class TestAdaptiveSpectralFilter:
    """Tests for the adaptive spectral patch method."""

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_dtype_shape_dims_and_coords_preserved(self, dtype) -> None:
        """Adaptive spectral should preserve patch structure and floating dtype."""
        patch = _patch((64, 64), ("distance", "time"), dtype=dtype)

        out = patch.adaptive_spectral_filter(
            distance=16,
            time=16,
            overlap={"distance": 7, "time": 7},
            samples=True,
            engine="scipy",
        )

        assert np.asarray(out.data).dtype == np.asarray(patch.data).dtype
        assert out.shape == patch.shape
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert out.attrs.history[-1].startswith("adaptive_spectral_filter")

    def test_time_distance_reversed_dims_are_supported(self) -> None:
        """Selected dimensions need not be in a fixed order."""
        patch = _patch((64, 80), ("time", "distance"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            time=16,
            distance=32,
            overlap={"time": 7, "distance": 14},
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape
        assert out.dims == ("time", "distance")
        assert np.isfinite(out.data).all()

    def test_arbitrary_2d_dimension_names_are_supported(self) -> None:
        """Adaptive spectral should work with any two patch dimensions."""
        patch = _patch((64, 64), ("channel", "sample"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            channel=16,
            sample=16,
            overlap={"channel": 7, "sample": 7},
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape
        assert out.dims == ("channel", "sample")

    def test_requires_explicit_dimension_kwargs(self) -> None:
        """At least one dimension window is required."""
        patch = _patch((64, 64), ("channel", "sample"), dtype=np.float32)

        with pytest.raises(ParameterError, match="one or two dimension window kwargs"):
            patch.adaptive_spectral_filter(samples=True, engine="scipy")

    def test_rejects_non_positive_window(self) -> None:
        """Window sizes must resolve to positive sample counts."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        with pytest.raises(ParameterError, match=r"window.*must be positive"):
            patch.adaptive_spectral_filter(
                distance=0, time=16, samples=True, engine="scipy"
            )

    def test_one_dimension_filter_is_supported(self) -> None:
        """A single selected dimension should run the 1D spectral path."""
        patch = _patch((8, 64), ("distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            time=16,
            overlap={"time": 7},
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert np.isfinite(out.data).all()

    def test_one_dimension_filter_batches_other_dims(self) -> None:
        """Unselected dimensions should be batches for 1D filtering."""
        patch = _patch((3, 8, 64), ("shot", "distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(time=16, samples=True, engine="scipy")
        expected = np.stack(
            [
                dc.Patch(
                    data=np.asarray(patch.data)[ind],
                    coords={
                        "distance": patch.get_array("distance"),
                        "time": patch.get_array("time"),
                    },
                    dims=("distance", "time"),
                )
                .adaptive_spectral_filter(time=16, samples=True, engine="scipy")
                .data
                for ind in range(patch.shape[0])
            ]
        )

        np.testing.assert_allclose(out.data, expected, rtol=1e-5, atol=1e-5)

    def test_coordinate_unit_window_and_overlap_conversion(self) -> None:
        """Coordinate units and sample counts should resolve identically."""
        patch = _patch(
            (64, 64),
            ("distance", "time"),
            dtype=np.float32,
            distance_step=2.0,
            time_step=np.timedelta64(4, "ms"),
        )

        by_units = patch.adaptive_spectral_filter(
            distance=32.0,
            time=np.timedelta64(64, "ms"),
            overlap={"distance": 14.0, "time": np.timedelta64(28, "ms")},
            samples=False,
            engine="scipy",
        )
        by_samples = patch.adaptive_spectral_filter(
            distance=16,
            time=16,
            overlap={"distance": 7, "time": 7},
            samples=True,
            engine="scipy",
        )

        np.testing.assert_allclose(by_units.data, by_samples.data, rtol=1e-5, atol=1e-5)

    def test_default_overlap_stays_in_samples_when_windows_use_units(self) -> None:
        """Computed overlap defaults should not be interpreted as coordinate units."""
        patch = _patch(
            (64, 64),
            ("distance", "time"),
            dtype=np.float32,
            distance_step=2.0,
            time_step=np.timedelta64(4, "ms"),
        )

        by_units = patch.adaptive_spectral_filter(
            distance=32.0,
            time=np.timedelta64(64, "ms"),
            samples=False,
            engine="scipy",
        )
        by_samples = patch.adaptive_spectral_filter(
            distance=16,
            time=16,
            overlap={"distance": 6, "time": 6},
            samples=True,
            engine="scipy",
        )

        np.testing.assert_allclose(by_units.data, by_samples.data, rtol=1e-5, atol=1e-5)

    def test_partial_overlap_defaults_stay_in_samples_with_units(self) -> None:
        """Missing overlap mapping entries should stay sample-count defaults."""
        patch = _patch(
            (64, 64),
            ("distance", "time"),
            dtype=np.float32,
            distance_step=2.0,
            time_step=np.timedelta64(4, "ms"),
        )

        by_units = patch.adaptive_spectral_filter(
            distance=32.0,
            time=np.timedelta64(64, "ms"),
            overlap={"time": np.timedelta64(24, "ms")},
            samples=False,
            engine="scipy",
        )
        by_samples = patch.adaptive_spectral_filter(
            distance=16,
            time=16,
            overlap={"distance": 6, "time": 6},
            samples=True,
            engine="scipy",
        )

        np.testing.assert_allclose(by_units.data, by_samples.data, rtol=1e-5, atol=1e-5)

    def test_1d_default_overlap_stays_in_samples_with_units(self) -> None:
        """Computed 1D overlap defaults should stay in sample counts."""
        patch = _patch(
            (8, 64),
            ("distance", "time"),
            dtype=np.float32,
            time_step=np.timedelta64(4, "ms"),
        )

        by_units = patch.adaptive_spectral_filter(
            time=np.timedelta64(64, "ms"),
            samples=False,
            engine="scipy",
        )
        by_samples = patch.adaptive_spectral_filter(
            time=16,
            overlap=6,
            samples=True,
            engine="scipy",
        )

        np.testing.assert_allclose(by_units.data, by_samples.data, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "shape,dims,kwargs",
        [
            ((3, 64, 64), ("shot", "distance", "time"), {"distance": 16, "time": 16}),
            (
                (2, 3, 64, 64),
                ("component", "shot", "distance", "time"),
                {"distance": 16, "time": 16},
            ),
            ((64, 3, 64), ("distance", "shot", "time"), {"distance": 16, "time": 16}),
        ],
    )
    def test_batches_over_non_selected_dimensions(
        self,
        shape: tuple[int, ...],
        dims: tuple[str, ...],
        kwargs: dict[str, int],
    ) -> None:
        """Extra dimensions should be processed as independent 2D batches."""
        patch = _patch(shape, dims, dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            **kwargs,
            overlap={dim: max(value // 2 - 2, 0) for dim, value in kwargs.items()},
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape
        assert out.dims == patch.dims
        assert out.coords == patch.coords
        assert np.isfinite(out.data).all()

    def test_batched_output_matches_independent_2d_calls(self) -> None:
        """Batched filtering should match independent 2D patch calls."""
        patch = _patch((4, 32, 32), ("depth", "distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            distance=16, time=16, samples=True, engine="scipy"
        )
        expected = np.stack(
            [
                dc.Patch(
                    data=np.asarray(patch.data)[ind],
                    coords={
                        "distance": patch.get_array("distance"),
                        "time": patch.get_array("time"),
                    },
                    dims=("distance", "time"),
                )
                .adaptive_spectral_filter(
                    distance=16, time=16, samples=True, engine="scipy"
                )
                .data
                for ind in range(patch.shape[0])
            ]
        )

        np.testing.assert_allclose(out.data, expected, rtol=1e-5, atol=1e-5)

    def test_rejects_unknown_overlap_dimension(self) -> None:
        """Overlap mappings may only name selected dimensions."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        with pytest.raises(ParameterError, match="overlap contains dimensions"):
            patch.adaptive_spectral_filter(
                distance=16,
                time=16,
                overlap={"distance": 7, "bad": 7},
                samples=True,
                engine="scipy",
            )

    def test_scalar_overlap_applies_to_both_dimensions(self) -> None:
        """A scalar overlap should apply to each selected dimension."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            distance=16,
            time=16,
            overlap=7,
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape

    def test_scalar_overlap_applies_to_one_dimension(self) -> None:
        """A scalar overlap should also work for 1D filtering."""
        patch = _patch((8, 64), ("distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            time=16,
            overlap=7,
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape

    def test_zero_overlap_is_supported(self) -> None:
        """Zero overlap should use an all-ones reconstruction taper."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(
            distance=16,
            time=16,
            overlap=0,
            samples=True,
            engine="scipy",
        )

        assert out.shape == patch.shape
        assert np.isfinite(out.data).all()

    def test_one_dimension_auto_uses_scipy(self) -> None:
        """Auto mode should use SciPy for 1D filtering."""
        patch = _patch((8, 64), ("distance", "time"), dtype=np.float32)

        out = patch.adaptive_spectral_filter(time=16, samples=True, engine="auto")

        assert out.shape == patch.shape

    def test_numba_rejects_one_dimension(self) -> None:
        """The optional numba engine is intentionally 2D-only."""
        patch = _patch((8, 64), ("distance", "time"), dtype=np.float32)

        with pytest.raises(ParameterError, match="two selected dimensions"):
            patch.adaptive_spectral_filter(time=16, samples=True, engine="numba")

    @pytest.mark.parametrize(
        "kwargs,match",
        [
            ({"exponent": np.nan}, "exponent must be finite"),
            ({"distance": 15}, "power of two"),
            ({"overlap": {"distance": 8}}, "too large"),
            ({"overlap": {"distance": -1}}, "non-negative"),
        ],
    )
    def test_patch_validation_branches(
        self, kwargs: dict[str, Any], match: str
    ) -> None:
        """Patch-level validation should raise ParameterError."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)
        call_kwargs = {"distance": 16, "time": 16, "samples": True, "engine": "scipy"}
        call_kwargs.update(kwargs)

        with pytest.raises(ParameterError, match=match):
            patch.adaptive_spectral_filter(**call_kwargs)

    def test_rejects_missing_dimension_kwarg(self) -> None:
        """Unknown dimensions should raise the normal patch coordinate error."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        with pytest.raises(PatchCoordinateError, match="not found"):
            patch.adaptive_spectral_filter(
                distance=16, missing=16, samples=True, engine="scipy"
            )

    def test_uneven_coordinate_conversion_raises_when_not_samples(self) -> None:
        """Coordinate-unit windows require evenly sampled coordinates."""
        patch = dc.get_example_patch("wacky_dim_coords_patch")

        with pytest.raises(CoordError):
            patch.adaptive_spectral_filter(
                distance=16, time=16, samples=False, engine="scipy"
            )

    def test_nan_values_remain_supported(self) -> None:
        """NaNs may propagate but should not produce infinities."""
        patch = dc.get_example_patch("patch_with_null", shape=(64, 64))

        out = patch.adaptive_spectral_filter(
            distance=16, time=16, samples=True, engine="scipy"
        )
        out_data = np.asarray(out.data)

        assert out.shape == patch.shape
        assert np.isnan(out_data).any()
        assert not np.isinf(out_data).any()

    def test_invalid_engine_raises(self) -> None:
        """Engine values should be constrained."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        with pytest.raises(ParameterError, match="engine"):
            patch.adaptive_spectral_filter(
                distance=16,
                time=16,
                samples=True,
                engine="bad",  # type: ignore[arg-type]
            )

    def test_proc_export_is_function(self) -> None:
        """The processing module should expose the direct patch function."""
        patch = _patch((64, 64), ("distance", "time"), dtype=np.float32)

        out = adaptive_spectral_filter_func(
            patch, distance=16, time=16, samples=True, engine="scipy"
        )

        assert out.shape == patch.shape


class TestAdaptiveSpectralCore:
    """Tests for plain array adaptive spectral helpers."""

    def test_triangular_taper_values(self) -> None:
        """The shared taper should match the overlap-add ramp geometry."""
        taper = _triangular_taper((8, 8), (2, 2))
        expected_1d = np.array([0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25])

        np.testing.assert_allclose(taper, expected_1d[:, None] * expected_1d[None, :])
        assert taper.dtype == np.float32

    def test_triangular_taper_all_ones_when_plateau_matches_window(self) -> None:
        """A full-window plateau should support zero-overlap reconstruction."""
        taper = _triangular_taper((8, 8), (8, 8))

        np.testing.assert_array_equal(taper, np.ones((8, 8), dtype=np.float32))

    def test_triangular_taper_one_dimensional(self) -> None:
        """The shared taper should also support 1D reconstruction."""
        taper = _triangular_taper((8,), (2,))
        expected = np.array([0.25, 0.5, 0.75, 1.0, 1.0, 0.75, 0.5, 0.25])

        np.testing.assert_allclose(taper, expected)
        assert taper.dtype == np.float32

    def test_triangular_taper_cache_is_not_mutated_by_callers(self) -> None:
        """Callers should receive a copy of the cached taper."""
        taper = _triangular_taper((16, 16), (2, 2))
        expected = taper.copy()

        taper[...] = -1.0
        actual = _triangular_taper((16, 16), (2, 2))

        np.testing.assert_array_equal(actual, expected)

    @pytest.mark.parametrize(
        "window_size,plateau,match",
        [
            ((16, 16), (17, 2), "Plateau cannot"),
            ((16, 16), (-1, 2), "non-negative"),
            ((15, 16), (2, 2), "Window sizes must be even"),
            ((16, 16), (2,), "same length"),
            ((16, 16, 16), (2, 2, 2), "one- and two-dimensional"),
        ],
    )
    def test_triangular_taper_rejects_invalid_geometry(
        self,
        window_size: tuple[int, int],
        plateau: tuple[int, int],
        match: str,
    ) -> None:
        """Invalid taper geometry should raise."""
        with pytest.raises(ValueError, match=match):
            _triangular_taper(window_size, plateau)

    @pytest.mark.parametrize(
        "window_size,overlap,match",
        [
            ((15, 16), (7, 7), "power of two"),
            ((4, 16), (1, 7), "greater than 4"),
            ((16, 16), (-1, 7), "non-negative"),
            ((16, 16), (8, 7), "too large"),
            ((16.0, 16), (7, 7), "must be an integer"),
            ((16, 16), (7.0, 7), "must be an integer"),
            ((16,), (7, 7), "match the input dimensionality"),
        ],
    )
    def test_core_rejects_invalid_window_and_overlap(
        self,
        window_size: tuple[Any, Any],
        overlap: tuple[Any, Any],
        match: str,
    ) -> None:
        """Direct array API should validate window geometry."""
        data = np.ones((32, 32), dtype=np.float32)

        with pytest.raises(ValueError, match=match):
            _adaptive_spectral_filter_scipy(
                data, window_size=window_size, overlap=overlap
            )

    def test_core_rejects_non_1d_or_2d_input(self) -> None:
        """Direct array API is 1D or 2D only."""
        data = np.ones((2, 16, 16), dtype=np.float32)

        with pytest.raises(ValueError, match="1D or 2D input"):
            _adaptive_spectral_filter_scipy(data, window_size=(16, 16), overlap=(7, 7))

    def test_core_rejects_non_finite_exponent(self) -> None:
        """Exponent must be finite."""
        data = np.ones((32, 32), dtype=np.float32)

        with pytest.raises(ValueError, match="exponent must be finite"):
            _adaptive_spectral_filter_scipy(
                data, window_size=(16, 16), overlap=(7, 7), exponent=np.nan
            )

    def test_direct_array_api_returns_float32_for_integer_inputs(self) -> None:
        """Non-floating array inputs should return float32 outputs."""
        data = np.ones((32, 32), dtype=np.int16)

        out = _adaptive_spectral_filter_scipy(
            data, window_size=(16, 16), overlap=(7, 7)
        )

        assert out.dtype == np.float32

    def test_direct_array_api_normalizes_power(self) -> None:
        """The SciPy path should run power normalization."""
        data = np.ones((32, 32), dtype=np.float32)

        out = _adaptive_spectral_filter_scipy(
            data,
            window_size=(16, 16),
            overlap=(7, 7),
            exponent=0.5,
            normalize_power=True,
        )

        assert out.shape == data.shape
        assert np.isfinite(out).all()

    def test_direct_array_api_filters_one_dimensional_data(self) -> None:
        """The SciPy helper should support 1D arrays."""
        data = np.ones(32, dtype=np.float32)

        out = _adaptive_spectral_filter_scipy(
            data,
            window_size=(16,),
            overlap=(7,),
            exponent=0.5,
            normalize_power=True,
        )

        assert out.shape == data.shape
        assert np.isfinite(out).all()

    def test_direct_array_api_supports_zero_overlap(self) -> None:
        """The direct SciPy helper should accept non-overlapping windows."""
        data = np.ones((32, 32), dtype=np.float32)

        out = _adaptive_spectral_filter_scipy(
            data, window_size=(16, 16), overlap=(0, 0)
        )

        assert out.shape == data.shape
        assert np.isfinite(out).all()

    def test_auto_engine_falls_back_when_numba_missing(self, monkeypatch) -> None:
        """Auto engine should fall back to SciPy when optional deps are absent."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dascore.proc._adaptive_spectral_filter_numba":
                raise ImportError("simulated missing numba engine")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        assert _get_engine("auto", 2) is _adaptive_spectral_filter_scipy

    def test_numba_engine_raises_when_missing(self, monkeypatch) -> None:
        """Explicit numba engine should raise when optional deps are absent."""
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "dascore.proc._adaptive_spectral_filter_numba":
                raise ImportError("simulated missing numba engine")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)

        with pytest.raises(MissingOptionalDependencyError, match="engine='numba'"):
            _get_engine("numba", 2)

    def test_get_engine_uses_scipy_for_one_dimensional_auto(self) -> None:
        """Auto mode should use SciPy when one dimension is selected."""
        assert _get_engine("auto", 1) is _adaptive_spectral_filter_scipy

    def test_get_engine_rejects_numba_for_one_dimension(self) -> None:
        """Numba is intentionally unavailable for the 1D helper path."""
        with pytest.raises(ParameterError, match="two selected dimensions"):
            _get_engine("numba", 1)

    def test_private_window_overlap_validator_rejects_negative_overlap(self) -> None:
        """The private window validator should guard negative overlaps."""
        with pytest.raises(ParameterError, match="non-negative"):
            _validate_window_and_overlap(("distance", "time"), (16, 16), (-1, 7), 0.3)

    @pytest.mark.parametrize("exponent", [0.0, 0.3])
    def test_numba_and_scipy_match_when_numba_available(self, exponent) -> None:
        """The optional Numba 2D path should match the SciPy implementation."""
        numba_mod = pytest.importorskip("dascore.proc._adaptive_spectral_filter_numba")
        rng = np.random.default_rng(20260511)
        data = rng.normal(size=(32, 32)).astype(np.float32)

        numba = numba_mod._adaptive_spectral_filter_numba(
            data,
            window_size=(16, 16),
            overlap=(7, 7),
            exponent=exponent,
            normalize_power=True,
        )
        scipy = _adaptive_spectral_filter_scipy(
            data,
            window_size=(16, 16),
            overlap=(7, 7),
            exponent=exponent,
            normalize_power=True,
        )

        np.testing.assert_allclose(scipy, numba, rtol=1e-5, atol=1e-5)

    def test_auto_engine_uses_numba_when_available(self) -> None:
        """Auto engine should use the Numba 2D path when optional deps import."""
        numba_mod = pytest.importorskip("dascore.proc._adaptive_spectral_filter_numba")

        assert _get_engine("auto", 2) is numba_mod._adaptive_spectral_filter_numba

    def test_numba_private_helpers_run_in_python(self) -> None:
        """The fast-engine helpers should be directly testable in Python."""
        numba_mod = pytest.importorskip("dascore.proc._adaptive_spectral_filter_numba")
        padded = np.arange(16, dtype=np.float32).reshape(4, 4)
        tile = np.zeros((2, 2), dtype=np.float32)

        assert numba_mod._tile_indices_from_parity_index(3, 2, 1, 0) == (3, 2)
        assert numba_mod._tile_bounds(1, 1, 2, 2, 1, 1, 4, 4) == (1, 1, 2, 2)
        numba_mod._copy_padded_tile(padded, tile, 1, 1, 2, 2)
        np.testing.assert_array_equal(tile, padded[1:3, 1:3])
        assert numba_mod._complex_power(3 + 4j) == np.float32(5.0)

        spec = np.array([[3 + 4j, 0j]], dtype=np.complex64)
        assert numba_mod._max_spectral_power(spec) == np.float32(5.0)
        assert numba_mod._max_spectral_power_numba_impl(spec) == np.float32(5.0)
        weighted = spec.copy()
        numba_mod._apply_spectral_weight(weighted, 1.0, False)
        np.testing.assert_allclose(weighted[0, 0], spec[0, 0] * 5.0)

        weighted = spec.copy()
        numba_mod._apply_spectral_weight(weighted, 0.3, True)
        assert np.isfinite(weighted).all()

        weighted = spec.copy()
        numba_mod._apply_spectral_weight_numba_impl(weighted, 0.3, True)
        assert np.isfinite(weighted).all()

        weighted = spec.copy()
        numba_mod._apply_spectral_weight_numba_impl(weighted, 1.0, False)
        np.testing.assert_allclose(weighted[0, 0], spec[0, 0] * 5.0)

        zeros = np.array([[0j]], dtype=np.complex64)
        numba_mod._apply_spectral_weight(zeros, 0.3, True)
        assert zeros[0, 0] == 0j

        zeros = np.array([[0j]], dtype=np.complex64)
        numba_mod._apply_spectral_weight_numba_impl(zeros, 0.3, True)
        assert zeros[0, 0] == 0j

        filtered = np.zeros_like(padded)
        taper = np.ones((2, 2), dtype=np.float32)
        numba_mod._overlap_add_tile(filtered, tile, taper, 1, 1, 2, 2)
        np.testing.assert_array_equal(filtered[1:3, 1:3], tile)

    def test_numba_private_tile_group_runs_in_python(self) -> None:
        """The tile group algorithm should run without JIT for coverage."""
        numba_mod = pytest.importorskip("dascore.proc._adaptive_spectral_filter_numba")
        data = np.ones((8, 8), dtype=np.float32)
        working, _, stride, taper, padded, filtered, n_tiles = (
            numba_mod._prepare_work_arrays(data, window_size=(8, 8), overlap=(3, 3))
        )

        numba_mod._process_tile_group_python(
            padded,
            filtered,
            taper,
            8,
            8,
            stride[0],
            stride[1],
            n_tiles[0],
            n_tiles[1],
            0,
            0,
            0.0,
            False,
        )
        numba_mod._process_tile_group_python(
            padded,
            filtered,
            taper,
            8,
            8,
            stride[0],
            stride[1],
            n_tiles[0],
            n_tiles[1],
            0,
            0,
            0.5,
            True,
        )
        numba_mod._process_tile_group_numba_impl(
            padded,
            filtered,
            taper,
            8,
            8,
            stride[0],
            stride[1],
            n_tiles[0],
            n_tiles[1],
            0,
            0,
            0.5,
            True,
        )
        out = numba_mod._finalize_output(filtered, working, data.dtype, stride)

        assert out.shape == data.shape
        assert np.isfinite(out).all()
