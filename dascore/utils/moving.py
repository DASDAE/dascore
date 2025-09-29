"""
Unified interface for moving window operations with automatic engine selection.

This module provides a generic interface for 1D moving window operations
that can use either scipy or bottleneck backends, with automatic fallback
when optional dependencies are missing.
"""

from __future__ import annotations

import importlib
import warnings
from collections.abc import Callable
from functools import cache
from typing import Literal

import numpy as np

from dascore.exceptions import ParameterError
from dascore.utils.misc import optional_import

bn = optional_import("bottleneck", on_missing="ignore")

# Operation registry mapping operations to engine implementations
OPERATION_REGISTRY = {
    "median": {
        "scipy": ("scipy.ndimage", "median_filter"),
        "bottleneck": ("bottleneck", "move_median"),
    },
    "mean": {
        "scipy": ("scipy.ndimage", "uniform_filter1d"),
        "bottleneck": ("bottleneck", "move_mean"),
    },
    "std": {
        "scipy": None,
        "bottleneck": ("bottleneck", "move_std"),
    },
    "sum": {
        "scipy": ("scipy.ndimage", "uniform_filter1d"),  # Needs scaling
        "bottleneck": ("bottleneck", "move_sum"),
    },
    "min": {
        "scipy": ("scipy.ndimage", "minimum_filter1d"),
        "bottleneck": ("bottleneck", "move_min"),
    },
    "max": {
        "scipy": ("scipy.ndimage", "maximum_filter1d"),
        "bottleneck": ("bottleneck", "move_max"),
    },
}


def _apply_scipy_operation(
    data: np.ndarray, window: int, operation: str, axis: int, **kwargs
) -> np.ndarray:
    """Apply scipy operation with proper handling."""
    module_name, func_name = OPERATION_REGISTRY[operation]["scipy"]
    func = _get_engine_function("scipy", operation)

    # Apply function with appropriate parameters
    if func_name == "uniform_filter1d":
        # Only promote integer or boolean dtypes to float, preserve float/complex
        data_kind = data.dtype.kind
        data = data.astype(np.float64) if data_kind in {"i", "b", "u"} else data
        uniform = func(data, size=window, axis=axis, **kwargs)
        if operation.lower() == "sum":
            # For sum, uniform_filter1d computes mean, so we scale by window size
            uniform *= window
        return uniform
    elif func_name == "median_filter":
        # Need full size tuple for median_filter
        size = [1] * data.ndim
        size[axis] = window
        return func(data, size=tuple(size), **kwargs)
    else:
        # 1D filters (minimum_filter1d, maximum_filter1d)
        return func(data, size=window, axis=axis, **kwargs)


def _apply_bottleneck_operation(
    data: np.ndarray, window: int, operation: str, axis: int, **kwargs
) -> np.ndarray:
    """Apply bottleneck operation with error handling."""
    func = _get_engine_function("bottleneck", operation)

    # Extract user-specified min_count with default of 1
    min_count = kwargs.pop("min_count", 1)

    # Filter out scipy-specific kwargs that bottleneck doesn't accept
    scipy_only_kwargs = {"mode", "origin", "cval"}
    bottleneck_kwargs = {k: v for k, v in kwargs.items() if k not in scipy_only_kwargs}

    result = func(
        data, window=window, axis=axis, min_count=min_count, **bottleneck_kwargs
    )
    return result


# Engine function wrappers (defined after functions)
ENGINE_WRAPPERS = {
    "scipy": _apply_scipy_operation,
    "bottleneck": _apply_bottleneck_operation,
}


@cache
def _get_module(module_name):
    """Import a module based on its name."""
    return importlib.import_module(module_name)


@cache
def _get_available_engines() -> tuple[str, ...]:
    """Get list of available engines (cached)."""
    bottle_list = [] if bn is None else ["bottleneck"]
    return tuple(["scipy", *bottle_list])


def _get_engine_function(engine: str, func_name: str) -> Callable | None:
    """Get and cache engine function."""
    module_name, func_name = OPERATION_REGISTRY[func_name][engine]
    mod = _get_module(module_name)
    return getattr(mod, func_name)


def _select_engine(preferred: str, operation: str) -> str:
    """Select best available engine with fallback."""
    available = _get_available_engines()
    preferred = "bottleneck" if preferred == "auto" else preferred

    if preferred not in available:
        engine = available[0]
        msg = (
            f"Preferred engine {preferred} is not available; falling back to {engine}. "
            "It may require an additional installation."
        )
        warnings.warn(msg, UserWarning, stacklevel=4)
        preferred = engine

    # Check if operation is available in preferred engine
    registry_entry = OPERATION_REGISTRY.get(operation, {}).get(preferred)
    if registry_entry is None:
        # Try fallback
        msg = f"Operation '{operation}' is not available for engine: {preferred}"
        raise ParameterError(msg)

    return preferred


def moving_window(
    data: np.ndarray,
    window: int,
    operation: str,
    axis: int = 0,
    engine: Literal["auto", "scipy", "bottleneck"] = "auto",
    **kwargs,
) -> np.ndarray:
    """
    Generic moving window operation with automatic engine selection.

    Parameters
    ----------
    data : array-like
        Input data
    window : int
        Window size
    operation : str
        Operation name ("median", "mean", "std", "sum", "min", "max")
    axis : int, default 0
        Axis along which to operate
    engine : {"auto", "scipy", "bottleneck"}, default "auto"
        Engine to use
    **kwargs
        Additional arguments passed to the engine function

    Returns
    -------
    np.ndarray
        Result of moving window operation
    """
    data = np.asarray(data)
    # Validate inputs
    if window <= 0:
        raise ParameterError("Window size must be positive")
    if operation not in OPERATION_REGISTRY:
        raise ParameterError(f"Unknown operation: {operation}")
    if window > data.shape[axis]:
        msg = (
            f"Window size ({window}) larger than data size ({data.shape[axis]}) "
            f"along axis {axis}"
        )
        raise ParameterError(msg)

    # Select engine and apply operation
    selected_engine = _select_engine(engine, operation)
    wrapper_func = ENGINE_WRAPPERS[selected_engine]

    return wrapper_func(data, window, operation, axis, **kwargs)


# Convenience functions - much simpler now!
def move_median(
    data: np.ndarray, window: int, axis: int = 0, engine: str = "auto", **kwargs
) -> np.ndarray:
    """Moving median filter."""
    return moving_window(data, window, "median", axis, engine, **kwargs)


def move_mean(
    data: np.ndarray, window: int, axis: int = 0, engine: str = "auto", **kwargs
) -> np.ndarray:
    """Moving mean filter."""
    return moving_window(data, window, "mean", axis, engine, **kwargs)


def move_std(
    data: np.ndarray, window: int, axis: int = 0, engine: str = "auto", **kwargs
) -> np.ndarray:
    """Moving standard deviation filter."""
    return moving_window(data, window, "std", axis, engine, **kwargs)


def move_sum(
    data: np.ndarray, window: int, axis: int = 0, engine: str = "auto", **kwargs
) -> np.ndarray:
    """Moving sum filter."""
    return moving_window(data, window, "sum", axis, engine, **kwargs)


def move_min(
    data: np.ndarray, window: int, axis: int = 0, engine: str = "auto", **kwargs
) -> np.ndarray:
    """Moving minimum filter."""
    return moving_window(data, window, "min", axis, engine, **kwargs)


def move_max(
    data: np.ndarray, window: int, axis: int = 0, engine: str = "auto", **kwargs
) -> np.ndarray:
    """Moving maximum filter."""
    return moving_window(data, window, "max", axis, engine, **kwargs)
