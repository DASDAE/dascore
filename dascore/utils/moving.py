"""
Unified interface for moving window operations with automatic engine selection.

This module provides a generic interface for 1D moving window operations
that can use either scipy or bottleneck backends, with automatic fallback
when optional dependencies are missing. Bottleneck median windows are aligned
with centered scipy semantics away from edges.
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
    data: np.ndarray,
    window: int,
    operation: str,
    axis: int,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
    ddof: int = 0,
) -> np.ndarray:
    """Apply scipy operation with proper handling."""
    module_name, func_name = OPERATION_REGISTRY[operation]["scipy"]
    func = _get_engine_function("scipy", operation)
    scipy_kwargs = {"mode": mode, "cval": cval, "origin": origin}

    # Apply function with appropriate parameters
    if func_name == "uniform_filter1d":
        # Only promote integer or boolean dtypes to float, preserve float/complex
        data_kind = data.dtype.kind
        data = data.astype(np.float64) if data_kind in {"i", "b", "u"} else data
        uniform = func(data, size=window, axis=axis, **scipy_kwargs)
        if operation.lower() == "sum":
            # For sum, uniform_filter1d computes mean, so we scale by window size
            uniform *= window
        return uniform
    elif func_name == "median_filter":
        # Need full size tuple for median_filter
        size = [1] * data.ndim
        size[axis] = window
        return func(data, size=tuple(size), **scipy_kwargs)
    else:
        # 1D filters (minimum_filter1d, maximum_filter1d)
        return func(data, size=window, axis=axis, **scipy_kwargs)


def _requires_scipy_boundary(mode: str, origin: int) -> bool:
    """Return True if scipy boundary handling is explicitly requested."""
    return mode != "reflect" or origin != 0


def _apply_bottleneck_median(
    data: np.ndarray,
    window: int,
    axis: int,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
) -> np.ndarray:
    """Apply bottleneck median with interior centered-window alignment."""
    if window % 2 == 0 or _requires_scipy_boundary(mode, origin):
        return _apply_scipy_operation(
            data,
            window,
            "median",
            axis,
            mode=mode,
            cval=cval,
            origin=origin,
            min_count=min_count,
        )

    func = _get_engine_function("bottleneck", "median")
    result = func(data, window=window, axis=axis, min_count=min_count)
    shift = window // 2
    if shift:
        destination = [slice(None)] * data.ndim
        source = [slice(None)] * data.ndim
        destination[axis] = slice(0, -shift)
        source[axis] = slice(shift, None)
        result[tuple(destination)] = result[tuple(source)]
    return result


def _apply_bottleneck_operation(
    data: np.ndarray,
    window: int,
    operation: str,
    axis: int,
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
    ddof: int = 0,
) -> np.ndarray:
    """Apply bottleneck operation with error handling."""
    if operation == "median":
        return _apply_bottleneck_median(
            data,
            window,
            axis,
            mode=mode,
            cval=cval,
            origin=origin,
            min_count=min_count,
        )

    func = _get_engine_function("bottleneck", operation)
    if _requires_scipy_boundary(mode, origin):
        if OPERATION_REGISTRY[operation]["scipy"] is None:
            msg = (
                f"Operation '{operation}' cannot use scipy boundary options with "
                "the bottleneck engine."
            )
            raise ParameterError(msg)
        return _apply_scipy_operation(
            data,
            window,
            operation,
            axis,
            mode=mode,
            cval=cval,
            origin=origin,
            min_count=min_count,
            ddof=ddof,
        )

    extra_kwargs = {"ddof": ddof} if operation == "std" else {}
    result = func(data, window=window, axis=axis, min_count=min_count, **extra_kwargs)
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
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
    ddof: int = 0,
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
    mode
        Boundary mode. Non-default values use scipy for operations where
        bottleneck cannot preserve scipy boundary semantics.
    cval
        Fill value when ``mode="constant"``.
    origin
        Filter placement. Nonzero values use scipy for operations where
        bottleneck cannot preserve scipy boundary semantics.
    min_count
        Minimum number of observations in a Bottleneck window.
    ddof
        Delta degrees of freedom for Bottleneck standard deviation.

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

    return wrapper_func(
        data,
        window,
        operation,
        axis,
        mode=mode,
        cval=cval,
        origin=origin,
        min_count=min_count,
        ddof=ddof,
    )


# Convenience functions - much simpler now!
def move_median(
    data: np.ndarray,
    window: int,
    axis: int = 0,
    engine: str = "auto",
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
) -> np.ndarray:
    """Moving median filter."""
    return moving_window(
        data,
        window,
        "median",
        axis,
        engine,
        mode=mode,
        cval=cval,
        origin=origin,
        min_count=min_count,
    )


def move_mean(
    data: np.ndarray,
    window: int,
    axis: int = 0,
    engine: str = "auto",
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
) -> np.ndarray:
    """Moving mean filter."""
    return moving_window(
        data,
        window,
        "mean",
        axis,
        engine,
        mode=mode,
        cval=cval,
        origin=origin,
        min_count=min_count,
    )


def move_std(
    data: np.ndarray,
    window: int,
    axis: int = 0,
    engine: str = "auto",
    min_count: int = 1,
    ddof: int = 0,
) -> np.ndarray:
    """Moving standard deviation filter."""
    return moving_window(
        data, window, "std", axis, engine, min_count=min_count, ddof=ddof
    )


def move_sum(
    data: np.ndarray,
    window: int,
    axis: int = 0,
    engine: str = "auto",
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
) -> np.ndarray:
    """Moving sum filter."""
    return moving_window(
        data,
        window,
        "sum",
        axis,
        engine,
        mode=mode,
        cval=cval,
        origin=origin,
        min_count=min_count,
    )


def move_min(
    data: np.ndarray,
    window: int,
    axis: int = 0,
    engine: str = "auto",
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
) -> np.ndarray:
    """Moving minimum filter."""
    return moving_window(
        data,
        window,
        "min",
        axis,
        engine,
        mode=mode,
        cval=cval,
        origin=origin,
        min_count=min_count,
    )


def move_max(
    data: np.ndarray,
    window: int,
    axis: int = 0,
    engine: str = "auto",
    mode: str = "reflect",
    cval: float = 0.0,
    origin: int = 0,
    min_count: int = 1,
) -> np.ndarray:
    """Moving maximum filter."""
    return moving_window(
        data,
        window,
        "max",
        axis,
        engine,
        mode=mode,
        cval=cval,
        origin=origin,
        min_count=min_count,
    )
