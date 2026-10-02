"""
Patch functions based on the Hilbert transform.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from dascore.constants import DIM_REDUCE_DOCS
from dascore.core.processor import PatchProcessor
from dascore.exceptions import ParameterError
from dascore.proc.basic import _as_float
from dascore.utils.array_api import array_namespace, asarray_like
from dascore.utils.docs import compose_docstring
from dascore.utils.imports import lazy_import

scipy_hilbert = lazy_import("scipy.signal", "hilbert")


def analytic_signal(data, axis: int):
    """
    Return the analytic signal of real data along an axis, with the array API.

    The discrete Hilbert transform scipy uses: zero the negative frequencies
    of the FFT, double the positive ones, and transform back.
    """
    xp = array_namespace(data)
    if xp.isdtype(data.dtype, "complex floating"):
        msg = "x must be real."
        raise ValueError(msg)
    # A chunked (dask) array transforms along one chunk only.
    if hasattr(data, "rechunk"):
        data = data.rechunk({axis: -1})
    single = data.dtype == xp.float32
    size = data.shape[axis]
    weights = np.zeros(size, dtype=np.float32 if single else np.float64)
    weights[0] = 1
    if size % 2 == 0:
        weights[size // 2] = 1
        weights[1 : size // 2] = 2
    else:
        weights[1 : (size + 1) // 2] = 2
    shape = [1] * data.ndim
    shape[axis] = size
    weights = asarray_like(weights.reshape(shape), data)
    # The standard's fft takes complex input only.
    complex_dtype = xp.complex64 if single else xp.complex128
    spectrum = xp.fft.fft(xp.astype(data, complex_dtype), axis=axis)
    return xp.fft.ifft(spectrum * weights, axis=axis)


class Hilbert(PatchProcessor):
    """
    Perform a Hilbert transform on a patch.

    The Hilbert transform returns the analytic signal (complex-valued)
    where the real part is the original signal and the imaginary part
    is the Hilbert transform of the signal.

    Parameters
    ----------
    patch
        The patch to transform.
    dim
        The dimension along which to apply the Hilbert transform.

    Returns
    -------
    PatchType
        A patch with a complex data array representing the analytic signal.

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>>
    >>> patch = dc.get_example_patch()
    >>> analytic = patch.hilbert(dim="time")
    >>> # Real part is original signal
    >>> assert np.allclose(analytic.data.real, patch.data)
    """

    dim: str

    data_type = ""

    def get_metadata(self, meta):
        """Return the axis, once it is known to be evenly sampled."""
        meta.get_coord(self.dim, require_evenly_sampled=True)
        return meta, {"axis": meta.get_axis(self.dim)}

    def numpy_kernel(self, data, *, axis):
        """Return the analytic signal along the axis."""
        return scipy_hilbert(data, axis=axis)

    def kernel(self, data, *, axis):
        """Return the analytic signal along the axis, through the FFT."""
        return analytic_signal(data, axis)


class Envelope(Hilbert):
    """
    Calculate the envelope of a signal using the Hilbert transform.

    The envelope is the magnitude of the analytic signal, which represents
    the instantaneous amplitude of the signal.

    Parameters
    ----------
    patch
        The patch to process.
    dim
        The dimension along which to calculate the envelope.

    Returns
    -------
    PatchType
        A patch containing the envelope (real-valued, positive).

    Examples
    --------
    >>> import dascore as dc
    >>> import numpy as np
    >>>
    >>> patch = dc.get_example_patch()
    >>> env = patch.envelope(dim="time")
    >>> # Envelope is always positive
    >>> assert np.all(env.data >= 0)
    """

    data_type = "envelope"

    def numpy_kernel(self, data, *, axis):
        """Return the magnitude of the analytic signal along the axis."""
        return np.abs(scipy_hilbert(data, axis=axis))

    def kernel(self, data, *, axis):
        """Return the magnitude of the analytic signal along the axis."""
        return array_namespace(data).abs(analytic_signal(data, axis))


def _infer_transform_dim(patch, stack_dim):
    """Try to infer transform dimension."""
    dims = set(patch.dims) - {stack_dim}
    if len(dims) > 1:
        msg = "Patch has more than two dimensions, can't infer transform dim."
        raise ParameterError(msg)
    if len(dims) == 0:
        msg = (
            f"Patch has one dimension: {patch.dims}. The phase_weighted_stack"
            f"requires at least two."
        )
        raise ParameterError(msg)
    return next(iter(dims))


def _phase_weighted_stack(data, analytic, axis: int, power, squeeze: bool):
    """Return the mean along an axis weighted by the coherence of its phases."""
    xp = array_namespace(data)
    # Get unit phasors. Use eps here to avoid unstable division by 0.
    eps = xp.finfo(xp.real(analytic).dtype).eps
    amp = xp.maximum(xp.abs(analytic), eps)
    unit_phasors = analytic / amp
    mean_phasor = xp.mean(unit_phasors, axis=axis, keepdims=True)
    # Weight by coherence: |mean_phasor| runs from 0 (random phases) to 1
    # (all phases aligned).
    weights = xp.abs(mean_phasor) ** power
    # Stack original data and apply weights (we can do this since weights
    # are common across all samples)
    out = xp.mean(_as_float(data), axis=axis, keepdims=True) * weights
    return xp.squeeze(out, axis=axis) if squeeze else out


@compose_docstring(dim_reduce=DIM_REDUCE_DOCS)
class PhaseWeightedStack(PatchProcessor):
    """
    Apply phase weighted stacking to enhance coherent signals.

    Phase weighted stacking uses the instantaneous phase coherence
    to weight the stacking process, enhancing coherent signals while
    suppressing incoherent noise.

    Parameters
    ----------
    patch
        The patch to stack.
    stack_dim
        The dimension over which the data should be stacked. For typical
        use cases this will be "distance".
    transform_dim
        The dimension along which to perform the Hilbert transform.
        For typical use cases this will be "time". If not provided, it will
        be inferred as the other dimension besides stack dim. If the patch
        has more than 2 dimensions and transform_dim is None, a ParameterError
        is raised.
    power
        The power to which the phase coherence is raised. Higher values
        give more weight to coherent signals.
    {dim_reduce}

    Returns
    -------
    PatchType
        A patch with the phase-weighted stack along the specified dimension.
        The specified dimension will have length 1 unless `dim_reduce="squeeze"`,
        in which case the dimension is removed.

    Notes
    -----
    Phase weighted stacking is described in @schimmel1997noise.

    Examples
    --------
    >>> import dascore as dc
    >>> from dascore.examples import ricker_moveout
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>>
    >>> # Create ricker wavelet with noise
    >>> ricker_patch = ricker_moveout(velocity=0)
    >>> noise_level = ricker_patch.data.max() * 0.2
    >>> noise = np.random.normal(size=ricker_patch.data.shape) * noise_level
    >>> patch = ricker_patch + noise
    >>>
    >>> # Make normal stack to phase weighted stack
    >>> stack = patch.mean("distance").squeeze().data
    >>> pws = patch.phase_weighted_stack("distance").squeeze().data
    >>>
    >>> # Plot results
    >>> fig, ax = plt.subplots(1, 1)
    >>> time = ricker_patch.get_array("time")
    >>> _ = ax.plot(time, stack, label="linear")
    >>> _ = ax.plot(time, pws, label="pws")
    >>> _ = ax.set_xlabel("time")
    >>> _ = ax.set_ylabel("amplitude")
    >>> _ = ax.legend()
    """

    stack_dim: Any
    transform_dim: Any = None
    power: Any = 2.0
    dim_reduce: Any = "empty"

    data_type = "phase_weighted_stack"

    def get_metadata(self, meta):
        """Return metadata with the stack dimension reduced, and the axes."""
        stack_dim, transform_dim = self.stack_dim, self.transform_dim
        # Ensure patch has both stack and transform dim. Raises nice Error if not.
        if transform_dim is None:
            transform_dim = _infer_transform_dim(meta, stack_dim)
        # Ensure evenly sampled transform dimension and get needed coords.
        meta.get_coord(transform_dim, require_evenly_sampled=True)
        stack_coord = meta.get_coord(stack_dim)
        plan = {
            "transform_axis": meta.get_axis(transform_dim),
            "stack_axis": meta.get_axis(stack_dim),
            "squeeze": self.dim_reduce == "squeeze",
        }
        new_coord = stack_coord.reduce_coord(dim_reduce=self.dim_reduce)
        cm = meta.coords.update(**{stack_dim: new_coord})
        return meta.new(coords=cm), plan

    def numpy_kernel(self, data, *, transform_axis, stack_axis, squeeze):
        """Return the phase weighted stack, through scipy's Hilbert transform."""
        analytic = scipy_hilbert(data, axis=transform_axis)
        return _phase_weighted_stack(data, analytic, stack_axis, self.power, squeeze)

    def kernel(self, data, *, transform_axis, stack_axis, squeeze):
        """Return the phase weighted stack, through the FFT."""
        analytic = analytic_signal(data, transform_axis)
        return _phase_weighted_stack(data, analytic, stack_axis, self.power, squeeze)
