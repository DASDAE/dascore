"""
Patch functions based on the Hilbert transform.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.signal

from dascore.constants import DIM_REDUCE_DOCS, PatchType
from dascore.exceptions import ParameterError
from dascore.utils.docs import compose_docstring
from dascore.utils.patch import patch_function


@patch_function()
def hilbert(patch: PatchType, dim: str) -> PatchType:
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
    # Get axis for the dimension
    patch.get_coord(dim, require_evenly_sampled=True)  # Ensure evenly sampled
    axis = patch.get_axis(dim)

    # Apply Hilbert transform
    analytic_signal = scipy.signal.hilbert(patch.data, axis=axis)

    # Return new patch with complex data
    return patch.new(data=analytic_signal)


@patch_function()
def envelope(patch: PatchType, dim: str) -> PatchType:
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
    # Get the analytic signal
    patch.get_coord(dim, require_evenly_sampled=True)  # Ensure evenly sampled
    axis = patch.get_axis(dim)
    data = scipy.signal.hilbert(patch.data, axis=axis)
    # Calculate envelope as magnitude of analytic signal
    envelope_data = np.abs(data)
    # Return new patch with envelope data
    return patch.new(data=envelope_data)


def __infer_transform_dim(patch, stack_dim):
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


@patch_function()
@compose_docstring(dim_reduce=DIM_REDUCE_DOCS)
def phase_weighted_stack(
    patch: PatchType,
    stack_dim: str,
    transform_dim: str | None = None,
    power: float = 2.0,
    dim_reduce: str | Callable = "empty",
) -> PatchType:
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
    # Ensure patch has both stack and transform dim. Raises nice Error if not.
    if transform_dim is None:
        transform_dim = __infer_transform_dim(patch, stack_dim)
    # Ensure evenly sampled transform dimension and get needed coords.
    patch.get_coord(transform_dim, require_evenly_sampled=True)
    stack_coord = patch.get_coord(stack_dim)
    # Get corresponding axes.
    transform_axis = patch.get_axis(transform_dim)
    stack_axis = patch.get_axis(stack_dim)
    data = patch.data
    # Get unit phasors. Use eps here to avoid unstable division by 0.
    analytic_data = scipy.signal.hilbert(data, axis=transform_axis)
    eps = np.finfo(analytic_data.real.dtype).eps
    amp = np.maximum(np.abs(analytic_data), eps)
    unit_phasors = analytic_data / amp
    mean_phasor = np.mean(unit_phasors, axis=stack_axis, keepdims=True)
    # Get weights based on coherence.
    # The coherence |mean_phasor| naturally ranges from 0 to 1:
    # - 0: completely incoherent (random phases)
    # - 1: perfectly coherent (all phases aligned)
    weights = np.abs(mean_phasor) ** power
    # Stack original data and apply weights (we can do this since weights
    # are common across all samples)
    stacked_data = (
        np.mean(data, axis=stack_axis, keepdims=True) * weights
    )  # Create new coord and coord manager, put patch back and return.
    new_coord = stack_coord.reduce_coord(dim_reduce=dim_reduce)
    cm = patch.coords.update(**{stack_dim: new_coord})
    if dim_reduce == "squeeze":
        stacked_data = np.squeeze(stacked_data, axis=stack_axis)
    return patch.new(data=stacked_data, coords=cm)
