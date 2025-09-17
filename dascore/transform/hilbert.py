"""
Patch functions based on the Hilbert transform.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import scipy.signal

from dascore.constants import PatchType
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
        A patch with a complex data array represneting the analytic signal.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> analytic = patch.hilbert(dim="time")
    >>> # Real part is original signal
    >>> assert np.allclose(analytic.data.real, patch.data)
    """
    # Get axis for the dimension
    axis = patch.dims.index(dim)

    # Apply Hilbert transform
    analytic_signal = scipy.signal.hilbert(patch.data, axis=axis)

    # Return new patch with complex data
    return patch.new(data=analytic_signal)


@patch_function()
def envelope(patch: PatchType, dim: str) -> PatchType:
    """
    Calculate the envelope (amplitude) of a signal using the Hilbert transform.

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
    >>> patch = dc.get_example_patch()
    >>> env = patch.envelope(dim="time")
    >>> # Envelope is always positive
    >>> assert np.all(env.data >= 0)
    """
    # Get the analytic signal
    analytic_patch = hilbert(patch, dim=dim)

    # Calculate envelope as magnitude of analytic signal
    envelope_data = np.abs(analytic_patch.data)

    # Return new patch with envelope data
    return patch.new(data=envelope_data)


@patch_function()
def phase_weighted_stack(
    patch: PatchType,
    transform_dim: str,
    stack_dim: str,
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
    transform_dim
        The dimension along which to perform the Hilbert transform.
        For typical use cases this will be "time".
    stack_dim
        The dimension over which the data should be stacked. For typical
        use cases this will be "distance".
    power
        The power to which the phase coherence is raised. Higher values
        give more weight to coherent signals. Default is 1.0.
    normalize
        If True, normalize the weights. Default is True.

    Returns
    -------
    PatchType
        A patch with the phase-weighted stack along the specified dimension.
        The specified dimension will have length 1.

    Notes
    -----
    Phase weighted stacking is described in:
    Schimmel, M., & Paulssen, H. (1997). Noise reduction and detection
    of weak, coherent signals through phase-weighted stacks.
    Geophysical Journal International, 130(2), 497-505.

    Examples
    --------
    >>> import dascore as dc
    >>> patch = dc.get_example_patch()
    >>> # Create multiple realizations along distance dimension for demo
    >>> stacked = patch.phase_weighted_stack(dim="distance", power=2.0)
    >>> assert stacked.shape[patch.dims.index("distance")] == 1
    """
    # Ensure patch has both stack and transform dim. Raises nice Error if not.
    stack_coord = patch.get_coord(stack_dim)
    _ = patch.get_coord(transform_dim)
    stack_axis = patch.dims.index(stack_dim)
    transform_axis = patch.dims.index(transform_dim)
    data = patch.data

    # Get unit phasors
    analytic_data = scipy.signal.hilbert(patch, axis=transform_axis)
    unit_phasors = analytic_data / np.abs(analytic_data)
    mean_phasor = np.mean(unit_phasors, axis=stack_axis, keepdims=True)

    # Get weights based on coherence.
    coherence = np.abs(mean_phasor.mean(axis=0)) ** power
    weights = coherence**power
    norm_weights = weights / np.max(weights)

    # Stack original data and apply weights
    stacked_data = np.mean(data, axis=stack_axis, keepdims=True) * norm_weights

    # Create new coord and coord manager, put patch back and return.
    new_coord = stack_coord  # Need to add this
    cm = patch.coords.update({stack_dim: new_coord})
    return patch.new(data=stacked_data, coords=cm)
