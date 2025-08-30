"""
Utilities for signal processing.
"""

from scipy.signal import windows

from dascore.exceptions import ParameterError

WINDOW_FUNCTIONS = dict(
    barthann=windows.barthann,
    bartlett=windows.bartlett,
    blackman=windows.blackman,
    blackmanharris=windows.blackmanharris,
    bohman=windows.bohman,
    hamming=windows.hamming,
    hann=windows.hann,
    cos=windows.hann,
    nuttall=windows.nuttall,
    parzen=windows.parzen,
    triang=windows.triang,
    ramp=windows.triang,
)


def _get_window_function(window_type):
    """Get the window function to use for taper."""
    # get taper function or raise if it isn't known.
    if window_type not in WINDOW_FUNCTIONS:
        msg = (
            f"'{window_type}' is not a known window type. "
            f"Options are: {sorted(WINDOW_FUNCTIONS)}"
        )
        raise ParameterError(msg)
    func = WINDOW_FUNCTIONS[window_type]
    return func
