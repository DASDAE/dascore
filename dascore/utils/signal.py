"""
Utilities for signal processing.
"""

from dascore.exceptions import ParameterError
from dascore.utils.imports import lazy_import

WINDOW_FUNCTIONS = dict(
    barthann=lazy_import("scipy.signal.windows", "barthann"),
    bartlett=lazy_import("scipy.signal.windows", "bartlett"),
    blackman=lazy_import("scipy.signal.windows", "blackman"),
    blackmanharris=lazy_import("scipy.signal.windows", "blackmanharris"),
    bohman=lazy_import("scipy.signal.windows", "bohman"),
    hamming=lazy_import("scipy.signal.windows", "hamming"),
    hann=lazy_import("scipy.signal.windows", "hann"),
    cos=lazy_import("scipy.signal.windows", "hann"),
    nuttall=lazy_import("scipy.signal.windows", "nuttall"),
    parzen=lazy_import("scipy.signal.windows", "parzen"),
    triang=lazy_import("scipy.signal.windows", "triang"),
    ramp=lazy_import("scipy.signal.windows", "triang"),
    boxcar=lazy_import("scipy.signal.windows", "boxcar"),
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
