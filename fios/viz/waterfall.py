"""
Module for waterfall plotting.
"""
from typing import Optional
import matplotlib.pyplot as plt

import fios


def waterfall(
        trace: 'fios.Trace2D',
        ax: Optional[plt.Axes] = None,
        ischan = False,
        cmap=plt.get_cmap('bwr'),
        timescale='second',
        use_timestamp=False,
        timefmt = '%m/%d %H:%M:%S',
        is_shorten=False
) -> plt.Figure:
    """
    Parameters
    ----------
    trace
        The Trace object.
    """
    if ax is None:
        _, ax = plt.subplots(1)
    data = trace.data
    time = trace.coords['time'].values
    ax.imshow(data, cmap=cmap, aspect='auto')
    # ax.set_xticks()q
    ax.set_label('time')
    # offset =
    breakpoint()

    plt.show()

