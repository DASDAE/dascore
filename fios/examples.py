"""
A small module for loading examples.
"""
import numpy as np

import fios
from fios.utils.misc import register_func


EXAMPLE_TRACES = {}


def get_example_trace(example_name="random_das"):
    """Load an example trace."""
    return EXAMPLE_TRACES[example_name]()


@register_func(EXAMPLE_TRACES, key="random_das")
def _random_trace():
    """Generate a random DAS trace"""
    rand = np.random.RandomState(13)
    array = rand.random(size=(300, 2_000))
    attrs = dict(dx=1, dt=1 / 250.0, category="DAS", id="test_data1")
    coords = dict(
        distance=np.arange(array.shape[0]) * attrs["dx"],
        time=np.arange(array.shape[1]) * attrs["dt"],
    )
    out = dict(
        data=array,
        coords=coords,
        attrs=attrs,
    )
    return fios.Trace2D(**out)
