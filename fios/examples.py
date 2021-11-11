"""
A small module for loading examples.
"""
import numpy as np

import fios
from fios.utils.misc import register_func

EXAMPLE_PATCHES = {}
EXAMPLE_STREAMS = {}


def get_example_patch(example_name="random_das"):
    """Load an example Patch."""
    return EXAMPLE_PATCHES[example_name]()


def get_example_stream(example_name="random_das"):
    """Load an example Patch."""
    return EXAMPLE_STREAMS[example_name]()


@register_func(EXAMPLE_PATCHES, key="random_das")
def _random_patch():
    """Generate a random DAS Patch"""
    rand = np.random.RandomState(13)
    array = rand.random(size=(300, 2_000))
    t1 = np.datetime64("2017-09-18")
    attrs = dict(dx=1, dt=1 / 250.0, category="DAS", id="test_data1", time_min=t1)
    coords = dict(
        distance=np.arange(array.shape[0]) * attrs["dx"],
        time=np.arange(array.shape[1]) * attrs["dt"],
    )
    out = dict(data=array, coords=coords, attrs=attrs)
    return fios.Patch(**out)


@register_func(EXAMPLE_STREAMS, key="random_das")
def _random_stream():
    """Generate a random DAS Patch"""
    out = _random_patch()
    return fios.Stream([out])
