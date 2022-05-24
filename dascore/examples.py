"""
A small module for loading examples.
"""
import numpy as np

import dascore
from dascore.utils.misc import register_func
from dascore.utils.time import to_timedelta64

EXAMPLE_PATCHES = {}
EXAMPLE_SPOOLS = {}


def get_example_patch(example_name="random_das", **kwargs):
    """
    Load an example Patch.

    kwargs are passed to the corresponding functions to generate data.
    """
    return EXAMPLE_PATCHES[example_name](**kwargs)


def get_example_spool(example_name="random_das", **kwargs):
    """
    Load an example spool.

    kwargs are passed to the corresponding functions to generate data.
    """
    return EXAMPLE_SPOOLS[example_name](**kwargs)


@register_func(EXAMPLE_PATCHES, key="random_das")
def _random_patch(starttime="2017-09-18"):
    """Generate a random DAS Patch"""
    rand = np.random.RandomState(13)
    array = rand.random(size=(300, 2_000))
    t1 = np.datetime64(starttime)
    attrs = dict(
        d_distance=1,
        d_time=to_timedelta64(1 / 250),
        category="DAS",
        id="test_data1",
        time_min=t1,
        tag="random_patch",
    )
    coords = dict(
        distance=np.arange(array.shape[0]) * attrs["d_distance"],
        time=np.arange(array.shape[1]) * attrs["d_time"],
    )
    out = dict(data=array, coords=coords, attrs=attrs)
    return dascore.Patch(**out)


@register_func(EXAMPLE_SPOOLS, key="random_das")
def _random_spool(d_time=0, length=3):
    """
    Generate several random patches in the spool.

    Parameters
    ----------
    d_time
        The difference in time between each patch.
    length
    """
    out = []
    starttime = "2020-01-03"
    for _ in range(length):
        patch = get_example_patch(starttime=starttime)
        out.append(patch)
        starttime = patch.attrs["time_max"] + to_timedelta64(d_time)
    return dascore.MemorySpool(out)
