"""
One call to every patch function, for comparing the routes to an operation.

`PATCHES` names the patches a call may be made against; `CALLS` holds one
entry per patch function DASCore defines, so a function added later fails
`test_every_patch_function_is_covered` until it is given a call here.

The arguments are chosen to do something -- an operation whose arguments the
example patch is indifferent to compares equal whichever route made it, and
would not notice an argument being dropped on the way.

Seeded from `scripts/differential_check.py::get_calls`, which picks its
arguments for the same reason.
"""

from __future__ import annotations

import importlib
import pkgutil
from functools import cache

import numpy as np
import pandas as pd

import dascore as dc
from dascore.utils.imports import _LazyImport


@cache
def patch_functions() -> tuple:
    """
    Return every patch function DASCore defines, each once.

    A private walk rather than a public helper: the only thing which wants
    the whole list is this suite, and importing every module in the package
    is not something to hand a user by accident.
    """
    # `dascore.viz` is deferred by `dascore/__init__.py`, and its functions
    # are patch functions like any other.
    import dascore.viz  # noqa: F401, PLC0415

    seen: set[int] = set()
    out = []
    for info in pkgutil.walk_packages(dc.__path__, f"{dc.__name__}."):
        try:
            module = importlib.import_module(info.name)
        except ImportError:  # a module needing something this install lacks
            continue
        # A snapshot: calling a function this finds can import a submodule
        # into the module it came from, which would resize the live view.
        for value in list(vars(module).values()):
            # Asked first: touching a lazy proxy performs the import it
            # exists to defer, and a patch function is never one.
            if isinstance(value, _LazyImport):
                continue
            operation = getattr(value, "op", None)
            # `is value`, not merely present: `functools.wraps` copies the
            # wrapped function's `__dict__`, so a decorator around a patch
            # function inherits its `op` and is not a second one.
            if operation is None or getattr(operation, "args", (None,))[0] is not value:
                continue
            if id(value) not in seen:
                seen.add(id(value))
                out.append(value)
    return tuple(out)


@cache
def get_patch(name: str):
    """Return the patch a call is made against."""
    patch = dc.get_example_patch("random_das")
    if name == "default":
        return patch
    if name == "null":
        return dc.get_example_patch("patch_with_null")
    if name == "dft":
        return patch.dft("time")
    if name == "stft":
        # The recipe `istft`'s own example uses, which round trips.
        from dascore.units import second  # noqa: PLC0415

        return dc.get_example_patch("chirp").stft(time=10 * second, overlap=4 * second)
    if name == "spectrum":
        # A Fourier coordinate holding real data, which `specplot` draws.
        return patch.dft("time", real=True).abs()
    if name == "velocity":
        return patch.update_attrs(data_type="velocity")
    if name == "lat_lon":
        return dc.get_example_patch("random_patch_with_lat_lon")
    if name == "xyz":
        return dc.get_example_patch("random_patch_with_xyz")
    if name == "dispersion":
        return dc.get_example_patch("dispersion_event")
    if name == "event":
        return dc.get_example_patch("example_event_1")
    if name == "wacky":
        return dc.get_example_patch("wacky_dim_coords_patch")
    if name == "collapsed":
        # Dimensions which carry no coordinate, which is the shape
        # `make_broadcastable_to` is for.
        return patch.mean()
    if name == "shifted":
        # Each channel carries the time it should be aligned by.
        small = patch.select(distance=(0, 50))
        shifts = np.arange(len(small.get_array("distance"))) * np.timedelta64(1, "ms")
        return small.update_coords(shift_times=("distance", shifts))
    if name == "tiled":
        # A stack of tiles, which `reassemble` blends back.
        return patch.tile_apply(
            _halve, mode="stack", time=64, distance=16, samples=True
        )
    if name == "inventory":
        from dascore.examples import inventory_patch_pair  # noqa: PLC0415

        return inventory_patch_pair()[0]
    msg = f"no example patch called {name!r}"
    raise LookupError(msg)


def _halve(tiles):
    """A function over a stack of tiles, for `tile_apply`."""
    return tiles / 2


def _slope_filter():
    """Return a filter for `slope_filter`, as its docstring builds one."""
    return np.array([1e3, 2e3, 8e3, 9e3])


def _coords_frame():
    """Return the frame `coords_from_df` reads distances from."""
    return pd.DataFrame({"distance": [0.0, 100.0, 200.0], "quality": [1.0, 2.0, 3.0]})


def _shot():
    """Return the origin `add_distance_to` measures from."""
    return pd.Series({"x": 10.0, "y": 10.0, "z": 0.0})


class Lazy:
    """
    An argument built when a test runs it, not when the catalogue is read.

    `CALLS` is a module-level tuple, so anything spelled out in it is built
    at import -- which for `enrich`'s inventory means real work during
    collection, whether or not the test which needs it is selected.
    """

    def __init__(self, make):
        self._make = make

    def get(self):
        """Return the argument."""
        return self._make()


def resolve(args: tuple) -> tuple:
    """Return a catalogue entry's arguments, built if they were deferred."""
    return tuple(x.get() if isinstance(x, Lazy) else x for x in args)


@cache
def _inventory():
    """Return the inventory `enrich` copies metadata from."""
    from dascore.examples import inventory_patch_pair  # noqa: PLC0415

    return inventory_patch_pair()[1]


# (function name, example patch, positional arguments, keyword arguments).
CALLS: tuple[tuple[str, str, tuple, dict], ...] = (
    # -- basic, no arguments at all
    ("abs", "default", (), {}),
    ("angle", "dft", (), {}),
    ("conj", "dft", (), {}),
    ("imag", "dft", (), {}),
    ("real", "dft", (), {}),
    ("simplify_units", "default", (), {}),
    ("drop_private_coords", "default", (), {}),
    ("istft", "stft", (), {}),
    # -- one dimension, given positionally where the signature allows it
    ("normalize", "default", ("time",), {"norm": "l1"}),
    ("normalize", "default", ("time",), {"norm": "l2", "window": 11, "samples": True}),
    ("pow_coord", "default", (), {"time": 2}),
    ("standardize", "default", ("time",), {}),
    ("detrend", "default", ("time",), {"type": "constant"}),
    ("demean", "default", (), {"dim": "distance"}),
    ("demedian", "default", (), {"dim": "distance"}),
    ("differentiate", "default", ("time",), {"order": 2}),
    ("integrate", "default", ("time",), {"definite": True}),
    ("hilbert", "default", ("time",), {}),
    ("envelope", "default", ("time",), {}),
    ("dropna", "null", ("time",), {"how": "all"}),
    ("squeeze", "default", (), {"dim": None}),
    ("idft", "dft", (), {"dim": "ft_time"}),
    ("sobel_filter", "default", ("time",), {"mode": "reflect"}),
    # -- aggregations
    ("aggregate", "default", (), {"dim": "time", "method": "mean"}),
    ("all", "default", (), {"dim": "time"}),
    ("any", "default", (), {"dim": "time"}),
    ("first", "default", (), {"dim": "time"}),
    ("idxmax", "default", ("time",), {}),
    ("idxmin", "default", ("time",), {}),
    ("last", "default", (), {"dim": "time"}),
    ("max", "default", (), {"dim": "time"}),
    ("mean", "default", (), {"dim": "time"}),
    ("median", "default", (), {"dim": "time"}),
    ("min", "default", (), {"dim": "time"}),
    ("std", "default", (), {"dim": "time"}),
    ("sum", "default", (), {"dim": "time"}),
    # -- a dimension given as an extra, which the signature does not name
    ("pass_filter", "default", (), {"time": (10, 100)}),
    ("notch_filter", "default", (30,), {"time": 100}),
    ("median_filter", "default", (), {"time": 3, "samples": True}),
    ("gaussian_filter", "default", (), {"time": 2, "samples": True}),
    ("savgol_filter", "default", (3,), {"time": 5, "samples": True}),
    ("slope_filter", "default", (Lazy(_slope_filter),), {}),
    ("wiener_filter", "default", (), {"time": 5, "samples": True}),
    ("hampel_filter", "default", (), {"time": 5, "samples": True}),
    ("adaptive_spectral_filter", "default", (), {"time": 32, "samples": True}),
    ("tile_apply", "default", (_halve,), {"time": 64, "distance": 16, "samples": True}),
    ("reassemble", "tiled", (), {"taper": "triang"}),
    ("select", "default", (), {"distance": (10, 40)}),
    ("unselect", "default", (), {"distance": (10, 40)}),
    ("order", "default", (), {"distance": (30, 10, 20), "samples": True}),
    ("decimate", "default", (), {"time": 2}),
    ("pad", "default", (), {"time": (2, 3)}),
    ("roll", "default", (), {"time": 5, "samples": True}),
    ("taper", "default", (), {"time": 0.05}),
    ("taper_range", "default", (), {"time": (1, 2, 3, 4), "samples": True}),
    ("resample", "default", (), {"time": 0.01}),
    ("interpolate", "default", (), {"distance": np.arange(0, 100, 3.0)}),
    ("convert_units", "default", (), {"distance": "ft"}),
    ("set_units", "default", (), {"distance": "m"}),
    ("rename_coords", "default", (), {"distance": "depth"}),
    ("update_coords", "default", (), {"distance": np.arange(300) * 2.0}),
    ("correlate", "default", (), {"distance": 0}),
    ("correlate_shift", "default", ("time",), {}),
    ("line_mute", "default", (), {"time": (0.1, 0.2)}),
    ("whiten", "default", (), {"time": (10, 100)}),
    ("stalta", "default", (), {"time": (0.1, 0.5)}),
    ("fbe", "default", (0.5,), {"time": (10, 100)}),
    ("align_to_coord", "shifted", (), {"time": "shift_times", "mode": "full"}),
    # -- a *args group
    ("transpose", "default", ("time", "distance"), {}),
    ("snap_coords", "wacky", ("time",), {"reverse": True}),
    ("sort_coords", "default", ("time",), {"reverse": True}),
    ("flip", "default", ("time",), {"flip_coords": True}),
    ("drop_coords", "lat_lon", ("latitude",), {}),
    ("append_dims", "default", ("new_dim",), {}),
    # -- a plain positional value
    ("fillna", "null", (0.0,), {}),
    ("full", "default", (1.0,), {}),
    ("where", "default", (True,), {"other": 0.0}),
    ("make_broadcastable_to", "collapsed", ((2, 3),), {}),
    ("coords_from_df", "default", (Lazy(_coords_frame),), {"extrapolate": True}),
    ("add_distance_to", "xyz", (Lazy(_shot),), {}),
    # -- transforms
    ("dft", "default", ("time",), {"real": True}),
    ("stft", "default", (), {"time": 1.0}),
    ("spectrogram", "default", (), {"dim": "time"}),
    ("velocity_to_strain_rate", "velocity", (), {"step_multiple": 2}),
    ("velocity_to_strain_rate_edgeless", "velocity", (), {"step_multiple": 2}),
    ("radians_to_strain", "default", (), {"gauge_length": 10.0}),
    ("dispersion_phase_shift", "dispersion", (np.arange(100, 1000, 100.0),), {}),
    ("tau_p", "event", (np.arange(1000, 6000, 500.0),), {}),
    ("kurtosis", "default", (), {"time": 5, "samples": True}),
    ("phase_weighted_stack", "default", ("distance",), {"transform_dim": "time"}),
    ("slope_mute", "default", ((1e3, 2e3),), {}),
    ("enrich", "inventory", (Lazy(_inventory),), {"attrs": ("gauge_length",)}),
    # -- the ones which draw rather than return a patch
    ("waterfall", "default", (), {"show": False}),
    ("wiggle", "default", (), {"dim": "time"}),
    ("specplot", "spectrum", (), {"show": False}),
    ("map_fiber", "xyz", (), {"x": "x", "y": "y", "color": "z"}),
)
