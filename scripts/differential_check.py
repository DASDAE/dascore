"""
Compare patch-function results with a git ref using exact fingerprints.

Verify NumPy results during the
[array API migration](https://data-apis.org/array-api/latest/).
Hashes detect every bit difference; no approximate comparison is used.

Usage:

    python scripts/differential_check.py --ref <git ref>

`get_calls` covers example patches with datetime coordinates, units, and complex data.
`MATRIX_CALLS` runs against every `make_arrays` dtype and edge case, including nan,
infinities, null slices, and overflow. Add rewritten functions to the appropriate list;
unlisted calls are not checked.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import dascore as dc
from dascore.utils.signal import WINDOW_NAMES

# Repeat fast calls for _TIMING_BUDGET; stop slow calls at _TIMING_MIN_ROUNDS.
_TIMING_MIN_ROUNDS = 3
_TIMING_MAX_ROUNDS = 200
_TIMING_BUDGET = 0.002
# Timings below this are noise on any machine, so they are not reported
# however far they appear to have moved.
_TIMING_FLOOR = 50e-6
# Measured with --ref HEAD: totals varied within 10%, individual calls within 35%.
_TIMING_NOISE = 0.35
# How many times each leg is run. Two is enough to stop the leg which
# happened to go first from looking slow.
_TIMING_PASSES = 3

# Keys a dump carries which are not calls.
_BOOKKEEPING = {"_timing", "_dascore_path"}

# Calls which raise on purpose, so that a rewrite keeps the message; and
# matrix inputs an operation cannot take. `--strict` skips these, but
# `compare` still fails a call which raises on one side only, or with a
# different message.
EXPECTED_ERRORS = (
    "rename_missing",
    "*_bad",
    "*_bad_*",
    "agg_no_dims*",
    "agg_squeeze_all",
    "agg_last_all",
    "hampel_even",
    "idxmax_none",
    "idxmax_tuple",
    "idxmax_partial",
    "idxmax_squeeze_only_dim",
    "pass_two_dims",
    "*_no_window",
    # One sample per dimension has no step to window or filter by.
    "matrix/tiny/*",
    "matrix/single_row/hampel_exact",
    # scipy refuses these dtypes and values.
    "matrix/complex*/median_filter",
    "matrix/complex*/hampel_exact",
    "matrix/bool/hampel_*",
    "matrix/*nan*/savgol_filter",
    "matrix/*inf*/savgol_filter",
    "matrix/bool/differentiate*",
    "matrix/bool/strain_rate*",
    "matrix/complex*/phase_weighted_stack",
    # One distance sample has no step, and too few for these windows.
    "matrix/single_row/differentiate_step",
    "matrix/single_row/slope_mute",
    "matrix/single_row/strain_rate*",
    "matrix/single_row/taper_*",
)


# The fingerprint fields which say what a call answered, as opposed to what
# it claims the answer is.
# Cover dtypes and edge values missed by example calls; this caught the
# float32 promotion regression in #921.
def make_arrays() -> dict:
    """Return arrays covering the dtypes and values patch data can hold."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=(6, 8))
    imaginary = rng.normal(size=(6, 8))
    arrays = {
        "float64": base,
        "float32": base.astype("float32"),
        "int32": (base * 100).astype("int32"),
        "int64": (base * 100).astype("int64"),
        "bool": base > 0,
        "complex128": base + 1j * imaginary,
        "complex64": (base + 1j * imaginary).astype("complex64"),
        "all_nan": np.full((6, 8), np.nan),
        "zeros": np.zeros((6, 8)),
        "tiny": np.full((1, 1), 3.0),
        "single_row": base[:1],
        "huge": base * 1e300,
    }
    nan = base.copy()
    nan[1, 2], nan[3, :] = np.nan, np.nan
    arrays["with_nan"] = nan
    infinite = base.copy()
    infinite[0, 0], infinite[0, 1] = np.inf, -np.inf
    arrays["with_inf"] = infinite
    both = nan.copy()
    both[2, 2], both[2, 3] = np.inf, -np.inf
    arrays["nan_and_inf"] = both
    return arrays


# Applied to every array above, so a dtype or a special value which
# changes an answer shows up wherever it happens.
MATRIX_CALLS = {
    "abs": lambda patch: patch.abs(),
    "add": lambda patch: patch + 1,
    "all": lambda patch: patch.all("time"),
    "angle": lambda patch: patch.angle(),
    "any": lambda patch: patch.any("time"),
    "conj": lambda patch: patch.conj(),
    "demean": lambda patch: patch.demean("time"),
    "demedian": lambda patch: patch.demedian("time"),
    "dropna_all": lambda patch: patch.dropna("distance", how="all"),
    "dropna_any": lambda patch: patch.dropna("time", how="any"),
    "fillna_0": lambda patch: patch.fillna(0),
    "idxmax": lambda patch: patch.idxmax("time"),
    "idxmin": lambda patch: patch.idxmin("time"),
    "fillna_noinf": lambda patch: patch.fillna(2, include_inf=False),
    "flip_both": lambda patch: patch.flip(*patch.dims),
    "flip_one": lambda patch: patch.flip("time"),
    "full_float": lambda patch: patch.full(1.5),
    "full_int": lambda patch: patch.full(2),
    "gt": lambda patch: patch > 0,
    "imag": lambda patch: patch.imag(),
    "max": lambda patch: patch.max("distance"),
    "mean": lambda patch: patch.mean("time"),
    "mean_all": lambda patch: patch.mean(),
    "median": lambda patch: patch.median("time"),
    "min": lambda patch: patch.min("time"),
    "mul": lambda patch: patch * 2,
    "norm_bit": lambda patch: patch.normalize("time", norm="bit"),
    "norm_l2_distance": lambda patch: patch.normalize("distance", norm="l2"),
    "demean_distance": lambda patch: patch.demean("distance"),
    "rename": lambda patch: patch.rename_coords(time="t"),
    "transpose_noop": lambda patch: patch.transpose(*patch.dims),
    "transpose_ell": lambda patch: patch.transpose(..., "distance"),
    "norm_l1": lambda patch: patch.normalize("time", norm="l1"),
    "norm_l2": lambda patch: patch.normalize("time", norm="l2"),
    "norm_max": lambda patch: patch.normalize("time", norm="max"),
    "np_exp": lambda patch: np.exp(patch),
    "pad": lambda patch: patch.pad(time=(1, 2), samples=True),
    "pad_both": lambda patch: patch.pad(time=1, distance=1, samples=True),
    "pad_fill": lambda patch: patch.pad(distance=1, samples=True, constant_values=7),
    "pad_noexpand": lambda patch: patch.pad(time=1, samples=True, expand_coords=False),
    "real": lambda patch: patch.real(),
    "reduce": lambda patch: patch.add.reduce(dim="time"),
    "roll": lambda patch: patch.roll(time=2, samples=True),
    "roll_coord": lambda patch: patch.roll(time=2, samples=True, update_coord=True),
    "square": lambda patch: patch**2,
    "standardize": lambda patch: patch.standardize("time"),
    "std": lambda patch: patch.std("time"),
    "sum": lambda patch: patch.sum("time"),
    "transpose": lambda patch: patch.transpose(),
    "update_coords": lambda patch: patch.update_coords(
        distance=patch.get_array("distance") + 1
    ),
    "where_arr": lambda patch: patch.where(np.asarray(patch.data) > 0),
    "where_other": lambda patch: patch.where(np.asarray(patch.data) > 0, other=0),
    # window filters and pass filters
    "median_filter": lambda patch: patch.median_filter(time=3, samples=True),
    "gaussian_filter": lambda patch: patch.gaussian_filter(time=1, samples=True),
    "savgol_filter": lambda patch: patch.savgol_filter(1, time=3, samples=True),
    "wiener_filter": lambda patch: patch.wiener_filter(time=3, samples=True),
    "hampel_filter": lambda patch: patch.hampel_filter(time=3, samples=True),
    "hampel_exact": lambda patch: patch.hampel_filter(
        time=3, distance=3, samples=True, approximate=False
    ),
    "pass_filter": lambda patch: patch.pass_filter(time=(None, 0.5), zerophase=False),
    # aggregations along several dims and with each kind of dim_reduce
    "first": lambda patch: patch.first("time"),
    "last": lambda patch: patch.last("distance"),
    "mean_both": lambda patch: patch.mean(("time", "distance")),
    "median_all": lambda patch: patch.median(),
    "sum_squeeze": lambda patch: patch.sum("time", dim_reduce="squeeze"),
    "max_reduce_mean": lambda patch: patch.max("distance", dim_reduce="mean"),
    "agg_median": lambda patch: patch.aggregate("distance", method="median"),
    "agg_callable": lambda patch: patch.aggregate("time", method=np.nanmax),
    "idxmax_distance": lambda patch: patch.idxmax("distance", dim_reduce="squeeze"),
    "idxmin_min": lambda patch: patch.idxmin("distance", dim_reduce="min"),
    # envelope multipliers
    "taper": lambda patch: patch.taper(time=0.25),
    "taper_distance": lambda patch: patch.taper(
        distance=(0.3, None), window_type="ramp"
    ),
    "taper_range": lambda patch: patch.taper_range(time=(1, 2, 4, 6), samples=True),
    "taper_range_invert": lambda patch: patch.taper_range(
        distance=(1, 4), samples=True, invert=True
    ),
    "line_mute": lambda patch: patch.line_mute(time=(0, 1.0)),
    "line_mute_smooth": lambda patch: patch.line_mute(time=(0.5, 2.0), smooth=0.2),
    "slope_mute": lambda patch: patch.slope_mute((0.5, 4)),
    "pow_coord": lambda patch: patch.pow_coord(time=2),
    "pow_coord_abs": lambda patch: patch.pow_coord(distance=1, relative=False),
    # calculus
    "differentiate": lambda patch: patch.differentiate("time"),
    "differentiate_step": lambda patch: patch.differentiate("distance", step=2),
    "differentiate_findiff": lambda patch: patch.differentiate("time", order=4),
    "integrate": lambda patch: patch.integrate("time"),
    "integrate_definite": lambda patch: patch.integrate("distance", definite=True),
    "strain_rate": lambda patch: patch.update_attrs(
        data_type="velocity"
    ).velocity_to_strain_rate(),
    "strain_rate_edgeless": lambda patch: patch.update_attrs(
        data_type="velocity"
    ).velocity_to_strain_rate_edgeless(step_multiple=2),
    "phase_weighted_stack": lambda patch: patch.phase_weighted_stack("distance"),
}


def _pinned(patch, label: str):
    """
    Give a patch built here the same ids in every process.

    A patch built in memory gets random ids, and every id downstream is
    derived from them, so without this no two runs could be compared.
    """
    names = [x for x in ("origin_id", "data_id") if x in dc.PatchAttrs.model_fields]
    fixed = hashlib.blake2b(label.encode(), digest_size=16).hexdigest()
    return patch.update_attrs(**dict.fromkeys(names, fixed))


def _matrix_patch(array, label: str = "matrix"):
    """Wrap an array in a patch with evenly sampled coordinates."""
    coords = {
        "distance": np.arange(array.shape[0]) * 1.0,
        "time": np.arange(array.shape[1]) * 0.5,
    }
    patch = dc.Patch(data=array, coords=coords, dims=("distance", "time"))
    return _pinned(patch, label)


def get_matrix_calls() -> dict:
    """Return every call in MATRIX_CALLS against every array."""
    out = {}
    for array_name, array in make_arrays().items():
        patch = _matrix_patch(array, array_name)
        out[f"matrix/{array_name}/input"] = lambda patch=patch: patch
        for call_name, call in MATRIX_CALLS.items():
            key = f"matrix/{array_name}/{call_name}"
            out[key] = lambda call=call, patch=patch: call(patch)
    return out


def _filter_calls(small, small_f32, spiky, hz, m, s) -> dict:
    """Return calls to the pass, notch and window filters."""
    return {
        "pass_band": lambda: small.pass_filter(time=(10, 100)),
        "pass_low_causal": lambda: small.pass_filter(time=(None, 50), zerophase=False),
        "pass_high_corners": lambda: small.pass_filter(time=(20, ...), corners=2),
        "pass_hz": lambda: small.pass_filter(time=(1 * hz, 10 * hz)),
        "pass_distance": lambda: small.pass_filter(distance=(None, 0.1)),
        "pass_wavelength": lambda: small.pass_filter(distance=(5 * m, 10 * m)),
        "pass_bad_range": lambda: small.pass_filter(time=(None, 1000)),
        "pass_two_dims": lambda: small.pass_filter(time=(1, 10), distance=(0.1, 0.2)),
        "notch_time": lambda: small.notch_filter(time=60, q=30),
        "notch_positional": lambda: small.notch_filter(10, time=60),
        "notch_both": lambda: small.notch_filter(time=60, distance=0.2, q=30),
        "notch_units": lambda: small.notch_filter(time=60 * hz, distance=5 * m, q=3),
        "notch_f32": lambda: small_f32.notch_filter(time=60, q=30),
        "notch_bad": lambda: small.notch_filter(time=500, q=30),
        "median_time": lambda: small.median_filter(time=0.012),
        "median_units": lambda: small.median_filter(time=0.012 * s, distance=2 * m),
        "median_samples": lambda: small.median_filter(time=3, distance=4, samples=True),
        "median_constant": lambda: small.median_filter(
            time=3, samples=True, mode="constant", cval=1.0
        ),
        "median_positional": lambda: small.median_filter(True, "nearest", time=3),
        "savgol_time": lambda: small.savgol_filter(polyorder=2, time=0.04),
        "savgol_distance": lambda: small.savgol_filter(2, distance=5, samples=True),
        "savgol_both": lambda: small.savgol_filter(distance=10, time=0.04, polyorder=4),
        "savgol_constant": lambda: small.savgol_filter(
            1, True, "constant", 2.0, time=5
        ),
        "savgol_bad": lambda: small.savgol_filter(polyorder=9, time=5, samples=True),
        "gauss_time": lambda: small.gaussian_filter(time=0.02),
        "gauss_samples": lambda: small.gaussian_filter(samples=True, distance=3),
        "gauss_both": lambda: small.gaussian_filter(time=0.02, distance=3 * m),
        "gauss_options": lambda: small.gaussian_filter(
            True, "constant", 1.0, 2.0, time=4
        ),
        "wiener_time": lambda: spiky.wiener_filter(time=5, samples=True),
        "wiener_noise": lambda: spiky.wiener_filter(time=5, samples=True, noise=0.01),
        "wiener_both": lambda: spiky.wiener_filter(time=5, distance=3, samples=True),
        "wiener_units": lambda: spiky.wiener_filter(time=0.02 * s),
        "wiener_no_window": lambda: spiky.wiener_filter(),
        "median_no_window": lambda: small.median_filter(),
        "pass_no_window": lambda: small.pass_filter(),
        "notch_no_window": lambda: small.notch_filter(3),
        "hampel_time": lambda: spiky.hampel_filter(time=0.02, threshold=3.5),
        "hampel_both": lambda: spiky.hampel_filter(time=5, distance=5, samples=True),
        "hampel_exact": lambda: spiky.hampel_filter(
            time=5, distance=5, samples=True, approximate=False
        ),
        "hampel_large": lambda: spiky.hampel_filter(
            time=11, distance=11, samples=True, approximate=False
        ),
        "hampel_bad_threshold": lambda: spiky.hampel_filter(time=5, threshold=-1),
        "hampel_even": lambda: spiky.hampel_filter(time=4, samples=True),
    }


def _aggregate_calls(patch, null_patch, int_patch, bool_patch, typed) -> dict:
    """Return aggregations with every kind of dim and dim_reduce."""
    collapsed = patch.mean("time")
    return {
        **{
            f"agg_{name}_reduce_{how}": (
                lambda name=name, how=how: getattr(patch, name)("time", dim_reduce=how)
            )
            for name in ("mean", "max", "first", "any")
            for how in ("squeeze", "mean", "min", "first", "last", "median", "sum")
        },
        "agg_reduce_callable": lambda: patch.mean("distance", dim_reduce=np.max),
        "agg_both_dims": lambda: patch.mean(("time", "distance")),
        "agg_both_dims_squeeze": lambda: patch.sum(["distance"], dim_reduce="squeeze"),
        "agg_reversed_dims": lambda: patch.std(("distance", "time"), dim_reduce="mean"),
        "agg_squeeze_all": lambda: patch.mean(dim_reduce="squeeze"),
        "agg_no_dims": lambda: patch.mean(()),
        "agg_no_dims_typed": lambda: typed.any(()),
        "agg_typed_any": lambda: typed.any("distance"),
        "agg_typed_mean": lambda: typed.mean("distance"),
        "agg_bad_dim": lambda: patch.mean("nope"),
        "agg_bad_reduce": lambda: patch.mean("time", dim_reduce="nope"),
        "agg_positional": lambda: patch.aggregate("time", "max", "squeeze"),
        "agg_positional_short": lambda: patch.min("distance", "mean"),
        **{
            f"agg_method_{method}": (
                lambda method=method: patch.aggregate("distance", method=method)
            )
            for method in ("median", "min", "max", "sum", "std", "first", "last")
        },
        "agg_method_callable": lambda: patch.aggregate("time", method=np.nanmax),
        "agg_method_all": lambda: patch.aggregate(method="mean"),
        "agg_median_all": lambda: patch.median(),
        "agg_median_distance": lambda: patch.median("distance"),
        "agg_first_all": lambda: patch.first(),
        "agg_last_all": lambda: patch.last(dim_reduce="squeeze"),
        "agg_any_all": lambda: bool_patch.any(),
        "agg_all_distance": lambda: bool_patch.all("distance"),
        "agg_null_median": lambda: null_patch.median("time"),
        "agg_null_first": lambda: null_patch.first("distance"),
        "agg_int_median": lambda: int_patch.median("distance"),
        "agg_int_std_all": lambda: int_patch.std(),
        "agg_collapsed_again": lambda: collapsed.mean("time"),
        "agg_collapsed_distance": lambda: collapsed.max("distance"),
        "idxmax_time": lambda: patch.idxmax("time"),
        "idxmin_time": lambda: patch.idxmin("time"),
        "idxmax_distance": lambda: patch.idxmax("distance"),
        "idxmax_squeeze": lambda: patch.idxmax("time", dim_reduce="squeeze"),
        "idxmin_reduce_mean": lambda: patch.idxmin("distance", dim_reduce="mean"),
        "idxmax_positional": lambda: patch.idxmax("distance", "max"),
        "idxmax_null": lambda: null_patch.idxmax("time"),
        "idxmin_null_distance": lambda: null_patch.idxmin("distance"),
        "idxmax_int": lambda: int_patch.idxmax("distance"),
        "idxmax_typed": lambda: typed.idxmax("time"),
        "idxmax_none": lambda: patch.idxmax(None),
        "idxmax_tuple": lambda: patch.idxmax(("time",)),
        "idxmax_partial": lambda: collapsed.idxmax("time"),
        "idxmax_squeeze_only_dim": lambda: collapsed.squeeze().idxmax(
            "distance", dim_reduce="squeeze"
        ),
    }


def _envelope_calls(patch, int_patch, f32, dft_patch, wacky, m, s) -> dict:
    """Return tapers, mutes and coordinate gains, with every spelling."""
    t1 = patch.get_coord("time").min() + np.timedelta64(1, "s")
    t2 = t1 + np.timedelta64(3, "s")
    percent = dc.get_unit("percent")
    velocity = ([0, 0.375], [0, 0.25]), ([0, 300], [0, 300])
    return {
        **{
            f"taper_{name}": (
                lambda name=name: patch.taper(time=0.05, window_type=name)
            )
            for name in sorted(WINDOW_NAMES)
        },
        "taper_tukey": lambda: patch.taper(time=0.1, window_type=("tukey", 0.5)),
        "taper_distance": lambda: patch.taper(
            distance=(0.10, None), window_type="triang"
        ),
        "taper_end_only": lambda: patch.taper(time=(None, 0.2)),
        "taper_percent": lambda: patch.taper(time=(20 * percent, 12 * percent)),
        "taper_meters": lambda: patch.taper(distance=15 * m),
        "taper_seconds": lambda: patch.taper(time=(1 * s, 2 * s)),
        "taper_timedelta": lambda: patch.taper(time=np.timedelta64(500, "ms")),
        "taper_positional": lambda: patch.taper("blackman", time=0.1),
        "taper_none": lambda: patch.taper(time=(None, None)),
        "taper_bad_zero": lambda: patch.taper(time=0),
        "taper_int": lambda: int_patch.taper(time=0.1),
        "taper_f32": lambda: f32.taper(time=0.1),
        "taper_complex": lambda: dft_patch.taper(ft_time=0.1),
        "taper_wacky": lambda: wacky.taper(distance=0.1),
        "taper_bad_overlap": lambda: patch.taper(time=0.6),
        "taper_bad_length": lambda: patch.taper(time=(0.1, 0.2, 0.3)),
        "taper_bad_dims": lambda: patch.taper(time=0.1, distance=0.1),
        "taper_range_abs": lambda: patch.taper_range(time=(t1, t2)),
        "taper_range_invert": lambda: patch.taper_range(time=(t1, t2), invert=True),
        "taper_range_relative": lambda: patch.taper_range(
            time=(1, 2, 5, 5), relative=True
        ),
        "taper_range_samples": lambda: patch.taper_range(
            distance=(10, 80), samples=True
        ),
        "taper_range_two": lambda: patch.taper_range(
            distance=((25, 50, 100, 125), (150, 175, 200, 225))
        ),
        "taper_range_ellipsis": lambda: patch.taper_range(
            distance=(..., 50, 100, None), window_type="ramp"
        ),
        "taper_range_positional": lambda: patch.taper_range(
            "hamming", True, False, True, distance=(10, 20, 30, 40)
        ),
        "taper_range_int": lambda: int_patch.taper_range(
            distance=(10, 80), samples=True
        ),
        "taper_range_f32": lambda: f32.taper_range(distance=(10, 80), samples=True),
        "taper_range_wacky": lambda: wacky.taper_range(distance=(5, 10, 20, 25)),
        "taper_range_bad_len": lambda: patch.taper_range(time=(1, 2, 3), samples=True),
        "taper_range_bad_none": lambda: patch.taper_range(time=(None, 2), samples=True),
        "taper_range_bad_scalar": lambda: patch.taper_range(time=2),
        "line_mute_time": lambda: patch.line_mute(time=(0, 0.5)),
        "line_mute_invert": lambda: patch.line_mute(time=(0.2, -0.2), invert=True),
        "line_mute_distance": lambda: patch.line_mute(distance=(50, 100)),
        "line_mute_absolute": lambda: patch.line_mute(
            distance=(50, 100), relative=False
        ),
        "line_mute_smooth_units": lambda: patch.line_mute(
            time=(0.2, 0.8), smooth=0.02 * s
        ),
        "line_mute_smooth_int": lambda: patch.line_mute(time=(0.2, 0.8), smooth=5),
        "line_mute_line": lambda: patch.line_mute(
            time=(0, [0, 0.3]), distance=(None, [0, 300]), smooth=0.02
        ),
        "line_mute_wedge": lambda: patch.line_mute(
            time=velocity[0], distance=velocity[1]
        ),
        "line_mute_wedge_invert": lambda: patch.line_mute(
            time=velocity[0], distance=velocity[1], invert=True
        ),
        "line_mute_parallel": lambda: patch.line_mute(
            time=([0, 0.2], [0.1, 0.3]), distance=([0, 100], [0, 100])
        ),
        "line_mute_smooth_dict": lambda: patch.line_mute(
            time=velocity[0], distance=velocity[1], smooth={"time": 0.01, "distance": 3}
        ),
        "line_mute_int": lambda: int_patch.line_mute(time=(0, 0.5)),
        "line_mute_f32": lambda: f32.line_mute(time=(0, 0.5), smooth=0.05),
        "line_mute_bad_none": lambda: patch.line_mute(),
        "line_mute_bad_three": lambda: patch.line_mute(time=(0, 1, 2)),
        "line_mute_bad_smooth": lambda: patch.line_mute(time=(0, 1), smooth=1.5),
        "line_mute_bad_dict": lambda: patch.line_mute(time=(0, 1), smooth={"x": 1}),
        "line_mute_bad_degenerate": lambda: patch.line_mute(
            time=([0, 0], [0, 0.3]), distance=([0, 0], [0, 300])
        ),
        "slope_mute": lambda: patch.slope_mute(slopes=(1000, 3000)),
        "slope_mute_invert": lambda: patch.slope_mute(slopes=(1500, 2500), invert=True),
        "slope_mute_smooth": lambda: patch.slope_mute(slopes=(1000, 3000), smooth=0.05),
        "slope_mute_slowness": lambda: patch.slope_mute(
            slopes=(0.0003, 0.001), dims=("time", "distance")
        ),
        "slope_mute_edges": lambda: patch.slope_mute(slopes=np.array([0, np.inf])),
        "slope_mute_flipped": lambda: patch.flip("distance").slope_mute((1000, 3000)),
        "slope_mute_int": lambda: int_patch.slope_mute((1000, 3000)),
        "slope_mute_bad_shape": lambda: patch.slope_mute(slopes=(1, 2, 3)),
        "slope_mute_bad_sign": lambda: patch.slope_mute(slopes=(-1, 2)),
        "pow_coord_time": lambda: patch.pow_coord(time=2),
        "pow_coord_distance": lambda: patch.pow_coord(distance=1),
        "pow_coord_both": lambda: patch.pow_coord(time=2, distance=1),
        "pow_coord_float": lambda: patch.pow_coord(time=0.5),
        "pow_coord_absolute": lambda: patch.pow_coord(distance=2, relative=False),
        "pow_coord_positional": lambda: patch.pow_coord(False, distance=1),
        "pow_coord_units": lambda: patch.update_attrs(data_units="m/s").pow_coord(
            distance=1, relative=False
        ),
        "pow_coord_int": lambda: int_patch.pow_coord(time=1),
        "pow_coord_f32": lambda: f32.pow_coord(time=2),
        "pow_coord_complex": lambda: dft_patch.pow_coord(ft_time=1, relative=False),
        "pow_coord_wacky": lambda: wacky.pow_coord(distance=1),
        "pow_coord_bad_time": lambda: patch.pow_coord(time=1, relative=False),
        "pow_coord_bad_negative": lambda: patch.pow_coord(distance=-1, relative=False),
    }


def _calculus_calls(patch, int_patch, f32, dft_patch, wacky, typed) -> dict:
    """Return derivatives, integrals, strain rates and phase weighted stacks."""
    velocity = _pinned(
        dc.get_example_patch("deformation_rate_event_1").isel(time=slice(0, 300)),
        "velocity",
    )
    three_d = _pinned(dc.get_example_patch("nd_patch", dim_count=3), "3d")
    return {
        "diff_time": lambda: patch.differentiate("time"),
        "diff_distance": lambda: patch.differentiate(dim="distance", order=2),
        "diff_all": lambda: patch.differentiate(None),
        "diff_tuple": lambda: patch.differentiate(("distance", "time")),
        "diff_order4": lambda: patch.differentiate("time", order=4),
        "diff_order6_all": lambda: patch.differentiate(None, order=6),
        "diff_step": lambda: patch.differentiate("distance", step=3),
        "diff_step_order4": lambda: patch.differentiate("time", order=4, step=2),
        "diff_positional": lambda: patch.differentiate("time", 2, 2),
        "diff_int": lambda: int_patch.differentiate("time"),
        "diff_int_step": lambda: int_patch.differentiate("time", step=2),
        "diff_f32": lambda: f32.differentiate("time"),
        "diff_complex": lambda: dft_patch.differentiate("ft_time"),
        "diff_typed": lambda: typed.differentiate("time"),
        "diff_units": lambda: patch.update_attrs(data_units="m").differentiate("time"),
        "diff_wacky": lambda: wacky.differentiate("distance"),
        "diff_wacky_step": lambda: wacky.differentiate("distance", step=2),
        "diff_wacky_findiff": lambda: wacky.differentiate("distance", order=4),
        "diff_bad_step_dims": lambda: patch.differentiate(None, step=2),
        "int_time": lambda: patch.integrate("time"),
        "int_distance": lambda: patch.integrate(dim="distance"),
        "int_all": lambda: patch.integrate(None),
        "int_definite": lambda: patch.integrate("time", definite=True),
        "int_definite_all": lambda: patch.integrate(None, True),
        "int_int": lambda: int_patch.integrate("time"),
        "int_f32": lambda: f32.integrate("distance", definite=True),
        "int_complex": lambda: dft_patch.integrate("ft_time"),
        "int_typed": lambda: typed.integrate("time"),
        "int_units": lambda: patch.update_attrs(data_units="m/s").integrate("time"),
        "int_wacky": lambda: wacky.integrate("distance"),
        "int_wacky_definite": lambda: wacky.integrate("distance", definite=True),
        "int_no_dims": lambda: patch.integrate(()),
        "strain_rate": lambda: velocity.velocity_to_strain_rate(),
        "strain_rate_4": lambda: velocity.velocity_to_strain_rate(step_multiple=4),
        "strain_rate_order4": lambda: velocity.velocity_to_strain_rate(4, 4),
        "strain_rate_f32": lambda: _pinned(
            velocity.new(data=np.asarray(velocity.data, "float32")), "vf32"
        ).velocity_to_strain_rate(),
        "strain_rate_bad_odd": lambda: velocity.velocity_to_strain_rate(
            step_multiple=3
        ),
        "strain_rate_bad_zero": lambda: velocity.velocity_to_strain_rate(
            step_multiple=0
        ),
        "strain_rate_bad_type": lambda: patch.velocity_to_strain_rate(),
        "edgeless_1": lambda: velocity.velocity_to_strain_rate_edgeless(),
        "edgeless_3": lambda: velocity.velocity_to_strain_rate_edgeless(
            step_multiple=3
        ),
        "edgeless_positional": lambda: velocity.velocity_to_strain_rate_edgeless(5),
        "edgeless_bad_zero": lambda: velocity.velocity_to_strain_rate_edgeless(0),
        "edgeless_bad_float": lambda: velocity.velocity_to_strain_rate_edgeless(1.0),
        "edgeless_long": lambda: velocity.velocity_to_strain_rate_edgeless(10_000),
        "edgeless_bad_type": lambda: patch.velocity_to_strain_rate_edgeless(),
        "pws_distance": lambda: patch.phase_weighted_stack("distance"),
        "pws_time": lambda: patch.phase_weighted_stack("time", "distance"),
        "pws_power": lambda: patch.phase_weighted_stack("distance", power=1),
        "pws_squeeze": lambda: patch.phase_weighted_stack(
            "distance", dim_reduce="squeeze"
        ),
        "pws_reduce_mean": lambda: patch.phase_weighted_stack(
            "distance", "time", 3, "mean"
        ),
        "pws_reduce_callable": lambda: patch.phase_weighted_stack(
            "distance", dim_reduce=np.max
        ),
        "pws_bad_reduce": lambda: patch.phase_weighted_stack("time", dim_reduce="x"),
        "pws_f32": lambda: f32.phase_weighted_stack("distance"),
        "pws_int": lambda: int_patch.phase_weighted_stack("distance"),
        "pws_3d": lambda: three_d.phase_weighted_stack("dim_1", transform_dim="dim_2"),
        "pws_bad_3d": lambda: three_d.phase_weighted_stack("dim_1"),
        "pws_bad_1d": lambda: (
            patch.mean("time").squeeze().phase_weighted_stack("distance")
        ),
        "pws_bad_complex": lambda: dft_patch.phase_weighted_stack("distance"),
    }


def get_calls() -> dict:
    """Return the calls to compare, keyed by a name for the report."""
    patch = _pinned(dc.get_example_patch(), "example")
    null_patch = _pinned(dc.get_example_patch("patch_with_null"), "null")
    dft_patch = patch.dft("time")
    # Pinned like the patches above: replacing a patch's data outside an
    # operation gives the result a random id, which no two runs share.
    int_patch = _pinned(
        patch.new(data=(np.asarray(patch.data) * 10).astype("int32")), "int"
    )
    bool_patch = _pinned(patch.new(data=np.asarray(patch.data) > 0.5), "bool")
    collapsed = patch.mean("time")
    # Use a nonempty data_type so failures to clear it are visible.
    typed = _pinned(patch.update_attrs(data_type="strain_rate"), "typed")
    with_nondim = patch.update_coords(
        quality=("distance", np.arange(patch.shape[0], dtype="float64"))
    )
    # Small enough for the slow window filters to be timed in a few rounds.
    small = _pinned(patch.isel(distance=slice(0, 40), time=slice(0, 400)), "small")
    small_f32 = _pinned(small.new(data=np.asarray(small.data, "float32")), "f32")
    spiky = np.asarray(small.data).copy()
    spiky[10, 5], spiky[20, 50] = 10.0, -8.0
    spiky = _pinned(small.new(data=spiky), "spiky")
    hz, m, s = dc.get_unit("Hz"), dc.get_unit("m"), dc.get_unit("s")
    f32 = _pinned(patch.new(data=np.asarray(patch.data, "float32")), "f32_full")
    # Unevenly sampled, so the derivatives and tapers take coordinate values.
    unsorted = dc.get_example_patch("wacky_dim_coords_patch").isel(time=slice(0, 200))
    wacky = _pinned(unsorted.sort_coords("distance"), "wacky")
    return {
        **_filter_calls(small, small_f32, spiky, hz, m, s),
        **_envelope_calls(patch, int_patch, f32, dft_patch, wacky, m, s),
        "taper_bad_unsorted": lambda: unsorted.taper(distance=0.1),
        "diff_bad_unsorted": lambda: unsorted.differentiate("distance"),
        **_calculus_calls(patch, int_patch, f32, dft_patch, wacky, typed),
        **_aggregate_calls(patch, null_patch, int_patch, bool_patch, typed),
        # The inputs themselves, so a difference in the examples cannot
        # masquerade as a difference in the functions.
        "input_patch": lambda: patch,
        "input_null": lambda: null_patch,
        "input_dft": lambda: dft_patch,
        # operators
        "add_scalar": lambda: patch + 1,
        "sub_patch": lambda: patch - patch,
        "mul_scalar": lambda: patch * 2.5,
        "pow": lambda: patch**2,
        "compare": lambda: patch > 0.5,
        "rsub": lambda: 1 - patch,
        "np_exp": lambda: np.exp(patch),
        "np_abs": lambda: np.abs(patch),
        "np_fmod": lambda: np.fmod(patch, 2),
        "units_mul": lambda: patch * dc.get_quantity("m"),
        "add_reduce": lambda: patch.add.reduce(dim="time"),
        "add_accumulate": lambda: patch.add.accumulate(dim="time"),
        "np_mean": lambda: np.mean(patch, axis=0),
        "int_add": lambda: int_patch + 1,
        "int_pow": lambda: int_patch**2,
        # aggregations
        **{
            f"agg_{name}": (lambda name=name: getattr(patch, name)("time"))
            for name in ("min", "max", "mean", "median", "std", "sum", "first", "last")
        },
        **{
            f"agg_{name}_all": (lambda name=name: getattr(patch, name)())
            for name in ("min", "max", "mean", "std", "sum")
        },
        "agg_any": lambda: patch.any("time"),
        "agg_all": lambda: patch.all("time"),
        "agg_squeeze": lambda: patch.mean("time", dim_reduce="squeeze"),
        "agg_method_str": lambda: patch.aggregate("time", method="mean"),
        "agg_null_mean": lambda: null_patch.mean("time"),
        "agg_null_std": lambda: null_patch.std("distance"),
        "agg_int_mean": lambda: int_patch.mean("time"),
        "agg_int_sum": lambda: int_patch.sum("time"),
        "agg_bool_sum": lambda: bool_patch.sum("time"),
        "agg_bool_min": lambda: bool_patch.min("time"),
        "agg_complex_mean": lambda: dft_patch.mean("ft_time"),
        "agg_complex_std": lambda: dft_patch.std("ft_time"),
        # basic
        **{
            f"norm_{norm}": (lambda norm=norm: patch.normalize("time", norm=norm))
            for norm in ("l1", "l2", "max", "bit")
        },
        "norm_int_l2": lambda: int_patch.normalize("time", norm="l2"),
        "norm_null_max": lambda: null_patch.normalize("time", norm="max"),
        "standardize": lambda: patch.standardize("time"),
        "standardize_int": lambda: int_patch.standardize("distance"),
        "demean": lambda: patch.demean("time"),
        "demedian": lambda: patch.demedian("time"),
        "abs": lambda: patch.abs(),
        "conj": lambda: dft_patch.conj(),
        "real": lambda: dft_patch.real(),
        "imag": lambda: dft_patch.imag(),
        "imag_real_data": lambda: patch.imag(),
        "angle": lambda: dft_patch.angle(),
        "angle_real": lambda: patch.angle(),
        "angle_int": lambda: int_patch.angle(),
        "fillna_0": lambda: null_patch.fillna(0),
        "fillna_no_inf": lambda: null_patch.fillna(1.5, include_inf=False),
        "fillna_int": lambda: int_patch.fillna(0),
        "dropna_any": lambda: null_patch.dropna("time", how="any"),
        "dropna_all": lambda: null_patch.dropna("distance", how="all"),
        "flip_time": lambda: patch.flip("time"),
        "flip_all": lambda: patch.flip(*patch.dims),
        "flip_no_coords": lambda: patch.flip("time", flip_coords=False),
        "full_float": lambda: patch.full(1.0),
        "full_int": lambda: patch.full(0),
        "roll": lambda: patch.roll(time=5, samples=True),
        "roll_coord": lambda: patch.roll(time=5, samples=True, update_coord=True),
        "where_array": lambda: patch.where(patch.data > 0.5),
        "where_patch": lambda: patch.where(patch > 0.5),
        "where_other": lambda: patch.where(patch.data > 0.5, other=0),
        "pad_tuple": lambda: patch.pad(time=(2, 3), samples=True),
        "pad_no_expand": lambda: patch.pad(time=2, samples=True, expand_coords=False),
        "pad_fill": lambda: patch.pad(time=1, samples=True, constant_values=1.0),
        "pad_two_dims": lambda: patch.pad(time=1, distance=2, samples=True),
        # data_type is cleared by these; only a typed input can show it
        "norm_typed": lambda: typed.normalize("time"),
        "standardize_typed": lambda: typed.standardize("time"),
        "abs_typed": lambda: typed.abs(),
        "conj_typed": lambda: typed.conj(),
        # the other axis, the other dtypes, and the default argument
        "norm_l2_distance": lambda: patch.normalize("distance", norm="l2"),
        "norm_complex_l2": lambda: dft_patch.normalize("ft_time", norm="l2"),
        "norm_units": lambda: patch.update_attrs(data_units="m/s").normalize("time"),
        "standardize_distance": lambda: patch.standardize("distance"),
        "standardize_complex": lambda: dft_patch.standardize("ft_time"),
        "demean_distance": lambda: patch.demean("distance"),
        "demean_complex": lambda: dft_patch.demean("ft_time"),
        "abs_complex": lambda: dft_patch.abs(),
        # the messages, which a rewrite can change without changing a number
        "norm_bad": lambda: patch.normalize("time", norm="nope"),
        "transpose_bad_dim": lambda: patch.transpose("nope"),
        "rename_missing": lambda: patch.rename_coords(nope="x"),
        # coords
        "transpose": lambda: patch.transpose(),
        # The no-op and ellipsis branches, which nothing else here reaches.
        # `transpose_noop` must keep handing back the patch it was given.
        "transpose_noop": lambda: patch.transpose(*patch.dims),
        "transpose_ell_last": lambda: patch.transpose(..., "distance"),
        "transpose_ell_first": lambda: patch.transpose("distance", ...),
        "rename_coords": lambda: patch.rename_coords(distance="depth"),
        "rename_nondim": lambda: with_nondim.rename_coords(quality="grade"),
        "transpose_named": lambda: patch.transpose("time", "distance"),
        "squeeze": lambda: patch.select(distance=0, samples=True).squeeze(),
        "broadcast": lambda: collapsed.make_broadcastable_to((collapsed.shape[0], 3)),
        "update_coords": lambda: patch.update_coords(
            distance=patch.get_array("distance") + 1
        ),
    }


def digest(patch) -> dict:
    """Return a fingerprint of everything a patch carries."""
    data = np.asarray(patch.data)
    coords = {
        name: _hash(patch.get_array(name)) for name in sorted(patch.coords.coord_map)
    }
    # Ignore argument reprs in history. The ids are a field of their own,
    # so `--fields` can leave them out against a ref which names or derives
    # them differently; the leaves are pinned, so a changed id otherwise
    # means a changed operation id or stamping rule.
    names = {"origin_id", "data_id", "patch_id", "processing_id"}
    dumped = patch.attrs.model_dump(exclude={"history", "coords"})
    ids = {i: str(v) for i, v in sorted(dumped.items()) if i in names}
    attrs = {i: v for i, v in dumped.items() if i not in names}
    return {
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "dims": list(patch.dims),
        "data_hash": _hash(data),
        "coords": coords,
        "attrs": {i: str(v) for i, v in sorted(attrs.items())},
        "ids": ids,
    }


def _hash(array) -> str:
    """Return a hash of an array's contents."""
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path: Path) -> None:
    """Write the fingerprint of every call to path."""
    warnings.simplefilter("ignore")
    calls = get_calls() | get_matrix_calls()
    # Hash inputs first to detect mutations that result fingerprints would miss.
    inputs_before = _input_digests(calls)
    out, timing = {}, {}
    for name, call in calls.items():
        try:
            seconds, patch = _timed(call)
            out[name] = digest(patch)
            timing[name] = seconds
        except Exception as error:
            out[name] = {"error": f"{type(error).__name__}: {error}"}
    inputs_after = _input_digests(calls)
    for name, before in inputs_before.items():
        if inputs_after.get(name) != before:
            out[name] = {"error": "the call changed the patch it was given"}
    # Recorded so the caller can prove which dascore was measured.
    out["_dascore_path"] = str(Path(dc.__file__).parent)
    out["_timing"] = timing
    path.write_text(json.dumps(out, indent=1, sort_keys=True))


def _timed(call) -> tuple[float, Any]:
    """
    Return the best repeated-call time and the last result.

    Repeat fast calls for a few milliseconds to reduce scheduling noise; limit slow
    calls to a few rounds.
    """
    best, patch, spent, rounds = None, None, 0.0, 0
    while rounds < _TIMING_MAX_ROUNDS and (
        rounds < _TIMING_MIN_ROUNDS or spent < _TIMING_BUDGET
    ):
        start = time.perf_counter()
        patch = call()
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
        spent += elapsed
        rounds += 1
    return best, patch


def _input_digests(calls) -> dict:
    """
    Fingerprint every input patch, including closures and default arguments.

    This detects mutations to any input, including matrix patches passed as defaults.
    """
    seen = {}
    for name, call in calls.items():
        held = [x.cell_contents for x in getattr(call, "__closure__", None) or ()]
        held.extend(getattr(call, "__defaults__", None) or ())
        held.extend((getattr(call, "__kwdefaults__", None) or {}).values())
        patches = [x for x in held if isinstance(x, dc.Patch)]
        for index, patch in enumerate(patches):
            seen[f"_input_of/{name}/{index}"] = _hash(np.asarray(patch.data))
    return seen


def compare(before: dict, after: dict, fields: set[str] | None = None) -> list[str]:
    """
    Report calls with different results.

    `fields` restricts comparison to selected fingerprint fields, allowing data
    comparisons across refs with different attribute schemas.
    """
    report = []
    # Timing is not a result; it is reported on its own and never compared.
    names = (set(before) | set(after)) - _BOOKKEEPING
    # An input which changed explains every result downstream of it, so no
    # operation is told to raise its version.
    picked = _picked(before, after, names, fields)
    changed_input = any(
        _is_leaf(before.get(name)) and old != new for name, (old, new) in picked.items()
    )
    for name in sorted(names):
        old, new = picked[name]
        if old == new:
            continue
        if old is None or new is None:
            report.append(f"{name}: only in {'after' if old is None else 'before'}")
            continue
        # Not `fields`, which says what is being compared for every call.
        differing = sorted(i for i in set(old) | set(new) if old.get(i) != new.get(i))
        gated = not changed_input and _same_recipe(before[name], after[name])
        report.append(_headline(name, differing, gated))
        report.extend(
            f"    {i}\n      before: {old.get(i)}\n      after:  {new.get(i)}"
            for i in differing
        )
    return report


def _headline(name: str, differing: list[str], gated: bool) -> str:
    """Return the line which says what kind of difference this is."""
    if gated and set(differing) - {"ids"}:
        gate = "same data_id, different content — raise the operation's version"
        return f"{name}: {gate}"
    return f"{name}: differs in {differing}"


def _picked(before: dict, after: dict, names, fields) -> dict:
    """Return each call's two fingerprints, narrowed to what is compared."""
    return {
        name: (_select(before.get(name), fields), _select(after.get(name), fields))
        for name in names
    }


def _same_recipe(old: dict, new: dict) -> bool:
    """
    Whether both sides are the result of one derivation.

    Whole fingerprints: `--fields` may leave the ids out of what is
    compared. A leaf's id is assigned rather than derived, so it names no
    recipe.
    """
    ids = [(x.get("ids") or {}).get("data_id") for x in (old, new)]
    derived = not _is_leaf(old) and not _is_leaf(new)
    return bool(ids[0]) and ids[0] == ids[1] and derived


def _is_leaf(fingerprint: dict | None) -> bool:
    """Whether a fingerprint is of an input, whose two ids are one."""
    ids = (fingerprint or {}).get("ids") or {}
    return bool(ids.get("data_id")) and ids.get("data_id") == ids.get("origin_id")


def _select(fingerprint: dict | None, fields: set[str] | None) -> dict | None:
    """Return the part of a fingerprint being compared."""
    if fingerprint is None or fields is None:
        return fingerprint
    # An error is never dropped: a call which raised on one side and not
    # the other is a difference whatever fields were asked for.
    kept = {i: v for i, v in fingerprint.items() if i in fields or i == "error"}
    return kept


def report_timing(before: dict, after: dict, slowest: int = 15) -> list[str]:
    """
    Return what the change cost, call by call.

    Best-of readings on one machine, so this is a guide rather than a
    benchmark: it says which calls moved, not by exactly how much.
    """
    old, new = before.get("_timing", {}), after.get("_timing", {})
    shared = sorted(set(old) & set(new))
    if not shared:
        return []
    rows = [
        (new[i] / old[i], old[i], new[i], i)
        for i in shared
        # A call too quick to time is not evidence of anything; reporting
        # it just fills the list with whichever ones the scheduler noticed.
        if old[i] >= _TIMING_FLOOR and new[i] >= _TIMING_FLOOR
    ]
    rows.sort(reverse=True)
    total_old = sum(old[i] for i in shared)
    total_new = sum(new[i] for i in shared)
    out = [
        f"timing over {len(shared)} calls: "
        f"{total_old * 1e3:.1f} ms before, {total_new * 1e3:.1f} ms after "
        f"({(total_new / total_old - 1) * 100:+.1f}%)",
        "  the two legs are separate processes, so a single call can read "
        f"{_TIMING_NOISE:.0%} out either way;",
        "  the total is the number to trust, and benchmarks/ is where a "
        "single operation gets measured properly.",
    ]
    moved = [x for x in rows if x[0] >= 1 + _TIMING_NOISE or x[0] <= 1 - _TIMING_NOISE]
    if not moved:
        out.append(f"  no call moved by more than {_TIMING_NOISE:.0%}.")
        return out
    out.append(f"  calls which moved more than {_TIMING_NOISE:.0%}:")
    out.extend(
        f"    {name:34} {before_s * 1e6:9.1f} us -> {after_s * 1e6:9.1f} us "
        f"({(ratio - 1) * 100:+6.1f}%)"
        for ratio, before_s, after_s, name in moved[:slowest]
    )
    return out


def _dump_at(worktree: Path, out_path: Path) -> dict:
    """Dump the fingerprints using the dascore in worktree."""
    # Script execution adds the script directory to sys.path; set PYTHONPATH
    # to import the requested worktree rather than another installed copy.
    env = {**os.environ, "PYTHONPATH": str(worktree)}
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--dump", str(out_path)],
        cwd=worktree,
        env=env,
        check=True,
    )
    result = json.loads(out_path.read_text())
    check_dascore_path(result.pop("_dascore_path"), worktree)
    return result


def check_dascore_path(used: str, worktree: Path) -> None:
    """Raise unless the dascore which ran is the one in worktree."""
    if used != str(worktree / "dascore"):
        msg = f"expected the dascore in {worktree}, imported {used}"
        raise RuntimeError(msg)


def main(ref: str, fields: set[str] | None = None, strict: bool = False) -> int:
    """Compare the working tree against a git ref."""
    repo = Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        worktree = temp / "baseline"
        try:
            subprocess.run(
                ["git", "worktree", "add", "--detach", str(worktree), ref],
                cwd=repo,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as error:
            # Otherwise an unknown ref is just a non-zero exit status.
            msg = f"could not check out {ref!r}: {error.stderr.strip()}"
            raise SystemExit(msg) from error
        try:
            # Alternate runs and keep each call's best time to reduce cache and CPU
            # warmup bias.
            before = _dump_at(worktree, temp / "before.json")
            after = _dump_at(repo, temp / "after.json")
            for index in range(_TIMING_PASSES - 1):
                after = _merge_timing(
                    after, _dump_at(repo, temp / f"after{index}.json")
                )
                before = _merge_timing(
                    before, _dump_at(worktree, temp / f"before{index}.json")
                )
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
                check=False,
                capture_output=True,
            )
    # Timing is reported whatever the verdict: a change which alters no
    # value can still cost, and that is worth seeing on a passing run.
    counted = len(before) - len(_BOOKKEEPING & set(before))
    if timing := report_timing(before, after):
        print("\n".join(timing), end="\n\n")  # noqa
    if strict and (raised := _raised(before) | _raised(after)):
        print(f"calls which raised on one side or both: {sorted(raised)}")  # noqa
        return 1
    if report := compare(before, after, fields):
        print(f"{counted} calls compared against {ref}; some differ:\n")  # noqa
        print("\n".join(report))  # noqa
        return 1
    print(f"{counted} calls compared against {ref}; all identical.")  # noqa
    return 0


def _merge_timing(kept: dict, other: dict) -> dict:
    """Return one dump holding the best timing of two runs of the same code."""
    best = dict(kept.get("_timing", {}))
    for name, seconds in other.get("_timing", {}).items():
        best[name] = min(seconds, best.get(name, seconds))
    return kept | {"_timing": best}


def _raised(dumped: dict) -> set[str]:
    """
    Return names of calls that recorded errors, other than the expected ones.

    Two matching errors compare equal, so without `--strict` a call broken on
    both sides would pass unnoticed.
    """
    return {
        i
        for i, v in dumped.items()
        if i not in _BOOKKEEPING
        and isinstance(v, dict)
        and "error" in v
        and not any(fnmatch.fnmatchcase(i, x) for x in EXPECTED_ERRORS)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="The git ref to compare against.")
    parser.add_argument("--dump", help="Write fingerprints here and exit.")
    parser.add_argument(
        "--fields",
        help=(
            "Comma separated fingerprint fields to compare, e.g. "
            "'dtype,shape,dims,data_hash,coords'. Use when the ref is far "
            "enough back that the attrs schema itself changed."
        ),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if any call outside EXPECTED_ERRORS raised, even where both "
            "sides raised alike."
        ),
    )
    args = parser.parse_args()
    if args.dump:
        dump(Path(args.dump))
    elif not args.ref:
        parser.error("--ref is required; it names the checkout to compare against.")
    else:
        chosen = {i.strip() for i in args.fields.split(",")} if args.fields else None
        sys.exit(main(args.ref, chosen, args.strict))
