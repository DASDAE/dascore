"""
Compare what patch functions return now against what they returned before.

The patch data path is being rewritten to use the
[array API standard](https://data-apis.org/array-api/latest/) so that patch
data can be backed by libraries other than numpy. Those rewrites must not
change a single number a numpy backed patch produces, and the test suite
cannot prove that on its own: it only exercises what someone thought to
assert, and it moves along with the code.

This script settles it directly. It runs the same calls against a checkout
of any git ref and against the working tree, fingerprints every result, and
reports which ones differ. Nothing is compared approximately; the data are
hashed, so a difference in the last bit is a difference.

Usage:

    python scripts/differential_check.py --ref <git ref>

There are two lists of comparisons. get_calls holds calls against the
example patches, which carry datetime coordinates, units, and complex data
from a transform. MATRIX_CALLS is run against every array make_arrays
builds, so each call is checked for every dtype and for the values
implementations tend to disagree about: nan, infinities, a whole slice of
nulls, and numbers big enough to overflow. Add to whichever fits when a
patch function is rewritten; one which isn't listed isn't checked.
"""

from __future__ import annotations

import argparse
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

# How hard to work at timing a call. A microsecond-scale call is repeated
# until it has run for _TIMING_BUDGET so the reading means something; a
# slow one stops at _TIMING_MIN_ROUNDS so the sweep stays quick.
_TIMING_MIN_ROUNDS = 3
_TIMING_MAX_ROUNDS = 200
_TIMING_BUDGET = 0.002
# Timings below this are noise on any machine, so they are not reported
# however far they appear to have moved.
_TIMING_FLOOR = 50e-6
# How far a single call can read out with nothing changed at all. Measured
# by running this script with --ref HEAD, which compares a checkout against
# itself: the totals landed within 10% and individual calls within 35%.
_TIMING_NOISE = 0.35
# How many times each leg is run. Two is enough to stop the leg which
# happened to go first from looking slow.
_TIMING_PASSES = 3

# Keys a dump carries which are not calls.
_BOOKKEEPING = {"_timing", "_dascore_path"}


# Data covering the dtypes patch data can hold and the values which
# implementations tend to disagree about. A hand written list of calls
# misses these; the where in #921 kept a float32 patch float32 where numpy
# had promoted it to float64, and only this matrix noticed.
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
    "flip_noop": lambda patch: patch.flip(),
    "flip_distance": lambda patch: patch.flip("distance"),
    "demedian_distance": lambda patch: patch.demedian("distance"),
    "full_bool": lambda patch: patch.full(True),
    "update_coords_replace": lambda patch: patch.update_coords(
        time=patch.get_array("time")
    ),
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
}


def _matrix_patch(array):
    """Wrap an array in a patch with evenly sampled coordinates."""
    coords = {
        "distance": np.arange(array.shape[0]) * 1.0,
        "time": np.arange(array.shape[1]) * 0.5,
    }
    return dc.Patch(data=array, coords=coords, dims=("distance", "time"))


def get_matrix_calls() -> dict:
    """Return every call in MATRIX_CALLS against every array."""
    out = {}
    for array_name, array in make_arrays().items():
        patch = _matrix_patch(array)
        out[f"matrix/{array_name}/input"] = lambda patch=patch: patch
        for call_name, call in MATRIX_CALLS.items():
            key = f"matrix/{array_name}/{call_name}"
            out[key] = lambda call=call, patch=patch: call(patch)
    return out


def get_calls() -> dict:
    """Return the calls to compare, keyed by a name for the report."""
    patch = dc.get_example_patch()
    null_patch = dc.get_example_patch("patch_with_null")
    dft_patch = patch.dft("time")
    int_patch = patch.new(data=(np.asarray(patch.data) * 10).astype("int32"))
    bool_patch = patch.new(data=np.asarray(patch.data) > 0.5)
    collapsed = patch.mean("time")
    # A patch which states a data_type, so that clearing it is visible.
    # Every other patch here already carries "", where a processor which
    # forgot to clear it would read the same as one which cleared it.
    typed = patch.update_attrs(data_type="strain_rate")
    with_nondim = patch.update_coords(
        quality=("distance", np.arange(patch.shape[0], dtype="float64"))
    )
    return {
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
        # The branches these five reach which nothing else here does.
        "flip_noop": lambda: patch.flip(),
        "flip_distance": lambda: patch.flip("distance"),
        "flip_complex": lambda: dft_patch.flip("ft_time"),
        "fillna_nothing_to_do": lambda: patch.fillna(0),
        "fillna_complex": lambda: dft_patch.fillna(0),
        "fillna_null_inf_only": lambda: null_patch.fillna(-1, include_inf=True),
        "full_complex": lambda: dft_patch.full(1 + 1j),
        "full_bool": lambda: patch.full(True),
        "full_on_int": lambda: int_patch.full(3),
        "demedian_distance": lambda: patch.demedian("distance"),
        "demedian_null": lambda: null_patch.demedian("time"),
        "demedian_int": lambda: int_patch.demedian("time"),
        "update_coords_new": lambda: patch.update_coords(
            quality=("distance", np.arange(patch.shape[0], dtype="float64"))
        ),
        "update_coords_replace": lambda: patch.update_coords(
            distance=patch.get_array("distance") * 2
        ),
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
    # History holds the repr of the arguments, which says nothing about the
    # answer, so it is left out. `patch_id` goes with it for a harder
    # reason: this dumps in two processes, and a patch not read from a
    # file mints one, so every patch would differ and the check would say
    # nothing. `processing_id` stays -- it is a digest of the route, the
    # same in both processes, so it catches a call which stopped being
    # stamped or started fingerprinting its arguments differently.
    attrs = patch.attrs.model_dump(exclude={"history", "coords", "patch_id"})
    return {
        "dtype": str(data.dtype),
        "shape": list(data.shape),
        "dims": list(patch.dims),
        "data_hash": _hash(data),
        "coords": coords,
        "attrs": {i: str(v) for i, v in sorted(attrs.items())},
    }


def _hash(array) -> str:
    """Return a hash of an array's contents."""
    return hashlib.md5(np.ascontiguousarray(array).tobytes()).hexdigest()


def dump(path: Path) -> None:
    """Write the fingerprint of every call to path."""
    warnings.simplefilter("ignore")
    calls = get_calls() | get_matrix_calls()
    # Hashed before anything runs, so a call which writes into its own
    # argument can be told from one which does not. Nothing else here
    # would notice: every digest is taken from the call's own result.
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
    Return how long a call took, and what it returned.

    Best of several, and enough of them to be worth reading: a call which
    takes a microsecond gets repeated until it has run for a few
    milliseconds, so the answer is about the code rather than about when
    the scheduler happened to look away. A slow call is measured a few
    times and left alone. The last result is the one fingerprinted; they
    are all the same call, so any of them would do.
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
    Return a fingerprint of every patch the calls were given.

    Both ways a call can be holding one: `get_calls` closes over its
    patches, and `get_matrix_calls` passes them as default arguments.
    Reading only the closures left the matrix half unchecked, which is
    the half carrying the dtypes and the special values -- so a call
    which wrote into one of those would have gone unnoticed.

    Every patch found is digested, not just the first: a call given two
    of them can spoil either.
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
    Return a report of the calls whose results differ.

    `fields` restricts the comparison to part of a fingerprint. Comparing
    against a ref far enough back that the attrs schema itself changed --
    master, at the time of writing -- otherwise reports every call as a
    difference and so reports nothing; the numbers are still worth
    checking there, and this is how.
    """
    report = []
    # Timing is not a result; it is reported on its own and never compared.
    names = (set(before) | set(after)) - _BOOKKEEPING
    for name in sorted(names):
        old, new = _select(before.get(name), fields), _select(after.get(name), fields)
        if old == new:
            continue
        if old is None or new is None:
            report.append(f"{name}: only in {'after' if old is None else 'before'}")
            continue
        fields = sorted(i for i in set(old) | set(new) if old.get(i) != new.get(i))
        report.append(f"{name}: differs in {fields}")
        report.extend(
            f"    {i}\n      before: {old.get(i)}\n      after:  {new.get(i)}"
            for i in fields
        )
    return report


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
    # PYTHONPATH, not the working directory: running a script file puts the
    # script's own directory on sys.path rather than the cwd, so a cwd of
    # the worktree would silently import the installed dascore instead.
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
            # Alternated, and more than once: run one leg after the other
            # and the first pays for a cold cache and a CPU which has not
            # yet ramped, which reads as the second being faster. Taking
            # the best of each pass per call takes most of that out.
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
    """Return the names of calls which recorded an error.

    An error is stored as its message and compared like any other field,
    so a call which fails the same way on both sides reads as a pass.
    `--strict` is how to notice that the check is not checking anything.
    """
    return {
        i
        for i, v in dumped.items()
        if i not in _BOOKKEEPING and isinstance(v, dict) and "error" in v
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
        help="Fail if any call raised, even where both sides raised alike.",
    )
    args = parser.parse_args()
    if args.dump:
        dump(Path(args.dump))
    elif not args.ref:
        parser.error("--ref is required; it names the checkout to compare against.")
    else:
        chosen = {i.strip() for i in args.fields.split(",")} if args.fields else None
        sys.exit(main(args.ref, chosen, args.strict))
