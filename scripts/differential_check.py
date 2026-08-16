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

CALLS below is the list of comparisons. Add to it whenever a patch function
is rewritten; a function which isn't listed isn't checked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

import dascore as dc


def get_calls() -> dict:
    """Return the calls to compare, keyed by a name for the report."""
    patch = dc.get_example_patch()
    null_patch = dc.get_example_patch("patch_with_null")
    dft_patch = patch.dft("time")
    int_patch = patch.new(data=(np.asarray(patch.data) * 10).astype("int32"))
    bool_patch = patch.new(data=np.asarray(patch.data) > 0.5)
    collapsed = patch.mean("time")
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
        # coords
        "transpose": lambda: patch.transpose(),
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
    # answer, so it is left out.
    attrs = patch.attrs.model_dump(exclude={"history", "coords"})
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
    out = {}
    for name, call in get_calls().items():
        try:
            out[name] = digest(call())
        except Exception as error:
            out[name] = {"error": f"{type(error).__name__}: {error}"}
    # Recorded so the caller can prove which dascore was measured.
    out["_dascore_path"] = str(Path(dc.__file__).parent)
    path.write_text(json.dumps(out, indent=1, sort_keys=True))


def compare(before: dict, after: dict) -> list[str]:
    """Return a report of the calls whose results differ."""
    report = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
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


def main(ref: str) -> int:
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
            before = _dump_at(worktree, temp / "before.json")
            after = _dump_at(repo, temp / "after.json")
        finally:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree)],
                cwd=repo,
                check=False,
                capture_output=True,
            )
    if report := compare(before, after):
        print(f"{len(before)} calls compared against {ref}; some differ:\n")  # noqa
        print("\n".join(report))  # noqa
        return 1
    print(f"{len(before)} calls compared against {ref}; all identical.")  # noqa
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", help="The git ref to compare against.")
    parser.add_argument("--dump", help="Write fingerprints here and exit.")
    args = parser.parse_args()
    if args.dump:
        dump(Path(args.dump))
    elif not args.ref:
        parser.error("--ref is required; it names the checkout to compare against.")
    else:
        sys.exit(main(args.ref))
