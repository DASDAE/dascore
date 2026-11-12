---
name: migrate-to-v1
description: Update code, files, and indexes written for DASCore 0.1.x to the DASCore 1.0 APIs, including behavior that changed without raising.
---

This guide covers the changes between DASCore 0.1.24 and 1.0 which can break existing code. Most of them raise a clear error, but a few change results without one; those are marked **silent** below and deserve a search of your code even if it runs. See the [announcement](dascore_1.0_announcement.qmd) for what is new.

An agent can follow this guide directly from this page; the project does not need DASCore 1.0 installed first, and 0.1.x has no skills command to load it from. Substitute the project environment's Python executable for `python` in the commands below.

## Workflow

1. Upgrade the environment to meet the minimum versions in DASCore's [`pyproject.toml`](https://github.com/DASDAE/dascore/blob/dev/pyproject.toml), and install DASCore 1.0 with `python -m pip install "dascore[all]>=1.0.0b1"`, which admits a DASCore pre-release while keeping other dependencies on stable releases. Confirm that the project's Python imports 1.0 or a 1.0 pre-release from that environment with `python -c "import dascore; print(dascore.__version__, dascore.__file__)"`.
2. Search the code for every old name in the tables below, and replace each with its 1.0 spelling. Read the installed docstring of each replacement before using it, since arguments changed too.
3. Search separately for the **silent** changes: coordinate fields written through attrs, `attrs.get` of coordinate fields, gap merging in `chunk`, glob negation, STFT time values, and correlation units.
4. Plan for existing files: 0.1.x DASDAE files need an explicit opt-in to read, old spool indexes are replaced, and DASDAE files written by 1.0 cannot be read by 0.1.x.
5. Run the project's tests. Treat new `ParameterError`, `UnitError`, `CoordError`, `PatchCoordinateError`, and `InvalidSpoolQueryError` exceptions as places where 0.1.x computed something questionable, not as noise to suppress.

## Requirements

DASCore 1.0 raises its minimum Python, NumPy, and pandas versions; check `requires-python` and `dependencies` in [`pyproject.toml`](https://github.com/DASDAE/dascore/blob/dev/pyproject.toml).

pandas 3 stores text columns as strings rather than `object`, which can affect code that inspects `Spool.get_contents()` or `dc.scan_to_df()` dtypes.

## Removed APIs

| 0.1.x | 1.0 |
|---|---|
| `patch.assign_coords(...)` | `patch.update_coords(...)` |
| `patch.iselect(time=(0, 10))` | `patch.select(time=(0, 10), samples=True)` or `patch.isel(time=slice(0, 10))` |
| `patch.iresample(...)` | `patch.resample(..., samples=True)` |
| `patch.tran.*` | the same method on the patch, e.g. `patch.dft(...)` |
| `dascore.transform.rfft`, `patch.rfft`, `patch.tran.rfft` | `patch.dft(dim, real=True)`. **Silent:** `dft` scales by the sample spacing, so its values are the old ones times the step, and it pads to a fast length unless `pad=False` |
| `patch.spectrogram(...)` | `patch.stft(time=<window>, ...)`, which returns complex coefficients rather than the real power values `spectrogram` returned; take `abs(...) ** 2` of the data for power |
| `patch.viz.spectrogram(nperseg=..., noverlap=...)` | `patch.viz.spectrogram(time=<window>, overlap=...)`, with the same window arguments as `patch.stft` |
| `Patch.from_parts(...)` | `dc.Patch(data=..., coords=..., attrs=...)` |
| `velocity_to_strain_rate(gauge_multiple=g)` | `velocity_to_strain_rate(step_multiple=2 * g)`; copying the value unchanged halves the gauge length |
| `correlate(..., lag=...)` | drop `lag`, which 0.1.x ignored; to restrict the output, select on its lag coordinate, e.g. `out.select(lag_time=(-1, 1))` |
| `slope_filter(..., notch=...)` | `slope_filter(..., invert=...)` |
| `MemorySpool`, `DirectorySpool`, `FileSpool`, `dascore.clients` | `dc.spool(...)`, which returns a `dc.Spool` |
| `select_kwargs` | call `spool.select(...)` |
| `spool.map(func, progress=True)` | `spool.map(func)`; `progress` takes `"standard"`, `"basic"`, `None`, or `False` |
| `dascore.io.PatchFileSummary` | `dc.PatchSummary` |
| `dascore.io.indexer.DirectoryIndexer` | `dc.spool(path)` |
| `dascore.io.sintela_binary` | `dascore.io.sintela` (reading through `dc.read` is unchanged) |
| `dascore.utils.hdf5.PyTablesReader`, `HDF5Reader`, `HDF5Writer` | `H5Reader`, `H5Writer`; these open h5py files, so code using the PyTables node API must move to h5py |
| `dascore.utils.array.hash_numpy_array` | `dascore.utils.array.hash_array` |
| `to_float.register`, `to_int.register` | register on the private `_to_float`, `_to_int` |
| `dc.proc.align_patch_coords` | `dascore.utils.patch.align_patch_coords` |
| `DEFAULT_ATTRS_TO_IGNORE`, `PATCH_MERGE_ATTRS`, `attrs_to_ignore=` | the `patch_kind_attrs` config option; see [Patch compatibility](../notes/patch_compatibility.qmd) |
| `dascore.utils.patch.merge_patches(patches)` | `dc.spool(patches).chunk(time=None)` |
| `dascore.utils.patch.concatenate_patches(patches, time=None)` | `dc.spool(patches).concatenate(time=None)`, which puts patches that cannot be concatenated together in separate outputs rather than skipping them |
| `dascore.utils.patch.get_patch_window_size`, `get_window_axis_step` | `dascore.utils.window.resolve_window(...)`; `.full_size()` gives the window along every axis |
| `dascore.utils.pd.get_regex` | `dascore.utils.misc.glob_to_regex`, which negates a character class with `[^...]` rather than fnmatch's `[!...]` |
| `dascore.utils.hdf5.unpack_scalar_h5_dataset(dataset)` | `dataset[()]`, taking the first element of a length-one array |
| `dascore.utils.misc.yield_sub_sequences` | slice the sequence directly |

## Attributes and coordinates

In 0.1.x, `patch.attrs` duplicated each coordinate's bounds, step, and units as flat fields such as `time_min` and `distance_step`, and every patch carried empty `network`, `station`, `instrument_id`, and `acquisition_id` fields. In 1.0, attrs hold only non-coordinate metadata, the coordinates are the single source of truth, and an attribute exists only when something sets it.

| 0.1.x | 1.0 |
|---|---|
| `patch.attrs.time_min`, `patch.attrs["time_max"]` | `patch.get_coord("time").min()`, `.max()`, or `patch.coords.min("time")` |
| `patch.attrs.time_step` | `patch.get_coord("time").step` |
| `patch.attrs.distance_units` | `patch.get_coord("distance").units` |
| `patch.attrs.coords`, `patch.attrs.dims` | `patch.coords`, `patch.dims` |
| `patch.update_attrs(time_min=t)` | `patch.update_coords(time_min=t)` |
| `patch.attrs.network`, `station` (always present, possibly empty) | `patch.attrs.get("network")`; set them explicitly, or use `acquisition_key` |
| `patch.attrs.acquisition_id` | `patch.attrs.acquisition_key` (`network.fiber_array.location.acquisition`) |
| `patch.attrs.instrument_id` | usually `patch.attrs["interrogator.serial_number"]`; the Febus and OptoDAS readers report `interrogator.name` |
| `*_units` companion attrs such as `gauge_length_units` or `pulse_width_units` | gone; attrs are stored in fixed units: lengths such as `gauge_length` in meters, `pulse_width` in seconds, `pulse_rate` in hertz |
| a pulse stated as a length, e.g. Febus `pulse_width` in meters | `pulse_length`, in meters |
| `patch.attrs.flat_dump()` with `time_min` style keys | `patch.summary.flat_dump()`; `patch.attrs.flat_dump()` no longer includes coordinate fields |
| missing numeric attrs (`gauge_length`, `pulse_width`, ...) as `nan` | `None` |

:::{.callout-warning}
Attribute names are no longer interpreted as coordinate metadata.

- `patch.update_attrs(time_min=...)`, and the same for `{dim}_max` and `{dim}_step`, used to change the coordinate. In 1.0 it stores an ordinary attribute, leaves the coordinate unchanged, and warns. Move these calls to `update_coords`.
- **Silent:** `update_attrs(distance_units=...)` and `patch.new(attrs={"time_min": ...})` change nothing about the coordinates and do not warn.
- **Silent:** `patch.attrs.get("time_min")` returns `None`, and `patch.attrs.flat_dump()` leaves coordinate fields out.
- **Silent:** numeric attrs read from files are converted to the fixed units above, so a `pulse_width` once stated in nanoseconds is now in seconds.
:::

An extra attribute holds one value. An array belongs on the patch as a coordinate: `patch.update_coords(name=(dims, array))`.

`dc.scan(path)` returns [`PatchSummary`](`dascore.PatchSummary`) objects rather than `PatchAttrs`. Read coordinate envelopes from `summary.get_coord_summary("time").min`, or use `dc.scan_to_df(path)`, whose `time_min` style columns remain.

## Spools

- **Indexes.** Directory indexes are now `.dascore_index.sqlite3`. Old `.dascore_index.h5` files are no longer used and can be deleted; DASCore warns when it finds one. SQLite indexes are not supported on network filesystems with unreliable locking.
- **`update`** is allowed only on the spool returned by `dc.spool(path)`, not on one derived from it by `select`, `chunk`, slicing, or `+`. Call `update()` first.
- **Unknown attributes.** `Spool.select` raises `InvalidSpoolQueryError` for a keyword which is neither a coordinate nor an attribute some patch states; 0.1.x ignored unknown names and treated the standard attributes as present but empty. Check `spool.get_contents().columns` before selecting on an optional attribute.
- **Contents.** `get_contents()` and `scan_to_df()` have no `patch`, `history`, or `{dim}_dtype` columns, and leave out attributes no patch states, such as an empty `station` or `data_units`. The `path`, `file_format`, and `file_version` columns are now `source_path`, `source_format`, and `source_version`.
- **Silent: joining.** `chunk` joins two patches only where the later starts one step after the earlier ends, give or take `max_shift` (0.2 of a step by default, from `dascore.config.max_shift`), where 0.1.x joined any seam up to `tolerance=1.5` steps wide. Seams 1.2 to 1.5 steps wide, and later patches more than 0.2 step off the earlier one's grid (including overlaps half a step off it), now start a new patch. Pass `tolerance=1.5` for the 0.1.x joining; `tolerance` is now off by default, a number is still a seam width in steps, and a quantity or timedelta is now the seam's width rather than the excess over one step. Each stretch of joined patches gets one evenly sampled grid: the roundest step its files state that keeps every file's ends within `max_shift`, else its span spread evenly as 0.1.x did, so `chunk(time=...)` windows rejoin exactly. Labels with no step join only under a quantity or timedelta `tolerance`. `get_gaps` and `get_coverage` take the same `tolerance` and `max_shift` and list where `chunk` splits, with a `missing` sample count.
- **Silent: gaps.** A `tolerance` wide enough to span a gap joins the patches and spreads their labels evenly over it, as 0.1.x did, with the same warning above 1.5 steps. Pass `fill_value=` as well to keep the declared step and write the missing samples instead, e.g. `chunk(time=None, tolerance=5, fill_value=np.nan)`.
- **`chunk(snap_coords=...)`** is deprecated and ignored, with a warning: `chunk` always fits one grid per stretch. Use `Spool.concatenate` to join patches keeping their recorded labels, and `Patch.snap_coords` to regularize labels.
- **Kinds.** `chunk`, `concatenate`, and `stack` combine only patches of the same kind, decided by the `patch_kind_attrs` config option; patches of other kinds come out separately, and `stack` warns. `data_type` is no longer part of a patch's kind, so chunking a spool which mixes, say, strain rate and velocity raises `CoordMergeError`, or merges them under `conflict="drop"`. Pass `group="data_type"` to keep them apart as 0.1.x did.
- **Directory names.** Hive-style `key=value` directory names become string attributes of the patches beneath them.

## Coordinates

The coordinate classes were unified into one run-based [`NumericCoord`](`dascore.core.coords.NumericCoord`), plus string coordinates. Build coordinates with [`dc.get_coord`](`dascore.get_coord`) rather than a class.

| 0.1.x | 1.0 |
|---|---|
| `CoordRange`, `CoordArray`, `CoordMonotonicArray`, `CoordPartial` | `NumericCoord`; build with `dc.get_coord(...)` |
| `isinstance(coord, CoordRange)` | `coord.evenly_sampled` |
| `coord.limits` | `(coord.min(), coord.max())` |
| `coord.start` | `coord.values[0]` |
| `coord.stop` (exclusive, in sampling order) | `coord.values[-1] + coord.step` |
| `coord.update_data(values)` | `coord.update(values=values)` |
| `hash(coord)` | `coord.data_id` |

`coord.data` remains as an alias of `coord.values`.

`dc.get_coord(data=..., step=...)` now checks that the values sit on the stated grid and raises when they do not; 0.1.x ignored the step.

Selecting samples by a mask or array (`select`, `dropna`) keeps the coordinate's step and records the removed samples as holes, where 0.1.x dropped the step. `differentiate`, `integrate`, `taper`, `sobel_filter`, and filtered `decimate` raise `CoordError` on a coordinate with holes or with labels off its step's grid (jitter), where 0.1.x computed on the labels. Use `fill_gaps`, `split_gaps`, or `snap_coords` first. A coordinate with no step, such as irregular sensor positions, is still an irregular axis these operations compute on by its labels, as in 0.1.x.

**Silent:** `dc.get_coord(data=...)` without a `step` reads labels exactly by default: labels within 0.1% of one spacing of a grid, which 0.1.x snapped onto it, are now jittered or have no step. Pass `snap=True` for the 0.1.x result, or call `snap_coords` on the patch. A stated `step` still absorbs float rounding unless `snap=False`.

## Processing

- Functional entry points take the patch positionally: `dc.proc.demean(patch)`, not `dc.proc.demean(patch=patch)`.
- `Patch.apply_ufunc` takes the patch first: `patch.apply_ufunc(np.add, other)`. Code calling `Patch.apply_ufunc(ufunc, patch)` should use `dascore.utils.array.apply_ufunc`.
- `np.sort`, `np.argsort`, `np.partition`, and `numpy.fft` raise on patches instead of returning mislabelled coordinates. Apply them to `patch.data`, or use `Patch.dft`.
- Arithmetic between patches no longer requires `data_units` to match; attributes in `patch_kind_attrs`, such as `tag`, still must. It raises `PatchCoordinateError` when a shared dimension has no overlapping values instead of returning an empty patch.
- `Patch.order` raises `UnknownCoordinateError` for a coordinate the patch does not have, as `select` already did, and `CoordError` for values (or, with `samples=True`, positions) the coordinate does not hold, which 0.1.x dropped. Order only by values the coordinate holds.
- `Patch.coords_from_df` raises `ParameterError` rather than overwrite an existing coordinate. Drop it first with `drop_coords`, or set it with `update_coords`.
- **Silent: stack labels.** `Spool.stack(dim_vary=...)` keeps the first patch's labels for `dim_vary`; 0.1.x relabelled it to start at zero (1970 for times).
- `hampel_filter` without a dimension raises instead of returning the data unchanged.
- **Silent: STFT.** `Patch.stft` places its windows differently, so depending on window length and overlap the output `time` values, and the number of windows, can differ from 0.1.x. The output also carries `{dim}_start` and `{dim}_stop` coordinates giving each window's edges.
- **Silent: correlation units.** `correlate` reports the square of the input data units.
- **Silent: taper.** `taper` builds its ramp from `2n + 1` samples rather than `2n`, changing edge values slightly.
- **Inventory enrichment.** `Patch.enrich` and `Spool.enrich` raise `PatchError` when a value the file header states disagrees with the inventory; 0.1.x kept the header value. Fix the inventory, or pass `enrich(conflict="keep_first")` for the old behaviour (`"keep_last"` lets the inventory win).

## Units and time

- `dc.to_float` raises `UnitError` for a dimensionless or non-time quantity; 0.1.x silently reduced it to base units, so `25 MB` became `2e8`. A time quantity converts to seconds. Use `convert_units` for other conversions.
- `notch_filter` raises `UnitError` for a dimensionless quantity such as `20 %` on a coordinate without units; 0.1.x read it as a bare number.
- `dc.to_datetime64` and `dc.to_timedelta64` raise `TimeOverflowError` for times outside the nanosecond range rather than wrapping or returning `NaT`.

## Files

- **Reading 0.1.x DASDAE files.** DASDAE files written by 0.1.x store coordinate metadata as a pickle, which 1.0 does not unpickle by default, so reading, scanning, or indexing them raises `DASDAEUnpickleError`. For files you trust, read them inside `with dc.config_context(allow_dasdae_format_unpickle=True):`, and rewrite them once with 1.0 to drop the need.
- **Reading pickles.** Unpickling a file can run arbitrary code, so 1.0 neither detects nor reads the PICKLE format by default. `dc.read` and `dc.get_format` raise `PickleFormatDisabledError` for a DASCore pickle, and scans and directory spools skip it with a warning, whatever its extension. For files you trust, read them inside `with dc.config_context(allow_pickle_format=True):`, and rewrite them as DASDAE. While the setting is on, a scan unpickles every file whose header names a DASCore patch or spool, so do not enable it for directories that may hold files from others. A directory index built with the setting off has no entries for its pickles; rebuild it after opting in. `dc.get_example_patch` reads a local path under your config rather than enabling any unpickling for it.
- **Writing DASDAE.** `dc.write(..., "DASDAE")` writes format version 2, which 0.1.x cannot read. Pass `file_version="1"` while collaborators still run 0.1.x. Version 1 cannot hold fractional steps or holes, and 0.1.x reads the `origin_id` and `data_id` such files carry as ordinary attributes, so merging patches with different ids there needs `conflict="drop"`.
- **Write options.** `dc.write` raises `ParameterError` for an option the format's writer does not declare, rather than ignoring it.
- **Silent: globs.** String metadata globs follow SQLite, so a character class is negated with `[^...]`. The 0.1.x spelling `[!...]` no longer negates, and selects a different set of patches without an error.
- **Patch names.** `get_patch_names` defaults to `acquisition_key` and `tag`, rather than `network`, `station`, and `tag`, which changes the names DASCore gives patches inside files it writes.

## Format authors

A 0.1.x `FiberIO` overrode `get_format`, `scan`, and `read`. In 1.0 a format implements `get_version`, `get_metadata`, and `read_array`, and detection, scanning, and reading are derived from them; a reader overriding only the old methods still works but warns. Write options are declared as named parameters of `write` rather than read from `**kwargs`. See [Adding a new format](../contributing/new_format.qmd).
