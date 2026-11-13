---
name: dascore-make-annotations
description: Build a DASCore annotation set from a table of picks, ranges, paths, or polygons, then query, plot, and save it.
---

# Make a DASCore annotation set

Use this skill when a user has located things in DAS data (picks from a picker or spreadsheet, event windows, tracks, outlines of regions) and wants them as a DASCore `AnnotationSet`, or wants to plot, query, save, or cut data with one.

Annotations are experimental: the API and file formats may change between minor releases. Check that the project's DASCore has `dc.AnnotationSet` before starting, and read the installed docstrings of the methods you use. The [annotations tutorial](https://dascore.netlify.app/tutorial/annotations.html) explains the model and shows each method; `dascore doc tutorial/annotations` prints the installed version of it.

## Procedure

1. **Inspect the source without changing it.** Read the table as text (`pd.read_csv(path, dtype=str, keep_default_na=False)`) and report its columns, row count, and a few rows. Establish what each row is (a pick, a window, a path vertex, a polygon vertex), and the units and reference of each position column: absolute or relative times, time zone, meters or channel numbers. Ask the user when the table does not say.
2. **Identify the data the annotations describe.** Open the patch or spool and note its dimensions, coordinate units, and extent (`patch.dims`, `patch.get_coord("time")`). Annotations are stated in coordinate values. Convert sample or channel indices with the coordinate (`patch.get_array("distance")[index]`), and relative times by adding the start time, before building the set.
3. **Map columns onto the model.** A single position becomes a dimension column (`time`, `distance`); a range becomes `{dim}_min` and `{dim}_max`, half-open; rows that make up one thing share a `feature_id`; vertices of a path or polygon are ordered by `seq`. Keep every other column under its own name. Rename in a copy; never edit the user's file.
4. **Build the set.** With the patch at hand use `dc.AnnotationSet.from_patch(patch, table)`, which records its `acquisition_key` and `data_id`; otherwise `dc.AnnotationSet(table, dims=...)`. Read files with `dc.annotations(path, dims=...)`, not plain pandas, which turns `001` into `1` and `NA` into a missing value. Add paths and polygons with `add_path` and `add_polygon`.
5. **Check the result.**
    - The set has as many annotation rows as the source had rows, and the expected number of features (`len(set.features)`).
    - Every feature falls within the data: `set.overlapping(patch)` keeps all of them, or the user knows why some fall outside. `set.bounds()` shows each feature's extent.
    - Spot-check a few rows against the source, including any converted times or channels.
    - Plot the set over the data (`ax = patch.viz.waterfall(); set.viz.plot(ax=ax, color=...)`), save the figure, and show it to the user.
6. **Save and verify the round trip.** A set of picks alone can be a CSV (`set.io.to_csv(path)`), which does not record the dimensions or `data_id`; Parquet (`set.io.to_parquet(path)`, needs pyarrow) keeps them. A set with features or bases is a directory: `set.io.save(path)`. Read the file back with `dc.annotations` and compare it with the set.
7. **Report** the file written, the row and feature counts, every assumption made (units, time zone, index conversion), and any rows that were dropped or changed, with why.

## Pitfalls

- A row's `_min`/`_max` range is half-open, a `select` tuple includes both ends, and `overlapping` given a patch includes the patch's last sample. Say which rule applies when a boundary matters.
- A row must state at least one dimension; reading refuses rows that state none.
- A `feature_id` that names no existing feature creates a group, so a typo creates a new feature rather than an error. Compare the feature ids with what the user expects.
- `Spool.cut` needs a pad, e.g. `time=("-1s", "1s")`, for features stated at a single value.
