---
name: dascore
description: >
  Python library for distributed fiber optic sensing (DAS/DTS/DSS). Use when
  reading, processing, transforming, or visualizing fiber-optic sensing data
  (e.g. HDF5/TDMS/SEGY DAS files from Terra15, Silixa, OptoDAS, Febus, etc.),
  or when converting such data between formats or to ObsPy/xarray objects.
license: LGPL-3.0-or-later
compatibility: Requires Python >=3.10.
---

# DASCore

DASCore reads, processes, and visualizes distributed acoustic sensing (DAS)
data. The two central types are:

- **`Patch`** — an n-D array (usually time × distance) with coordinates and
  metadata. Immutable: every method returns a *new* patch.
- **Spool** (`dc.spool(...)`) — a collection of patches, backed by memory, a
  single file, or a directory of files. Iterate it to get patches.

## Install

```bash
pip install dascore        # or: conda install -c conda-forge dascore
```

## Core workflow

```python
import dascore as dc

spool = dc.spool("path/to/file_or_directory")   # lazy; indexes directories
spool = spool.select(time=("2023-01-01", ...))  # filter before loading
spool = spool.chunk(time=60)                    # re-chunk to 60 s patches
for patch in spool:
    out = (
        patch.detrend("time")                  # most proc funcs are methods
        .pass_filter(time=(1, 100))            # units in Hz for time dim
        .velocity_to_strain_rate()
    )
```

## Decision table

| Task | Use | Not |
|---|---|---|
| Read data files | `dc.spool(path)[0]` or iterate | `dc.read` (low-level) |
| Discover file metadata cheaply | `dc.scan(path)` / `dc.scan_to_df(path)` | reading whole files |
| Subset by time/distance values | `patch.select(time=(t1, t2))` | index math |
| Subset by sample index | `patch.select(time=(0, 100), samples=True)` | `iselect` shorthand also exists |
| Merge contiguous patches | `spool.chunk(time=None)` | manual `np.concatenate` |
| Fixed-length windows | `spool.chunk(time=30, overlap=5)` | manual slicing |
| Parallel processing | `spool.map(func, client=executor)` | multiprocessing on patches directly |
| Get example data | `dc.get_example_patch("random_das")` | downloading files in tests/docs |
| Save patches | `patch.io.write(path, "dasdae")` or `dc.write(...)` | pickle |
| Convert to ObsPy/xarray/pandas | `patch.io.to_obspy()` etc. | manual conversion |

## Gotchas

- **Patches are immutable.** `patch.pass_filter(...)` returns a new patch;
  the original is unchanged. Chain calls or reassign.
- **Time is numpy `datetime64`/`timedelta64`.** Use `dc.to_datetime64("2023-01-01")`
  and `dc.to_timedelta64(1.5)` to build values; select accepts strings,
  datetime64, and floats-as-seconds for relative offsets.
- **`...` (Ellipsis) means "open ended"** in ranges: `time=(start, ...)`.
- **Filter arguments are ranges keyed by dimension**:
  `patch.pass_filter(time=(1, 100))` band-passes 1–100 Hz along the time dim;
  `None` on one side makes it low/high pass.
- **Units**: many methods accept pint quantities, e.g.
  `patch.convert_units(distance="ft")`; `dc.get_quantity("10 m/s")` parses
  strings. Do not cache pint quantities across registry resets.
- **Directory spools need an index update** after files change:
  `spool.update()`.
- **`spool.select` on non-coordinate attrs** matches exact values or
  collections (e.g. `network={"das1", "das2"}`).
- **Processing functions live in `dascore.proc` but are attached to `Patch`
  as methods**; transforms live in `dascore.transform`; plots in
  `dascore.viz` (e.g. `patch.viz.waterfall()`).

## Capability boundaries

- DASCore handles single-experiment array data, not seismic network
  workflows — use ObsPy for station/inventory-based seismology.
- Supported formats and their read/scan/write capabilities are listed at
  https://dascore.org/supported_formats.html — writing is only supported
  for a few formats (DASDAE, and converters via ObsPy/xarray).
- For very large datasets, prefer `spool.select(...).chunk(...)` before
  loading; patches load lazily from directory spools.

## Resources

- Full documentation: https://dascore.org/
- LLM index: https://dascore.org/llms.txt (full: /llms-full.txt)
- Tutorials: https://dascore.org/tutorial/concepts.html
