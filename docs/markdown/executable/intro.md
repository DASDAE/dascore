---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
# Quickstart

A quickstart for DASCore, a python library for fiber-optic sensing.

## Patch
A section of contiguous (or nearly so) fiber data is called a Patch. These can be generated in a few ways:


### 1. Load an example patch (for simple demonstrations)

```python
import dascore
pa = dascore.get_example_patch()
```

### 2. Load a file

We first download an example fiber file (you need an internet connection).
Next, we simply read it into a [Stream](#Stream) object then get the first (and only) patch.

```python
# get a fiber file
import dascore
from dascore.utils.downloader import fetch
path = fetch("terra15_das_1_trimmed.hdf5")  # path to a datafile

pa = dascore.read(path)[0]
```

### 3. Create from Arrays

Patches can also be created from numpy arrays and dictionaries. You need to specify:

- The data array
- The coordinates for labeling each axis
- The attributes (optional)


```python
import numpy as np

import dascore
from dascore.utils.time import to_timedelta64


array = np.random.random(size=(300, 2_000))
t1 = np.datetime64("2017-09-18")
attrs = dict(
    d_distance=1,
    d_time=to_timedelta64(1 / 250),
    category="DAS",
    id="test_data1",
    time_min=t1,
)

coords = dict(
    distance=np.arange(array.shape[0]) * attrs["d_distance"],
    time=np.arange(array.shape[1]) * attrs["d_time"],
)
pa = dascore.Patch(data=array, coords=coords, attrs=attrs)
print(pa)
```

## Processing
The patch has several methods which are intended to be chained together via a [fluent interface](https://en.wikipedia.org/wiki/Fluent_interface).

```python
import dascore
pa = dascore.get_example_patch()

out = (
    pa.decimate(8)  # decimate to reduce data volume by 8 along time dimension
    .detrend(dim='distance')  # detrend along distance dimension
    .pass_filter(time=(None, 10))  # apply a low-pass 10 Hz butterworth filter
)
```

## Visualization

```{code-cell}
import dascore
pa = dascore.get_example_patch()
pa.viz.waterfall(show=True)
```

## Modifying Patches

Patches should be treated an immutable, which means you can't just modify
them with normal item assignment. There are a few methods that return new
patches with modifications, however, that are functionally nearly the same.

### new

```python
import dascore
pa = dascore.get_example_patch()
# create a copy of patch with new data but coords and attrs stay the same
pa.new(data=pa.data * 10)
```

### update attrs

```python
import dascore
pa = dascore.get_example_patch()
# update existing attribute 'network' and create new attr 'new_attr'
pa1 = pa.update_attrs(**{'network': 'experiment_1', 'new_attr': 42})
```


# Spool

Spools are containers/managers of patches. These come in a few varieties
including `MemorySpool` for managing a group of patches loaded into memory,
`FileSpool` for managing archives of local files, and a variety of clients
for accessing remote resources. However, despite a few subtle differences,
Spools of any type have a few main methods:

## get_contents

If supported, returns a dataframe listing contents.

```{code-cell}
import dascore
spool = dascore.get_example_spool()
# entire contents of spool
print(spool.get_contents())
```

## select

Selects a subset of spool and returns a new spool. `get_contents` will now
reflect subset of the original data requested by the select operation.

```{code-cell}
import dascore
spool = dascore.get_example_spool()
# select a spool with
subspool = spool.select(time=('2020-01-03T00:00:09', None)
```

## chunk

Chunk controls how data are grouped together in patches. It can be used to
merge contiguous patches together, specify size of patches for processing,
etc.

```{code-cell}
import dascore
spool = dascore.get_example_spool()
# chunk spool for 3600 second increments with 10 second overlaps
# and keep any segements that are partials
subspool = spool.chunk(time=3600, overlap=10, keep_partials=True)

# merge all contiguous segments along time dimension
merged_spool = spool.chunk(time=None)
```

## iterate

Patches are extracted from the spool via simple iteration or indexing:

```{code-cell}
import dascore
spool = dascore.get_example_spool()
# chunk spool for 3600 second increments with 10 second overlaps
patch = spool[0]  # extract first patch
# iterate patchs
for patch in spool:
    print(patch)
# extract a range of patches into new spool
new_spool = spool[1:]
```
