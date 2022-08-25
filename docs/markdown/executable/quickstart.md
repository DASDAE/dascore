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

DASCore has two main data structures; the Patch and the Spool.

## Patch
A section of contiguous, uniformly sampled data is called a Patch. These can be generated in a few ways:


### 1. Load an example patch (for simple demonstrations)

```{code-cell}
import dascore as dc
pa = dc.get_example_patch()
```

### 2. Load a file

We first download an example fiber file (you need an internet connection).
Next, we simply read it into a [spool](#spool) object then get the first (and only) patch.

```{code-cell}
# get a fiber file
import dascore as dc
from dascore.utils.downloader import fetch
path = fetch("terra15_das_1_trimmed.hdf5")  # path to a datafile

pa = dc.read(path)[0]
```

### 3. Create from Arrays

Patches can also be created from numpy arrays and dictionaries. This requires:

- The data array
- The coordinates for labeling each axis
- The attributes (optional)


```{code-cell}
import numpy as np

import dascore as dc
from dascore.utils.time import to_timedelta64


# Create the patch data
array = np.random.random(size=(300, 2_000))

# Create attributes, or metadata
t1 = np.datetime64("2017-09-18")
attrs = dict(
    d_distance=1,
    d_time=to_timedelta64(1 / 250),
    category="DAS",
    id="test_data1",
    time_min=t1,
)

# Create coordinates, labels for each axis in the array.
coords = dict(
    distance=np.arange(array.shape[0]) * attrs["d_distance"],
    time=np.arange(array.shape[1]) * attrs["d_time"],
)
pa = dc.Patch(data=array, coords=coords, attrs=attrs)
print(pa)
```

## Processing
For various reasons, Patches should be treated as *immutable*, meaning they should
not be modified in place, but rather new patches created when something needs to be
modified.

The patch has several methods which are intended to be chained together via a
[fluent interface](https://en.wikipedia.org/wiki/Fluent_interface), meaning each
method returns a new `Patch` instance.

```{code-cell}
import dascore as dc
pa = dc.get_example_patch()

out = (
    pa.decimate(time=8)  # decimate to reduce data volume by 8 along time dimension
    .detrend(dim='distance')  # detrend along distance dimension
    .pass_filter(time=(None, 10))  # apply a low-pass 10 Hz butterworth filter
)
```
The processing methods are located in the dascore.proc package.

## Visualization

DASCore provides various visualization functions found in the `dascore.viz`
package or using the `Patch.viz` namespace.

```{code-cell}
import dascore as dc
pa = dc.get_example_patch()
pa.viz.waterfall(show=True)
```

## Modifying Patches

Because patches should be treated as immutable objects, you can't just modify
them with normal item assignment. There are a few methods that return new
patches with modifications, however, that are functionally the same.

### new

Often you may wish to modify one aspect of the patch. `Patch.new` is designed
for this purpose:

```{code-cell}
import dascore as dc
pa = dc.get_example_patch()

# create a copy of patch with new data but coords and attrs stay the same
new = pa.new(data=pa.data * 10)
```

### update attrs
`Patch.update_attrs()` is for making small changes to the patch attrs (metadata) while
keeping the unaffected metadata (`Patch.new` would require you replace the entirety of attrs).

```{code-cell}
import dascore as dc
pa = dc.get_example_patch()

# update existing attribute 'network' and create new attr 'new_attr'
pa1 = pa.update_attrs(**{'network': 'experiment_1', 'new_attr': 42})
```

`Patch.update_attrs` also tries to keep the patch attributes consistent.
For example, changing the start, end, or sampling of a dimension should
update the other attributes affected by the change.

```{code-cell}
import dascore as dc
pa = dc.get_example_patch()

# update start time should also shift endtime
pa1 = pa.update_attrs(time_min='2000-01-01')
print(pa.attrs['time_min'])
print(pa1.attrs['time_min'])
```

## Spool

Spools are containers/managers of patches. These come in a few varieties which
can manage a group of patches loaded into memory, archives of local files,
and (in the future) a variety of clients for accessing remote resources.

The simplest way to get the appropriate spool for a specified input is to use
the `load` method, which should just work in the vast majority of cases.


```{code-cell}
import dascore as dc

# create a list of patches
patch_list = [dc.get_example_patch()]

# get a spool for managing in-memory patches
spool1 = dc.spool(patch_list)

# get a spool from a das file
from dascore.utils.downloader import fetch
path_to_das_file = fetch("terra15_das_1_trimmed.hdf5")
spool2 = dc.spool(path_to_das_file)

# get a spool from a directory of DAS files
directory_path = path_to_das_file.parent
spool3 = dc.spool(directory_path)
```

Despite some implementation differences, all spools have common behavior/methods.

### Accessing patches

Patches are extracted from the spool via simple iteration or indexing. New
spools are returned via slicing.

```{code-cell}
import dascore as dc
spool = dc.get_example_spool()

patch = spool[0]  # extract first patch

# iterate patchs
for patch in spool:
    ...

# slice spool to create new spool which excludes first patch.
new_spool = spool[1:]
```


### get_contents

Returns a dataframe listing contents. This method may not be supported on all
spools, especially those interfacing with vast remote resources.

```{code-cell}
import dascore as dc
spool = dc.get_example_spool()

# Return dataframe with contents of spool (each row has metadata of a patch)
print(spool.get_contents())
```

### select

Selects a subset of spool and returns a new spool. `get_contents` will now
reflect subset of the original data requested by the select operation.

```{code-cell}
import dascore as dc
spool = dc.get_example_spool()

# select a spool with
subspool = spool.select(time=('2020-01-03T00:00:09', None))
```

In addition to trimming the data along a specified dimension (as shown above),
select can be used to filter patches that meet a specified criteria.


```{code-cell}
import dascore as dc
# load a spool which has many diverse patches
spool = dc.get_example_spool('diverse_das')

# Only include patches which are in network 'das2' or 'das3'
subspool = spool.select(network={'das2', 'das3'})

# only include spools which match some unix-style query on their tags.
subspool = spool.select(tag='some*')
```

### chunk

Chunk controls how data are grouped together in patches. It can be used to
merge contiguous patches together, specify size of patches for processing,
overlap with previous segments, etc.

```{code-cell}
import dascore as dc
spool = dc.get_example_spool()

# chunk spool for 3 second increments with 1 second overlaps
# and keep any segements that don't have full 3600 seconds
subspool = spool.chunk(time=3, overlap=1, keep_partial=True)

# merge all contiguous segments along time dimension
merged_spool = spool.chunk(time=None)
```
