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

# Working with Files

DASCore contains two data structures which are useful for working with directories of
fiber data.

## FileSpool
The file spool is used to retrieve data from a directory of dascore-readable files.
It has the same interface as other spools mentioned in the
[quickstart](./quickstart.md).

For example:

```python
import dascore
from dascore import examples as ex

# Get a directory with several files
diverse_spool = dascore.get_example_spool('diverse_das')
path = ex.spool_to_directory(diverse_spool)

# Create a spool for interacting with the files in the directory.
spool = (
    dascore.get_spool(path)
    .select(network='das2')  # sub-select das2 network
    .select(time=(None, '2022-01-01'))  # unselect anything after 2022
    .chunk(time=2, overlap=0.5)  # change the chunking of the patches
)

# Iterate each patch and do something with it
for patch in spool:
    ...
```

## Directory Indexer
The 'DirectoryIndexer' is used to track the contents of a directory which
contains fiber data. It creates a small, hidden HDF index file at the top
of the directory which can be efficiently queried for retrieving data
(it is used internally by the `DirectorySpool`) or to ask questions about
the data archive.


```python
import dascore
from dascore.io.indexer import DirectoryIndexer
from dascore import examples as ex

# Get a directory with several files
diverse_spool = dascore.get_example_spool('diverse_das')
path = ex.spool_to_directory(diverse_spool)

# Create an indexer and update the index. This will include any new files
# with timestamps newer than the last update, or create a new HDF index file
# if one does not yet exist.
indexer = DirectoryIndexer(path).update()

# get the contents of the directory's files
df = indexer.get_contents()

# This dataframe can be used to ascertain data availability, detect gaps, etc.
```
