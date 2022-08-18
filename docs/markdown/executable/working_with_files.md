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

DASCore contains two data strutures which are useful in managing directories of
fiber data.

## FileSpool
The file spool is used to retrieve data from a directory of dascore-readable files.
It has the same interface as other spools mentioned in the
[quickstart](./quickstart.md).

```{code-cell}
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
