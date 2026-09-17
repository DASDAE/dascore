"""Read XDAS NetCDF arrays and collections.

The ``xdas`` reader returns one Patch per signal, including signals in nested
collection groups. Both legacy and current linear coordinate interpolation,
sampled segments, and explicit coordinates are supported. Scanning does not
load signal samples; selections and ``read_array`` slice the stored arrays.

Use ``dc.read(path, file_format="xdas")`` to explicitly select this reader.
Files with XDAS coordinate metadata or collections are detected automatically;
plain single-array CF files can also be read by the generic NetCDF reader.
Install ``xarray`` and ``h5netcdf`` to read, and ``hdf5plugin`` for filters such
as ZFP. Writing XDAS files and XDAS tile manifests are not supported.
"""
