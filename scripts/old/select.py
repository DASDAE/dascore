"""Old select method without xarray."""

def select(self, **kwargs):
    """
    Return a subset of the trace based on query parameters.

    Any dimension of the data can be passed as key, and the values
    should either be a Slice or a tuple of (min, max) for that
    dimension.

    The time dimension is handled specially in that either floats,
    datetime64 or datetime objects can be used to specify relative
    or absolute times, respectively.

    Examples
    --------
    >>> from fios.examples import get_example_trace
    >>> tr = get_example_trace()
    >>> new = tr.select(distance=(50,300))
    """
    if not len(kwargs):
        return self
    assert len(kwargs) <= 1, "only one dim supported for now"
    dim = list(kwargs)[0]
    vals = kwargs[dim]
    coord = self.coords[dim]
    start = vals[0] if vals[0] is not None else coord.min()
    stop = vals[1] if vals[1] is not None else coord.max()
    missing_dimension = set(kwargs) - set(self.coords)
    if missing_dimension:
        msg = f"Trace does to have dimension(s): {missing_dimension}"
        raise MissingDimensions(msg)
    index1 = np.searchsorted(coord, start, side="left")
    index2 = np.searchsorted(coord, stop, side="right")
    # create new coords and slice arrays
    coords = dict(self.coords)
    coords[dim] = coords[dim][slice(index1, index2)]
    # slice np array
    slices = [slice(None)] * len(self.data.shape)
    dim_ind = self.dims.index(dim)
    slices[dim_ind] = slice(index1, index2)
    data = self.data[tuple(slices)]

    return self.new(data=data, coords=coords)