"""Synthetic PUNIU fixtures for reader and common IO tests."""

import h5py
import numpy as np
import pytest

import dascore as dc
from dascore.io.puniu import PuniuV1


def make_file(tmp_path, shape=(12, 4), order="time, channel", **updates):
    """Create a deterministic PUNIU file from the observed v1 schema."""
    p = tmp_path / "synthetic.h5"
    data = np.arange(np.prod(shape), dtype="float32").reshape(shape)
    nc = shape[order.split(", ").index("channel")]
    attrs = dict(
        version=1,
        format="differential",
        row_major_order=order,
        column_major_order=", ".join(order.split(", ")[::-1]),
        channel_count=nc,
        start_channel="306",
        step="5",
        end_channel=str(306 + nc * 5 - 1),
        epoch=1722470616,
        ns=552952169,
        sampling_rate=1000,
        spatial_resolution=1.6,
        spatial_sampling_rate=0.2,
    )
    attrs.update(updates)
    with h5py.File(p, "w") as f:
        ds = f.create_dataset("default", data=data)
        ds.attrs.update(attrs)
    return p, data


@pytest.fixture
def puniu_path(tmp_path):
    """Return a synthetic file with sufficient extent for interval tests."""
    return make_file(tmp_path, shape=(128, 32))[0]


@pytest.fixture
def io_path_tuple(puniu_path):
    """Bind the PUNIU reader to the common IO tests."""
    return PuniuV1(), puniu_path


@pytest.fixture
def data_file_path(puniu_path):
    """Provide the synthetic file to common scanning tests."""
    return puniu_path


@pytest.fixture
def read_spool(puniu_path):
    """Read the synthetic file through the public API."""
    return dc.read(puniu_path)


@pytest.fixture
def scanned_attrs(puniu_path):
    """Scan the synthetic file through the public API."""
    return dc.scan(puniu_path)
