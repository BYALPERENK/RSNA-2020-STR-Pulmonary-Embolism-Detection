"""Tests for the storage formats in src/pe/data/formats.py."""
import numpy as np
import pytest

from pe.data.dicom import HU_MAX, HU_MIN
from pe.data.formats import FORMATS, npy_encode, read_series_slices


@pytest.mark.parametrize("name", ["npy16", "png16", "jp2", "png8x3"])
def test_lossless_formats_give_back_the_stored_array(name):
    rng = np.random.default_rng(0)
    hu = rng.integers(HU_MIN, HU_MAX + 1, size=(512, 512)).astype(np.float32)
    f = FORMATS[name]
    a = f.to_array(hu)
    assert np.array_equal(f.decode(f.encode(a)), a)


def test_png16_keeps_every_hu_value():
    hu = np.arange(HU_MIN, HU_MAX + 1, dtype=np.float32).reshape(64, 64)
    f = FORMATS["png16"]
    back = f.decode(f.encode(f.to_array(hu))).astype(np.int32) + HU_MIN
    assert np.array_equal(back, hu)


def test_read_series_slices_reads_only_the_asked_slices(tmp_path):
    volume = np.arange(5 * 4 * 4, dtype=np.int16).reshape(5, 4, 4)
    path = tmp_path / "s.npy"
    path.write_bytes(npy_encode(volume))
    assert np.array_equal(read_series_slices(path, 1, 3), volume[1:4])
