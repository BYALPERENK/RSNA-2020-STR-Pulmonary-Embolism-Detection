"""Tests for the DICOM to Hounsfield unit conversion in src/pe/data/dicom.py."""
import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, JPEGLosslessSV1, generate_uid

from pe.data.dicom import HU_MAX, HU_MIN, orient, read_hu, to_hu, to_square

AXIAL = [1, 0, 0, 0, 1, 0]


def to_jpeg_lossless(src, dst):
    """A copy of `src` as JPEG lossless, the syntax of 6.6% of the train files (01, 1.2). pydicom cannot encode it, GDCM can."""
    import gdcm
    reader = gdcm.ImageReader()
    reader.SetFileName(str(src))
    assert reader.Read()
    change = gdcm.ImageChangeTransferSyntax()
    change.SetTransferSyntax(gdcm.TransferSyntax(gdcm.TransferSyntax.JPEGLosslessProcess14_1))
    change.SetInput(reader.GetImage())
    assert change.Change()
    writer = gdcm.ImageWriter()
    writer.SetFileName(str(dst))
    writer.SetFile(reader.GetFile())
    writer.SetImage(change.GetOutput())
    assert writer.Write()
    return dst


def write_dicom(path, pixels, signed=True, intercept=-1024, orientation=AXIAL, pixel_data=None):
    """A minimal CT file with the given stored values; `pixel_data` replaces the pixel bytes."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID, ds.SOPInstanceUID = meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns = pixels.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
    ds.PixelRepresentation = int(signed)
    ds.RescaleIntercept, ds.RescaleSlope = str(intercept), "1"
    ds.ImageOrientationPatient = [str(v) for v in orientation]
    ds.PixelData = pixels.astype(np.int16 if signed else np.uint16).tobytes() if pixel_data is None else pixel_data
    ds.save_as(path, enforce_file_format=True)
    return path


def test_values_are_cast_before_the_rescale():
    stored = np.array([0, 24, 4095], dtype=np.uint16)
    assert to_hu(stored, 1, -1024).tolist() == [-1024, -1000, 3071]


def test_to_hu_does_not_change_its_input():
    stored = np.array([3096.0, 40.0], dtype=np.float32)
    to_hu(stored, 1, -1024)
    assert stored.tolist() == [3096, 40]


@pytest.mark.parametrize("orientation, expected", [(AXIAL, [[1, 2], [3, 4]]), ([-1, 0, 0, 0, -1, 0], [[4, 3], [2, 1]]),
                                                   ([-1, 0, 0, 0, 1, 0], [[2, 1], [4, 3]]), ([1, 0, 0, 0, -1, 0], [[3, 4], [1, 2]])])
def test_orient_flips_reversed_rows_and_columns(orientation, expected):
    assert orient(np.array([[1, 2], [3, 4]]), orientation).tolist() == expected


def test_a_tilted_image_is_not_flipped():
    hu = np.array([[1, 2], [3, 4]])
    assert orient(hu, [0.9984, 0.0557, -0.0031, -0.0558, 0.9969, -0.0548]).tolist() == hu.tolist()


def test_non_square_image_is_padded_with_air_in_the_centre():
    square = to_square(np.full((4, 2), 100, dtype=np.float32), size=4)
    assert square.shape == (4, 4)
    assert (square[:, [0, 3]] == HU_MIN).all() and (square[:, 1:3] == 100).all()


def test_512_image_is_not_resized():
    hu = np.arange(512 * 512, dtype=np.float32).reshape(512, 512)
    assert np.array_equal(to_square(hu), hu)


def test_read_hu_clips_the_padding_and_the_metal(tmp_path):
    stored = np.array([[-3024, -1000], [40, 10466]], dtype=np.int16)  # the padding and the metal of the intercept-0 group (01, 6.1)
    hu = read_hu(write_dicom(tmp_path / "a.dcm", stored, intercept=0), size=2)
    assert hu.dtype == np.float32 and hu.tolist() == [[HU_MIN, -1000], [40, HU_MAX]]


def test_read_hu_flips_a_reversed_file(tmp_path):
    stored = np.array([[0, 1024], [2048, 3072]], dtype=np.uint16)
    hu = read_hu(write_dicom(tmp_path / "b.dcm", stored, signed=False, orientation=[-1, 0, 0, 0, -1, 0]), size=2)
    assert hu.tolist() == [[2048, 1024], [0, -1024]]


def test_jpeg_lossless_gives_the_same_hu(tmp_path):
    rng = np.random.default_rng(0)
    stored = rng.integers(0, 4096, size=(64, 64)).astype(np.uint16)
    path = write_dicom(tmp_path / "c.dcm", stored, signed=False)
    jpeg = to_jpeg_lossless(path, tmp_path / "d.dcm")
    assert pydicom.dcmread(jpeg).file_meta.TransferSyntaxUID == JPEGLosslessSV1
    assert np.array_equal(read_hu(path, size=64), read_hu(jpeg, size=64))


def test_read_hu_raises_on_missing_pixel_data(tmp_path):
    stored = np.zeros((64, 64), dtype=np.int16)
    path = write_dicom(tmp_path / "e.dcm", stored, pixel_data=stored.tobytes()[:3000])
    with pytest.raises(ValueError):
        read_hu(path)
