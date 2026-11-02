"""Reading the DICOM files."""
import io
import os
import zlib

import cv2
import numpy as np
import pydicom
from pydicom.multival import MultiValue

# The UIDs in this data are shortened hashes, not valid DICOM UIDs; without this pydicom warns on every file.
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

# Below -1024 is padding outside the scanned circle, above 3071 metal; clipping makes the padding the same in every group (01, 6.3).
HU_MIN, HU_MAX = -1024, 3071
IMAGE_SIZE = 512


def as_text(value) -> str:
    """A header value as text; multi-valued fields are joined with a backslash, as stored in DICOM."""
    if isinstance(value, (MultiValue, list)):
        return "\\".join(str(v) for v in value)
    return str(value)


def read_header(path: str, crc: bool = False) -> dict:
    """Every header field of one file as text (file meta included, pixels not decoded), plus the file size.

    With crc=True the whole file is read once and its CRC-32 is added, to compare with the zip it came from.
    A file that cannot be read gets a `read_error` instead of its fields, so no file is lost silently.
    """
    row = {"file_size": os.path.getsize(path)}
    try:
        if crc:
            with open(path, "rb") as f:
                data = f.read()
            row["crc32"] = zlib.crc32(data)
            ds = pydicom.dcmread(io.BytesIO(data), stop_before_pixels=True)
        else:
            ds = pydicom.dcmread(path, stop_before_pixels=True)
    except Exception as e:
        row["read_error"] = repr(e)
        return row
    for elem in [*ds.file_meta, *ds]:
        if elem.keyword:
            row[elem.keyword] = as_text(elem.value)
    return row


def to_hu(stored: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    """Stored pixel values to Hounsfield units, as float32. The values are cast first: uint16 plus a negative intercept would wrap around."""
    return np.array(stored, dtype=np.float32) * np.float32(slope) + np.float32(intercept)


def orient(hu: np.ndarray, orientation) -> np.ndarray:
    """Flip an image whose rows run towards the patient's right or whose columns run towards the front, so that every
    image is seen from the feet with the patient's right on the left (`orientation`: ImageOrientationPatient, 6 numbers)."""
    if orientation[0] < -0.9:
        hu = hu[:, ::-1]
    if orientation[4] < -0.9:
        hu = hu[::-1, :]
    return np.ascontiguousarray(hu)


def to_square(hu: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Pad a non-square image with air (HU_MIN) to a centred square, then resize it to size x size. Every image of the
    data is already 512 x 512 (01, 3.5); this is for a hidden test file that is not."""
    rows, cols = hu.shape
    if rows != cols:
        side = max(rows, cols)
        top, left = (side - rows) // 2, (side - cols) // 2
        square = np.full((side, side), HU_MIN, dtype=hu.dtype)
        square[top:top + rows, left:left + cols] = hu
        hu = square
    if hu.shape[0] != size:
        interpolation = cv2.INTER_AREA if hu.shape[0] > size else cv2.INTER_LINEAR
        hu = cv2.resize(hu, (size, size), interpolation=interpolation)
    return hu


def read_hu(path, size: int = IMAGE_SIZE) -> np.ndarray:
    """One slice in Hounsfield units: float32, clipped to [HU_MIN, HU_MAX], flipped to the common orientation, size x size.

    Raises if the pixels cannot be read; a broken file never becomes an empty image.
    """
    ds = pydicom.dcmread(path)
    hu = to_hu(ds.pixel_array, float(ds.RescaleSlope), float(ds.RescaleIntercept))
    hu = orient(np.clip(hu, HU_MIN, HU_MAX), [float(v) for v in ds.ImageOrientationPatient])
    return to_square(hu, size)
