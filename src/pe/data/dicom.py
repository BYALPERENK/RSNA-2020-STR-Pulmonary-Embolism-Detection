"""Reading the DICOM files."""
import io
import os
import zlib

import pydicom
from pydicom.multival import MultiValue

# The UIDs in this data are shortened hashes, not valid DICOM UIDs; without this pydicom warns on every file.
pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE


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
