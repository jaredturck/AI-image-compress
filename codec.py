import struct
from pathlib import Path

MAGIC = b"NIC2"
VERSION = 2
HEADER = struct.Struct("<4sB16sIIIIII")


def write_file(path, model_id, width, height, shape, y_stream, z_stream):
    path = Path(path)
    z_height, z_width = shape
    header = HEADER.pack(
        MAGIC,
        VERSION,
        model_id,
        width,
        height,
        z_width,
        z_height,
        len(y_stream),
        len(z_stream),
    )
    path.write_bytes(header + y_stream + z_stream)


def read_file(path):
    path = Path(path)
    data = path.read_bytes()
    if len(data) < HEADER.size:
        return None

    magic, version, model_id, width, height, z_width, z_height, y_size, z_size = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        return None
    if len(data) != HEADER.size + y_size + z_size:
        return None

    offset = HEADER.size
    y_stream = data[offset:offset + y_size]
    z_stream = data[offset + y_size:]
    return {
        "model_id": model_id,
        "width": width,
        "height": height,
        "shape": (z_height, z_width),
        "y_stream": y_stream,
        "z_stream": z_stream,
        "file_size": len(data),
    }
