import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision.io import decode_image, write_png

from codec import read_file, write_file
from model import ImageCodec


def checkpoint_id(checkpoint_path):
    digest = hashlib.sha256()
    with Path(checkpoint_path).open("rb") as file:
        while True:
            block = file.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.digest()[:16]


def pad_image(image):
    height, width = image.shape[-2:]
    pad_height = (-height) % 64
    pad_width = (-width) % 64
    return F.pad(image, (0, pad_width, 0, pad_height), mode="replicate")


def load_model(checkpoint_path, device=None):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ImageCodec()
    model.load_state_dict(checkpoint["model"])
    model.update(force=True)
    model.eval().to(device)
    return model, checkpoint_id(checkpoint_path), device


def compress_image(input_path, output_path, model, model_id, device):
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = decode_image(str(input_path), mode="RGB").float().div(255.0)
    height, width = image.shape[-2:]
    image = pad_image(image.unsqueeze(0).to(device))

    packed = model.compress(image)
    y_stream = packed["strings"][0][0]
    z_stream = packed["strings"][1][0]
    write_file(output_path, model_id, width, height, packed["shape"], y_stream, z_stream)

    file_size = output_path.stat().st_size
    return {
        "input_size": input_path.stat().st_size,
        "compressed_size": file_size,
        "bpp": file_size * 8.0 / (width * height),
        "width": width,
        "height": height,
    }


def decompress_image(input_path, output_path, model, model_id, device):
    input_path = Path(input_path)
    output_path = Path(output_path)
    packed = read_file(input_path)
    if packed is None:
        return {"error": "Not a valid NIC2 compressed image."}
    if packed["model_id"] != model_id:
        return {"error": "This .nic file was created with a different checkpoint."}

    strings = [[packed["y_stream"]], [packed["z_stream"]]]
    reconstruction = model.decompress(strings, packed["shape"])
    reconstruction = reconstruction[0, :, :packed["height"], :packed["width"]]
    reconstruction = reconstruction.mul(255.0).round().to(torch.uint8).cpu()
    write_png(reconstruction, str(output_path))

    return {
        "compressed_size": packed["file_size"],
        "output_size": output_path.stat().st_size,
        "width": packed["width"],
        "height": packed["height"],
    }


if __name__ == "__main__":
    print("Use gui.py for interactive compression/decompression, or import compress_image/decompress_image from infer.py.")
