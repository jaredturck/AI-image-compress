import hashlib
import struct
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import torch
import torch.nn.functional as F
from torchvision.io import decode_image, write_png

from model import ImageCodec

MAGIC = b"NIC2"
VERSION = 2
HEADER = struct.Struct("<4sB16sIIIIII")


def write_file(path, model_id, width, height, shape, y_stream, z_stream):
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
    Path(path).write_bytes(header + y_stream + z_stream)


def read_file(path):
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        return None

    magic, version, model_id, width, height, z_width, z_height, y_size, z_size = HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        return None
    if len(data) != HEADER.size + y_size + z_size:
        return None

    offset = HEADER.size
    return {
        "model_id": model_id,
        "width": width,
        "height": height,
        "shape": (z_height, z_width),
        "y_stream": data[offset:offset + y_size],
        "z_stream": data[offset + y_size:],
        "file_size": len(data),
    }


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


def decompress_image(input_path, output_path, model, model_id):
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


class CodecGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Image Codec")
        self.root.geometry("620x300")
        self.root.resizable(False, False)

        self.model = None
        self.model_id = None
        self.device = None
        self.checkpoint_var = tk.StringVar(value="checkpoints/latest.pt")
        self.status_var = tk.StringVar(value="Load a trained checkpoint first.")

        tk.Label(root, text="Checkpoint").pack(anchor="w", padx=18, pady=(18, 4))
        checkpoint_row = tk.Frame(root)
        checkpoint_row.pack(fill="x", padx=18)
        tk.Entry(checkpoint_row, textvariable=self.checkpoint_var).pack(side="left", fill="x", expand=True)
        tk.Button(checkpoint_row, text="Browse", command=self.browse_checkpoint).pack(side="left", padx=(8, 0))
        tk.Button(checkpoint_row, text="Load", command=self.load_checkpoint).pack(side="left", padx=(8, 0))

        action_row = tk.Frame(root)
        action_row.pack(pady=28)
        self.compress_button = tk.Button(action_row, text="Compress image", width=20, state="disabled", command=self.compress)
        self.compress_button.pack(side="left", padx=8)
        self.decompress_button = tk.Button(action_row, text="Decompress .nic", width=20, state="disabled", command=self.decompress)
        self.decompress_button.pack(side="left", padx=8)

        tk.Label(root, textvariable=self.status_var, justify="left", wraplength=580).pack(anchor="w", padx=18, pady=8)

    def browse_checkpoint(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pt"), ("All files", "*")])
        if path:
            self.checkpoint_var.set(path)

    def load_checkpoint(self):
        loaded = load_model(self.checkpoint_var.get())
        if loaded is None:
            self.status_var.set("Checkpoint file not found.")
            return

        self.model, self.model_id, self.device = loaded
        self.compress_button.config(state="normal")
        self.decompress_button.config(state="normal")
        self.status_var.set(f"Loaded checkpoint on {self.device}.")

    def compress(self):
        input_path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.webp"), ("All files", "*")]
        )
        if not input_path:
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".nic",
            initialfile=Path(input_path).stem + ".nic",
            filetypes=[("Neural image", "*.nic")],
        )
        if not output_path:
            return

        self.status_var.set("Compressing...")
        self.root.update_idletasks()
        stats = compress_image(input_path, output_path, self.model, self.model_id, self.device)
        ratio = stats["input_size"] / max(stats["compressed_size"], 1)
        self.status_var.set(
            f"Compressed to {stats['compressed_size']:,} bytes ({stats['bpp']:.3f} bpp). "
            f"Source-file ratio: {ratio:.2f}x.\n{output_path}"
        )

    def decompress(self):
        input_path = filedialog.askopenfilename(filetypes=[("Neural image", "*.nic"), ("All files", "*")])
        if not input_path:
            return

        output_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            initialfile=Path(input_path).stem + "_decoded.png",
            filetypes=[("PNG image", "*.png")],
        )
        if not output_path:
            return

        self.status_var.set("Decompressing...")
        self.root.update_idletasks()
        stats = decompress_image(input_path, output_path, self.model, self.model_id)
        if "error" in stats:
            self.status_var.set(stats["error"])
            return

        self.status_var.set(f"Decompressed {stats['width']}×{stats['height']} image.\n{output_path}")


def main():
    root = tk.Tk()
    CodecGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
