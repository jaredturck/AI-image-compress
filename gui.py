import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from infer import compress_image, decompress_image, load_model


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

        default_name = Path(input_path).stem + ".nic"
        output_path = filedialog.asksaveasfilename(defaultextension=".nic", initialfile=default_name, filetypes=[("Neural image", "*.nic")])
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

        default_name = Path(input_path).stem + "_decoded.png"
        output_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile=default_name, filetypes=[("PNG image", "*.png")])
        if not output_path:
            return

        self.status_var.set("Decompressing...")
        self.root.update_idletasks()
        stats = decompress_image(input_path, output_path, self.model, self.model_id, self.device)
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
