''' Compress and decompress images with a trained codec. '''

import hashlib, struct, tkinter as tk
from pathlib import Path
from tkinter import filedialog

import torch, torch.nn.functional as F
from torchvision.io import decode_image, write_png

from model import ImageCodec

MAGIC = b'NIC3'
HEADER = struct.Struct('<4s16sIII')

def read_file(path):
    ''' Read and validate a compressed NIC image file. '''
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        return None

    magic, model_id, width, height, latent_size = HEADER.unpack_from(data)
    if magic != MAGIC or latent_size > len(data) - HEADER.size:
        return None

    offset = HEADER.size
    return {
        'model_id': model_id,
        'width': width,
        'height': height,
        'latent_stream': data[offset:offset + latent_size],
        'hyper_stream': data[offset + latent_size:],
    }

def load_model(checkpoint_path):
    ''' Load a trained image codec checkpoint. '''
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    model = ImageCodec()
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
    model.update(force=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    with checkpoint_path.open('rb') as file:
        model_id = hashlib.file_digest(file, 'sha256').digest()[:16]

    return model, model_id

def compress_image(input_path, output_path, model, model_id):
    ''' Compress an image into the NIC file format. '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    image = decode_image(str(input_path), mode='RGB').float().div(255.0)
    height, width = image.shape[-2:]
    pad_height = (-height) % 64
    pad_width = (-width) % 64
    device = next(model.parameters()).device
    image = F.pad(image.unsqueeze(0).to(device), (0, pad_width, 0, pad_height), mode='replicate')

    latent_stream, hyper_stream = model.compress(image)
    header = HEADER.pack(MAGIC, model_id, width, height, len(latent_stream))
    output_path.write_bytes(header + latent_stream + hyper_stream)

    input_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    return {
        'input_size': input_size,
        'compressed_size': compressed_size,
        'bpp': compressed_size * 8.0 / (width * height),
    }

def decompress_image(input_path, output_path, model, model_id):
    ''' Decompress a NIC file into a PNG image. '''
    output_path = Path(output_path)
    packed = read_file(input_path)
    if packed is None:
        return {'error': 'Not a valid NIC3 compressed image.'}
    if packed['model_id'] != model_id:
        return {'error': 'This .nic file was created with a different checkpoint.'}

    hyper_shape = ((packed['height'] + 63) // 64, (packed['width'] + 63) // 64)
    reconstruction = model.decompress(packed['latent_stream'], packed['hyper_stream'], hyper_shape)
    reconstruction = reconstruction[0, :, :packed['height'], :packed['width']]
    reconstruction = reconstruction.mul(255.0).round().to(torch.uint8).cpu()
    write_png(reconstruction, str(output_path))
    return {'width': packed['width'], 'height': packed['height']}

class CodecGUI:
    ''' Provide a small desktop interface for compression and decompression. '''

    def __init__(self, root):
        ''' Build the codec interface. '''
        self.root = root
        self.root.title('Neural Image Codec')
        self.root.geometry('620x300')
        self.root.resizable(False, False)

        self.model = None
        self.model_id = None
        self.checkpoint_var = tk.StringVar(value='checkpoints/latest.pt')
        self.status_var = tk.StringVar(value='Load a trained checkpoint first.')

        tk.Label(root, text='Checkpoint').pack(anchor='w', padx=18, pady=(18, 4))
        checkpoint_row = tk.Frame(root)
        checkpoint_row.pack(fill='x', padx=18)
        tk.Entry(checkpoint_row, textvariable=self.checkpoint_var).pack(side='left', fill='x', expand=True)
        tk.Button(checkpoint_row, text='Browse', command=self.browse_checkpoint).pack(side='left', padx=(8, 0))
        tk.Button(checkpoint_row, text='Load', command=self.load_checkpoint).pack(side='left', padx=(8, 0))

        action_row = tk.Frame(root)
        action_row.pack(pady=28)
        self.compress_button = tk.Button(action_row, text='Compress image', width=20, state='disabled', command=self.compress)
        self.compress_button.pack(side='left', padx=8)
        self.decompress_button = tk.Button(action_row, text='Decompress .nic', width=20, state='disabled', command=self.decompress)
        self.decompress_button.pack(side='left', padx=8)

        tk.Label(root, textvariable=self.status_var, justify='left', wraplength=580).pack(anchor='w', padx=18, pady=8)

    def browse_checkpoint(self):
        ''' Select a checkpoint file. '''
        path = filedialog.askopenfilename(filetypes=[('PyTorch checkpoint', '*.pt'), ('All files', '*')])
        if path:
            self.checkpoint_var.set(path)

    def load_checkpoint(self):
        ''' Load the selected checkpoint. '''
        loaded = load_model(self.checkpoint_var.get())
        if loaded is None:
            self.status_var.set('Checkpoint file not found.')
            return

        self.model, self.model_id = loaded
        self.compress_button.config(state='normal')
        self.decompress_button.config(state='normal')
        self.status_var.set(f'Loaded checkpoint on {next(self.model.parameters()).device}.')

    def compress(self):
        ''' Choose an image and compress it. '''
        input_path = filedialog.askopenfilename(filetypes=[('Images', '*.jpg *.png'), ('All files', '*')])
        if not input_path:
            return

        output_path = filedialog.asksaveasfilename(defaultextension='.nic', initialfile=Path(input_path).stem + '.nic',
            filetypes=[('Neural image', '*.nic')])
        if not output_path:
            return

        self.status_var.set('Compressing...')
        self.root.update_idletasks()
        stats = compress_image(input_path, output_path, self.model, self.model_id)
        ratio = stats['input_size'] / max(stats['compressed_size'], 1)
        self.status_var.set(f'Compressed to {stats["compressed_size"]:,} bytes ({stats["bpp"]:.3f} bpp). '
            f'Source-file ratio: {ratio:.2f}x.\n{output_path}')

    def decompress(self):
        ''' Choose a NIC file and decompress it. '''
        input_path = filedialog.askopenfilename(filetypes=[('Neural image', '*.nic'), ('All files', '*')])
        if not input_path:
            return

        output_path = filedialog.asksaveasfilename(defaultextension='.png', initialfile=Path(input_path).stem + '_decoded.png',
            filetypes=[('PNG image', '*.png')])
        if not output_path:
            return

        self.status_var.set('Decompressing...')
        self.root.update_idletasks()
        stats = decompress_image(input_path, output_path, self.model, self.model_id)
        if 'error' in stats:
            self.status_var.set(stats['error'])
            return

        self.status_var.set(f'Decompressed {stats["width"]}×{stats["height"]} image.\n{output_path}')

def main():
    ''' Start the desktop inference interface. '''
    root = tk.Tk()
    CodecGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
