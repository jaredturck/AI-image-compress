''' Compress and reconstruct images with a trained codec. '''

import hashlib, shutil, struct, subprocess, tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import torch, torch.nn.functional as F
from torchvision.io import decode_image
from torchvision.transforms.functional import to_pil_image

from model import ImageCodec

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / 'checkpoints'
COMPRESSED_DIR = PROJECT_DIR / 'compressed_images'
MAGIC = b'NIC3'
HEADER = struct.Struct('<4s16sIII')
PREVIEW_PADDING = 24
PREVIEW_RESIZE_DELAY = 50

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
    model.update(force=True, update_quantiles=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    with checkpoint_path.open('rb') as file:
        model_id = hashlib.file_digest(file, 'sha256').digest()[:16]

    return model, model_id, checkpoint_path

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
        'output_path': output_path,
    }

def decompress_image(input_path, model, model_id):
    ''' Reconstruct an image by reading a NIC file from disk. '''
    packed = read_file(input_path)
    if packed is None:
        return {'error': 'Not a valid NIC3 compressed image.'}
    if packed['model_id'] != model_id:
        return {'error': 'This .nic file was created with a different checkpoint.'}

    hyper_shape = ((packed['height'] + 63) // 64, (packed['width'] + 63) // 64)
    reconstruction = model.decompress(packed['latent_stream'], packed['hyper_stream'], hyper_shape)
    reconstruction = reconstruction[0, :, :packed['height'], :packed['width']]
    reconstruction = reconstruction.mul(255.0).round().to(torch.uint8).cpu()
    return {'width': packed['width'], 'height': packed['height'], 'image': reconstruction}

def choose_file(title, start_dir, kdialog_filter, filetypes):
    ''' Choose a file with KDialog when available and Tk as a fallback. '''
    kdialog = shutil.which('kdialog')
    if kdialog:
        try:
            result = subprocess.run([kdialog, '--getopenfilename', str(start_dir), kdialog_filter], stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True)

        except OSError:
            result = None

        if result is not None:
            if result.returncode == 0:
                return result.stdout.strip()
            return ''

    return filedialog.askopenfilename(title=title, initialdir=str(start_dir), filetypes=filetypes)

def format_file_size(size):
    ''' Format a byte count for the inference interface. '''
    if size >= 1024 ** 2:
        return f'{size / (1024 ** 2):.2f} MB'
    if size >= 1024:
        return f'{size / 1024:.1f} KB'
    return f'{size} bytes'

def make_preview(image, available_width, available_height):
    ''' Build a fitted CustomTkinter image without changing codec input. '''
    available_width = max(1, available_width - PREVIEW_PADDING)
    available_height = max(1, available_height - PREVIEW_PADDING)
    scale = min(available_width / image.width, available_height / image.height, 1.0)
    size = (max(1, int(image.width * scale)), max(1, int(image.height * scale)))
    preview = image.copy()
    preview.thumbnail(size)
    return ctk.CTkImage(light_image=preview, dark_image=preview, size=preview.size)

class CodecGUI:
    ''' Provide a dark desktop interface for compression and reconstruction. '''

    def __init__(self, root):
        ''' Build the codec interface. '''
        self.root = root
        self.root.title('Neural Image Codec')
        self.root.geometry('1280x820')
        self.root.minsize(1080, 720)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self.model = None
        self.model_id = None
        self.checkpoint_path = None
        self.source_path = None
        self.source_image = None
        self.source_preview = None
        self.reconstruction_image = None
        self.reconstruction_preview = None
        self.compressed_path = None
        self.preview_resize_job = None
        self.worker_pool = ThreadPoolExecutor(max_workers=1)
        self.worker_future = None
        self.worker_callback = None

        self.status_var = tk.StringVar(value='Loading the latest checkpoint...')
        self.source_name_var = tk.StringVar(value='No image selected')
        self.source_info_var = tk.StringVar(value='Open an image to begin.')
        self.output_info_var = tk.StringVar(value='Compress and decode an image to compare the reconstruction.')
        self.checkpoint_var = tk.StringVar(value='No checkpoint loaded')
        self.metrics_var = tk.StringVar(value='No compression statistics yet')

        self.build_header()
        self.build_panels()
        self.build_status_bar()
        self.update_action_states()
        self.root.after(100, self.load_latest_checkpoint)

    def build_header(self):
        ''' Build the application header and checkpoint controls. '''
        header = ctk.CTkFrame(self.root, corner_radius=0)
        header.grid(row=0, column=0, sticky='ew')
        header.grid_columnconfigure(0, weight=1)

        title_frame = ctk.CTkFrame(header, fg_color='transparent')
        title_frame.grid(row=0, column=0, sticky='w', padx=24, pady=16)
        ctk.CTkLabel(title_frame, text='Neural Image Codec', font=ctk.CTkFont(size=24, weight='bold')).pack(anchor='w')
        ctk.CTkLabel(title_frame, text='Compress, decode, and compare reconstruction quality.', text_color='#9c9c9c').pack(anchor='w')

        checkpoint_frame = ctk.CTkFrame(header, fg_color='transparent')
        checkpoint_frame.grid(row=0, column=1, sticky='e', padx=24, pady=16)
        ctk.CTkLabel(checkpoint_frame, text='Checkpoint', text_color='#9c9c9c').grid(row=0, column=0, sticky='e', padx=(0, 10))
        ctk.CTkLabel(checkpoint_frame, textvariable=self.checkpoint_var, width=220, anchor='e').grid(row=0, column=1, sticky='e', padx=(0, 10))
        self.checkpoint_button = ctk.CTkButton(checkpoint_frame, text='Change', width=90, command=self.browse_checkpoint)
        self.checkpoint_button.grid(row=0, column=2)

    def build_panels(self):
        ''' Build the side-by-side source and reconstruction panels. '''
        content = ctk.CTkFrame(self.root, fg_color='transparent')
        content.grid(row=1, column=0, sticky='nsew', padx=18, pady=(18, 10))
        content.grid_columnconfigure(0, weight=1, uniform='panel')
        content.grid_columnconfigure(1, weight=1, uniform='panel')
        content.grid_rowconfigure(0, weight=1)

        self.build_source_panel(content)
        self.build_output_panel(content)

    def build_source_panel(self, parent):
        ''' Build the source image panel. '''
        panel = ctk.CTkFrame(parent, corner_radius=16)
        panel.grid(row=0, column=0, sticky='nsew', padx=(0, 9))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(panel, fg_color='transparent')
        header.grid(row=0, column=0, sticky='ew', padx=18, pady=(18, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text='Source image', font=ctk.CTkFont(size=18, weight='bold')).grid(row=0, column=0, sticky='w')
        self.open_button = ctk.CTkButton(header, text='Open Image', width=110, command=self.open_image)
        self.open_button.grid(row=0, column=1, sticky='e')

        ctk.CTkLabel(panel, textvariable=self.source_name_var, anchor='w', font=ctk.CTkFont(size=15, weight='bold')).grid(
            row=1, column=0, sticky='ew', padx=18)
        ctk.CTkLabel(panel, textvariable=self.source_info_var, anchor='w', text_color='#9c9c9c').grid(
            row=2, column=0, sticky='ew', padx=18, pady=(2, 10))

        self.source_preview_frame = ctk.CTkFrame(panel, corner_radius=12, fg_color='#171717')
        self.source_preview_frame.grid(row=3, column=0, sticky='nsew', padx=18, pady=(0, 14))
        self.source_preview_frame.grid_columnconfigure(0, weight=1)
        self.source_preview_frame.grid_rowconfigure(0, weight=1)
        self.source_preview_frame.bind('<Configure>', self.schedule_preview_resize)
        self.source_preview_label = ctk.CTkLabel(self.source_preview_frame, text='No source image', text_color='#777777')
        self.source_preview_label.grid(row=0, column=0, sticky='nsew', padx=12, pady=12)

        self.compress_button = ctk.CTkButton(panel, text='Compress', height=42, state='disabled', command=self.compress)
        self.compress_button.grid(row=4, column=0, sticky='ew', padx=18, pady=(0, 18))

    def build_output_panel(self, parent):
        ''' Build the reconstructed image and compression statistics panel. '''
        panel = ctk.CTkFrame(parent, corner_radius=16)
        panel.grid(row=0, column=1, sticky='nsew', padx=(9, 0))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        ctk.CTkLabel(panel, text='Reconstruction', font=ctk.CTkFont(size=18, weight='bold')).grid(
            row=0, column=0, sticky='w', padx=18, pady=(18, 8))
        ctk.CTkLabel(panel, textvariable=self.output_info_var, anchor='w', text_color='#9c9c9c').grid(
            row=1, column=0, sticky='ew', padx=18)
        ctk.CTkLabel(panel, text='Decoded from the saved .nic file on disk.', anchor='w', text_color='#6f9fd8').grid(
            row=2, column=0, sticky='ew', padx=18, pady=(2, 10))

        self.output_preview_frame = ctk.CTkFrame(panel, corner_radius=12, fg_color='#171717')
        self.output_preview_frame.grid(row=3, column=0, sticky='nsew', padx=18, pady=(0, 12))
        self.output_preview_frame.grid_columnconfigure(0, weight=1)
        self.output_preview_frame.grid_rowconfigure(0, weight=1)
        self.output_preview_frame.bind('<Configure>', self.schedule_preview_resize)
        self.output_preview_label = ctk.CTkLabel(self.output_preview_frame, text='No reconstruction yet', text_color='#777777')
        self.output_preview_label.grid(row=0, column=0, sticky='nsew', padx=12, pady=12)

        metrics = ctk.CTkLabel(panel, textvariable=self.metrics_var, anchor='w', height=30, corner_radius=8,
            fg_color='#303030', text_color='#b5b5b5', font=ctk.CTkFont(size=12))
        metrics.grid(row=4, column=0, sticky='ew', padx=18, pady=(0, 10))

        self.decode_button = ctk.CTkButton(panel, text='Decode', height=42, state='disabled', command=self.decode)
        self.decode_button.grid(row=5, column=0, sticky='ew', padx=18, pady=(0, 18))

    def build_status_bar(self):
        ''' Build the status line at the bottom of the window. '''
        ctk.CTkLabel(self.root, textvariable=self.status_var, anchor='w', text_color='#9c9c9c').grid(
            row=2, column=0, sticky='ew', padx=24, pady=(0, 12))

    def load_latest_checkpoint(self):
        ''' Load the newest project checkpoint when one exists. '''
        checkpoints = list(CHECKPOINT_DIR.glob('checkpoint_*.pt'))
        if not checkpoints:
            self.status_var.set('No checkpoint found. Choose a checkpoint to enable compression.')
            self.update_action_states()
            return

        checkpoint_path = max(checkpoints, key=lambda path: path.stat().st_mtime)
        self.start_checkpoint_load(checkpoint_path)

    def browse_checkpoint(self):
        ''' Select and load a checkpoint file. '''
        default_dir = CHECKPOINT_DIR if CHECKPOINT_DIR.exists() else PROJECT_DIR
        start_dir = self.checkpoint_path.parent if self.checkpoint_path else default_dir
        path = choose_file('Choose checkpoint', start_dir, 'PyTorch checkpoint (*.pt)', [('PyTorch checkpoint', '*.pt'), ('All files', '*')])
        if path:
            self.start_checkpoint_load(Path(path))

    def start_checkpoint_load(self, checkpoint_path):
        ''' Load a checkpoint away from the GUI event loop. '''
        self.status_var.set(f'Loading {checkpoint_path.name}...')
        self.start_worker(load_model, self.finish_checkpoint_load, checkpoint_path)

    def finish_checkpoint_load(self, loaded):
        ''' Apply a checkpoint loaded by the background worker. '''
        if loaded is None:
            self.status_var.set('Checkpoint file not found.')
            return

        self.model, self.model_id, self.checkpoint_path = loaded
        self.checkpoint_var.set(self.checkpoint_path.name)
        self.clear_compressed_result()
        device = next(self.model.parameters()).device
        self.status_var.set(f'Loaded {self.checkpoint_path.name} on {device}.')

    def open_image(self):
        ''' Select an image and show the exact pixels used by inference. '''
        pictures_dir = Path.home() / 'Pictures'
        default_dir = pictures_dir if pictures_dir.exists() else Path.home()
        start_dir = self.source_path.parent if self.source_path else default_dir
        path = choose_file('Choose source image', start_dir, 'Images (*.png *.jpg *.jpeg)',
            [('Images', '*.png *.jpg *.jpeg'), ('All files', '*')])
        if not path:
            return

        try:
            image = decode_image(path, mode='RGB')

        except (OSError, RuntimeError) as error:
            self.status_var.set(f'Could not open image: {error}')
            return

        self.source_path = Path(path)
        self.source_image = to_pil_image(image)
        self.refresh_previews()
        self.source_name_var.set(self.source_path.name)
        self.source_info_var.set(f'{image.shape[2]} x {image.shape[1]} pixels | original dimensions preserved')
        self.clear_compressed_result()
        self.status_var.set('Image ready. Click Compress to create the .nic file.')
        self.update_action_states()

    def clear_compressed_result(self):
        ''' Clear output state that belongs to a previous source or checkpoint. '''
        self.compressed_path = None
        # CustomTkinter leaves the underlying Tk image unchanged when image=None.
        self.output_preview_label.configure(image='', text='No reconstruction yet')
        self.reconstruction_image = None
        self.reconstruction_preview = None
        self.output_info_var.set('Compress and decode an image to compare the reconstruction.')
        self.metrics_var.set('No compression statistics yet')
        self.update_action_states()

    def schedule_preview_resize(self, _event=None):
        ''' Refresh image previews after the preview area changes size. '''
        if self.preview_resize_job is not None:
            self.root.after_cancel(self.preview_resize_job)
        self.preview_resize_job = self.root.after(PREVIEW_RESIZE_DELAY, self.refresh_previews)

    def refresh_previews(self):
        ''' Fit complete source and reconstruction images inside their preview areas. '''
        self.preview_resize_job = None

        if self.source_image is not None:
            self.source_preview = make_preview(self.source_image, self.source_preview_frame.winfo_width(), self.source_preview_frame.winfo_height())
            self.source_preview_label.configure(image=self.source_preview, text='')

        if self.reconstruction_image is not None:
            self.reconstruction_preview = make_preview(self.reconstruction_image, self.output_preview_frame.winfo_width(),
                self.output_preview_frame.winfo_height())
            self.output_preview_label.configure(image=self.reconstruction_preview, text='')

    def compress(self):
        ''' Compress the selected source image into the project output folder. '''
        if self.source_path is None or self.model is None:
            return

        self.clear_compressed_result()
        COMPRESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_path = COMPRESSED_DIR / f'{self.source_path.stem}.nic'
        self.status_var.set('Compressing...')
        self.start_worker(compress_image, self.finish_compress, self.source_path, output_path, self.model, self.model_id)

    def finish_compress(self, stats):
        ''' Display statistics for a completed compression operation. '''
        self.compressed_path = stats['output_path']
        ratio = stats['input_size'] / stats['compressed_size']
        reduction = (1.0 - stats['compressed_size'] / stats['input_size']) * 100.0
        original_size = format_file_size(stats['input_size'])
        compressed_size = format_file_size(stats['compressed_size'])
        metrics_text = f'Original {original_size} | Compressed {compressed_size} | Reduction {reduction:.1f}% | ' \
            f'Ratio {ratio:.2f}x | {stats["bpp"]:.3f} bpp'
        self.metrics_var.set(metrics_text)
        saved_path = self.compressed_path.relative_to(PROJECT_DIR)
        self.status_var.set(f'Compression complete: {saved_path}. Click Decode to reconstruct it.')

    def decode(self):
        ''' Read the saved NIC file from disk and reconstruct its image. '''
        if self.compressed_path is None or self.model is None:
            return

        self.status_var.set('Decoding saved .nic file...')
        self.start_worker(decompress_image, self.finish_decode, self.compressed_path, self.model, self.model_id)

    def finish_decode(self, stats):
        ''' Display a reconstruction produced from the saved compressed file. '''
        if 'error' in stats:
            self.status_var.set(stats['error'])
            return

        self.reconstruction_image = to_pil_image(stats['image'])
        self.refresh_previews()
        self.output_info_var.set(f'{stats["width"]} x {stats["height"]} pixels')
        self.status_var.set('Decode complete. The reconstruction was read from the saved .nic file.')

    def start_worker(self, function, callback, *args):
        ''' Run one expensive codec operation outside the GUI event loop. '''
        if self.worker_future is not None:
            return

        self.worker_callback = callback
        self.worker_future = self.worker_pool.submit(function, *args)
        self.update_action_states()
        self.root.after(50, self.check_worker)

    def check_worker(self):
        ''' Finish a background operation when its result becomes available. '''
        if self.worker_future is None:
            return
        if not self.worker_future.done():
            self.root.after(50, self.check_worker)
            return

        future = self.worker_future
        callback = self.worker_callback
        self.worker_future = None
        self.worker_callback = None

        try:
            result = future.result()

        except Exception as error:  # noqa: BLE001
            self.status_var.set(f'Operation failed: {error}')
            self.update_action_states()
            return

        callback(result)
        self.update_action_states()

    def update_action_states(self):
        ''' Enable controls that are valid for the current application state. '''
        busy = self.worker_future is not None
        self.open_button.configure(state='disabled' if busy else 'normal')
        self.checkpoint_button.configure(state='disabled' if busy else 'normal')

        can_compress = not busy and self.model is not None and self.source_path is not None
        can_decode = not busy and self.model is not None and self.compressed_path is not None
        self.compress_button.configure(state='normal' if can_compress else 'disabled')
        self.decode_button.configure(state='normal' if can_decode else 'disabled')

def main():
    ''' Start the desktop inference interface. '''
    ctk.set_appearance_mode('dark')
    ctk.set_default_color_theme('blue')
    root = ctk.CTk()
    CodecGUI(root)
    root.mainloop()

if __name__ == '__main__':
    main()
