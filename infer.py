''' Compress and reconstruct images with a trained codec. '''

import hashlib, io, shutil, struct, subprocess, tkinter as tk, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk
import cv2, numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torchvision.io import decode_image
from torchvision.transforms.functional import to_pil_image

from model import ImageCodec

PROJECT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = PROJECT_DIR / 'checkpoints'
COMPRESSED_DIR = PROJECT_DIR / 'compressed_images'
MAGIC = b'NIC4'
HEADER = struct.Struct('<4s16sIIIII')
PREVIEW_PADDING = 24
PREVIEW_RESIZE_DELAY = 50
PREVIEW_ZOOM_STEP = 1.2
PREVIEW_MAX_ZOOM = 8.0
FILTERS = ['Original', 'Grayscale', 'Edges', 'Detail']

def read_file(path):
    ''' Read and validate a compressed NIC image file. '''
    data = Path(path).read_bytes()
    if len(data) < HEADER.size:
        return None

    magic, model_id, width, height, latent_size, hyper_size, texture_size = HEADER.unpack_from(data)
    payload_size = latent_size + hyper_size + texture_size
    if magic != MAGIC or payload_size != len(data) - HEADER.size:
        return None

    latent_offset = HEADER.size
    hyper_offset = latent_offset + latent_size
    texture_offset = hyper_offset + hyper_size
    return {
        'model_id': model_id,
        'width': width,
        'height': height,
        'latent_stream': data[latent_offset:hyper_offset],
        'hyper_stream': data[hyper_offset:texture_offset],
        'texture_stream': data[texture_offset:],
    }

def pack_texture_tokens(texture_mask, texture_indices):
    ''' Store a sparse texture mask and one-byte VQ indices with optional zlib compression. '''
    mask = texture_mask.detach().to(torch.uint8).cpu().numpy().reshape(-1)
    packed_mask = np.packbits(mask, bitorder='little').tobytes()
    raw_indices = texture_indices.detach().to(torch.uint8).cpu().numpy().tobytes()
    raw_texture = packed_mask + raw_indices
    compressed_texture = zlib.compress(raw_texture, level=9)
    if len(compressed_texture) < len(raw_texture):
        return b'\x01' + compressed_texture
    return b'\x00' + raw_texture

def unpack_texture_tokens(texture_stream, width, height):
    ''' Restore the sparse H/8 texture mask and selected VQ indices. '''
    if not texture_stream:
        return None

    padded_width = ((width + 63) // 64) * 64
    padded_height = ((height + 63) // 64) * 64
    token_width = padded_width // 8
    token_height = padded_height // 8
    token_count = token_width * token_height
    mask_size = (token_count + 7) // 8

    if texture_stream[0] == 1:
        raw_texture = zlib.decompress(texture_stream[1:])
    elif texture_stream[0] == 0:
        raw_texture = texture_stream[1:]
    else:
        return None

    if len(raw_texture) < mask_size:
        return None

    packed_mask = np.frombuffer(raw_texture[:mask_size], dtype=np.uint8)
    mask = np.unpackbits(packed_mask, bitorder='little')[:token_count].astype(np.bool_).reshape(token_height, token_width)
    raw_indices = raw_texture[mask_size:]
    if len(raw_indices) != int(mask.sum()):
        return None

    indices = np.frombuffer(raw_indices, dtype=np.uint8).copy()
    return torch.from_numpy(mask), torch.from_numpy(indices)

def encode_jpeg(image, quality):
    ''' Encode one in-memory JPEG candidate for bitrate comparison. '''
    buffer = io.BytesIO()
    image.save(buffer, format='JPEG', quality=quality, subsampling=2, optimize=True)
    return buffer.getvalue()

def make_size_matched_jpeg(image, target_size):
    ''' Find the JPEG quality whose encoded size is closest to the NIC file. '''
    low = 1
    high = 95
    best_quality = 1
    best_data = encode_jpeg(image, best_quality)
    best_difference = abs(len(best_data) - target_size)

    while low <= high:
        quality = (low + high) // 2
        data = encode_jpeg(image, quality)
        difference = abs(len(data) - target_size)

        if difference < best_difference:
            best_quality = quality
            best_data = data
            best_difference = difference

        if len(data) < target_size:
            low = quality + 1
        elif len(data) > target_size:
            high = quality - 1
        else:
            break

    for quality in range(max(1, high - 2), min(95, low + 2) + 1):
        data = encode_jpeg(image, quality)
        difference = abs(len(data) - target_size)
        if difference < best_difference:
            best_quality = quality
            best_data = data
            best_difference = difference

    jpeg_image = Image.open(io.BytesIO(best_data)).convert('RGB')
    return jpeg_image, len(best_data), best_quality

def load_model(checkpoint_path):
    ''' Load a trained image codec checkpoint. '''
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return None

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model = ImageCodec()
    model.load_state_dict(checkpoint['model'])
    model.update(force=True, update_quantiles=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)

    with checkpoint_path.open('rb') as file:
        model_id = hashlib.file_digest(file, 'sha256').digest()[:16]

    return model, model_id, checkpoint_path, checkpoint['metadata']

def compress_image(input_path, output_path, model, model_id):
    ''' Compress an image into the NIC file format and build a size-matched JPEG. '''
    input_path = Path(input_path)
    output_path = Path(output_path)
    source = decode_image(str(input_path), mode='RGB')
    jpeg_source = to_pil_image(source)
    image = source.float().div(255.0)
    height, width = image.shape[-2:]
    pad_height = (-height) % 64
    pad_width = (-width) % 64
    device = next(model.parameters()).device
    image = F.pad(image.unsqueeze(0).to(device), (0, pad_width, 0, pad_height), mode='replicate')

    latent_stream, hyper_stream, texture_mask, texture_indices = model.compress(image)
    texture_stream = pack_texture_tokens(texture_mask, texture_indices)
    header = HEADER.pack(MAGIC, model_id, width, height, len(latent_stream), len(hyper_stream), len(texture_stream))
    output_path.write_bytes(header + latent_stream + hyper_stream + texture_stream)

    input_size = input_path.stat().st_size
    compressed_size = output_path.stat().st_size
    jpeg_image, jpeg_size, jpeg_quality = make_size_matched_jpeg(jpeg_source, compressed_size)
    return {
        'input_size': input_size,
        'compressed_size': compressed_size,
        'bpp': compressed_size * 8.0 / (width * height),
        'output_path': output_path,
        'jpeg_image': jpeg_image,
        'jpeg_size': jpeg_size,
        'jpeg_quality': jpeg_quality,
    }

def decompress_image(input_path, model, model_id):
    ''' Reconstruct an image by reading a NIC file from disk. '''
    packed = read_file(input_path)
    if packed is None:
        return {'error': 'Not a valid NIC4 compressed image.'}
    if packed['model_id'] != model_id:
        return {'error': 'This .nic file was created with a different checkpoint.'}

    try:
        texture_tokens = unpack_texture_tokens(packed['texture_stream'], packed['width'], packed['height'])

    except zlib.error:
        texture_tokens = None

    if texture_tokens is None:
        return {'error': 'The NIC texture stream is invalid.'}

    texture_mask, texture_indices = texture_tokens
    hyper_shape = ((packed['height'] + 63) // 64, (packed['width'] + 63) // 64)
    reconstruction = model.decompress(packed['latent_stream'], packed['hyper_stream'], texture_mask, texture_indices, hyper_shape)
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

def apply_filter(image, filter_name):
    ''' Apply a display-only OpenCV quality inspection filter. '''
    if filter_name == 'Original':
        return image

    image_array = np.asarray(image)
    grayscale = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)

    if filter_name == 'Grayscale':
        filtered = grayscale
    elif filter_name == 'Edges':
        gradient_x = cv2.Sobel(grayscale, cv2.CV_32F, 1, 0, ksize=3)
        gradient_y = cv2.Sobel(grayscale, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(gradient_x, gradient_y)
        filtered = cv2.convertScaleAbs(magnitude, alpha=0.25)
    else:
        laplacian = cv2.Laplacian(grayscale, cv2.CV_16S, ksize=3)
        filtered = cv2.convertScaleAbs(laplacian)

    filtered = cv2.cvtColor(filtered, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(filtered)

def get_preview_scale(image, available_width, available_height, zoom):
    ''' Calculate the source-to-preview scale for the shared viewport. '''
    available_width = max(1, available_width - PREVIEW_PADDING)
    available_height = max(1, available_height - PREVIEW_PADDING)
    return min(available_width / image.width, available_height / image.height) * zoom

def get_preview_center_limits(image, available_width, available_height, zoom):
    ''' Calculate valid normalized viewport center limits. '''
    available_width = max(1, available_width - PREVIEW_PADDING)
    available_height = max(1, available_height - PREVIEW_PADDING)
    display_scale = min(available_width / image.width, available_height / image.height) * zoom
    visible_width = min(float(image.width), available_width / display_scale)
    visible_height = min(float(image.height), available_height / display_scale)
    x_margin = visible_width / (2.0 * image.width)
    y_margin = visible_height / (2.0 * image.height)
    return x_margin, 1.0 - x_margin, y_margin, 1.0 - y_margin

def make_preview(image, available_width, available_height, zoom, center_x, center_y):
    ''' Build a fitted or zoomed CustomTkinter image from the shared viewport. '''
    available_width = max(1, available_width - PREVIEW_PADDING)
    available_height = max(1, available_height - PREVIEW_PADDING)
    display_scale = min(available_width / image.width, available_height / image.height) * zoom
    visible_width = min(float(image.width), available_width / display_scale)
    visible_height = min(float(image.height), available_height / display_scale)
    left = center_x * image.width - visible_width / 2.0
    top = center_y * image.height - visible_height / 2.0
    left = max(0.0, min(left, image.width - visible_width))
    top = max(0.0, min(top, image.height - visible_height))
    right = min(image.width, left + visible_width)
    bottom = min(image.height, top + visible_height)
    crop_box = (int(round(left)), int(round(top)), int(round(right)), int(round(bottom)))
    preview = image.crop(crop_box)
    target_width = max(1, min(available_width, int(round(preview.width * display_scale))))
    target_height = max(1, min(available_height, int(round(preview.height * display_scale))))
    preview = preview.resize((target_width, target_height), Image.Resampling.LANCZOS)
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
        self.source_display_image = None
        self.source_preview = None
        self.reconstruction_image = None
        self.reconstruction_display_image = None
        self.reconstruction_preview = None
        self.jpeg_image = None
        self.jpeg_display_image = None
        self.jpeg_size = None
        self.jpeg_quality = None
        self.output_mode = 'Neural'
        self.compressed_path = None
        self.preview_resize_job = None
        self.preview_zoom = 1.0
        self.preview_center_x = 0.5
        self.preview_center_y = 0.5
        self.pan_x = 0
        self.pan_y = 0
        self.filter_name = 'Original'
        self.worker_pool = ThreadPoolExecutor(max_workers=1)
        self.worker_future = None
        self.worker_callback = None

        self.status_var = tk.StringVar(value='Loading the latest checkpoint...')
        self.source_name_var = tk.StringVar(value='No image selected')
        self.source_info_var = tk.StringVar(value='Open an image to begin.')
        self.output_info_var = tk.StringVar(value='Compress and decode an image to compare the reconstruction.')
        self.checkpoint_var = tk.StringVar(value='Checkpoint: No checkpoint loaded')
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
        ctk.CTkLabel(checkpoint_frame, textvariable=self.checkpoint_var, text_color='#b5b5b5').grid(row=0, column=0, sticky='e', padx=(0, 18))
        ctk.CTkLabel(checkpoint_frame, text='Filter:', text_color='#9c9c9c').grid(row=0, column=1, sticky='e', padx=(0, 6))
        self.filter_menu = ctk.CTkOptionMenu(checkpoint_frame, values=FILTERS, width=110, command=self.change_filter)
        self.filter_menu.set(self.filter_name)
        self.filter_menu.grid(row=0, column=2, sticky='e', padx=(0, 10))
        self.checkpoint_button = ctk.CTkButton(checkpoint_frame, text='Change', width=90, command=self.browse_checkpoint)
        self.checkpoint_button.grid(row=0, column=3)

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
        self.bind_preview_events(self.source_preview_label)

        self.compress_button = ctk.CTkButton(panel, text='Compress', height=42, state='disabled', command=self.compress)
        self.compress_button.grid(row=4, column=0, sticky='ew', padx=18, pady=(0, 18))

    def build_output_panel(self, parent):
        ''' Build the reconstructed image and compression statistics panel. '''
        panel = ctk.CTkFrame(parent, corner_radius=16)
        panel.grid(row=0, column=1, sticky='nsew', padx=(9, 0))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(3, weight=1)

        header = ctk.CTkFrame(panel, fg_color='transparent')
        header.grid(row=0, column=0, sticky='ew', padx=18, pady=(18, 8))
        header.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header, text='Reconstruction', font=ctk.CTkFont(size=18, weight='bold')).grid(row=0, column=0, sticky='w')
        self.output_toggle = ctk.CTkSegmentedButton(header, values=['Neural', 'JPEG'], width=160, command=self.change_output_mode)
        self.output_toggle.set(self.output_mode)
        self.output_toggle.grid(row=0, column=1, sticky='e')
        ctk.CTkLabel(panel, textvariable=self.output_info_var, anchor='w', text_color='#9c9c9c').grid(
            row=1, column=0, sticky='ew', padx=18)
        ctk.CTkLabel(panel, text='Toggle between the saved .nic reconstruction and a size-matched JPEG.', anchor='w', text_color='#6f9fd8').grid(
            row=2, column=0, sticky='ew', padx=18, pady=(2, 10))

        self.output_preview_frame = ctk.CTkFrame(panel, corner_radius=12, fg_color='#171717')
        self.output_preview_frame.grid(row=3, column=0, sticky='nsew', padx=18, pady=(0, 12))
        self.output_preview_frame.grid_columnconfigure(0, weight=1)
        self.output_preview_frame.grid_rowconfigure(0, weight=1)
        self.output_preview_frame.bind('<Configure>', self.schedule_preview_resize)
        self.output_preview_label = ctk.CTkLabel(self.output_preview_frame, text='No reconstruction yet', text_color='#777777')
        self.output_preview_label.grid(row=0, column=0, sticky='nsew', padx=12, pady=12)
        self.bind_preview_events(self.output_preview_label)

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

        self.model, self.model_id, self.checkpoint_path, metadata = loaded
        self.checkpoint_var.set(f'Checkpoint: {self.checkpoint_path.name}, loss {metadata["loss"]:.4f}, {metadata["dataset_images"]:,} images')
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
        self.source_display_image = apply_filter(self.source_image, self.filter_name)
        self.preview_zoom = 1.0
        self.preview_center_x = 0.5
        self.preview_center_y = 0.5
        self.source_name_var.set(self.source_path.name)
        self.source_info_var.set(f'{image.shape[2]} x {image.shape[1]} pixels | original dimensions preserved')
        self.clear_compressed_result()
        self.refresh_previews()
        self.status_var.set('Image ready. Click Compress to create the .nic file.')
        self.update_action_states()

    def clear_compressed_result(self):
        ''' Clear output state that belongs to a previous source or checkpoint. '''
        self.compressed_path = None
        # CustomTkinter leaves the underlying Tk image unchanged when image=None.
        self.output_preview_label.configure(image='', text='No reconstruction yet')
        self.reconstruction_image = None
        self.reconstruction_display_image = None
        self.reconstruction_preview = None
        self.jpeg_image = None
        self.jpeg_display_image = None
        self.jpeg_size = None
        self.jpeg_quality = None
        self.output_mode = 'Neural'
        self.output_toggle.set(self.output_mode)
        self.output_info_var.set('Compress and decode an image to compare the reconstruction.')
        self.metrics_var.set('No compression statistics yet')
        self.update_action_states()

    def schedule_preview_resize(self, _event=None):
        ''' Refresh image previews after the preview area changes size. '''
        if self.preview_resize_job is not None:
            self.root.after_cancel(self.preview_resize_job)
        self.preview_resize_job = self.root.after(PREVIEW_RESIZE_DELAY, self.refresh_previews)

    def refresh_previews(self):
        ''' Render the source and selected comparison from the same shared viewport. '''
        self.preview_resize_job = None
        available_width = min(self.source_preview_frame.winfo_width(), self.output_preview_frame.winfo_width())
        available_height = min(self.source_preview_frame.winfo_height(), self.output_preview_frame.winfo_height())

        if self.source_display_image is not None:
            self.source_preview = make_preview(self.source_display_image, available_width, available_height, self.preview_zoom,
                self.preview_center_x, self.preview_center_y)
            self.source_preview_label.configure(image=self.source_preview, text='')

        output_image = self.reconstruction_display_image if self.output_mode == 'Neural' else self.jpeg_display_image
        if output_image is not None:
            self.reconstruction_preview = make_preview(output_image, available_width, available_height, self.preview_zoom,
                self.preview_center_x, self.preview_center_y)
            self.output_preview_label.configure(image=self.reconstruction_preview, text='')
        else:
            self.reconstruction_preview = None
            empty_text = 'No reconstruction yet' if self.output_mode == 'Neural' else 'No JPEG comparison yet'
            self.output_preview_label.configure(image='', text=empty_text)

    def bind_preview_events(self, label):
        ''' Bind synchronized zoom and pan controls to an image preview. '''
        label.bind('<MouseWheel>', self.zoom_preview)
        label.bind('<Button-4>', self.zoom_preview)
        label.bind('<Button-5>', self.zoom_preview)
        label.bind('<ButtonPress-1>', self.start_pan)
        label.bind('<B1-Motion>', self.pan_preview)
        label.bind('<Double-Button-1>', self.reset_viewport)

    def change_filter(self, filter_name):
        ''' Apply one quality inspection filter to all cached comparison images. '''
        self.filter_name = filter_name

        if self.source_image is not None:
            self.source_display_image = apply_filter(self.source_image, filter_name)
        if self.reconstruction_image is not None:
            self.reconstruction_display_image = apply_filter(self.reconstruction_image, filter_name)
        if self.jpeg_image is not None:
            self.jpeg_display_image = apply_filter(self.jpeg_image, filter_name)

        self.refresh_previews()

    def change_output_mode(self, output_mode):
        ''' Switch the right preview between neural and size-matched JPEG output. '''
        self.output_mode = output_mode
        self.update_output_info()
        self.refresh_previews()

    def update_output_info(self):
        ''' Describe the comparison image currently selected in the right panel. '''
        if self.output_mode == 'JPEG':
            if self.jpeg_image is None:
                self.output_info_var.set('Compress an image to build the JPEG comparison.')
                return

            width, height = self.jpeg_image.size
            self.output_info_var.set(f'{width} x {height} pixels | JPEG q{self.jpeg_quality} | {format_file_size(self.jpeg_size)}')
            return

        if self.reconstruction_image is None:
            self.output_info_var.set('Decode the saved .nic file to view the neural reconstruction.')
            return

        self.output_info_var.set(f'{self.reconstruction_image.width} x {self.reconstruction_image.height} pixels | neural reconstruction')

    def zoom_preview(self, event):
        ''' Zoom both image previews together. '''
        if self.source_display_image is None:
            return 'break'

        zoom_in = getattr(event, 'num', 0) == 4 or getattr(event, 'delta', 0) > 0
        if zoom_in:
            self.preview_zoom = min(PREVIEW_MAX_ZOOM, self.preview_zoom * PREVIEW_ZOOM_STEP)
        else:
            self.preview_zoom = max(1.0, self.preview_zoom / PREVIEW_ZOOM_STEP)

        available_width = min(self.source_preview_frame.winfo_width(), self.output_preview_frame.winfo_width())
        available_height = min(self.source_preview_frame.winfo_height(), self.output_preview_frame.winfo_height())
        min_x, max_x, min_y, max_y = get_preview_center_limits(self.source_display_image, available_width, available_height, self.preview_zoom)
        self.preview_center_x = max(min_x, min(max_x, self.preview_center_x))
        self.preview_center_y = max(min_y, min(max_y, self.preview_center_y))
        self.refresh_previews()
        return 'break'

    def start_pan(self, event):
        ''' Start dragging the shared image viewport. '''
        self.pan_x = event.x_root
        self.pan_y = event.y_root

    def pan_preview(self, event):
        ''' Pan both image previews together while zoomed. '''
        if self.source_display_image is None or self.preview_zoom <= 1.0:
            return

        delta_x = event.x_root - self.pan_x
        delta_y = event.y_root - self.pan_y
        self.pan_x = event.x_root
        self.pan_y = event.y_root
        available_width = min(self.source_preview_frame.winfo_width(), self.output_preview_frame.winfo_width())
        available_height = min(self.source_preview_frame.winfo_height(), self.output_preview_frame.winfo_height())
        display_scale = get_preview_scale(self.source_display_image, available_width, available_height, self.preview_zoom)
        display_width = self.source_display_image.width * display_scale
        display_height = self.source_display_image.height * display_scale
        min_x, max_x, min_y, max_y = get_preview_center_limits(self.source_display_image, available_width, available_height, self.preview_zoom)
        self.preview_center_x = max(min_x, min(max_x, self.preview_center_x - delta_x / display_width))
        self.preview_center_y = max(min_y, min(max_y, self.preview_center_y - delta_y / display_height))
        self.refresh_previews()

    def reset_viewport(self, _event=None):
        ''' Reset synchronized image inspection to the fitted full-image view. '''
        self.preview_zoom = 1.0
        self.preview_center_x = 0.5
        self.preview_center_y = 0.5
        self.refresh_previews()

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
        ''' Display neural compression statistics and cache the size-matched JPEG. '''
        self.compressed_path = stats['output_path']
        self.jpeg_image = stats['jpeg_image']
        self.jpeg_display_image = apply_filter(self.jpeg_image, self.filter_name)
        self.jpeg_size = stats['jpeg_size']
        self.jpeg_quality = stats['jpeg_quality']
        ratio = stats['input_size'] / stats['compressed_size']
        reduction = (1.0 - stats['compressed_size'] / stats['input_size']) * 100.0
        original_size = format_file_size(stats['input_size'])
        compressed_size = format_file_size(stats['compressed_size'])
        jpeg_size = format_file_size(stats['jpeg_size'])
        metrics_text = f'Original {original_size} | NIC {compressed_size} | Reduction {reduction:.1f}% | Ratio {ratio:.2f}x | ' \
            f'{stats["bpp"]:.3f} bpp | JPEG {jpeg_size} q{stats["jpeg_quality"]}'
        self.metrics_var.set(metrics_text)
        self.update_output_info()
        self.refresh_previews()
        saved_path = self.compressed_path.relative_to(PROJECT_DIR)
        self.status_var.set(f'Compression complete: {saved_path}. JPEG comparison ready; click Decode for the neural reconstruction.')

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
        self.reconstruction_display_image = apply_filter(self.reconstruction_image, self.filter_name)
        self.update_output_info()
        self.refresh_previews()
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
        self.filter_menu.configure(state='disabled' if busy else 'normal')
        self.output_toggle.configure(state='normal' if not busy and self.jpeg_image is not None else 'disabled')

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
