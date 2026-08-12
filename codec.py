import bisect
import math
import struct
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from model import GROUP_SIZES, HYPER_CHANNELS, LATENT_CHANNELS, SCALE_LEVELS, SYMBOL_MAX


MAGIC = b"NIC1"
VERSION = 1
CDF_TOTAL = 1 << 15
HEADER = struct.Struct("<4sB16sIIIIII")


class BitWriter:
    def __init__(self):
        self.data = bytearray()
        self.current = 0
        self.count = 0

    def write(self, bit):
        self.current = (self.current << 1) | int(bit)
        self.count += 1
        if self.count == 8:
            self.data.append(self.current)
            self.current = 0
            self.count = 0

    def finish(self):
        if self.count:
            self.current <<= 8 - self.count
            self.data.append(self.current)
        return bytes(self.data)


class BitReader:
    def __init__(self, data):
        self.data = data
        self.byte_index = 0
        self.bit_index = 0

    def read(self):
        if self.byte_index >= len(self.data):
            return 0
        byte = self.data[self.byte_index]
        bit = (byte >> (7 - self.bit_index)) & 1
        self.bit_index += 1
        if self.bit_index == 8:
            self.bit_index = 0
            self.byte_index += 1
        return bit


class ArithmeticEncoder:
    def __init__(self):
        self.state_bits = 32
        self.mask = (1 << self.state_bits) - 1
        self.half = 1 << (self.state_bits - 1)
        self.quarter = self.half >> 1
        self.three_quarter = self.quarter * 3
        self.low = 0
        self.high = self.mask
        self.pending = 0
        self.writer = BitWriter()

    def write_pending(self, bit):
        self.writer.write(bit)
        inverse = 1 - bit
        for _ in range(self.pending):
            self.writer.write(inverse)
        self.pending = 0

    def update(self, cum_low, cum_high, total):
        current_range = self.high - self.low + 1
        self.high = self.low + current_range * cum_high // total - 1
        self.low = self.low + current_range * cum_low // total

        while True:
            if self.high < self.half:
                self.write_pending(0)
            elif self.low >= self.half:
                self.write_pending(1)
                self.low -= self.half
                self.high -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarter:
                self.pending += 1
                self.low -= self.quarter
                self.high -= self.quarter
            else:
                break

            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1

    def finish(self):
        self.pending += 1
        if self.low < self.quarter:
            self.write_pending(0)
        else:
            self.write_pending(1)
        return self.writer.finish()


class ArithmeticDecoder:
    def __init__(self, data):
        self.state_bits = 32
        self.mask = (1 << self.state_bits) - 1
        self.half = 1 << (self.state_bits - 1)
        self.quarter = self.half >> 1
        self.three_quarter = self.quarter * 3
        self.low = 0
        self.high = self.mask
        self.reader = BitReader(data)
        self.code = 0
        for _ in range(self.state_bits):
            self.code = (self.code << 1) | self.reader.read()

    def target(self, total):
        current_range = self.high - self.low + 1
        return ((self.code - self.low + 1) * total - 1) // current_range

    def update(self, cum_low, cum_high, total):
        current_range = self.high - self.low + 1
        self.high = self.low + current_range * cum_high // total - 1
        self.low = self.low + current_range * cum_low // total

        while True:
            if self.high < self.half:
                pass
            elif self.low >= self.half:
                self.low -= self.half
                self.high -= self.half
                self.code -= self.half
            elif self.low >= self.quarter and self.high < self.three_quarter:
                self.low -= self.quarter
                self.high -= self.quarter
                self.code -= self.quarter
            else:
                break

            self.low = (self.low << 1) & self.mask
            self.high = ((self.high << 1) & self.mask) | 1
            self.code = ((self.code << 1) & self.mask) | self.reader.read()


class GaussianCDFTable:
    def __init__(self, scales):
        self.scales = [float(scale) for scale in scales]
        self.cdfs = [self.build_cdf(scale) for scale in self.scales]

    def gaussian_cdf(self, value, scale):
        return 0.5 * (1.0 + math.erf(value / (scale * math.sqrt(2.0))))

    def build_cdf(self, scale):
        probabilities = []
        for symbol in range(-SYMBOL_MAX, SYMBOL_MAX + 1):
            lower = -math.inf if symbol == -SYMBOL_MAX else symbol - 0.5
            upper = math.inf if symbol == SYMBOL_MAX else symbol + 0.5
            lower_cdf = 0.0 if math.isinf(lower) else self.gaussian_cdf(lower, scale)
            upper_cdf = 1.0 if math.isinf(upper) else self.gaussian_cdf(upper, scale)
            probabilities.append(max(0.0, upper_cdf - lower_cdf))

        alphabet = len(probabilities)
        remaining = CDF_TOTAL - alphabet
        weighted = [probability * remaining for probability in probabilities]
        extras = [int(value) for value in weighted]
        leftover = remaining - sum(extras)
        fractions = sorted(range(alphabet), key=lambda index: weighted[index] - extras[index], reverse=True)
        for index in fractions[:leftover]:
            extras[index] += 1

        frequencies = [extra + 1 for extra in extras]
        cdf = [0]
        total = 0
        for frequency in frequencies:
            total += frequency
            cdf.append(total)
        return cdf


class EntropyCodec:
    def __init__(self, model):
        self.model = model
        self.cdf_table = GaussianCDFTable(model.scale_table.detach().cpu().tolist())

    def encode_symbols(self, encoder, symbols, scale_indexes):
        symbol_list = symbols.detach().to("cpu", torch.int16).reshape(-1).tolist()
        scale_list = scale_indexes.detach().to("cpu", torch.int16).reshape(-1).tolist()
        for symbol, scale_index in zip(symbol_list, scale_list):
            cdf = self.cdf_table.cdfs[scale_index]
            index = symbol + SYMBOL_MAX
            encoder.update(cdf[index], cdf[index + 1], CDF_TOTAL)

    def decode_symbols(self, decoder, scale_indexes, shape, device):
        scale_list = scale_indexes.detach().to("cpu", torch.int16).reshape(-1).tolist()
        symbols = []
        for scale_index in scale_list:
            cdf = self.cdf_table.cdfs[scale_index]
            target = decoder.target(CDF_TOTAL)
            index = bisect.bisect_right(cdf, target) - 1
            decoder.update(cdf[index], cdf[index + 1], CDF_TOTAL)
            symbols.append(index - SYMBOL_MAX)
        return torch.tensor(symbols, dtype=torch.float32, device=device).reshape(shape)


def inference_autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_context(parts, reference, start):
    if not parts:
        return torch.zeros_like(reference)
    return torch.cat(parts + [torch.zeros_like(reference[:, start:])], dim=1)


def encode_latents(model, image):
    image_device = image.device
    entropy_device = model.z_scale_param.device
    entropy = EntropyCodec(model)

    with torch.no_grad():
        with inference_autocast(image_device):
            y = model.encoder(image)

        y = y.float().to(entropy_device)
        z = model.hyper_encoder(y)
        q_z = z.round().clamp(-SYMBOL_MAX, SYMBOL_MAX)
        hyper_mean, hyper_scale = model.hyper_stats(q_z)

        z_scale = model.z_scales().expand_as(q_z)
        _, z_scale_index = model.quantize_scale(z_scale)
        z_encoder = ArithmeticEncoder()
        entropy.encode_symbols(z_encoder, q_z, z_scale_index)
        z_stream = z_encoder.finish()

        y_encoder = ArithmeticEncoder()
        context_parts = []
        start = 0

        for size in GROUP_SIZES:
            end = start + size
            q_context = build_context(context_parts, y, start)
            mean, _, scale_index = model.group_stats(hyper_mean, hyper_scale, q_context, start, end)
            q = (y[:, start:end] - mean).round().clamp(-SYMBOL_MAX, SYMBOL_MAX)
            entropy.encode_symbols(y_encoder, q, scale_index)
            context_parts.append(q)
            start = end

        y_stream = y_encoder.finish()

    return z_stream, y_stream


def decode_latents(model, z_stream, y_stream, padded_height, padded_width, device):
    entropy_device = model.z_scale_param.device
    entropy = EntropyCodec(model)
    y_height = padded_height // 16
    y_width = padded_width // 16
    z_height = padded_height // 64
    z_width = padded_width // 64

    with torch.no_grad():
        z_shape = (1, HYPER_CHANNELS, z_height, z_width)
        z_scale = model.z_scales().expand(z_shape)
        _, z_scale_index = model.quantize_scale(z_scale)
        z_decoder = ArithmeticDecoder(z_stream)
        q_z = entropy.decode_symbols(z_decoder, z_scale_index, z_shape, entropy_device)
        hyper_mean, hyper_scale = model.hyper_stats(q_z)

        reference = torch.zeros((1, LATENT_CHANNELS, y_height, y_width), device=entropy_device)
        context_parts = []
        y_parts = []
        start = 0
        y_decoder = ArithmeticDecoder(y_stream)

        for size in GROUP_SIZES:
            end = start + size
            q_context = build_context(context_parts, reference, start)
            mean, _, scale_index = model.group_stats(hyper_mean, hyper_scale, q_context, start, end)
            shape = (1, size, y_height, y_width)
            q = entropy.decode_symbols(y_decoder, scale_index, shape, entropy_device)
            context_parts.append(q)
            y_parts.append(q + mean)
            start = end

        y_hat = torch.cat(y_parts, dim=1).to(device)

        with inference_autocast(device):
            reconstruction = model.decoder(y_hat)

    return reconstruction.float().clamp(0.0, 1.0)


def pad_image(image):
    height, width = image.shape[-2:]
    padded_height = math.ceil(height / 64) * 64
    padded_width = math.ceil(width / 64) * 64
    pad_bottom = padded_height - height
    pad_right = padded_width - width
    padded = F.pad(image, (0, pad_right, 0, pad_bottom), mode="replicate")
    return padded, padded_height, padded_width


def write_file(path, model_id, width, height, padded_width, padded_height, z_stream, y_stream):
    header = HEADER.pack(
        MAGIC,
        VERSION,
        model_id,
        width,
        height,
        padded_width,
        padded_height,
        len(z_stream),
        len(y_stream),
    )
    Path(path).write_bytes(header + z_stream + y_stream)


def read_file(path):
    data = Path(path).read_bytes()
    values = HEADER.unpack(data[:HEADER.size])
    magic, version, model_id, width, height, padded_width, padded_height, z_length, y_length = values
    if magic != MAGIC or version != VERSION:
        return None
    start = HEADER.size
    z_stream = data[start:start + z_length]
    y_stream = data[start + z_length:start + z_length + y_length]
    return {
        "model_id": model_id,
        "width": width,
        "height": height,
        "padded_width": padded_width,
        "padded_height": padded_height,
        "z_stream": z_stream,
        "y_stream": y_stream,
        "file_size": len(data),
    }
