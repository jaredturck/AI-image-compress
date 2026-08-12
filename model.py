import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

LATENT_CHANNELS = 256
HYPER_CHANNELS = 128
GROUP_SIZES = (16, 16, 32, 64, 128)
MIN_SCALE = 0.11
MAX_SCALE = 256.0
SYMBOL_MAX = 255
SCALE_LEVELS = 64
MEAN_STEPS = 16

class GDN(nn.Module):
    def __init__(self, channels, inverse=False):
        super().__init__()
        self.inverse = inverse
        self.beta_param = nn.Parameter(torch.ones(channels))
        gamma = torch.eye(channels) * math.sqrt(0.1)
        self.gamma_param = nn.Parameter(gamma)

    def forward(self, x):
        beta = self.beta_param.square() + 1e-6
        gamma = self.gamma_param.square()
        norm = F.conv2d(x.square(), gamma[:, :, None, None], beta)
        if self.inverse:
            return x * torch.sqrt(norm)
        return x * torch.rsqrt(norm)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(F.silu(self.conv1(x)))


class FlashAttentionBlock(nn.Module):
    def __init__(self, channels, heads=8, window_size=16):
        super().__init__()
        self.channels = channels
        self.heads = heads
        self.head_dim = channels // heads
        self.window_size = window_size
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.ff_norm = nn.LayerNorm(channels)
        self.ff1 = nn.Linear(channels, channels * 2)
        self.ff2 = nn.Linear(channels * 2, channels)

    def forward(self, x):
        batch, channels, height, width = x.shape
        window = self.window_size
        pad_height = (window - height % window) % window
        pad_width = (window - width % window) % window
        padded = F.pad(x, (0, pad_width, 0, pad_height))
        padded_height, padded_width = padded.shape[-2:]
        rows = padded_height // window
        columns = padded_width // window

        tokens = padded.view(batch, channels, rows, window, columns, window)
        tokens = tokens.permute(0, 2, 4, 3, 5, 1).reshape(batch * rows * columns, window * window, channels)
        residual = tokens
        tokens = self.norm(tokens)
        qkv = self.qkv(tokens).view(tokens.shape[0], window * window, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if x.is_cuda and x.dtype in (torch.float16, torch.bfloat16):
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attended = F.scaled_dot_product_attention(q, k, v)
        else:
            attended = F.scaled_dot_product_attention(q, k, v)

        attended = attended.transpose(1, 2).reshape(tokens.shape[0], window * window, channels)
        tokens = residual + self.proj(attended)
        tokens = tokens + self.ff2(F.silu(self.ff1(self.ff_norm(tokens))))
        tokens = tokens.view(batch, rows, columns, window, window, channels)
        tokens = tokens.permute(0, 5, 1, 3, 2, 4).reshape(batch, channels, padded_height, padded_width)
        return tokens[:, :, :height, :width]

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 128, 5, stride=2, padding=2)
        self.gdn1 = GDN(128)
        self.res1 = nn.Sequential(ResidualBlock(128))

        self.down2 = nn.Conv2d(128, 192, 5, stride=2, padding=2)
        self.gdn2 = GDN(192)
        self.res2 = nn.Sequential(ResidualBlock(192), ResidualBlock(192))

        self.down3 = nn.Conv2d(192, 256, 5, stride=2, padding=2)
        self.gdn3 = GDN(256)
        self.res3 = nn.Sequential(ResidualBlock(256), ResidualBlock(256), ResidualBlock(256))
        self.attn3 = FlashAttentionBlock(256, 8)

        self.down4 = nn.Conv2d(256, 320, 5, stride=2, padding=2)
        self.gdn4 = GDN(320)
        self.res4 = nn.Sequential(*[ResidualBlock(320) for _ in range(6)])
        self.attn4 = FlashAttentionBlock(320, 8)
        self.to_latent = nn.Conv2d(320, LATENT_CHANNELS, 3, padding=1)

    def forward(self, x):
        x = self.res1(self.gdn1(self.down1(x)))
        x = self.res2(self.gdn2(self.down2(x)))
        x = self.res3(self.gdn3(self.down3(x)))
        x = self.attn3(x)
        x = self.res4(self.gdn4(self.down4(x)))
        x = self.attn4(x)
        return self.to_latent(x)

class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, blocks, attention=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 5, padding=2)
        self.gdn = GDN(out_channels, inverse=True)
        self.res = nn.Sequential(*[ResidualBlock(out_channels) for _ in range(blocks)])
        self.attn = FlashAttentionBlock(out_channels, 8) if attention else nn.Identity()

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = self.gdn(self.conv(x))
        x = self.res(x)
        return self.attn(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.from_latent = nn.Conv2d(LATENT_CHANNELS, 320, 3, padding=1)
        self.attn0 = FlashAttentionBlock(320, 8)
        self.res0 = nn.Sequential(*[ResidualBlock(320) for _ in range(12)])
        self.up1 = UpsampleBlock(320, 256, 3, attention=True)
        self.up2 = UpsampleBlock(256, 192, 2)
        self.up3 = UpsampleBlock(192, 128, 1)
        self.up4 = UpsampleBlock(128, 64, 1)
        self.out = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.from_latent(x)
        x = self.attn0(x)
        x = self.res0(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        return torch.sigmoid(self.out(x))


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(LATENT_CHANNELS, 192, 3, padding=1)
        self.conv2 = nn.Conv2d(192, 160, 5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(160, HYPER_CHANNELS, 5, stride=2, padding=2)

    def forward(self, y):
        x = F.silu(self.conv1(y.abs()))
        x = F.silu(self.conv2(x))
        return self.conv3(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(HYPER_CHANNELS, 160, 5, padding=2)
        self.conv2 = nn.Conv2d(160, 192, 5, padding=2)
        self.conv3 = nn.Conv2d(192, LATENT_CHANNELS * 2, 3, padding=1)

    def forward(self, z):
        x = F.interpolate(z, scale_factor=2, mode="nearest")
        x = F.silu(self.conv1(x))
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        x = F.silu(self.conv2(x))
        return self.conv3(x)


class ChannelContext(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(LATENT_CHANNELS, 192, 3, padding=1)
        self.conv2 = nn.Conv2d(192, 192, 3, padding=1)
        self.conv3 = nn.Conv2d(192, LATENT_CHANNELS * 2, 1)
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, q_context):
        x = F.silu(self.conv1(q_context))
        x = F.silu(self.conv2(x))
        return self.conv3(x)


class ImageCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.context = ChannelContext()
        self.z_scale_param = nn.Parameter(torch.zeros(HYPER_CHANNELS))
        scale_table = torch.exp(torch.linspace(math.log(MIN_SCALE), math.log(MAX_SCALE), SCALE_LEVELS))
        self.register_buffer("scale_table", scale_table)

    def quantize_mean(self, mean):
        quantized = torch.round(mean * MEAN_STEPS) / MEAN_STEPS
        if self.training:
            return mean + (quantized - mean).detach()
        return quantized

    def quantize_scale(self, scale):
        scale = scale.clamp(MIN_SCALE, MAX_SCALE)
        position = (torch.log(scale) - math.log(MIN_SCALE)) / (math.log(MAX_SCALE) - math.log(MIN_SCALE))
        index = torch.round(position * (SCALE_LEVELS - 1)).long().clamp(0, SCALE_LEVELS - 1)
        quantized = self.scale_table[index]
        if self.training:
            quantized = scale + (quantized - scale).detach()
        return quantized, index

    def z_scales(self):
        scale = MIN_SCALE + F.softplus(self.z_scale_param)
        scale = scale.clamp(MIN_SCALE, MAX_SCALE)[None, :, None, None]
        scale, _ = self.quantize_scale(scale)
        return scale

    def analysis(self, x):
        y = self.encoder(x)
        z = self.hyper_encoder(y)
        return y, z

    def hyper_stats(self, z_hat):
        stats = self.hyper_decoder(z_hat)
        mean, raw_scale = stats.chunk(2, dim=1)
        scale = (MIN_SCALE + F.softplus(raw_scale)).clamp(MIN_SCALE, MAX_SCALE)
        return mean, scale

    def group_stats(self, hyper_mean, hyper_scale, q_context, start, end):
        context_stats = self.context(q_context)
        context_mean, context_raw_scale = context_stats.chunk(2, dim=1)
        mean = hyper_mean[:, start:end] + context_mean[:, start:end]
        scale = hyper_scale[:, start:end] * torch.exp(torch.tanh(context_raw_scale[:, start:end]) * 2.0)
        mean = self.quantize_mean(mean)
        scale, scale_index = self.quantize_scale(scale)
        return mean, scale, scale_index

    def gaussian_likelihood(self, values, scales):
        scales = scales.clamp(MIN_SCALE, MAX_SCALE)
        upper = (values + 0.5) / scales
        lower = (values - 0.5) / scales
        upper_cdf = 0.5 * (1.0 + torch.erf(upper / math.sqrt(2.0)))
        lower_cdf = 0.5 * (1.0 + torch.erf(lower / math.sqrt(2.0)))
        return (upper_cdf - lower_cdf).clamp_min(1e-9)

    def quantize_noise(self, x):
        if self.training:
            return (x + torch.empty_like(x).uniform_(-0.5, 0.5)).clamp(-SYMBOL_MAX, SYMBOL_MAX)
        return x.round().clamp(-SYMBOL_MAX, SYMBOL_MAX)

    def forward(self, x):
        y = self.encoder(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            y_float = y.float()
            z = self.hyper_encoder(y_float)
            z_hat = self.quantize_noise(z)
            z_prob = self.gaussian_likelihood(z_hat, self.z_scales())
            hyper_mean, hyper_scale = self.hyper_stats(z_hat)

            context_parts = []
            y_hat_parts = []
            y_bits = torch.zeros((), device=x.device, dtype=torch.float32)
            start = 0

            for size in GROUP_SIZES:
                end = start + size
                if context_parts:
                    q_context = torch.cat(context_parts + [torch.zeros_like(y_float[:, start:])], dim=1)
                else:
                    q_context = torch.zeros_like(y_float)

                mean, scale, _ = self.group_stats(hyper_mean, hyper_scale, q_context, start, end)
                residual = y_float[:, start:end] - mean
                q = self.quantize_noise(residual)
                likelihood = self.gaussian_likelihood(q, scale)
                y_bits = y_bits - torch.log2(likelihood.float()).sum()
                y_hat_parts.append(q + mean)
                context_parts.append(q)
                start = end

            y_hat = torch.cat(y_hat_parts, dim=1)
            z_bits = -torch.log2(z_prob.float()).sum()
            pixels = x.shape[0] * x.shape[2] * x.shape[3]
            bpp = (y_bits + z_bits) / pixels

        reconstruction = self.decoder(y_hat)
        mse = F.mse_loss(reconstruction.float(), x.float())
        return reconstruction, bpp, mse



def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
