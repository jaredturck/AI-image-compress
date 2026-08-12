import torch
import torch.nn as nn
import torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel

CHANNELS = 128
LATENT_CHANNELS = 192


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, CHANNELS, 5, stride=2, padding=2),
            GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2),
            GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2),
            GDN(CHANNELS),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(LATENT_CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1),
            GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1),
            GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1),
            GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, 3, 5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class HyperEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2),
        )

    def forward(self, x):
        return self.layers(x)


class HyperDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class ImageCodec(CompressionModel):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.entropy_bottleneck = EntropyBottleneck(CHANNELS)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x):
        y = self.encoder(x)

        with torch.autocast(device_type=x.device.type, enabled=False):
            y = y.float()
            z = self.hyper_encoder(torch.abs(y))
            z_hat, z_likelihoods = self.entropy_bottleneck(z)
            scales_hat = self.hyper_decoder(z_hat)
            y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
            pixels = x.shape[0] * x.shape[2] * x.shape[3]
            y_bits = -torch.log2(y_likelihoods).sum()
            z_bits = -torch.log2(z_likelihoods).sum()
            bpp = (y_bits + z_bits) / pixels

        reconstruction = self.decoder(y_hat)
        mse = F.mse_loss(reconstruction.float(), x.float())
        return reconstruction, bpp, mse

    @torch.no_grad()
    def compress(self, x):
        y = self.encoder(x)
        z = self.hyper_encoder(torch.abs(y))
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        scales_hat = self.hyper_decoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {
            "strings": [y_strings, z_strings],
            "shape": z.shape[-2:],
        }

    @torch.no_grad()
    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.hyper_decoder(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        return self.decoder(y_hat).clamp_(0, 1)
