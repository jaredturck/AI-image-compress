''' Neural image compression model. '''

import torch, torch.nn as nn
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel

CHANNELS = 128
LATENT_CHANNELS = 192

class ImageCodec(CompressionModel):
    ''' Combine the learned transforms and entropy models into an image codec. '''

    def __init__(self):
        ''' Build the complete image codec. '''
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 5, stride=2, padding=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(LATENT_CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, 3, 5, stride=2, padding=2, output_padding=1)
        )
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, CHANNELS, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2)
        )
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 3, padding=1), nn.ReLU(inplace=True)
        )
        self.entropy_bottleneck = EntropyBottleneck(CHANNELS)
        self.entropy_bottleneck.quantiles.requires_grad_(False)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, image):
        ''' Run the differentiable training path. '''
        latent = self.encoder(image)

        with torch.autocast(device_type=image.device.type, enabled=False):
            latent = latent.float()
            hyper_latent = self.hyper_encoder(torch.abs(latent))
            hyper_latent_hat, hyper_likelihoods = self.entropy_bottleneck(hyper_latent)
            scales = self.hyper_decoder(hyper_latent_hat)
            latent_hat, latent_likelihoods = self.gaussian_conditional(latent, scales)
            pixels = image.shape[0] * image.shape[2] * image.shape[3]
            latent_bits = -torch.log2(latent_likelihoods).sum()
            hyper_bits = -torch.log2(hyper_likelihoods).sum()
            bpp = (latent_bits + hyper_bits) / pixels

        return self.decoder(latent_hat), bpp

    @torch.no_grad()
    def compress(self, image):
        ''' Compress one image tensor into entropy-coded byte strings. '''
        latent = self.encoder(image)
        hyper_latent = self.hyper_encoder(torch.abs(latent))
        hyper_strings = self.entropy_bottleneck.compress(hyper_latent)
        hyper_latent_hat = self.entropy_bottleneck.decompress(hyper_strings, hyper_latent.shape[-2:])
        scales = self.hyper_decoder(hyper_latent_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        latent_strings = self.gaussian_conditional.compress(latent, indexes)
        return latent_strings[0], hyper_strings[0]

    @torch.no_grad()
    def decompress(self, latent_stream, hyper_stream, shape):
        ''' Reconstruct one image tensor from entropy-coded byte strings. '''
        hyper_latent_hat = self.entropy_bottleneck.decompress([hyper_stream], shape)
        scales = self.hyper_decoder(hyper_latent_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        latent_hat = self.gaussian_conditional.decompress([latent_stream], indexes, hyper_latent_hat.dtype)
        return self.decoder(latent_hat).clamp_(0, 1)
