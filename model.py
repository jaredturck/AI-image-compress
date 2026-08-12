''' Neural image compression model. '''

import torch, torch.nn as nn, torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel

CHANNELS = 128
LATENT_CHANNELS = 192

class Encoder(nn.Module):
    ''' Encode images into a compact latent representation. '''

    def __init__(self):
        ''' Build the image encoder. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 5, stride=2, padding=2)
        )

    def forward(self, image):
        ''' Encode an image tensor. '''
        return self.layers(image)

class Decoder(nn.Module):
    ''' Reconstruct images from the latent representation. '''

    def __init__(self):
        ''' Build the image decoder. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(LATENT_CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), GDN(CHANNELS, inverse=True),
            nn.ConvTranspose2d(CHANNELS, 3, 5, stride=2, padding=2, output_padding=1)
        )

    def forward(self, latent):
        ''' Decode a latent tensor into an image. '''
        return self.layers(latent)

class HyperEncoder(nn.Module):
    ''' Encode latent statistics for the scale hyperprior. '''

    def __init__(self):
        ''' Build the hyperprior encoder. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, CHANNELS, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2)
        )

    def forward(self, latent):
        ''' Encode latent statistics. '''
        return self.layers(latent)

class HyperDecoder(nn.Module):
    ''' Decode hyperprior values into latent scales. '''

    def __init__(self):
        ''' Build the hyperprior decoder. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(CHANNELS, CHANNELS, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 3, padding=1), nn.ReLU(inplace=True)
        )

    def forward(self, hyper_latent):
        ''' Decode hyperprior values into scales. '''
        return self.layers(hyper_latent)

class ImageCodec(CompressionModel):
    ''' Combine the learned transforms and entropy models into an image codec. '''

    def __init__(self):
        ''' Build the complete image codec. '''
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.hyper_encoder = HyperEncoder()
        self.hyper_decoder = HyperDecoder()
        self.entropy_bottleneck = EntropyBottleneck(CHANNELS)
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

        reconstruction = self.decoder(latent_hat)
        mse = F.mse_loss(reconstruction.float(), image.float())
        return reconstruction, bpp, mse

    @torch.no_grad()
    def compress(self, image):
        ''' Compress an image tensor into entropy-coded byte strings. '''
        latent = self.encoder(image)
        hyper_latent = self.hyper_encoder(torch.abs(latent))
        hyper_strings = self.entropy_bottleneck.compress(hyper_latent)
        hyper_latent_hat = self.entropy_bottleneck.decompress(hyper_strings, hyper_latent.shape[-2:])
        scales = self.hyper_decoder(hyper_latent_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        latent_strings = self.gaussian_conditional.compress(latent, indexes)
        return {'strings': [latent_strings, hyper_strings], 'shape': hyper_latent.shape[-2:]}

    @torch.no_grad()
    def decompress(self, strings, shape):
        ''' Reconstruct an image tensor from entropy-coded byte strings. '''
        hyper_latent_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales = self.hyper_decoder(hyper_latent_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        latent_hat = self.gaussian_conditional.decompress(strings[0], indexes, hyper_latent_hat.dtype)
        return self.decoder(latent_hat).clamp_(0, 1)
