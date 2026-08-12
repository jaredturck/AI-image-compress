''' Neural image compression model. '''

import torch, torch.nn as nn, torch.nn.functional as F
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN
from compressai.models import CompressionModel

CHANNELS = 128
LATENT_CHANNELS = 192
TEXTURE_CHANNELS = 96
TEXTURE_DIM = 48
TEXTURE_CODEBOOK_SIZE = 256
TEXTURE_COMMITMENT = 0.25
TEXTURE_TOKEN_FRACTION = 0.25

def extract_texture(image):
    ''' Separate local high-frequency texture from coarse image structure. '''
    smooth = F.avg_pool2d(image, kernel_size=5, stride=1, padding=2, count_include_pad=False)
    return image - smooth

def select_texture_mask(texture):
    ''' Select the most detailed H/8 regions for the dedicated texture stream. '''
    detail_energy = texture.abs().mean(dim=1, keepdim=True)
    detail_energy = F.avg_pool2d(detail_energy, kernel_size=8, stride=8)
    flattened = detail_energy.flatten(1)
    selected_count = max(1, int(flattened.shape[1] * TEXTURE_TOKEN_FRACTION))
    selected = flattened.topk(selected_count, dim=1, sorted=False).indices
    mask = torch.zeros_like(flattened, dtype=torch.bool)
    mask.scatter_(1, selected, True)
    return mask.view(texture.shape[0], detail_energy.shape[2], detail_energy.shape[3])

class ResidualBlock(nn.Module):
    ''' Refine generative image features without changing their dimensions. '''

    def __init__(self, channels):
        ''' Build a compact residual feature block. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.GroupNorm(32, channels), nn.SiLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels), nn.SiLU(inplace=True), nn.Conv2d(channels, channels, 3, padding=1)
        )

    def forward(self, features):
        ''' Add learned feature refinement to the residual input. '''
        return features + self.layers(features)

class StructureEncoder(nn.Module):
    ''' Preserve image geometry, color, composition, and coarse appearance. '''

    def __init__(self):
        ''' Build the structure analysis transform. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, CHANNELS, 5, stride=2, padding=2), GDN(CHANNELS),
            nn.Conv2d(CHANNELS, LATENT_CHANNELS, 5, stride=2, padding=2)
        )

    def forward(self, image):
        ''' Encode coarse structure into the entropy-coded latent. '''
        return self.layers(image)

class TextureEncoder(nn.Module):
    ''' Convert local high-frequency image information into VQ features. '''

    def __init__(self):
        ''' Build the texture analysis transform. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.SiLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), ResidualBlock(64),
            nn.Conv2d(64, TEXTURE_CHANNELS, 4, stride=2, padding=1), ResidualBlock(TEXTURE_CHANNELS),
            nn.GroupNorm(32, TEXTURE_CHANNELS), nn.SiLU(inplace=True), nn.Conv2d(TEXTURE_CHANNELS, TEXTURE_DIM, 1)
        )

    def forward(self, texture):
        ''' Encode high-frequency residuals into local texture features. '''
        return self.layers(texture)

class TextureCodebook(nn.Module):
    ''' Quantize selected texture features into a learned discrete vocabulary. '''

    def __init__(self):
        ''' Build the learned texture vocabulary. '''
        super().__init__()
        self.embedding = nn.Embedding(TEXTURE_CODEBOOK_SIZE, TEXTURE_DIM)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def nearest_indices(self, features):
        ''' Find the nearest learned texture token for selected feature vectors. '''
        with torch.autocast(device_type=features.device.type, enabled=False):
            flattened = features.detach().float()
            embedding = self.embedding.weight.detach().float()
            distances = flattened.square().sum(dim=1, keepdim=True) + embedding.square().sum(dim=1) - 2.0 * flattened @ embedding.t()
            indices = distances.argmin(dim=1)

        return indices

    def selected_features(self, features, mask):
        ''' Gather selected feature vectors in stable row-major order. '''
        flattened = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        return flattened, flattened[mask.reshape(-1)]

    def encode(self, features, mask):
        ''' Convert selected texture features into compact codebook indices. '''
        _, selected = self.selected_features(features, mask)
        indices = self.nearest_indices(selected)
        return indices.view(features.shape[0], -1)

    def decode(self, mask, indices):
        ''' Expand sparse texture indices into a dense feature map with empty gaps. '''
        batch_size, height, width = mask.shape
        flattened = self.embedding.weight.new_zeros((batch_size * height * width, TEXTURE_DIM))
        flattened[mask.reshape(-1)] = self.embedding(indices.reshape(-1).long())
        return flattened.view(batch_size, height, width, TEXTURE_DIM).permute(0, 3, 1, 2).contiguous()

    def forward(self, features, mask):
        ''' Quantize selected texture features with straight-through VQ training. '''
        flattened, selected = self.selected_features(features, mask)
        indices = self.nearest_indices(selected)
        quantized_selected = self.embedding(indices)
        codebook_loss = F.mse_loss(quantized_selected, selected.detach())
        commitment_loss = F.mse_loss(selected, quantized_selected.detach())
        loss = codebook_loss + TEXTURE_COMMITMENT * commitment_loss
        quantized_selected = selected + (quantized_selected - selected).detach()
        quantized = torch.zeros_like(flattened)
        quantized[mask.reshape(-1)] = quantized_selected
        quantized = quantized.view(features.shape[0], features.shape[2], features.shape[3], features.shape[1]).permute(0, 3, 1, 2).contiguous()
        return quantized, indices.view(features.shape[0], -1), loss

class GenerativeDecoder(nn.Module):
    ''' Reconstruct structure faithfully while synthesizing missing fine detail. '''

    def __init__(self):
        ''' Build the structure-conditioned texture generator. '''
        super().__init__()
        self.structure_projection = nn.Sequential(
            nn.Conv2d(LATENT_CHANNELS, 128, 3, padding=1), ResidualBlock(128),
            nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1), ResidualBlock(128)
        )
        self.texture_projection = nn.Sequential(
            nn.Conv2d(TEXTURE_DIM + 1, 64, 3, padding=1), ResidualBlock(64)
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(192, 128, 3, padding=1), ResidualBlock(128), ResidualBlock(128)
        )
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(128, 96, 4, stride=2, padding=1), ResidualBlock(96),
            nn.ConvTranspose2d(96, 64, 4, stride=2, padding=1),
            nn.GroupNorm(32, 64), nn.SiLU(inplace=True), nn.Conv2d(64, 12, 3, padding=1), nn.PixelShuffle(2)
        )

    def forward(self, structure, texture, texture_mask):
        ''' Fuse compressed structure and sparse texture cues into an RGB reconstruction. '''
        structure = self.structure_projection(structure)
        texture = torch.cat([texture, texture_mask.unsqueeze(1).to(texture.dtype)], dim=1)
        texture = self.texture_projection(texture)
        features = self.fusion(torch.cat([structure, texture], dim=1))
        return self.generator(features)

class PatchDiscriminator(nn.Module):
    ''' Judge local image patches to train the decoder toward sharp plausible detail. '''

    def __init__(self):
        ''' Build a lightweight training-only patch discriminator. '''
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.GroupNorm(8, 64), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.GroupNorm(16, 128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.GroupNorm(32, 256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 1, 3, padding=1)
        )

    def forward(self, image):
        ''' Score local patches as real or reconstructed. '''
        return self.layers(image)

class ImageCodec(CompressionModel):
    ''' Combine structure compression, sparse texture tokens, and generative reconstruction. '''

    def __init__(self):
        ''' Build the complete image codec. '''
        super().__init__()
        self.encoder = StructureEncoder()
        self.texture_encoder = TextureEncoder()
        self.texture_codebook = TextureCodebook()
        self.decoder = GenerativeDecoder()
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
        ''' Run the differentiable two-stream training path. '''
        latent = self.encoder(image)
        texture_target = extract_texture(image)
        texture_mask = select_texture_mask(texture_target)
        texture_features = self.texture_encoder(texture_target)
        texture_quantized, texture_indices, vq_loss = self.texture_codebook(texture_features, texture_mask)
        hyper_latent = self.hyper_encoder(torch.abs(latent))

        with torch.autocast(device_type=image.device.type, enabled=False):
            hyper_latent_hat, hyper_likelihoods = self.entropy_bottleneck(hyper_latent.float())

        scales = self.hyper_decoder(hyper_latent_hat)

        with torch.autocast(device_type=image.device.type, enabled=False):
            latent_hat, latent_likelihoods = self.gaussian_conditional(latent.float(), scales.float())
            pixels = image.shape[0] * image.shape[2] * image.shape[3]
            structure_bpp = -(torch.log2(latent_likelihoods).sum() + torch.log2(hyper_likelihoods).sum()) / pixels
            texture_bpp = image.new_tensor((texture_mask.numel() + texture_indices.numel() * 8.0) / pixels)

        reconstruction = torch.sigmoid(self.decoder(latent_hat, texture_quantized, texture_mask))
        return reconstruction, structure_bpp, texture_bpp, vq_loss, texture_target

    @torch.inference_mode()
    def compress(self, image):
        ''' Compress one image into entropy streams and sparse texture tokens. '''
        latent = self.encoder(image)
        texture_target = extract_texture(image)
        texture_mask = select_texture_mask(texture_target)
        texture_features = self.texture_encoder(texture_target)
        texture_indices = self.texture_codebook.encode(texture_features, texture_mask)
        hyper_latent = self.hyper_encoder(torch.abs(latent))
        hyper_strings = self.entropy_bottleneck.compress(hyper_latent)
        hyper_latent_hat = self.entropy_bottleneck.decompress(hyper_strings, hyper_latent.shape[-2:])
        scales = self.hyper_decoder(hyper_latent_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        latent_strings = self.gaussian_conditional.compress(latent, indexes)
        return latent_strings[0], hyper_strings[0], texture_mask[0], texture_indices[0]

    @torch.inference_mode()
    def decompress(self, latent_stream, hyper_stream, texture_mask, texture_indices, shape):
        ''' Reconstruct one image from structure streams and sparse texture tokens. '''
        hyper_latent_hat = self.entropy_bottleneck.decompress([hyper_stream], shape)
        scales = self.hyper_decoder(hyper_latent_hat)
        indexes = self.gaussian_conditional.build_indexes(scales)
        latent_hat = self.gaussian_conditional.decompress([latent_stream], indexes, hyper_latent_hat.dtype)
        device = self.texture_codebook.embedding.weight.device
        texture_mask = texture_mask.to(device=device, dtype=torch.bool).unsqueeze(0)
        texture_indices = texture_indices.to(device=device, dtype=torch.long).unsqueeze(0)
        texture_quantized = self.texture_codebook.decode(texture_mask, texture_indices)
        return torch.sigmoid(self.decoder(latent_hat, texture_quantized, texture_mask))
