import torch

from codec import decode_latents, encode_latents
from model import ImageCodec, count_parameters


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImageCodec().eval()
    model.encoder.to(device)
    model.decoder.to(device)
    image = torch.rand(1, 3, 64, 64, device=device)

    z_stream, y_stream = encode_latents(model, image)
    reconstruction = decode_latents(model, z_stream, y_stream, 64, 64, device)

    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"z bytes: {len(z_stream)}")
    print(f"y bytes: {len(y_stream)}")
    print(f"output: {tuple(reconstruction.shape)}")
    print(f"finite: {torch.isfinite(reconstruction).all().item()}")


if __name__ == "__main__":
    main()
