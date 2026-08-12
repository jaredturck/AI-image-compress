import torch

from model import ImageCodec, count_parameters


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ImageCodec().eval()
    model.update(force=True)
    model.to(device)
    image = torch.rand(1, 3, 64, 64, device=device)

    packed = model.compress(image)
    reconstruction = model.decompress(packed["strings"], packed["shape"])

    print(f"parameters: {count_parameters(model) / 1e6:.2f}M")
    print(f"y bytes: {len(packed['strings'][0][0])}")
    print(f"z bytes: {len(packed['strings'][1][0])}")
    print(f"output: {tuple(reconstruction.shape)}")
    print(f"finite: {torch.isfinite(reconstruction).all().item()}")


if __name__ == "__main__":
    main()
