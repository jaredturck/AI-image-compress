# Neural Image Codec

A small learned image-compression experiment built around a custom convolutional autoencoder and the scale-hyperprior design from Ballé et al. (ICLR 2018).

The project deliberately keeps the learned image transforms in this repository while using CompressAI for standard compression plumbing: GDN, entropy models, CDF construction, and rANS coding. No pretrained or ready-made autoencoder is imported.

## Core pipeline

```text
image
  -> convolutional encoder
  -> main latent y
  -> quantization + entropy coding
  -> .nic file
  -> entropy decoding
  -> convolutional decoder
  -> reconstructed image
```

A small hyperprior provides side information used to predict the probability scale of each `y` value:

```text
                         -> hyper encoder -> z -> entropy bottleneck -> z_hat
                        /                                      |
image -> encoder -> y --                                       v
                        \-------------------------- hyper decoder
                                                         |
                                                    scales for y
                                                         |
                                                Gaussian conditional
```

This keeps the useful 2018 hyperprior while removing the later attention, residual-stack, autoregressive context, and uneven channel-group machinery.

## Architecture

### Encoder

```text
RGB
  -> Conv 3 -> 128, stride 2 + GDN
  -> Conv 128 -> 128, stride 2 + GDN
  -> Conv 128 -> 128, stride 2 + GDN
  -> Conv 128 -> 192, stride 2
  -> y
```

### Decoder

```text
y_hat
  -> ConvTranspose 192 -> 128, stride 2 + IGDN
  -> ConvTranspose 128 -> 128, stride 2 + IGDN
  -> ConvTranspose 128 -> 128, stride 2 + IGDN
  -> ConvTranspose 128 -> 3, stride 2
  -> reconstructed RGB
```

### Hyperprior

```text
abs(y)
  -> Conv 192 -> 128
  -> Conv 128 -> 128, stride 2
  -> Conv 128 -> 128, stride 2
  -> z

z_hat
  -> ConvTranspose 128 -> 128, stride 2
  -> ConvTranspose 128 -> 128, stride 2
  -> Conv 128 -> 192
  -> scales for y
```

`z` is entropy-coded with `EntropyBottleneck`. The decoded `z_hat` predicts scales for `GaussianConditional`, which entropy-codes `y`.

## What CompressAI is used for

Only standard learned-compression components are imported:

```text
compressai.layers.GDN
compressai.entropy_models.EntropyBottleneck
compressai.entropy_models.GaussianConditional
compressai.models.CompressionModel
```

The encoder, decoder, hyper encoder, and hyper decoder are defined directly in `model.py` using PyTorch convolutional layers.

## Training objective

```text
loss = estimated_bpp + lambda * 255^2 * MSE
```

The main optimizer trains the transforms and entropy-model parameters. CompressAI's entropy bottleneck also has a small auxiliary loss for its quantile parameters, trained with a separate auxiliary optimizer.

## `.nic` files

The current format is `NIC2` and contains:

```text
header
  magic/version
  checkpoint identifier
  original width/height
  hyper-latent shape
  y/z stream lengths

compressed y stream
compressed z stream
```

The checkpoint itself is not stored in every image. A 16-byte SHA-256 checkpoint identifier ensures a `.nic` file is decoded with the same learned model that created it.

`NIC2` is intentionally incompatible with the previous experimental `NIC1` format.

## Setup

```bash
pip install -r requirements.txt
```

On Arch Linux, Tkinter normally comes from the `tk` package if it is not already installed.

Edit `DATA_DIR` in `config.py` so it points at your image folder.

## Training

The included Accelerate configuration uses two GPUs and BF16:

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

Checkpoints are written to:

```text
checkpoints/latest.pt
```

Reconstruction previews are written to:

```text
checkpoints/previews/
```

## GUI

After training:

```bash
python gui.py
```

Load the checkpoint, compress an image to `.nic`, or decompress a `.nic` file back to PNG.

## Smoke test

```bash
python smoke_test.py
```

The smoke test uses a randomly initialized model to exercise CompressAI's real entropy encode/decode path. It checks plumbing only; image quality requires training.

## Project files

```text
model.py                 custom convolutional codec + scale hyperprior
codec.py                 small NIC2 file container
config.py                training settings
data.py                  image dataset and preprocessing
train.py                 Accelerate training loop
infer.py                 checkpoint loading and image compression API
gui.py                   minimal Tk GUI
smoke_test.py            entropy-code/decode sanity check
accelerate_config.yaml   two-GPU BF16 launcher config
requirements.txt         dependencies
```

## Research basis

The baseline intentionally follows the simpler branch of learned image compression:

- Ballé, Laparra, Simoncelli — *End-to-end Optimized Image Compression* (ICLR 2017): convolutional analysis/synthesis transforms, GDN, differentiable quantization, and rate-distortion training.
- Ballé, Minnen, Singh, Hwang, Johnston — *Variational Image Compression with a Scale Hyperprior* (ICLR 2018): the `z` hyperprior and conditional scale model used here.

Later ideas such as autoregressive context models, uneven channel groups, deep residual transforms, and attention are intentionally not part of this baseline.
