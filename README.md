# Neural Image Codec

A from-scratch learned image compression experiment built around PyTorch. The learned architecture is defined directly in `model.py`; there are no imported pretrained models, codec networks, entropy-model packages, or CompressAI components.

The only non-PyTorch training dependency is Hugging Face Accelerate for two-GPU data-parallel training. Attention inside the image transform uses PyTorch's fused FlashAttention SDPA backend on CUDA BF16/FP16 tensors. Attention is applied in fixed 16x16 spatial windows so inference cost remains reasonable on images larger than the 256x256 training crop.

## What the codec actually stores

The encoder does **not** dump floating-point latents to disk.

1. The image encoder produces a main latent tensor `y` at 1/16 spatial resolution.
2. A small hyper-encoder produces `z` at 1/64 spatial resolution.
3. `z` is rounded to integers and entropy-coded first.
4. Decoded `z` predicts image-specific mean/scale statistics for `y`.
5. `y` is encoded in uneven channel groups: `16, 16, 32, 64, 128`.
6. Previously encoded integer residual groups feed a small context network, improving the probability estimates for later groups.
7. Each integer symbol is range-coded using a Gaussian CDF selected from 64 logarithmically spaced scale levels.
8. Training explicitly minimizes estimated bitrate plus reconstruction distortion.

The `.nic` file contains only a small header plus the range-coded `z` and `y` streams. The exact trained checkpoint is the decoder/codebook and is intentionally not duplicated inside each image file.

## Architecture

Approximately **60.87M parameters** total.

### Main encoder

```text
RGB 256x256
  -> Conv 3->128, stride 2 + GDN + ResBlock
  -> Conv 128->192, stride 2 + GDN + 2 ResBlocks
  -> Conv 192->256, stride 2 + GDN + 3 ResBlocks + windowed FlashAttention
  -> Conv 256->320, stride 2 + GDN + 6 ResBlocks + windowed FlashAttention
  -> Conv 320->256
  -> y: 256x16x16
```

### Hyperprior

```text
abs(y)
  -> Conv 256->192
  -> Conv 192->160, stride 2
  -> Conv 160->128, stride 2
  -> z: 128x4x4

z
  -> upsample + Conv 128->160
  -> upsample + Conv 160->192
  -> Conv 192->512
  -> per-location mean and scale for y
```

### Channel context

A transparent three-convolution PyTorch network reads already encoded integer residual groups and predicts corrections to the mean/scale estimates for later groups.

### Main decoder

```text
y_hat: 256x16x16
  -> Conv 256->320 + windowed FlashAttention + 12 ResBlocks
  -> upsample + Conv 320->256 + IGDN + 3 ResBlocks + windowed FlashAttention
  -> upsample + Conv 256->192 + IGDN + 2 ResBlocks
  -> upsample + Conv 192->128 + IGDN + ResBlock
  -> upsample + Conv 128->64 + IGDN + ResBlock
  -> Conv 64->3 + sigmoid
```

The large number of low-resolution residual blocks gives the decoder useful capacity without paying for that depth at full image resolution.

## Training objective

```text
loss = estimated_bpp + lambda * 255^2 * MSE
```

`estimated_bpp` is computed from the same quantized Gaussian probability model used by the real arithmetic coder. Quantization uses uniform `[-0.5, 0.5]` noise during training and integer rounding during compression.

## Image preparation

`data.py` recursively finds JPG/JPEG/PNG/WEBP files and uses TorchVision only:

```text
decode_image(..., mode="RGB")
  -> RandomResizedCrop(256x256, square crop, antialias=True)
  -> RandomHorizontalFlip
  -> float32 scaled to [0, 1]
```

No ImageNet mean/std normalization is used because the model reconstructs RGB values directly.

Validation uses aspect-preserving resize, center crop, and `[0, 1]` conversion.

## Setup

Create an environment and install the dependencies:

```bash
pip install -r requirements.txt
```

On Arch Linux, Tkinter normally comes from the `tk` package if it is not already installed.

Edit `DATA_DIR` in `config.py` so it points at your image folder.

## Train on both RTX 3090s

The included Accelerate config is set for two local GPUs and BF16:

```bash
accelerate launch --config_file accelerate_config.yaml train.py
```

Equivalent explicit launch:

```bash
accelerate launch --multi_gpu --mixed_precision=bf16 --num_processes=2 train.py
```

Checkpoints are written to:

```text
checkpoints/latest.pt
```

Reconstruction previews are written to `checkpoints/previews/`. Each preview places original images first and reconstructions second.

The main knobs are all in `config.py`, especially:

```text
BATCH_SIZE
LEARNING_RATE
LAMBDA
MAX_STEPS
```

For two 24 GB RTX 3090s, the starting per-process batch size is 8. Reduce it if your particular PyTorch/CUDA build runs out of memory.

## GUI

After training:

```bash
python gui.py
```

1. Load `checkpoints/latest.pt`.
2. Choose **Compress image** and save a `.nic` file.
3. Choose **Decompress .nic** and save the reconstructed PNG.

The GUI reports the actual compressed byte count and bits per pixel.

A `.nic` file is tied to the exact checkpoint that created it. A 16-byte SHA-256 checkpoint identifier is stored in the file header so the GUI refuses to decode with the wrong network weights.

## Programmatic inference

`infer.py` exposes:

```text
load_model(...)
compress_image(...)
decompress_image(...)
```

The codec pads arbitrary input dimensions to a multiple of 64 for the network, stores the original dimensions in the header, and removes the padding after decompression. The main encoder/decoder run on the selected GPU; the small entropy/hyper/context path and arithmetic coder run in deterministic float32 on CPU so the bitstream does not depend on BF16 FlashAttention numerics.

## Smoke test

```bash
python smoke_test.py
```

This uses a randomly initialized network and verifies the full latent entropy-code/decode path. It does not test image quality; useful reconstructions require training.

## Project files

```text
model.py                 neural architecture, quantization, rate estimate
codec.py                 arithmetic coder, Gaussian CDFs, .nic bitstream
data.py                  image discovery and TorchVision preprocessing
config.py                training settings
train.py                 Accelerate two-GPU training loop
infer.py                 compression/decompression API
gui.py                   minimal Tk GUI
smoke_test.py            end-to-end codec sanity check
accelerate_config.yaml   two-GPU BF16 launcher config
requirements.txt         dependencies
```

## Research basis

The design intentionally uses a small subset of ideas that materially affect file size while remaining understandable from the source:

- Ballé et al., *Variational Image Compression with a Scale Hyperprior* — side information/hyperprior and rate-distortion training.
- Minnen et al., *Joint Autoregressive and Hierarchical Priors for Learned Image Compression* — learned conditional entropy models.
- He et al., *ELIC* — uneven channel grouping and practical context modelling.
- Dao et al., *FlashAttention / FlashAttention-2* — fused exact attention; this project reaches it through PyTorch SDPA rather than importing a transformer model.

The project deliberately does not implement a pixel-by-pixel autoregressive context model: that can improve rate-distortion performance but makes decoding serial and much slower. The channel-group context is the compromise used here.
