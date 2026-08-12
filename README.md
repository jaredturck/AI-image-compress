# Neural Image Codec

A learned image compression project built with a custom convolutional autoencoder and scale hyperprior. Images are compressed into a compact `.nic` format and reconstructed using the trained decoder.

## Training

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py
```

Set `DATA_DIR` at the top of `train.py`, then run this command to train across two GPUs using BF16 mixed precision.

## Inference

```bash
python infer.py
```

Starts the inference interface for compressing images to `.nic` files and reconstructing them back into images.
