# Neural Image Codec

A learned image compression project built with a custom convolutional autoencoder and scale hyperprior. Images are compressed into a compact `.nic` format and reconstructed using the trained decoder.

## Training

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py
```

Set `DATA_DIR` at the top of `train.py`, then run this command to train across two GPUs using BF16 mixed precision. Training builds a deterministic uint8 RAM cache from the source image folder and stops automatically after the configured loss target is sustained.

## Inference

```bash
python infer.py
```

The CustomTkinter interface loads the newest checkpoint automatically when one is available. Open a source image, click **Compress** to save it under `compressed_images/`, then click **Decode** to read that `.nic` file back from disk and compare the reconstruction side by side.

On KDE, the image and checkpoint pickers use `kdialog` when it is installed. The standard Tk file picker is used as a fallback when KDialog is unavailable.
