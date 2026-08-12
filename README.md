# Neural Image Codec

A learned image compression project with a structure hyperprior, sparse VQ texture stream, and generative decoder. Images are compressed into a compact `.nic` format that carries coarse image structure plus selected texture cues for sharp perceptual reconstruction.

## Training

```bash
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 train.py
```

Set `DATA_DIR` at the top of `train.py`, then run this command to train across two GPUs using BF16 mixed precision. Training builds a deterministic uint8 RAM cache from the source image folder and stops automatically after the configured loss plateau is sustained.

## Inference

```bash
python infer.py
```

The CustomTkinter interface loads the newest checkpoint automatically when one is available. Open a source image, click **Compress** to save it under `compressed_images/`, then click **Decode** to read that `.nic` file back from disk and compare the reconstruction side by side.

The reconstruction panel includes a **Neural / JPEG** toggle. The JPEG comparison is encoded as close as possible to the `.nic` file size so both codecs can be inspected at approximately the same bitrate with the same zoom, pan, and quality filters.

On KDE, the image and checkpoint pickers use `kdialog` when it is installed. The standard Tk file picker is used as a fallback when KDialog is unavailable.
