import math

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed
from torchvision.utils import save_image

import config
from data import build_dataloaders
from model import ImageCodec, count_parameters


def evaluate(model, validation_loader, accelerator, max_batches):
    model.eval()
    total_bpp = torch.zeros((), device=accelerator.device)
    total_mse = torch.zeros((), device=accelerator.device)
    count = 0

    with torch.no_grad():
        for images in validation_loader:
            _, bpp, mse = model(images)
            total_bpp += bpp
            total_mse += mse
            count += 1
            if count >= max_batches:
                break

    metrics = torch.stack([total_bpp / count, total_mse / count])
    metrics = accelerator.reduce(metrics, reduction="mean")
    model.train()
    return metrics[0].item(), metrics[1].item()


def save_checkpoint(model, accelerator, step):
    accelerator.wait_for_everyone()
    if not accelerator.is_main_process:
        return

    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(model)
    checkpoint = {
        "model": unwrapped.state_dict(),
        "step": step,
        "parameters": count_parameters(unwrapped),
    }
    accelerator.save(checkpoint, config.CHECKPOINT_PATH)


def save_preview(images, reconstruction, step, accelerator):
    if not accelerator.is_main_process:
        return
    preview_dir = config.CHECKPOINT_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview = torch.cat([images[:4].float(), reconstruction[:4].float()], dim=0)
    save_image(preview, preview_dir / f"step_{step:07d}.png", nrow=4)


def main():
    accelerator = Accelerator(mixed_precision="bf16")
    set_seed(config.SEED)

    if not config.DATA_DIR.exists():
        if accelerator.is_main_process:
            print(f"Set DATA_DIR in config.py first. Current value: {config.DATA_DIR}")
        return

    train_loader, validation_loader, image_count = build_dataloaders(
        config.DATA_DIR,
        config.IMAGE_SIZE,
        config.BATCH_SIZE,
        config.NUM_WORKERS,
        config.VALIDATION_SPLIT,
        config.SEED,
    )

    if train_loader is None:
        if accelerator.is_main_process:
            print("The image folder needs at least two JPG, PNG, or WEBP images.")
        return

    model = ImageCodec()
    main_parameters = [parameter for name, parameter in model.named_parameters() if not name.endswith(".quantiles")]
    aux_parameters = [parameter for name, parameter in model.named_parameters() if name.endswith(".quantiles")]
    optimizer = torch.optim.Adam(main_parameters, lr=config.LEARNING_RATE)
    aux_optimizer = torch.optim.Adam(aux_parameters, lr=config.AUX_LEARNING_RATE)

    model, optimizer, aux_optimizer, train_loader, validation_loader = accelerator.prepare(
        model,
        optimizer,
        aux_optimizer,
        train_loader,
        validation_loader,
    )
    optimizer_parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]

    if accelerator.is_main_process:
        parameters = count_parameters(accelerator.unwrap_model(model))
        print(f"images: {image_count:,}")
        print(f"parameters: {parameters / 1e6:.2f}M")
        print(f"devices: {accelerator.num_processes}")
        print(f"mixed precision: {accelerator.mixed_precision}")

    model.train()
    step = 0

    while step < config.MAX_STEPS:
        for images in train_loader:
            optimizer.zero_grad(set_to_none=True)
            reconstruction, bpp, mse = model(images)
            loss = bpp + config.LAMBDA * (255.0 ** 2) * mse
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(optimizer_parameters, config.GRAD_CLIP)
            optimizer.step()

            aux_optimizer.zero_grad(set_to_none=True)
            aux_loss = accelerator.unwrap_model(model).aux_loss()
            accelerator.backward(aux_loss)
            aux_optimizer.step()
            step += 1

            if step % config.LOG_EVERY == 0:
                metrics = torch.stack([loss.detach(), bpp.detach(), mse.detach()])
                metrics = accelerator.reduce(metrics, reduction="mean")
                psnr = -10.0 * math.log10(max(metrics[2].item(), 1e-12))
                if accelerator.is_main_process:
                    print(
                        f"step {step:7d} | loss {metrics[0].item():.4f} | "
                        f"bpp {metrics[1].item():.4f} | psnr {psnr:.2f} dB"
                    )

            if step % config.PREVIEW_EVERY == 0:
                save_preview(images, reconstruction, step, accelerator)

            if step % config.SAVE_EVERY == 0:
                validation_bpp, validation_mse = evaluate(
                    model,
                    validation_loader,
                    accelerator,
                    config.VALIDATION_BATCHES,
                )
                validation_psnr = -10.0 * math.log10(max(validation_mse, 1e-12))
                if accelerator.is_main_process:
                    print(f"validation | bpp {validation_bpp:.4f} | psnr {validation_psnr:.2f} dB")
                save_checkpoint(model, accelerator, step)

            if step >= config.MAX_STEPS:
                break

    save_checkpoint(model, accelerator, step)


if __name__ == "__main__":
    main()
