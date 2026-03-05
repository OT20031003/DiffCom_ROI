import argparse
import os
import random
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from _djscc.network import ADJSCC
from _djscc.standalone_utils import (
    ImportanceImageDataset,
    RunningAverages,
    batch_psnr,
    pearson_corr_loss_map,
    reconstruction_loss,
    resolve_split_dirs,
    sample_snr_db,
    set_seed,
)
from channel.channel import Channel


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone training for importance-aware DeepJSCC.")
    parser.add_argument("--train-dir", type=str, default="data/train", help="Split dir with images/importance.")
    parser.add_argument("--val-dir", type=str, default="data/val", help="Validation split dir with images/importance.")
    parser.add_argument("--train-images-dir", type=str, default=None, help="Optional override for train images dir.")
    parser.add_argument(
        "--train-importance-dir", type=str, default=None, help="Optional override for train importance dir."
    )
    parser.add_argument("--val-images-dir", type=str, default=None, help="Optional override for val images dir.")
    parser.add_argument("--val-importance-dir", type=str, default=None, help="Optional override for val importance dir.")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--channel-num", type=int, default=2)
    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--snr", type=float, default=None, help="Fixed training SNR in dB.")
    parser.add_argument("--snr-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument(
        "--lambda-corr",
        type=float,
        default=0.01,
        help="Weight of Pearson-correlation regularization term.",
    )
    parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--save-dir", type=str, default="results/djscc_train")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=str, default=None, help="Resume checkpoint path.")
    parser.add_argument(
        "--init-ckpt", type=str, default=None, help="Optional init checkpoint (supports legacy Encoder/Decoder keys)."
    )
    parser.add_argument("--disable-importance-gating", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


def split_dirs(
    split_dir: str,
    images_override: Optional[str],
    importance_override: Optional[str],
) -> Tuple[str, Optional[str]]:
    images_dir, importance_dir = resolve_split_dirs(split_dir)
    if images_override is not None:
        images_dir = images_override
    if importance_override is not None:
        importance_dir = importance_override
    return images_dir, importance_dir


def unwrap_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        if "model_state" in checkpoint_obj:
            return checkpoint_obj["model_state"]
        if "state_dict" in checkpoint_obj:
            return checkpoint_obj["state_dict"]
    return checkpoint_obj


def remap_legacy_keys(state_dict):
    remapped = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        if new_key.startswith("distortion_loss"):
            continue
        if new_key.startswith("Encoder"):
            new_key = new_key.replace("Encoder", "jscc_encoder", 1)
        elif new_key.startswith("Decoder"):
            new_key = new_key.replace("Decoder", "jscc_decoder", 1)
        remapped[new_key] = value
    return remapped


def load_model_weights(model: nn.Module, ckpt_path: str, device: torch.device, strict: bool = False):
    ckpt_obj = torch.load(ckpt_path, map_location=device)
    state_dict = remap_legacy_keys(unwrap_state_dict(ckpt_obj))
    load_result = model.load_state_dict(state_dict, strict=strict)
    return ckpt_obj, load_result


def run_epoch(model, loader, optimizer, device, args, rng, train: bool, epoch: int):
    model.train(mode=train)
    averages = RunningAverages()
    use_mse = args.loss_type == "mse"
    sample_count = 0
    recon_loss_sum = 0.0
    corr_loss_sum = 0.0
    total_loss_sum = 0.0
    mode_name = "train" if train else "val"
    pbar = tqdm(
        loader,
        total=len(loader),
        desc=f"{mode_name} epoch {epoch}/{args.epochs}",
        dynamic_ncols=True,
        leave=False,
    )

    for step, (images, importance, _names) in enumerate(pbar, start=1):
        images = images.to(device, non_blocking=True)
        importance = importance.to(device, non_blocking=True)

        snr_db = sample_snr_db(args.snr, args.snr_range, rng)

        with torch.set_grad_enabled(train):
            # Importance is injected encoder-side only. Decoder receives latent + SNR, not importance.
            recon = model(
                images,
                given_SNR=snr_db,
                importance_map=None if args.disable_importance_gating else importance,
            )
            recon_loss = reconstruction_loss(images, recon, use_mse=use_mse)
            err_map = torch.mean(torch.abs(images - recon), dim=1, keepdim=True)
            corr_loss = pearson_corr_loss_map(importance, err_map)
            total_loss = recon_loss + args.lambda_corr * corr_loss
            if train:
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                optimizer.step()

        psnr_value = batch_psnr(images, recon).mean().item()
        batch_size = images.size(0)
        sample_count += batch_size
        recon_loss_sum += float(recon_loss.detach().item()) * batch_size
        corr_loss_sum += float(corr_loss.detach().item()) * batch_size
        total_loss_sum += float(total_loss.detach().item()) * batch_size

        averages.update(
            loss=float(total_loss.detach().item()),
            recon_loss=float(recon_loss.detach().item()),
            psnr=psnr_value,
            n=batch_size,
        )

        current = averages.compute()
        running_recon_loss = recon_loss_sum / max(sample_count, 1)
        running_corr_loss = corr_loss_sum / max(sample_count, 1)
        running_total_loss = total_loss_sum / max(sample_count, 1)
        pbar.set_postfix(
            recon=f"{running_recon_loss:.4f}",
            corr_loss=f"{running_corr_loss:.4f}",
            total=f"{running_total_loss:.4f}",
            psnr=f"{current['psnr']:.2f}",
            snr=f"{snr_db:.2f}dB",
        )

    result = averages.compute()
    result["corr"] = corr_loss_sum / max(sample_count, 1)
    result["recon_loss"] = recon_loss_sum / max(sample_count, 1)
    result["corr_loss"] = corr_loss_sum / max(sample_count, 1)
    result["total_loss"] = total_loss_sum / max(sample_count, 1)
    return result


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    rng = random.Random(args.seed)

    train_images_dir, train_importance_dir = split_dirs(
        args.train_dir, args.train_images_dir, args.train_importance_dir
    )
    # Importance maps are expected as pre-generated PNGs in the importance directory.
    # In this workflow they can be synthetic random structured priors, not semantic masks.
    train_dataset = ImportanceImageDataset(
        images_dir=train_images_dir,
        importance_dir=train_importance_dir,
        image_size=args.image_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    val_loader = None
    val_images_dir, val_importance_dir = split_dirs(args.val_dir, args.val_images_dir, args.val_importance_dir)
    if os.path.isdir(val_images_dir):
        try:
            val_dataset = ImportanceImageDataset(
                images_dir=val_images_dir,
                importance_dir=val_importance_dir,
                image_size=args.image_size,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=(device.type == "cuda"),
                drop_last=False,
            )
        except ValueError:
            val_loader = None

    init_snr = args.snr if args.snr is not None else 10.0
    channel = Channel(args.channel_type, init_snr, logger=None, device=device, rescale=True)
    model = ADJSCC(
        C=args.channel_num,
        channel=channel,
        device=device,
        use_importance=(not args.disable_importance_gating),
        multiple_snr=None if args.snr_range is None else [float(args.snr_range[0]), float(args.snr_range[1])],
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_epoch = 1
    best_val_loss = float("inf")

    if args.init_ckpt is not None:
        _, load_result = load_model_weights(model, args.init_ckpt, device=device, strict=False)
        print(f"Loaded init checkpoint from {args.init_ckpt}: {load_result}")

    if args.resume is not None:
        ckpt_obj, load_result = load_model_weights(model, args.resume, device=device, strict=False)
        print(f"Resumed model from {args.resume}: {load_result}")
        if isinstance(ckpt_obj, dict) and "optimizer_state" in ckpt_obj:
            optimizer.load_state_dict(ckpt_obj["optimizer_state"])
        if isinstance(ckpt_obj, dict) and "epoch" in ckpt_obj:
            start_epoch = int(ckpt_obj["epoch"]) + 1
        if isinstance(ckpt_obj, dict) and "best_val_loss" in ckpt_obj:
            best_val_loss = float(ckpt_obj["best_val_loss"])

    for epoch in range(start_epoch, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, optimizer, device, args, rng, train=True, epoch=epoch)
        print(
            f"[epoch {epoch}] train recon_loss={train_stats['recon_loss']:.6f} "
            f"corr_loss={train_stats['corr_loss']:.6f} total_loss={train_stats['total_loss']:.6f} "
            f"psnr={train_stats['psnr']:.2f} corr={train_stats['corr']:.4f}"
        )

        val_stats = None
        if val_loader is not None:
            with torch.no_grad():
                val_stats = run_epoch(model, val_loader, optimizer, device, args, rng, train=False, epoch=epoch)
            print(
                f"[epoch {epoch}] val   recon_loss={val_stats['recon_loss']:.6f} "
                f"corr_loss={val_stats['corr_loss']:.6f} total_loss={val_stats['total_loss']:.6f} "
                f"psnr={val_stats['psnr']:.2f} corr={val_stats['corr']:.4f}"
            )

        save_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val_loss,
        }

        if epoch % args.save_every == 0:
            latest_path = os.path.join(args.save_dir, "latest.pth")
            torch.save(save_payload, latest_path)

        target_loss = val_stats["loss"] if val_stats is not None else train_stats["loss"]
        if target_loss < best_val_loss:
            best_val_loss = float(target_loss)
            save_payload["best_val_loss"] = best_val_loss
            best_path = os.path.join(args.save_dir, "best.pth")
            torch.save(save_payload, best_path)
            print(f"[epoch {epoch}] saved best checkpoint: {best_path}")

    final_path = os.path.join(args.save_dir, "final.pth")
    torch.save(
        {
            "epoch": args.epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args": vars(args),
            "best_val_loss": best_val_loss,
        },
        final_path,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
