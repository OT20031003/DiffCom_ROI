import argparse
import os
import random
from typing import Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from _djscc.network import ADJSCC
from _djscc.standalone_utils import (
    ImportanceImageDataset,
    RunningAverages,
    batch_psnr,
    effective_alpha,
    resolve_split_dirs,
    sample_snr_db,
    set_seed,
    weighted_recon_loss,
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
    parser.add_argument("--alpha", type=float, default=2.0, help="Importance-region loss weight.")
    parser.add_argument("--beta", type=float, default=1.0, help="Background-region loss weight.")
    parser.add_argument(
        "--alpha-low-snr-threshold",
        type=float,
        default=None,
        help="If set, increase alpha below this SNR threshold.",
    )
    parser.add_argument(
        "--alpha-low-snr-gain",
        type=float,
        default=0.0,
        help="Gain for alpha boost when SNR is below threshold.",
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


def run_epoch(model, loader, optimizer, device, args, rng, train: bool):
    model.train(mode=train)
    averages = RunningAverages()
    use_mse = args.loss_type == "mse"

    for step, (images, importance, _names) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        importance = importance.to(device, non_blocking=True)

        snr_db = sample_snr_db(args.snr, args.snr_range, rng)
        alpha_eff = effective_alpha(
            args.alpha,
            snr_db,
            low_snr_threshold=args.alpha_low_snr_threshold,
            low_snr_gain=args.alpha_low_snr_gain,
        )

        with torch.set_grad_enabled(train):
            # Importance is injected encoder-side only. Decoder receives latent + SNR, not importance.
            recon = model(
                images,
                given_SNR=snr_db,
                importance_map=None if args.disable_importance_gating else importance,
            )
            loss, l_roi, l_bg = weighted_recon_loss(
                images, recon, importance, alpha=alpha_eff, beta=args.beta, use_mse=use_mse
            )
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        psnr_value = batch_psnr(images, recon).mean().item()
        batch_size = images.size(0)
        averages.update(
            loss=float(loss.item()),
            l_roi=float(l_roi.item()),
            l_bg=float(l_bg.item()),
            psnr=psnr_value,
            n=batch_size,
        )

        if train and step % args.log_interval == 0:
            current = averages.compute()
            print(
                f"[train] step={step}/{len(loader)} snr={snr_db:.2f}dB "
                f"alpha_eff={alpha_eff:.4f} loss={current['loss']:.6f} psnr={current['psnr']:.2f}"
            )

    return averages.compute()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    set_seed(args.seed)
    device = resolve_device(args.device)
    rng = random.Random(args.seed)

    train_images_dir, train_importance_dir = split_dirs(
        args.train_dir, args.train_images_dir, args.train_importance_dir
    )
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
        train_stats = run_epoch(model, train_loader, optimizer, device, args, rng, train=True)
        print(
            f"[epoch {epoch}] train loss={train_stats['loss']:.6f} "
            f"l_roi={train_stats['l_roi']:.6f} l_bg={train_stats['l_bg']:.6f} psnr={train_stats['psnr']:.2f}"
        )

        val_stats = None
        if val_loader is not None:
            with torch.no_grad():
                val_stats = run_epoch(model, val_loader, optimizer, device, args, rng, train=False)
            print(
                f"[epoch {epoch}] val   loss={val_stats['loss']:.6f} "
                f"l_roi={val_stats['l_roi']:.6f} l_bg={val_stats['l_bg']:.6f} psnr={val_stats['psnr']:.2f}"
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
