import argparse
import os
import random
from typing import Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from _djscc.network import ADJSCC
from _djscc.standalone_utils import (
    ImportanceImageDataset,
    RunningAverages,
    batch_psnr,
    effective_alpha,
    resolve_split_dirs,
    sample_snr_db,
    weighted_recon_loss,
)
from channel.channel import Channel


def parse_args():
    parser = argparse.ArgumentParser(description="Standalone inference/testing for DeepJSCC.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path.")
    parser.add_argument("--input-dir", type=str, default="data/val", help="Input dir (or split dir with images/).")
    parser.add_argument("--images-dir", type=str, default=None, help="Optional override for image directory.")
    parser.add_argument("--importance-dir", type=str, default=None, help="Optional override for importance directory.")
    parser.add_argument("--output-dir", type=str, default="results/djscc_test")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--channel-num", type=int, default=2)
    parser.add_argument("--channel-type", type=str, default="awgn", choices=["awgn", "rayleigh"])
    parser.add_argument("--snr", type=float, default=10.0, help="Fixed SNR in dB.")
    parser.add_argument("--snr-range", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    parser.add_argument("--alpha", type=float, default=2.0, help="Importance-region loss weight for reporting.")
    parser.add_argument("--beta", type=float, default=1.0, help="Background-region loss weight for reporting.")
    parser.add_argument("--alpha-low-snr-threshold", type=float, default=None)
    parser.add_argument("--alpha-low-snr-gain", type=float, default=0.0)
    parser.add_argument("--loss-type", type=str, default="l1", choices=["l1", "mse"])
    parser.add_argument("--disable-importance-gating", action="store_true")
    parser.add_argument(
        "--report-correlation",
        action="store_true",
        help="Report Pearson correlation between importance map and reconstruction error map.",
    )
    parser.add_argument(
        "--print-per-image-correlation",
        action="store_true",
        help="Print per-image correlation values.",
    )
    parser.add_argument(
        "--correlation-eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon for Pearson correlation denominator.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_arg)


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


def split_input_dirs(
    input_dir: str,
    images_override: Optional[str],
    importance_override: Optional[str],
) -> Tuple[str, Optional[str]]:
    images_dir, importance_dir = resolve_split_dirs(input_dir)
    if images_override is not None:
        images_dir = images_override
    if importance_override is not None:
        importance_dir = importance_override
    return images_dir, importance_dir


def choose_snr(args, rng: random.Random) -> float:
    snr_range: Optional[Sequence[float]] = args.snr_range
    return sample_snr_db(args.snr, snr_range, rng)


def pearson_corr(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    a = a - a.mean()
    b = b - b.mean()
    denom = torch.sqrt(torch.sum(a * a) * torch.sum(b * b)) + eps
    if denom.item() <= eps:
        return 0.0
    return float((torch.sum(a * b) / denom).item())


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    rng = random.Random(args.seed)

    images_dir, importance_dir = split_input_dirs(args.input_dir, args.images_dir, args.importance_dir)
    dataset = ImportanceImageDataset(images_dir=images_dir, importance_dir=importance_dir, image_size=args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    channel = Channel(args.channel_type, args.snr, logger=None, device=device, rescale=True)
    model = ADJSCC(
        C=args.channel_num,
        channel=channel,
        device=device,
        use_importance=(not args.disable_importance_gating),
        multiple_snr=None if args.snr_range is None else [float(args.snr_range[0]), float(args.snr_range[1])],
    ).to(device)

    checkpoint_obj = torch.load(args.checkpoint, map_location=device)
    state_dict = remap_legacy_keys(unwrap_state_dict(checkpoint_obj))
    load_result = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded checkpoint {args.checkpoint}: {load_result}")

    use_mse = args.loss_type == "mse"
    averages = RunningAverages()
    corr_sum = 0.0
    corr_count = 0
    model.eval()
    with torch.no_grad():
        for images, importance, names in loader:
            images = images.to(device, non_blocking=True)
            importance = importance.to(device, non_blocking=True)
            snr_db = choose_snr(args, rng)
            alpha_eff = effective_alpha(
                args.alpha,
                snr_db,
                low_snr_threshold=args.alpha_low_snr_threshold,
                low_snr_gain=args.alpha_low_snr_gain,
            )

            # Importance maps are used only by the encoder. Decoder does not consume them explicitly.
            recon = model(
                images,
                given_SNR=snr_db,
                importance_map=None if args.disable_importance_gating else importance,
            )
            loss, l_roi, l_bg = weighted_recon_loss(
                images, recon, importance, alpha=alpha_eff, beta=args.beta, use_mse=use_mse
            )
            psnr = batch_psnr(images, recon).mean().item()
            averages.update(
                loss=float(loss.item()),
                l_roi=float(l_roi.item()),
                l_bg=float(l_bg.item()),
                psnr=psnr,
                n=images.size(0),
            )

            if args.report_correlation:
                err_map = torch.mean(torch.abs(images - recon), dim=1, keepdim=True)
                for i, name in enumerate(names):
                    corr = pearson_corr(importance[i], err_map[i], eps=args.correlation_eps)
                    corr_sum += corr
                    corr_count += 1
                    if args.print_per_image_correlation:
                        print(f"[corr] {name}: {corr:.6f}")

            recon = recon.clamp(0.0, 1.0)
            for i, name in enumerate(names):
                save_path = os.path.join(args.output_dir, name)
                save_image(recon[i], save_path)

    stats = averages.compute()
    print(
        f"Test complete | avg_loss={stats['loss']:.6f} "
        f"avg_l_roi={stats['l_roi']:.6f} avg_l_bg={stats['l_bg']:.6f} avg_psnr={stats['psnr']:.2f}"
    )
    if args.report_correlation:
        avg_corr = corr_sum / max(corr_count, 1)
        print(f"Pearson correlation (importance vs error map): avg={avg_corr:.6f}, samples={corr_count}")
    print(f"Saved reconstructions to: {args.output_dir}")


if __name__ == "__main__":
    main()
