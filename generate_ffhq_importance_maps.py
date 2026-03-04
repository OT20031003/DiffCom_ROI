#!/usr/bin/env python3
"""
Generate synthetic random structured importance-map PNGs.

This script intentionally does NOT use facial landmark/segmentation/recognition models.
Maps are random spatial priors (blobs, soft patches, stripes, optional center bias).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass
class Stats:
    total_images: int = 0
    generated: int = 0
    failed_to_read: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate random structured importance-map PNGs.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k",
        help="Input image directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/mnt/d/WSL_Work/diffcom/testsets/ffhq_train_70k_importance",
        help="Output directory for grayscale PNG maps.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Optional square resize before map generation. 0 keeps original image size.",
    )
    parser.add_argument("--background-weight", type=float, default=0.15, help="Base low importance in [0,1].")
    parser.add_argument("--min-blobs", type=int, default=3, help="Minimum number of smooth random blobs.")
    parser.add_argument("--max-blobs", type=int, default=8, help="Maximum number of smooth random blobs.")
    parser.add_argument(
        "--min-blob-radius-frac",
        type=float,
        default=0.04,
        help="Minimum blob radius as fraction of min(H, W).",
    )
    parser.add_argument(
        "--max-blob-radius-frac",
        type=float,
        default=0.22,
        help="Maximum blob radius as fraction of min(H, W).",
    )
    parser.add_argument("--center-prob", type=float, default=0.65, help="Probability of adding center-prior blob.")
    parser.add_argument("--patch-prob", type=float, default=0.50, help="Probability of adding soft rectangle patches.")
    parser.add_argument("--stripe-prob", type=float, default=0.35, help="Probability of adding soft stripe regions.")
    parser.add_argument("--max-patches", type=int, default=2, help="Maximum number of patch regions.")
    parser.add_argument("--max-stripes", type=int, default=2, help="Maximum number of stripe regions.")
    parser.add_argument("--blur-kernel", type=int, default=31, help="Final Gaussian blur kernel (odd integer).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--progress-interval", type=int, default=200, help="Print progress every N images.")
    return parser.parse_args()


def ensure_odd(x: int, minimum: int = 3) -> int:
    x = max(int(x), minimum)
    return x if x % 2 == 1 else x + 1


def gather_images(input_dir: Path, recursive: bool) -> List[Path]:
    if recursive:
        paths = [p for p in input_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    else:
        paths = [p for p in input_dir.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(paths)


def add_gaussian_blob(
    canvas: np.ndarray,
    rng: np.random.Generator,
    cx: float,
    cy: float,
    rx: float,
    ry: float,
    weight: float,
) -> None:
    h, w = canvas.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = ((xx - cx) ** 2) / (2.0 * rx * rx + 1e-6) + ((yy - cy) ** 2) / (2.0 * ry * ry + 1e-6)
    blob = np.exp(-dist, dtype=np.float32)
    canvas += weight * blob.astype(np.float32)


def add_soft_patch(canvas: np.ndarray, rng: np.random.Generator, blur_kernel: int) -> None:
    h, w = canvas.shape
    patch_mask = np.zeros_like(canvas, dtype=np.float32)
    pw = int(rng.uniform(0.12, 0.45) * w)
    ph = int(rng.uniform(0.10, 0.38) * h)
    x0 = int(rng.integers(0, max(1, w - pw)))
    y0 = int(rng.integers(0, max(1, h - ph)))
    patch_mask[y0 : y0 + ph, x0 : x0 + pw] = 1.0
    patch_mask = cv2.GaussianBlur(patch_mask, (blur_kernel, blur_kernel), 0)
    if patch_mask.max() > 1e-6:
        patch_mask /= patch_mask.max()
    canvas += float(rng.uniform(0.15, 0.55)) * patch_mask


def add_soft_stripe(canvas: np.ndarray, rng: np.random.Generator, blur_kernel: int) -> None:
    h, w = canvas.shape
    stripe_mask = np.zeros_like(canvas, dtype=np.float32)
    orientation = int(rng.integers(0, 3))  # 0: horizontal, 1: vertical, 2: diagonal

    if orientation == 0:
        y = int(rng.integers(0, h))
        thickness = int(max(2, rng.uniform(0.03, 0.10) * h))
        cv2.line(stripe_mask, (0, y), (w - 1, y), 1.0, thickness)
    elif orientation == 1:
        x = int(rng.integers(0, w))
        thickness = int(max(2, rng.uniform(0.03, 0.10) * w))
        cv2.line(stripe_mask, (x, 0), (x, h - 1), 1.0, thickness)
    else:
        y0 = int(rng.integers(0, h))
        y1 = int(rng.integers(0, h))
        thickness = int(max(2, rng.uniform(0.02, 0.08) * min(h, w)))
        cv2.line(stripe_mask, (0, y0), (w - 1, y1), 1.0, thickness)

    stripe_mask = cv2.GaussianBlur(stripe_mask, (blur_kernel, blur_kernel), 0)
    if stripe_mask.max() > 1e-6:
        stripe_mask /= stripe_mask.max()
    canvas += float(rng.uniform(0.10, 0.35)) * stripe_mask


def random_structured_importance_map(
    h: int,
    w: int,
    rng: np.random.Generator,
    args: argparse.Namespace,
) -> np.ndarray:
    map_raw = np.zeros((h, w), dtype=np.float32)
    min_hw = float(min(h, w))

    # Random smooth blobs.
    min_blobs = max(0, int(args.min_blobs))
    max_blobs = max(min_blobs, int(args.max_blobs))
    n_blobs = int(rng.integers(min_blobs, max_blobs + 1))
    for _ in range(n_blobs):
        cx = float(rng.uniform(0, w - 1))
        cy = float(rng.uniform(0, h - 1))
        rmin = max(1.0, args.min_blob_radius_frac * min_hw)
        rmax = max(rmin + 1.0, args.max_blob_radius_frac * min_hw)
        rx = float(rng.uniform(rmin, rmax))
        ry = float(rng.uniform(rmin, rmax))
        weight = float(rng.uniform(0.25, 1.00))
        add_gaussian_blob(map_raw, rng, cx, cy, rx, ry, weight)

    # Optional center-prior smooth blob.
    if rng.uniform(0.0, 1.0) < float(args.center_prob):
        cx = float(w * (0.5 + rng.uniform(-0.08, 0.08)))
        cy = float(h * (0.5 + rng.uniform(-0.08, 0.08)))
        rx = float(rng.uniform(0.18, 0.35) * w)
        ry = float(rng.uniform(0.18, 0.35) * h)
        add_gaussian_blob(map_raw, rng, cx, cy, rx, ry, weight=float(rng.uniform(0.35, 0.85)))

    blur_kernel = ensure_odd(args.blur_kernel)

    # Optional smooth patches.
    if rng.uniform(0.0, 1.0) < float(args.patch_prob):
        n_patches = int(rng.integers(1, max(2, int(args.max_patches)) + 1))
        for _ in range(n_patches):
            add_soft_patch(map_raw, rng, blur_kernel=ensure_odd(blur_kernel // 2))

    # Optional stripe priors.
    if rng.uniform(0.0, 1.0) < float(args.stripe_prob):
        n_stripes = int(rng.integers(1, max(2, int(args.max_stripes)) + 1))
        for _ in range(n_stripes):
            add_soft_stripe(map_raw, rng, blur_kernel=ensure_odd(blur_kernel // 2))

    # Final smoothing.
    map_raw = cv2.GaussianBlur(map_raw, (blur_kernel, blur_kernel), 0)
    map_raw -= map_raw.min()
    denom = map_raw.max()
    if denom > 1e-6:
        map_raw /= denom
    else:
        map_raw.fill(0.0)

    # Add low background base and keep range in [0,1].
    background = float(np.clip(args.background_weight, 0.0, 1.0))
    out = background + (1.0 - background) * map_raw
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def output_path_for(in_path: Path, input_root: Path, output_root: Path, recursive: bool) -> Path:
    if recursive:
        rel = in_path.relative_to(input_root)
        out_dir = output_root / rel.parent
    else:
        out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{in_path.stem}.png"


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_paths = gather_images(input_dir, recursive=args.recursive)
    stats = Stats(total_images=len(image_paths))
    if stats.total_images == 0:
        raise RuntimeError(f"No supported images found in {input_dir}")

    print("Generating synthetic random structured importance maps (no face model used).")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Total images: {stats.total_images}")
    print(f"Seed: {args.seed}")

    # Deterministic random stream if seed is fixed.
    rng = np.random.default_rng(int(args.seed))

    for idx, image_path in enumerate(image_paths, start=1):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            stats.failed_to_read += 1
            continue

        if args.image_size and int(args.image_size) > 0:
            img = cv2.resize(img, (int(args.image_size), int(args.image_size)), interpolation=cv2.INTER_AREA)

        h, w = img.shape[:2]
        imp = random_structured_importance_map(h, w, rng, args)
        imp_u8 = (imp * 255.0 + 0.5).astype(np.uint8)

        out_path = output_path_for(image_path, input_root=input_dir, output_root=output_dir, recursive=args.recursive)
        cv2.imwrite(str(out_path), imp_u8)
        stats.generated += 1

        if args.progress_interval > 0 and (idx % args.progress_interval == 0 or idx == stats.total_images):
            print(f"Processed {idx}/{stats.total_images}")

    print("Done.")
    print(f"Generated maps: {stats.generated}")
    print(f"Failed-to-read images: {stats.failed_to_read}")
    print(f"Output path: {output_dir}")
    print("Note: these importance maps are synthetic random structured priors, not semantic annotations.")


if __name__ == "__main__":
    main()
