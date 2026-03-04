import os
import random
from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _collect_images(image_dir: str) -> List[str]:
    image_paths: List[str] = []
    for ext in IMG_EXTENSIONS:
        image_paths.extend(glob(os.path.join(image_dir, f"*{ext}")))
        image_paths.extend(glob(os.path.join(image_dir, f"*{ext.upper()}")))
    image_paths = sorted(set(image_paths))
    if not image_paths:
        raise ValueError(f"No images found in {image_dir}")
    return image_paths


def resolve_split_dirs(split_dir: str) -> Tuple[str, Optional[str]]:
    images_dir = os.path.join(split_dir, "images")
    importance_dir = os.path.join(split_dir, "importance")
    if os.path.isdir(images_dir):
        return images_dir, importance_dir if os.path.isdir(importance_dir) else None
    return split_dir, None


def _importance_path_for_image(image_path: str, importance_dir: Optional[str]) -> Optional[str]:
    if importance_dir is None:
        return None
    image_name = os.path.basename(image_path)
    full_match = os.path.join(importance_dir, image_name)
    if os.path.isfile(full_match):
        return full_match

    stem, _ = os.path.splitext(image_name)
    for ext in IMG_EXTENSIONS:
        candidate = os.path.join(importance_dir, f"{stem}{ext}")
        if os.path.isfile(candidate):
            return candidate
        candidate_upper = os.path.join(importance_dir, f"{stem}{ext.upper()}")
        if os.path.isfile(candidate_upper):
            return candidate_upper
    return None


class ImportanceImageDataset(Dataset):
    """
    Images are loaded from `images_dir`.
    Importance maps are optional; missing maps fall back to all-ones.
    """

    def __init__(self, images_dir: str, image_size: int, importance_dir: Optional[str] = None):
        self.images_dir = images_dir
        self.importance_dir = importance_dir
        self.image_paths = _collect_images(images_dir)
        self.image_size = image_size
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.importance_resize = transforms.Resize(
            (image_size, image_size), interpolation=InterpolationMode.BILINEAR
        )
        self.to_tensor = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image_name = os.path.basename(image_path)

        image = Image.open(image_path).convert("RGB")
        image_tensor = self.image_transform(image)

        imp_path = _importance_path_for_image(image_path, self.importance_dir)
        if imp_path is None:
            importance_tensor = torch.ones((1, self.image_size, self.image_size), dtype=image_tensor.dtype)
        else:
            imp_image = Image.open(imp_path).convert("L")
            imp_image = self.importance_resize(imp_image)
            importance_tensor = self.to_tensor(imp_image).clamp(0.0, 1.0)

        return image_tensor, importance_tensor, image_name


def sample_snr_db(
    fixed_snr: Optional[float],
    snr_range: Optional[Sequence[float]],
    generator: random.Random,
) -> float:
    if fixed_snr is not None:
        return float(fixed_snr)
    if snr_range is not None:
        low, high = float(snr_range[0]), float(snr_range[1])
        if high < low:
            low, high = high, low
        return generator.uniform(low, high)
    return 10.0


def reconstruction_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    use_mse: bool = False,
) -> torch.Tensor:
    if use_mse:
        diff = (x - x_hat) ** 2
    else:
        diff = (x - x_hat).abs()
    return diff.mean()


def batch_psnr(x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mse = torch.mean((x - x_hat) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


def pearson_corr_loss_map(
    importance: torch.Tensor,
    err_map: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable batch-mean Pearson correlation between importance and error maps.
    Inputs are expected in shape [B, 1, H, W].
    Returns a scalar tensor.
    """
    if importance.shape != err_map.shape:
        raise ValueError(f"Shape mismatch: importance {importance.shape} vs err_map {err_map.shape}")

    imp = importance.reshape(importance.size(0), -1).float()
    err = err_map.reshape(err_map.size(0), -1).float()

    imp = imp - imp.mean(dim=1, keepdim=True)
    err = err - err.mean(dim=1, keepdim=True)

    numerator = torch.sum(imp * err, dim=1)
    denom = torch.sqrt(torch.sum(imp * imp, dim=1) * torch.sum(err * err, dim=1) + eps)
    corr = numerator / (denom + eps)
    return corr.mean()


@dataclass
class RunningAverages:
    loss: float = 0.0
    recon_loss: float = 0.0
    psnr: float = 0.0
    count: int = 0

    def update(self, loss: float, recon_loss: float, psnr: float, n: int) -> None:
        self.loss += loss * n
        self.recon_loss += recon_loss * n
        self.psnr += psnr * n
        self.count += n

    def compute(self):
        denom = max(self.count, 1)
        return {
            "loss": self.loss / denom,
            "recon_loss": self.recon_loss / denom,
            "psnr": self.psnr / denom,
        }
