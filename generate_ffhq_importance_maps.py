#!/usr/bin/env python3
"""
Generate soft grayscale importance maps for FFHQ-style face images.

Detection priority:
1) MediaPipe Face Mesh (if installed)
2) dlib 68-point landmarks (if installed and predictor path is provided)
3) OpenCV Haar face detector
4) Center-face prior fallback (always available)

Output: one grayscale PNG per input image. Brighter means more important.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np  # type: ignore
except Exception:
    np = None

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


SUPPORTED_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


# MediaPipe Face Mesh landmark groups (indices).
MP_LEFT_EYE = [33, 133, 160, 159, 158, 157, 173, 144, 145, 153, 154, 155]
MP_RIGHT_EYE = [362, 263, 387, 386, 385, 384, 398, 373, 374, 380, 381, 382]
MP_LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
MP_RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276]
MP_NOSE = [168, 197, 195, 5, 4, 1, 2, 98, 327]
MP_MOUTH = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317,
    14, 87, 178, 88, 95, 185, 40, 39, 37, 0, 267, 269, 270, 409, 415, 310, 311, 312,
    13, 82, 81, 42, 183, 78
]
MP_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
    103, 67, 109
]


@dataclass
class Stats:
    total: int = 0
    success: int = 0
    fallback: int = 0
    failed_to_read: int = 0
    mediapipe_used: int = 0
    dlib_used: int = 0
    opencv_used: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate FFHQ importance-map PNGs.")
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
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories. Default: non-recursive.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        help="Optional resize to square size before processing (0 keeps original size).",
    )
    parser.add_argument(
        "--blur-kernel",
        type=int,
        default=31,
        help="Gaussian blur kernel for soft masks (odd integer).",
    )
    parser.add_argument(
        "--background-weight",
        type=float,
        default=0.15,
        help="Base background importance in [0,1].",
    )
    parser.add_argument(
        "--fallback-center-prior",
        type=float,
        default=1.0,
        help="Scale factor for center-prior fallback strength.",
    )
    parser.add_argument(
        "--dlib-shape-predictor",
        type=str,
        default="",
        help="Path to dlib shape_predictor_68_face_landmarks.dat (optional).",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=200,
        help="Print progress every N images.",
    )
    return parser.parse_args()


def ensure_odd(x: int, minimum: int = 3) -> int:
    x = max(int(x), minimum)
    return x if x % 2 == 1 else x + 1


def gather_images(input_dir: str, recursive: bool) -> List[Path]:
    root = Path(input_dir)
    if not root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    else:
        paths = [p for p in root.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    return sorted(paths)


class DetectorStack:
    def __init__(self, dlib_shape_predictor: str = ""):
        self.mp_face_mesh = None
        self.dlib_detector = None
        self.dlib_predictor = None
        self.cv2_face_cascade = None

        self._init_mediapipe()
        self._init_dlib(dlib_shape_predictor)
        self._init_opencv()

    def _init_mediapipe(self) -> None:
        try:
            import mediapipe as mp  # type: ignore

            self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
        except Exception:
            self.mp_face_mesh = None

    def _init_dlib(self, predictor_path: str) -> None:
        if not predictor_path:
            return
        if not os.path.isfile(predictor_path):
            return
        try:
            import dlib  # type: ignore

            self.dlib_detector = dlib.get_frontal_face_detector()
            self.dlib_predictor = dlib.shape_predictor(predictor_path)
        except Exception:
            self.dlib_detector = None
            self.dlib_predictor = None

    def _init_opencv(self) -> None:
        cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.isfile(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if not cascade.empty():
                self.cv2_face_cascade = cascade

    def status(self) -> Dict[str, bool]:
        return {
            "mediapipe": self.mp_face_mesh is not None,
            "dlib": self.dlib_detector is not None and self.dlib_predictor is not None,
            "opencv": self.cv2_face_cascade is not None,
        }

    def close(self) -> None:
        if self.mp_face_mesh is not None:
            self.mp_face_mesh.close()


class ImportanceMapBuilder:
    def __init__(self, blur_kernel: int, background_weight: float, fallback_center_prior: float):
        self.blur_kernel = ensure_odd(blur_kernel)
        self.background_weight = float(np.clip(background_weight, 0.0, 1.0))
        self.fallback_center_prior = max(float(fallback_center_prior), 0.0)

        # Relative region weights.
        self.w_eye = 0.85
        self.w_brow = 0.60
        self.w_nose = 0.60
        self.w_mouth = 0.70
        self.w_face = 0.45
        self.w_hair = 0.35

    def _add_soft_polygon(
        self,
        out: np.ndarray,
        points: np.ndarray,
        weight: float,
        blur_scale: float = 1.0,
    ) -> None:
        if points.shape[0] < 3:
            return
        mask = np.zeros_like(out, dtype=np.float32)
        hull = cv2.convexHull(points.astype(np.int32))
        cv2.fillConvexPoly(mask, hull, 1.0)
        k = ensure_odd(int(self.blur_kernel * blur_scale))
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        if mask.max() > 1e-6:
            mask /= mask.max()
        out += weight * mask

    def _add_soft_ellipse(
        self,
        out: np.ndarray,
        center: Tuple[int, int],
        axes: Tuple[int, int],
        angle: float,
        weight: float,
        blur_scale: float = 1.0,
    ) -> None:
        mask = np.zeros_like(out, dtype=np.float32)
        axes = (max(int(axes[0]), 1), max(int(axes[1]), 1))
        cv2.ellipse(mask, center, axes, angle, 0, 360, 1.0, -1)
        k = ensure_odd(int(self.blur_kernel * blur_scale))
        mask = cv2.GaussianBlur(mask, (k, k), 0)
        if mask.max() > 1e-6:
            mask /= mask.max()
        out += weight * mask

    def _center_prior(self, h: int, w: int) -> np.ndarray:
        out = np.full((h, w), self.background_weight, dtype=np.float32)
        cx, cy = w // 2, int(h * 0.56)
        self._add_soft_ellipse(
            out, (cx, cy), (int(0.28 * w), int(0.36 * h)), 0, self.w_face * self.fallback_center_prior, blur_scale=1.2
        )
        self._add_soft_ellipse(
            out, (cx, int(h * 0.30)), (int(0.32 * w), int(0.24 * h)), 0, self.w_hair * self.fallback_center_prior, blur_scale=1.3
        )
        self._add_soft_ellipse(
            out, (int(w * 0.37), int(h * 0.50)), (int(0.07 * w), int(0.05 * h)), 0, self.w_eye * self.fallback_center_prior
        )
        self._add_soft_ellipse(
            out, (int(w * 0.63), int(h * 0.50)), (int(0.07 * w), int(0.05 * h)), 0, self.w_eye * self.fallback_center_prior
        )
        self._add_soft_ellipse(
            out, (cx, int(h * 0.58)), (int(0.06 * w), int(0.08 * h)), 0, self.w_nose * self.fallback_center_prior
        )
        self._add_soft_ellipse(
            out, (cx, int(h * 0.68)), (int(0.12 * w), int(0.06 * h)), 0, self.w_mouth * self.fallback_center_prior
        )
        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _build_from_rect(self, h: int, w: int, rect: Tuple[int, int, int, int]) -> np.ndarray:
        out = np.full((h, w), self.background_weight, dtype=np.float32)
        x, y, rw, rh = rect
        cx, cy = int(x + rw * 0.5), int(y + rh * 0.58)

        self._add_soft_ellipse(out, (cx, cy), (int(rw * 0.54), int(rh * 0.65)), 0, self.w_face, blur_scale=1.2)
        self._add_soft_ellipse(out, (cx, int(y + rh * 0.18)), (int(rw * 0.70), int(rh * 0.42)), 0, self.w_hair, blur_scale=1.2)

        self._add_soft_ellipse(out, (int(x + rw * 0.33), int(y + rh * 0.43)), (int(rw * 0.10), int(rh * 0.06)), 0, self.w_eye)
        self._add_soft_ellipse(out, (int(x + rw * 0.67), int(y + rh * 0.43)), (int(rw * 0.10), int(rh * 0.06)), 0, self.w_eye)
        self._add_soft_ellipse(out, (int(x + rw * 0.33), int(y + rh * 0.36)), (int(rw * 0.11), int(rh * 0.04)), 0, self.w_brow)
        self._add_soft_ellipse(out, (int(x + rw * 0.67), int(y + rh * 0.36)), (int(rw * 0.11), int(rh * 0.04)), 0, self.w_brow)
        self._add_soft_ellipse(out, (cx, int(y + rh * 0.56)), (int(rw * 0.08), int(rh * 0.11)), 0, self.w_nose)
        self._add_soft_ellipse(out, (cx, int(y + rh * 0.72)), (int(rw * 0.16), int(rh * 0.07)), 0, self.w_mouth)

        np.clip(out, 0.0, 1.0, out=out)
        return out

    @staticmethod
    def _points_from_indices(points: np.ndarray, indices: Sequence[int]) -> np.ndarray:
        valid = [idx for idx in indices if 0 <= idx < points.shape[0]]
        if not valid:
            return np.zeros((0, 2), dtype=np.int32)
        return points[valid].astype(np.int32)

    def _build_from_mediapipe(self, h: int, w: int, points: np.ndarray) -> np.ndarray:
        out = np.full((h, w), self.background_weight, dtype=np.float32)

        left_eye = self._points_from_indices(points, MP_LEFT_EYE)
        right_eye = self._points_from_indices(points, MP_RIGHT_EYE)
        left_brow = self._points_from_indices(points, MP_LEFT_BROW)
        right_brow = self._points_from_indices(points, MP_RIGHT_BROW)
        nose = self._points_from_indices(points, MP_NOSE)
        mouth = self._points_from_indices(points, MP_MOUTH)
        face_oval = self._points_from_indices(points, MP_FACE_OVAL)

        self._add_soft_polygon(out, left_eye, self.w_eye)
        self._add_soft_polygon(out, right_eye, self.w_eye)
        self._add_soft_polygon(out, left_brow, self.w_brow)
        self._add_soft_polygon(out, right_brow, self.w_brow)
        self._add_soft_polygon(out, nose, self.w_nose)
        self._add_soft_polygon(out, mouth, self.w_mouth)

        if face_oval.shape[0] >= 3:
            x, y, rw, rh = cv2.boundingRect(face_oval)
            cx, cy = int(x + rw * 0.5), int(y + rh * 0.56)
            self._add_soft_ellipse(out, (cx, cy), (int(rw * 0.50), int(rh * 0.62)), 0, self.w_face, blur_scale=1.25)
            self._add_soft_ellipse(out, (cx, int(y + rh * 0.18)), (int(rw * 0.70), int(rh * 0.38)), 0, self.w_hair, blur_scale=1.3)
        else:
            out += self._center_prior(h, w) * 0.4

        np.clip(out, 0.0, 1.0, out=out)
        return out

    def _build_from_dlib(self, h: int, w: int, points68: np.ndarray) -> np.ndarray:
        out = np.full((h, w), self.background_weight, dtype=np.float32)
        jaw = points68[0:17]
        left_brow = points68[17:22]
        right_brow = points68[22:27]
        nose = points68[27:36]
        left_eye = points68[36:42]
        right_eye = points68[42:48]
        mouth = points68[48:68]

        self._add_soft_polygon(out, left_eye, self.w_eye)
        self._add_soft_polygon(out, right_eye, self.w_eye)
        self._add_soft_polygon(out, left_brow, self.w_brow)
        self._add_soft_polygon(out, right_brow, self.w_brow)
        self._add_soft_polygon(out, nose, self.w_nose)
        self._add_soft_polygon(out, mouth, self.w_mouth)

        x, y, rw, rh = cv2.boundingRect(jaw.astype(np.int32))
        cx, cy = int(x + rw * 0.5), int(y + rh * 0.56)
        self._add_soft_ellipse(out, (cx, cy), (int(rw * 0.52), int(rh * 0.66)), 0, self.w_face, blur_scale=1.2)
        self._add_soft_ellipse(out, (cx, int(y + rh * 0.08)), (int(rw * 0.70), int(rh * 0.45)), 0, self.w_hair, blur_scale=1.25)

        np.clip(out, 0.0, 1.0, out=out)
        return out

    def build(
        self,
        image_bgr: np.ndarray,
        detectors: DetectorStack,
    ) -> Tuple[np.ndarray, str]:
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # 1) MediaPipe Face Mesh
        if detectors.mp_face_mesh is not None:
            try:
                result = detectors.mp_face_mesh.process(rgb)
                if result.multi_face_landmarks:
                    lm = result.multi_face_landmarks[0].landmark
                    points = np.array(
                        [[int(np.clip(p.x * w, 0, w - 1)), int(np.clip(p.y * h, 0, h - 1))] for p in lm],
                        dtype=np.int32,
                    )
                    return self._build_from_mediapipe(h, w, points), "mediapipe"
            except Exception:
                pass

        # 2) dlib 68 landmarks
        if detectors.dlib_detector is not None and detectors.dlib_predictor is not None:
            try:
                rects = detectors.dlib_detector(gray, 1)
                if len(rects) > 0:
                    rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
                    shape = detectors.dlib_predictor(gray, rect)
                    points = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)], dtype=np.int32)
                    return self._build_from_dlib(h, w, points), "dlib"
            except Exception:
                pass

        # 3) OpenCV Haar detector
        if detectors.cv2_face_cascade is not None:
            faces = detectors.cv2_face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(40, 40),
            )
            if len(faces) > 0:
                x, y, rw, rh = max(faces, key=lambda b: b[2] * b[3])
                return self._build_from_rect(h, w, (int(x), int(y), int(rw), int(rh))), "opencv"

        # 4) Generic center prior fallback
        return self._center_prior(h, w), "fallback"


def map_to_uint8(importance: np.ndarray) -> np.ndarray:
    importance = np.clip(importance, 0.0, 1.0)
    return (importance * 255.0 + 0.5).astype(np.uint8)


def output_path_for(
    in_path: Path,
    input_root: Path,
    output_root: Path,
    recursive: bool,
) -> Path:
    if recursive:
        rel = in_path.relative_to(input_root)
        out_dir = output_root / rel.parent
    else:
        out_dir = output_root
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{in_path.stem}.png"


def process_images(args: argparse.Namespace) -> Stats:
    if np is None:
        raise RuntimeError("NumPy is required. Please install it with `pip install numpy`.")
    if cv2 is None:
        raise RuntimeError(
            "OpenCV is required for this script. Please install it with "
            "`pip install opencv-python` (or `opencv-python-headless`)."
        )

    images = gather_images(args.input_dir, args.recursive)
    stats = Stats(total=len(images))
    if stats.total == 0:
        raise RuntimeError(f"No supported images found in {args.input_dir}")

    input_root = Path(args.input_dir).resolve()
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    detectors = DetectorStack(dlib_shape_predictor=args.dlib_shape_predictor)
    status = detectors.status()
    print(
        "Detector availability:",
        f"MediaPipe={status['mediapipe']}, dlib={status['dlib']}, OpenCV={status['opencv']}",
    )
    if status["dlib"] is False and args.dlib_shape_predictor:
        print("dlib predictor was requested but not loaded. Check --dlib-shape-predictor path/dependencies.")

    builder = ImportanceMapBuilder(
        blur_kernel=args.blur_kernel,
        background_weight=args.background_weight,
        fallback_center_prior=args.fallback_center_prior,
    )

    for idx, image_path in enumerate(images, start=1):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            stats.failed_to_read += 1
            continue

        if args.image_size and args.image_size > 0:
            img = cv2.resize(img, (args.image_size, args.image_size), interpolation=cv2.INTER_AREA)

        importance, method = builder.build(img, detectors)
        out_u8 = map_to_uint8(importance)

        out_path = output_path_for(
            image_path,
            input_root=input_root,
            output_root=output_root,
            recursive=args.recursive,
        )
        cv2.imwrite(str(out_path), out_u8)

        if method == "mediapipe":
            stats.success += 1
            stats.mediapipe_used += 1
        elif method == "dlib":
            stats.success += 1
            stats.dlib_used += 1
        elif method == "opencv":
            stats.success += 1
            stats.opencv_used += 1
        else:
            stats.fallback += 1

        if args.progress_interval > 0 and (idx % args.progress_interval == 0 or idx == stats.total):
            print(f"Processed {idx}/{stats.total}")

    detectors.close()
    return stats


def main() -> None:
    args = parse_args()
    print(f"script start")
    stats = process_images(args)
    print("Done.")
    print(f"Total images: {stats.total}")
    print(f"Successful face detections: {stats.success}")
    print(f"  - MediaPipe used: {stats.mediapipe_used}")
    print(f"  - dlib used: {stats.dlib_used}")
    print(f"  - OpenCV used: {stats.opencv_used}")
    print(f"Fallback count: {stats.fallback}")
    print(f"Failed-to-read count: {stats.failed_to_read}")
    print(f"Output path: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
