"""Deterministic image perturbations for robustness testing (Phase 4b).

`apply_perturbation(img, name, severity, seed)`: img is a uint8 array (H, W) or (H, W, C); the result has the same
shape and dtype. `severity` is an index: 0 is the identity, 1..n picks the n-th entry of LEVELS[name].

  blur        Gaussian blur, sigma in pixels
  brightness  multiply pixel values by a factor
  contrast    scale deviations from the image mean by a factor
  noise       additive Gaussian noise, sigma on a 0-1 intensity scale
  jpeg        JPEG re-compression at the given quality

Only `noise` is random; it draws from numpy's default_rng(seed), so the same (img, severity, seed) always gives the
same output. The other perturbations ignore the seed.
"""

import zlib
from typing import Dict, Tuple

import cv2
import numpy as np

LEVELS: Dict[str, Tuple[float, ...]] = {
    "blur": (1.0, 2.0, 3.0),
    "brightness": (0.6, 0.8, 1.2, 1.4),
    "contrast": (0.6, 1.4),
    "noise": (0.02, 0.05, 0.1),
    "jpeg": (50, 20),
}


def level_value(name: str, severity: int) -> float:
    """The perturbation parameter for a severity index (severity 0 has no parameter: NaN)."""
    check(name, severity)
    return float("nan") if severity == 0 else LEVELS[name][severity - 1]


def check(name: str, severity: int) -> None:
    if name not in LEVELS:
        raise ValueError(f"unknown perturbation {name!r}; choose from {sorted(LEVELS)}")
    if not isinstance(severity, (int, np.integer)) or not 0 <= severity <= len(LEVELS[name]):
        raise ValueError(f"severity for {name!r} must be an int in 0..{len(LEVELS[name])}, got {severity!r}")


def _to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def image_seed(image_name: str, base_seed: int = 0) -> int:
    """Stable per-image seed (same on every run and platform), so noise differs between images."""
    return (zlib.crc32(image_name.encode("utf-8")) + int(base_seed)) % (2 ** 32)


def apply_perturbation(img: np.ndarray, name: str, severity: int, seed: int = 0) -> np.ndarray:
    """Return a perturbed copy of a uint8 image. Severity 0 returns an unchanged copy."""
    check(name, severity)
    if img.dtype != np.uint8 or img.ndim not in (2, 3) or (img.ndim == 3 and img.shape[2] not in (1, 3)):
        raise ValueError(f"expected a uint8 array of shape (H, W), (H, W, 1) or (H, W, 3), got {img.dtype} {img.shape}")
    if severity == 0:
        return img.copy()
    v = LEVELS[name][severity - 1]
    if name == "blur":
        out = cv2.GaussianBlur(img, (0, 0), sigmaX=float(v), borderType=cv2.BORDER_REFLECT_101)
    elif name == "brightness":
        out = _to_uint8(img.astype(np.float64) * v)
    elif name == "contrast":
        x = img.astype(np.float64)
        out = _to_uint8((x - x.mean()) * v + x.mean())
    elif name == "noise":
        rng = np.random.default_rng(seed)
        x = img.astype(np.float64) / 255.0 + rng.normal(0.0, v, img.shape)
        out = _to_uint8(np.clip(x, 0.0, 1.0) * 255.0)
    else:  # jpeg
        gray = img.ndim == 2 or img.shape[2] == 1
        ok, buf = cv2.imencode(".jpg", img.reshape(img.shape[:2]) if gray else img,
                               [cv2.IMWRITE_JPEG_QUALITY, int(v)])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        out = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR)
    return out.reshape(img.shape)


def all_conditions():
    """[(name, severity)] for every perturbation level (excluding the identity)."""
    return [(n, s) for n, lv in LEVELS.items() for s in range(1, len(lv) + 1)]
