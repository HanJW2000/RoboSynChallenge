"""Render a bell location as an RGB Gaussian without changing image shape."""

from __future__ import annotations

import math

import numpy as np

from policy.dp.target_locator import BellDetection


def _validate_image(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] < 3:
        raise ValueError(f"expected HWC image with at least 3 channels, got {array.shape}")
    if array.dtype not in (np.dtype(np.uint8), np.dtype(np.float32)):
        raise TypeError(f"expected uint8 or float32 image, got {array.dtype}")
    if array.dtype == np.float32:
        if not np.isfinite(array).all():
            raise ValueError("image contains non-finite values")
        if array.size and (float(array.min()) < 0.0 or float(array.max()) > 1.0):
            raise ValueError("float32 image values must be in [0, 1]")
    return array


def _validate_detection(detection: BellDetection) -> None:
    values = (
        detection.cx,
        detection.cy,
        detection.width,
        detection.height,
        detection.confidence,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("detection values must be finite")
    if not 0.0 <= detection.cx <= 1.0 or not 0.0 <= detection.cy <= 1.0:
        raise ValueError("detection center must be in [0, 1]")
    if not 0.0 < detection.width <= 1.0 or not 0.0 < detection.height <= 1.0:
        raise ValueError("detection size must be in (0, 1]")
    if not 0.0 <= detection.confidence <= 1.0:
        raise ValueError("detection confidence must be in [0, 1]")


def overlay_bell_heatmap(
    image: np.ndarray,
    detection: BellDetection | None,
) -> np.ndarray:
    """Alpha-blend a red Gaussian on the first three channels."""
    source = _validate_image(image)
    if detection is None:
        return source.copy()
    _validate_detection(detection)

    height, width = source.shape[:2]
    center_x = detection.cx * width
    center_y = detection.cy * height
    sigma = float(
        np.clip(
            0.35 * max(detection.width * width, detection.height * height),
            6.0,
            32.0,
        )
    )
    yy, xx = np.ogrid[:height, :width]
    heat = np.exp(
        -((xx - center_x) ** 2 + (yy - center_y) ** 2) / (2.0 * sigma * sigma)
    )
    alpha = (0.55 * detection.confidence * heat)[..., None].astype(np.float32)
    scale = 255.0 if source.dtype == np.uint8 else 1.0
    target_red = np.asarray(
        [scale, (32.0 / 255.0) * scale, (32.0 / 255.0) * scale],
        dtype=np.float32,
    )

    output = source.copy()
    rgb = source[..., :3].astype(np.float32)
    blended = (1.0 - alpha) * rgb + alpha * target_red
    if source.dtype == np.uint8:
        blended = np.rint(blended)
    output[..., :3] = np.clip(blended, 0.0, scale).astype(source.dtype)
    return output


def overlay_bell_heatmap_torch(image, detection: BellDetection | None):
    """Torch-native equivalent for HWC or BHWC tensors."""
    import torch

    if not isinstance(image, torch.Tensor):
        raise TypeError(f"expected torch.Tensor, got {type(image).__name__}")
    if image.ndim not in (3, 4) or image.shape[-1] < 3:
        raise ValueError(f"expected HWC or BHWC image, got {tuple(image.shape)}")
    if image.dtype not in (torch.uint8, torch.float32):
        raise TypeError(f"expected uint8 or float32 image, got {image.dtype}")
    if image.dtype == torch.float32:
        if not torch.isfinite(image).all().item():
            raise ValueError("image contains non-finite values")
        if image.numel() and (image.min().item() < 0.0 or image.max().item() > 1.0):
            raise ValueError("float32 image values must be in [0, 1]")
    if detection is None:
        return image.clone()
    _validate_detection(detection)

    height, width = int(image.shape[-3]), int(image.shape[-2])
    center_x = detection.cx * width
    center_y = detection.cy * height
    sigma = max(
        6.0,
        min(
            32.0,
            0.35 * max(detection.width * width, detection.height * height),
        ),
    )
    yy = torch.arange(height, dtype=torch.float32, device=image.device).view(height, 1)
    xx = torch.arange(width, dtype=torch.float32, device=image.device).view(1, width)
    heat = torch.exp(
        -((xx - center_x).square() + (yy - center_y).square()) / (2.0 * sigma * sigma)
    )
    alpha = (0.55 * detection.confidence * heat).unsqueeze(-1)
    if image.ndim == 4:
        alpha = alpha.unsqueeze(0)
    scale = 255.0 if image.dtype == torch.uint8 else 1.0
    target_red = torch.tensor(
        [scale, (32.0 / 255.0) * scale, (32.0 / 255.0) * scale],
        dtype=torch.float32,
        device=image.device,
    )
    output = image.clone()
    rgb = image[..., :3].to(dtype=torch.float32)
    blended = (1.0 - alpha) * rgb + alpha * target_red
    if image.dtype == torch.uint8:
        blended = blended.round()
    output[..., :3] = blended.clamp(0.0, scale).to(dtype=image.dtype)
    return output
