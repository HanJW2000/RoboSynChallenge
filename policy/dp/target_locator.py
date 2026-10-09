"""Text-conditioned bell localization with per-episode caching."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class DetectionCandidate:
    xyxy: tuple[float, float, float, float]
    confidence: float
    label: str


@dataclass(frozen=True)
class BellDetection:
    cx: float
    cy: float
    width: float
    height: float
    confidence: float
    label: str


def _normalise_for_color(image: np.ndarray) -> np.ndarray:
    """Convert HWC (3/4 channel) uint8 or float [0, 1] images to HWC uint8 RGB."""
    array = np.asarray(image)
    if array.ndim == 4 and array.shape[1] in (3, 4):
        array = np.asarray(array[0]).transpose(1, 2, 0)
    if array.ndim != 3 or array.shape[2] < 3:
        raise ValueError(f"expected HWC image, got shape {array.shape}")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all():
            raise ValueError("floating image contains non-finite values")
        array = np.clip(array, 0.0, 1.0) * 255.0
    return np.ascontiguousarray(np.clip(array, 0, 255).astype(np.uint8))


def teal_box_fraction(
    image: np.ndarray, xyxy: tuple[float, float, float, float]
) -> float:
    """Fraction of in-box pixels matching the teal bell base in HSV space.

    Calibrated on 135 click_bell first frames under full domain
    randomisation (lighting + camera intrinsics/extrinsics): boxes centred
    on the bell score 0.39-0.51, while boxes on white paper cups or the
    black robot base score <= 0.06. Any threshold inside (0.06, 0.39)
    separates the two classes.
    """
    x1, y1, x2, y2 = (float(value) for value in xyxy)
    array = _normalise_for_color(image)
    height, width = array.shape[:2]
    x1i, y1i = max(0, int(x1)), max(0, int(y1))
    x2i, y2i = min(width, int(x2)), min(height, int(y2))
    if x2i <= x1i or y2i <= y1i:
        return 0.0
    crop = array[y1i:y2i, x1i:x2i].astype(np.float32) / 255.0
    maxc = crop.max(axis=2)
    minc = crop.min(axis=2)
    delta = maxc - minc
    saturation = np.where(maxc > 0.0, delta / np.maximum(maxc, 1e-6), 0.0)
    red, green, blue = crop[..., 0], crop[..., 1], crop[..., 2]
    hue6 = np.zeros_like(delta)
    mask = delta > 1e-6
    is_red = mask & (maxc == red)
    hue6[is_red] = ((green[is_red] - blue[is_red]) / delta[is_red]) % 6
    is_green = mask & (maxc == green)
    hue6[is_green] = (blue[is_green] - red[is_green]) / delta[is_green] + 2
    is_blue = mask & (maxc == blue)
    hue6[is_blue] = (red[is_blue] - green[is_blue]) / delta[is_blue] + 4
    hue = hue6 * 60.0
    teal = (
        (hue >= 140.0)
        & (hue <= 220.0)
        & (saturation > 0.20)
        & (maxc > 0.15)
        & (maxc < 0.95)
    )
    return float(teal.mean())


def _to_detection(
    confidence: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str,
    image_width: int,
    image_height: int,
) -> BellDetection:
    return BellDetection(
        cx=((x1 + x2) * 0.5) / image_width,
        cy=((y1 + y2) * 0.5) / image_height,
        width=(x2 - x1) / image_width,
        height=(y2 - y1) / image_height,
        confidence=confidence,
        label=label,
    )


def select_bell_candidate(
    candidates: Iterable[DetectionCandidate],
    image_width: int,
    image_height: int,
    box_threshold: float = 0.30,
    max_area_fraction: float = 0.10,
    image: np.ndarray | None = None,
    min_teal_fraction: float = 0.10,
) -> BellDetection | None:
    """Validate candidates and select the first color-plausible one.

    Candidates are ordered by (-confidence, area, x, y). When
    ``min_teal_fraction`` is positive, the first candidate whose in-box
    teal fraction reaches the threshold wins; high-confidence candidates
    that only match white cups or the robot base are skipped in favour of
    later candidates. ``image`` is required in that case.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not 0.0 <= box_threshold <= 1.0:
        raise ValueError("box_threshold must be in [0, 1]")
    if not 0.0 < max_area_fraction <= 1.0:
        raise ValueError("max_area_fraction must be in (0, 1]")
    if not 0.0 <= min_teal_fraction <= 1.0:
        raise ValueError("min_teal_fraction must be in [0, 1]")

    valid: list[tuple[float, float, float, float, float, float, str]] = []
    image_area = float(image_width * image_height)
    for candidate in candidates:
        try:
            x1, y1, x2, y2 = (float(value) for value in candidate.xyxy)
            confidence = float(candidate.confidence)
        except (TypeError, ValueError):
            continue
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2, confidence)):
            continue
        if confidence < box_threshold:
            continue
        if not (0.0 <= x1 < x2 <= image_width and 0.0 <= y1 < y2 <= image_height):
            continue
        area = (x2 - x1) * (y2 - y1)
        if area / image_area > max_area_fraction:
            continue
        valid.append((confidence, area, x1, y1, x2, y2, str(candidate.label)))

    if not valid:
        return None
    ordered = sorted(
        valid, key=lambda item: (-item[0], item[1], item[2], item[3])
    )
    if min_teal_fraction > 0.0:
        if image is None:
            raise ValueError("image is required when min_teal_fraction > 0")
        for confidence, _area, x1, y1, x2, y2, label in ordered:
            if teal_box_fraction(image, (x1, y1, x2, y2)) < min_teal_fraction:
                continue
            return _to_detection(
                confidence, x1, y1, x2, y2, label, image_width, image_height
            )
        return None
    confidence, _area, x1, y1, x2, y2, label = ordered[0]
    return _to_detection(
        confidence, x1, y1, x2, y2, label, image_width, image_height
    )


class GroundingDinoBackend:
    """Lazy, offline-only GroundingDINO adapter."""

    def __init__(
        self,
        model_path: str,
        revision: str,
        device: str = "cuda",
        prompt: str = "service bell. desk bell. call bell.",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
    ) -> None:
        if not model_path:
            raise ValueError("model_path is required")
        if not revision:
            raise ValueError("an exact model revision is required")
        self.model_path = model_path
        self.revision = revision
        self.device = device
        self.prompt = prompt
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._processor = None
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(
            self.model_path,
            revision=self.revision,
            local_files_only=True,
        )
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self.model_path,
            revision=self.revision,
            local_files_only=True,
        ).to(self.device)
        self._model.eval()

    def __call__(self, image: np.ndarray) -> list[DetectionCandidate]:
        import torch

        self._load()
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"expected HWC RGB image, got shape {array.shape}")
        rgb = np.ascontiguousarray(array[..., :3])
        inputs = self._processor(
            images=rgb,
            text=self.prompt,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        result = self._processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[rgb.shape[:2]],
        )[0]
        candidates = []
        for box, score, label in zip(
            result["boxes"], result["scores"], result["labels"], strict=True
        ):
            candidates.append(
                DetectionCandidate(
                    tuple(float(value) for value in box.detach().cpu().tolist()),
                    float(score.detach().cpu().item()),
                    str(label),
                )
            )
        return candidates


_NOT_CACHED = object()


class BellLocator:
    """Select and cache at most one bell detection per episode."""

    def __init__(
        self,
        backend: Callable[[np.ndarray], Sequence[DetectionCandidate]],
        box_threshold: float = 0.30,
        max_area_fraction: float = 0.10,
        max_consecutive_errors: int = 3,
        min_teal_fraction: float = 0.10,
    ) -> None:
        if max_consecutive_errors <= 0:
            raise ValueError("max_consecutive_errors must be positive")
        self.backend = backend
        self.box_threshold = float(box_threshold)
        self.max_area_fraction = float(max_area_fraction)
        self.max_consecutive_errors = int(max_consecutive_errors)
        self.min_teal_fraction = float(min_teal_fraction)
        self.consecutive_errors = 0
        self.last_error: str | None = None
        self.last_latency_s: float | None = None
        self._cached: BellDetection | None | object = _NOT_CACHED
        self.cache_hits = 0

    @property
    def has_cached_result(self) -> bool:
        return self._cached is not _NOT_CACHED

    def reset(self) -> None:
        self._cached = _NOT_CACHED
        self.cache_hits = 0
        self.last_error = None
        self.last_latency_s = None

    def detect_once(self, image: np.ndarray) -> BellDetection | None:
        if self._cached is not _NOT_CACHED:
            self.cache_hits += 1
            return self._cached  # type: ignore[return-value]

        array = np.asarray(image)
        if array.ndim != 3 or array.shape[-1] < 3:
            raise ValueError(f"expected HWC image, got shape {array.shape}")
        started_at = time.perf_counter()
        try:
            candidates = self.backend(array)
            detection = select_bell_candidate(
                candidates,
                image_width=int(array.shape[1]),
                image_height=int(array.shape[0]),
                box_threshold=self.box_threshold,
                max_area_fraction=self.max_area_fraction,
                image=array,
                min_teal_fraction=self.min_teal_fraction,
            )
        except Exception as exc:
            self.last_latency_s = time.perf_counter() - started_at
            self.consecutive_errors += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise RuntimeError(
                    f"bell locator failed for {self.consecutive_errors} consecutive episodes"
                ) from exc
            self._cached = None
            return None

        self.last_latency_s = time.perf_counter() - started_at
        self.consecutive_errors = 0
        self.last_error = None
        self._cached = detection
        return detection
