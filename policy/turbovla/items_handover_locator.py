"""Inference-only visual target adapter for items_handover.

The adapter uses only RGB observations. GroundingDINO detects candidates in
the first cam_high frame; image-space priors learned from the training split
disambiguate distractor pens/cups. No simulator pose or privileged state is
read at evaluation time.
"""

from __future__ import annotations

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


PHASE_PROMPTS = (
    "Stage GRASP_PEN: locate and pick up the pen with the right gripper",
    "Stage MOVE_TO_HANDOVER: carry the pen to the left gripper",
    "Stage HANDOVER: transfer the pen from the right gripper to the left gripper",
    "Stage INSERT_HOLDER: insert the pen into the pen holder with the left gripper",
    "Stage RELEASE_RETREAT: release the pen and move both grippers away",
)

STAGE_SUFFIXES = (
    "grasp the pen with the right gripper",
    "move the pen toward the left gripper",
    "transfer the pen to the left gripper",
    "insert the pen into the holder with the left gripper",
    "release the pen and retreat",
)

# Median and robust image-space scale measured from the 1,000 training episodes.
PRIORS = {
    "pen": {"cx": 0.7988, "cy": 0.3525, "sx": 0.1770, "sy": 0.1054,
            "w": 0.0636, "h": 0.2301},
    "holder": {"cx": 0.3928, "cy": 0.2527, "sx": 0.1672, "sy": 0.1319,
               "w": 0.1514, "h": 0.2499},
}


class ItemsHandoverInferenceAdapter:
    camera_key = "observation.images.cam_high"

    def __init__(self, device: str):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_path = os.environ.get(
            "ITEMS_HANDOVER_DINO_PATH",
            "/supconit-data/robosyn/checkpoints/pretrained/grounding-dino-base",
        )
        revision = os.environ.get(
            "ITEMS_HANDOVER_DINO_REVISION",
            "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        )
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, revision=revision, local_files_only=True
        )
        self.detector = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_path, revision=revision, local_files_only=True
        ).to(device)
        self.detector.eval()
        self.prompt = (
            "writing pen. marker pen. pencil. thin rod. cylindrical cup. "
            "container. pen holder."
        )
        self.mode = os.environ.get(
            "ITEMS_HANDOVER_LOCATOR_MODE", "v1_dense"
        ).strip().lower()
        if self.mode not in {"v1_dense", "v2_sparse"}:
            raise ValueError(
                "ITEMS_HANDOVER_LOCATOR_MODE must be v1_dense or v2_sparse, "
                f"got {self.mode!r}"
            )
        self.overlay_strength = float(
            os.environ.get("ITEMS_HANDOVER_LOCATOR_STRENGTH", "0.18")
        )
        if not 0.0 <= self.overlay_strength <= 1.0:
            raise ValueError(
                "ITEMS_HANDOVER_LOCATOR_STRENGTH must be in [0, 1], "
                f"got {self.overlay_strength}"
            )
        self.reset()

    def reset(self):
        self.action_count = 0
        self.targets = None

    @staticmethod
    def _phase(action_count: int) -> int:
        if action_count < 94:
            return 0
        if action_count < 207:
            return 1
        if action_count < 220:
            return 2
        if action_count < 295:
            return 3
        return 4

    @staticmethod
    def _valid_box(box, width: int, height: int) -> bool:
        x1, y1, x2, y2 = box
        if not all(math.isfinite(float(v)) for v in box):
            return False
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return False
        area = ((x2 - x1) * (y2 - y1)) / float(width * height)
        return 0.0005 <= area <= 0.10

    @staticmethod
    def _kind(label: str):
        label = label.lower()
        is_pen = any(t in label for t in ("pen", "pencil", "marker", "rod"))
        is_holder = any(t in label for t in ("cup", "container", "holder"))
        return is_pen, is_holder

    def _rank(self, candidate: dict, kind: str) -> float:
        p = PRIORS[kind]
        spatial = ((candidate["cx"] - p["cx"]) / p["sx"]) ** 2
        spatial += ((candidate["cy"] - p["cy"]) / p["sy"]) ** 2
        shape = ((candidate["width"] - p["w"]) / max(p["w"], 0.04)) ** 2
        shape += ((candidate["height"] - p["h"]) / max(p["h"], 0.08)) ** 2
        return 1.2 * spatial + 0.25 * shape - 2.0 * candidate["confidence"]

    def _detect(self, image: torch.Tensor) -> dict:
        rgb = image.detach().float().cpu().clamp(0, 1)
        array = (rgb.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        height, width = array.shape[:2]
        inputs = self.processor(
            images=Image.fromarray(array), text=self.prompt, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            outputs = self.detector(**inputs)
        result = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.10,
            text_threshold=0.10,
            target_sizes=[(height, width)],
        )[0]
        labels = result.get("text_labels", result.get("labels", []))
        candidates = {"pen": [], "holder": []}
        for box_t, score_t, label_value in zip(
            result["boxes"], result["scores"], labels, strict=True
        ):
            box = [float(v) for v in box_t.detach().cpu().tolist()]
            if not self._valid_box(box, width, height):
                continue
            x1, y1, x2, y2 = box
            row = {
                "cx": (x1 + x2) / (2.0 * width),
                "cy": (y1 + y2) / (2.0 * height),
                "width": (x2 - x1) / width,
                "height": (y2 - y1) / height,
                "confidence": float(score_t.detach().cpu().item()),
                "label": str(label_value),
            }
            is_pen, is_holder = self._kind(row["label"])
            if is_pen:
                candidates["pen"].append(row)
            if is_holder:
                candidates["holder"].append(row)

        selected = {}
        for kind in ("pen", "holder"):
            rows = candidates[kind]
            selected[kind] = min(rows, key=lambda row: self._rank(row, kind)) if rows else PRIORS[kind]
            print(
                f"[items_handover_locator] {kind}: candidates={len(rows)} "
                f"selected=({selected[kind]['cx']:.3f},{selected[kind]['cy']:.3f}) "
                f"confidence={selected[kind].get('confidence', 0.0):.3f}",
                flush=True,
            )
        return selected

    @staticmethod
    def _blend(
        image: torch.Tensor,
        target: dict,
        color: tuple[float, float, float],
        strength: float = 0.55,
    ):
        height, width = image.shape[-2:]
        low_h, low_w = 120, 160
        yy, xx = torch.meshgrid(
            torch.arange(low_h, device=image.device, dtype=torch.float32),
            torch.arange(low_w, device=image.device, dtype=torch.float32),
            indexing="ij",
        )
        sx = max(1.25, float(target["width"]) * low_w * 0.70)
        sy = max(1.25, float(target["height"]) * low_h * 0.70)
        heat = torch.exp(-0.5 * (((xx - target["cx"] * low_w) / sx) ** 2 +
                                 ((yy - target["cy"] * low_h) / sy) ** 2))
        heat = F.interpolate(
            heat[None, None], size=(height, width), mode="bilinear", align_corners=False
        )[0, 0].to(dtype=image.dtype)
        weight = (float(strength) * heat).unsqueeze(0)
        overlay = torch.tensor(color, dtype=image.dtype, device=image.device).view(3, 1, 1)
        return image * (1.0 - weight) + overlay * weight

    def prepare(self, obs: dict, original_task: str):
        image = obs[self.camera_key]
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError(f"unexpected cam_high shape {tuple(image.shape)}")
        if self.targets is None:
            self.targets = self._detect(image)
        phase = self._phase(self.action_count)
        if self.mode == "v2_sparse":
            # Keep the natural RGB view intact during transport/handover.  A
            # weak cue is shown only when the corresponding static target is
            # immediately actionable.
            if phase == 0:
                image = self._blend(
                    image,
                    self.targets["pen"],
                    (0.0, 1.0, 1.0),
                    self.overlay_strength,
                )
            elif phase == 3:
                image = self._blend(
                    image,
                    self.targets["holder"],
                    (1.0, 0.0, 1.0),
                    self.overlay_strength,
                )
            original = str(original_task).strip().rstrip(".")
            task = f"{original}. Current stage: {STAGE_SUFFIXES[phase]}."
        else:
            if phase == 0:
                image = self._blend(image, self.targets["pen"], (0.0, 1.0, 1.0))
            image = self._blend(image, self.targets["holder"], (1.0, 0.0, 1.0))
            task = PHASE_PROMPTS[phase]
        obs[self.camera_key] = image.clamp(0.0, 1.0)
        return obs, task

    def advance(self, n_actions: int):
        self.action_count += int(n_actions)
