#!/usr/bin/env python3
# ----------------------------------------------------------------------------
# TurboVLA inference worker for RoboSynChallenge protocol eval.
#
# Runs in the dedicated turbovla venv (not the protocol eval venv). Loads the
# model once, then serves JSON-line RPC over stdin/stdout:
#   {"cmd": "ping"}                       -> {"ok": true}
#   {"cmd": "reset"}                      -> {"ok": true}
#   {"cmd": "infer", "obs_path": "...", "lang": "..."}
#     -> {"ok": true, "actions": [[14 floats] x horizon]}
#
# The npz passed to "infer" must contain one uint8 HWC array per camera in
# --cam-order plus a "state" array (14D qpos).
#
# Set ITEMS_HANDOVER_LOCATOR=1 to run the RGB-only GroundingDINO target
# locator on cam_high (items_handover task): it blends the same cyan pen /
# magenta holder heatmaps used at training time and replaces the task text
# with the current stage prompt. No simulator state is read.
# ----------------------------------------------------------------------------

import argparse
import contextlib
import json
import os
import sys

import numpy as np
import torch


DEFAULT_CAM_ORDER = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
BINARY_THRESHOLD = 0.49
EPS = 1e-8


def np_to_pil(array) -> "object":
    from PIL import Image

    return Image.fromarray(np.ascontiguousarray(np.asarray(array, dtype=np.uint8)[..., :3]))


def apply_locator(adapter, images, lang: str, cam_order=None):
    """Blend cam_high locator heatmaps and swap in the phase prompt.

    Returns (images, lang); identity when adapter is None. Wrist cameras
    pass through unchanged, matching the training-time augmentation.
    """
    if adapter is None:
        return images, lang
    order = list(cam_order) if cam_order is not None else list(DEFAULT_CAM_ORDER)
    if not order or order[0] != "cam_high":
        raise ValueError(f"locator requires cam_high first in cam_order, got {order}")
    tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(images[0]))[..., :3])
    tensor = tensor.permute(2, 0, 1).float().div_(255.0)
    obs = {"observation.images.cam_high": tensor}
    obs, lang = adapter.prepare(obs, lang)
    blended = obs["observation.images.cam_high"].clamp(0.0, 1.0)
    blended = (blended * 255.0).round().to(torch.uint8).permute(1, 2, 0).numpy()
    return [np_to_pil(blended), *images[1:]], lang


def build_overrides(args) -> list[str]:
    """Map CLI args onto the runtime's framework config schema.

    The runtime registers `framework.{text,vision,interaction,initialization,
    action}`; released robotwin checkpoints store the legacy schema
    (`framework.{dinov3,action_model,fusion,groundingdino,text}`) whose
    non-overlapping keys are dropped by merge_framework_config, so legacy
    checkpoints need explicit overrides (paths, attention, name, init flags).
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    from starVLA.model.framework.base_framework import (
        _auto_import_framework_modules,
    )
    from starVLA.model.tools import FRAMEWORK_REGISTRY

    overrides = []
    if args.attn_implementation:
        overrides.append(
            f"framework.vision.attn_implementation={args.attn_implementation}"
        )
        overrides.append(
            f"framework.text.attn_implementation={args.attn_implementation}"
        )
    if args.dinov3_path:
        overrides.append(f"framework.vision.model_path={args.dinov3_path}")
        overrides.append("framework.vision.local_files_only=True")
    if args.bert_path:
        overrides.append(f"framework.text.bert_path={args.bert_path}")
        overrides.append("framework.text.local_files_only=True")
    if args.init_ckpt:
        # _load_initialization always runs at build time and requires a real
        # file; the flags below it only select which tensors are copied.
        overrides.append(f"framework.initialization.pretrained_ckpt={args.init_ckpt}")

    ckpt = Path(args.checkpoint)
    stored_name = None
    for parent in ckpt.parents:
        cfg_path = parent / "config.yaml"
        if cfg_path.is_file() and (parent / "dataset_statistics.json").is_file():
            stored_name = OmegaConf.load(str(cfg_path)).framework.name
            break

    if stored_name is not None:
        _auto_import_framework_modules()
        registry = FRAMEWORK_REGISTRY._registry
        if stored_name not in registry and "TurboVLA" in registry:
            print(
                f"framework.name {stored_name!r} not registered; mapping to 'TurboVLA'",
                flush=True,
            )
            overrides.append("framework.name=TurboVLA")
    return overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to steps_*_ema_model.safetensors; config.yaml and "
        "dataset_statistics.json must exist in one of its parent directories.",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument(
        "--device",
        default=None,
        help="Device override, e.g. cpu (default: cuda:<gpu-id>).",
    )
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        help="sdpa keeps the worker runnable without flash-attn.",
    )
    parser.add_argument(
        "--dinov3-path",
        default=None,
        help="Local DINOv3 model dir; overrides framework.dinov3.model_path.",
    )
    parser.add_argument(
        "--bert-path",
        default=None,
        help="Local BERT model dir; overrides framework.text.bert_path.",
    )
    parser.add_argument(
        "--init-ckpt",
        default=None,
        help="GroundingDINO init .pth; overrides framework.initialization."
        "pretrained_ckpt (required by the build; weights are overwritten by "
        "the checkpoint state dict afterwards).",
    )
    parser.add_argument(
        "--strict-load",
        action="store_true",
        help="Require the checkpoint to contain every model tensor. Released "
        "checkpoints exclude frozen encoders and need the default partial load.",
    )
    parser.add_argument("--cam-order", nargs="+", default=DEFAULT_CAM_ORDER)
    args = parser.parse_args()

    device = args.device or f"cuda:{args.gpu_id}"

    from PIL import Image

    from starVLA.model.framework.base_framework import baseframework

    vla = baseframework.from_pretrained(
        args.checkpoint,
        config_overrides=build_overrides(args),
        allow_partial_load=not args.strict_load,
    )
    vla = vla.to(device).eval()

    adapter = None
    if os.environ.get("ITEMS_HANDOVER_LOCATOR", "0") == "1":
        with contextlib.redirect_stdout(sys.stderr):
            from items_handover_locator import ItemsHandoverInferenceAdapter

            adapter = ItemsHandoverInferenceAdapter(device)

    norm_stats = dict(vla.norm_stats)
    if not norm_stats:
        raise RuntimeError("Checkpoint has no normalization statistics.")
    embodiment_key = next(iter(norm_stats))
    stats = norm_stats[embodiment_key]
    action_stats = stats["action"]
    state_stats = stats["state"]
    action_high = np.asarray(action_stats["max"], dtype=np.float64).reshape(-1)
    action_low = np.asarray(action_stats["min"], dtype=np.float64).reshape(-1)
    state_high = np.asarray(state_stats["max"], dtype=np.float64).reshape(-1)
    state_low = np.asarray(state_stats["min"], dtype=np.float64).reshape(-1)
    action_mask = np.asarray(
        action_stats.get("mask", np.ones(action_high.shape, dtype=bool)), dtype=bool
    )
    if action_mask.shape != action_high.shape:
        raise RuntimeError(
            f"Action mask shape {action_mask.shape} != action stats shape {action_high.shape}."
        )
    action_valid = (action_high - action_low) > EPS
    state_valid = (state_high - state_low) > EPS

    action_horizon = int(getattr(vla, "action_horizon", 50))
    print(
        json.dumps(
            {
                "status": "ready",
                "action_horizon": action_horizon,
                "action_dim": int(action_high.shape[0]),
                "embodiment": embodiment_key,
                "locator": adapter is not None,
            }
        ),
        flush=True,
    )

    def normalize_state(state: np.ndarray) -> np.ndarray:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        span = np.where(state_valid, state_high - state_low, 1.0)
        return np.where(state_valid, (state - state_low) / span * 2.0 - 1.0, state)

    def unnormalize_actions(normalized: np.ndarray) -> np.ndarray:
        normalized = np.clip(normalized, -1.0, 1.0)
        span = np.where(action_valid, action_high - action_low, 1.0)
        continuous = action_mask & action_valid
        actions = np.where(
            continuous,
            0.5 * (normalized + 1.0) * span + action_low,
            (normalized > BINARY_THRESHOLD).astype(np.float64),
        )
        # Dims that never varied in training (e.g. a constant gripper) must
        # stay at their trained constant value, not flip through the binary
        # branch where their (meaningless) normalized output could exceed
        # BINARY_THRESHOLD.
        actions = np.where(action_valid, actions, action_low)
        return actions

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            print(json.dumps({"ok": False, "error": "invalid JSON request"}), flush=True)
            continue

        cmd = request.get("cmd")
        try:
            if cmd == "ping":
                response = {"ok": True}
            elif cmd == "reset":
                # TurboVLA is stateless across episodes; nothing to reset.
                if adapter is not None:
                    with contextlib.redirect_stdout(sys.stderr):
                        adapter.reset()
                response = {"ok": True}
            elif cmd == "infer":
                data = np.load(request["obs_path"])
                images = []
                for cam in args.cam_order:
                    img = np.asarray(data[cam], dtype=np.uint8)
                    if img.ndim != 3 or img.shape[-1] not in (1, 3, 4):
                        raise ValueError(
                            f"expected HWC uint8 image for {cam}, got {img.shape}"
                        )
                    if img.shape[-1] == 1:
                        img = np.repeat(img, 3, axis=-1)
                    images.append(Image.fromarray(img[..., :3]))
                state = normalize_state(data["state"])
                lang = str(request.get("lang", ""))
                if adapter is not None:
                    # The locator's detection prints must stay off the RPC
                    # stdout channel.
                    with contextlib.redirect_stdout(sys.stderr):
                        images, lang = apply_locator(
                            adapter, images, lang, args.cam_order
                        )
                with torch.inference_mode():
                    output = vla.predict_action(
                        [
                            {
                                "image": images,
                                "lang": lang,
                                "state": state,
                            }
                        ]
                    )
                normalized_actions = np.asarray(
                    output["normalized_actions"], dtype=np.float64
                )[0]
                actions = unnormalize_actions(normalized_actions).tolist()
                if adapter is not None:
                    with contextlib.redirect_stdout(sys.stderr):
                        # Every returned chunk action is eventually executed
                        # by the parent's action queue, so advancing by the
                        # chunk length tracks executed steps.
                        adapter.advance(len(actions))
                response = {"ok": True, "actions": actions}
            else:
                response = {"ok": False, "error": f"unknown cmd: {cmd!r}"}
        except Exception as exc:  # report failures back over RPC
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(response), flush=True)


if __name__ == "__main__":
    main()
