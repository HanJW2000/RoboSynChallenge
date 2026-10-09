# ----------------------------------------------------------------------------
# TurboVLA Policy Adapter for RoboSynChallenge
#
# Worker-subprocess pattern (same as policy/smolvla): the model runs in the
# dedicated turbovla venv (separate dependency stack) and is driven over
# stdin/stdout JSON-line RPC. This adapter only handles env stepping and
# action-queue management in the protocol eval process.
#
# Unified evaluation interface:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> final_obs, info, truncated, inference_times_s
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

from __future__ import annotations

import atexit
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from policy.inference_timing import finish_inference, start_inference


DEFAULT_TURBOVLA_PYTHON = "/supconit-data/robosyn/envs/turbovla/bin/python"
DEFAULT_TURBOVLA_ROOT = "/supconit-data/code/RoboSynChallenge_ws/TurboVLA"
DEFAULT_CHECKPOINTS_ROOT = "/supconit-data/robosyn/checkpoints/turbovla"
DEFAULT_CAM_ORDER = ["cam_high", "cam_left_wrist", "cam_right_wrist"]
DEFAULT_TASK = "Click the bell"


def _to_numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _extract_image(obs, sensor_name: str, env_index: int = 0) -> np.ndarray:
    """Return an HWC uint8 numpy image from an obs sensor entry."""
    image = _to_numpy(obs["sensor"][sensor_name]["color"])
    # The protocol env may or may not carry a leading env-batch dim.
    if image.ndim == 4 and image.shape[0] <= 8:
        image = image[env_index]
    if image.ndim != 3:
        raise ValueError(f"Expected image for {sensor_name} to be 3D, got {image.shape}")

    # Normalize to HWC.
    if image.shape[-1] in (1, 3, 4):
        hwc = image
    elif image.shape[0] in (1, 3, 4):
        hwc = np.moveaxis(image, 0, -1)
    else:
        raise ValueError(
            f"Cannot detect channel layout for {sensor_name}: {image.shape}"
        )

    if hwc.shape[-1] == 1:
        hwc = np.repeat(hwc, 3, axis=-1)
    elif hwc.shape[-1] > 3:
        hwc = hwc[..., :3]

    if np.issubdtype(hwc.dtype, np.floating):
        if float(np.nanmax(hwc)) <= 1.0:
            hwc = hwc * 255.0
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)
    elif hwc.dtype != np.uint8:
        hwc = np.clip(hwc, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(hwc)


def _extract_state(obs, env_index: int = 0) -> np.ndarray:
    value = _to_numpy(obs["robot"]["qpos"])
    # Same env-batch-dim heuristic as _extract_image.
    if value.ndim == 2 and value.shape[0] <= 8:
        value = value[env_index]
    value = value.astype(np.float64, copy=False).reshape(-1)
    return np.ascontiguousarray(value)


class TurboVLAWorkerClient:
    def __init__(self, usr_args: dict):
        if not usr_args.get("checkpoint_path"):
            raise ValueError(
                "checkpoint_path must be provided (path to steps_*_ema_model.safetensors)."
            )
        self.checkpoint_path = Path(usr_args["checkpoint_path"]).expanduser().resolve()
        self.python_bin = (
            usr_args.get("turbovla_python")
            or os.environ.get("TURBOVLA_PYTHON")
            or DEFAULT_TURBOVLA_PYTHON
        )
        self.tv_root = Path(
            usr_args.get("turbovla_root")
            or os.environ.get("TURBOVLA_ROOT")
            or DEFAULT_TURBOVLA_ROOT
        )
        self.runtime_root = Path(
            usr_args.get("turbovla_starvla_runtime")
            or (self.tv_root / "third_party" / "starvla_runtime")
        )
        checkpoints_root = Path(
            usr_args.get("turbovla_checkpoints_root") or DEFAULT_CHECKPOINTS_ROOT
        )
        self.dinov3_path = usr_args.get("turbovla_dinov3_path") or str(
            checkpoints_root / "dinov3-vitl16"
        )
        self.bert_path = usr_args.get("turbovla_bert_path") or str(
            checkpoints_root / "bert-base-uncased"
        )
        self.gpu_id = int(usr_args.get("gpu_id", 0))
        self.device = usr_args.get("turbovla_device") or None
        self.cuda_visible_devices = str(
            usr_args.get("turbovla_cuda_visible_devices", usr_args.get("gpu_id", 0))
        )
        self.steps_per_eval = int(usr_args.get("turbovla_steps", 10))
        if self.steps_per_eval <= 0:
            raise ValueError(f"turbovla_steps must be positive, got {self.steps_per_eval}.")
        self.cam_order = list(usr_args.get("turbovla_cam_order") or DEFAULT_CAM_ORDER)
        self.task_description = usr_args.get("task_description") or DEFAULT_TASK
        self.task_description_explicit = bool(usr_args.get("task_description"))
        self.action_queue: list[np.ndarray] = []

        worker_path = Path(__file__).resolve().parent / "turbovla_worker.py"
        worker_env = os.environ.copy()
        # Keep the simulator parent unmasked (DexSim uses physical GPU
        # ordinals); mask only the worker, like the SmolVLA adapter.
        worker_env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices
        worker_env["PYTHONPATH"] = os.pathsep.join(
            [str(self.tv_root), str(self.runtime_root)]
            + ([worker_env["PYTHONPATH"]] if worker_env.get("PYTHONPATH") else [])
        )
        worker_env["HF_HUB_OFFLINE"] = "1"
        worker_env["TRANSFORMERS_OFFLINE"] = "1"
        worker_env["HF_HUB_DISABLE_XET"] = "1"
        # Checkpoint configs may reference these via ${oc.env:...}; resolution
        # happens inside the worker's read_mode_config.
        self.init_ckpt = usr_args.get("turbovla_init_ckpt") or str(
            checkpoints_root / "groundingdino" / "groundingdino_swint_ogc.pth"
        )
        worker_env["DINOV3_MODEL_PATH"] = self.dinov3_path
        worker_env["BERT_MODEL_PATH"] = self.bert_path
        worker_env["TURBOVLA_INIT_CKPT"] = self.init_ckpt
        worker_cmd = [
            self.python_bin,
            str(worker_path),
            "--checkpoint",
            str(self.checkpoint_path),
            "--gpu-id",
            # the worker env is masked by turbovla_cuda_visible_devices, so it
            # sees exactly one device; always address it as cuda:0
            "0",
            "--dinov3-path",
            self.dinov3_path,
            "--bert-path",
            self.bert_path,
            "--init-ckpt",
            self.init_ckpt,
        ]
        if self.device:
            worker_cmd.extend(["--device", str(self.device)])
        self.proc = subprocess.Popen(
            worker_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=worker_env,
        )
        atexit.register(self.close)
        self._wait_ready()

    def _wait_ready(self) -> None:
        while True:
            if self.proc.stdout is None:
                raise RuntimeError("TurboVLA worker has no stdout pipe.")
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"TurboVLA worker exited before reporting ready (rc={self.proc.poll()})."
                )
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") == "ready":
                self.action_horizon = int(payload.get("action_horizon", 50))
                return
            if not payload.get("ok", False):
                raise RuntimeError(
                    payload.get("error", "TurboVLA worker failed to start.")
                )

    def _rpc(self, payload: dict) -> dict:
        if self.proc.stdin is None or self.proc.stdout is None:
            raise RuntimeError("TurboVLA worker pipes are not available.")
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"TurboVLA worker exited unexpectedly (rc={self.proc.poll()})."
                )
            line = line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not response.get("ok", False):
                raise RuntimeError(response.get("error", "TurboVLA worker error"))
            return response

    def close(self) -> None:
        self.action_queue.clear()
        if getattr(self, "proc", None) is None or self.proc.poll() is not None:
            return
        if self.proc.stdin is not None:
            try:
                self.proc.stdin.close()
            except Exception:
                pass
        try:
            self.proc.terminate()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
            self.proc.wait(timeout=5)

    def infer(self, obs, lang: str) -> list[np.ndarray]:
        state = _extract_state(obs)
        fd, obs_path = tempfile.mkstemp(prefix="turbovla_obs_", suffix=".npz")
        os.close(fd)
        arrays = {"state": state}
        try:
            for cam in self.cam_order:
                arrays[cam] = _extract_image(obs, cam)
            np.savez(obs_path, **arrays)
            response = self._rpc({"cmd": "infer", "obs_path": obs_path, "lang": lang})
        finally:
            if os.path.exists(obs_path):
                os.remove(obs_path)
        return [np.asarray(action, dtype=np.float32) for action in response["actions"]]

    def reset(self) -> None:
        self.action_queue.clear()
        self._rpc({"cmd": "reset"})


def get_model(usr_args):
    return TurboVLAWorkerClient(usr_args)


def eval(env, model, obs):
    final_obs = obs
    info = {}
    truncated = False
    inference_times_s = []

    for _ in range(model.steps_per_eval):
        if len(model.action_queue) == 0:
            started_at = start_inference()
            lang = (
                model.task_description
                if model.task_description_explicit
                else (getattr(env, "_current_instruction", None) or model.task_description)
            )
            model.action_queue.extend(model.infer(final_obs, lang))
            finish_inference(started_at, inference_times_s)

        action = model.action_queue.pop(0)
        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        if action.shape[0] != env_action_dim:
            raise ValueError(
                f"TurboVLA action has dim {action.shape[0]}, but env expects {env_action_dim}."
            )
        action_tensor = torch.as_tensor(
            action, dtype=torch.float32, device=env.unwrapped.device
        ).unsqueeze(0)
        final_obs, reward, terminated, truncated, info = env.step(action_tensor)
        if env.get_wrapper_attr("is_task_success")():
            break
        if isinstance(truncated, torch.Tensor):
            is_truncated = truncated.any().item()
        elif isinstance(truncated, np.ndarray):
            is_truncated = truncated.any()
        else:
            is_truncated = bool(truncated)
        if is_truncated:
            break

    return final_obs, info, truncated, inference_times_s


def reset_model(model):
    model.reset()


def close_model(model):
    model.close()
