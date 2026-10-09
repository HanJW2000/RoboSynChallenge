# ----------------------------------------------------------------------------
# LeRobot Diffusion Policy Adapter for RoboSynChallenge
#
# Follows the same unified evaluation interface as policy/pi0:
#   - get_model(usr_args) -> model
#   - eval(env, model, obs) -> obs, info, truncated
#   - reset_model(model) -> None
# ----------------------------------------------------------------------------

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from policy.dp.target_locator import BellLocator, GroundingDinoBackend
from policy.dp.target_overlay import overlay_bell_heatmap, overlay_bell_heatmap_torch
from policy.inference_timing import finish_inference, start_inference


DEFAULT_NUM_INFERENCE_STEPS = 10


def _first_rgb_numpy(image):
    if isinstance(image, torch.Tensor):
        frame = image[0] if image.ndim == 4 else image
        return frame.detach().cpu().numpy()[..., :3]
    array = np.asarray(image)
    frame = array[0] if array.ndim == 4 else array
    return np.ascontiguousarray(frame[..., :3])


def _apply_target_overlay(model, camera_name, image):
    """Apply the cached bell hint to cam_high while preserving input layout."""
    locator = getattr(model, "bell_locator", None)
    if camera_name != "cam_high" or locator is None:
        return image
    detector_input = image if locator.has_cached_result else _first_rgb_numpy(image)
    detection = locator.detect_once(detector_input)
    if isinstance(image, torch.Tensor):
        return overlay_bell_heatmap_torch(image, detection)
    array = np.asarray(image)
    if array.ndim == 4:
        return np.stack(
            [overlay_bell_heatmap(frame, detection) for frame in array], axis=0
        )
    return overlay_bell_heatmap(array, detection)


def get_model(usr_args):
    checkpoint_path = usr_args.get("checkpoint_path")
    if checkpoint_path is None:
        raise ValueError("checkpoint_path must be provided in usr_args.")

    device = usr_args.get("device", usr_args.get("pytorch_device", "cuda"))
    cli_overrides = [f"--device={device}"]
    num_inference_steps = usr_args.get("dp_num_inference_steps")
    if num_inference_steps is None:
        num_inference_steps = DEFAULT_NUM_INFERENCE_STEPS
    num_inference_steps = int(num_inference_steps)
    cli_overrides.append(f"--num_inference_steps={num_inference_steps}")
    try:
        policy = DiffusionPolicy.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
    except TypeError as exc:
        if "cli_overrides" not in str(exc):
            raise
        config = PreTrainedConfig.from_pretrained(
            checkpoint_path,
            cli_overrides=cli_overrides,
        )
        policy = DiffusionPolicy.from_pretrained(checkpoint_path, config=config)
    policy.eval()

    image_key_map = {
        "observation.images.cam_high": "cam_high",
        "observation.images.cam_right_wrist": "cam_right_wrist",
        "observation.images.cam_left_wrist": "cam_left_wrist",
    }
    image_key_map.update(usr_args.get("image_key_map") or {})
    image_keys = list(policy.config.image_features.keys())
    for image_key in image_keys:
        image_key_map.setdefault(
            image_key,
            image_key.removeprefix("observation.images."),
        )

    policy.dp_device = next(policy.parameters()).device
    policy.dp_step = int(usr_args.get("dp_step", 8))
    policy.state_obs_path = usr_args.get("state_obs_path", "robot/qpos")
    policy.strict_action_dim = bool(usr_args.get("strict_action_dim", True))
    policy.dp_image_keys = image_keys
    policy.image_key_map = image_key_map
    policy.bell_locator = None
    if bool(usr_args.get("target_locator_enabled", False)):
        backend = GroundingDinoBackend(
            model_path=usr_args.get("target_locator_model_path"),
            revision=usr_args.get("target_locator_revision"),
            device=usr_args.get("target_locator_device", str(policy.dp_device)),
            prompt=usr_args.get(
                "target_locator_prompt", "service bell. desk bell. call bell."
            ),
            box_threshold=float(usr_args.get("target_locator_box_threshold", 0.30)),
            text_threshold=float(usr_args.get("target_locator_text_threshold", 0.25)),
        )
        policy.bell_locator = BellLocator(
            backend,
            box_threshold=float(usr_args.get("target_locator_box_threshold", 0.30)),
            max_area_fraction=float(
                usr_args.get("target_locator_max_area_fraction", 0.10)
            ),
            min_teal_fraction=float(
                usr_args.get("target_locator_min_teal_fraction", 0.0)
            ),
        )

    if policy.dp_step <= 0:
        raise ValueError(f"dp_step must be positive, got {policy.dp_step}.")

    return policy


def eval(env, model, obs):
    final_obs = obs
    info = None
    truncated = False
    inference_times_s = []

    for _ in range(model.dp_step):
        action_queue = getattr(model, "_queues", {}).get("action")
        runs_model_inference = action_queue is None or len(action_queue) == 0
        started_at = start_inference(model.dp_device) if runs_model_inference else None
        state = final_obs
        for key in str(model.state_obs_path).split("/"):
            if key:
                state = state[key]
        state_tensor = state.detach().to(device=model.dp_device, dtype=torch.float32) if isinstance(state, torch.Tensor) else torch.as_tensor(state, dtype=torch.float32, device=model.dp_device)
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)

        batch = {"observation.state": state_tensor}
        for image_key in model.dp_image_keys:
            camera_name = model.image_key_map.get(
                image_key,
                image_key.removeprefix("observation.images."),
            )
            image = final_obs["sensor"][camera_name]["color"]
            image = _apply_target_overlay(model, camera_name, image)
            image_tensor = image.detach().to(device=model.dp_device, dtype=torch.float32) if isinstance(image, torch.Tensor) else torch.as_tensor(image, dtype=torch.float32, device=model.dp_device)
            if image_tensor.ndim == 3:
                image_tensor = image_tensor.unsqueeze(0)
            image_tensor = image_tensor[..., :3].permute(0, 3, 1, 2).contiguous()
            if torch.max(image_tensor) > 1.5:
                image_tensor = image_tensor / 255.0
            batch[image_key] = image_tensor

        action = model.select_action(batch)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(f"Expected policy action shape [B, D], but got {tuple(action.shape)}.")

        env_action_dim = int(np.prod(env.unwrapped.single_action_space.shape))
        policy_action_dim = int(action.shape[-1])
        if policy_action_dim != env_action_dim:
            message = (
                f"Policy action has dim {policy_action_dim}, but env expects {env_action_dim}. "
                "RoboSynChallenge CobotMagic simulation tasks typically expect 14D actions; "
                "HF RoboSynChallenge/Piper_real_* checkpoints are Piper/Aloha real-robot "
                "policies and commonly output 32D actions, so they cannot be evaluated "
                "directly in CobotMagic simulation without an explicit action mapping."
            )
            if model.strict_action_dim or policy_action_dim < env_action_dim:
                raise ValueError(message)
            action = action[:, :env_action_dim]

        action_tensor = action.detach().to(
            device=env.unwrapped.device,
            dtype=torch.float32,
        )
        if runs_model_inference:
            finish_inference(started_at, inference_times_s, model.dp_device)
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
    locator = getattr(model, "bell_locator", None)
    if locator is not None:
        locator.reset()
