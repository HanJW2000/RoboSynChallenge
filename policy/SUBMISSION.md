# scv-think-tank policy-only evaluation package

This directory is self-contained for the RoboSynChallenge preparation process,
which retains only `policy/` from the participant repository. No participant
source file outside this directory is required for installation or evaluation.

## Included participant components

- `turbovla/`: RoboSyn adapter, worker, locator, evaluator launcher, and the
  complete vendored TurboVLA source;
- `turbovla/third_party/TurboVLA/third_party/starvla_runtime/starVLA/`:
  complete vendored starVLA runtime source;
- `turbovla/third_party/TurboVLA/{pyproject.toml,uv.lock,.python-version}`:
  reproducible Python dependency specification;
- `turbovla/third_party/TurboVLA/scripts/robotwin/requirements.txt`:
  readable task-stack requirements;
- `turbovla/install_uv.sh`: separate TurboVLA UV environment installer;
- `eval_submission_task.sh`: ten-task checkpoint and policy router;
- `download_submission_checkpoints.sh`: immutable checkpoint downloader;
- `task_checkpoints.json`: task-to-policy/checkpoint manifest.

No virtual environment, Python bytecode, build output, model weight, simulator
source, evaluator source, task setting, or scoring implementation is vendored
under `policy/`.

## Install TurboVLA

The validated inference stack used Python 3.11.16, PyTorch 2.7.0+cu126,
torchvision 0.22.0+cu126, Transformers 4.57.6, timm 1.0.29, NumPy 1.26.4,
OmegaConf 2.3.0, and Accelerate 1.5.2.

From the official RoboSynChallenge repository root after inserting this policy
tree:

```bash
bash policy/turbovla/install_uv.sh
export TURBOVLA_PYTHON="$PWD/policy/turbovla/third_party/TurboVLA/.venv/bin/python"
```

The installer runs:

```bash
uv sync \
  --project policy/turbovla/third_party/TurboVLA \
  --python 3.11 \
  --frozen \
  --extra robotwin \
  --no-dev
```

The TurboVLA environment is separate from the RoboSynChallenge/EmbodiChain
simulator environment. FlashAttention is not required: the worker defaults to
PyTorch SDPA.

## Download immutable checkpoints

```bash
bash policy/download_submission_checkpoints.sh /absolute/path/to/checkpoint-root
```

This downloads submission revision
`924324d49eeca83e3d23a035ee3a83168a0ac527` plus the three pinned official
fallback repositories recorded in `policy/task_checkpoints.json`.

## Evaluate any of the ten tasks

```bash
bash policy/eval_submission_task.sh \
  table_rearrangement random 0 /absolute/path/to/checkpoint-root \
  --headless True --max_episodes 100 --eval_video_log True
```

Arguments are `<task> <setting> <gpu_id> <checkpoint_root> [extra overrides]`.
The same command supports all ten task names in `task_checkpoints.json`.

Runtime paths are configurable:

- `ROBOSYN_VENV_DIR`: simulator Python environment;
- `EMBODICHAIN_ROOT`: EmbodiChain checkout;
- `TURBOVLA_ROOT`: vendored source override (default is inside `policy/`);
- `TURBOVLA_PYTHON`: TurboVLA worker Python override;
- `TURBOVLA_CHECKPOINTS_ROOT`: BERT/DINOv3/GroundingDINO weights override.

The packaging correction changes only file placement and path discovery. Model
preprocessing, inference, action generation, action-queue control, and timing
behavior are unchanged.
