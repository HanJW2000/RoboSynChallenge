# scv-think-tank RoboSynChallenge submission

This repository is the reproducible evaluation code for the
`HanJW2000/RoboSynChallenge` checkpoint repository.  It keeps the official
RoboSynChallenge simulator and evaluation entry points, and adds the exact
TurboVLA and Diffusion Policy adapters used for the submitted results.

## Selected policies

| Task | Policy | Local seed-0 success |
|---|---|---:|
| `table_rearrangement` | TurboVLA | 99/100 |
| `handle_basket` | TurboVLA | 96/100 |
| `mixer_operating` | TurboVLA | 95/100 |
| `item_assembly` | TurboVLA | 76/100 |
| `water_pouring` | TurboVLA | 74/100 |
| `items_handover` | TurboVLA | 64/100 |
| `click_bell` | Diffusion Policy Hard450 | 66/100 |
| `manipulate_pipette` | official ACT fallback | 61/100 official reference |
| `drawer_open_place` | official SmolVLA fallback | 62/100 official reference |
| `sample_loading` | official SmolVLA fallback | 6/100 official reference |

The 69.9% aggregate is a public-seed reference, not a claim about the hidden
organizer evaluation.

## 1. Source completeness

The official preparation retains only `policy/`, so all participant-provided
files are fully vendored there. TurboVLA is not a Git submodule and no
source checkout from the training machine is required:

- `policy/turbovla/third_party/TurboVLA/turbovla`: TurboVLA package;
- `policy/turbovla/third_party/TurboVLA/third_party/starvla_runtime/starVLA`:
  starVLA runtime;
- `policy/turbovla/third_party/TurboVLA/pyproject.toml`: package declaration;
- `policy/turbovla/third_party/TurboVLA/uv.lock`: immutable UV dependency lock;
- `policy/turbovla/third_party/TurboVLA/scripts/robotwin/requirements.txt`:
  readable RoboTwin
  dependency list.

The checkpoint repository contains model weights only.  BERT, DINOv3, and
GroundingDINO under `dependencies/` are runtime weights, not source-code
replacements.

## 2. Install

Follow the official repository setup to install RoboSynChallenge and
EmbodiChain.  Install the policy environments as documented under
`policy/act`, `policy/dp`, and `policy/smolvla`.  TurboVLA source is vendored at
`policy/turbovla/third_party/TurboVLA`. The submitted adapter was validated with Python
3.11.16, PyTorch 2.7.0, torchvision 0.22.0, Transformers 4.57.6, and timm
1.0.29.  The patch component of Python 3.11 may vary.
Create its separate UV environment from the repository root:

```bash
bash policy/turbovla/install_uv.sh
```

The equivalent explicit commands are:

```bash
uv sync \
  --project policy/turbovla/third_party/TurboVLA \
  --python 3.11 \
  --frozen \
  --extra robotwin \
  --no-dev
```

FlashAttention is not required for evaluation: the submitted worker defaults
to PyTorch SDPA.  Set the two environment Python paths before evaluation:

```bash
export ROBOSYN_VENV_DIR=/absolute/path/to/EmbodiChain/.venv
export TURBOVLA_PYTHON="$PWD/policy/turbovla/third_party/TurboVLA/.venv/bin/python"
```

`ROBOSYN_VENV_DIR` is the organizer's simulator environment;
`TURBOVLA_PYTHON` is the separate UV environment above.  The evaluation
adapter communicates with TurboVLA through JSON-line RPC, so the two dependency
stacks do not need to be merged.

## 3. Download checkpoints

```bash
bash policy/download_submission_checkpoints.sh /absolute/path/to/checkpoints
```

This downloads Hugging Face revision
`924324d49eeca83e3d23a035ee3a83168a0ac527` and the three pinned official
fallback revisions.  Verify the submitted files with:

```bash
cd /absolute/path/to/checkpoints/submission
sha256sum -c SHA256SUMS
```

## 4. Evaluate one task

```bash
bash policy/eval_submission_task.sh \
  table_rearrangement random 0 /absolute/path/to/checkpoints \
  --headless True --max_episodes 100 --eval_video_log True
```

Arguments are `<task> <setting> <gpu_id> <checkpoint_root> [extra overrides]`.
Run the same command with any of the ten task names in the table above.  The
router selects the submitted policy and exact checkpoint automatically.

## Data declaration

The policies use the released RoboSynChallenge simulation datasets.  The
`click_bell` checkpoint additionally uses 450 hard-position trajectories
generated with the official simulator, task definition, and assets.  No
external demonstration dataset is used.  TurboVLA uses pretrained vision and
text weights, all included in the checkpoint repository for offline loading.
