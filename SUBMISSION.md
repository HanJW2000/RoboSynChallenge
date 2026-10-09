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

## 1. Install

Follow the official repository setup to install RoboSynChallenge and
EmbodiChain.  Install the policy environments as documented under
`policy/act`, `policy/dp`, and `policy/smolvla`.  TurboVLA source is vendored at
`third_party/TurboVLA`; install it into a separate environment:

```bash
python -m venv .venv-turbovla
source .venv-turbovla/bin/activate
pip install -e third_party/TurboVLA
```

Set the environment Python paths before evaluation:

```bash
export ROBOSYN_VENV_DIR=/absolute/path/to/EmbodiChain/.venv
export TURBOVLA_PYTHON=/absolute/path/to/.venv-turbovla/bin/python
```

## 2. Download checkpoints

```bash
bash scripts/download_submission_checkpoints.sh /absolute/path/to/checkpoints
```

This downloads the submitted HF repository and the three pinned official
fallback checkpoints.  Verify the submitted files with:

```bash
cd /absolute/path/to/checkpoints/submission
sha256sum -c SHA256SUMS
```

## 3. Evaluate one task

```bash
bash scripts/eval_submission_task.sh \
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
