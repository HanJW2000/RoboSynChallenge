#!/usr/bin/env bash
set -euo pipefail

TASK="${1:?Usage: $0 <task> <setting> <gpu_id> <checkpoint_root> [extra overrides...]}"
SETTING="${2:?Missing setting}"
GPU_ID="${3:?Missing gpu_id}"
ROOT="${4:?Missing checkpoint_root}"
shift 4

POLICY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$POLICY_ROOT/.." && pwd)"
SUBMISSION="$ROOT/submission"
OFFICIAL="$ROOT/official"

case "$TASK" in
  table_rearrangement|handle_basket|mixer_operating|item_assembly|water_pouring|items_handover)
    CHECKPOINT="$SUBMISSION/checkpoints/$TASK/turbovla/final_model/ema_ema_pytorch_model.pt"
    export TURBOVLA_ROOT="${TURBOVLA_ROOT:-$POLICY_ROOT/turbovla/third_party/TurboVLA}"
    bash "$POLICY_ROOT/turbovla/eval.sh" \
      "$TASK" "$SETTING" "$CHECKPOINT" "$GPU_ID" \
      --turbovla_checkpoints_root "$SUBMISSION/dependencies" "$@"
    ;;
  click_bell)
    CHECKPOINT="$SUBMISSION/checkpoints/click_bell/dp/pretrained_model"
    bash "$POLICY_ROOT/dp/eval.sh" \
      "$TASK" "$SETTING" "$CHECKPOINT" "$GPU_ID" "$@"
    ;;
  manipulate_pipette)
    CHECKPOINT="$OFFICIAL/ACT_sim_manipulate_pipette"
    bash "$POLICY_ROOT/act/eval.sh" \
      "$TASK" "$SETTING" "$CHECKPOINT" "$GPU_ID" "$@"
    ;;
  drawer_open_place)
    CHECKPOINT="$OFFICIAL/SmolVLA_sim_drawer_open_place"
    bash "$POLICY_ROOT/smolvla/eval.sh" \
      "$TASK" "$SETTING" "$CHECKPOINT" "$GPU_ID" "$@"
    ;;
  sample_loading)
    CHECKPOINT="$OFFICIAL/SmolVLA_sim_sample_loading"
    bash "$POLICY_ROOT/smolvla/eval.sh" \
      "$TASK" "$SETTING" "$CHECKPOINT" "$GPU_ID" "$@"
    ;;
  *)
    echo "Unknown task: $TASK" >&2
    exit 2
    ;;
esac
