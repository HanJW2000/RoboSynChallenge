#!/bin/bash
# ----------------------------------------------------------------------------
# bash eval.sh <task_name> <setting> <checkpoint_safetensors> [gpu_id] [extra_opts...]
# bash eval.sh click_bell random /path/to/steps_55000_ema_model.safetensors 1 \
#   --headless True --max_episodes 100
# ----------------------------------------------------------------------------

export EMBODICHAIN_SIM_EXIT_PROCESS=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
if [[ -z "${EMBODICHAIN_ROOT:-}" ]]; then
    if [[ -d "$WORKSPACE_ROOT/EmbodiChain" ]]; then
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    else
        EMBODICHAIN_ROOT="$WORKSPACE_ROOT/EmbodiChain"
    fi
fi
VENV_DIR="${ROBOSYN_VENV_DIR:-$EMBODICHAIN_ROOT/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
TURBOVLA_ROOT="${TURBOVLA_ROOT:-$SCRIPT_DIR/third_party/TurboVLA}"
TURBOVLA_PYTHON="${TURBOVLA_PYTHON:-$TURBOVLA_ROOT/.venv/bin/python}"

POLICY_NAME=turbovla

TASK_NAME="${1:?Usage: bash policy/turbovla/eval.sh <task_name> <setting> <checkpoint_safetensors> [gpu_id] [extra_opts...]}"
SETTING="${2:?Usage: bash policy/turbovla/eval.sh <task_name> <setting> <checkpoint_safetensors> [gpu_id] [extra_opts...]}"
CHECKPOINT_PATH="${3:?Usage: bash policy/turbovla/eval.sh <task_name> <setting> <checkpoint_safetensors> [gpu_id] [extra_opts...]}"
GPU_ID="${4:-0}"

shift 4 2>/dev/null || true
EXTRA_ARGS=("$@")

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

EXTRA_ARGS+=(--turbovla_python "$TURBOVLA_PYTHON")
EXTRA_ARGS+=(--turbovla_root "$TURBOVLA_ROOT")

# DexSim uses physical GPU ordinals from --gpu_id. Keep the simulator parent
# unmasked, then mask only the TurboVLA worker via --turbovla_cuda_visible_devices.
unset CUDA_VISIBLE_DEVICES
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  TurboVLA Policy Evaluation"
echo "  Task:       $TASK_NAME ($SETTING)"
echo "  Checkpoint: $CHECKPOINT_PATH"
echo "  GPU:        $GPU_ID"
echo "  Python:     $PYTHON_BIN"
echo "========================================="

if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: cannot find executable Python: $PYTHON_BIN" >&2
    exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
    echo "Error: checkpoint file does not exist: $CHECKPOINT_PATH" >&2
    exit 1
fi

if [[ ! -x "$TURBOVLA_PYTHON" ]]; then
    echo "Error: cannot find TurboVLA Python: $TURBOVLA_PYTHON" >&2
    echo "Run: bash policy/turbovla/install_uv.sh" >&2
    exit 1
fi

if [[ ! -d "$TURBOVLA_ROOT/third_party/starvla_runtime/starVLA" ]]; then
    echo "Error: vendored starVLA source is missing under: $TURBOVLA_ROOT" >&2
    exit 1
fi

cd "$REPO_ROOT"

PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config policy/$POLICY_NAME/deploy_policy.yml \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --model_name "$(basename "$(dirname "$CHECKPOINT_PATH")")" \
    --gpu_id "$GPU_ID" \
    --turbovla_cuda_visible_devices "$GPU_ID" \
    "${EXTRA_ARGS[@]}"
