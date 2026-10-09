#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:?Usage: $0 <checkpoint_root>}"
mkdir -p "$ROOT/submission" "$ROOT/official"

hf download HanJW2000/RoboSynChallenge \
  --local-dir "$ROOT/submission"

hf download RoboSynChallenge/ACT_sim_manipulate_pipette \
  --revision ca013b2e202d594bf347530c61c3ba28c06d1bc2 \
  --local-dir "$ROOT/official/ACT_sim_manipulate_pipette"

hf download RoboSynChallenge/SmolVLA_sim_drawer_open_place \
  --revision c0088d84a568f93fb4401aabafcc41cf643efcdd \
  --local-dir "$ROOT/official/SmolVLA_sim_drawer_open_place"

hf download RoboSynChallenge/SmolVLA_sim_sample_loading \
  --revision 2a4f41ab3be8e5f544c890d6054bc8de3be5c53b \
  --local-dir "$ROOT/official/SmolVLA_sim_sample_loading"

echo "Downloaded submission checkpoints under: $ROOT"
