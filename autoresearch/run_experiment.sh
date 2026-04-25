#!/bin/bash
# Autoresearch v2 experiment runner for CF algorithm
# Usage: bash autoresearch/run_experiment.sh <env_name> [gpu_id]
#
# - Runs CF training with paper-aligned config (NUM_ENVS=64 NUM_STEPS=128)
# - 3e8 timesteps, 7 hour hard timeout (6h budget + 1h grace)
# - Uses wandb online + run_with_hook.py to print METRIC: lines
#
# Outputs lines:
#   METRIC: env_step=... returned_episode_returns=...   (shaped, training reward)
#   METRIC_RAW: env_step=... returned_episode_original_returns=...   (paper-aligned, only for SVO/CF wrapped envs)

set -uo pipefail

ENV_NAME="${1:?Usage: run_experiment.sh <env_name> [gpu_id]}"
GPU_ID="${2:-0}"
TIMEOUT_SECONDS=25200  # 7h hard kill (6h budget + 1h grace)

# Map environment name to CF script filename
declare -A ENV_TO_CONFIG=(
    ["clean_up"]="cf_cnn_cleanup"
    ["coop_mining"]="cf_cnn_coop_mining"
    ["mushrooms"]="cf_cnn_mushrooms"
    ["harvest_common_open"]="cf_cnn_harvest_common"
)

CONFIG_NAME="${ENV_TO_CONFIG[$ENV_NAME]:-}"
if [ -z "$CONFIG_NAME" ]; then
    echo "ERROR: Unknown environment: $ENV_NAME"
    echo "Available: ${!ENV_TO_CONFIG[*]}"
    exit 1
fi

SCRIPT_FILE="algorithms/CF/${CONFIG_NAME}.py"
if [ ! -f "$SCRIPT_FILE" ]; then
    echo "ERROR: Script not found: $SCRIPT_FILE"
    exit 1
fi

# GUARD: reject if code or config hardcodes shared_rewards=True
CONFIG_FILE="algorithms/CF/config/${CONFIG_NAME}.yaml"
if grep -q "shared_rewards.*True\|shared_rewards.*true" "$SCRIPT_FILE" "$CONFIG_FILE" 2>/dev/null; then
    echo "ERROR: BLOCKED — shared_rewards=True detected in code/config!"
    echo "This VIOLATES the decentralized constraint. Fix the code first."
    exit 2
fi

# GUARD: reject if CF reward model or counterfactual computation has been removed
if ! grep -q "compute_cf_shaped_reward\|RewardModel\|reward_model" "$SCRIPT_FILE" 2>/dev/null; then
    echo "ERROR: BLOCKED — CF reward model / counterfactual computation not found!"
    echo "The CF pipeline (reward model + counterfactual + regret) MUST be present."
    echo "You cannot replace CF with TEAM_BLEND or other non-CF methods."
    exit 2
fi

PY=/home/shuqing/.conda/envs/melting-jax/bin/python
export PYTHONPATH="./socialjax:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"

echo "=========================================="
echo "AUTORESEARCH v2 EXPERIMENT"
echo "Environment: $ENV_NAME"
echo "Script: $SCRIPT_FILE"
echo "GPU: $GPU_ID"
echo "Timeout: ${TIMEOUT_SECONDS}s"
echo "Started: $(date)"
echo "=========================================="

# HARD CONSTRAINT: shared_rewards MUST be False (decentralized requirement)
# This override is applied LAST so it cannot be overridden by the CF script's config
timeout "$TIMEOUT_SECONDS" $PY scripts/run_with_hook.py "$SCRIPT_FILE" \
    TOTAL_TIMESTEPS=3e8 \
    WANDB_MODE=online \
    SEED=30 \
    NUM_ENVS=64 \
    NUM_STEPS=128 \
    NUM_MINIBATCHES=16 \
    UPDATE_EPOCHS=2 \
    ++ENV_KWARGS.shared_rewards=False

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 124 ]; then
    echo "FINAL_METRIC:TIMEOUT"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "FINAL_METRIC:CRASHED_${EXIT_CODE}"
fi

exit $EXIT_CODE
