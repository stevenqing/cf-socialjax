#!/bin/bash
# Autoresearch experiment runner for CF algorithm
# Usage: bash autoresearch/run_experiment.sh <env_name> [gpu_id]
#
# Runs CF training with 3e8 timesteps and 6-hour timeout.
# Outputs FINAL_METRIC:<value> for parsing.

set -euo pipefail

ENV_NAME="${1:?Usage: run_experiment.sh <env_name> [gpu_id]}"
GPU_ID="${2:-0}"
TIMEOUT_SECONDS=25200  # 7 hours hard kill (6h budget + 1h grace)

# Map environment name to config file name
declare -A ENV_TO_CONFIG=(
    ["coin_game"]="cf_cnn_coins"
    ["harvest_common_open"]="cf_cnn_harvest_common"
    ["clean_up"]="cf_cnn_cleanup"
    ["coop_mining"]="cf_cnn_coop_mining"
    ["territory_open"]="cf_cnn_territory_open"
    ["pd_arena"]="cf_cnn_pd_arena"
    ["mushrooms"]="cf_cnn_mushrooms"
    ["gift"]="cf_cnn_gift"
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

export PYTHONPATH="./socialjax:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "=========================================="
echo "AUTORESEARCH EXPERIMENT"
echo "Environment: $ENV_NAME"
echo "Config: $CONFIG_NAME"
echo "GPU: $GPU_ID"
echo "Timeout: ${TIMEOUT_SECONDS}s"
echo "Started: $(date)"
echo "=========================================="

# Override total timesteps to 3e8 via Hydra
# Also disable wandb to avoid clutter (set to offline)
timeout "$TIMEOUT_SECONDS" python "$SCRIPT_FILE" \
    TOTAL_TIMESTEPS=3e8 \
    WANDB_MODE=offline \
    TUNE=False \
    SEED=30

EXIT_CODE=$?

echo "=========================================="
echo "Finished: $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="

if [ $EXIT_CODE -eq 124 ]; then
    echo "FINAL_METRIC:TIMEOUT"
elif [ $EXIT_CODE -ne 0 ]; then
    echo "FINAL_METRIC:CRASHED"
fi

exit $EXIT_CODE
