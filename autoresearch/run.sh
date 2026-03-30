#!/bin/bash
# Autoresearch main runner
# Usage: bash autoresearch/run.sh [gpu_id]
#
# Launches Claude Code to autonomously optimize CF algorithm across ALL environments.
# Each invocation runs one full experiment cycle per environment.

set -euo pipefail

cd "$(dirname "$0")/.."

GPU_ID="${1:-0}"

# All environments in optimization order
ENVS=(
    "coin_game"
    "pd_arena"
    "gift"
    "coop_mining"
    "mushrooms"
    "clean_up"
    "harvest_common_open"
    "territory_open"
)

# Map env name to CF script filename
declare -A ENV_TO_SCRIPT=(
    ["coin_game"]="cf_cnn_coins"
    ["pd_arena"]="cf_cnn_pd_arena"
    ["gift"]="cf_cnn_gift"
    ["coop_mining"]="cf_cnn_coop_mining"
    ["mushrooms"]="cf_cnn_mushrooms"
    ["clean_up"]="cf_cnn_cleanup"
    ["harvest_common_open"]="cf_cnn_harvest_common"
    ["territory_open"]="cf_cnn_territory_open"
)

# Create autoresearch branch
BRANCH="autoresearch/all_envs_$(date +%Y%m%d_%H%M%S)"
git checkout -b "$BRANCH"

echo "=========================================="
echo "AUTORESEARCH RUNNER - ALL ENVIRONMENTS"
echo "GPU: $GPU_ID"
echo "Branch: $BRANCH"
echo "Environments: ${ENVS[*]}"
echo "Started: $(date)"
echo "=========================================="

EXPERIMENT_NUM=0

while true; do
    for ENV_NAME in "${ENVS[@]}"; do
        EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
        SCRIPT_NAME="${ENV_TO_SCRIPT[$ENV_NAME]}"

        echo ""
        echo ">>> Experiment #${EXPERIMENT_NUM} | ${ENV_NAME} | starting at $(date)"
        echo ""

        ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic" \
        ANTHROPIC_AUTH_TOKEN="0ed2bd422a494115900375d3095578f6.xvYnx5wzrCUKgC3E" \
        claude --model GLM-5.1 \
            --dangerously-skip-permissions \
            --print \
            "You are an autonomous ML research agent optimizing the CF algorithm for SocialJax.

TARGET: ${ENV_NAME} | GPU: ${GPU_ID} | Experiment #${EXPERIMENT_NUM}
CF script: algorithms/CF/${SCRIPT_NAME}.py
CF config: algorithms/CF/config/${SCRIPT_NAME}.yaml

## Your task for THIS invocation

1. Read autoresearch/program.md for research directions and context.
2. Read autoresearch/results.tsv to see past experiment results and the current best for ${ENV_NAME}.
3. Read the current CF code: algorithms/CF/${SCRIPT_NAME}.py and its config.
4. Think of ONE improvement idea. Consider what has been tried (results.tsv) and what hasn't.
5. Modify the CF code (algorithms/CF/${SCRIPT_NAME}.py) or config with your idea.
6. Git commit with message: 'autoresearch(${ENV_NAME}): <brief description of change>'
7. Run: bash autoresearch/run_experiment.sh ${ENV_NAME} ${GPU_ID} 2>&1 | tee run.log
8. Extract result: grep 'FINAL_METRIC:' run.log | tail -1
9. Compare to best known result for ${ENV_NAME} in results.tsv.
10. If improved: keep commit, append to results.tsv with status 'keep'.
11. If NOT improved or crashed: run 'git reset --hard HEAD~1', append to results.tsv with status 'revert'.

IMPORTANT:
- Make exactly ONE change per experiment.
- Only modify algorithms/CF/${SCRIPT_NAME}.py and algorithms/CF/config/${SCRIPT_NAME}.yaml.
- Keep code clean and JAX-compatible (no dynamic shapes inside jax.lax.scan).
- If the run crashes, revert and note why in results.tsv.
- After logging results, you are DONE for this invocation."

        echo ""
        echo ">>> Experiment #${EXPERIMENT_NUM} | ${ENV_NAME} | completed at $(date)"
        echo ""

        sleep 5
    done

    echo ""
    echo "=== Completed one full round of all environments. Starting next round... ==="
    echo ""
done
