#!/bin/bash
# Single GPU worker for autoresearch
# Usage: bash autoresearch/run_gpu_worker.sh <gpu_id> "<env_list>" <timestamp>
#
# Runs in a git worktree, cycles through assigned environments,
# coordinates with other GPU workers via shared results.tsv.

set -euo pipefail

GPU_ID="$1"
ENV_LIST=($2)
TIMESTAMP="$3"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKTREE="${REPO_ROOT}/autoresearch/worktree-gpu${GPU_ID}"
SHARED_DIR="${REPO_ROOT}/autoresearch/shared"
LOGFILE="${REPO_ROOT}/autoresearch/gpu${GPU_ID}.log"

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

cd "$WORKTREE"

echo "[GPU ${GPU_ID}] Worker started at $(date)" | tee -a "$LOGFILE"
echo "[GPU ${GPU_ID}] Worktree: ${WORKTREE}" | tee -a "$LOGFILE"
echo "[GPU ${GPU_ID}] Environments: ${ENV_LIST[*]}" | tee -a "$LOGFILE"

EXPERIMENT_NUM=0

while true; do
    for ENV_NAME in "${ENV_LIST[@]}"; do
        EXPERIMENT_NUM=$((EXPERIMENT_NUM + 1))
        SCRIPT_NAME="${ENV_TO_SCRIPT[$ENV_NAME]}"

        # Sync: pull latest best code from other GPU before each experiment
        bash "${SHARED_DIR}/sync_best.sh" pull "$ENV_NAME" "$SCRIPT_NAME" "$WORKTREE" 2>&1 | tee -a "$LOGFILE"

        # Get current best for this env from shared results
        CURRENT_BEST=$(bash "${SHARED_DIR}/get_best.sh" "$ENV_NAME" 2>/dev/null || echo "0")

        echo "" | tee -a "$LOGFILE"
        echo "[GPU ${GPU_ID}] === Experiment #${EXPERIMENT_NUM} | ${ENV_NAME} | best=${CURRENT_BEST} | $(date) ===" | tee -a "$LOGFILE"
        echo "" | tee -a "$LOGFILE"

        ANTHROPIC_BASE_URL="https://open.bigmodel.cn/api/anthropic" \
        ANTHROPIC_AUTH_TOKEN="0ed2bd422a494115900375d3095578f6.xvYnx5wzrCUKgC3E" \
        claude --model GLM-5.1 \
            --dangerously-skip-permissions \
            -p \
            "You are an autonomous ML research agent optimizing the CF algorithm for SocialJax.

TARGET: ${ENV_NAME} | GPU: ${GPU_ID} | Experiment #${EXPERIMENT_NUM}
CF script: algorithms/CF/${SCRIPT_NAME}.py
CF config: algorithms/CF/config/${SCRIPT_NAME}.yaml
CURRENT BEST for ${ENV_NAME}: ${CURRENT_BEST}

You are running in a git worktree at: ${WORKTREE}
Shared results file: ${SHARED_DIR}/results.tsv

## Your task for THIS invocation

1. Read autoresearch/program.md for research directions and full context.
2. Read ${SHARED_DIR}/results.tsv to see ALL past experiments (from both GPUs).
3. Read the current CF code: algorithms/CF/${SCRIPT_NAME}.py and its config.
4. Think of ONE improvement idea. Look at what BOTH GPUs have tried (results.tsv) to avoid duplicating experiments.
5. Modify algorithms/CF/${SCRIPT_NAME}.py or its config with your idea.
6. Git commit: git commit -am 'autoresearch(gpu${GPU_ID}/${ENV_NAME}): <brief description>'
7. Run the experiment:
   export PYTHONPATH=./socialjax:\${PYTHONPATH:-}
   export CUDA_VISIBLE_DEVICES=${GPU_ID}
   timeout 25200 python algorithms/CF/${SCRIPT_NAME}.py TOTAL_TIMESTEPS=3e8 WANDB_MODE=offline TUNE=False SEED=30 2>&1 | tee run.log
8. Extract: grep 'FINAL_METRIC:' run.log | tail -1
9. Compare to CURRENT BEST: ${CURRENT_BEST}
10. Log result (use flock for safety):
    bash ${SHARED_DIR}/lock.sh '<commit>\t${ENV_NAME}\t<metric>\t${GPU_ID}\t<keep|revert>\t<description>'
11. If IMPROVED: publish best code for the other GPU to sync:
    bash ${SHARED_DIR}/sync_best.sh publish ${ENV_NAME} ${SCRIPT_NAME} ${WORKTREE}
12. If NOT improved or crashed: git reset --hard HEAD~1
13. Done. Exit cleanly.

IMPORTANT:
- Make exactly ONE change per experiment.
- Only modify algorithms/CF/${SCRIPT_NAME}.py and algorithms/CF/config/${SCRIPT_NAME}.yaml.
- Keep code JAX-compatible (no dynamic shapes in jax.lax.scan).
- Check shared results.tsv to see what the OTHER GPU has already tried — don't repeat failed ideas.
- If the other GPU found an improvement on a DIFFERENT env, that's fine — focus on YOUR env.
- After logging results, you are DONE." 2>&1 | tee -a "$LOGFILE"

        echo "[GPU ${GPU_ID}] Experiment #${EXPERIMENT_NUM} | ${ENV_NAME} done at $(date)" | tee -a "$LOGFILE"
        sleep 5
    done

    echo "[GPU ${GPU_ID}] === Completed one full round. Starting next... ===" | tee -a "$LOGFILE"
done
