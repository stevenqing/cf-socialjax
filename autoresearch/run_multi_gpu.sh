#!/bin/bash
# Autoresearch multi-GPU coordinator
# Usage: bash autoresearch/run_multi_gpu.sh
#
# Launches 2 GPU workers in parallel, each with its own git worktree.
# Workers coordinate through shared results.tsv with file locking.
#
# GPU 0: coin_game, gift, mushrooms, harvest_common_open
# GPU 1: pd_arena, coop_mining, clean_up, territory_open

set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$(pwd)"
SHARED_DIR="${REPO_ROOT}/autoresearch/shared"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "AUTORESEARCH MULTI-GPU COORDINATOR"
echo "Repo: $REPO_ROOT"
echo "Started: $(date)"
echo "=========================================="

# Create worktrees for each GPU
for GPU_ID in 0 1; do
    WORKTREE="${REPO_ROOT}/autoresearch/worktree-gpu${GPU_ID}"
    BRANCH="autoresearch/gpu${GPU_ID}_${TIMESTAMP}"

    # Clean up old worktree if exists
    if [ -d "$WORKTREE" ]; then
        echo "Cleaning up old worktree for GPU ${GPU_ID}..."
        git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
    fi

    echo "Creating worktree for GPU ${GPU_ID} at ${WORKTREE}..."
    git worktree add -b "$BRANCH" "$WORKTREE" HEAD

    # Symlink shared directory into worktree so both workers see it
    ln -sf "$SHARED_DIR" "${WORKTREE}/autoresearch/shared"
done

echo ""
echo "Launching GPU workers..."
echo ""

# Launch both GPU workers in parallel
bash autoresearch/run_gpu_worker.sh 0 "coin_game pd_arena gift coop_mining" "$TIMESTAMP" &
PID_GPU0=$!

bash autoresearch/run_gpu_worker.sh 1 "mushrooms clean_up harvest_common_open territory_open" "$TIMESTAMP" &
PID_GPU1=$!

echo "GPU 0 worker PID: $PID_GPU0"
echo "GPU 1 worker PID: $PID_GPU1"
echo ""
echo "To stop: kill $PID_GPU0 $PID_GPU1"
echo "Logs: autoresearch/gpu0.log, autoresearch/gpu1.log"

# Wait for both (they run forever until killed)
wait $PID_GPU0 $PID_GPU1
