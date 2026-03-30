#!/bin/bash
# Sync best code between GPU workers
#
# PUBLISH: After a successful experiment, publish the improved files
#   bash sync_best.sh publish <env_name> <script_name> <worktree_path>
#
# PULL: Before an experiment, pull the latest best code for this env
#   bash sync_best.sh pull <env_name> <script_name> <worktree_path>

set -euo pipefail

ACTION="$1"
ENV_NAME="$2"
SCRIPT_NAME="$3"
WORKTREE="$4"

SHARED_DIR="$(cd "$(dirname "$0")" && pwd)"
BEST_DIR="${SHARED_DIR}/best_code/${ENV_NAME}"
LOCKFILE="${SHARED_DIR}/.sync_${ENV_NAME}.lock"

case "$ACTION" in
    publish)
        # Copy improved code to shared best_code directory
        (
            flock -w 10 200 || { echo "SYNC: Could not acquire lock"; exit 1; }
            mkdir -p "$BEST_DIR"
            cp "${WORKTREE}/algorithms/CF/${SCRIPT_NAME}.py" "${BEST_DIR}/${SCRIPT_NAME}.py"
            cp "${WORKTREE}/algorithms/CF/config/${SCRIPT_NAME}.yaml" "${BEST_DIR}/${SCRIPT_NAME}.yaml"
            echo "$(date +%s)" > "${BEST_DIR}/.timestamp"
            echo "SYNC: Published ${ENV_NAME} best code"
        ) 200>"$LOCKFILE"
        ;;

    pull)
        # Pull latest best code if newer than local
        (
            flock -w 10 200 || { echo "SYNC: Could not acquire lock"; exit 1; }

            if [ ! -f "${BEST_DIR}/${SCRIPT_NAME}.py" ]; then
                echo "SYNC: No shared best code for ${ENV_NAME} yet, using local"
                exit 0
            fi

            # Check if shared code is different from local
            if diff -q "${BEST_DIR}/${SCRIPT_NAME}.py" "${WORKTREE}/algorithms/CF/${SCRIPT_NAME}.py" > /dev/null 2>&1; then
                echo "SYNC: Local code already matches best for ${ENV_NAME}"
            else
                cp "${BEST_DIR}/${SCRIPT_NAME}.py" "${WORKTREE}/algorithms/CF/${SCRIPT_NAME}.py"
                cp "${BEST_DIR}/${SCRIPT_NAME}.yaml" "${WORKTREE}/algorithms/CF/config/${SCRIPT_NAME}.yaml"
                cd "$WORKTREE"
                git add "algorithms/CF/${SCRIPT_NAME}.py" "algorithms/CF/config/${SCRIPT_NAME}.yaml"
                git commit -m "autoresearch: sync best code for ${ENV_NAME} from other GPU" --allow-empty 2>/dev/null || true
                echo "SYNC: Pulled latest best code for ${ENV_NAME}"
            fi
        ) 200>"$LOCKFILE"
        ;;

    *)
        echo "Usage: sync_best.sh <publish|pull> <env_name> <script_name> <worktree_path>"
        exit 1
        ;;
esac
