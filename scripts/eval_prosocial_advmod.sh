#!/bin/bash
# Evaluate PROSOCIAL10 + CF_ADVMOD on all 4 target environments
# Uses the winning harvest config: CF_PROSOCIAL_ALPHA=10, CF_ADVMOD_COEF=1.0
# One env per GPU (GPU 0, 1, 2 for first 3, then GPU 0 for 4th)
set -uo pipefail

cd /home/shuqing/cf-socialjax
export PYTHONPATH=./socialjax:${PYTHONPATH:-}
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
PY=/home/shuqing/.conda/envs/melting-jax/bin/python
RESULTS="eval_prosocial_advmod_results.tsv"
echo -e "env_name\treturned_episode_returns\tpeak\tseed\tgpu\tstatus" > "$RESULTS"

run_eval() {
    local SCRIPT=$1 ENV=$2 GPU=$3 SEED=${4:-30}
    echo "$(date) | ${ENV} | GPU ${GPU} | seed ${SEED}"
    local LOG="eval_logs/prosocial_advmod_${ENV}_seed${SEED}.log"
    mkdir -p eval_logs

    CUDA_VISIBLE_DEVICES=$GPU timeout 25200 $PY scripts/run_with_hook.py "$SCRIPT" \
        TOTAL_TIMESTEPS=3e8 \
        WANDB_MODE=online \
        SEED=$SEED \
        NUM_ENVS=64 \
        NUM_STEPS=128 \
        NUM_MINIBATCHES=16 \
        UPDATE_EPOCHS=2 \
        CF_PROSOCIAL_ALPHA=10.0 \
        CF_ADVMOD_COEF=1.0 \
        ++ENV_KWARGS.shared_rewards=False \
        ++TUNE=False \
        2>&1 | tee "$LOG" || true

    local EC=${PIPESTATUS[0]}
    local FINAL=$(grep "^METRIC:" "$LOG" 2>/dev/null | tail -1 | grep -oP 'returned_episode_returns=\K-?[\d.]+' || echo "CRASHED")
    local PEAK=$(grep "^METRIC:" "$LOG" 2>/dev/null | grep -oP 'returned_episode_returns=\K-?[\d.]+' | sort -rn | head -1 || echo "0")
    echo -e "${ENV}\t${FINAL}\t${PEAK}\t${SEED}\t${GPU}\t${EC}" >> "$RESULTS"
    echo ">>> ${ENV}: final=${FINAL} peak=${PEAK}"
}

# Run 3 envs in parallel on GPU 0, 1, 2
run_eval algorithms/CF/cf_cnn_mushrooms.py mushrooms 0 &
PID0=$!
run_eval algorithms/CF/cf_cnn_coop_mining.py coop_mining 1 &
PID1=$!
run_eval algorithms/CF/cf_cnn_harvest_common.py harvest_common_open 2 &
PID2=$!

echo "Running mushrooms (GPU0), coop_mining (GPU1), harvest (GPU2) in parallel..."
wait $PID0 $PID1 $PID2

# Then clean_up on GPU 0
run_eval algorithms/CF/cf_cnn_cleanup.py clean_up 0

echo ""
echo "========== ALL EVALUATIONS DONE =========="
cat "$RESULTS"
