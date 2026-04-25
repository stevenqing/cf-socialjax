#!/bin/bash
# Sweep CF_PROSOCIAL_ALPHA on coop_mining (GPU 1)
set -uo pipefail
cd /home/shuqing/cf-socialjax
export PYTHONPATH=./socialjax:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=1
export XLA_FLAGS="--xla_gpu_strict_conv_algorithm_picker=false"
PY=/home/shuqing/.conda/envs/melting-jax/bin/python
RESULTS="eval_coop_mining_sweep.tsv"
echo -e "alpha\treturned_episode_returns\tpeak\tstatus" > "$RESULTS"

for ALPHA in 2.0 5.0 1.0 3.0; do
    echo ""; echo "$(date) | coop_mining | alpha=${ALPHA}"
    LOG="eval_logs/coop_mining_alpha${ALPHA}_seed30.log"
    mkdir -p eval_logs

    timeout 25200 $PY scripts/run_with_hook.py algorithms/CF/cf_cnn_coop_mining.py \
        TOTAL_TIMESTEPS=3e8 \
        WANDB_MODE=online \
        SEED=30 \
        NUM_ENVS=64 \
        NUM_STEPS=128 \
        NUM_MINIBATCHES=16 \
        UPDATE_EPOCHS=2 \
        CF_PROSOCIAL_ALPHA=${ALPHA} \
        CF_ADVMOD_COEF=1.0 \
        ++ENV_KWARGS.shared_rewards=False \
        ++TUNE=False \
        2>&1 | tee "$LOG" || true

    EC=${PIPESTATUS[0]}
    FINAL=$(grep "^METRIC:" "$LOG" 2>/dev/null | tail -1 | grep -oP 'returned_episode_returns=\K-?[\d.]+' || echo "CRASHED")
    PEAK=$(grep "^METRIC:" "$LOG" 2>/dev/null | grep -oP 'returned_episode_returns=\K-?[\d.]+' | sort -rn | head -1 || echo "0")
    echo -e "${ALPHA}\t${FINAL}\t${PEAK}\t${EC}" >> "$RESULTS"
    echo ">>> alpha=${ALPHA}: final=${FINAL} peak=${PEAK}"
done

echo ""; echo "=== COOP_MINING SWEEP DONE ==="; cat "$RESULTS"
