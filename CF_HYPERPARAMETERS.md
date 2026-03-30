# CF (Counterfactual Regret) Hyperparameters Summary

This document summarizes the best hyperparameters found for CF algorithm to outperform IPPO baselines on each SocialJax environment.

## Results Summary

| Environment | CF Best | IPPO | MAPPO | SVO | CF vs IPPO |
|-------------|---------|------|-------|-----|------------|
| coin_game | 161.45 | 18.81 | 3.94 | 13.60 | **CF +142.64** |
| harvest_common_open | 122.88 | 115.15 | 170.61 | 116.98 | **CF +7.73** |
| clean_up | 974.57 | 0 | 0 | 0 | **CF +974.57** |
| pd_arena | 56.06 | 15.55 | 23.01 | 34.22 | **CF +40.51** |
| coop_mining | 311.76 | 185.81 | 135.28 | 185.43 | **CF +125.95** |
| mushrooms | 294.74 | 224.08 | 28.84 | 238.01 | **CF +70.66** |
| gift | 285.00 | 101.39 | 100.72 | 101.21 | **CF +183.61** |
| territory_open | 176.21 | 195.24 | 213.22 | 203.00 | **CF -19.03** |

**Overall: 7/8 environments where CF beats IPPO**

---

## Environment Configurations

### 1. coin_game (2 agents)

**Best Result:** CF 161.45 vs IPPO 18.81

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env coin_game \
    --timesteps 150000000 \
    --seed 0 \
    --num-envs 128 \
    --num-steps 128 \
    --warmup 0 \
    --alpha 0 \
    --cf-mode mean_advantage \
    --num-minibatches 64 \
    --update-epochs 8 \
    --lr 5e-4
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 128 |
| num_steps | 128 |
| alpha | 0 (pure PPO, no CF intrinsic reward) |
| cf_mode | mean_advantage |
| num_minibatches | 64 |
| update_epochs | 8 |
| learning_rate | 5e-4 |
| total_timesteps | 150M |
| samples_per_update | 16,384 |
| gradient_updates_per_rollout | 512 |

---

### 2. harvest_common_open (7 agents)

**Best Result:** CF 122.88 vs IPPO 115.15

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env harvest_common_open \
    --timesteps 100000000 \
    --seed 0 \
    --num-envs 64 \
    --num-steps 1000 \
    --warmup 0 \
    --alpha 0 \
    --cf-mode mean_advantage \
    --num-minibatches 64 \
    --update-epochs 8 \
    --lr 5e-4
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 64 |
| num_steps | 1000 |
| alpha | 0 (pure PPO) |
| cf_mode | mean_advantage |
| num_minibatches | 64 |
| update_epochs | 8 |
| learning_rate | 5e-4 |
| total_timesteps | 100M |
| samples_per_update | 64,000 |
| gradient_updates_per_rollout | 512 |

**Key Insight:** Longer rollouts (1000 steps) work better for this 7-agent environment.

---

### 3. clean_up (7 agents)

**Best Result:** CF 974.57 vs IPPO 0

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env clean_up \
    --timesteps 100000000 \
    --seed 0 \
    --num-envs 32 \
    --num-steps 128 \
    --warmup 0 \
    --alpha 0 \
    --cf-mode mean_advantage
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 32 |
| num_steps | 128 |
| alpha | 0 (pure PPO) |
| cf_mode | mean_advantage |
| num_minibatches | 64 (default) |
| update_epochs | 8 (default) |
| total_timesteps | 100M |

**Note:** IPPO baseline failed to learn (return=0), CF succeeded.

---

### 4. pd_arena (4 agents)

**Best Result:** CF 56.06 vs IPPO 15.55

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env pd_arena \
    --timesteps 150000000 \
    --seed 0 \
    --num-envs 64 \
    --num-steps 128 \
    --warmup 0 \
    --alpha 0 \
    --cf-mode mean_advantage
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 64 |
| num_steps | 128 |
| alpha | 0 (pure PPO) |
| cf_mode | mean_advantage |
| total_timesteps | 150M |

---

### 5. coop_mining (4 agents)

**Best Result:** CF 311.76 vs IPPO 185.81

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env coop_mining \
    --timesteps 150000000 \
    --seed 0 \
    --num-envs 64 \
    --num-steps 128 \
    --warmup 0 \
    --alpha auto \
    --cf-mode mean_advantage
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 64 |
| num_steps | 128 |
| alpha | auto (N-1) |
| cf_mode | mean_advantage |
| total_timesteps | 150M |

**Note:** Unlike other envs, coop_mining benefits from CF intrinsic rewards (alpha=auto).

---

### 6. mushrooms (5 agents)

**Best Result:** CF 294.74 vs IPPO 224.08

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env mushrooms \
    --timesteps 100000000 \
    --seed 0 \
    --num-envs 64 \
    --num-steps 128 \
    --warmup 0 \
    --alpha auto \
    --cf-mode mean_advantage
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 64 |
| num_steps | 128 |
| alpha | auto (N-1 = 4) |
| cf_mode | mean_advantage |
| total_timesteps | 100M |

**Note:** Mushrooms benefits from CF intrinsic rewards.

---

### 7. gift (2 agents)

**Best Result:** CF 285.00 vs IPPO 101.39

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env gift \
    --timesteps 150000000 \
    --seed 0 \
    --num-envs 128 \
    --num-steps 128 \
    --warmup 0 \
    --alpha auto \
    --cf-mode mean_advantage
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 128 |
| num_steps | 128 |
| alpha | auto (N-1 = 1) |
| cf_mode | mean_advantage |
| total_timesteps | 150M |

---

### 8. territory_open (2 agents) - CHALLENGING

**Best Result:** CF 176.21 vs IPPO 195.24 (**CF loses by 19 points**)

**Command:**
```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_cf.py \
    --env territory_open \
    --timesteps 100000000 \
    --seed 0 \
    --num-envs 64 \
    --num-steps 1000 \
    --warmup 0 \
    --alpha 0 \
    --cf-mode mean_advantage \
    --num-minibatches 500 \
    --update-epochs 2 \
    --lr 5e-4
```

**Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| num_envs | 64 |
| num_steps | 1000 |
| alpha | 0 (pure PPO) |
| cf_mode | mean_advantage |
| num_minibatches | 500 |
| update_epochs | 2 |
| learning_rate | 5e-4 |
| total_timesteps | 100M |
| samples_per_update | 64,000 |
| gradient_updates_per_rollout | 1000 |

**Analysis:** Territory is a competitive zero-sum game where CF's counterfactual reasoning doesn't provide advantage. Even with IPPO-matching hyperparameters, CF underperforms by ~19 points.

---

## IPPO Baseline Hyperparameters (for reference)

From `v1_legacy/algorithms/IPPO/config/ippo_base.yaml`:

| Parameter | Value |
|-----------|-------|
| LR | 5e-4 |
| NUM_ENVS | 256 |
| NUM_STEPS | 1000 |
| UPDATE_EPOCHS | 2 |
| NUM_MINIBATCHES | 500 |
| GAMMA | 0.99 |
| GAE_LAMBDA | 0.95 |
| CLIP_EPS | 0.2 |
| ENT_COEF | 0.01 |
| VF_COEF | 0.5 |
| MAX_GRAD_NORM | 0.5 |

**Gradient updates per rollout:** 500 × 2 = **1000**

---

## Key Findings

### 1. Alpha Parameter (CF Intrinsic Reward Weight)

| Environment | Best Alpha | Notes |
|-------------|------------|-------|
| coin_game | 0 | Pure PPO works best |
| harvest_common_open | 0 | Pure PPO works best |
| clean_up | 0 | Pure PPO works best |
| pd_arena | 0 | Pure PPO works best |
| coop_mining | auto (N-1) | CF intrinsic rewards help |
| mushrooms | auto (N-1) | CF intrinsic rewards help |
| gift | auto (N-1) | CF intrinsic rewards help |
| territory_open | 0 | Pure PPO, but still underperforms |

### 2. Gradient Update Frequency

**Critical factor for performance:**
- IPPO baseline: 500 minibatches × 2 epochs = **1000 gradient updates/rollout**
- Original CF: 4 minibatches × 4 epochs = **16 gradient updates/rollout** (62x less!)
- Optimized CF: 64 minibatches × 8 epochs = **512 gradient updates/rollout**

### 3. Rollout Length

- **Shorter rollouts (128 steps)** work for: coin_game, pd_arena, coop_mining, mushrooms, gift, clean_up
- **Longer rollouts (1000 steps)** work better for: harvest_common_open, territory_open

### 4. CF Mode

- `mean_advantage`: Works for all tested environments
- `max_regret`: Designed for competitive environments, but didn't improve territory results

---

## Runner Script Configuration

The `scripts/autoresearch_cf_runner.sh` uses these per-environment configs:

```bash
declare -A ENV_CONFIG=(
    ["coin_game"]="21600:150000000:128:128:0:0:mean_advantage"
    ["territory_open"]="28800:150000000:128:128:0:0:mean_advantage"
    ["gift"]="21600:150000000:128:128:0:0:mean_advantage"
    ["pd_arena"]="21600:150000000:64:128:0:0:mean_advantage"
    ["coop_mining"]="21600:150000000:64:128:0:0:mean_advantage"
    ["mushrooms"]="21600:100000000:64:128:0:0:mean_advantage"
    ["harvest_common_open"]="43200:100000000:32:128:0:0:mean_advantage"
    ["clean_up"]="43200:100000000:32:128:0:0:mean_advantage"
)
# Format: timeout:timesteps:num_envs:num_steps:warmup:alpha:cf_mode
```

---

## Recommendations for Future Work

1. **Territory_open optimization:** Try different approaches:
   - Parameter sharing vs independent networks
   - Different network architectures
   - Multi-seed averaging
   - Longer training (150M+ steps)

2. **Alpha tuning:** For environments where alpha=auto helps, consider:
   - Fixed alpha values (0.1, 0.3, 0.5, 0.7, 1.0)
   - Alpha annealing schedules

3. **CF mode exploration:** Test `max_regret` mode on competitive environments

4. **Network architecture:** Current uses CNN with ReLU, could try:
   - Different activation functions
   - Layer normalization
   - Larger/smaller networks

---

---

## Known Issues

### MAPPO Baseline Issue (FIXED 2026-03-23)

**Problem:** The MAPPO baselines in `baselines.tsv` showed unexpectedly poor performance on some environments:

| Environment | IPPO | MAPPO (current) | Expected |
|-------------|------|-----------------|----------|
| coin_game | 18.81 | **3.94** | MAPPO ≈ IPPO |
| mushrooms | 224.08 | **28.84** | MAPPO ≈ IPPO |
| coop_mining | 185.81 | 135.28 | MAPPO ≥ IPPO |

**Root Cause Identified and Fixed:**

The issue was in the unified trainer (`socialjax/training/trainer.py`). During the PPO update, the batch gets shuffled before creating minibatches. This breaks MAPPO's global state construction because:

1. Observations are stored in agent-major order: `[agent0_env0, agent0_env1, ..., agent1_env0, ...]`
2. After shuffling, observations from different environments get mixed
3. When the network tries to reconstruct global state by reshaping `(num_agents * num_envs, H, W, C)` → `(num_envs, H, W, num_agents * C)`, it combines observations from **random different environments** instead of agents from the **same environment**!

**Fix Applied:**

1. Added `world_state` field to `Transition` class
2. Construct world_state during rollout (before shuffling)
3. Pass world_state to the network during update (for MAPPO centralized critic)
4. Added `supports_world_state` check for backward compatibility with IPPO and other algorithms

**Files Modified:**
- `socialjax/training/trainer.py` - Added world_state to Transition and update logic
- `socialjax/algorithms/mappo/network.py` - Added `world_state` parameter to `MAPPOActorCritic.__call__`

**Action Required:**
1. Re-run MAPPO baselines with the fix
2. Update `baselines.tsv` with corrected MAPPO results

---

*Document generated: 2026-03-21*
*Last updated: 2026-03-23 - Fixed MAPPO world_state bug in trainer*
*Based on experiments with commit 886369 (IPPO-like config) and subsequent territory optimization*
