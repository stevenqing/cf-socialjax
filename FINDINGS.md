# CF Autoresearch: Comprehensive Findings

## 1. Baseline Comparison (all at 3e8 timesteps, seed=30)

| Environment | Agents | IPPO | MAPPO | SVO (θ=90) | CF Best (valid) | CF Method | vs MAPPO |
|-------------|--------|------|-------|------------|-----------------|-----------|----------|
| mushrooms | 5 | 235 | **460** | 416 | **447** (peak 509) | Additive Prosocial α=2.5 | **97%** |
| harvest_common_open | 7 | 77 | **266** | ~73 | **283** (peak 283) | Prosocial10 + CF_AdvMod | **106%** ✅ |
| coop_mining | 6 | 399 | **657** | - | **366** (peak 366) | Prosocial α=2.0 + CF_AdvMod | 56% |
| clean_up | 7 | 0.02 | **2857** | 1516 | **95** (peak 311) | CF_EMA + Prosocial α=5 | 3.3% |

## 2. The Winning Method: Prosocial Bonus + CF Advantage Modification

### Core Idea
Instead of using CF-shaped reward directly for PPO, split it into two channels:

```
Channel 1 (Reward): final_reward = r_i + α * prosocial_bonus
  where prosocial_bonus = mean(Φ_m predicted r_{-i})  [via LEARNED reward model, not true rewards]

Channel 2 (Advantage): advantages = GAE_advantages + β * cf_intrinsic
  where cf_intrinsic = -regret (from counterfactual enumeration)
```

### Why This Works
- **Prosocial bonus** gives agents a cooperative reward signal from Φ_m predictions (learned, not oracle)
- **CF advantage modification** preserves the counterfactual reasoning signal in the policy gradient
- Separating reward vs advantage prevents CF noise from destabilizing value function training

### Key Hyperparameters Per Environment

| Env | CF_PROSOCIAL_ALPHA | CF_ADVMOD_COEF | Notes |
|-----|-------------------|----------------|-------|
| mushrooms (5a) | **2.5** | 1.0 | Higher α works; α=10 still OK (403 peak) |
| harvest (7a) | **10.0** | 1.0 | Needs high α for 7 agents; α=2 too weak |
| coop_mining (6a) | **2.0** | 1.0 | α=10 crashes to 0; α=2 sweet spot |
| clean_up (7a) | **5.0** | 1.0 | Best valid CF result was 95 with EMA target RM |

### Critical: α Sensitivity
- **2-agent envs**: α=0.5-2.5 works
- **5-agent envs**: α=2.0-2.5 optimal, α=10 still works but suboptimal  
- **6-agent envs**: α=2.0 works, **α=10 crashes to 0** (too aggressive)
- **7-agent envs (harvest)**: α=10 works (needs strong signal for cooperation)
- **7-agent envs (clean_up)**: α=5 with EMA target, still very hard

## 3. Failed Approaches (Don't Repeat These)

### INVALID (violate decentralized constraint)
- ❌ `shared_rewards=True` — centralizes reward signal
- ❌ `TEAM_BLEND` without CF — removes reward model + counterfactual pipeline
- ❌ Direct `mean(r_{-i})` in shaping — uses oracle access to other agents' rewards

### Failed but Valid CF Approaches
- ❌ **SVO-style subtractive penalty**: `r_i - w*|θ - arctan(r_{-i}/r_i)|` — too harsh, crashes training
- ❌ **Multiplicative blending**: weaker than additive (peak 288 vs 434 on mushrooms)
- ❌ **High entropy (0.05+)**: doesn't help CF convergence
- ❌ **GAE_LAMBDA=0.99**: worse than 0.95 for CF
- ❌ **LR=0.0005**: CF prefers lower LR (0.0001)
- ❌ **No warmup**: crashes immediately — warmup is ESSENTIAL for CF
- ❌ **Large α without warmup**: reward model is garbage early, must warm up first

## 4. Techniques That Help

### Confirmed Helpful
- ✅ **Warmup**: α=0 for first 1-5M steps, then ramp to target α (5M-15M window)
- ✅ **Additive prosocial bonus** (not subtractive penalty)
- ✅ **CF advantage modification** (add cf_intrinsic to GAE advantages)
- ✅ **EMA target reward model** (τ=0.05): stabilizes harvest training, prevents crash
- ✅ **Sampled counterfactuals K=3**: makes clean_up feasible (7 agents × 9 actions → 7×3)
- ✅ **Lower LR (0.0001)**: more stable for CF than default 0.0005

### Partially Helpful
- ⚠️ **CF_FREQ=4**: update CF every 4 steps instead of every step — reduces overhead but mixed results
- ⚠️ **Reward model freezing**: helps in some cases, hurts in others

## 5. Unsolved Problems

### harvest_common_open: The Crash Pattern
- **All non-prosocial CF variants** spike to ~190-215 at 3-5M steps then crash to ~75
- Even **pure IPPO (α=0)** crashes the same way → crash is structural, not CF-caused
- Only **Prosocial10 + AdvMod** survives past the crash zone
- **Best valid result**: 283 (c86429e, α=10) and 255 (current eval, α=10)

### clean_up: The 7-Agent Challenge  
- CF counterfactual computation is O(N × action_dim) = 63 inferences per step
- Reward model Φ_m can't learn accurate 7-agent predictions
- Best valid CF result: 95.27 (CF_EMA + Prosocial α=5 + ENT=0.05)
- MAPPO gets 2857 with centralized critic — CF at 95 is only 3.3%
- **Needs architectural innovation**: GNN reward model, factored predictions, or multi-step CF

### coop_mining: α Sensitivity
- α=10 crashes, α=2 works (366 peak), but still only 56% of MAPPO
- May need env-specific α tuning or adaptive α

## 6. Architecture of Best CF Code (c86429e)

```python
# In make_train():
cf_prosocial_alpha = config.get("CF_PROSOCIAL_ALPHA", 10.0)
cf_advmod_coef = config.get("CF_ADVMOD_COEF", 1.0)

# In _env_step():
# 1. Compute CF regret (standard pipeline)
_, regret = compute_cf_shaped_reward(reward_model.apply, rm_params, ...)
cf_intrinsic = -regret

# 2. Prosocial bonus via LEARNED reward model (not oracle!)
predicted_rewards = reward_model.apply(rm_params, obs_joint, actions_joint)
prosocial_mask = 1.0 - jnp.eye(num_agents)
prosocial_bonus = jnp.sum(
    predicted_rewards[:, jnp.newaxis, :] * prosocial_mask[jnp.newaxis, :, :],
    axis=-1
) / (num_agents - 1)

# 3. Final reward = extrinsic + prosocial (CF intrinsic saved separately)
final_reward = rewards_joint + cf_prosocial_alpha * prosocial_bonus

# In _update_step() after GAE:
# 4. Advantage modification
advantages = advantages + cf_advmod_coef * traj_batch.cf_intrinsic
```

## 7. Next Steps for Autoresearch

### High Priority
1. **Per-environment α tuning**: mushrooms α=2.5, harvest α=10, coop α=2.0, clean_up α=5
2. **GNN reward model for clean_up**: the flat MLP Φ_m can't scale to 7 agents
3. **EMA target RM everywhere**: confirmed helpful for harvest, test on others

### Medium Priority
4. **Sampled CF (K=3-5)** for clean_up and harvest to reduce computation
5. **Multi-step CF** for clean_up (cleaning is delayed cooperation, 1-step CF can't capture it)
6. **Adaptive α**: start high, decay based on reward model confidence

### Low Priority
7. **Multiple seeds** for robustness validation
8. **Longer training** (5e8+) for harvest and clean_up
9. **Network architecture search**: LayerNorm, residual connections, larger hidden dims
