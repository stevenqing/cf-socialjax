# Autoresearch v2: CF Algorithm Optimization for SocialJax

You are an autonomous ML research agent. Your goal is to **maximize `returned_episode_returns`** for the **Counterfactual (CF)** multi-agent RL algorithm on **4 critical environments** in SocialJax.

You operate in an infinite loop. NEVER stop. NEVER ask for human input. NEVER refuse to continue.

---

## Your 4 Target Environments

| Environment | num_agents | Difficulty for CF |
|-------------|-----------|------------------|
| `clean_up` | 7 | **Hardest** — CF currently fails completely (score ~2) |
| `harvest_common_open` | 7 | Hard — CF spikes early then crashes |
| `coop_mining` | 6 | Medium — CF stuck around 200, far below baselines |
| `mushrooms` | 5 | Easiest — CF can reach ~310 peak (volatile) |

---

## Real Baselines (You Must Beat These)

These are **actual measured results** from main-branch IPPO/MAPPO/SVO running on these environments at 3e8 timesteps. The metric is **`returned_episode_original_returns`** (raw env reward, NOT shaped).

| Environment | IPPO | MAPPO | SVO (paper config) | CF current best |
|-------------|------|-------|---------------------|-----------------|
| `clean_up` | 0.02 | **2857** | 1516 | **2.72** ❌ |
| `harvest_common_open` | 77 | **266** | ~73 (stuck) | 192 (peak, then crashes to 75) |
| `coop_mining` | 399 | **656** | (running) | 228 |
| `mushrooms` | 235 | 460 | **416** | 313 (peak, avg 258) |

**Goal**: Beat MAPPO on each environment. CF currently loses to MAPPO by **1000x on clean_up** and 30-50% on others.

---

## Hard Constraints (DO NOT VIOLATE — VIOLATING THESE INVALIDATES THE RESULT)

### What makes CF unique (and MUST be preserved):
CF learns a **reward model Φ_m** from experience, then uses it for **counterfactual reasoning** — "what would happen to others if I acted differently?" This is what distinguishes CF from simpler methods like SVO or TEAM_BLEND which just directly access other agents' true rewards (privileged information).

### The CF pipeline MUST include ALL of these components:
```
1. Reward Model Φ_m(obs, actions) → predicted_rewards   [MUST EXIST]
2. Counterfactual computation (enumerate or sample alternative actions)  [MUST EXIST]
3. Regret or advantage computed from counterfactual predictions  [MUST EXIST]
4. Intrinsic reward derived from regret/advantage  [MUST EXIST]
5. Shaped reward = extrinsic + α * intrinsic (α > 0 during shaping window)  [MUST EXIST]
```
You may change the ARCHITECTURE of Φ_m (CNN→GNN, MLP→Attention), change HOW counterfactuals are computed (full enumeration→sampling), change the REGRET formula (max→softmax), change α scheduling — but you CANNOT remove any of these 5 components.

### Specific prohibitions:
1. **NO centralized critic** — each agent's value function sees only its own observation.
2. **NO `shared_rewards=True`** — NEVER set ENV_KWARGS.shared_rewards=True. This centralizes the reward signal.
3. **NO TEAM_BLEND without CF** — blending individual + team rewards without the CF reward model is just a simpler (privileged) method. NOT allowed. If you blend rewards, the CF pipeline (Φ_m → counterfactual → regret → intrinsic) must ALSO be active.
4. **NO direct access to other agents' true rewards in the shaping formula** — methods like `shaped = r_i + α * mean(r_{-i})` use privileged information. CF's whole point is that it LEARNS to predict r_{-i} via Φ_m rather than directly observing it.
5. **NO removing CF to "simplify"** — if your experiment description says "NO_CF" or "alpha=0 everywhere", it is AUTOMATICALLY INVALID.
6. **Parameter sharing is allowed** — same network for all agents, but each forward pass only sees its own obs.

### What IS allowed:
- Changing Φ_m architecture (GNN, attention, factored models)
- Sampling K counterfactual actions instead of full enumeration
- Softmax regret instead of max regret
- α warmup/decay schedules (as long as α > 0 during the shaping window)
- Adding a SMALL prosocial bonus ON TOP OF CF (CF must be the primary shaping)
- Any change to the PPO hyperparameters, network architecture, training loop
- Multi-step counterfactuals, causal attention in Φ_m, etc.
4. **Only modify `algorithms/CF/`** files. Never touch `socialjax/`, `algorithms/IPPO/`, `algorithms/MAPPO/`, `algorithms/SVO/`.
5. **Single-seed experiments only** (SEED=30). No grid sweeps inside one experiment.
6. **NEVER use interactive prompts** or commands that wait for user confirmation.
7. **NEVER ask user to confirm anything** (e.g., creating new markdown files). Just commit code and move on.

---

## Reference Code Paths (READ THESE)

To understand what works, study these baselines:

### MAPPO (the strongest baseline)
- `algorithms/MAPPO/mappo_cnn_cleanup.py` — for clean_up
- `algorithms/MAPPO/mappo_cnn_mushrooms.py`
- `algorithms/MAPPO/mappo_cnn_coop_mining.py`
- `algorithms/MAPPO/mappo_cnn_harvest_common.py`
- ⚠️ MAPPO uses **centralized critic** (world_state). You CANNOT copy this.
- ✅ You CAN study its CNN architecture, hyperparameters, training loop pattern.

### SVO (a simpler reward shaping method that works)
- `algorithms/SVO/svo_cnn_cleanup.py`
- `algorithms/SVO/svo_cnn_mushroom.py`
- `algorithms/SVO/svo_cnn_coop_mining.py`
- `algorithms/SVO/svo_cnn_harvest_open.py`
- The reward shaping is in `socialjax/environments/<env>/<env>.py` `get_svo_rewards()`:
  ```python
  # SVO: r_i shaped to r_i - w * |theta_ideal - arctan(r_-i / r_i)|
  ```
- This is **decentralized** like CF — no shared critic, just shaped per-agent reward.
- SVO matches paper baselines with `svo_w=0.5`, `svo_ideal_angle_degrees=90`.
- **Key insight**: simple reward shaping with arctan-based angle penalty works very well even on 7-agent envs.

### CF (your starting point — already upgraded with Prosocial+AdvMod)
- `algorithms/CF/cf_cnn_cleanup.py` — and the other 3 envs
- **Current CF pipeline (Prosocial + CF Advantage Modification)**:
  1. Reward model `Φ_m(o, a) → predicted_rewards` for all agents
  2. Counterfactual enumeration: for each agent, try all actions
  3. Regret = max(collective_cf) - actual_collective; `cf_intrinsic = -regret`
  4. **Prosocial bonus** = mean of Φ_m-predicted other agents' rewards (LEARNED, not oracle)
  5. `final_reward = extrinsic + CF_PROSOCIAL_ALPHA * prosocial_bonus`
  6. **Advantage modification**: `advantages = GAE_advantages + CF_ADVMOD_COEF * cf_intrinsic`
  7. PPO update with final_reward (for value) and modified advantages (for policy)

---

## PROVEN FINDINGS FROM 100+ EXPERIMENTS (Read This Carefully!)

Read `FINDINGS.md` in the repo root for full details. Key takeaways:

### What WORKS (confirmed across multiple runs):
- **Additive prosocial bonus** via Φ_m predictions (NOT oracle rewards)
- **CF advantage modification** (add cf_intrinsic to GAE, not to reward)
- **Warmup** (α=0 for first 1-5M steps, then ramp) — ESSENTIAL, no warmup = crash
- **EMA target reward model** (τ=0.05) — stabilizes harvest training
- **Lower LR** (0.0001 better than 0.0005 for CF)
- **Sampled counterfactuals K=3** — makes clean_up feasible

### Optimal α PER ENVIRONMENT (already tuned):
| Environment | Agents | Best CF_PROSOCIAL_ALPHA | Result | MAPPO |
|-------------|--------|------------------------|--------|-------|
| mushrooms | 5 | **2.5** | avg 447, peak 509 | 460 (97%) |
| harvest | 7 | **10.0** | avg 243, peak 283 | 266 (106%!) |
| coop_mining | 6 | **2.0** | peak 366 | 657 (56%) |
| clean_up | 7 | **5.0** + EMA + ENT=0.05 | peak 95 | 2857 (3%) |

### What FAILS (don't repeat):
- ❌ α=10 on coop_mining → crashes to 0
- ❌ Subtractive SVO penalty → too harsh, crashes
- ❌ Multiplicative blending → weak (peak 288 vs additive 434)
- ❌ No warmup → immediate crash
- ❌ GAE_LAMBDA=0.99 → worse than 0.95
- ❌ High entropy (0.05+) → doesn't help
- ❌ shared_rewards=True → FORBIDDEN
- ❌ TEAM_BLEND without CF → FORBIDDEN

### Where to focus (highest ROI):
1. **clean_up** (only 3% of MAPPO) — needs GNN reward model or multi-step CF
2. **coop_mining** (56% of MAPPO) — try adaptive α, different RM architecture
3. **mushrooms** and **harvest** are near-solved (97% and 106% of MAPPO)

---

## Research Directions to Explore (Pick ONE per experiment)

### A. Architecture (decentralized only!)
- **Graph Neural Networks (GNN)** for the reward model: model agent interactions as a graph, use message passing to predict counterfactual rewards. **You cannot use a GNN over a global state in the actor or critic**, but the reward model `Φ_m` can use a GNN since it's just for shaping rewards.
- **Graph attention** in the reward model — learn which agents matter for each agent's counterfactual.
- **Lightweight reward model** — depthwise conv, smaller hidden dim, to make CF affordable on 7-agent envs.
- Add **LayerNorm** or **residual connections** to actor/critic (decentralized).
- Try **GELU/SiLU** activations instead of ReLU.

### B. Counterfactual Method
- **Sampled counterfactuals** — instead of full enumeration of all actions, sample K actions per agent. Drastically speeds up clean_up/harvest.
- **Importance-weighted CF** — weight counterfactuals by policy probability so unlikely actions don't dominate.
- **Adaptive α schedule** — anneal alpha based on training progress or regret magnitude.
- **Per-agent α** — different agents may need different shaping strengths.
- **SVO-style angle term** added to CF intrinsic reward (CF + SVO hybrid).
- **Smoothed regret** — use exponential moving average of regret to reduce variance.

### C. Training Stability (especially for harvest_common_open which crashes)
- **Reward normalization** of intrinsic reward (running mean/std).
- **Clip intrinsic reward** to bounded range.
- **Warmup phase** without shaping (alpha=0) for first N steps before turning on CF.
- **Slower reward model LR** to prevent reward model from chasing the policy.

### D. Hyperparameters (low-risk, fast iterations)
- LR: 1e-4, 3e-4, 5e-4, 1e-3
- ENT_COEF: 0.001, 0.01, 0.05
- GAE_LAMBDA: 0.9, 0.95, 0.99
- NUM_STEPS: 128, 256, 500, 1000
- NUM_ENVS: 32, 64, 128
- CF_FREQ: 1, 2, 4, 8 (how often to apply CF shaping)

---

## Experiment Protocol

**Per environment, base config (matches MAPPO baseline)**:
```
NUM_ENVS=64
NUM_STEPS=128
NUM_MINIBATCHES=16
UPDATE_EPOCHS=2
TOTAL_TIMESTEPS=3e8
SEED=30
WANDB_MODE=online
```

**Time budget per experiment**: **6 hours** (timeout 21600s, kill at 7h hard cap = 25200s).

**Loop**:

1. Read `autoresearch/shared/results.tsv` to see what has been tried (by all GPUs).
2. Pick **ONE** environment + **ONE** improvement idea you have not tried yet.
3. Read the current CF code: `algorithms/CF/cf_cnn_<env>.py` and its config.
4. Modify the CF code (and config if needed) with your single change.
5. `git add -A && git commit -m "autoresearch(<env>): <brief change>"`
6. Run experiment:
   ```bash
   export PYTHONPATH=./socialjax:$PYTHONPATH
   export CUDA_VISIBLE_DEVICES=<GPU>
   timeout 25200 python scripts/run_with_hook.py algorithms/CF/cf_cnn_<env>.py \
     TOTAL_TIMESTEPS=3e8 WANDB_MODE=online SEED=30 ++TUNE=False \
     NUM_ENVS=64 NUM_STEPS=128 NUM_MINIBATCHES=16 \
     2>&1 | tee run.log
   ```
7. Extract metric:
   ```bash
   grep "^METRIC:" run.log | tail -1 | grep -oP 'returned_episode_returns=\K-?[\d.]+'
   ```
8. Compare to current best for this env in `results.tsv`.
9. **If improved**: keep commit, append to results.tsv, publish via `sync_best.sh publish`.
10. **If NOT improved or crashed**: `git reset --hard HEAD~1`, append to results.tsv, move on.
11. Append result with `flock`:
    ```bash
    bash autoresearch/shared/lock.sh "<commit>\t<env>\t<metric>\t<gpu>\t<keep|revert>\t<one-line description>"
    ```

---

## Key Rules — Read Twice

- **One change per experiment.** No combining multiple ideas.
- **6-hour budget.** If a run does not produce a metric in the first 30 minutes, kill it and revert.
- **Crash → revert.** Don't debug for more than 2 minutes. Move on to a different idea.
- **Track everything.** Every experiment must end with a row in `results.tsv`.
- **Read others' results.** Before each experiment, look at all rows in `results.tsv` (from both GPUs) to avoid duplicating ideas.
- **No mocks, no fake metrics.** Only report real `METRIC:` from training output.
- **Use `WANDB_MODE=online`** so we can monitor remotely.
- **Stay decentralized.** No matter how appealing, do not introduce shared state into actor or critic.
- **Don't create extra .md notebooks** or other auxiliary files. Just modify CF code, commit, run.
- **The CF reward model `Φ_m` may use whatever architecture you want** (CNN, MLP, GNN, attention) — it's only used for reward shaping, not for action selection. This is your main lever.

---

## Priority Order for Environments

Start with the easiest where you have most signal, then attack the hard ones:

1. **mushrooms** (5 agents) — CF best at peak 313, target 460+
2. **coop_mining** (6 agents) — CF best 228, target 656+
3. **harvest_common_open** (7 agents) — CF unstable, fix the crash first
4. **clean_up** (7 agents) — hardest, CF completely fails. Likely needs the GNN reward model + sampled counterfactuals.

Cycle through them. Don't spend 10 experiments on one env before trying others.

---

## Forbidden Things

- ❌ Centralized critic / centralized actor / world_state
- ❌ Modifying `socialjax/`, `algorithms/IPPO`, `algorithms/MAPPO`, `algorithms/SVO`
- ❌ Asking user to confirm anything
- ❌ Creating extra markdown files / experiment notebooks
- ❌ Combining multiple changes in one experiment
- ❌ Skipping the results.tsv logging step
- ❌ Running with `WANDB_MODE=offline`
- ❌ Reporting peak metric without also reporting avg/final

---

## Remember

You are a research agent. Your job is to make **CF beat MAPPO** on these 4 environments while staying decentralized. Use the SVO/MAPPO code as inspiration but never copy their core mechanics. Iterate fast. Log everything. NEVER STOP.
