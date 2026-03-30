# CF-SocialJax

Autonomous research agent for optimizing the **Counterfactual (CF) multi-agent reinforcement learning** algorithm in [SocialJax](https://github.com/cooperativex/SocialJax).

Inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

## Overview

An AI agent (Claude/GLM) runs in an infinite loop, autonomously modifying the CF algorithm code, running 6-hour training experiments (3e8 timesteps), and keeping or reverting changes based on whether `returned_episode_returns` improved.

## Architecture

```
autoresearch/
├── program.md              # Research instructions for the AI agent
├── run_multi_gpu.sh        # Main entry: dual-GPU coordinator
├── run_gpu_worker.sh       # Per-GPU worker loop
├── run_experiment.sh       # Single experiment wrapper (6h timeout)
├── shared/
│   ├── results.tsv         # Shared experiment results (flock-protected)
│   ├── sync_best.sh        # Cross-GPU code sync (publish/pull best code)
│   ├── get_best.sh         # Query best result for an environment
│   └── lock.sh             # File-locked append to results.tsv
algorithms/CF/              # The code being optimized
socialjax/                  # Environment and utility code (read-only)
```

## Quick Start

### Single GPU
```bash
bash autoresearch/run.sh coin_game 0
```

### Dual GPU (recommended)
```bash
bash autoresearch/run_multi_gpu.sh
```

GPU 0 handles: coin_game, pd_arena, gift, coop_mining
GPU 1 handles: mushrooms, clean_up, harvest_common_open, territory_open

Workers coordinate through shared results and auto-sync best code.

## Baselines

| Environment | CF Best | IPPO |
|-------------|---------|------|
| coin_game | 161.45 | 18.81 |
| harvest_common_open | 122.88 | 115.15 |
| clean_up | 974.57 | 0 |
| pd_arena | 56.06 | 15.55 |
| coop_mining | 311.76 | 185.81 |
| mushrooms | 294.74 | 224.08 |
| gift | 285.00 | 101.39 |
| territory_open | 176.21 | 195.24 |

## Research Directions

- Graph Neural Networks / Graph Causal Attention
- Improved reward model architectures
- Alpha scheduling and counterfactual computation
- Hyperparameter optimization
- Network architecture (LayerNorm, residual connections, attention)

## Setup

```bash
pip install -r requirements.txt
pip install jaxlib==0.4.23+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
export PYTHONPATH=./socialjax:$PYTHONPATH
```
