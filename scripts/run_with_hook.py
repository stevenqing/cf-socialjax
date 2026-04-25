"""Wrapper that patches wandb.log to print metrics, then runs the target script.
Usage: python scripts/run_with_hook.py <target_script.py> [hydra args...]

Prints both:
  METRIC: shaped_returns  (includes SVO/CF reward shaping)
  METRIC_RAW: original_returns  (raw environment reward, comparable to paper)
"""
import sys
import wandb
import wandb.sdk.wandb_run

def _emit(data):
    if not isinstance(data, dict):
        return
    s = data.get("env_step", "?")
    if "returned_episode_returns" in data:
        r = data["returned_episode_returns"]
        print(f"METRIC: env_step={s} returned_episode_returns={float(r):.4f}", flush=True)
    if "returned_episode_original_returns" in data:
        r = data["returned_episode_original_returns"]
        print(f"METRIC_RAW: env_step={s} returned_episode_original_returns={float(r):.4f}", flush=True)

_orig_log = wandb.log
def _patched_log(data, *a, **kw):
    _orig_log(data, *a, **kw)
    _emit(data)
wandb.log = _patched_log

_orig_run_log = wandb.sdk.wandb_run.Run.log
def _patched_run_log(self, data, *a, **kw):
    _orig_run_log(self, data, *a, **kw)
    _emit(data)
wandb.sdk.wandb_run.Run.log = _patched_run_log

import runpy
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name="__main__")
