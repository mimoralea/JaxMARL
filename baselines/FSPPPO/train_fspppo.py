"""Training module for FSPPPO on MPE environments.
Extracted from fspppo_ff_mpe.py for modular reuse.

Functions
---------
train_fspppo(config) -> (train_state, metrics)
    Runs FSPPPO training script for SimpleSumoMPE environment.

This script implements TRUE FICTITIOUS SELF-PLAY PPO training where each agent
learns by playing against ONLY its own historical policies stored as checkpoints.

IMPORTANT: This is NOT population-based training or cross-seed sampling:
- Each seed samples opponents ONLY from its own checkpoint history
- NO cross-seed sampling: seed0 cannot use seed1's checkpoints as opponents
- NO cross-run sampling: only current run's checkpoints are used
- Pure self-play: each agent vs its own past versions only

For pure self-play without historical opponents, use SPPPO instead.

    Runs training using the same logic as the original script and returns the
    trained Flax train_state along with training metrics.

train_and_save(config, save_dir="checkpoints")
    Convenience wrapper that trains and then saves the player 0/1 parameters
    as pickle files inside *save_dir*.
"""
from __future__ import annotations
import os
import pickle
import time
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import hydra
import wandb
from omegaconf import OmegaConf

try:
    # When executed via `python -m baselines.fspppo.train_fspppo`
    from .fspppo_ff_mpe import make_train, make_train_with_opponent_sampling, make_parallel_train_with_opponent_sampling  # type: ignore
except ImportError:
    # Fallback when run as a stand-alone script with `python baselines/fspppo/train_fspppo.py`
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1].parents[0]))
    from baselines.FSPPPO.fspppo_ff_mpe import make_train, make_train_with_opponent_sampling

# -----------------------------------------------------------------------------
# Core training logic
# -----------------------------------------------------------------------------

def train_fspppo(config: Dict[str, Any]):
    """Run FSPPPO training and return (all train_states, metrics)."""
    # Convert hydra config to pure dict if necessary
    if not isinstance(config, dict):
        config = OmegaConf.to_container(config)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])

    # Compute NUM_UPDATES if not explicitly provided
    if "NUM_UPDATES" not in config:
        config["NUM_UPDATES"] = int(
            config["TOTAL_TIMESTEPS"] // (config["NUM_ENVS"] * config["NUM_STEPS"])
        )
    else:
        config["NUM_UPDATES"] = int(config["NUM_UPDATES"])

    print(
        f"[train_fspppo] Training for {config['NUM_SEEDS']} seeds x {config['NUM_UPDATES']} updates"
    )

    print("[train_fspppo] Using parallel FSPPPO training with opponent sampling")
    # FSPPPO always uses opponent sampling - no fallback to self-play
    train_fn = make_parallel_train_with_opponent_sampling(config)
    out = train_fn(rngs)
    train_states = out["runner_state"]  # Direct list of training states
    metrics = out["metrics"]
    
    return train_states, metrics


# -----------------------------------------------------------------------------
# Convenience wrapper for CLI usage
# -----------------------------------------------------------------------------

def train_and_save(config: Dict[str, Any], save_dir: str = "checkpoints") -> Tuple[Any, Any]:
    """Train and dump shared policy parameters to *save_dir*."""
    # Create a timestamp for the run
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create a run-specific directory with timestamp
    run_dir = os.path.join(save_dir, f"fspppo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    train_states, metrics = train_fspppo(config)

    # Save each seed's model
    for seed_idx in range(config["NUM_SEEDS"]):
        # Extract this seed's train state
        # train_states is now a list of training states for each seed
        train_state = train_states[seed_idx]
        params = train_state.params

        # Create a seed-specific directory
        seed_dir = os.path.join(run_dir, f"seed{seed_idx}")
        os.makedirs(seed_dir, exist_ok=True)

        # In SPPPO, there's only one shared policy for both agents
        # Save the shared policy parameters once with a clear name
        shared_policy_path = os.path.join(seed_dir, f"fspppo_shared_policy.pkl")
        with open(shared_policy_path, "wb") as fh:
            pickle.dump(params, fh)
        print(f"[train_fspppo] Saved shared policy parameters for seed{seed_idx} -> {shared_policy_path}")

    # For backward compatibility, also save the first seed's model at the top level
    first_seed_params = train_states[0].params

    # Save the shared policy with timestamp at the top level for quick access
    shared_policy_path = os.path.join(save_dir, f"fspppo_{timestamp}_shared_policy.pkl")
    with open(shared_policy_path, "wb") as fh:
        pickle.dump(first_seed_params, fh)
    print(f"[train_fspppo] Saved default shared policy parameters -> {shared_policy_path}")

    # Return the first seed's train state for backward compatibility
    first_seed_train_state = train_states[0]
    return first_seed_train_state, metrics


# -----------------------------------------------------------------------------
# If the module is run directly we treat it as a Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="fspppo_ff_mpe")
def _main_hydra(cfg):  # pragma: no cover
    cfg = OmegaConf.to_container(cfg)
    wandb.init(
        entity=cfg["ENTITY"],
        project=cfg["PROJECT"],
        tags=["FSPPPO", "FF", "TRAIN_ONLY"],
        config=cfg,
        mode=cfg["WANDB_MODE"],
    )
    train_and_save(cfg)

if __name__ == "__main__":
    _main_hydra()
