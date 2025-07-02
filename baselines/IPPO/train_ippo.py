"""Training module for IPPO on MPE environments.
Extracted from ippo_ff_mpe.py for modular reuse.

Functions
---------
train_ippo(config) -> (train_state, metrics)
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
    # When executed via `python -m baselines.IPPO.train_ippo`
    from .ippo_ff_mpe import make_train  # type: ignore
except ImportError:
    # Fallback when run as a stand-alone script with `python baselines/IPPO/train_ippo.py`
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1].parents[0]))
    from baselines.IPPO.ippo_ff_mpe import make_train

# -----------------------------------------------------------------------------
# Core training logic
# -----------------------------------------------------------------------------

def train_ippo(config: Dict[str, Any]):
    """Run IPPO training and return (all train_states, metrics)."""
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
        f"[train_ippo] Training for {config['NUM_SEEDS']} seeds x {config['NUM_UPDATES']} updates"
    )

    # JIT-compile the training function
    train_jit = jax.jit(make_train(config))

    out = jax.vmap(train_jit)(rngs)
    # Return all train states instead of just the first one
    train_states = out["runner_state"][0]
    metrics = out["metrics"]
    return train_states, metrics


# -----------------------------------------------------------------------------
# Convenience wrapper for CLI usage
# -----------------------------------------------------------------------------

def train_and_save(config: Dict[str, Any], save_dir: str = "checkpoints") -> Tuple[Any, Any]:
    """Train and save a single shared policy pickle per seed (IPPO currently uses one set of params)."""
    # Create a timestamp for the run
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Create a run-specific directory with timestamp
    run_dir = os.path.join(save_dir, f"ippo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    train_states, metrics = train_ippo(config)

    # Save each seed's model (single shared policy)
    for seed_idx in range(config["NUM_SEEDS"]):
        # Extract this seed's train state
        train_state = jax.tree_util.tree_map(lambda x: x[seed_idx], train_states)
        params0, params1 = train_state.params

        # Create a seed-specific directory
        seed_dir = os.path.join(run_dir, f"seed{seed_idx}")
        os.makedirs(seed_dir, exist_ok=True)

        for i, p in enumerate((params0, params1)):
            p_path = os.path.join(seed_dir, f"ippo_player_{i}.pkl")
            with open(p_path, "wb") as fh:
                pickle.dump(p, fh)
            print(f"[train_ippo] Saved player_{i} params for seed{seed_idx} -> {p_path}")

    # Also save the first seed's model at the top level for quick access
    first_seed_params0, first_seed_params1 = jax.tree_util.tree_map(lambda x: x[0], train_states).params
    for i, p in enumerate((first_seed_params0, first_seed_params1)):
        p_path = os.path.join(save_dir, f"ippo_{timestamp}_player_{i}.pkl")
        with open(p_path, "wb") as fh:
            pickle.dump(p, fh)
        print(f"[train_ippo] Saved default player_{i} params -> {p_path}")

    # Return the first seed's train state for backward compatibility
    first_seed_train_state = jax.tree_util.tree_map(lambda x: x[0], train_states)
    return first_seed_train_state, metrics


# -----------------------------------------------------------------------------
# If the module is run directly we treat it as a Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="ippo_ff_mpe")
def _main_hydra(cfg):  # pragma: no cover
    cfg = OmegaConf.to_container(cfg)
    wandb.init(
        entity=cfg["ENTITY"],
        project=cfg["PROJECT"],
        tags=["IPPO", "FF", "TRAIN_ONLY"],
        config=cfg,
        mode=cfg["WANDB_MODE"],
    )
    train_and_save(cfg)

if __name__ == "__main__":
    _main_hydra()
