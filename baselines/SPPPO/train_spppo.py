"""Training module for SPPPO on MPE environments.
Extracted from spppo_ff_mpe.py for modular reuse.

Functions
---------
train_spppo(config) -> (train_state, metrics)
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
    # When executed via `python -m baselines.spppo.train_spppo`
    from .spppo_ff_mpe import make_train  # type: ignore
except ImportError:
    # Fallback when run as a stand-alone script with `python baselines/spppo/train_spppo.py`
    import sys, pathlib
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1].parents[0]))
    from baselines.SPPPO.spppo_ff_mpe import make_train

# -----------------------------------------------------------------------------
# Core training logic
# -----------------------------------------------------------------------------

def train_spppo(config: Dict[str, Any]):
    """Run SPPPO training and return (all train_states, metrics)."""
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
        f"[train_spppo] Training for {config['NUM_SEEDS']} seeds x {config['NUM_UPDATES']} updates"
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
    """Train and dump player_0/player_1 parameter pickles to *save_dir*."""
    # Create a timestamp for the run
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Create a run-specific directory with timestamp
    run_dir = os.path.join(save_dir, f"spppo_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    train_states, metrics = train_spppo(config)

    # Save each seed's model
    for seed_idx in range(config["NUM_SEEDS"]):
        # Extract this seed's train state
        train_state = jax.tree_util.tree_map(lambda x: x[seed_idx], train_states)
        params = train_state.params
        
        # Create a seed-specific directory
        seed_dir = os.path.join(run_dir, f"seed{seed_idx}")
        os.makedirs(seed_dir, exist_ok=True)
        
        # Save parameters for both players
        for i in range(2):
            file_path = os.path.join(seed_dir, f"spppo_player_{i}.pkl")
            with open(file_path, "wb") as fh:
                pickle.dump(params, fh)
            print(f"[train_spppo] Saved parameters for seed{seed_idx} player_{i} -> {file_path}")

    # For backward compatibility, also save the first seed's model at the top level
    first_seed_params = jax.tree_util.tree_map(lambda x: x[0], train_states).params
    for i in range(2):
        file_path = os.path.join(save_dir, f"spppo_player_{i}.pkl")
        with open(file_path, "wb") as fh:
            pickle.dump(first_seed_params, fh)
        print(f"[train_spppo] Saved parameters for default player_{i} -> {file_path}")

    # Return the first seed's train state for backward compatibility
    first_seed_train_state = jax.tree_util.tree_map(lambda x: x[0], train_states)
    return first_seed_train_state, metrics


# -----------------------------------------------------------------------------
# If the module is run directly we treat it as a Hydra entry-point
# -----------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="config", config_name="spppo_ff_mpe")
def _main_hydra(cfg):  # pragma: no cover
    cfg = OmegaConf.to_container(cfg)
    wandb.init(
        entity=cfg["ENTITY"],
        project=cfg["PROJECT"],
        tags=["SPPPO", "FF", "TRAIN_ONLY"],
        config=cfg,
        mode=cfg["WANDB_MODE"],
    )
    train_and_save(cfg)

if __name__ == "__main__":
    _main_hydra()
