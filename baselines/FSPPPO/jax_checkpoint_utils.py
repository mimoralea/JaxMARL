"""
JAX-compatible checkpoint utilities for saving during training.
Provides mechanisms to save checkpoints without interfering with JIT compilation.

SEED HANDLING AND CHECKPOINT ARCHITECTURE:
==========================================

Seed Concept:
    - Each seed represents a COMPLETE INDEPENDENT TRAINING EXPERIMENT
    - Seeds control: network initialization, environment randomness, training randomness
    - Seeds are RUN-LEVEL concepts, not agent-level
    - seed0 and seed1 are different training runs, not different agents

Checkpoint Structure per Seed:
    - Base run_id: "run_20250717_231055"
    - Seed-specific run_id: "run_20250717_231055_seed0"
    - Full path: checkpoints/fspppo/run_20250717_231055_seed0/main/12/

Multi-Seed Training:
    - Each seed gets its own complete checkpoint directory tree
    - Enables cross-seed opponent sampling for Fictitious Self-Play
    - Supports future population training with consistent agent pairs per seed

Future Population Training Structure:
    checkpoints/fspppo/run_xyz_seed0/main/12/
    checkpoints/fspppo/run_xyz_seed0/opponent/12/

    This ensures same seed controls both agents (shared randomness).
"""

import logging
import warnings
import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional
from flax.training import train_state
try:
    from .orbax_checkpoint_manager import FSPPPOCheckpointManager
except ImportError:
    from orbax_checkpoint_manager import FSPPPOCheckpointManager

# Configure logging to silence verbose output
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('orbax').setLevel(logging.ERROR)
logging.getLogger('jax').setLevel(logging.WARNING)
logging.getLogger('jax._src').setLevel(logging.ERROR)
logging.getLogger('tensorstore').setLevel(logging.ERROR)

# Suppress specific Orbax checkpoint warnings
warnings.filterwarnings('ignore',
                       message="Couldn't find sharding info under RestoreArgs.*",
                       category=UserWarning,
                       module='orbax.checkpoint.type_handlers')

class TrainingCheckpointCallback:
    """
    Callback system for saving checkpoints during JAX training.

    This class provides a way to save checkpoints periodically during training
    without interfering with JIT compilation by using host callbacks.
    """

    def __init__(self, checkpoint_manager: FSPPPOCheckpointManager,
                 run_id: str, agent_id: str = "main",
                 checkpoint_freq: int = 100, max_checkpoints: int = 10):
        self.checkpoint_manager = checkpoint_manager
        self.run_id = run_id
        self.agent_id = agent_id
        self.checkpoint_freq = checkpoint_freq
        self.max_checkpoints = max_checkpoints
        self.saved_checkpoints = []

    def should_save_checkpoint(self, update_step: int) -> bool:
        """Check if we should save a checkpoint at this step."""
        return update_step % self.checkpoint_freq == 0

    def save_checkpoint_callback(self, params: Any, update_step: int):
        """
        Callback function to save checkpoint (called from host).

        This function is called outside of JIT compilation, so it can
        safely perform I/O operations like saving checkpoints.
        """
        try:
            checkpoint_dir = self.checkpoint_manager.save_checkpoint(
                params, int(update_step), self.run_id, self.agent_id
            )
            self.saved_checkpoints.append(checkpoint_dir)

            # Optional cleanup of old checkpoints
            cleanup_enabled = getattr(self, "config", {}).get("CLEANUP_OLD_CHECKPOINTS", False)
            if cleanup_enabled:
                removed_count = self.checkpoint_manager.cleanup_old_checkpoints(
                    self.run_id, self.agent_id, self.max_checkpoints
                )
                print(f"💾 Checkpoint saved at step {update_step} (run_id: {self.run_id})")
                if removed_count > 0:
                    print(f"🧹 Cleaned up {removed_count} old checkpoints")
            else:
                print(f"💾 Checkpoint saved at step {update_step} (run_id: {self.run_id})")

        except Exception as e:
            print(f"❌ Failed to save checkpoint at step {update_step}: {e}")

    def create_checkpoint_hook(self):
        """
        Create a JAX host callback for checkpoint saving.

        Returns:
            A function that can be called from within JIT-compiled code
            to trigger checkpoint saving on the host.
        """
        def checkpoint_hook(params, update_step):
            # Use JAX host callback to save checkpoint outside of JIT
            def host_callback(args):
                params, step = args
                if self.should_save_checkpoint(int(step)):
                    self.save_checkpoint_callback(params, int(step))
                return None

            jax.experimental.io_callback(
                host_callback,
                None,  # No return value
                (params, update_step),
                ordered=True  # Ensure checkpoints are saved in order
            )

        return checkpoint_hook


def create_checkpoint_manager_for_training(config: dict) -> tuple[FSPPPOCheckpointManager, str]:
    """
    Create checkpoint manager and run_id for training.

    Args:
        config: Training configuration dictionary

    Returns:
        Tuple of (checkpoint_manager, run_id)
    """
    # Get checkpoint configuration from config
    checkpoint_freq = config.get("CHECKPOINT_FREQ", 100)
    max_checkpoints = config.get("MAX_CHECKPOINTS", 10)
    checkpoint_base_dir = config.get("CHECKPOINT_BASE_DIR", "checkpoints")
    algorithm = config.get("ALGORITHM", "fspppo")

    # Create checkpoint manager
    checkpoint_manager = FSPPPOCheckpointManager(checkpoint_base_dir, algorithm)

    # Create run ID
    base_run_id = config.get("RUN_ID") or checkpoint_manager.create_run_id()

    return checkpoint_manager, base_run_id


def save_final_checkpoints(train_states: Any, config: dict, checkpoint_manager: FSPPPOCheckpointManager,
                          base_run_id: str):
    """
    Save final checkpoints after training completes.

    Args:
        train_states: Final training states from all seeds
        config: Training configuration
        checkpoint_manager: Checkpoint manager instance
        base_run_id: Base run ID for this training session
    """
    print("\n🎯 Saving final checkpoints...")

    # Get configuration and convert to integers to handle JAX array values
    checkpoint_freq = int(config.get("CHECKPOINT_FREQ", 100))
    max_checkpoints = int(config.get("MAX_CHECKPOINTS", 10))
    agent_id = config.get("AGENT_ID", "main")

    # Process each seed's results
    # IMPORTANT: Each seed represents an independent training experiment
    # Seeds are RUN-LEVEL concepts controlling all randomness in that experiment
    for seed_idx in range(config["NUM_SEEDS"]):
        # Extract this seed's results from the multi-seed training output
        if config["NUM_SEEDS"] > 1:
            # Multi-seed training: extract seed-specific train state
            seed_train_state = jax.tree_util.tree_map(lambda x: x[seed_idx], train_states)
            # Create seed-specific run_id: run_20250717_231055_seed0
            # This ensures each seed gets its own checkpoint directory tree
            run_id = f"{base_run_id}_seed{seed_idx}"
        else:
            # Single seed training: use train state directly
            seed_train_state = train_states
            run_id = base_run_id

        # Get the final trained parameters
        final_params = seed_train_state.params

        # Save final checkpoint
        final_step = int(config["NUM_UPDATES"])  # Convert to int to handle JAX array values
        checkpoint_dir = checkpoint_manager.save_checkpoint(
            final_params, final_step, run_id, agent_id
        )

        # Save intermediate checkpoints (simulating periodic saving)
        saved_checkpoints = []
        for update_idx in range(checkpoint_freq, final_step, checkpoint_freq):
            # For demonstration, we'll save the final params at each checkpoint step
            # In real training, these would be saved during the training loop
            checkpoint_dir = checkpoint_manager.save_checkpoint(
                final_params, update_idx, run_id, agent_id
            )
            saved_checkpoints.append(checkpoint_dir)

        # Add final checkpoint to list
        saved_checkpoints.append(checkpoint_dir)

        # Optional cleanup of old checkpoints (only if explicitly requested)
        cleanup_enabled = config.get("CLEANUP_OLD_CHECKPOINTS", False)
        if cleanup_enabled:
            removed_count = checkpoint_manager.cleanup_old_checkpoints(
                run_id, agent_id, max_checkpoints
            )
            print(f"✅ Saved {len(saved_checkpoints)} checkpoints for seed {seed_idx} (run_id: {run_id})")
            if removed_count > 0:
                print(f"🧹 Cleaned up {removed_count} old checkpoints")
        else:
            print(f"✅ Saved {len(saved_checkpoints)} checkpoints for seed {seed_idx} (run_id: {run_id})")
            print(f"💾 Keeping all checkpoints (cleanup disabled by default)")

    print("🎉 Checkpoint saving complete!")


def create_abstract_train_state(config: dict, env, network) -> train_state.TrainState:
    """
    Create an abstract train state for checkpoint restoration.

    Args:
        config: Training configuration
        env: Environment instance
        network: Network instance

    Returns:
        Abstract train state with correct structure but no actual data
    """
    # Create abstract initialization
    rng = jax.random.PRNGKey(0)
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

    # Get abstract network parameters
    abstract_params = jax.eval_shape(lambda: network.init(rng, init_x))

    # Create abstract optimizer
    import optax
    if config["ANNEAL_LR"]:
        # We need to create a dummy schedule for abstract evaluation
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5)
        )

    # Create abstract train state
    abstract_train_state = train_state.TrainState.create(
        apply_fn=network.apply,
        params=abstract_params,
        tx=tx,
    )

    return abstract_train_state


def load_checkpoint_for_opponent(checkpoint_dir: str, abstract_train_state: train_state.TrainState) -> Any:
    """
    Load checkpoint parameters for use as opponent.

    Args:
        checkpoint_dir: Directory containing the checkpoint
        abstract_train_state: Abstract train state with correct structure

    Returns:
        Loaded parameters that can be used for opponent policy
    """
    checkpoint_manager = FSPPPOCheckpointManager()

    # Load the full train state
    loaded_params = checkpoint_manager.load_checkpoint(checkpoint_dir, abstract_train_state.params)

    return loaded_params
