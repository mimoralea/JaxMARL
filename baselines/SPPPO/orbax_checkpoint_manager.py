"""Orbax-based checkpoint management for SPPPO.

Provides unified checkpoint saving/loading for SPPPO using the same structure
as FSPPPO to enable cross-algorithm evaluation and comparison.

Structure: checkpoints/spppo/run_{run_id}_seed{X}/main/step_{step}/
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import jax
import orbax.checkpoint as ocp
from flax.training.train_state import TrainState


class SPPPOCheckpointManager:
    """Checkpoint manager for SPPPO using Orbax."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: Optional[int] = 10,
        agent_name: str = "main",
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints (e.g., "checkpoints/spppo/run_xyz_seed0")
            max_to_keep: Maximum number of checkpoints to keep (None = keep all)
            agent_name: Name of the agent directory (default: "main")
        """
        self.checkpoint_dir = Path(
            checkpoint_dir
        ).resolve()  # Ensure absolute path
        self.agent_name = agent_name
        self.agent_dir = self.checkpoint_dir / agent_name
        self.max_to_keep = max_to_keep

        # Create directories
        self.agent_dir.mkdir(parents=True, exist_ok=True)

        # Initialize Orbax manager
        self.manager = ocp.CheckpointManager(
            directory=str(self.agent_dir.resolve()),  # Ensure absolute path
            checkpointers={
                "train_state": ocp.PyTreeCheckpointer(),
                "metadata": ocp.StandardCheckpointer(),
            },
            options=ocp.CheckpointManagerOptions(
                max_to_keep=max_to_keep,
                create=True,
            ),
        )

    def save_checkpoint(
        self,
        step: int,
        train_state: TrainState,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Save a checkpoint at the given step.

        Args:
            step: Training step number
            train_state: Flax TrainState to save
            metadata: Optional metadata dictionary

        Returns:
            Path to saved checkpoint
        """
        if metadata is None:
            metadata = {}

        # Add default metadata (only numeric types for StandardCheckpointer)
        metadata.update(
            {
                "step": step,
                "timestamp": time.time(),
                # Note: 'algorithm' and 'agent_type' strings removed for StandardCheckpointer compatibility
            }
        )

        # Save checkpoint
        save_args = {
            "train_state": train_state,
            "metadata": metadata,
        }

        # Convert step to int to handle float values from JAX
        step_int = int(step)
        
        # Orbax creates directory with just the step number (no step_ prefix)
        actual_checkpoint_path = self.agent_dir / str(step_int)
        
        # Log the actual checkpoint path that Orbax will create
        print(f"[SPPPO] Saving main checkpoint to: {actual_checkpoint_path.resolve()}")
        
        self.manager.save(step_int, save_args)
        return str(actual_checkpoint_path.resolve())

    def load_checkpoint(self, step: int) -> Tuple[TrainState, Dict[str, Any]]:
        """Load a checkpoint from the given step.

        Args:
            step: Training step number to load

        Returns:
            Tuple of (train_state, metadata)
        """
        restored = self.manager.restore(step)
        return restored["train_state"], restored["metadata"]

    def load_latest_checkpoint(
        self,
    ) -> Optional[Tuple[TrainState, Dict[str, Any]]]:
        """Load the most recent checkpoint.

        Returns:
            Tuple of (train_state, metadata) or None if no checkpoints exist
        """
        latest_step = self.manager.latest_step()
        if latest_step is None:
            return None

        return self.load_checkpoint(latest_step)

    def list_checkpoints(self) -> List[int]:
        """List all available checkpoint steps.

        Returns:
            Sorted list of checkpoint step numbers
        """
        return sorted(self.manager.all_steps())

    def cleanup_old_checkpoints(self) -> int:
        """Remove old checkpoints beyond max_to_keep limit.

        Returns:
            Number of checkpoints removed
        """
        if self.max_to_keep is None:
            return 0

        checkpoints = self.list_checkpoints()
        if len(checkpoints) <= self.max_to_keep:
            return 0

        # Remove oldest checkpoints
        to_remove = checkpoints[: -self.max_to_keep]
        removed_count = 0

        for step in to_remove:
            try:
                checkpoint_path = self.agent_dir / f"step_{step:06d}"
                if checkpoint_path.exists():
                    import shutil

                    shutil.rmtree(checkpoint_path)
                    removed_count += 1
            except Exception as e:
                print(f"Warning: Failed to remove checkpoint {step}: {e}")

        return removed_count

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about this checkpoint run.

        Returns:
            Dictionary with run information
        """
        checkpoints = self.list_checkpoints()

        info = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "agent_name": self.agent_name,
            "num_checkpoints": len(checkpoints),
            "steps": checkpoints,
            "latest_step": checkpoints[-1] if checkpoints else None,
            "max_to_keep": self.max_to_keep,
        }

        # Add metadata from latest checkpoint if available
        if checkpoints:
            try:
                _, metadata = self.load_checkpoint(checkpoints[-1])
                info["latest_metadata"] = metadata
            except Exception:
                pass

        return info


def create_spppo_checkpoint_manager(
    run_id: str,
    seed: int,
    base_dir: str = "checkpoints",
    max_to_keep: Optional[int] = 10,
) -> SPPPOCheckpointManager:
    """Create a checkpoint manager for SPPPO with standard directory structure.

    Args:
        run_id: Run identifier (e.g., "run_20250718_170900")
        seed: Seed number
        base_dir: Base checkpoint directory
        max_to_keep: Maximum checkpoints to keep

    Returns:
        Configured SPPPOCheckpointManager
    """
    checkpoint_dir = os.path.join(base_dir, "spppo", f"{run_id}_seed{seed}")
    return SPPPOCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep,
        agent_name="main",
    )


# Callback for integration with JAX training loops
class SPPPOCheckpointCallback:
    """Callback for saving checkpoints during SPPPO training."""

    def __init__(
        self,
        checkpoint_manager: SPPPOCheckpointManager,
        save_frequency: int = 325,
        save_at_end: bool = True,
    ):
        """Initialize checkpoint callback.

        Args:
            checkpoint_manager: The checkpoint manager to use
            save_frequency: Save checkpoint every N training iterations
            save_at_end: Whether to save a final checkpoint
        """
        self.checkpoint_manager = checkpoint_manager
        self.save_frequency = save_frequency
        self.save_at_end = save_at_end
        self.saved_steps = set()

    def __call__(
        self, step: int, train_state: TrainState, is_final: bool = False
    ) -> Optional[str]:
        """Save checkpoint if conditions are met.

        Args:
            step: Current training step
            train_state: Current training state
            is_final: Whether this is the final step

        Returns:
            Path to saved checkpoint or None
        """
        should_save = (
            (step % self.save_frequency == 0 and step > 0)
            or (is_final and self.save_at_end)
            or step == 0  # Always save initial checkpoint
        )

        if should_save and step not in self.saved_steps:
            metadata = {
                "training_step": step,
                "is_final": int(
                    is_final
                ),  # Convert bool to int for StandardCheckpointer
                # Note: 'algorithm' string removed for StandardCheckpointer compatibility
            }

            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                step=step, train_state=train_state, metadata=metadata
            )

            self.saved_steps.add(step)
            return checkpoint_path

        return None

    def save_final_checkpoint(
        self, train_state: TrainState, step: int
    ) -> Optional[str]:
        """Save final checkpoint at the end of training.

        Args:
            train_state: Current training state
            step: Final training step

        Returns:
            Path to saved checkpoint or None
        """
        return self.__call__(step=step, train_state=train_state, is_final=True)


# Utility functions for backward compatibility
def load_spppo_pickle_checkpoint(pickle_path: str) -> Any:
    """Load legacy SPPPO pickle checkpoint.

    Args:
        pickle_path: Path to pickle file

    Returns:
        Loaded parameters
    """
    import pickle

    with open(pickle_path, "rb") as f:
        params = pickle.load(f)

    return params


def convert_pickle_to_orbax(
    pickle_path: str,
    output_manager: SPPPOCheckpointManager,
    step: int = 0,
    network_apply_fn=None,
    tx=None,
) -> str:
    """Convert legacy pickle checkpoint to Orbax format.

    Args:
        pickle_path: Path to pickle file
        output_manager: Orbax checkpoint manager for output
        step: Step number to assign
        network_apply_fn: Network apply function for TrainState
        tx: Optimizer for TrainState

    Returns:
        Path to converted checkpoint
    """
    # Load pickle parameters
    params = load_spppo_pickle_checkpoint(pickle_path)

    # Create TrainState (requires network and optimizer)
    if network_apply_fn is None or tx is None:
        raise ValueError("network_apply_fn and tx required for conversion")

    train_state = TrainState.create(
        apply_fn=network_apply_fn,
        params=params,
        tx=tx,
    )

    # Save as Orbax checkpoint
    metadata = {
        "converted_from_pickle": True,
        "original_path": pickle_path,
        "conversion_timestamp": time.time(),
    }

    return output_manager.save_checkpoint(step, train_state, metadata)
