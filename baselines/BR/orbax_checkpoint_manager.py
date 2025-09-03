"""Orbax-based checkpoint management for BR.

Provides unified checkpoint saving/loading for BR using the same structure
as FSPPPO to enable cross-algorithm evaluation and comparison.

Structure: checkpoints/br/run_{run_id}_seed{X}/agent_{0|1}/step_{step}/
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import orbax.checkpoint as ocp
from flax.training.train_state import TrainState


class BRCheckpointManager:
    """Checkpoint manager for BR using Orbax."""

    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: Optional[int] = 10,
        agent_names: List[str] = None,
    ):
        """Initialize checkpoint manager.

        Args:
            checkpoint_dir: Base directory for checkpoints (e.g., "checkpoints/br/run_xyz_seed0")
            max_to_keep: Maximum number of checkpoints to keep (None = keep all)
            agent_names: Names of the agents (default: ["main", "opponent"])
        """
        # Convert to absolute path to satisfy Orbax requirements
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.agent_names = agent_names or ["main", "opponent"]
        self.max_to_keep = max_to_keep

        # Create agent directories and managers
        self.agent_dirs = {}
        self.managers = {}

        for agent_name in self.agent_names:
            agent_dir = self.checkpoint_dir / agent_name
            agent_dir.mkdir(parents=True, exist_ok=True)
            self.agent_dirs[agent_name] = agent_dir

            # Initialize Orbax manager for this agent (use absolute path)
            self.managers[agent_name] = ocp.CheckpointManager(
                directory=str(agent_dir.resolve()),
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
        train_states: Dict[str, TrainState],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Save checkpoints for all agents at the given step.

        Args:
            step: Training step number
            train_states: Dictionary mapping agent names to TrainStates
            metadata: Optional metadata dictionary

        Returns:
            Dictionary mapping agent names to checkpoint paths
        """
        if metadata is None:
            metadata = {}

        # Add default metadata
        metadata.update(
            {
                "step": step,
                "timestamp": time.time(),
                "algorithm": "br",
                "num_agents": len(self.agent_names),
            }
        )

        checkpoint_paths = {}

        for agent_name in self.agent_names:
            if agent_name not in train_states:
                raise ValueError(
                    f"Missing train_state for agent: {agent_name}"
                )

            # Convert metadata to supported types (no strings)
            clean_metadata = {
                "step": step,
                "agent_id": 0 if agent_name == "main" else 1,
                "training_step": metadata.get("training_step", step),
                "is_final": int(metadata.get("is_final", False)),
            }

            save_args = {
                "train_state": train_states[agent_name],
                "metadata": clean_metadata,
            }

            # Convert step to int to handle float values from JAX
            step_int = int(step)

            # Orbax creates directory with just the step number (no step_ prefix)
            actual_checkpoint_path = self.agent_dirs[agent_name] / str(step_int)

            # Log the actual checkpoint path that Orbax will create
            print(f"[BR] Saving {agent_name} checkpoint to: {actual_checkpoint_path.resolve()}")

            self.managers[agent_name].save(step_int, save_args)
            checkpoint_paths[agent_name] = str(actual_checkpoint_path.resolve())

        return checkpoint_paths

    def load_checkpoint(
        self, step: int
    ) -> Tuple[Dict[str, TrainState], Dict[str, Any]]:
        """Load checkpoints for all agents from the given step.

        Args:
            step: Training step number to load

        Returns:
            Tuple of (train_states_dict, metadata)
        """
        train_states = {}
        metadata = None

        for agent_name in self.agent_names:
            restored = self.managers[agent_name].restore(step)
            train_states[agent_name] = restored["train_state"]

            # Use metadata from first agent (should be consistent)
            if metadata is None:
                metadata = restored["metadata"]

        return train_states, metadata

    def load_latest_checkpoint(
        self,
    ) -> Optional[Tuple[Dict[str, TrainState], Dict[str, Any]]]:
        """Load the most recent checkpoint for all agents.

        Returns:
            Tuple of (train_states_dict, metadata) or None if no checkpoints exist
        """
        # Get latest step from first agent (should be consistent across agents)
        first_agent = self.agent_names[0]
        latest_step = self.managers[first_agent].latest_step()

        if latest_step is None:
            return None

        return self.load_checkpoint(latest_step)

    def list_checkpoints(self) -> List[int]:
        """List all available checkpoint steps.

        Returns:
            Sorted list of checkpoint step numbers
        """
        # Use first agent's checkpoints (should be consistent across agents)
        first_agent = self.agent_names[0]
        return sorted(self.managers[first_agent].all_steps())

    def cleanup_old_checkpoints(self) -> Dict[str, int]:
        """Remove old checkpoints beyond max_to_keep limit for all agents.

        Returns:
            Dictionary mapping agent names to number of checkpoints removed
        """
        if self.max_to_keep is None:
            return {agent: 0 for agent in self.agent_names}

        removed_counts = {}

        for agent_name in self.agent_names:
            checkpoints = sorted(self.managers[agent_name].all_steps())
            if len(checkpoints) <= self.max_to_keep:
                removed_counts[agent_name] = 0
                continue

            # Remove oldest checkpoints
            to_remove = checkpoints[: -self.max_to_keep]
            removed_count = 0

            for step in to_remove:
                try:
                    checkpoint_path = (
                        self.agent_dirs[agent_name] / f"step_{step:06d}"
                    )
                    if checkpoint_path.exists():
                        import shutil

                        shutil.rmtree(checkpoint_path)
                        removed_count += 1
                except Exception as e:
                    print(
                        f"Warning: Failed to remove checkpoint {step} for {agent_name}: {e}"
                    )

            removed_counts[agent_name] = removed_count

        return removed_counts

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about this checkpoint run.

        Returns:
            Dictionary with run information
        """
        checkpoints = self.list_checkpoints()

        info = {
            "checkpoint_dir": str(self.checkpoint_dir),
            "agent_names": self.agent_names,
            "num_agents": len(self.agent_names),
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


def create_br_checkpoint_manager(
    run_id: str,
    seed: int,
    base_dir: str = "checkpoints",
    max_to_keep: Optional[int] = 10,
    agent_names: List[str] = None,
) -> BRCheckpointManager:
    """Create a checkpoint manager for BR with standard directory structure.

    Args:
        run_id: Run identifier (e.g., "run_20250718_170900")
        seed: Seed number
        base_dir: Base checkpoint directory
        max_to_keep: Maximum checkpoints to keep
        agent_names: Names of the agents (default: ["main", "opponent"])

    Returns:
        Configured BRCheckpointManager
    """
    checkpoint_dir = os.path.join(base_dir, "br", f"{run_id}_seed{seed}")
    return BRCheckpointManager(
        checkpoint_dir=checkpoint_dir,
        max_to_keep=max_to_keep,
        agent_names=agent_names or ["main", "opponent"],
    )


# Callback for integration with JAX training loops
class BRCheckpointCallback:
    """Callback for saving checkpoints during BR training."""

    def __init__(
        self,
        checkpoint_manager: BRCheckpointManager,
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
        self,
        step: int,
        train_states: Dict[str, TrainState],
        is_final: bool = False,
    ) -> Optional[Dict[str, str]]:
        """Save checkpoint if conditions are met.

        Args:
            step: Current training step
            train_states: Dictionary of current training states for all agents
            is_final: Whether this is the final step

        Returns:
            Dictionary of checkpoint paths or None
        """
        should_save = (
            (step % self.save_frequency == 0 and step > 0)
            or (is_final and self.save_at_end)
            or step == 0  # Always save initial checkpoint
        )

        if should_save and step not in self.saved_steps:
            metadata = {
                "training_step": step,
                "is_final": is_final,
                "algorithm": "br",
            }

            checkpoint_paths = self.checkpoint_manager.save_checkpoint(
                step=step, train_states=train_states, metadata=metadata
            )

            self.saved_steps.add(step)
            return checkpoint_paths

        return None

    def save_final_checkpoint(
        self, train_state, step: int
    ) -> Optional[Dict[str, str]]:
        """Save final checkpoint for BR training.

        Args:
            train_state: Training state containing params for both agents
            step: Final training step

        Returns:
            Dictionary of checkpoint paths or None
        """
        # Extract individual agent states from the combined train_state
        train_states = {
            "main": train_state.replace(params=train_state.params[0]),
            "opponent": train_state.replace(params=train_state.params[1]),
        }

        return self.__call__(
            step=step, train_states=train_states, is_final=True
        )


# Utility functions for backward compatibility
def load_br_pickle_checkpoint(pickle_path: str) -> Any:
    """Load legacy BR pickle checkpoint.

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
    pickle_paths: Dict[str, str],
    output_manager: BRCheckpointManager,
    step: int = 0,
    network_apply_fn=None,
    tx=None,
) -> Dict[str, str]:
    """Convert legacy pickle checkpoints to Orbax format.

    Args:
        pickle_paths: Dictionary mapping agent names to pickle file paths
        output_manager: Orbax checkpoint manager for output
        step: Step number to assign
        network_apply_fn: Network apply function for TrainState
        tx: Optimizer for TrainState

    Returns:
        Dictionary of paths to converted checkpoints
    """
    # Load pickle parameters
    train_states = {}

    for agent_name, pickle_path in pickle_paths.items():
        params = load_br_pickle_checkpoint(pickle_path)

        # Create TrainState (requires network and optimizer)
        if network_apply_fn is None or tx is None:
            raise ValueError("network_apply_fn and tx required for conversion")

        train_state = TrainState.create(
            apply_fn=network_apply_fn,
            params=params,
            tx=tx,
        )

        train_states[agent_name] = train_state

    # Save as Orbax checkpoint
    metadata = {
        "converted_from_pickle": True,
        "original_paths": pickle_paths,
        "conversion_timestamp": time.time(),
    }

    return output_manager.save_checkpoint(step, train_states, metadata)
