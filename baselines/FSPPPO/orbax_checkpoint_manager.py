"""
Orbax-based checkpoint management utilities for FSPPPO and other algorithms using Orbax.
Provides hierarchical storage: checkpoints/{algorithm}/{run_id}/{agent_id}/{step}/
Uses Orbax for proper JAX/Flax checkpointing without JIT compilation issues.

CHECKPOINT DIRECTORY STRUCTURE DESIGN:
=====================================

Current Structure (CORRECT):
    checkpoints/fspppo/run_xyz_seed[0-9]/agent_id/{step}/

Example:
    checkpoints/fspppo/run_20250717_231055_seed0/main/12/
    checkpoints/fspppo/run_20250717_231055_seed1/main/12/

Why This Structure is Optimal:

1. **Conceptual Clarity**: Each seed = independent training experiment
   - Seeds control: network initialization, environment randomness, training randomness
   - Seeds are RUN-LEVEL concepts, not agent-level
   - seed0 and seed1 are different training runs, not different agents

2. **Future Population Training Support**:
   - Same seed controls both agents in training run (shared randomness)
   - Structure: run_xyz_seed0/main/ AND run_xyz_seed0/opponent/
   - Easy to load consistent agent pairs from same seed

3. **Opponent Sampling Benefits**:
   - Sample opponents from any seed's history across all runs
   - Mix opponents across different training runs (cross-seed sampling)
   - Simple recency-biased sampling across all available checkpoints

4. **Standard ML Practice**: Follows common ML conventions where seeds are experiment-level

Alternative Structure (REJECTED):
    checkpoints/fspppo/run_xyz/agent_id/seed[0-9]/{step}/

Problems with rejected approach:
    - Implies seeds are agent-specific (conceptually wrong)
    - Harder to maintain consistency across agents in population training
    - More complex opponent sampling logic
    - Doesn't follow standard ML experiment organization

Implementation Details:
    - Base run_id: "run_20250717_231055"
    - Seed-specific run_id: "run_20250717_231055_seed0"
    - Each seed gets independent checkpoint directory tree
    - Metadata tracks seed-specific information

This structure supports: single-agent training, population training,
Fictitious Self-Play, and cross-seed opponent sampling.
"""

import os
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp

# Configure logging to silence verbose output
logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("orbax").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.ERROR)
logging.getLogger("tensorstore").setLevel(logging.ERROR)


class FSPPPOCheckpointManager:
    """
    Checkpoint manager for FSPPPO using Orbax.

    This class handles saving and loading checkpoints during training,
    with support for hierarchical storage and cleanup of old checkpoints.
    """

    def __init__(
        self, base_dir: str = "checkpoints", algorithm: str = "fspppo"
    ):
        self.base_dir = Path(base_dir)
        self.algorithm = algorithm
        self.checkpointer = ocp.StandardCheckpointer()

    def create_run_id(self) -> str:
        """Generate timestamped run ID: run_YYYYMMDD_HHMMSS"""
        return datetime.now().strftime("run_%Y%m%d_%H%M%S")

    def get_checkpoint_dir(
        self, run_id: str, agent_id: str, update_step: int
    ) -> Path:
        """Get directory path for a checkpoint."""
        # Convert update_step to int to handle JAX array values
        step_int = int(update_step)
        checkpoint_dir = (
            self.base_dir
            / self.algorithm
            / run_id
            / agent_id
            / str(step_int)
        )
        return checkpoint_dir.resolve()  # Convert to absolute path

    def save_checkpoint(
        self,
        params: Any,
        update_step: int,
        run_id: Optional[str] = None,
        agent_id: str = "main",
    ) -> str:
        """
        Save checkpoint using Orbax.

        Args:
            params: JAX parameters to save (typically from train_state.params)
            update_step: Training update step number
            run_id: Training run ID (auto-generated if None)
            agent_id: Agent identifier

        Returns:
            Path to saved checkpoint directory
        """
        if run_id is None:
            run_id = self.create_run_id()

        # Convert update_step to int to handle JAX array values
        step_int = int(update_step)

        checkpoint_dir = self.get_checkpoint_dir(run_id, agent_id, step_int)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Log the checkpoint path before saving
        print(f"[FSPPPO] Saving {agent_id} checkpoint to: {checkpoint_dir.resolve()}")

        # Save checkpoint using Orbax (force=True allows overwriting)
        self.checkpointer.save(checkpoint_dir, params, force=True)

        # Wait for Orbax to complete the save operation
        self.checkpointer.wait_until_finished()

        # Save metadata after Orbax is done
        metadata = {
            "update_step": step_int,
            "timestamp": datetime.now().isoformat(),
            "algorithm": self.algorithm,
            "run_id": run_id,
            "agent_id": agent_id,
            "checkpoint_dir": str(checkpoint_dir),
        }

        metadata_path = checkpoint_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Update agent-level metadata
        self._update_agent_metadata(
            run_id, agent_id, step_int, str(checkpoint_dir)
        )

        return str(checkpoint_dir)

    def load_checkpoint(
        self, checkpoint_dir: str, abstract_params: Any
    ) -> Any:
        """
        Load checkpoint using Orbax.

        Args:
            checkpoint_dir: Path to checkpoint directory
            abstract_params: Abstract parameter structure for restoration

        Returns:
            Loaded parameters
        """
        return self.checkpointer.restore(checkpoint_dir, abstract_params)

    def _update_agent_metadata(
        self, run_id: str, agent_id: str, update_step: int, checkpoint_dir: str
    ):
        """Update agent-level metadata with new checkpoint info."""
        agent_dir = self.base_dir / self.algorithm / run_id / agent_id
        agent_metadata_path = agent_dir / "agent_metadata.json"

        # Load existing metadata
        if agent_metadata_path.exists():
            with open(agent_metadata_path, "r") as f:
                agent_metadata = json.load(f)
        else:
            agent_metadata = {"checkpoints": []}

        # Add new checkpoint info
        checkpoint_info = {
            "update_step": update_step,
            "timestamp": datetime.now().isoformat(),
            "checkpoint_dir": checkpoint_dir,
        }

        # Remove any existing entry for this step (in case of overwrite)
        agent_metadata["checkpoints"] = [
            cp
            for cp in agent_metadata["checkpoints"]
            if cp["update_step"] != update_step
        ]

        # Add new entry and sort by update step
        agent_metadata["checkpoints"].append(checkpoint_info)
        agent_metadata["checkpoints"].sort(key=lambda x: x["update_step"])

        # Save updated metadata
        with open(agent_metadata_path, "w") as f:
            json.dump(agent_metadata, f, indent=2)

    def get_available_runs(self) -> List[str]:
        """List all training runs for the algorithm."""
        algorithm_dir = self.base_dir / self.algorithm
        if not algorithm_dir.exists():
            return []

        runs = [
            d.name
            for d in algorithm_dir.iterdir()
            if d.is_dir() and d.name.startswith("run_")
        ]
        return sorted(runs)

    def get_agent_checkpoints(
        self, run_id: str, agent_id: str = "main"
    ) -> List[Dict]:
        """Get all checkpoints for a specific agent in a run."""
        agent_dir = self.base_dir / self.algorithm / run_id / agent_id
        agent_metadata_path = agent_dir / "agent_metadata.json"

        if not agent_metadata_path.exists():
            return []

        with open(agent_metadata_path, "r") as f:
            agent_metadata = json.load(f)

        return agent_metadata.get("checkpoints", [])

    def cleanup_old_checkpoints(
        self,
        run_id: str,
        agent_id: str = "main",
        max_checkpoints: int = 10,
    ) -> int:
        """
        Keep only N most recent checkpoints for an agent.

        Returns:
            Number of checkpoints removed
        """
        checkpoints = self.get_agent_checkpoints(run_id, agent_id)

        if len(checkpoints) <= max_checkpoints:
            return 0

        # Remove oldest checkpoints
        to_remove = checkpoints[:-max_checkpoints]
        removed_count = 0

        for checkpoint_info in to_remove:
            checkpoint_dir = Path(checkpoint_info["checkpoint_dir"])
            if checkpoint_dir.exists():
                # Remove the entire checkpoint directory
                import shutil

                shutil.rmtree(checkpoint_dir)
                removed_count += 1

        # Update agent metadata
        agent_dir = self.base_dir / self.algorithm / run_id / agent_id
        agent_metadata_path = agent_dir / "agent_metadata.json"

        if agent_metadata_path.exists():
            with open(agent_metadata_path, "r") as f:
                agent_metadata = json.load(f)

            # Keep only the recent checkpoints
            agent_metadata["checkpoints"] = checkpoints[-max_checkpoints:]

            with open(agent_metadata_path, "w") as f:
                json.dump(agent_metadata, f, indent=2)

        return removed_count

    def get_latest_checkpoint(
        self, run_id: str, agent_id: str = "main"
    ) -> Optional[Dict]:
        """Get the most recent checkpoint for an agent."""
        checkpoints = self.get_agent_checkpoints(run_id, agent_id)
        return checkpoints[-1] if checkpoints else None

    def get_checkpoint_for_step(
        self, run_id: str, agent_id: str, update_step: int
    ) -> Optional[Dict]:
        """Get checkpoint for a specific update step."""
        checkpoints = self.get_agent_checkpoints(run_id, agent_id)
        for checkpoint in checkpoints:
            if checkpoint["update_step"] == update_step:
                return checkpoint
        return None

    def print_checkpoint_summary(
        self, run_id: str, agent_id: str = "main"
    ):
        """Print a summary of checkpoints for debugging."""
        checkpoints = self.get_agent_checkpoints(run_id, agent_id)

        if not checkpoints:
            return

        # Checkpoint summary (commented out to reduce verbosity)
        # print(f"\nCheckpoint Summary for {self.algorithm}/{run_id}/{agent_id}:")
        # print(f"{'Update Step':<12} {'Timestamp':<20} {'Checkpoint Dir':<50}")
        # print("-" * 85)

        # for checkpoint in checkpoints:
        #     timestamp = checkpoint['timestamp'][:19] if len(checkpoint['timestamp']) > 19 else checkpoint['timestamp']
        #     checkpoint_dir = Path(checkpoint['checkpoint_dir']).name
        #     print(f"{checkpoint['update_step']:<12} {timestamp:<20} {checkpoint_dir:<50}")


# Convenience functions for backward compatibility
def create_run_id() -> str:
    """Generate timestamped run ID: run_YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def save_checkpoint(
    params: Any,
    update_step: int,
    algorithm: str = "fspppo",
    run_id: Optional[str] = None,
    agent_id: str = "main",
    base_dir: str = "checkpoints",
) -> str:
    """
    Save checkpoint using Orbax (convenience function).

    Args:
        params: JAX parameters to save
        update_step: Training update step number
        algorithm: Algorithm name (e.g., "fspppo", "mappo")
        run_id: Training run ID (auto-generated if None)
        agent_id: Agent identifier
        base_dir: Base checkpoint directory

    Returns:
        Path to saved checkpoint directory
    """
    manager = FSPPPOCheckpointManager(base_dir, algorithm)
    return manager.save_checkpoint(params, update_step, run_id, agent_id)


def load_checkpoint(checkpoint_dir: str, abstract_params: Any) -> Any:
    """Load parameters from checkpoint directory (convenience function)."""
    manager = FSPPPOCheckpointManager()
    return manager.load_checkpoint(checkpoint_dir, abstract_params)


def cleanup_old_checkpoints(
    algorithm: str = "fspppo",
    run_id: str = None,
    agent_id: str = "main",
    max_checkpoints: int = 10,
    base_dir: str = "checkpoints",
) -> int:
    """
    Keep only N most recent checkpoints for an agent (convenience function).

    Returns:
        Number of checkpoints removed
    """
    manager = FSPPPOCheckpointManager(base_dir, algorithm)

    if run_id is None:
        runs = manager.get_available_runs()
        if not runs:
            return 0
        run_id = runs[-1]  # Use most recent run

    return manager.cleanup_old_checkpoints(run_id, agent_id, max_checkpoints)


def get_agent_checkpoints(
    algorithm: str = "fspppo",
    run_id: str = None,
    agent_id: str = "main",
    base_dir: str = "checkpoints",
) -> List[Dict]:
    """Get all checkpoints for a specific agent in a run (convenience function)."""
    manager = FSPPPOCheckpointManager(base_dir, algorithm)

    if run_id is None:
        runs = manager.get_available_runs()
        if not runs:
            return []
        run_id = runs[-1]  # Use most recent run

    return manager.get_agent_checkpoints(run_id, agent_id)
