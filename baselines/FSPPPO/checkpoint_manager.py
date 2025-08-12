"""
Checkpoint management utilities for FSPPPO and other algorithms using Orbax.
Provides hierarchical storage: checkpoints/{algorithm}/{run_id}/{agent_id}/step_{step}/
"""

import os
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp


def create_run_id() -> str:
    """Generate timestamped run ID: run_YYYYMMDD_HHMMSS"""
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def get_checkpoint_path(algorithm: str, run_id: str, agent_id: str,
                       update_step: int, base_dir: str = "checkpoints") -> str:
    """Get full path for a checkpoint file."""
    filename = f"checkpoint_{update_step:06d}.pkl"
    return os.path.join(base_dir, algorithm, run_id, agent_id, filename)


def save_checkpoint(params: Any, update_step: int, algorithm: str = "fspppo",
                   run_id: Optional[str] = None, agent_id: str = "main_agent",
                   base_dir: str = "checkpoints") -> str:
    """
    Save checkpoint with hierarchical structure.

    Args:
        params: JAX parameters to save
        update_step: Training update step number
        algorithm: Algorithm name (e.g., "fspppo", "mappo")
        run_id: Training run ID (auto-generated if None)
        agent_id: Agent identifier
        base_dir: Base checkpoint directory

    Returns:
        Path to saved checkpoint file
    """
    if run_id is None:
        run_id = create_run_id()

    # Create directory structure
    checkpoint_dir = os.path.join(base_dir, algorithm, run_id, agent_id)
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save checkpoint
    checkpoint_path = get_checkpoint_path(algorithm, run_id, agent_id, update_step, base_dir)
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(params, f)

    # Update metadata
    metadata_path = os.path.join(checkpoint_dir, "metadata.json")
    metadata = load_metadata(metadata_path) if os.path.exists(metadata_path) else {}

    # Calculate MD5 hash for validation
    checkpoint_hash = calculate_checkpoint_hash(checkpoint_path)

    metadata[f"checkpoint_{update_step:06d}"] = {
        "update_step": update_step,
        "timestamp": datetime.now().isoformat(),
        "file_path": checkpoint_path,
        "md5_hash": checkpoint_hash
    }

    save_metadata(metadata, metadata_path)

    return checkpoint_path


def load_checkpoint(checkpoint_path: str) -> Any:
    """Load parameters from checkpoint file."""
    with open(checkpoint_path, 'rb') as f:
        return pickle.load(f)


def calculate_checkpoint_hash(checkpoint_path: str) -> str:
    """Calculate MD5 hash of checkpoint file for validation."""
    hash_md5 = hashlib.md5()
    with open(checkpoint_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_available_runs(algorithm: str = "fspppo", base_dir: str = "checkpoints") -> List[str]:
    """List all training runs for an algorithm."""
    algorithm_dir = os.path.join(base_dir, algorithm)
    if not os.path.exists(algorithm_dir):
        return []

    runs = [d for d in os.listdir(algorithm_dir)
            if os.path.isdir(os.path.join(algorithm_dir, d)) and d.startswith("run_")]
    return sorted(runs)


def get_agent_checkpoints(algorithm: str = "fspppo", run_id: str = None,
                         agent_id: str = "main_agent", base_dir: str = "checkpoints") -> List[Dict]:
    """Get all checkpoints for a specific agent in a run."""
    if run_id is None:
        runs = get_available_runs(algorithm, base_dir)
        if not runs:
            return []
        run_id = runs[-1]  # Use most recent run

    agent_dir = os.path.join(base_dir, algorithm, run_id, agent_id)
    metadata_path = os.path.join(agent_dir, "metadata.json")

    if not os.path.exists(metadata_path):
        return []

    metadata = load_metadata(metadata_path)
    checkpoints = []

    for checkpoint_key, checkpoint_info in metadata.items():
        if checkpoint_key.startswith("checkpoint_"):
            checkpoints.append(checkpoint_info)

    return sorted(checkpoints, key=lambda x: x["update_step"])


def cleanup_old_checkpoints(algorithm: str = "fspppo", run_id: str = None,
                           agent_id: str = "main_agent", max_checkpoints: int = 10,
                           base_dir: str = "checkpoints") -> int:
    """
    Keep only N most recent checkpoints for an agent.

    Returns:
        Number of checkpoints removed
    """
    checkpoints = get_agent_checkpoints(algorithm, run_id, agent_id, base_dir)

    if len(checkpoints) <= max_checkpoints:
        return 0

    # Remove oldest checkpoints
    to_remove = checkpoints[:-max_checkpoints]
    removed_count = 0

    for checkpoint_info in to_remove:
        checkpoint_path = checkpoint_info["file_path"]
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            removed_count += 1

    # Update metadata
    if run_id is None:
        runs = get_available_runs(algorithm, base_dir)
        if runs:
            run_id = runs[-1]

    if run_id:
        agent_dir = os.path.join(base_dir, algorithm, run_id, agent_id)
        metadata_path = os.path.join(agent_dir, "metadata.json")

        if os.path.exists(metadata_path):
            metadata = load_metadata(metadata_path)
            # Remove metadata entries for deleted checkpoints
            for checkpoint_info in to_remove:
                checkpoint_key = f"checkpoint_{checkpoint_info['update_step']:06d}"
                if checkpoint_key in metadata:
                    del metadata[checkpoint_key]
            save_metadata(metadata, metadata_path)

    return removed_count


def load_metadata(metadata_path: str) -> Dict:
    """Load metadata from JSON file."""
    try:
        with open(metadata_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_metadata(metadata: Dict, metadata_path: str):
    """Save metadata to JSON file."""
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def validate_checkpoint_differences(checkpoint_paths: List[str]) -> Dict[str, str]:
    """
    Validate that checkpoints are different by comparing MD5 hashes.

    Returns:
        Dictionary mapping checkpoint paths to their MD5 hashes
    """
    hashes = {}
    for path in checkpoint_paths:
        if os.path.exists(path):
            hashes[path] = calculate_checkpoint_hash(path)
    return hashes


def print_checkpoint_summary(algorithm: str = "fspppo", run_id: str = None,
                           agent_id: str = "main_agent", base_dir: str = "checkpoints"):
    """Print a summary of checkpoints for debugging."""
    checkpoints = get_agent_checkpoints(algorithm, run_id, agent_id, base_dir)

    if not checkpoints:
        print(f"No checkpoints found for {algorithm}/{run_id}/{agent_id}")
        return

    print(f"\nCheckpoint Summary for {algorithm}/{run_id}/{agent_id}:")
    print(f"{'Update Step':<12} {'Timestamp':<20} {'MD5 Hash':<32}")
    print("-" * 70)

    for checkpoint in checkpoints:
        print(f"{checkpoint['update_step']:<12} {checkpoint['timestamp'][:19]:<20} {checkpoint['md5_hash']:<32}")
