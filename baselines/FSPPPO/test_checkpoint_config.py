#!/usr/bin/env python3
"""
Test script to verify checkpoint configuration parameters are working correctly.
"""

import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from omegaconf import OmegaConf
from jax_checkpoint_utils import create_checkpoint_manager_for_training

def test_checkpoint_config():
    """Test that checkpoint configuration parameters are properly loaded and used."""
    
    # Load the config file
    config_path = Path(__file__).parent / "config" / "fspppo_ff_mpe.yaml"
    config = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config)
    
    print("=== Checkpoint Configuration Test ===")
    print(f"CHECKPOINT_FREQ: {config.get('CHECKPOINT_FREQ', 'NOT SET')}")
    print(f"SAVE_CHECKPOINT_AT_END: {config.get('SAVE_CHECKPOINT_AT_END', 'NOT SET')}")
    print(f"MAX_CHECKPOINTS: {config.get('MAX_CHECKPOINTS', 'NOT SET')}")
    print(f"CHECKPOINT_BASE_DIR: {config.get('CHECKPOINT_BASE_DIR', 'NOT SET')}")
    print(f"AGENT_ID: {config.get('AGENT_ID', 'NOT SET')}")
    
    # Test checkpoint manager creation
    try:
        checkpoint_manager, base_run_id = create_checkpoint_manager_for_training(config)
        print(f"\n✅ Checkpoint manager created successfully!")
        print(f"Base run ID: {base_run_id}")
        print(f"Checkpoint manager algorithm: {checkpoint_manager.algorithm}")
        print(f"Checkpoint manager base_dir: {checkpoint_manager.base_dir}")
    except Exception as e:
        print(f"\n❌ Failed to create checkpoint manager: {e}")
        return False
    
    # Test config parameter usage
    checkpoint_freq = config.get("CHECKPOINT_FREQ", 0)
    save_checkpoint_at_end = config.get("SAVE_CHECKPOINT_AT_END", True)
    
    print(f"\n=== Configuration Logic Test ===")
    print(f"Checkpoint frequency: {checkpoint_freq} iterations")
    print(f"Save at end: {save_checkpoint_at_end}")
    
    if checkpoint_freq > 0:
        print(f"✅ Periodic checkpointing enabled (every {checkpoint_freq} iterations)")
    else:
        print("ℹ️  Periodic checkpointing disabled")
    
    if save_checkpoint_at_end:
        print("✅ Final checkpoint saving enabled")
    else:
        print("ℹ️  Final checkpoint saving disabled")
    
    # Test training iteration logic
    print(f"\n=== Training Iteration Logic Test ===")
    num_updates = 100  # Example
    
    checkpoint_steps = []
    for update_idx in range(num_updates):
        if checkpoint_freq > 0 and (update_idx + 1) % checkpoint_freq == 0:
            checkpoint_steps.append(update_idx + 1)
    
    if checkpoint_steps:
        print(f"Checkpoints would be saved at iterations: {checkpoint_steps}")
    else:
        print("No periodic checkpoints would be saved")
    
    if save_checkpoint_at_end:
        print(f"Final checkpoint would be saved at iteration: {num_updates}")
    
    print(f"\n✅ All checkpoint configuration tests passed!")
    return True

if __name__ == "__main__":
    success = test_checkpoint_config()
    sys.exit(0 if success else 1)
