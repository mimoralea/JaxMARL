#!/usr/bin/env python3
"""
Test script for checkpoint management system.
Validates that checkpoints are created, saved correctly, and are different from each other.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add the parent directory to the path so we can import the modules
sys.path.append(str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from baselines.FSPPPO.checkpoint_manager import (
    save_checkpoint, load_checkpoint, get_available_runs,
    get_agent_checkpoints, validate_checkpoint_differences,
    print_checkpoint_summary, cleanup_old_checkpoints
)
from baselines.FSPPPO.fspppo_ff_mpe import make_train


def test_checkpoint_basic_functionality():
    """Test basic checkpoint save/load functionality."""
    print("🧪 Testing basic checkpoint save/load functionality...")

    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create dummy parameters
        dummy_params = {
            'layer1': jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            'layer2': jnp.array([5.0, 6.0])
        }

        # Save checkpoint
        checkpoint_path = save_checkpoint(
            dummy_params,
            update_step=100,
            algorithm="test_algo",
            run_id="test_run",
            agent_id="test_agent",
            base_dir=temp_dir
        )

        # Verify file was created
        assert os.path.exists(checkpoint_path), f"Checkpoint file not created: {checkpoint_path}"
        print(f"✅ Checkpoint saved successfully: {checkpoint_path}")

        # Load checkpoint
        loaded_params = load_checkpoint(checkpoint_path)

        # Verify parameters match
        for key in dummy_params:
            assert jnp.allclose(dummy_params[key], loaded_params[key]), f"Parameter mismatch for {key}"

        print("✅ Checkpoint loaded successfully and parameters match!")

        # Test directory structure
        expected_dir = os.path.join(temp_dir, "test_algo", "test_run", "test_agent")
        assert os.path.exists(expected_dir), f"Directory structure not created: {expected_dir}"
        print("✅ Directory structure created correctly!")


def test_checkpoint_differences():
    """Test that different checkpoints have different content."""
    print("\n🧪 Testing checkpoint differences...")

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_paths = []

        # Create multiple checkpoints with different parameters
        for i in range(3):
            params = {
                'weights': jnp.array([[i * 1.0, i * 2.0], [i * 3.0, i * 4.0]]),
                'bias': jnp.array([i * 5.0, i * 6.0])
            }

            checkpoint_path = save_checkpoint(
                params,
                update_step=(i + 1) * 100,
                algorithm="test_algo",
                run_id="test_run",
                agent_id="test_agent",
                base_dir=temp_dir
            )
            checkpoint_paths.append(checkpoint_path)

        # Validate that checkpoints are different
        hashes = validate_checkpoint_differences(checkpoint_paths)

        print(f"📊 Checkpoint hashes:")
        for path, hash_val in hashes.items():
            print(f"  {os.path.basename(path)}: {hash_val}")

        # Verify all hashes are different
        hash_values = list(hashes.values())
        assert len(set(hash_values)) == len(hash_values), "Some checkpoints have identical content!"
        print("✅ All checkpoints have different content (verified by MD5 hash)!")


def test_checkpoint_metadata():
    """Test checkpoint metadata functionality."""
    print("\n🧪 Testing checkpoint metadata...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save multiple checkpoints
        for i in range(3):
            params = {'data': jnp.array([i * 1.0, i * 2.0])}
            save_checkpoint(
                params,
                update_step=(i + 1) * 50,
                algorithm="test_algo",
                run_id="test_run",
                agent_id="test_agent",
                base_dir=temp_dir
            )

        # Get checkpoint list
        checkpoints = get_agent_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="test_agent",
            base_dir=temp_dir
        )

        assert len(checkpoints) == 3, f"Expected 3 checkpoints, got {len(checkpoints)}"
        print(f"✅ Found {len(checkpoints)} checkpoints in metadata!")

        # Verify checkpoints are sorted by update step
        update_steps = [cp['update_step'] for cp in checkpoints]
        assert update_steps == sorted(update_steps), "Checkpoints not sorted by update step!"
        print("✅ Checkpoints are properly sorted by update step!")

        # Print summary
        print_checkpoint_summary(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="test_agent",
            base_dir=temp_dir
        )


def test_checkpoint_cleanup():
    """Test checkpoint cleanup functionality."""
    print("\n🧪 Testing checkpoint cleanup...")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Save more checkpoints than the limit
        for i in range(15):
            params = {'data': jnp.array([i * 1.0])}
            save_checkpoint(
                params,
                update_step=(i + 1) * 10,
                algorithm="test_algo",
                run_id="test_run",
                agent_id="test_agent",
                base_dir=temp_dir
            )

        # Verify all checkpoints exist
        checkpoints_before = get_agent_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="test_agent",
            base_dir=temp_dir
        )
        assert len(checkpoints_before) == 15, f"Expected 15 checkpoints, got {len(checkpoints_before)}"
        print(f"✅ Created {len(checkpoints_before)} checkpoints!")

        # Cleanup to keep only 5 checkpoints
        removed_count = cleanup_old_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="test_agent",
            max_checkpoints=5,
            base_dir=temp_dir
        )

        # Verify cleanup worked
        checkpoints_after = get_agent_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="test_agent",
            base_dir=temp_dir
        )

        assert len(checkpoints_after) == 5, f"Expected 5 checkpoints after cleanup, got {len(checkpoints_after)}"
        assert removed_count == 10, f"Expected 10 checkpoints removed, got {removed_count}"
        print(f"✅ Cleanup successful! Removed {removed_count} old checkpoints, kept {len(checkpoints_after)}!")

        # Verify the remaining checkpoints are the most recent ones
        remaining_steps = [cp['update_step'] for cp in checkpoints_after]
        expected_steps = [110, 120, 130, 140, 150]  # Last 5 checkpoints
        assert remaining_steps == expected_steps, f"Expected {expected_steps}, got {remaining_steps}"
        print("✅ Kept the most recent checkpoints as expected!")


def test_training_integration():
    """Test checkpoint integration with actual training."""
    print("\n🧪 Testing checkpoint integration with training...")

    # Load minimal config for testing
    config = OmegaConf.load('baselines/FSPPPO/config/fspppo_ff_mpe.yaml')
    config = OmegaConf.to_container(config)

    # Use minimal settings for quick test
    config.update({
        'TOTAL_TIMESTEPS': 1024,
        'NUM_ENVS': 2,
        'NUM_STEPS': 32,
        'UPDATE_EPOCHS': 1,
        'NUM_MINIBATCHES': 1,
        'NUM_SEEDS': 1,
        'CHECKPOINT_FREQ': 10,  # Save every 10 updates
        'MAX_CHECKPOINTS': 5,
        'CHECKPOINT_BASE_DIR': 'test_checkpoints'
    })

    # Clean up any existing test checkpoints
    if os.path.exists('test_checkpoints'):
        shutil.rmtree('test_checkpoints')

    try:
        # Run training
        print("🚀 Running training with checkpoint integration...")
        train_fn = make_train(config)
        rng = jax.random.PRNGKey(42)
        train_jit = jax.jit(train_fn)
        result = train_jit(rng)

        run_id = result.get('run_id')
        assert run_id is not None, "Run ID not returned from training!"
        print(f"✅ Training completed with run_id: {run_id}")

        # Check that checkpoints were created
        checkpoints = get_agent_checkpoints(
            algorithm="fspppo",
            run_id=run_id,
            agent_id="main_agent",
            base_dir="test_checkpoints"
        )

        assert len(checkpoints) > 0, "No checkpoints were created during training!"
        print(f"✅ Created {len(checkpoints)} checkpoints during training!")

        # Validate checkpoint differences
        checkpoint_paths = [cp['file_path'] for cp in checkpoints]
        hashes = validate_checkpoint_differences(checkpoint_paths)

        if len(hashes) > 1:
            hash_values = list(hashes.values())
            unique_hashes = len(set(hash_values))
            print(f"✅ Checkpoint validation: {unique_hashes}/{len(hashes)} unique checkpoints!")

        # Print summary
        print_checkpoint_summary(
            algorithm="fspppo",
            run_id=run_id,
            agent_id="main_agent",
            base_dir="test_checkpoints"
        )

    finally:
        # Clean up test checkpoints
        if os.path.exists('test_checkpoints'):
            shutil.rmtree('test_checkpoints')
            print("🧹 Cleaned up test checkpoints")


def main():
    """Run all checkpoint tests."""
    print("🎯 Starting Checkpoint Management System Tests\n")

    try:
        test_checkpoint_basic_functionality()
        test_checkpoint_differences()
        test_checkpoint_metadata()
        test_checkpoint_cleanup()
        test_training_integration()

        print("\n🎉 All checkpoint tests passed successfully!")
        print("✅ Checkpoint management system is working correctly!")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
