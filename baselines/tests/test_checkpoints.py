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

import hashlib
import pytest

# Skip this test module if required deps are missing
pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("optax")
pytest.importorskip("orbax")

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
from flax.training import train_state
import optax

from baselines.FSPPPO.orbax_checkpoint_manager import (
    save_checkpoint, load_checkpoint,
    get_agent_checkpoints, cleanup_old_checkpoints,
    FSPPPOCheckpointManager,
)
from baselines.FSPPPO.jax_checkpoint_utils import (
    create_checkpoint_manager_for_training,
    save_final_checkpoints,
)


# Helpers
def make_abstract_from_params(params):
    """Create abstract shape/dtype structure for Orbax restore."""
    return jax.tree_util.tree_map(
        lambda x: jax.ShapeDtypeStruct(x.shape, x.dtype), params
    )


def hash_params(params) -> str:
    """Compute MD5 hash of a PyTree of arrays (deterministic order)."""
    m = hashlib.md5()
    leaves, _ = jax.tree_util.tree_flatten(params)
    for leaf in leaves:
        arr = np.asarray(leaf)
        m.update(arr.tobytes())
    return m.hexdigest()


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
            agent_id="main",
            base_dir=temp_dir
        )

        # Verify file was created
        assert os.path.exists(checkpoint_path), f"Checkpoint file not created: {checkpoint_path}"
        print(f"✅ Checkpoint saved successfully: {checkpoint_path}")

        # Load checkpoint
        abstract = make_abstract_from_params(dummy_params)
        loaded_params = load_checkpoint(checkpoint_path, abstract)

        # Verify parameters match
        for key in dummy_params:
            assert jnp.allclose(dummy_params[key], loaded_params[key]), f"Parameter mismatch for {key}"

        print("✅ Checkpoint loaded successfully and parameters match!")

        # Test directory structure (numeric step dir)
        expected_dir = os.path.join(temp_dir, "test_algo", "test_run", "main", "100")
        assert os.path.exists(expected_dir), f"Directory structure not created: {expected_dir}"
        assert os.path.exists(os.path.join(expected_dir, "metadata.json")), "Metadata file missing!"
        print("✅ Directory structure and metadata created correctly!")


def test_checkpoint_differences():
    """Test that different checkpoints have different content."""
    print("\n🧪 Testing checkpoint differences...")

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_paths = []
        saved_params_list = []

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
                agent_id="main",
                base_dir=temp_dir
            )
            checkpoint_paths.append(checkpoint_path)
            saved_params_list.append(params)

        # Validate that checkpoints are different by restoring and hashing
        hashes = {}
        for ckpt_path, original_params in zip(checkpoint_paths, saved_params_list):
            abstract = make_abstract_from_params(original_params)
            restored = load_checkpoint(ckpt_path, abstract)
            hashes[ckpt_path] = hash_params(restored)

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
                agent_id="main",
                base_dir=temp_dir
            )

        # Get checkpoint list
        checkpoints = get_agent_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="main",
            base_dir=temp_dir
        )

        assert len(checkpoints) == 3, f"Expected 3 checkpoints, got {len(checkpoints)}"
        print(f"✅ Found {len(checkpoints)} checkpoints in metadata!")

        # Verify checkpoints are sorted by update step
        update_steps = [cp['update_step'] for cp in checkpoints]
        assert update_steps == sorted(update_steps), "Checkpoints not sorted by update step!"
        print("✅ Checkpoints are properly sorted by update step!")

        # Print summary (optional)
        manager = FSPPPOCheckpointManager(temp_dir, "test_algo")
        manager.print_checkpoint_summary("test_run", "main")


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
                agent_id="main",
                base_dir=temp_dir
            )

        # Verify all checkpoints exist
        checkpoints_before = get_agent_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="main",
            base_dir=temp_dir
        )
        assert len(checkpoints_before) == 15, f"Expected 15 checkpoints, got {len(checkpoints_before)}"
        print(f"✅ Created {len(checkpoints_before)} checkpoints!")

        # Cleanup to keep only 5 checkpoints
        removed_count = cleanup_old_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="main",
            max_checkpoints=5,
            base_dir=temp_dir
        )

        # Verify cleanup worked
        checkpoints_after = get_agent_checkpoints(
            algorithm="test_algo",
            run_id="test_run",
            agent_id="main",
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
    """Test checkpoint integration using jax_checkpoint_utils (no heavy training)."""
    print("\n🧪 Testing checkpoint integration (lightweight)...")

    # Minimal config
    config = {
        'CHECKPOINT_FREQ': 10,
        'MAX_CHECKPOINTS': 5,
        'CHECKPOINT_BASE_DIR': tempfile.mkdtemp(),
        'ALGORITHM': 'fspppo',
        'NUM_SEEDS': 1,
        'NUM_UPDATES': 30,
        'AGENT_ID': 'main',
    }

    # Simple network for train state
    class SimpleNet(nn.Module):
        features: int = 8
        @nn.compact
        def __call__(self, x):
            x = nn.Dense(self.features)(x)
            return x

    def create_test_train_state(rng_key, input_shape=(4,)):
        net = SimpleNet()
        dummy_input = jnp.zeros(input_shape)
        params = net.init(rng_key, dummy_input)
        tx = optax.adam(1e-3)
        return train_state.TrainState.create(apply_fn=net.apply, params=params, tx=tx)

    try:
        checkpoint_manager, base_run_id = create_checkpoint_manager_for_training(config)
        rng = jax.random.PRNGKey(0)
        ts0 = create_test_train_state(rng)

        # Save final checkpoints for single seed
        save_final_checkpoints(ts0, config, checkpoint_manager, base_run_id)

        # Verify per-seed checkpoints exist
        for seed_idx in range(config['NUM_SEEDS']):
            run_id = base_run_id
            cps = get_agent_checkpoints(
                algorithm=config['ALGORITHM'],
                run_id=run_id,
                agent_id=config['AGENT_ID'],
                base_dir=config['CHECKPOINT_BASE_DIR']
            )
            assert len(cps) > 0, f"No checkpoints for seed {seed_idx}"
        print("✅ Lightweight training integration passed!")
    finally:
        shutil.rmtree(config['CHECKPOINT_BASE_DIR'], ignore_errors=True)


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
