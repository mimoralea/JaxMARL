"""
Test suite for Orbax-based checkpoint management system.
Tests basic functionality, integration with JAX/Flax, and training integration.
"""

import os
import tempfile
import shutil
import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training import train_state
import optax
from orbax_checkpoint_manager import FSPPPOCheckpointManager
from jax_checkpoint_utils import create_checkpoint_manager_for_training, save_final_checkpoints


class SimpleNetwork(nn.Module):
    """Simple network for testing."""
    features: int = 64
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.features)(x)
        x = nn.relu(x)
        x = nn.Dense(self.features)(x)
        return x


def create_test_train_state(rng_key, input_shape=(4,)):
    """Create a test training state."""
    network = SimpleNetwork()
    
    # Initialize parameters
    dummy_input = jnp.zeros(input_shape)
    params = network.init(rng_key, dummy_input)
    
    # Create optimizer
    tx = optax.adam(learning_rate=1e-3)
    
    # Create training state
    train_state_obj = train_state.TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    
    return train_state_obj, network


def test_basic_checkpoint_save_load():
    """Test basic checkpoint saving and loading with Orbax."""
    print("🧪 Testing basic Orbax checkpoint save/load functionality...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create checkpoint manager
        manager = FSPPPOCheckpointManager(temp_dir, "test_algo")
        
        # Create test training state
        rng = jax.random.PRNGKey(42)
        train_state_obj, network = create_test_train_state(rng)
        
        # Save checkpoint
        run_id = "test_run"
        agent_id = "test_agent"
        update_step = 100
        
        checkpoint_dir = manager.save_checkpoint(
            train_state_obj.params, update_step, run_id, agent_id
        )
        
        print(f"✅ Checkpoint saved to: {checkpoint_dir}")
        
        # Verify directory structure
        expected_dir = os.path.join(temp_dir, "test_algo", run_id, agent_id, f"step_{update_step:06d}")
        assert os.path.exists(expected_dir), f"Checkpoint directory not found: {expected_dir}"
        assert os.path.exists(os.path.join(expected_dir, "metadata.json")), "Metadata file not found"
        
        # Load checkpoint
        abstract_params = jax.eval_shape(lambda: network.init(rng, jnp.zeros((4,))))
        loaded_params = manager.load_checkpoint(checkpoint_dir, abstract_params)
        
        # Verify parameters match
        def params_equal(p1, p2):
            return jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), p1, p2)
            )
        
        assert params_equal(train_state_obj.params, loaded_params), "Loaded parameters don't match saved parameters"
        
        print("✅ Checkpoint loaded successfully and parameters match!")
        
        # Test metadata
        checkpoints = manager.get_agent_checkpoints(run_id, agent_id)
        assert len(checkpoints) == 1, f"Expected 1 checkpoint, found {len(checkpoints)}"
        assert checkpoints[0]["update_step"] == update_step, "Update step mismatch in metadata"
        
        print("✅ Metadata is correct!")


def test_multiple_checkpoints_and_cleanup():
    """Test multiple checkpoints and cleanup functionality."""
    print("\n🧪 Testing multiple checkpoints and cleanup...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = FSPPPOCheckpointManager(temp_dir, "test_algo")
        
        # Create test training state
        rng = jax.random.PRNGKey(42)
        train_state_obj, network = create_test_train_state(rng)
        
        run_id = "test_run"
        agent_id = "test_agent"
        
        # Save multiple checkpoints
        saved_dirs = []
        for step in [100, 200, 300, 400, 500]:
            # Modify parameters slightly for each checkpoint
            modified_params = jax.tree_util.tree_map(
                lambda x: x + jnp.ones_like(x) * 0.01 * step, 
                train_state_obj.params
            )
            
            checkpoint_dir = manager.save_checkpoint(
                modified_params, step, run_id, agent_id
            )
            saved_dirs.append(checkpoint_dir)
        
        print(f"✅ Saved {len(saved_dirs)} checkpoints")
        
        # Verify all checkpoints exist
        checkpoints = manager.get_agent_checkpoints(run_id, agent_id)
        assert len(checkpoints) == 5, f"Expected 5 checkpoints, found {len(checkpoints)}"
        
        # Test cleanup - keep only 3 most recent
        removed_count = manager.cleanup_old_checkpoints(run_id, agent_id, max_checkpoints=3)
        assert removed_count == 2, f"Expected to remove 2 checkpoints, removed {removed_count}"
        
        # Verify only 3 checkpoints remain
        remaining_checkpoints = manager.get_agent_checkpoints(run_id, agent_id)
        assert len(remaining_checkpoints) == 3, f"Expected 3 remaining checkpoints, found {len(remaining_checkpoints)}"
        
        # Verify the correct checkpoints remain (most recent ones)
        remaining_steps = [cp["update_step"] for cp in remaining_checkpoints]
        expected_steps = [300, 400, 500]
        assert remaining_steps == expected_steps, f"Expected steps {expected_steps}, got {remaining_steps}"
        
        print("✅ Cleanup functionality works correctly!")


def test_training_integration():
    """Test integration with training configuration."""
    print("\n🧪 Testing training integration...")
    
    # Create test config
    config = {
        "CHECKPOINT_FREQ": 50,
        "MAX_CHECKPOINTS": 5,
        "CHECKPOINT_BASE_DIR": tempfile.mkdtemp(),
        "ALGORITHM": "fspppo",
        "NUM_SEEDS": 2,
        "NUM_UPDATES": 150,
        "AGENT_ID": "main_agent"
    }
    
    try:
        # Create checkpoint manager
        checkpoint_manager, base_run_id = create_checkpoint_manager_for_training(config)
        
        print(f"✅ Created checkpoint manager with run_id: {base_run_id}")
        
        # Create mock training states for multiple seeds
        rng = jax.random.PRNGKey(42)
        train_state_obj, network = create_test_train_state(rng)
        
        # Simulate multiple seeds by creating slightly different parameters
        mock_train_states = []
        for seed in range(config["NUM_SEEDS"]):
            seed_params = jax.tree_util.tree_map(
                lambda x: x + jnp.ones_like(x) * 0.01 * seed, 
                train_state_obj.params
            )
            seed_train_state = train_state_obj.replace(params=seed_params)
            mock_train_states.append(seed_train_state)
        
        # Stack train states to simulate multi-seed training output
        if config["NUM_SEEDS"] > 1:
            stacked_train_states = jax.tree_util.tree_map(
                lambda *args: jnp.stack(args), *mock_train_states
            )
        else:
            stacked_train_states = mock_train_states[0]
        
        # Test final checkpoint saving
        save_final_checkpoints(stacked_train_states, config, checkpoint_manager, base_run_id)
        
        # Verify checkpoints were saved for each seed
        for seed_idx in range(config["NUM_SEEDS"]):
            if config["NUM_SEEDS"] > 1:
                run_id = f"{base_run_id}_seed{seed_idx}"
            else:
                run_id = base_run_id
            
            checkpoints = checkpoint_manager.get_agent_checkpoints(run_id, config["AGENT_ID"])
            assert len(checkpoints) > 0, f"No checkpoints found for seed {seed_idx}"
            
            # Verify final checkpoint exists
            final_checkpoint = checkpoint_manager.get_checkpoint_for_step(
                run_id, config["AGENT_ID"], config["NUM_UPDATES"]
            )
            assert final_checkpoint is not None, f"Final checkpoint not found for seed {seed_idx}"
            
            print(f"✅ Checkpoints saved correctly for seed {seed_idx}")
        
        print("✅ Training integration test passed!")
        
    finally:
        # Cleanup
        shutil.rmtree(config["CHECKPOINT_BASE_DIR"], ignore_errors=True)


def test_checkpoint_loading_for_opponent():
    """Test loading checkpoints for opponent use."""
    print("\n🧪 Testing checkpoint loading for opponent...")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        manager = FSPPPOCheckpointManager(temp_dir, "test_algo")
        
        # Create test training state
        rng = jax.random.PRNGKey(42)
        train_state_obj, network = create_test_train_state(rng)
        
        # Save a checkpoint
        run_id = "test_run"
        agent_id = "main_agent"
        update_step = 100
        
        checkpoint_dir = manager.save_checkpoint(
            train_state_obj.params, update_step, run_id, agent_id
        )
        
        # Test loading for opponent
        from jax_checkpoint_utils import load_checkpoint_for_opponent, create_abstract_train_state
        
        # Create abstract train state (simulating what we'd have in opponent sampling)
        mock_config = {"ANNEAL_LR": False, "LR": 1e-3, "MAX_GRAD_NORM": 0.5}
        
        # Create a mock environment-like object
        class MockEnv:
            def __init__(self):
                self.agents = ["agent_0"]
            
            def observation_space(self, agent):
                class MockSpace:
                    def __init__(self):
                        self.shape = (4,)
                return MockSpace()
        
        mock_env = MockEnv()
        abstract_train_state = create_abstract_train_state(mock_config, mock_env, network)
        
        # Load checkpoint for opponent
        opponent_params = load_checkpoint_for_opponent(checkpoint_dir, abstract_train_state)
        
        # Verify parameters match
        def params_equal(p1, p2):
            return jax.tree_util.tree_all(
                jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y), p1, p2)
            )
        
        assert params_equal(train_state_obj.params, opponent_params), "Opponent parameters don't match"
        
        print("✅ Checkpoint loading for opponent works correctly!")


def main():
    """Run all tests."""
    print("🎯 Starting Orbax Checkpoint Management System Tests\n")
    
    try:
        test_basic_checkpoint_save_load()
        test_multiple_checkpoints_and_cleanup()
        test_training_integration()
        test_checkpoint_loading_for_opponent()
        
        print("\n🎉 All tests passed! Orbax checkpoint system is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
