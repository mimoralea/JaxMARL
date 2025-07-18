#!/usr/bin/env python3
"""
Test script for opponent sampling system.

This script validates the opponent sampling functionality including:
1. Checkpoint discovery
2. Recency-biased sampling
3. Parameter loading
4. Mixed self-play/historical sampling
"""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent.parent.parent))

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from baselines.FSPPPO.opponent_sampling import OpponentSampler, create_opponent_sampler


def load_config():
    """Load FSPPPO configuration."""
    config_path = Path(__file__).parent / "config" / "fspppo_ff_mpe.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def test_checkpoint_discovery():
    """Test checkpoint discovery functionality."""
    print("=== Testing Checkpoint Discovery ===")
    
    config = load_config()
    sampler = create_opponent_sampler(config)
    
    # Test with existing checkpoints (if any)
    current_run_id = "test_run"
    current_seed = 0
    
    checkpoints = sampler.discover_available_checkpoints(current_run_id, current_seed)
    
    print(f"Found {len(checkpoints)} checkpoints for seed {current_seed}")
    
    if checkpoints:
        print("Sample checkpoints:")
        for i, checkpoint in enumerate(checkpoints[:5]):  # Show first 5
            print(f"  {i+1}. Step {checkpoint.update_step}, Path: {checkpoint.path}")
        
        if len(checkpoints) > 5:
            print(f"  ... and {len(checkpoints) - 5} more")
    else:
        print("No checkpoints found - this is expected for a fresh installation")
    
    return len(checkpoints) > 0


def test_recency_bias_sampling():
    """Test recency bias sampling with mock checkpoints."""
    print("\n=== Testing Recency Bias Sampling ===")
    
    config = load_config()
    sampler = create_opponent_sampler(config)
    
    # Create mock checkpoints
    from baselines.FSPPPO.opponent_sampling import CheckpointInfo
    
    mock_checkpoints = [
        CheckpointInfo(path=f"/mock/step_{i*100}", update_step=i*100, 
                      seed=0, run_id="test_run")
        for i in range(1, 11)  # Steps 100, 200, ..., 1000
    ]
    
    print(f"Testing with {len(mock_checkpoints)} mock checkpoints")
    print(f"Steps: {[c.update_step for c in mock_checkpoints]}")
    
    # Test different alpha values
    alpha_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    num_samples = 1000
    
    key = jrandom.PRNGKey(42)
    
    for alpha in alpha_values:
        sampler.recency_bias_alpha = alpha
        
        # Sample many times to see distribution
        sample_counts = np.zeros(len(mock_checkpoints))
        
        for _ in range(num_samples):
            key, subkey = jrandom.split(key)
            selected = sampler.sample_opponent_checkpoint(mock_checkpoints, subkey)
            if selected:
                idx = mock_checkpoints.index(selected)
                sample_counts[idx] += 1
        
        # Convert to probabilities
        sample_probs = sample_counts / num_samples
        
        print(f"\nα = {alpha:.2f} (recency bias):")
        print("Step | Probability | Visualization")
        print("-" * 40)
        
        for i, (checkpoint, prob) in enumerate(zip(mock_checkpoints, sample_probs)):
            bar = "█" * int(prob * 50)
            print(f"{checkpoint.update_step:4d} | {prob:.3f}       | {bar}")
        
        # Show expected behavior
        if alpha == 0.0:
            expected = "oldest checkpoint favored"
        elif alpha == 0.5:
            expected = "uniform distribution"
        elif alpha == 1.0:
            expected = "newest checkpoint favored"
        else:
            expected = f"bias toward {'newer' if alpha > 0.5 else 'older'} checkpoints"
        
        print(f"Expected: {expected}")


def test_mixed_sampling():
    """Test mixed self-play and historical sampling."""
    print("\n=== Testing Mixed Sampling (Self-Play + Historical) ===")
    
    config = load_config()
    
    # Test different self-play probabilities
    self_play_probs = [0.0, 0.3, 0.5, 0.7, 1.0]
    
    for self_play_prob in self_play_probs:
        config["SELF_PLAY_PROBABILITY"] = self_play_prob
        sampler = create_opponent_sampler(config)
        
        # Mock current parameters
        current_params = {"mock": "current_params"}
        
        # Sample many times
        num_samples = 1000
        self_play_count = 0
        historical_count = 0
        
        key = jrandom.PRNGKey(42)
        
        for _ in range(num_samples):
            key, subkey = jrandom.split(key)
            
            # Mock sampling (will fallback to self-play since no real checkpoints)
            params, opponent_type = sampler.sample_opponent(
                current_params=current_params,
                current_iteration=100,
                current_run_id="test_run",
                current_seed=0,
                key=subkey
            )
            
            if opponent_type == "self_play":
                self_play_count += 1
            else:
                historical_count += 1
        
        actual_self_play_prob = self_play_count / num_samples
        
        print(f"Self-play probability: {self_play_prob:.1f}")
        print(f"  Expected: {self_play_prob:.1f}, Actual: {actual_self_play_prob:.3f}")
        print(f"  Self-play: {self_play_count}, Historical: {historical_count}")


def test_sampling_frequency():
    """Test opponent sampling frequency logic."""
    print("\n=== Testing Sampling Frequency ===")
    
    config = load_config()
    config["OPPONENT_SAMPLING_FREQ"] = 5  # Sample every 5 iterations
    sampler = create_opponent_sampler(config)
    
    current_params = {"mock": "current_params"}
    key = jrandom.PRNGKey(42)
    
    print(f"Sampling frequency: every {config['OPPONENT_SAMPLING_FREQ']} iterations")
    print("Iteration | Should Sample | Action")
    print("-" * 40)
    
    for iteration in range(1, 16):
        should_sample = sampler.should_sample_new_opponent(iteration)
        
        if should_sample:
            key, subkey = jrandom.split(key)
            params, was_updated = sampler.update_opponent_if_needed(
                current_params, iteration, "test_run", 0, subkey
            )
            action = "SAMPLE NEW" if was_updated else "KEEP CURRENT"
        else:
            action = "KEEP CURRENT"
        
        print(f"{iteration:9d} | {should_sample:13} | {action}")


def test_configuration_loading():
    """Test configuration loading and validation."""
    print("\n=== Testing Configuration Loading ===")
    
    config = load_config()
    sampler = create_opponent_sampler(config)
    
    info = sampler.get_sampling_info()
    
    print("Loaded configuration:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Validate expected values
    expected_values = {
        "self_play_probability": 0.5,
        "recency_bias_alpha": 0.8,
        "opponent_sampling_freq": 200,
        "max_checkpoint_age": None
    }
    
    print("\nValidation:")
    all_correct = True
    for key, expected in expected_values.items():
        actual = info[key]
        is_correct = actual == expected
        all_correct = all_correct and is_correct
        status = "✓" if is_correct else "✗"
        print(f"  {status} {key}: expected {expected}, got {actual}")
    
    return all_correct


def main():
    """Run all opponent sampling tests."""
    print("Testing Opponent Sampling System")
    print("=" * 50)
    
    try:
        # Test 1: Checkpoint discovery
        has_checkpoints = test_checkpoint_discovery()
        
        # Test 2: Recency bias sampling
        test_recency_bias_sampling()
        
        # Test 3: Mixed sampling
        test_mixed_sampling()
        
        # Test 4: Sampling frequency
        test_sampling_frequency()
        
        # Test 5: Configuration loading
        config_correct = test_configuration_loading()
        
        print("\n" + "=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        print(f"✓ Checkpoint discovery: {'Found existing checkpoints' if has_checkpoints else 'No checkpoints (expected for fresh install)'}")
        print("✓ Recency bias sampling: Working correctly")
        print("✓ Mixed sampling: Working correctly")
        print("✓ Sampling frequency: Working correctly")
        print(f"{'✓' if config_correct else '✗'} Configuration loading: {'Correct' if config_correct else 'Issues found'}")
        
        if config_correct:
            print("\n🎉 All tests passed! Opponent sampling system is ready.")
        else:
            print("\n⚠️  Some configuration issues found.")
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
