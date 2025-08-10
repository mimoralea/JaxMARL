#!/usr/bin/env python3
"""
Minimal Tournament Test

This test validates the core tournament functionality without relying on
potentially broken imports or complex setup. It focuses on testing the
essential components we need to trust for experimental results.
"""

import sys
import tempfile
from pathlib import Path

import jax
import pytest

# Add baselines to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Test basic environment and scripted behavior functionality
def test_environment_basic_functionality():
    """Test that the environment works correctly."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
        
        # Initialize environment
        env = SimpleSumoMPE(random_spawn=False)
        env = LogWrapper(env)
        
        # Test environment reset
        rng_key = jax.random.PRNGKey(42)
        obs, state = env.reset(rng_key)
        
        assert obs is not None
        assert state is not None
        assert len(env.agents) == 2
        
        # Test environment step
        actions = {agent: 0 for agent in env.agents}  # NOOP actions
        rng_key, step_key = jax.random.split(rng_key)
        obs, state, rewards, dones, infos = env.step(step_key, state, actions)
        
        assert obs is not None
        assert rewards is not None
        assert dones is not None
        
        print("✓ Environment basic functionality test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        return False


def test_scripted_behaviors_basic():
    """Test that scripted behaviors work correctly."""
    try:
        from baselines.scripted_behaviors import get_scripted_action, list_scripted_behaviors
        
        # Get available behaviors
        behaviors = list_scripted_behaviors()
        assert len(behaviors) > 0, "No scripted behaviors found"
        
        print(f"Found {len(behaviors)} scripted behaviors: {behaviors}")
        
        # Test each behavior can generate actions
        rng_key = jax.random.PRNGKey(42)
        dummy_obs = jax.numpy.zeros(4)  # Simple dummy observation
        
        for behavior in behaviors[:3]:  # Test first 3 to save time
            try:
                rng_key, action_key = jax.random.split(rng_key)
                action, _ = get_scripted_action(dummy_obs, behavior, action_key)
                assert action is not None
                print(f"  ✓ {behavior} behavior works")
            except Exception as e:
                print(f"  ✗ {behavior} behavior failed: {e}")
                return False
        
        print("✓ Scripted behaviors basic test passed")
        return True
        
    except Exception as e:
        print(f"✗ Scripted behaviors test failed: {e}")
        return False


def test_episode_execution():
    """Test that we can run a complete episode with scripted players."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
        from baselines.scripted_behaviors import get_scripted_action
        
        # Initialize environment
        env = SimpleSumoMPE(random_spawn=False)
        env = LogWrapper(env)
        
        # Reset environment
        rng_key = jax.random.PRNGKey(42)
        obs, state = env.reset(rng_key)
        
        # Run episode with scripted behaviors
        episode_length = 0
        max_steps = 50
        
        while not state.done and episode_length < max_steps:
            actions = {}
            
            # Agent 0: seek behavior
            rng_key, action_key = jax.random.split(rng_key)
            action, _ = get_scripted_action(obs[env.agents[0]], "seek", action_key)
            actions[env.agents[0]] = action
            
            # Agent 1: noop behavior
            rng_key, action_key = jax.random.split(rng_key)
            action, _ = get_scripted_action(obs[env.agents[1]], "noop", action_key)
            actions[env.agents[1]] = action
            
            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = env.step(step_key, state, actions)
            
            episode_length += 1
        
        assert episode_length > 0, "Episode didn't run"
        assert episode_length <= max_steps, "Episode ran too long"
        
        # Check rewards are reasonable
        total_rewards = {agent: 0.0 for agent in env.agents}
        for agent in env.agents:
            if agent in rewards:
                total_rewards[agent] = float(rewards[agent])
        
        print(f"Episode completed in {episode_length} steps")
        print(f"Final rewards: {total_rewards}")
        
        print("✓ Episode execution test passed")
        return True
        
    except Exception as e:
        print(f"✗ Episode execution test failed: {e}")
        return False


def test_determinism():
    """Test that episodes are deterministic with same seed."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
        from baselines.scripted_behaviors import get_scripted_action
        
        def run_episode(seed):
            env = SimpleSumoMPE(random_spawn=False)
            env = LogWrapper(env)
            
            rng_key = jax.random.PRNGKey(seed)
            obs, state = env.reset(rng_key)
            
            episode_rewards = {agent: 0.0 for agent in env.agents}
            episode_length = 0
            max_steps = 20
            
            while not state.done and episode_length < max_steps:
                actions = {}
                
                # Both agents use noop for deterministic behavior
                for agent in env.agents:
                    rng_key, action_key = jax.random.split(rng_key)
                    action, _ = get_scripted_action(obs[agent], "noop", action_key)
                    actions[agent] = action
                
                rng_key, step_key = jax.random.split(rng_key)
                obs, state, rewards, dones, infos = env.step(step_key, state, actions)
                
                for agent in env.agents:
                    if agent in rewards:
                        episode_rewards[agent] += float(rewards[agent])
                
                episode_length += 1
            
            return episode_length, episode_rewards
        
        # Run same episode twice with same seed
        length1, rewards1 = run_episode(42)
        length2, rewards2 = run_episode(42)
        
        assert length1 == length2, f"Different episode lengths: {length1} vs {length2}"
        
        for agent in rewards1:
            assert abs(rewards1[agent] - rewards2[agent]) < 1e-6, \
                f"Different rewards for {agent}: {rewards1[agent]} vs {rewards2[agent]}"
        
        print(f"Determinism test: {length1} steps, rewards {rewards1}")
        print("✓ Determinism test passed")
        return True
        
    except Exception as e:
        print(f"✗ Determinism test failed: {e}")
        return False


def test_different_behaviors_produce_different_results():
    """Test that different behaviors produce different outcomes."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from jaxmarl.wrappers.baselines import MPELogWrapper as LogWrapper
        from baselines.scripted_behaviors import get_scripted_action
        
        def run_episode_with_behaviors(behavior1, behavior2, seed):
            env = SimpleSumoMPE(random_spawn=False)
            env = LogWrapper(env)
            
            rng_key = jax.random.PRNGKey(seed)
            obs, state = env.reset(rng_key)
            
            episode_rewards = {agent: 0.0 for agent in env.agents}
            episode_length = 0
            max_steps = 30
            
            while not state.done and episode_length < max_steps:
                actions = {}
                
                # Agent 0 uses behavior1
                rng_key, action_key = jax.random.split(rng_key)
                action, _ = get_scripted_action(obs[env.agents[0]], behavior1, action_key)
                actions[env.agents[0]] = action
                
                # Agent 1 uses behavior2
                rng_key, action_key = jax.random.split(rng_key)
                action, _ = get_scripted_action(obs[env.agents[1]], behavior2, action_key)
                actions[env.agents[1]] = action
                
                rng_key, step_key = jax.random.split(rng_key)
                obs, state, rewards, dones, infos = env.step(step_key, state, actions)
                
                for agent in env.agents:
                    if agent in rewards:
                        episode_rewards[agent] += float(rewards[agent])
                
                episode_length += 1
            
            return episode_length, episode_rewards
        
        # Test seek vs noop
        length1, rewards1 = run_episode_with_behaviors("seek", "noop", 42)
        
        # Test noop vs seek (flipped)
        length2, rewards2 = run_episode_with_behaviors("noop", "seek", 42)
        
        print(f"seek vs noop: {length1} steps, rewards {rewards1}")
        print(f"noop vs seek: {length2} steps, rewards {rewards2}")
        
        # Results should be different (seek should be more aggressive)
        different_outcomes = (length1 != length2 or 
                            abs(rewards1[env.agents[0]] - rewards2[env.agents[0]]) > 0.1)
        
        assert different_outcomes, "Different behaviors produced identical results"
        
        print("✓ Different behaviors test passed")
        return True
        
    except Exception as e:
        print(f"✗ Different behaviors test failed: {e}")
        return False


def run_all_minimal_tests():
    """Run all minimal validation tests."""
    print("=" * 60)
    print("MINIMAL TOURNAMENT VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_environment_basic_functionality,
        test_scripted_behaviors_basic,
        test_episode_execution,
        test_determinism,
        test_different_behaviors_produce_different_results
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            print(f"\n{'-' * 40}")
            print(f"Running {test.__name__}...")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1
    
    print(f"\n{'=' * 60}")
    print(f"MINIMAL TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    
    if failed == 0:
        print("🎉 All minimal tests passed! Core functionality is validated.")
        return True
    else:
        print("❌ Some minimal tests failed. Core functionality needs attention.")
        return False


if __name__ == "__main__":
    success = run_all_minimal_tests()
    sys.exit(0 if success else 1)
