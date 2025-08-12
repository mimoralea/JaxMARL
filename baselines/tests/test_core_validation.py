#!/usr/bin/env python3
"""
Core Validation Tests

This test suite focuses on validating the essential components needed for
tournament evaluation, working around known issues with the current codebase.
"""

import sys
import tempfile
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

# Add baselines to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def test_jax_environment_basic():
    """Test basic JAX environment functionality."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE

        # Initialize environment without wrapper first
        env = SimpleSumoMPE(random_spawn=False)

        # Test environment reset
        rng_key = jax.random.PRNGKey(42)
        obs, state = env.reset(rng_key)

        assert obs is not None
        assert state is not None
        assert len(env.agents) == 2
        print(f"Environment agents: {env.agents}")
        print(f"Observation space: {env.observation_space(env.agents[0])}")
        print(f"Action space: {env.action_space(env.agents[0])}")

        # Test environment step with simple actions
        actions = {agent: 0 for agent in env.agents}  # NOOP actions
        rng_key, step_key = jax.random.split(rng_key)
        obs, new_state, rewards, dones, infos = env.step(step_key, state, actions)

        assert obs is not None
        assert rewards is not None
        assert dones is not None

        # Check state progression
        assert new_state.step == state.step + 1

        print("✓ Basic JAX environment test passed")
        return True

    except Exception as e:
        print(f"✗ JAX environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripted_behaviors_import():
    """Test that scripted behaviors can be imported and listed."""
    try:
        from baselines.scripted_behaviors import list_scripted_behaviors

        # Get available behaviors
        behaviors = list_scripted_behaviors()
        assert isinstance(behaviors, dict), f"Expected dict, got {type(behaviors)}"
        assert len(behaviors) > 0, "No scripted behaviors found"

        print(f"Found {len(behaviors)} scripted behaviors:")
        for name, description in behaviors.items():
            print(f"  - {name}: {description}")

        print("✓ Scripted behaviors import test passed")
        return True

    except Exception as e:
        print(f"✗ Scripted behaviors import test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_scripted_action_generation():
    """Test scripted action generation with proper observation format."""
    try:
        from baselines.scripted_behaviors import get_scripted_action, list_scripted_behaviors
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE

        # Initialize environment to get proper observation format
        env = SimpleSumoMPE(random_spawn=False)
        rng_key = jax.random.PRNGKey(42)
        obs, state = env.reset(rng_key)

        # Get a real observation from the environment
        agent = env.agents[0]
        real_obs = obs[agent]

        print(f"Real observation shape: {real_obs.shape}")
        print(f"Real observation type: {type(real_obs)}")

        # Test with available behaviors
        behaviors = list_scripted_behaviors()

        for behavior_name in list(behaviors.keys())[:3]:  # Test first 3
            try:
                rng_key, action_key = jax.random.split(rng_key)
                action = get_scripted_action(real_obs, behavior_name, action_key)

                # Validate action
                assert action is not None
                assert isinstance(action, (int, jnp.ndarray))

                # Check action is in valid range
                action_space_size = env.action_space(agent).n
                if isinstance(action, jnp.ndarray):
                    action_val = int(action)
                else:
                    action_val = action

                assert 0 <= action_val < action_space_size, \
                    f"Action {action_val} out of range [0, {action_space_size})"

                print(f"  ✓ {behavior_name}: action={action_val}")

            except Exception as e:
                print(f"  ✗ {behavior_name} failed: {e}")
                return False

        print("✓ Scripted action generation test passed")
        return True

    except Exception as e:
        print(f"✗ Scripted action generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_episode_with_raw_environment():
    """Test running an episode with the raw environment (no wrapper)."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from baselines.scripted_behaviors import get_scripted_action

        # Initialize environment without wrapper
        env = SimpleSumoMPE(random_spawn=False)

        # Reset environment
        rng_key = jax.random.PRNGKey(42)
        obs, state = env.reset(rng_key)

        # Run episode
        episode_length = 0
        max_steps = 20
        episode_rewards = {agent: 0.0 for agent in env.agents}

        while episode_length < max_steps:
            actions = {}

            # Generate actions for each agent
            for i, agent in enumerate(env.agents):
                behavior = "seek" if i == 0 else "noop"
                rng_key, action_key = jax.random.split(rng_key)
                action = get_scripted_action(obs[agent], behavior, action_key)
                actions[agent] = action

            # Step environment
            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = env.step(step_key, state, actions)

            # Accumulate rewards
            for agent in env.agents:
                if agent in rewards:
                    episode_rewards[agent] += float(rewards[agent])

            episode_length += 1

            # Check if episode is done
            if dones.get("__all__", False):
                break

        print(f"Episode completed in {episode_length} steps")
        print(f"Final rewards: {episode_rewards}")

        # Basic validation
        assert episode_length > 0, "Episode didn't run"
        assert all(isinstance(r, (int, float)) for r in episode_rewards.values()), \
            "Invalid reward types"

        print("✓ Raw environment episode test passed")
        return True

    except Exception as e:
        print(f"✗ Raw environment episode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_deterministic_behavior():
    """Test that episodes are deterministic with same seed."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from baselines.scripted_behaviors import get_scripted_action

        def run_short_episode(seed):
            env = SimpleSumoMPE(random_spawn=False)
            rng_key = jax.random.PRNGKey(seed)
            obs, state = env.reset(rng_key)

            # Run just a few steps for determinism test
            actions_taken = []
            rewards_received = []

            for step in range(5):
                actions = {}

                # Use deterministic behavior (noop)
                for agent in env.agents:
                    rng_key, action_key = jax.random.split(rng_key)
                    action = get_scripted_action(obs[agent], "noop", action_key)
                    actions[agent] = action

                actions_taken.append(actions.copy())

                rng_key, step_key = jax.random.split(rng_key)
                obs, state, rewards, dones, infos = env.step(step_key, state, actions)

                rewards_received.append({k: float(v) for k, v in rewards.items()})

                if dones.get("__all__", False):
                    break

            return actions_taken, rewards_received

        # Run same episode twice
        actions1, rewards1 = run_short_episode(42)
        actions2, rewards2 = run_short_episode(42)

        # Compare results
        assert len(actions1) == len(actions2), "Different number of steps"

        for i, (a1, a2) in enumerate(zip(actions1, actions2)):
            for agent in a1:
                assert a1[agent] == a2[agent], \
                    f"Different actions at step {i} for {agent}: {a1[agent]} vs {a2[agent]}"

        for i, (r1, r2) in enumerate(zip(rewards1, rewards2)):
            for agent in r1:
                assert abs(r1[agent] - r2[agent]) < 1e-6, \
                    f"Different rewards at step {i} for {agent}: {r1[agent]} vs {r2[agent]}"

        print(f"Determinism test: {len(actions1)} steps, consistent results")
        print("✓ Deterministic behavior test passed")
        return True

    except Exception as e:
        print(f"✗ Deterministic behavior test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reward_consistency():
    """Test that rewards are consistent and reasonable."""
    try:
        import jaxmarl
        from jaxmarl.environments.mpe.simple_sumo import SimpleSumoMPE
        from baselines.scripted_behaviors import get_scripted_action

        env = SimpleSumoMPE(random_spawn=False)
        rng_key = jax.random.PRNGKey(42)
        obs, state = env.reset(rng_key)

        # Run episode and collect reward statistics
        all_rewards = {agent: [] for agent in env.agents}

        for step in range(10):
            actions = {}

            # Use different behaviors for each agent
            behaviors = ["seek", "noop"]
            for i, agent in enumerate(env.agents):
                behavior = behaviors[i % len(behaviors)]
                rng_key, action_key = jax.random.split(rng_key)
                action = get_scripted_action(obs[agent], behavior, action_key)
                actions[agent] = action

            rng_key, step_key = jax.random.split(rng_key)
            obs, state, rewards, dones, infos = env.step(step_key, state, actions)

            # Collect rewards
            for agent in env.agents:
                if agent in rewards:
                    reward_val = float(rewards[agent])
                    all_rewards[agent].append(reward_val)

                    # Basic sanity checks
                    assert not jnp.isnan(reward_val), f"NaN reward for {agent}"
                    assert not jnp.isinf(reward_val), f"Infinite reward for {agent}"
                    assert -100 <= reward_val <= 100, f"Extreme reward for {agent}: {reward_val}"

            if dones.get("__all__", False):
                break

        # Analyze reward patterns
        for agent, rewards_list in all_rewards.items():
            if rewards_list:
                avg_reward = sum(rewards_list) / len(rewards_list)
                min_reward = min(rewards_list)
                max_reward = max(rewards_list)

                print(f"{agent}: avg={avg_reward:.3f}, min={min_reward:.3f}, max={max_reward:.3f}")

        print("✓ Reward consistency test passed")
        return True

    except Exception as e:
        print(f"✗ Reward consistency test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_core_tests():
    """Run all core validation tests."""
    print("=" * 60)
    print("CORE VALIDATION TESTS")
    print("=" * 60)

    tests = [
        test_jax_environment_basic,
        test_scripted_behaviors_import,
        test_scripted_action_generation,
        test_episode_with_raw_environment,
        test_deterministic_behavior,
        test_reward_consistency
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
    print(f"CORE TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed == 0:
        print("🎉 All core tests passed! Essential functionality is validated.")
        return True
    else:
        print("❌ Some core tests failed. Essential functionality needs attention.")
        return False


if __name__ == "__main__":
    success = run_all_core_tests()
    sys.exit(0 if success else 1)
