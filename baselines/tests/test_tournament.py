#!/usr/bin/env python3
"""Verify tournament system works correctly."""

import sys
from pathlib import Path

import jax
import pytest

# Add baselines to path for local testing.
# A better solution is to install the project in editable mode (`pip install -e .`)
# but this is a quick fix for the test runner.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.run_tournament import TournamentEvaluator, TournamentPlayer # noqa: E402


@pytest.mark.skip(reason="Depends on a hardcoded, non-existent checkpoint.")
def test_ippo_vs_scripted():
    """Test IPPO checkpoint loading against scripted baseline."""
    # This test remains as a template for future checkpoint-based tests.
    pass


def test_scripted_vs_scripted():
    """Test scripted vs. scripted with a deterministic, non-random env."""
    # 1. Set up the evaluator
    # We don't need to create the real output dir for this unit test.
    evaluator = TournamentEvaluator(
        env_name="MPE_simple_sumo_v3",
        output_dir="/tmp/test_results",
    )
    # Manually set a deterministic environment
    evaluator.env_kwargs = {"random_spawn": False}

    # 2. Create players
    green_player = TournamentPlayer(name="scripted_seek", player_type="scripted")
    red_player = TournamentPlayer(name="scripted_guardian", player_type="scripted")

    # 3. Run a single episode
    rng_key = jax.random.PRNGKey(42)
    result = evaluator._run_episode_with_positions(
        green_player=green_player,
        red_player=red_player,
        rng_key=rng_key,
        episode_id=0,
        side=1,
    )

    # 4. Assert correctness
    assert result is not None
    # The winner is the player name, not the color.
    assert result["winner"] == "scripted_seek"
    # In this deterministic setup, 'seek' (green) wins.
    assert result["green_reward"] > result["red_reward"]
