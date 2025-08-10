#!/usr/bin/env python3
"""Integration tests for the full tournament evaluation script."""

import csv
import sys
from pathlib import Path

import jax
import pytest

# Add baselines to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.run_tournament import TournamentEvaluator  # noqa: E402


def test_full_tournament_run_and_csv_output(tmp_path):
    """
    Tests a full tournament run, checking for correct CSV and summary output.

    This is an integration test that validates the main functionality of
    the TournamentEvaluator, ensuring that it can run a small tournament
    and produce verifiable output files.
    """
    run_output_dir = tmp_path / "tournament_results"

    # 1. Initialize and configure the evaluator for a small, deterministic run
    evaluator = TournamentEvaluator(
        env_name="MPE_simple_sumo_v3",
        episodes_per_matchup=2,  # 1 episode per side is enough for this test
        output_dir=str(run_output_dir),
        max_episode_steps=100,
    )
    evaluator.env_kwargs = {"random_spawn": False}  # Ensure deterministic results

    # 2. Setup and run the tournament
    selected_players = ["scripted_seek", "scripted_guardian"]
    evaluator.setup_tournament(selected_players=selected_players, latest_only=True)

    rng_key = jax.random.PRNGKey(42)
    evaluator.run_tournament(rng_key)

    # 3. Verify that the output files were created
    # The evaluator creates a timestamped sub-directory, so we find it first.
    run_dirs = list(run_output_dir.iterdir())
    assert len(run_dirs) == 1, "Expected one timestamped run directory"
    output_dir = run_dirs[0]

    csv_file = output_dir / "tournament_results.csv"
    summary_file = output_dir / "tournament_summary.txt"

    assert csv_file.exists(), "tournament_results.csv was not created"
    assert summary_file.exists(), "tournament_summary.txt was not created"

    # 4. Validate the content of the CSV file
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        results = list(reader)

    # A vs B and B vs A, with 1 episode per side = 2 total episodes per matchup
    # Matchup: seek vs guardian. Total episodes = 2.
    assert len(results) == 2, "CSV should contain results for 2 episodes"

    # Check the deterministic outcomes
    # Side 1: seek (green) vs guardian (red) -> seek wins
    side1 = results[0]
    assert side1["player1"] == "scripted_seek"
    assert side1["player2"] == "scripted_guardian"
    assert side1["winner"] == "scripted_seek"

    # Side 2: guardian (green) vs seek (red) -> seek wins
    side2 = results[1]
    assert side2["player1"] == "scripted_guardian"
    assert side2["player2"] == "scripted_seek"
    assert side2["winner"] == "scripted_seek"
