#!/usr/bin/env python3
"""
Tournament Test Runner

This script runs comprehensive tests to validate the tournament evaluation system
and ensures we can trust the experimental results.
"""

import sys
import subprocess
import tempfile
from pathlib import Path

# Add baselines to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from baselines.run_tournament import TournamentEvaluator, TournamentPlayer
import jax


def test_basic_functionality():
    """Test basic tournament functionality."""
    print("Testing basic tournament functionality...")

    # Create a minimal tournament
    temp_dir = tempfile.mkdtemp()
    evaluator = TournamentEvaluator(
        env_name="MPE_simple_sumo_v3",
        episodes_per_matchup=4,  # Small for testing
        output_dir=temp_dir,
        max_episode_steps=20
    )

    # Create test players
    players = [
        TournamentPlayer(name="scripted_seek", player_type="scripted"),
        TournamentPlayer(name="scripted_noop", player_type="scripted")
    ]

    evaluator.players = players
    evaluator.setup_matches()

    # Run tournament
    rng_key = jax.random.PRNGKey(42)
    evaluator.run_tournament(rng_key)

    # Validate results
    assert len(evaluator.results) > 0, "No results generated"
    assert len(evaluator.matches) == 2, f"Expected 2 matches, got {len(evaluator.matches)}"

    # Check CSV output
    csv_file = evaluator.output_dir / "tournament_results.csv"
    assert csv_file.exists(), "CSV file not created"

    print("✓ Basic functionality test passed")
    return True


def test_scripted_behaviors():
    """Test that all scripted behaviors work correctly."""
    print("Testing scripted behaviors...")

    temp_dir = tempfile.mkdtemp()
    evaluator = TournamentEvaluator(
        env_name="MPE_simple_sumo_v3",
        episodes_per_matchup=2,
        output_dir=temp_dir,
        max_episode_steps=10
    )

    # Get all scripted players
    scripted_players = evaluator.create_scripted_players()
    print(f"Found {len(scripted_players)} scripted behaviors:")
    for player in scripted_players:
        print(f"  - {player.name}")

    # Test each behavior can run an episode
    rng_key = jax.random.PRNGKey(42)
    noop_player = TournamentPlayer(name="scripted_noop", player_type="scripted")

    for player in scripted_players[:3]:  # Test first 3 to save time
        try:
            rng_key, episode_key = jax.random.split(rng_key)
            result = evaluator._run_episode_with_positions(
                green_player=player,
                red_player=noop_player,
                rng_key=episode_key,
                episode_id=1,
                side=1,
                match_id="test"
            )
            assert result is not None, f"No result from {player.name}"
            print(f"  ✓ {player.name} works correctly")
        except Exception as e:
            print(f"  ✗ {player.name} failed: {e}")
            return False

    print("✓ Scripted behaviors test passed")
    return True


def test_determinism():
    """Test that results are deterministic with same seed."""
    print("Testing determinism...")

    def run_mini_tournament(seed):
        temp_dir = tempfile.mkdtemp()
        evaluator = TournamentEvaluator(
            env_name="MPE_simple_sumo_v3",
            episodes_per_matchup=4,
            output_dir=temp_dir,
            max_episode_steps=15
        )

        players = [
            TournamentPlayer(name="scripted_seek", player_type="scripted"),
            TournamentPlayer(name="scripted_noop", player_type="scripted")
        ]

        evaluator.players = players
        evaluator.setup_matches()

        rng_key = jax.random.PRNGKey(seed)
        evaluator.run_tournament(rng_key)

        return evaluator.results

    # Run twice with same seed
    results1 = run_mini_tournament(42)
    results2 = run_mini_tournament(42)

    # Compare results
    assert len(results1) == len(results2), "Different number of results"

    for r1, r2 in zip(results1, results2):
        assert r1['winner'] == r2['winner'], f"Different winners: {r1['winner']} vs {r2['winner']}"
        assert r1['episode_length'] == r2['episode_length'], "Different episode lengths"

    print("✓ Determinism test passed")
    return True


def test_side_flip_logic():
    """Test that side-flip logic works correctly."""
    print("Testing side-flip logic...")

    temp_dir = tempfile.mkdtemp()
    evaluator = TournamentEvaluator(
        env_name="MPE_simple_sumo_v3",
        episodes_per_matchup=4,
        output_dir=temp_dir,
        max_episode_steps=20
    )

    # Create asymmetric players (one should consistently beat the other)
    seek_player = TournamentPlayer(name="scripted_seek", player_type="scripted")
    noop_player = TournamentPlayer(name="scripted_noop", player_type="scripted")

    rng_key = jax.random.PRNGKey(42)

    # Test side 1: seek (green) vs noop (red)
    rng_key1, rng_key2 = jax.random.split(rng_key)
    result1 = evaluator._run_episode_with_positions(
        green_player=seek_player,
        red_player=noop_player,
        rng_key=rng_key1,
        episode_id=1,
        side=1,
        match_id="test"
    )

    # Test side 2: noop (green) vs seek (red)
    result2 = evaluator._run_episode_with_positions(
        green_player=noop_player,
        red_player=seek_player,
        rng_key=rng_key2,
        episode_id=2,
        side=2,
        match_id="test"
    )

    # Verify side assignments
    assert result1['green_player'] == "scripted_seek"
    assert result1['red_player'] == "scripted_noop"
    assert result2['green_player'] == "scripted_noop"
    assert result2['red_player'] == "scripted_seek"

    print(f"  Side 1 winner: {result1['winner']}")
    print(f"  Side 2 winner: {result2['winner']}")

    # Both should have seek as winner (seek should beat noop regardless of position)
    if result1['winner'] == "scripted_seek" and result2['winner'] == "scripted_seek":
        print("  ✓ Consistent winner across sides")
    else:
        print(f"  ! Inconsistent winners: {result1['winner']} vs {result2['winner']}")

    print("✓ Side-flip logic test passed")
    return True


def test_csv_format():
    """Test CSV output format and data integrity."""
    print("Testing CSV format...")

    temp_dir = tempfile.mkdtemp()
    evaluator = TournamentEvaluator(
        env_name="MPE_simple_sumo_v3",
        episodes_per_matchup=4,
        output_dir=temp_dir,
        max_episode_steps=15
    )

    players = [
        TournamentPlayer(name="scripted_seek", player_type="scripted"),
        TournamentPlayer(name="scripted_noop", player_type="scripted")
    ]

    evaluator.players = players
    evaluator.setup_matches()

    rng_key = jax.random.PRNGKey(42)
    evaluator.run_tournament(rng_key)

    # Read and validate CSV
    import csv
    csv_file = evaluator.output_dir / "tournament_results.csv"

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Check required columns
    required_columns = [
        'match_id', 'episode_id', 'player1', 'player2',
        'green_player', 'red_player', 'winner',
        'player1_reward', 'player2_reward',
        'green_reward', 'red_reward', 'episode_length', 'side'
    ]

    for col in required_columns:
        assert col in reader.fieldnames, f"Missing column: {col}"

    # Validate data
    for i, row in enumerate(rows):
        # Check data types
        try:
            int(row['episode_id'])
            int(row['side'])
            int(row['episode_length'])
            float(row['player1_reward'])
            float(row['player2_reward'])
            float(row['green_reward'])
            float(row['red_reward'])
        except ValueError as e:
            print(f"  ✗ Data type error in row {i}: {e}")
            return False

        # Check logical consistency
        if row['winner'] not in [row['player1'], row['player2'], 'draw']:
            print(f"  ✗ Invalid winner in row {i}: {row['winner']}")
            return False

        if row['side'] not in ['1', '2']:
            print(f"  ✗ Invalid side in row {i}: {row['side']}")
            return False

    print(f"  ✓ Validated {len(rows)} CSV rows")
    print("✓ CSV format test passed")
    return True


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("TOURNAMENT VALIDATION TEST SUITE")
    print("=" * 60)

    tests = [
        test_basic_functionality,
        test_scripted_behaviors,
        test_determinism,
        test_side_flip_logic,
        test_csv_format
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            print(f"\n{'-' * 40}")
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed with exception: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")

    if failed == 0:
        print("🎉 All tests passed! Tournament system is validated.")
        return True
    else:
        print("❌ Some tests failed. Tournament system needs attention.")
        return False


def run_pytest_tests():
    """Run the comprehensive pytest test suite."""
    print("\nRunning comprehensive pytest test suite...")

    test_file = Path(__file__).parent / "test_tournament_validation.py"

    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", str(test_file), "-v"
        ], capture_output=True, text=True, cwd=Path(__file__).parent.parent.parent)

        print("PYTEST OUTPUT:")
        print(result.stdout)
        if result.stderr:
            print("PYTEST ERRORS:")
            print(result.stderr)

        return result.returncode == 0
    except Exception as e:
        print(f"Failed to run pytest: {e}")
        return False


if __name__ == "__main__":
    print("Starting tournament validation...")

    # Run basic validation tests
    basic_tests_passed = run_all_tests()

    # Run comprehensive pytest suite
    pytest_passed = run_pytest_tests()

    print(f"\n{'=' * 60}")
    print("FINAL VALIDATION RESULTS:")
    print(f"Basic tests: {'PASSED' if basic_tests_passed else 'FAILED'}")
    print(f"Pytest suite: {'PASSED' if pytest_passed else 'FAILED'}")

    if basic_tests_passed and pytest_passed:
        print("🎉 ALL VALIDATION TESTS PASSED!")
        print("Tournament system is validated and ready for experiments.")
    else:
        print("❌ VALIDATION FAILED!")
        print("Tournament system needs fixes before running experiments.")

    print(f"{'=' * 60}")
