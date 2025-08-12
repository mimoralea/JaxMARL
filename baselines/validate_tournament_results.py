#!/usr/bin/env python3
"""
Tournament Results Validation Utility

This script validates tournament results for consistency, accuracy, and reliability.
It performs cross-checks to ensure experimental data can be trusted.
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any

import pandas as pd
import numpy as np


class TournamentResultValidator:
    """Validates tournament results for consistency and accuracy."""

    def __init__(self, results_file: str):
        """Initialize validator with results CSV file."""
        self.results_file = Path(results_file)
        self.results = []
        self.validation_errors = []
        self.validation_warnings = []

        if not self.results_file.exists():
            raise FileNotFoundError(f"Results file not found: {results_file}")

        self.load_results()

    def load_results(self):
        """Load results from CSV file."""
        try:
            with open(self.results_file, 'r') as f:
                reader = csv.DictReader(f)
                self.results = list(reader)
            print(f"Loaded {len(self.results)} results from {self.results_file}")
        except Exception as e:
            raise ValueError(f"Failed to load results: {e}")

    def validate_data_integrity(self):
        """Validate basic data integrity."""
        print("\n" + "="*50)
        print("VALIDATING DATA INTEGRITY")
        print("="*50)

        errors = []

        # Check required columns
        if not self.results:
            errors.append("No results found")
            return errors

        required_columns = [
            'match_id', 'episode_id', 'player1', 'player2',
            'green_player', 'red_player', 'winner',
            'player1_reward', 'player2_reward',
            'green_reward', 'red_reward', 'episode_length', 'side'
        ]

        first_row = self.results[0]
        missing_columns = [col for col in required_columns if col not in first_row]
        if missing_columns:
            errors.append(f"Missing columns: {missing_columns}")

        # Validate each row
        for i, row in enumerate(self.results):
            try:
                # Check data types
                int(row['episode_id'])
                int(row['side'])
                int(row['episode_length'])
                float(row['player1_reward'])
                float(row['player2_reward'])
                float(row['green_reward'])
                float(row['red_reward'])

                # Check logical consistency
                if row['winner'] not in [row['player1'], row['player2'], 'draw']:
                    errors.append(f"Row {i}: Invalid winner '{row['winner']}'")

                if row['side'] not in ['1', '2']:
                    errors.append(f"Row {i}: Invalid side '{row['side']}'")

                if int(row['episode_length']) <= 0:
                    errors.append(f"Row {i}: Invalid episode length {row['episode_length']}")

            except ValueError as e:
                errors.append(f"Row {i}: Data type error - {e}")

        if errors:
            self.validation_errors.extend(errors)
            print(f"❌ Found {len(errors)} data integrity errors")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("✅ Data integrity validation passed")

        return errors

    def validate_match_completeness(self):
        """Validate that all matches are complete."""
        print("\n" + "="*50)
        print("VALIDATING MATCH COMPLETENESS")
        print("="*50)

        errors = []

        # Group results by match
        match_groups = defaultdict(list)
        for result in self.results:
            match_groups[result['match_id']].append(result)

        print(f"Found {len(match_groups)} unique matches")

        # Check each match
        expected_episodes = None
        episode_counts = []

        for match_id, match_results in match_groups.items():
            episode_count = len(match_results)
            episode_counts.append(episode_count)

            if expected_episodes is None:
                expected_episodes = episode_count
            elif episode_count != expected_episodes:
                errors.append(f"Match {match_id}: {episode_count} episodes, expected {expected_episodes}")

            # Check side distribution
            side_counts = Counter(r['side'] for r in match_results)
            if len(side_counts) != 2 or side_counts['1'] != side_counts['2']:
                errors.append(f"Match {match_id}: Uneven side distribution {dict(side_counts)}")

        if errors:
            self.validation_errors.extend(errors)
            print(f"❌ Found {len(errors)} match completeness errors")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"✅ All matches complete with {expected_episodes} episodes each")
            print(f"✅ Episodes per side: {expected_episodes // 2}")

        return errors

    def validate_side_flip_consistency(self):
        """Validate side-flip logic consistency."""
        print("\n" + "="*50)
        print("VALIDATING SIDE-FLIP CONSISTENCY")
        print("="*50)

        errors = []
        warnings = []

        # Group by match
        match_groups = defaultdict(list)
        for result in self.results:
            match_groups[result['match_id']].append(result)

        for match_id, match_results in match_groups.items():
            # Separate by side
            side1_results = [r for r in match_results if r['side'] == '1']
            side2_results = [r for r in match_results if r['side'] == '2']

            if not side1_results or not side2_results:
                errors.append(f"Match {match_id}: Missing results for one side")
                continue

            # Check player assignments
            side1_green = side1_results[0]['green_player']
            side1_red = side1_results[0]['red_player']
            side2_green = side2_results[0]['green_player']
            side2_red = side2_results[0]['red_player']

            # In side 2, players should be flipped
            if side1_green != side2_red or side1_red != side2_green:
                errors.append(f"Match {match_id}: Incorrect side-flip assignments")

            # Check consistency within each side
            for results, side_name in [(side1_results, "side 1"), (side2_results, "side 2")]:
                green_players = set(r['green_player'] for r in results)
                red_players = set(r['red_player'] for r in results)

                if len(green_players) > 1 or len(red_players) > 1:
                    errors.append(f"Match {match_id} {side_name}: Inconsistent player assignments")

        if errors:
            self.validation_errors.extend(errors)
            print(f"❌ Found {len(errors)} side-flip consistency errors")
            for error in errors:
                print(f"  - {error}")
        else:
            print("✅ Side-flip consistency validation passed")

        return errors

    def validate_reward_consistency(self):
        """Validate reward consistency with winner determination."""
        print("\n" + "="*50)
        print("VALIDATING REWARD CONSISTENCY")
        print("="*50)

        errors = []
        warnings = []

        for i, result in enumerate(self.results):
            green_reward = float(result['green_reward'])
            red_reward = float(result['red_reward'])
            winner = result['winner']
            green_player = result['green_player']
            red_player = result['red_player']

            # Check reward-winner consistency
            if winner == green_player and green_reward < red_reward:
                errors.append(f"Episode {i}: Green player won but has lower reward ({green_reward} < {red_reward})")
            elif winner == red_player and red_reward < green_reward:
                errors.append(f"Episode {i}: Red player won but has lower reward ({red_reward} < {green_reward})")
            elif winner == 'draw' and green_reward != red_reward:
                # This might be a warning rather than error, depending on environment
                warnings.append(f"Episode {i}: Draw declared but rewards differ ({green_reward} vs {red_reward})")

            # Check player1/player2 reward consistency
            player1_reward = float(result['player1_reward'])
            player2_reward = float(result['player2_reward'])

            # Determine which player is green/red
            if result['player1'] == green_player:
                if player1_reward != green_reward:
                    errors.append(f"Episode {i}: Player1 reward mismatch with green reward")
                if player2_reward != red_reward:
                    errors.append(f"Episode {i}: Player2 reward mismatch with red reward")
            else:
                if player1_reward != red_reward:
                    errors.append(f"Episode {i}: Player1 reward mismatch with red reward")
                if player2_reward != green_reward:
                    errors.append(f"Episode {i}: Player2 reward mismatch with green reward")

        if errors:
            self.validation_errors.extend(errors)
            print(f"❌ Found {len(errors)} reward consistency errors")
            for error in errors[:10]:
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
        else:
            print("✅ Reward consistency validation passed")

        if warnings:
            self.validation_warnings.extend(warnings)
            print(f"⚠️  Found {len(warnings)} reward consistency warnings")
            for warning in warnings[:5]:
                print(f"  - {warning}")

        return errors

    def analyze_player_performance(self):
        """Analyze and validate player performance statistics."""
        print("\n" + "="*50)
        print("ANALYZING PLAYER PERFORMANCE")
        print("="*50)

        # Calculate player statistics
        player_stats = defaultdict(lambda: {
            'wins': 0, 'losses': 0, 'draws': 0, 'episodes': 0,
            'total_reward': 0.0, 'rewards': []
        })

        for result in self.results:
            p1, p2 = result['player1'], result['player2']
            winner = result['winner']
            p1_reward = float(result['player1_reward'])
            p2_reward = float(result['player2_reward'])

            # Update episode counts
            player_stats[p1]['episodes'] += 1
            player_stats[p2]['episodes'] += 1

            # Update rewards
            player_stats[p1]['total_reward'] += p1_reward
            player_stats[p2]['total_reward'] += p2_reward
            player_stats[p1]['rewards'].append(p1_reward)
            player_stats[p2]['rewards'].append(p2_reward)

            # Update win/loss/draw counts
            if winner == p1:
                player_stats[p1]['wins'] += 1
                player_stats[p2]['losses'] += 1
            elif winner == p2:
                player_stats[p2]['wins'] += 1
                player_stats[p1]['losses'] += 1
            else:
                player_stats[p1]['draws'] += 1
                player_stats[p2]['draws'] += 1

        # Display statistics
        print(f"{'Player':<20} {'Win Rate':<10} {'Avg Reward':<12} {'Episodes':<10}")
        print("-" * 60)

        # Sort by win rate
        sorted_players = sorted(
            player_stats.items(),
            key=lambda x: x[1]['wins'] / max(x[1]['episodes'], 1),
            reverse=True
        )

        for player_name, stats in sorted_players:
            win_rate = stats['wins'] / max(stats['episodes'], 1) * 100
            avg_reward = stats['total_reward'] / max(stats['episodes'], 1)

            print(f"{player_name:<20} {win_rate:>7.1f}% {avg_reward:>10.3f} {stats['episodes']:>8}")

        # Check for suspicious patterns
        warnings = []
        for player_name, stats in player_stats.items():
            # Check for perfect win/loss rates (might indicate bugs)
            if stats['episodes'] > 10:  # Only for players with sufficient data
                win_rate = stats['wins'] / stats['episodes']
                if win_rate == 1.0:
                    warnings.append(f"{player_name}: Perfect win rate (100%) - check for bugs")
                elif win_rate == 0.0:
                    warnings.append(f"{player_name}: Zero win rate (0%) - check for bugs")

                # Check for constant rewards (might indicate deterministic issues)
                if len(set(stats['rewards'])) == 1 and len(stats['rewards']) > 5:
                    warnings.append(f"{player_name}: Constant reward values - check for deterministic issues")

        if warnings:
            self.validation_warnings.extend(warnings)
            print(f"\n⚠️  Performance analysis warnings:")
            for warning in warnings:
                print(f"  - {warning}")

        return player_stats

    def generate_validation_report(self):
        """Generate comprehensive validation report."""
        print("\n" + "="*60)
        print("TOURNAMENT RESULTS VALIDATION REPORT")
        print("="*60)

        # Run all validations
        data_errors = self.validate_data_integrity()
        match_errors = self.validate_match_completeness()
        side_errors = self.validate_side_flip_consistency()
        reward_errors = self.validate_reward_consistency()
        player_stats = self.analyze_player_performance()

        # Summary
        total_errors = len(self.validation_errors)
        total_warnings = len(self.validation_warnings)

        print(f"\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        print(f"Results file: {self.results_file}")
        print(f"Total episodes: {len(self.results)}")
        print(f"Total players: {len(player_stats)}")
        print(f"Validation errors: {total_errors}")
        print(f"Validation warnings: {total_warnings}")

        if total_errors == 0:
            print("\n🎉 VALIDATION PASSED!")
            print("Tournament results are consistent and reliable.")
        else:
            print(f"\n❌ VALIDATION FAILED!")
            print(f"Found {total_errors} errors that need to be addressed.")

        if total_warnings > 0:
            print(f"\n⚠️  {total_warnings} warnings found - review recommended.")

        return total_errors == 0


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate tournament results for consistency and accuracy"
    )
    parser.add_argument(
        "results_file",
        help="Path to tournament results CSV file"
    )
    parser.add_argument(
        "--detailed", action="store_true",
        help="Show detailed error messages"
    )

    args = parser.parse_args()

    try:
        validator = TournamentResultValidator(args.results_file)
        validation_passed = validator.generate_validation_report()

        if args.detailed and validator.validation_errors:
            print(f"\n" + "="*60)
            print("DETAILED ERROR REPORT")
            print("="*60)
            for i, error in enumerate(validator.validation_errors, 1):
                print(f"{i:3d}. {error}")

        sys.exit(0 if validation_passed else 1)

    except Exception as e:
        print(f"Validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
