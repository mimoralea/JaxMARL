#!/usr/bin/env python3
"""Tournament Results Analysis Script.

This script analyzes tournament CSV results and generates comprehensive visualizations
similar to the demo_evaluation outputs. It processes the tournament data to create:

1. Win rate comparisons across algorithms
2. Performance heatmaps for algorithm vs algorithm matchups
3. Scripted baseline performance analysis
4. Statistical summaries and insights

Usage:
    python -m baselines.analyze_tournament_results --csv tournament_results/tournament_results_20250728_125004.csv
    python -m baselines.analyze_tournament_results --directory tournament_results/
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import datetime

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def load_tournament_results(csv_path: str) -> pd.DataFrame:
    """Load and preprocess tournament results CSV."""

    print(f"📊 Loading tournament results from: {csv_path}")

    df = pd.read_csv(csv_path)

    # Add derived columns for easier analysis
    df['green_algorithm'] = df['green_player'].apply(lambda x: 'SCRIPTED' if x.startswith('scripted_') else x.split('_')[0].upper())
    df['red_algorithm'] = df['red_player'].apply(lambda x: 'SCRIPTED' if x.startswith('scripted_') else x.split('_')[0].upper())
    df['green_spec'] = df['green_player']
    df['red_spec'] = df['red_player']

    # Convert outcome to numeric for easier analysis
    df['green_wins'] = (df['winner'] == df['green_player']).astype(int)
    df['red_wins'] = (df['winner'] == df['red_player']).astype(int)
    df['draws'] = (df['winner'] == 'draw').astype(int)

    # Infer spawn mode from episode ordering if not present
    if 'spawn_mode' not in df.columns:
        print("⚠️  spawn_mode column not found, inferring from episode ordering...")
        df['spawn_mode'] = 'deterministic'  # Default

        # For each match, infer spawn mode based on episode ordering
        # Episodes are ordered: deterministic side 1, deterministic side 2, random side 1, random side 2
        for match_id in df['match_id'].unique():
            match_episodes = df[df['match_id'] == match_id].copy()
            total_episodes = len(match_episodes)
            episodes_per_side = total_episodes // 2
            det_per_side = episodes_per_side // 2

            # Mark random episodes (second half of each side)
            match_episodes = match_episodes.sort_values('episode_id')
            episode_indices = match_episodes.index

            # Side 1 random episodes (episodes det_per_side to episodes_per_side-1)
            side1_random_start = det_per_side
            side1_random_end = episodes_per_side

            # Side 2 random episodes (episodes episodes_per_side+det_per_side to total-1)
            side2_random_start = episodes_per_side + det_per_side
            side2_random_end = total_episodes

            for i, idx in enumerate(episode_indices):
                if (side1_random_start <= i < side1_random_end) or (side2_random_start <= i < side2_random_end):
                    df.loc[idx, 'spawn_mode'] = 'random'

    print(f"✅ Loaded {len(df)} episodes from {len(df.groupby(['green_player', 'red_player']))} unique matchups")
    if 'spawn_mode' in df.columns:
        spawn_counts = df['spawn_mode'].value_counts()
        print(f"📊 Spawn modes: {dict(spawn_counts)}")

    return df


def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive win rates for each player/algorithm."""

    print("🧮 Calculating win rates...")

    # Get all unique players
    all_players = list(set(df['green_player'].unique()) | set(df['red_player'].unique()))

    win_rate_data = []

    for player in all_players:
        # Games where player was green
        green_games = df[df['green_player'] == player]
        green_wins = len(green_games[green_games['winner'] == 'green'])

        # Games where player was red
        red_games = df[df['red_player'] == player]
        red_wins = len(red_games[red_games['winner'] == 'red'])

        # Total statistics
        total_games = len(green_games) + len(red_games)
        total_wins = green_wins + red_wins
        total_draws = len(green_games[green_games['winner'] == 'draw']) + len(red_games[red_games['winner'] == 'draw'])
        total_losses = total_games - total_wins - total_draws

        win_rate = total_wins / total_games if total_games > 0 else 0

        # Algorithm classification
        if player.startswith('scripted_'):
            algorithm = 'SCRIPTED'
            player_type = 'scripted'
        else:
            algorithm = player.split('_')[0].upper()
            player_type = 'learned'

        # Calculate performance vs different opponent types
        vs_scripted_games = 0
        vs_scripted_wins = 0
        vs_learned_games = 0
        vs_learned_wins = 0

        # As green player
        for _, row in green_games.iterrows():
            if row['red_player'].startswith('scripted_'):
                vs_scripted_games += 1
                if row['winner'] == 'green':
                    vs_scripted_wins += 1
            else:
                vs_learned_games += 1
                if row['winner'] == 'green':
                    vs_learned_wins += 1

        # As red player
        for _, row in red_games.iterrows():
            if row['green_player'].startswith('scripted_'):
                vs_scripted_games += 1
                if row['winner'] == 'red':
                    vs_scripted_wins += 1
            else:
                vs_learned_games += 1
                if row['winner'] == 'red':
                    vs_learned_wins += 1

        vs_scripted_rate = vs_scripted_wins / vs_scripted_games if vs_scripted_games > 0 else 0
        vs_learned_rate = vs_learned_wins / vs_learned_games if vs_learned_games > 0 else 0

        win_rate_data.append({
            'player': player,
            'algorithm': algorithm,
            'player_type': player_type,
            'total_games': total_games,
            'wins': total_wins,
            'losses': total_losses,
            'draws': total_draws,
            'win_rate': win_rate,
            'vs_scripted_games': vs_scripted_games,
            'vs_scripted_wins': vs_scripted_wins,
            'vs_scripted_rate': vs_scripted_rate,
            'vs_learned_games': vs_learned_games,
            'vs_learned_wins': vs_learned_wins,
            'vs_learned_rate': vs_learned_rate,
        })

    win_rates_df = pd.DataFrame(win_rate_data)
    win_rates_df = win_rates_df.sort_values('win_rate', ascending=False)

    print(f"✅ Calculated win rates for {len(win_rates_df)} players")

    return win_rates_df


def create_visualizations(df: pd.DataFrame, win_rates: pd.DataFrame, output_dir: str) -> List[str]:
    """Create comprehensive visualizations for tournament analysis."""

    print("🎨 Creating visualizations...")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    artifacts = []

    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")

    # Check if spawn mode data is available
    has_spawn_mode = 'spawn_mode' in df.columns and len(df['spawn_mode'].unique()) > 1

    # Main win rate bar chart
    plt.figure(figsize=(14, 10))

    # Main win rate bar chart
    plt.subplot(2, 3, 1)
    colors = ['lightcoral' if player_type == 'scripted' else 'skyblue'
              for player_type in win_rates['player_type']]
    bars = plt.bar(range(len(win_rates)), win_rates['win_rate'], color=colors, alpha=0.8)
    plt.title('Overall Win Rates by Player', fontsize=12, fontweight='bold')
    plt.ylabel('Win Rate')
    plt.ylim(0, 1)

    # Add value labels on bars
    for i, (bar, rate) in enumerate(zip(bars, win_rates['win_rate'])):
        plt.text(bar.get_x() + bar.get_width()/2., rate + 0.01,
                f'{rate:.2f}', ha='center', va='bottom', fontsize=8)

    plt.xticks(range(len(win_rates)), win_rates['player'], rotation=45, ha='right')

    # Add legend
    scripted_patch = plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.8, label='Scripted')
    learned_patch = plt.Rectangle((0,0),1,1, facecolor='skyblue', alpha=0.8, label='Learned')
    plt.legend(handles=[learned_patch, scripted_patch], loc='upper right')

    # 2. Algorithm-level Win Rates
    plt.subplot(2, 3, 2)
    alg_win_rates = win_rates.groupby('algorithm').agg({
        'win_rate': 'mean',
        'total_games': 'sum'
    }).reset_index()
    alg_win_rates = alg_win_rates.sort_values('win_rate', ascending=False)

    bars = plt.bar(alg_win_rates['algorithm'], alg_win_rates['win_rate'],
                   color='lightgreen', alpha=0.8)
    plt.title('Average Win Rates by Algorithm', fontsize=12, fontweight='bold')
    plt.ylabel('Average Win Rate')
    plt.ylim(0, 1)

    for bar, rate in zip(bars, alg_win_rates['win_rate']):
        plt.text(bar.get_x() + bar.get_width()/2., rate + 0.01,
                f'{rate:.2f}', ha='center', va='bottom')

    plt.xticks(rotation=45)

    # 3. Win Rate vs Opponent Type Breakdown
    plt.subplot(2, 3, 3)
    learned_players = win_rates[win_rates['player_type'] == 'learned']

    if len(learned_players) > 0:
        x = range(len(learned_players))
        width = 0.35

        plt.bar([i - width/2 for i in x], learned_players['vs_scripted_rate'],
               width, label='vs Scripted', color='lightcoral', alpha=0.8)
        plt.bar([i + width/2 for i in x], learned_players['vs_learned_rate'],
               width, label='vs Learned', color='lightgreen', alpha=0.8)

        plt.title('Learned Agents: Performance by Opponent Type', fontsize=12, fontweight='bold')
        plt.ylabel('Win Rate')
        plt.xlabel('Learned Agent')
        plt.xticks(x, learned_players['player'], rotation=45, ha='right')
        plt.legend()
        plt.ylim(0, 1)

    # 4. Games Played Distribution
    plt.subplot(2, 3, 4)
    plt.bar(range(len(win_rates)), win_rates['total_games'], color='gold', alpha=0.8)
    plt.title('Total Games per Player', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games')
    plt.xticks(range(len(win_rates)), win_rates['player'], rotation=45, ha='right')

    # 5. Win/Loss/Draw Distribution
    plt.subplot(2, 3, 5)
    bottom_losses = win_rates['losses']
    bottom_draws = win_rates['losses'] + win_rates['draws']

    plt.bar(range(len(win_rates)), win_rates['losses'], color='lightcoral', alpha=0.8, label='Losses')
    plt.bar(range(len(win_rates)), win_rates['draws'], bottom=win_rates['losses'],
           color='lightyellow', alpha=0.8, label='Draws')
    plt.bar(range(len(win_rates)), win_rates['wins'], bottom=bottom_draws,
           color='lightgreen', alpha=0.8, label='Wins')

    plt.title('Win/Loss/Draw Distribution', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Games')
    plt.xticks(range(len(win_rates)), win_rates['player'], rotation=45, ha='right')
    plt.legend()

    # 6. Performance Consistency
    plt.subplot(2, 3, 6)
    scatter = plt.scatter(win_rates['total_games'], win_rates['win_rate'],
                         s=100, c=win_rates['vs_scripted_rate'], cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Win Rate vs Scripted')
    plt.xlabel('Total Games')
    plt.ylabel('Overall Win Rate')
    plt.title('Performance vs Experience', fontsize=12, fontweight='bold')

    # Add player labels
    for _, row in win_rates.iterrows():
        plt.annotate(row['player'][:8] + '...' if len(row['player']) > 8 else row['player'],
                    (row['total_games'], row['win_rate']),
                    xytext=(5, 5), textcoords='offset points',
                    fontsize=7, alpha=0.8)

    plt.tight_layout()

    # Save main visualization
    main_viz_path = output_path / "tournament_analysis_overview.png"
    plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append(str(main_viz_path))
    print(f"✅ Main tournament overview: {main_viz_path}")

    # 7. Detailed Matchup Heatmap
    learned_players = win_rates[win_rates['player_type'] == 'learned']['player'].tolist()

    if len(learned_players) > 1:
        plt.figure(figsize=(10, 8))

        # Create win rate matrix
        matrix = np.zeros((len(learned_players), len(learned_players)))

        for i, player1 in enumerate(learned_players):
            for j, player2 in enumerate(learned_players):
                if player1 != player2:
                    # Get matches between these players
                    matches = df[((df['green_player'] == player1) & (df['red_player'] == player2)) |
                               ((df['green_player'] == player2) & (df['red_player'] == player1))]

                    if len(matches) > 0:
                        # Calculate win rate for player1 (row) vs player2 (column)
                        player1_wins = len(matches[matches['winner'] == player1])
                        win_rate = player1_wins / len(matches)
                        matrix[i, j] = win_rate
                    else:
                        matrix[i, j] = np.nan
                else:
                    matrix[i, j] = np.nan

        # Create heatmap with red-to-green color scheme
        mask = np.eye(len(learned_players), dtype=bool)
        ax = sns.heatmap(
            matrix,
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',  # Red-Yellow-Green colormap (red=loss, green=win)
            center=0.5,
            vmin=0,
            vmax=1,
            mask=mask,
            square=True,
            linewidths=0.5,
            cbar_kws={'label': 'Win Rate - Red=Loss, Green=Win'}
        )

        plt.title('Learned Agent vs Learned Agent Win Rates\n(Row beats Column)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Opponent (Red)')
        plt.ylabel('Player (Green)')

        heatmap_path = output_path / "learned_vs_learned_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(str(heatmap_path))
        print(f"✅ Learned vs learned heatmap: {heatmap_path}")

    # 8. Episode length distribution
    plt.figure(figsize=(10, 6))
    if has_spawn_mode:
        # Split by spawn mode
        for spawn_mode in df['spawn_mode'].unique():
            subset = df[df['spawn_mode'] == spawn_mode]
            plt.hist(subset['steps'], bins=30, alpha=0.6, label=f'{spawn_mode.title()} starts', edgecolor='black')
        plt.legend()
    else:
        plt.hist(df['steps'], bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Episode Length (Steps)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Episode Lengths' + (' by Spawn Mode' if has_spawn_mode else ''))
    plt.grid(True, alpha=0.3)

    length_file = output_path / "episode_length_distribution.png"
    plt.savefig(length_file, dpi=300, bbox_inches='tight')
    plt.close()
    artifacts.append(str(length_file))

    # 9. Spawn mode comparison (if available)
    if has_spawn_mode:
        plt.figure(figsize=(12, 8))

        # Calculate win rates by spawn mode for each algorithm
        spawn_comparison = []
        for algorithm in win_rates['algorithm'].unique():
            if algorithm == 'SCRIPTED':
                continue
            alg_players = win_rates[win_rates['algorithm'] == algorithm]['player'].tolist()

            for spawn_mode in df['spawn_mode'].unique():
                subset = df[(df['spawn_mode'] == spawn_mode) &
                           ((df['green_player'].isin(alg_players)) | (df['red_player'].isin(alg_players)))]

                wins = 0
                total = 0
                for player in alg_players:
                    green_wins = len(subset[(subset['green_player'] == player) & (subset['winner'] == player)])
                    red_wins = len(subset[(subset['red_player'] == player) & (subset['winner'] == player)])
                    green_total = len(subset[subset['green_player'] == player])
                    red_total = len(subset[subset['red_player'] == player])

                    wins += green_wins + red_wins
                    total += green_total + red_total

                win_rate = wins / total if total > 0 else 0
                spawn_comparison.append({
                    'algorithm': algorithm,
                    'spawn_mode': spawn_mode,
                    'win_rate': win_rate,
                    'games': total
                })

        if spawn_comparison:
            spawn_df = pd.DataFrame(spawn_comparison)
            pivot_data = spawn_df.pivot(index='algorithm', columns='spawn_mode', values='win_rate')

            sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0.5)
            plt.title('Algorithm Win Rates by Spawn Mode')
            plt.ylabel('Algorithm')
            plt.xlabel('Spawn Mode')

            spawn_file = output_path / "spawn_mode_comparison.png"
            plt.savefig(spawn_file, dpi=300, bbox_inches='tight')
            plt.close()
            artifacts.append(str(spawn_file))

    # 10. Scripted Baseline Performance Analysis
    scripted_players = win_rates[win_rates['player_type'] == 'scripted']['player'].tolist()

    if len(scripted_players) > 0 and len(learned_players) > 0:
        plt.figure(figsize=(12, 8))

        # Create matrix: learned agents (rows) vs scripted agents (columns)
        matrix = np.zeros((len(learned_players), len(scripted_players)))

        for i, learned in enumerate(learned_players):
            for j, scripted in enumerate(scripted_players):
                # Get matches between learned and scripted
                matches = df[((df['green_player'] == learned) & (df['red_player'] == scripted)) |
                           ((df['green_player'] == scripted) & (df['red_player'] == learned))]

                if len(matches) > 0:
                    # Calculate win rate for learned agent
                    learned_wins = len(matches[matches['winner'] == learned])
                    win_rate = learned_wins / len(matches)
                    matrix[i, j] = win_rate
                else:
                    matrix[i, j] = np.nan

        # Create heatmap
        sns.heatmap(matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0.5,
                   xticklabels=[p.replace('scripted_', '') for p in scripted_players],
                   yticklabels=[p[:10] for p in learned_players],
                   cbar_kws={'label': 'Win Rate'})

        plt.title('Learned Agents vs Scripted Baselines\n(Higher is Better for Learned Agents)',
                 fontsize=14, fontweight='bold')
        plt.xlabel('Scripted Baseline')
        plt.ylabel('Learned Agent')

        baseline_path = output_path / "learned_vs_scripted_heatmap.png"
        plt.savefig(baseline_path, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(str(baseline_path))
        print(f"✅ Learned vs scripted heatmap: {baseline_path}")

    return artifacts


def generate_analysis_summary(df: pd.DataFrame, win_rates: pd.DataFrame,
                            artifacts: List[str], output_dir: str) -> str:
    """Generate comprehensive analysis summary."""

    print("📝 Generating analysis summary...")

    output_dir = Path(output_dir)
    summary_file = output_dir / f"tournament_analysis_summary_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    with open(summary_file, 'w') as f:
        f.write("🏆 TOURNAMENT ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Episodes: {len(df)}\n")
        f.write(f"Unique Players: {len(win_rates)}\n")
        f.write(f"Unique Matchups: {len(df.groupby(['green_player', 'red_player']))}\n\n")

        # Player Statistics
        f.write("PLAYER PERFORMANCE RANKINGS\n")
        f.write("-" * 30 + "\n\n")

        for i, (_, row) in enumerate(win_rates.iterrows(), 1):
            f.write(f"{i}. {row['player']} ({row['algorithm']})\n")
            f.write(f"   Win Rate: {row['win_rate']:.1%} ({row['wins']}/{row['total_games']})\n")
            f.write(f"   W/L/D: {row['wins']}/{row['losses']}/{row['draws']}\n")
            if row['player_type'] == 'learned':
                f.write(f"   vs Scripted: {row['vs_scripted_rate']:.1%}\n")
                f.write(f"   vs Learned: {row['vs_learned_rate']:.1%}\n")
            f.write("\n")

        # Algorithm Summary
        f.write("ALGORITHM SUMMARY\n")
        f.write("-" * 20 + "\n\n")

        alg_summary = win_rates.groupby('algorithm').agg({
            'win_rate': ['mean', 'std', 'count'],
            'total_games': 'sum'
        }).round(3)

        for alg in alg_summary.index:
            f.write(f"{alg}:\n")
            f.write(f"  Players: {alg_summary.loc[alg, ('win_rate', 'count')]}\n")
            f.write(f"  Avg Win Rate: {alg_summary.loc[alg, ('win_rate', 'mean')]:.1%}\n")
            f.write(f"  Std Dev: {alg_summary.loc[alg, ('win_rate', 'std')]:.3f}\n")
            f.write(f"  Total Games: {alg_summary.loc[alg, ('total_games', 'sum')]}\n\n")

        # Key Insights
        f.write("KEY INSIGHTS\n")
        f.write("-" * 15 + "\n\n")

        # Best performer
        best_player = win_rates.iloc[0]
        f.write(f"• Best Overall Performer: {best_player['player']} ({best_player['win_rate']:.1%})\n")

        # Best algorithm
        best_alg = win_rates.groupby('algorithm')['win_rate'].mean().idxmax()
        best_alg_rate = win_rates.groupby('algorithm')['win_rate'].mean().max()
        f.write(f"• Best Algorithm: {best_alg} ({best_alg_rate:.1%} avg)\n")

        # Most games played
        max_games_idx = win_rates['total_games'].idxmax()
        f.write(f"• Most games played: {win_rates.loc[max_games_idx, 'player']} ({win_rates['total_games'].max()} games)\n")

        # Draw rate
        total_draws = df['draws'].sum()
        draw_rate = total_draws / len(df)
        f.write(f"• Overall draw rate: {draw_rate:.1%} ({total_draws}/{len(df)} episodes)\n")

        f.write("\n")
        f.write("GENERATED ARTIFACTS\n")
        f.write("-" * 20 + "\n\n")
        for artifact in artifacts:
            artifact_name = Path(artifact).name
            f.write(f"• {artifact_name}\n")

        f.write("\n" + "=" * 50 + "\n")
        f.write("Analysis generated by JaxMARL Tournament Analysis System\n")

    print(f"✅ Analysis summary: {summary_file}")
    return str(summary_file)


def main():
    parser = argparse.ArgumentParser(description="Analyze tournament results and create visualizations")
    parser.add_argument("--csv", help="Path to tournament results CSV file")
    parser.add_argument("--directory", help="Directory containing tournament results (will use latest CSV)")
    parser.add_argument("--output-dir", default="tournament_analysis", help="Output directory for analysis")

    args = parser.parse_args()

    # Determine CSV file to analyze
    csv_path = None
    if args.csv:
        csv_path = args.csv
    elif args.directory:
        # Find latest CSV in directory
        csv_files = glob.glob(f"{args.directory}/tournament_results_*.csv")
        if csv_files:
            # Filter out intermediate files and get latest
            final_files = [f for f in csv_files if 'intermediate' not in f]
            if final_files:
                csv_path = max(final_files, key=lambda x: Path(x).stat().st_mtime)
            else:
                csv_path = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
        else:
            print(f"❌ No tournament results CSV found in {args.directory}")
            return
    else:
        # Default to tournament_results directory
        csv_files = glob.glob("tournament_results/tournament_results_*.csv")
        if csv_files:
            final_files = [f for f in csv_files if 'intermediate' not in f]
            if final_files:
                csv_path = max(final_files, key=lambda x: Path(x).stat().st_mtime)
            else:
                csv_path = max(csv_files, key=lambda x: Path(x).stat().st_mtime)
        else:
            print("❌ No tournament results found. Please specify --csv or --directory")
            return

    if not Path(csv_path).exists():
        print(f"❌ CSV file not found: {csv_path}")
        return

    print("🎯 Tournament Results Analysis")
    print("=" * 40)
    print(f"📁 Input CSV: {csv_path}")
    print(f"📁 Output directory: {args.output_dir}")

    # Load and analyze results
    df = load_tournament_results(csv_path)
    if df.empty:
        print("❌ No valid results to analyze.")
        return

    # Calculate win rates
    win_rates = calculate_win_rates(df)

    # Create visualizations
    artifacts = create_visualizations(df, win_rates, args.output_dir)

    # Generate analysis summary
    summary_file = generate_analysis_summary(df, win_rates, artifacts, args.output_dir)

    # Final summary
    print(f"\n🎉 Tournament Analysis Complete!")
    print("=" * 40)
    print(f"📁 Output directory: {args.output_dir}")
    print(f"📊 Input CSV: {csv_path}")
    print(f"📝 Analysis summary: {summary_file}")
    print(f"📈 Visualizations: {len(artifacts)} files")

    print(f"\n📋 Top 3 Performers:")
    for i, (_, row) in enumerate(win_rates.head(3).iterrows(), 1):
        print(f"  {i}. {row['player']}: {row['win_rate']:.1%} win rate")

    print(f"\n🚀 Analysis ready for review!")


if __name__ == "__main__":
    main()
