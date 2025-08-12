#!/usr/bin/env python3
"""
Episode Length Analysis for Tournament Results

Analyzes game duration patterns across different matchup types:
- Learned vs Learned matchups
- Learned vs Scripted matchups
- Scripted vs Scripted matchups

Provides insights into strategic efficiency and decisiveness.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from datetime import datetime

def load_tournament_data(csv_path):
    """Load tournament results from CSV file."""
    print(f"📊 Loading tournament data from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"✅ Loaded {len(df)} episodes from {df['player1'].nunique() + df['player2'].nunique()} unique players")
    return df

def categorize_players(df):
    """Categorize players into learned algorithms and scripted opponents."""
    learned_algorithms = ['IPPO', 'SPPPO', 'FSPPPO']

    def get_player_type(player_name):
        if any(alg in player_name for alg in learned_algorithms):
            return 'learned'
        elif 'scripted_' in player_name:
            return 'scripted'
        else:
            return 'unknown'

    def get_algorithm(player_name):
        for alg in learned_algorithms:
            if alg in player_name:
                return alg
        if 'scripted_' in player_name:
            return player_name.replace('scripted_', 'scripted_')
        return 'unknown'

    # Add player type and algorithm columns
    df['player1_type'] = df['player1'].apply(get_player_type)
    df['player2_type'] = df['player2'].apply(get_player_type)
    df['player1_algorithm'] = df['player1'].apply(get_algorithm)
    df['player2_algorithm'] = df['player2'].apply(get_algorithm)

    # Categorize matchup types
    def get_matchup_type(row):
        if row['player1_type'] == 'learned' and row['player2_type'] == 'learned':
            return 'learned_vs_learned'
        elif row['player1_type'] == 'learned' and row['player2_type'] == 'scripted':
            return 'learned_vs_scripted'
        elif row['player1_type'] == 'scripted' and row['player2_type'] == 'learned':
            return 'learned_vs_scripted'
        elif row['player1_type'] == 'scripted' and row['player2_type'] == 'scripted':
            return 'scripted_vs_scripted'
        else:
            return 'unknown'

    df['matchup_type'] = df.apply(get_matchup_type, axis=1)

    return df

def analyze_episode_lengths(df):
    """Analyze episode length patterns across different matchup types."""
    print("\n🔍 EPISODE LENGTH ANALYSIS")
    print("=" * 50)

    # Overall statistics
    print(f"\n📈 Overall Episode Length Statistics:")
    print(f"   Mean: {df['episode_length'].mean():.1f} steps")
    print(f"   Median: {df['episode_length'].median():.1f} steps")
    print(f"   Std: {df['episode_length'].std():.1f} steps")
    print(f"   Min: {df['episode_length'].min()} steps")
    print(f"   Max: {df['episode_length'].max()} steps")

    # Analysis by matchup type
    print(f"\n📊 Episode Length by Matchup Type:")
    matchup_stats = df.groupby('matchup_type')['episode_length'].agg([
        'count', 'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
    print(matchup_stats)

    # Analysis by specific algorithm matchups
    print(f"\n🤖 Episode Length by Algorithm Matchups:")
    df['matchup_pair'] = df.apply(lambda row: f"{row['player1_algorithm']}_vs_{row['player2_algorithm']}", axis=1)
    algorithm_stats = df.groupby('matchup_pair')['episode_length'].agg([
        'count', 'mean', 'median', 'std'
    ]).round(1).sort_values('mean')
    print(algorithm_stats)

    # Timeout analysis (episodes reaching max length)
    max_length = df['episode_length'].max()
    timeout_analysis = df.groupby('matchup_type').apply(
        lambda x: (x['episode_length'] == max_length).sum() / len(x) * 100
    ).round(1)

    print(f"\n⏱️ Timeout Rate Analysis (% reaching {max_length} steps):")
    for matchup_type, timeout_rate in timeout_analysis.items():
        print(f"   {matchup_type}: {timeout_rate}%")

    # Quick finish analysis (episodes finishing in <50 steps)
    quick_threshold = 50
    quick_analysis = df.groupby('matchup_type').apply(
        lambda x: (x['episode_length'] < quick_threshold).sum() / len(x) * 100
    ).round(1)

    print(f"\n⚡ Quick Finish Rate Analysis (% finishing in <{quick_threshold} steps):")
    for matchup_type, quick_rate in quick_analysis.items():
        print(f"   {matchup_type}: {quick_rate}%")

    return matchup_stats, algorithm_stats, timeout_analysis, quick_analysis

def create_episode_length_visualizations(df, output_dir):
    """Create visualizations for episode length analysis."""
    print(f"\n📈 Creating Episode Length Visualizations")
    print("=" * 50)

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Episode Length Analysis Across Tournament Matchups', fontsize=16, fontweight='bold')

    # 1. Episode length distribution by matchup type
    ax1 = axes[0, 0]
    matchup_types = df['matchup_type'].unique()
    for matchup_type in matchup_types:
        data = df[df['matchup_type'] == matchup_type]['episode_length']
        ax1.hist(data, alpha=0.7, label=matchup_type, bins=20)
    ax1.set_xlabel('Episode Length (steps)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Episode Length Distribution by Matchup Type')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Box plot of episode lengths by matchup type
    ax2 = axes[0, 1]
    df.boxplot(column='episode_length', by='matchup_type', ax=ax2)
    ax2.set_xlabel('Matchup Type')
    ax2.set_ylabel('Episode Length (steps)')
    ax2.set_title('Episode Length Distribution by Matchup Type')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    # 3. Mean episode length by algorithm pairs
    ax3 = axes[0, 2]
    df['matchup_pair'] = df.apply(lambda row: f"{row['player1_algorithm']}_vs_{row['player2_algorithm']}", axis=1)
    mean_lengths = df.groupby('matchup_pair')['episode_length'].mean().sort_values()
    mean_lengths.plot(kind='barh', ax=ax3)
    ax3.set_xlabel('Mean Episode Length (steps)')
    ax3.set_title('Mean Episode Length by Algorithm Matchups')
    ax3.grid(True, alpha=0.3)

    # 4. Timeout rate by matchup type
    ax4 = axes[1, 0]
    max_length = df['episode_length'].max()
    timeout_rates = df.groupby('matchup_type').apply(
        lambda x: (x['episode_length'] == max_length).sum() / len(x) * 100
    )
    timeout_rates.plot(kind='bar', ax=ax4, color='orange')
    ax4.set_ylabel('Timeout Rate (%)')
    ax4.set_title(f'Timeout Rate by Matchup Type (% reaching {max_length} steps)')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(True, alpha=0.3)

    # 5. Episode length vs outcome
    ax5 = axes[1, 1]
    outcome_lengths = df.groupby('outcome')['episode_length'].mean()
    colors = ['red' if outcome == -1 else 'blue' if outcome == 1 else 'gray' for outcome in outcome_lengths.index]
    outcome_lengths.plot(kind='bar', ax=ax5, color=colors)
    ax5.set_ylabel('Mean Episode Length (steps)')
    ax5.set_title('Mean Episode Length by Outcome')
    ax5.set_xlabel('Outcome (1=Player1 wins, -1=Player2 wins, 0=Draw)')
    ax5.grid(True, alpha=0.3)

    # 6. Algorithm efficiency heatmap
    ax6 = axes[1, 2]
    # Create algorithm vs algorithm mean episode length matrix
    algorithms = ['IPPO_latest_', 'SPPPO_latest', 'FSPPPO_latest']
    scripted = [col for col in df['player1'].unique() if 'scripted_' in col]

    # Focus on learned vs scripted for efficiency analysis
    learned_vs_scripted = df[df['matchup_type'] == 'learned_vs_scripted'].copy()

    if not learned_vs_scripted.empty:
        # Create pivot table for heatmap
        pivot_data = []
        for learned in algorithms:
            row_data = []
            for script in scripted:
                # Get episodes where learned agent plays against scripted
                episodes = learned_vs_scripted[
                    ((learned_vs_scripted['player1'] == learned) & (learned_vs_scripted['player2'] == script)) |
                    ((learned_vs_scripted['player1'] == script) & (learned_vs_scripted['player2'] == learned))
                ]
                if len(episodes) > 0:
                    row_data.append(episodes['episode_length'].mean())
                else:
                    row_data.append(np.nan)
            pivot_data.append(row_data)

        pivot_df = pd.DataFrame(pivot_data, index=algorithms, columns=scripted)
        sns.heatmap(pivot_df, annot=True, fmt='.1f', ax=ax6, cmap='RdYlBu_r')
        ax6.set_title('Mean Episode Length: Learned vs Scripted')
        ax6.set_xlabel('Scripted Opponents')
        ax6.set_ylabel('Learned Algorithms')

    plt.tight_layout()

    # Save the visualization
    output_path = output_dir / 'episode_length_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Episode length visualization saved: {output_path}")
    plt.close()

def generate_episode_length_summary(df, matchup_stats, algorithm_stats, timeout_analysis, quick_analysis, output_dir):
    """Generate a comprehensive episode length analysis summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = output_dir / f'episode_length_summary_{timestamp}.txt'

    with open(summary_path, 'w') as f:
        f.write("🎮 EPISODE LENGTH ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Episodes: {len(df)}\n")
        f.write(f"Unique Players: {df['player1'].nunique() + df['player2'].nunique()}\n")
        f.write(f"Max Episode Length: {df['episode_length'].max()} steps\n\n")

        f.write("OVERALL EPISODE LENGTH STATISTICS\n")
        f.write("-" * 35 + "\n")
        f.write(f"Mean: {df['episode_length'].mean():.1f} steps\n")
        f.write(f"Median: {df['episode_length'].median():.1f} steps\n")
        f.write(f"Standard Deviation: {df['episode_length'].std():.1f} steps\n")
        f.write(f"Min: {df['episode_length'].min()} steps\n")
        f.write(f"Max: {df['episode_length'].max()} steps\n\n")

        f.write("EPISODE LENGTH BY MATCHUP TYPE\n")
        f.write("-" * 32 + "\n")
        f.write(matchup_stats.to_string() + "\n\n")

        f.write("EPISODE LENGTH BY ALGORITHM MATCHUPS\n")
        f.write("-" * 37 + "\n")
        f.write(algorithm_stats.to_string() + "\n\n")

        f.write("TIMEOUT ANALYSIS (% reaching max length)\n")
        f.write("-" * 40 + "\n")
        for matchup_type, rate in timeout_analysis.items():
            f.write(f"{matchup_type}: {rate}%\n")
        f.write("\n")

        f.write("QUICK FINISH ANALYSIS (% finishing <50 steps)\n")
        f.write("-" * 45 + "\n")
        for matchup_type, rate in quick_analysis.items():
            f.write(f"{matchup_type}: {rate}%\n")
        f.write("\n")

        f.write("KEY INSIGHTS\n")
        f.write("-" * 12 + "\n")

        # Generate insights based on the data
        learned_vs_learned_mean = matchup_stats.loc['learned_vs_learned', 'mean'] if 'learned_vs_learned' in matchup_stats.index else 0
        learned_vs_scripted_mean = matchup_stats.loc['learned_vs_scripted', 'mean'] if 'learned_vs_scripted' in matchup_stats.index else 0
        scripted_vs_scripted_mean = matchup_stats.loc['scripted_vs_scripted', 'mean'] if 'scripted_vs_scripted' in matchup_stats.index else 0

        f.write(f"1. Learned vs Learned matchups average {learned_vs_learned_mean:.1f} steps\n")
        f.write(f"2. Learned vs Scripted matchups average {learned_vs_scripted_mean:.1f} steps\n")
        f.write(f"3. Scripted vs Scripted matchups average {scripted_vs_scripted_mean:.1f} steps\n")

        if learned_vs_learned_mean > learned_vs_scripted_mean:
            f.write("4. Learned agents take longer to resolve matches against each other than against scripted opponents\n")
        else:
            f.write("4. Learned agents resolve matches faster against each other than against scripted opponents\n")

        # Timeout insights
        if 'learned_vs_learned' in timeout_analysis:
            ll_timeout = timeout_analysis['learned_vs_learned']
            ls_timeout = timeout_analysis['learned_vs_scripted']
            f.write(f"5. Learned vs Learned timeout rate: {ll_timeout}%\n")
            f.write(f"6. Learned vs Scripted timeout rate: {ls_timeout}%\n")

            if ll_timeout > ls_timeout:
                f.write("7. Learned agents are more likely to reach timeout against each other (indecisive matches)\n")
            else:
                f.write("7. Learned agents are more decisive against each other than against scripted opponents\n")

    print(f"✅ Episode length summary saved: {summary_path}")
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='Analyze episode lengths from tournament results')
    parser.add_argument('--csv-path', type=str,
                       default='tournament_results/tournament_results_20250728_141435.csv',
                       help='Path to tournament results CSV file')
    parser.add_argument('--output-dir', type=str, default='tournament_analysis',
                       help='Output directory for analysis results')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load and process data
    df = load_tournament_data(args.csv_path)
    df = categorize_players(df)

    # Analyze episode lengths
    matchup_stats, algorithm_stats, timeout_analysis, quick_analysis = analyze_episode_lengths(df)

    # Create visualizations
    create_episode_length_visualizations(df, output_dir)

    # Generate summary report
    summary_path = generate_episode_length_summary(
        df, matchup_stats, algorithm_stats, timeout_analysis, quick_analysis, output_dir
    )

    print(f"\n🎉 Episode Length Analysis Complete!")
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Visualization: {output_dir}/episode_length_analysis.png")
    print(f"📝 Summary report: {summary_path}")

if __name__ == "__main__":
    main()
