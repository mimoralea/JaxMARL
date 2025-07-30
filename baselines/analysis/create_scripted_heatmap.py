#!/usr/bin/env python3
"""
Create Scripted vs Scripted Heatmap for Validation

This script creates a detailed heatmap showing win rates between scripted opponents
to validate that the tournament results are correct (e.g., seek should dominate noop).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_and_filter_scripted_data(csv_path):
    """Load tournament data and filter for scripted vs scripted matchups."""
    print(f"📊 Loading tournament data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Filter for scripted vs scripted matchups
    scripted_df = df[
        (df['player1'].str.contains('scripted_')) & 
        (df['player2'].str.contains('scripted_'))
    ].copy()
    
    print(f"✅ Found {len(scripted_df)} scripted vs scripted episodes")
    print(f"   Unique scripted players: {sorted(scripted_df['player1'].unique())}")
    
    return scripted_df

def calculate_scripted_win_rates(df):
    """Calculate win rates for scripted vs scripted matchups."""
    print("\n🔍 Calculating scripted vs scripted win rates...")
    
    # Get all unique scripted players
    scripted_players = sorted(set(df['player1'].unique()) | set(df['player2'].unique()))
    print(f"   Scripted players: {scripted_players}")
    
    # Create win rate matrix
    win_rate_matrix = pd.DataFrame(index=scripted_players, columns=scripted_players, dtype=float)
    episode_count_matrix = pd.DataFrame(index=scripted_players, columns=scripted_players, dtype=int)
    
    for player1 in scripted_players:
        for player2 in scripted_players:
            if player1 == player2:
                # Self-play should be 50% win rate
                win_rate_matrix.loc[player1, player2] = 50.0
                episode_count_matrix.loc[player1, player2] = 0
                continue
            
            # Get all episodes between these two players
            matchup_episodes = df[
                ((df['player1'] == player1) & (df['player2'] == player2)) |
                ((df['player1'] == player2) & (df['player2'] == player1))
            ]
            
            if len(matchup_episodes) == 0:
                win_rate_matrix.loc[player1, player2] = np.nan
                episode_count_matrix.loc[player1, player2] = 0
                continue
            
            # Calculate win rate for player1
            player1_wins = 0
            total_games = len(matchup_episodes)
            
            for _, episode in matchup_episodes.iterrows():
                if episode['winner'] == player1:
                    player1_wins += 1
            
            win_rate = (player1_wins / total_games) * 100
            win_rate_matrix.loc[player1, player2] = win_rate
            episode_count_matrix.loc[player1, player2] = total_games
    
    return win_rate_matrix, episode_count_matrix

def create_scripted_heatmap(win_rate_matrix, episode_count_matrix, output_dir):
    """Create a heatmap visualization for scripted vs scripted matchups."""
    print(f"\n📈 Creating scripted vs scripted heatmap...")
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    
    # Create annotations that show both win rate and episode count
    annotations = []
    for i in range(len(win_rate_matrix.index)):
        row = []
        for j in range(len(win_rate_matrix.columns)):
            win_rate = win_rate_matrix.iloc[i, j]
            episode_count = episode_count_matrix.iloc[i, j]
            
            if pd.isna(win_rate):
                row.append("N/A")
            elif i == j:  # Self-play
                row.append("Self")
            else:
                row.append(f"{win_rate:.1f}%\n({episode_count} games)")
        annotations.append(row)
    
    # Create the heatmap with red-to-green color scheme
    mask = win_rate_matrix.isna()
    ax = sns.heatmap(
        win_rate_matrix, 
        annot=annotations,
        fmt='',
        cmap='RdYlGn',  # Red-Yellow-Green colormap (red=loss, green=win)
        center=50,
        vmin=0,
        vmax=100,
        mask=mask,
        square=True,
        linewidths=0.5,
        cbar_kws={'label': 'Win Rate (%) - Red=Loss, Green=Win'}
    )
    
    # Customize the plot
    plt.title('Scripted vs Scripted Win Rate Matrix\n(Row player win rate against column player)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Opponent (Column Player)', fontsize=12)
    plt.ylabel('Player (Row Player)', fontsize=12)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Add a note about expected results
    plt.figtext(0.02, 0.02, 
                "Expected: seek should dominate noop/random, dodge should avoid damage, centaur should be balanced",
                fontsize=10, style='italic')
    
    plt.tight_layout()
    
    # Save the heatmap
    output_path = output_dir / 'scripted_vs_scripted_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Scripted vs scripted heatmap saved: {output_path}")
    plt.close()
    
    return output_path

def print_validation_analysis(win_rate_matrix):
    """Print validation analysis of scripted opponent behaviors."""
    print(f"\n🔍 VALIDATION ANALYSIS")
    print("=" * 30)
    
    # Expected behaviors to validate
    expected_dominance = [
        ("scripted_seek", "scripted_noop", "seek should dominate noop"),
        ("scripted_seek", "scripted_random", "seek should beat random"),
        ("scripted_dodge", "scripted_seek", "dodge should avoid seek"),
        ("scripted_random", "scripted_noop", "random should beat noop"),
    ]
    
    print("\n📋 Expected Behavior Validation:")
    for dominant, weak, description in expected_dominance:
        if dominant in win_rate_matrix.index and weak in win_rate_matrix.columns:
            win_rate = win_rate_matrix.loc[dominant, weak]
            if pd.notna(win_rate):
                status = "✅" if win_rate > 60 else "⚠️" if win_rate > 40 else "❌"
                print(f"   {status} {description}: {win_rate:.1f}%")
            else:
                print(f"   ❓ {description}: No data")
        else:
            print(f"   ❓ {description}: Players not found")
    
    # Print top and bottom performers
    print(f"\n🏆 Best Overall Performers:")
    avg_win_rates = win_rate_matrix.mean(axis=1, skipna=True)
    top_performers = avg_win_rates.nlargest(3)
    for player, avg_rate in top_performers.items():
        print(f"   {player}: {avg_rate:.1f}% average win rate")
    
    print(f"\n📉 Weakest Performers:")
    bottom_performers = avg_win_rates.nsmallest(3)
    for player, avg_rate in bottom_performers.items():
        print(f"   {player}: {avg_rate:.1f}% average win rate")

def main():
    # Configuration
    csv_path = 'tournament_results/tournament_results_20250728_141435.csv'
    output_dir = Path('tournament_analysis')
    output_dir.mkdir(exist_ok=True)
    
    # Load and process data
    scripted_df = load_and_filter_scripted_data(csv_path)
    
    # Calculate win rates
    win_rate_matrix, episode_count_matrix = calculate_scripted_win_rates(scripted_df)
    
    # Create heatmap
    heatmap_path = create_scripted_heatmap(win_rate_matrix, episode_count_matrix, output_dir)
    
    # Print validation analysis
    print_validation_analysis(win_rate_matrix)
    
    print(f"\n🎉 Scripted vs Scripted Analysis Complete!")
    print(f"📊 Heatmap saved: {heatmap_path}")

if __name__ == "__main__":
    main()
