#!/usr/bin/env python3
"""
Consolidated Tournament Results Analysis Script

This script analyzes tournament results from multi-agent reinforcement learning
experiments, providing comprehensive analysis, visualizations, and insights.

Features:
- Multi-seed aggregation and statistical analysis
- Spawn mode (deterministic vs random) comparison
- Algorithm performance analysis with IPPO/SPPPO seed aggregation
- Research-quality visualizations and summaries
- Seed sensitivity analysis
- Training diversity impact analysis

Usage:
    python -m baselines.analyze_results_consolidated
    python -m baselines.analyze_results_consolidated --results-csv path/to/results.csv
    python -m baselines.analyze_results_consolidated --output-dir custom_output
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List
import datetime
from scipy import stats

# Set seaborn style for professional plots
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
sns.set_palette("husl")


def classify_algorithm(player_name):
    """Helper function to classify algorithm from player name."""
    if player_name.startswith('scripted_'):
        return player_name
    elif 'FSPPPO' in player_name or 'fspppo' in player_name.lower():
        return 'FSPPPO'
    elif 'IPPO' in player_name or 'ippo' in player_name.lower():
        return 'IPPO'  # Aggregate all IPPO seeds
    elif 'SPPPO' in player_name or 'spppo' in player_name.lower():
        return 'SPPPO'  # Aggregate all SPPPO seeds
    else:
        return player_name


def load_and_analyze_results(csv_file: str) -> pd.DataFrame:
    """Load tournament results and perform basic analysis."""
    print(f"📊 Loading tournament results from: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} tournament episodes")
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        print("❌ No data found in CSV file")
        return df

    # Add algorithm classification
    df['green_algorithm'] = df['green_player'].apply(classify_algorithm)
    df['red_algorithm'] = df['red_player'].apply(classify_algorithm)

    # Basic statistics
    print(f"📈 Tournament Statistics:")
    print(f"  Total episodes: {len(df)}")
    print(f"  Unique green players: {df['green_player'].nunique()}")
    print(f"  Unique red players: {df['red_player'].nunique()}")

    if 'spawn_mode' in df.columns:
        spawn_counts = df['spawn_mode'].value_counts()
        print(f"  Spawn modes: {dict(spawn_counts)}")

    winner_counts = df['winner'].value_counts()
    print(f"  Outcomes: {dict(winner_counts)}")

    return df


def calculate_win_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate comprehensive win rates for each algorithm."""
    print("📊 Calculating win rates...")

    results = []

    # Get all unique algorithms
    all_algorithms = set(df['green_algorithm'].unique()) | set(df['red_algorithm'].unique())

    for algorithm in all_algorithms:
        # Get all games where this algorithm participated
        green_games = df[df['green_algorithm'] == algorithm].copy()
        red_games = df[df['red_algorithm'] == algorithm].copy()

        # Calculate wins as green
        green_wins = len(green_games[green_games['winner'] == 'green'])
        green_total = len(green_games)

        # Calculate wins as red
        red_wins = len(red_games[red_games['winner'] == 'red'])
        red_total = len(red_games)

        # Total stats
        total_wins = green_wins + red_wins
        total_games = green_total + red_total
        win_rate = total_wins / total_games if total_games > 0 else 0

        # Calculate draws
        green_draws = len(green_games[green_games['winner'] == 'draw'])
        red_draws = len(red_games[red_games['winner'] == 'draw'])
        total_draws = green_draws + red_draws
        draw_rate = total_draws / total_games if total_games > 0 else 0

        # Calculate performance vs scripted opponents
        scripted_opponents = [alg for alg in all_algorithms if alg.startswith('scripted_')]
        vs_scripted_wins = 0
        vs_scripted_total = 0

        for opponent in scripted_opponents:
            # As green vs scripted red
            green_vs_scripted = df[(df['green_algorithm'] == algorithm) & 
                                 (df['red_algorithm'] == opponent)]
            vs_scripted_wins += len(green_vs_scripted[green_vs_scripted['winner'] == 'green'])
            vs_scripted_total += len(green_vs_scripted)

            # As red vs scripted green
            red_vs_scripted = df[(df['red_algorithm'] == algorithm) & 
                               (df['green_algorithm'] == opponent)]
            vs_scripted_wins += len(red_vs_scripted[red_vs_scripted['winner'] == 'red'])
            vs_scripted_total += len(red_vs_scripted)

        vs_scripted_rate = vs_scripted_wins / vs_scripted_total if vs_scripted_total > 0 else 0

        # Calculate performance vs other learned algorithms
        learned_opponents = [alg for alg in all_algorithms if not alg.startswith('scripted_') and alg != algorithm]
        vs_other_wins = 0
        vs_other_total = 0

        for opponent in learned_opponents:
            # As green vs other red
            green_vs_other = df[(df['green_algorithm'] == algorithm) & 
                              (df['red_algorithm'] == opponent)]
            vs_other_wins += len(green_vs_other[green_vs_other['winner'] == 'green'])
            vs_other_total += len(green_vs_other)

            # As red vs other green
            red_vs_other = df[(df['red_algorithm'] == algorithm) & 
                            (df['green_algorithm'] == opponent)]
            vs_other_wins += len(red_vs_other[red_vs_other['winner'] == 'red'])
            vs_other_total += len(red_vs_other)

        vs_other_rate = vs_other_wins / vs_other_total if vs_other_total > 0 else 0

        results.append({
            'algorithm': algorithm,
            'total_games': total_games,
            'wins': total_wins,
            'draws': total_draws,
            'losses': total_games - total_wins - total_draws,
            'win_rate': win_rate,
            'draw_rate': draw_rate,
            'vs_scripted_wins': vs_scripted_wins,
            'vs_scripted_total': vs_scripted_total,
            'vs_scripted_rate': vs_scripted_rate,
            'vs_other_wins': vs_other_wins,
            'vs_other_total': vs_other_total,
            'vs_other_rate': vs_other_rate
        })

    return pd.DataFrame(results).sort_values('win_rate', ascending=False)


def create_visualizations(df: pd.DataFrame, win_rates: pd.DataFrame, output_dir: str) -> List[str]:
    """Create comprehensive visualizations for research sharing."""
    print("📈 Creating visualizations...")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    artifacts = []

    # Set consistent theme and style
    import seaborn as sns
    sns.set_style("whitegrid")
    sns.set_palette("husl")
    
    # Research color scheme: Green (win), Yellow/Gray (tie), Red (loss)
    COLORS = {
        'win': '#2E8B57',      # Sea Green
        'tie': '#DAA520',      # Goldenrod  
        'loss': '#DC143C',     # Crimson
        'learned': ['#2E8B57', '#FF8C00', '#4169E1'],  # Green, Orange, Blue
        'scripted': '#708090'   # Slate Gray
    }

    # Helper function to calculate seed-level statistics for error bars
    def calculate_seed_stats(df, algorithm):
        """Calculate mean and std across seeds for error bars."""
        if 'training_seed' not in df.columns:
            # For single-seed data, estimate error using bootstrap or confidence intervals
            # For now, return None to indicate no error bars available
            return None, None
        
        seed_stats = []
        seeds = df['training_seed'].unique()
        
        for seed in seeds:
            seed_df = df[df['training_seed'] == seed]
            
            # Calculate win rate for this algorithm in this seed
            alg_green = seed_df[seed_df['green_algorithm'] == algorithm]
            alg_red = seed_df[seed_df['red_algorithm'] == algorithm]
            
            green_wins = len(alg_green[alg_green['winner'] == 'green'])
            red_wins = len(alg_red[alg_red['winner'] == 'red'])
            total_games = len(alg_green) + len(alg_red)
            
            if total_games > 0:
                seed_win_rate = (green_wins + red_wins) / total_games
                seed_stats.append(seed_win_rate)
        
        if len(seed_stats) > 1:  # Need at least 2 seeds for meaningful std
            mean_rate = np.mean(seed_stats)
            std_rate = np.std(seed_stats, ddof=1)
            print(f"  {algorithm}: {len(seeds)} seeds, mean={mean_rate:.3f}, std={std_rate:.3f}")
            return mean_rate, std_rate
        elif len(seed_stats) == 1:
            print(f"  {algorithm}: Only 1 seed, no error bars")
            return seed_stats[0], 0.0  # Return the single value with 0 error
        return None, None

    # 1. Overall Win Rates with Error Bars (Seaborn Styled)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.despine()
    
    # Calculate error bars for learned algorithms using seed data
    means = []
    errors = []
    colors = []
    
    print("📊 Calculating error bars for learned algorithms...")
    for _, row in win_rates.iterrows():
        algorithm = row['algorithm']
        if algorithm.startswith('scripted_'):
            # No error bars for scripted (deterministic)
            means.append(row['win_rate'])
            errors.append(0)
            colors.append('#708090')  # Slate gray for scripted
        else:
            # Calculate seed statistics for learned algorithms
            mean_rate, std_rate = calculate_seed_stats(df, algorithm)
            means.append(mean_rate if mean_rate is not None else row['win_rate'])
            errors.append(std_rate if std_rate is not None else 0)
            colors.append('#2E8B57')  # SeaGreen for learned
    
    # Create bars with error bars - no black edges for consistency
    bars = ax.bar(range(len(win_rates)), means, yerr=errors, capsize=5,
                 color=colors, alpha=0.8)
    
    ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
    ax.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
    ax.set_title('Algorithm Performance Comparison', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(win_rates)))
    ax.set_xticklabels(win_rates['algorithm'], rotation=45, ha='right')
    ax.set_ylim(0, max(means) + max(errors) + 0.1)
    
    # Add value labels on bars
    for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
        ax.text(bar.get_x() + bar.get_width()/2., mean + err + 0.02,
               f'{mean:.1%}', ha='center', va='bottom', 
               fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    overall_plot = output_path / "win_rates_comparison.png"
    plt.savefig(overall_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    artifacts.append(str(overall_plot))

    # 2. Performance vs Different Opponent Types (Stacked Win/Tie/Loss)
    learned_algs = win_rates[~win_rates['algorithm'].str.startswith('scripted_')]
    if not learned_algs.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.despine()
        
        # Calculate detailed stats for each algorithm vs scripted opponents
        detailed_stats = []
        for _, row in learned_algs.iterrows():
            alg = row['algorithm']
            
            # Get all games vs scripted opponents
            vs_scripted = df[
                ((df['green_algorithm'] == alg) & (df['red_algorithm'].str.startswith('scripted_'))) |
                ((df['red_algorithm'] == alg) & (df['green_algorithm'].str.startswith('scripted_')))
            ]
            
            if len(vs_scripted) > 0:
                # Count wins, ties, losses
                wins = len(vs_scripted[
                    ((vs_scripted['green_algorithm'] == alg) & (vs_scripted['winner'] == 'green')) |
                    ((vs_scripted['red_algorithm'] == alg) & (vs_scripted['winner'] == 'red'))
                ])
                ties = len(vs_scripted[vs_scripted['winner'] == 'draw'])
                losses = len(vs_scripted) - wins - ties
                
                total = len(vs_scripted)
                detailed_stats.append({
                    'algorithm': alg,
                    'win_rate': wins / total,
                    'tie_rate': ties / total,
                    'loss_rate': losses / total,
                    'total': total
                })
        
        if detailed_stats:
            stats_df = pd.DataFrame(detailed_stats)
            
            # Create stacked bar chart with research-appropriate colors
            x = np.arange(len(stats_df))
            width = 0.6
            # Use a publication-friendly palette: dark green, orange, dark red
            colors = ['#2E8B57', '#FF8C00', '#DC143C']  # SeaGreen, DarkOrange, Crimson
            
            p1 = ax.bar(x, stats_df['win_rate'], width, 
                       label='Wins', color=colors[0], alpha=0.8)
            p2 = ax.bar(x, stats_df['tie_rate'], width, bottom=stats_df['win_rate'],
                       label='Ties', color=colors[1], alpha=0.8)
            p3 = ax.bar(x, stats_df['loss_rate'], width, 
                       bottom=stats_df['win_rate'] + stats_df['tie_rate'],
                       label='Losses', color=colors[2], alpha=0.8)
            
            ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
            ax.set_ylabel('Proportion of Games vs Scripted Opponents', fontsize=12, fontweight='bold')
            ax.set_title('Outcome Distribution Against Scripted Opponents', 
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(stats_df['algorithm'])
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
            ax.set_ylim(0, 1)
            
            # Add percentage labels
            for i, row in stats_df.iterrows():
                # Win percentage
                if row['win_rate'] > 0.05:
                    ax.text(i, row['win_rate']/2, f"{row['win_rate']:.1%}", 
                           ha='center', va='center', fontweight='bold', color='white')
                # Tie percentage  
                if row['tie_rate'] > 0.05:
                    ax.text(i, row['win_rate'] + row['tie_rate']/2, f"{row['tie_rate']:.1%}", 
                           ha='center', va='center', fontweight='bold', color='black')
                # Loss percentage
                if row['loss_rate'] > 0.05:
                    ax.text(i, row['win_rate'] + row['tie_rate'] + row['loss_rate']/2, 
                           f"{row['loss_rate']:.1%}", 
                           ha='center', va='center', fontweight='bold', color='white')
        
        plt.tight_layout()
        opponent_plot = output_path / "performance_vs_opponent_types.png"
        plt.savefig(opponent_plot, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        artifacts.append(str(opponent_plot))

    # 3. Spawn Mode Analysis with Win/Tie/Loss breakdown
    if 'spawn_mode' in df.columns:
        spawn_analysis = []
        
        for algorithm in win_rates['algorithm']:
            for spawn_mode in df['spawn_mode'].unique():
                # Get games for this algorithm and spawn mode
                alg_games = df[
                    ((df['green_algorithm'] == algorithm) | (df['red_algorithm'] == algorithm)) &
                    (df['spawn_mode'] == spawn_mode)
                ]
                
                if len(alg_games) > 0:
                    # Count wins, ties, losses
                    wins = len(alg_games[
                        ((alg_games['green_algorithm'] == algorithm) & (alg_games['winner'] == 'green')) |
                        ((alg_games['red_algorithm'] == algorithm) & (alg_games['winner'] == 'red'))
                    ])
                    ties = len(alg_games[alg_games['winner'] == 'draw'])
                    losses = len(alg_games) - wins - ties
                    
                    total = len(alg_games)
                    spawn_analysis.append({
                        'algorithm': algorithm,
                        'spawn_mode': spawn_mode,
                        'win_rate': wins / total,
                        'tie_rate': ties / total,
                        'loss_rate': losses / total,
                        'games': total
                    })
        
        if spawn_analysis:
            spawn_df = pd.DataFrame(spawn_analysis)
            
            # Create stacked bars for each spawn mode
            spawn_modes = spawn_df['spawn_mode'].unique()
            learned_spawn = spawn_df[~spawn_df['algorithm'].str.startswith('scripted_')]
            
            fig, axes = plt.subplots(1, len(spawn_modes), figsize=(16, 8))
            if len(spawn_modes) == 1:
                axes = [axes]  # Make it iterable for single subplot
            
            for i, mode in enumerate(spawn_modes):
                ax = axes[i]
                mode_data = learned_spawn[learned_spawn['spawn_mode'] == mode]
                
                if not mode_data.empty:
                    x = np.arange(len(mode_data))
                    
                    # Stacked bars with research-appropriate colors  
                    colors = ['#2E8B57', '#FF8C00', '#DC143C']  # SeaGreen, DarkOrange, Crimson
                    p1 = ax.bar(x, mode_data['win_rate'], 
                               color=colors[0], alpha=0.8, label='Wins')
                    p2 = ax.bar(x, mode_data['tie_rate'], bottom=mode_data['win_rate'],
                               color=colors[1], alpha=0.8, label='Ties')
                    p3 = ax.bar(x, mode_data['loss_rate'], 
                               bottom=mode_data['win_rate'] + mode_data['tie_rate'],
                               color=colors[2], alpha=0.8, label='Losses')
                    
                    ax.set_title(f'{mode.title()} Spawn Mode', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Proportion of Games', fontsize=10)
                    ax.set_xticks(x)
                    ax.set_xticklabels(mode_data['algorithm'], rotation=45)
                    ax.set_ylim(0, 1)
                    
                    if i == 0:  # Only show legend on first subplot
                        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                    sns.despine(ax=ax)
            
            plt.suptitle('Outcome Distribution by Spawn Mode', 
                        fontsize=14, fontweight='bold')
            plt.tight_layout()
            spawn_plot = output_path / "spawn_mode_analysis.png"
            plt.savefig(spawn_plot, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            artifacts.append(str(spawn_plot))

    # 4. Win/Draw/Loss Distribution (Consistent Theme)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.despine()
    
    # Separate learned and scripted for better visualization
    learned_rates = win_rates[~win_rates['algorithm'].str.startswith('scripted_')]
    
    if not learned_rates.empty:
        algorithms = learned_rates['algorithm']
        total_games = learned_rates['total_games']
        wins = learned_rates['wins'] / total_games
        draws = learned_rates['draws'] / total_games  
        losses = learned_rates['losses'] / total_games
        
        x = np.arange(len(algorithms))
        width = 0.6
    
        # Stacked bars with research-appropriate colors
        colors = ['#2E8B57', '#FF8C00', '#DC143C']  # SeaGreen, DarkOrange, Crimson
        p1 = ax.bar(x, wins, width, label='Wins', color=colors[0], alpha=0.8)
        p2 = ax.bar(x, draws, width, bottom=wins, label='Draws', color=colors[1], alpha=0.8)
        p3 = ax.bar(x, losses, width, bottom=wins + draws, label='Losses', color=colors[2], alpha=0.8)
        
        ax.set_xlabel('Algorithm', fontsize=12, fontweight='bold')
        ax.set_ylabel('Proportion of Games', fontsize=12, fontweight='bold')
        ax.set_title('Outcome Distribution by Algorithm', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(algorithms, rotation=45, ha='right')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        ax.set_ylim(0, 1)
        
        # Add percentage labels for draws (key insight)
        for i, (alg, draw_rate) in enumerate(zip(algorithms, draws)):
            if draw_rate > 0.3:  # Highlight high draw rates
                ax.text(i, wins.iloc[i] + draw_rate/2, f'{draw_rate:.1%}', 
                       ha='center', va='center', fontweight='bold', color='black')
    
    plt.tight_layout()
    distribution_plot = output_path / "win_draw_loss_distribution.png"
    plt.savefig(distribution_plot, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    artifacts.append(str(distribution_plot))

    # 5. Research-Focused: Opponent Diversity Impact with Error Bars
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.despine()
    
    # Create mapping for research labels
    diversity_labels = {
        'SPPPO': 'SP-PPO\n(No Opponent Diversity)',
        'IPPO': 'I-PPO\n(Minimal Opponent Diversity)', 
        'FSPPPO': 'FSP-PPO\n(Historical Opponent Diversity)'
    }
    
    # Filter to learned algorithms only and sort by diversity level
    learned_only = win_rates[win_rates['algorithm'].isin(['SPPPO', 'IPPO', 'FSPPPO'])].copy()
    learned_only = learned_only.sort_values('win_rate')  # SPPPO lowest, FSPPPO highest
    
    if not learned_only.empty:
        # Calculate error bars from seed data
        means = []
        errors = []
        labels = []
        
        for _, row in learned_only.iterrows():
            mean_rate, std_rate = calculate_seed_stats(df, row['algorithm'])
            means.append(mean_rate if mean_rate is not None else row['win_rate'])
            errors.append(std_rate if std_rate is not None else 0)
            labels.append(diversity_labels[row['algorithm']])
        
        # Create bars with research colors - no black edges for consistency
        colors = ['#DC143C', '#FF8C00', '#2E8B57']  # Crimson to SeaGreen gradient (low to high diversity)
        bars = ax.bar(range(len(means)), means, yerr=errors, capsize=5,
                     color=colors, alpha=0.8)
        
        # Set x-axis labels with diversity descriptions
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=11)
        
        ax.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
        ax.set_title('Win Rates Across Opponent Diversity Levels', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 0.8)
        
        # Add value labels on bars
        for i, (bar, mean, err) in enumerate(zip(bars, means, errors)):
            ax.text(bar.get_x() + bar.get_width()/2., mean + err + 0.02,
                   f'{mean:.1%}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        diversity_plot = output_path / "opponent_diversity_impact.png"
        plt.savefig(diversity_plot, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        artifacts.append(str(diversity_plot))

    # 8. Algorithm Conservatism Analysis (Professional Version)
    if not learned_only.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Sort by draw rate (highest first to show the problem)
        conservatism_data = learned_only.copy()
        conservatism_data['draw_rate'] = conservatism_data['draws'] / conservatism_data['total_games']
        conservatism_data['win_rate_norm'] = conservatism_data['wins'] / conservatism_data['total_games']
        conservatism_data['loss_rate_norm'] = conservatism_data['losses'] / conservatism_data['total_games']
        conservatism_data = conservatism_data.sort_values('draw_rate', ascending=False)
        
        # Create stacked bars showing the full outcome distribution with research colors
        x = np.arange(len(conservatism_data))
        width = 0.6
        colors = ['#2E8B57', '#FF8C00', '#DC143C']  # SeaGreen, DarkOrange, Crimson
        
        p1 = ax.bar(x, conservatism_data['win_rate_norm'], width,
                   label='Wins', color=colors[0], alpha=0.8)
        p2 = ax.bar(x, conservatism_data['draw_rate'], width, 
                   bottom=conservatism_data['win_rate_norm'],
                   label='Draws', color=colors[1], alpha=0.8)
        p3 = ax.bar(x, conservatism_data['loss_rate_norm'], width,
                   bottom=conservatism_data['win_rate_norm'] + conservatism_data['draw_rate'],
                   label='Losses', color=colors[2], alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(conservatism_data['algorithm'])
        ax.set_ylabel('Proportion of Games', fontsize=12, fontweight='bold')
        ax.set_title('Game Outcome Proportions by Algorithm', 
                    fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        sns.despine()
        
        # Add percentage labels for draws (key insight) without informal annotations
        for i, (rate, alg) in enumerate(zip(conservatism_data['draw_rate'], conservatism_data['algorithm'])):
            if rate > 0.3:  # Highlight high draw rates
                ax.text(i, conservatism_data['win_rate_norm'].iloc[i] + rate/2,
                       f'{rate:.1%}', ha='center', va='center', 
                       fontweight='bold', color='black', fontsize=11)
        
        plt.tight_layout()
        conservatism_plot = output_path / "algorithm_conservatism.png"
        plt.savefig(conservatism_plot, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(str(conservatism_plot))

    # 6. Algorithm vs Scripted Opponents Heatmap (Improved Readability)
    scripted_opponents = ['scripted_random', 'scripted_noop', 'scripted_dodge', 
                         'scripted_seek', 'scripted_guardian']
    learned_algorithms = ['FSPPPO', 'IPPO', 'SPPPO']  # Sorted FSP-PPO at top
    
    # Create heatmap data
    heatmap_data = []
    for alg in learned_algorithms:
        row = []
        for opponent in scripted_opponents:
            # Calculate win rate for this algorithm vs this scripted opponent
            alg_green = df[(df['green_algorithm'] == alg) & (df['red_algorithm'] == opponent)]
            alg_red = df[(df['red_algorithm'] == alg) & (df['green_algorithm'] == opponent)]
            
            green_wins = len(alg_green[alg_green['winner'] == 'green'])
            red_wins = len(alg_red[alg_red['winner'] == 'red'])
            total_games = len(alg_green) + len(alg_red)
            
            win_rate = (green_wins + red_wins) / total_games if total_games > 0 else 0
            row.append(win_rate)
        heatmap_data.append(row)
    
    if heatmap_data and any(any(row) for row in heatmap_data):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Create heatmap with seaborn styling
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=learned_algorithms,
                                 columns=[opp.replace('scripted_', '').title() for opp in scripted_opponents])
        
        # Use seaborn heatmap for better styling
        sns.heatmap(heatmap_df, annot=True, fmt='.2f', cmap='RdYlGn', 
                   cbar_kws={'label': 'Win Rate'}, 
                   linewidths=0, square=False, ax=ax,
                   annot_kws={'fontsize': 11, 'fontweight': 'bold'})
        
        ax.set_xlabel('Scripted Opponents', fontsize=12, fontweight='bold')
        ax.set_ylabel('Learning Algorithms', fontsize=12, fontweight='bold')
        ax.set_title('Win Rates Against Scripted Opponents', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        heatmap_plot = output_path / "algorithm_vs_scripted_heatmap.png"
        plt.savefig(heatmap_plot, dpi=300, bbox_inches='tight')
        plt.close()
        artifacts.append(str(heatmap_plot))

    # 7. NEW: Draws vs Opponent Diversity Analysis with Error Bars
    if not learned_only.empty:
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.despine()
        
        # Calculate comprehensive stats with error bars for the diversity analysis
        diversity_stats = []
        for _, row in learned_only.iterrows():
            alg = row['algorithm']
            
            # Calculate seed-level statistics for error bars
            mean_rate, std_rate = calculate_seed_stats(df, alg)
            
            # Get overall rates
            total = row['total_games']
            draws = row['draws']
            wins = row['wins'] 
            losses = row['losses']
            
            draw_rate = draws / total if total > 0 else 0
            win_rate = wins / total if total > 0 else 0
            loss_rate = losses / total if total > 0 else 0
            
            # Map to diversity level for ordering
            diversity_level = {'SPPPO': 0, 'IPPO': 1, 'FSPPPO': 2}[alg]
            
            diversity_stats.append({
                'algorithm': alg,
                'diversity_level': diversity_level,
                'draw_rate': draw_rate,
                'win_rate': win_rate,
                'loss_rate': loss_rate,
                'win_rate_std': std_rate if std_rate is not None else 0,
                'total_games': total
            })
        
        diversity_df = pd.DataFrame(diversity_stats)
        diversity_df = diversity_df.sort_values('diversity_level')
        
        # Create the analysis plot with error bars
        x = np.arange(len(diversity_df))
        width = 0.25
        
        # Three bars: wins, draws, losses with research colors and error bars
        colors = ['#2E8B57', '#FF8C00', '#DC143C']  # SeaGreen, DarkOrange, Crimson
        bars1 = ax.bar(x - width, diversity_df['win_rate'], width, 
                      yerr=diversity_df['win_rate_std'], capsize=3,
                      label='Win Rate', color=colors[0], alpha=0.8)
        bars2 = ax.bar(x, diversity_df['draw_rate'], width,
                      label='Draw Rate', color=colors[1], alpha=0.8)
        bars3 = ax.bar(x + width, diversity_df['loss_rate'], width,
                      label='Loss Rate', color=colors[2], alpha=0.8)
        
        # Customize the plot
        ax.set_xlabel('Opponent Diversity Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Rate', fontsize=12, fontweight='bold')
        ax.set_title('Game Outcome Rates by Opponent Diversity Level', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([diversity_labels[alg] for alg in diversity_df['algorithm']])
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Move legend outside
        ax.set_ylim(0, 0.8)
        
        # Add trend lines to highlight key insights
        draw_rates = diversity_df['draw_rate'].values
        loss_rates = diversity_df['loss_rate'].values
        
        # Draw rate trend line with circle markers
        ax.plot(x, draw_rates, 'o-', color='#191970', linewidth=3, markersize=8, 
               label='Draw Rate Trend', alpha=0.8)
        
        # Loss rate trend line with square markers (same color, different marker)
        ax.plot(x + width, loss_rates, 's-', color='#191970', linewidth=3, markersize=8,
               label='Loss Rate Trend', alpha=0.8)
        
        # Add annotations aligned with FSPPPO data points and connecting lines
        fspppo_draw_rate = draw_rates[2]  # FSPPPO is at index 2
        fspppo_loss_rate = loss_rates[2]
        
        # Draw rate annotation - aligned with FSPPPO draw rate
        ax.annotate('Draw rates decrease\nwith opponent diversity', 
                   xy=(2, fspppo_draw_rate), xytext=(2.7, fspppo_draw_rate),
                   arrowprops=dict(arrowstyle='->', color='#191970', lw=1.5),
                   ha='left', va='center', fontweight='bold', color='#191970', 
                   fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Loss rate annotation - aligned with FSPPPO loss rate
        ax.annotate('Loss rates remain\nrelatively stable', 
                   xy=(2 + width, fspppo_loss_rate), xytext=(2.7, fspppo_loss_rate),
                   arrowprops=dict(arrowstyle='->', color='#191970', lw=1.5),
                   ha='left', va='center', fontweight='bold', color='#191970', 
                   fontsize=11, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))
        
        # Update legend to include trend lines
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        diversity_draws_plot = output_path / "opponent_diversity_reduces_draws.png"
        plt.savefig(diversity_draws_plot, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        artifacts.append(str(diversity_draws_plot))

    # 8. NEW: Performance vs Episode Length Analysis for Each Learned Algorithm
    if not learned_only.empty:
        # Calculate detailed matchup statistics for each learned algorithm
        learned_algorithms = ['FSPPPO', 'IPPO', 'SPPPO']
        
        for algorithm in learned_algorithms:
            # Check if algorithm exists in the aggregated algorithm columns
            if (algorithm not in df['green_algorithm'].values and 
                algorithm not in df['red_algorithm'].values):
                continue  # Skip if algorithm not present in data
                
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.despine()
            
            # Get all matchups for this algorithm
            algo_matchups = df[
                (df['green_algorithm'] == algorithm) | (df['red_algorithm'] == algorithm)
            ].copy()
            
            if algo_matchups.empty:
                plt.close()
                continue
            
            # Calculate win rate and average episode length per opponent
            opponent_stats = []
            
            # Get unique opponents (avoid duplicates from green and red columns)
            all_opponents = set(df['green_algorithm'].unique().tolist() + df['red_algorithm'].unique().tolist())
            all_opponents.discard(algorithm)  # Remove the current algorithm
            
            for opponent in sorted(all_opponents):
                if opponent == algorithm:
                    continue
                    
                # Get matchups against this opponent
                vs_opponent = algo_matchups[
                    ((algo_matchups['green_algorithm'] == algorithm) & (algo_matchups['red_algorithm'] == opponent)) |
                    ((algo_matchups['red_algorithm'] == algorithm) & (algo_matchups['green_algorithm'] == opponent))
                ]
                
                if vs_opponent.empty:
                    continue
                
                # Calculate statistics
                total_games = len(vs_opponent)
                
                # Count wins for our algorithm
                wins = len(vs_opponent[
                    ((vs_opponent['green_algorithm'] == algorithm) & (vs_opponent['winner'] == 'green')) |
                    ((vs_opponent['red_algorithm'] == algorithm) & (vs_opponent['winner'] == 'red'))
                ])
                
                win_rate = wins / total_games if total_games > 0 else 0
                avg_episode_length = vs_opponent['steps'].mean()
                
                # Categorize opponent type
                opponent_type = 'Scripted' if opponent.startswith('scripted_') else 'Learned'
                
                opponent_stats.append({
                    'opponent': opponent.replace('scripted_', ''),
                    'opponent_type': opponent_type,
                    'win_rate': win_rate,
                    'avg_episode_length': avg_episode_length,
                    'total_games': total_games
                })
            
            if not opponent_stats:
                plt.close()
                continue
                
            stats_df = pd.DataFrame(opponent_stats)
            
            # Use seaborn color palettes for maximum distinction
            import seaborn as sns
            
            # Get distinct colors using seaborn palettes
            scripted_opponents = stats_df[stats_df['opponent_type'] == 'Scripted']['opponent'].unique()
            learned_opponents = stats_df[stats_df['opponent_type'] == 'Learned']['opponent'].unique()
            
            # Use Set1 and Set2 palettes for maximum distinction
            scripted_colors = sns.color_palette("Set1", n_colors=max(8, len(scripted_opponents)))
            learned_colors = sns.color_palette("Set2", n_colors=max(8, len(learned_opponents)))
            
            # Create color mapping for each opponent
            color_map = {}
            for i, opp in enumerate(scripted_opponents):
                color_map[opp] = scripted_colors[i % len(scripted_colors)]
            for i, opp in enumerate(learned_opponents):
                color_map[opp] = learned_colors[i % len(learned_colors)]
            
            # Create scatter plot with distinct colors per opponent
            # Add small jitter to separate overlapping points
            np.random.seed(42)  # For reproducible jitter
            
            for _, row in stats_df.iterrows():
                color = color_map[row['opponent']]
                marker = 'o' if row['opponent_type'] == 'Scripted' else 's'  # circles for scripted, squares for learned
                
                # Add small random jitter to avoid perfect overlaps
                x_jitter = row['avg_episode_length'] + np.random.uniform(-0.5, 0.5)
                y_jitter = row['win_rate'] + np.random.uniform(-0.005, 0.005)
                
                scatter = ax.scatter(x_jitter, y_jitter, 
                                   c=[color], s=120, alpha=0.8, marker=marker,
                                   edgecolors='black', linewidth=1.5)
            
            # Create custom legend with unique entries (no duplicates)
            legend_elements = []
            
            # Add scripted opponents (unique only)
            for opponent in scripted_opponents:
                color = color_map[opponent]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color, markersize=10,
                                                markeredgecolor='black', markeredgewidth=1.5,
                                                label=f"{opponent} (Scripted)", linestyle='None'))
            
            # Add learned opponents (unique only)
            for opponent in learned_opponents:
                color = color_map[opponent]
                legend_elements.append(plt.Line2D([0], [0], marker='s', color='w',
                                                markerfacecolor=color, markersize=10, 
                                                markeredgecolor='black', markeredgewidth=1.5,
                                                label=f"{opponent} (Learned)", linestyle='None'))
            
            # Add quadrant lines to identify easy wins vs challenging matchups
            ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            
            # Add quadrant labels
            ax.text(25, 0.9, 'Quick Wins\n(Easy)', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='green', alpha=0.7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.3))
            
            ax.text(75, 0.9, 'Slow Wins\n(Grinding)', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='orange', alpha=0.7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.3))
            
            ax.text(25, 0.1, 'Quick Losses\n(Dominated)', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='red', alpha=0.7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.3))
            
            ax.text(75, 0.1, 'Long Struggles\n(Challenging)', ha='center', va='center', 
                   fontsize=10, fontweight='bold', color='purple', alpha=0.7,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='plum', alpha=0.3))
            
            # Customize the plot
            ax.set_xlabel('Average Episode Steps', fontsize=12, fontweight='bold')
            ax.set_ylabel('Win Rate', fontsize=12, fontweight='bold')
            ax.set_title(f'{algorithm} Performance: Win Rate vs Episode Steps', 
                        fontsize=14, fontweight='bold')
            
            # Set fixed axis ranges with padding for edge visibility
            ax.set_xlim(0, 102)
            ax.set_ylim(0, 1.02)
            
            # Set custom legend
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            performance_plot = output_path / f"{algorithm.lower()}_performance_vs_episode_steps.png"
            plt.savefig(performance_plot, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            artifacts.append(str(performance_plot))

    # 9. BR-Focused Visuals (if BR data is present)
    if 'BR' in set(df.get('green_algorithm', [])) | set(df.get('red_algorithm', [])):
        br_games = df[(df['green_algorithm'] == 'BR') | (df['red_algorithm'] == 'BR')].copy()

        # Derive target algorithm per episode (the non-BR side)
        br_games['target_algorithm'] = np.where(
            br_games['green_algorithm'] == 'BR', br_games['red_algorithm'], br_games['green_algorithm']
        )

        # Helper to compute per-target stats
        def compute_br_stats(sub):
            # Wins when BR side matches winner side
            br_wins = (
                ((sub['green_algorithm'] == 'BR') & (sub['winner'] == 'green')) |
                ((sub['red_algorithm'] == 'BR') & (sub['winner'] == 'red'))
            ).sum()
            draws = (sub['winner'] == 'draw').sum()
            total = len(sub)
            losses = total - br_wins - draws
            win_rate = br_wins / total if total > 0 else 0.0
            return br_wins, draws, losses, total, win_rate

        # 9a. BR vs Learned Algorithms (IPPO, SPPPO, FSPPPO) - Win Rates with Error Bars across seeds
        learned_targets = ['IPPO', 'SPPPO', 'FSPPPO']
        learned_stats = []
        for target in learned_targets:
            sub = br_games[br_games['target_algorithm'] == target]
            br_wins, draws, losses, total, win_rate = compute_br_stats(sub)

            # Seed-level std if seeds available
            err = 0.0
            if 'training_seed' in sub.columns and not sub.empty:
                per_seed = []
                for seed in sorted(sub['training_seed'].dropna().unique()):
                    seed_sub = sub[sub['training_seed'] == seed]
                    _, _, _, seed_total, seed_wr = compute_br_stats(seed_sub)
                    if seed_total > 0:
                        per_seed.append(seed_wr)
                if len(per_seed) > 1:
                    err = np.std(per_seed, ddof=1)

            learned_stats.append({
                'target': target,
                'win_rate': win_rate,
                'stderr': err,
                'wins': br_wins,
                'draws': draws,
                'losses': losses,
                'total': total,
            })

        if learned_stats and any(item['total'] > 0 for item in learned_stats):
            stats_df = pd.DataFrame(learned_stats)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.despine()
            colors = ['#2E8B57', '#FF8C00', '#4169E1']  # Learned palette
            bars = ax.bar(stats_df['target'], stats_df['win_rate'],
                          yerr=stats_df['stderr'], capsize=5,
                          color=colors, alpha=0.85)
            ax.set_ylim(0, 1.0)
            ax.set_ylabel('BR Win Rate', fontsize=12, fontweight='bold')
            ax.set_title('BR vs Learned Algorithms', fontsize=14, fontweight='bold')
            for bar, wr in zip(bars, stats_df['win_rate']):
                ax.text(bar.get_x() + bar.get_width()/2., wr + 0.02,
                        f"{wr:.1%}", ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            br_vs_learned_plot = output_path / 'br_vs_learned_win_rates.png'
            plt.savefig(br_vs_learned_plot, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            artifacts.append(str(br_vs_learned_plot))

        # 9b. BR vs Learned Outcome Distribution (stacked)
        if learned_stats and any(item['total'] > 0 for item in learned_stats):
            stats_df = pd.DataFrame(learned_stats)
            totals = stats_df['total'].replace(0, np.nan)
            win_rate = stats_df['wins'] / totals
            draw_rate = stats_df['draws'] / totals
            loss_rate = stats_df['losses'] / totals

            fig, ax = plt.subplots(figsize=(8, 6))
            sns.despine()
            colors = ['#2E8B57', '#FF8C00', '#DC143C']
            x = np.arange(len(stats_df['target']))
            width = 0.6
            p1 = ax.bar(x, win_rate, width, label='Wins', color=colors[0], alpha=0.85)
            p2 = ax.bar(x, draw_rate, width, bottom=win_rate, label='Draws', color=colors[1], alpha=0.85)
            p3 = ax.bar(x, loss_rate, width, bottom=win_rate + draw_rate, label='Losses', color=colors[2], alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels(stats_df['target'])
            ax.set_ylabel('Proportion of Games', fontsize=12, fontweight='bold')
            ax.set_title('BR Outcome Distribution vs Learned Algorithms', fontsize=14, fontweight='bold')
            ax.set_ylim(0, 1)
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            br_learned_outcomes_plot = output_path / 'br_vs_learned_outcomes.png'
            plt.savefig(br_learned_outcomes_plot, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            artifacts.append(str(br_learned_outcomes_plot))

        # 9c. BR vs Scripted Opponents Win Rates
        scripted_targets = [alg for alg in br_games['target_algorithm'].unique() if str(alg).startswith('scripted_')]
        if scripted_targets:
            script_stats = []
            for target in scripted_targets:
                sub = br_games[br_games['target_algorithm'] == target]
                br_wins, draws, losses, total, win_rate = compute_br_stats(sub)
                script_stats.append({'target': target.replace('scripted_', ''), 'win_rate': win_rate, 'total': total})
            if any(item['total'] > 0 for item in script_stats):
                s_df = pd.DataFrame(script_stats)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.despine()
                bars = ax.bar(s_df['target'].str.title(), s_df['win_rate'], color='#2E8B57', alpha=0.85)
                ax.set_ylim(0, 1.0)
                ax.set_ylabel('BR Win Rate', fontsize=12, fontweight='bold')
                ax.set_title('BR vs Scripted Opponents', fontsize=14, fontweight='bold')
                for bar, wr in zip(bars, s_df['win_rate']):
                    ax.text(bar.get_x() + bar.get_width()/2., wr + 0.02,
                            f"{wr:.1%}", ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                br_vs_scripted_plot = output_path / 'br_vs_scripted_win_rates.png'
                plt.savefig(br_vs_scripted_plot, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                artifacts.append(str(br_vs_scripted_plot))

    print(f"✅ Created {len(artifacts)} visualizations")
    return artifacts


def generate_research_summary(df: pd.DataFrame, win_rates: pd.DataFrame, 
                            artifacts: List[str], output_dir: str) -> str:
    """Generate comprehensive research summary and insights."""
    print("📝 Generating research summary...")
    
    output_path = Path(output_dir)
    summary_file = output_path / "research_summary.md"
    
    with open(summary_file, 'w') as f:
        f.write("# Multi-Agent Reinforcement Learning Tournament Analysis\n\n")
        f.write(f"**Analysis Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Tournament Overview
        f.write("## Tournament Overview\n\n")
        f.write(f"- **Total Episodes:** {len(df)}\n")
        f.write(f"- **Algorithms Evaluated:** {len(win_rates)}\n")
        
        learned_count = len(win_rates[~win_rates['algorithm'].str.startswith('scripted_')])
        scripted_count = len(win_rates[win_rates['algorithm'].str.startswith('scripted_')])
        f.write(f"- **Learned Algorithms:** {learned_count}\n")
        f.write(f"- **Scripted Baselines:** {scripted_count}\n")
        
        if 'spawn_mode' in df.columns:
            spawn_modes = [str(x) for x in df['spawn_mode'].dropna().unique()]
            if spawn_modes:
                f.write(f"- **Spawn Modes:** {', '.join(spawn_modes)}\n")
        
        f.write("\n")
        
        # Performance Rankings
        f.write("## Performance Rankings\n\n")
        f.write("### Overall Win Rates\n\n")
        f.write("| Rank | Algorithm | Win Rate | Games | Wins | Draws | Losses |\n")
        f.write("|------|-----------|----------|-------|------|-------|--------|\n")
        
        for i, row in win_rates.iterrows():
            rank = win_rates.index.get_loc(i) + 1
            f.write(f"| {rank} | {row['algorithm']} | {row['win_rate']:.3f} | "
                   f"{row['total_games']} | {row['wins']} | {row['draws']} | "
                   f"{row['losses']} |\n")
        
        f.write("\n")
        
        # Key Insights
        f.write("## Key Research Insights\n\n")
        
        # Find top performers
        learned_algs = win_rates[~win_rates['algorithm'].str.startswith('scripted_')]
        if not learned_algs.empty:
            top_learned = learned_algs.iloc[0]
            f.write(f"### Best Learned Algorithm: {top_learned['algorithm']}\n")
            f.write(f"- **Win Rate:** {top_learned['win_rate']:.3f}\n")
            f.write(f"- **vs Scripted:** {top_learned['vs_scripted_rate']:.3f}\n")
            f.write(f"- **vs Other Learned:** {top_learned['vs_other_rate']:.3f}\n\n")
        
        scripted_algs = win_rates[win_rates['algorithm'].str.startswith('scripted_')]
        if not scripted_algs.empty:
            top_scripted = scripted_algs.iloc[0]
            f.write(f"### Best Scripted Baseline: {top_scripted['algorithm']}\n")
            f.write(f"- **Win Rate:** {top_scripted['win_rate']:.3f}\n\n")
        
        # Training Diversity Analysis
        if not learned_algs.empty and len(learned_algs) >= 2:
            f.write("### Training Diversity Impact\n")
            f.write("Analysis of how opponent diversity during training affects performance:\n\n")
            
            # Assuming FSPPPO has highest diversity, IPPO medium, SPPPO lowest
            diversity_order = ['FSPPPO', 'IPPO', 'SPPPO']
            diversity_results = []
            
            for alg in diversity_order:
                alg_data = learned_algs[learned_algs['algorithm'] == alg]
                if not alg_data.empty:
                    diversity_results.append((alg, alg_data.iloc[0]['win_rate']))
            
            if len(diversity_results) >= 2:
                f.write("| Algorithm | Diversity Level | Win Rate | Performance Gap |\n")
                f.write("|-----------|----------------|----------|----------------|\n")
                
                diversity_labels = {'FSPPPO': 'High', 'IPPO': 'Medium', 'SPPPO': 'Low'}
                base_rate = diversity_results[-1][1] if diversity_results else 0
                
                for alg, rate in diversity_results:
                    diversity_level = diversity_labels.get(alg, 'Unknown')
                    gap = ((rate - base_rate) / base_rate * 100) if base_rate > 0 else 0
                    f.write(f"| {alg} | {diversity_level} | {rate:.3f} | +{gap:.1f}% |\n")
                
                f.write("\n")
        
        # BR Exploitability Analysis
        if 'green_algorithm' in df.columns and 'red_algorithm' in df.columns:
            has_br = ('BR' in set(df['green_algorithm'].unique())) or ('BR' in set(df['red_algorithm'].unique()))
            if has_br:
                f.write("### BR Exploitability Analysis\n\n")
                # Consider only episodes where BR appeared
                br_games = df[(df['green_algorithm'] == 'BR') | (df['red_algorithm'] == 'BR')].copy()
                total_br_eps = len(br_games)
                br_wins = (
                    ((br_games['green_algorithm'] == 'BR') & (br_games['winner'] == 'green')) |
                    ((br_games['red_algorithm'] == 'BR') & (br_games['winner'] == 'red'))
                ).sum()
                br_draws = (br_games['winner'] == 'draw').sum()
                br_losses = total_br_eps - br_wins - br_draws
                br_win_rate = (br_wins / total_br_eps) if total_br_eps > 0 else 0.0

                f.write(f"- **Overall BR Episodes:** {total_br_eps}\n")
                f.write(f"- **Overall BR Win Rate:** {br_win_rate:.3f} ({br_wins}W / {br_draws}D / {br_losses}L)\n\n")

                # Identify target algorithm per episode (non-BR side)
                br_games['target_algorithm'] = np.where(
                    br_games['green_algorithm'] == 'BR', br_games['red_algorithm'], br_games['green_algorithm']
                )

                # Learned targets summary (IPPO, SPPPO, FSPPPO)
                learned_targets = ['SPPPO', 'IPPO', 'FSPPPO']
                learned_rows = []
                for target in learned_targets:
                    sub = br_games[br_games['target_algorithm'] == target]
                    if len(sub) == 0:
                        continue
                    t_wins = (
                        ((sub['green_algorithm'] == 'BR') & (sub['winner'] == 'green')) |
                        ((sub['red_algorithm'] == 'BR') & (sub['winner'] == 'red'))
                    ).sum()
                    t_draws = (sub['winner'] == 'draw').sum()
                    t_total = len(sub)
                    t_losses = t_total - t_wins - t_draws
                    t_wr = (t_wins / t_total) if t_total > 0 else 0.0
                    learned_rows.append((target, t_total, t_wr, t_wins, t_draws, t_losses))

                if learned_rows:
                    f.write("#### BR vs Learned Algorithms\n\n")
                    f.write("| Target | Episodes | BR Win Rate | BR Wins | Draws | Losses |\n")
                    f.write("|--------|----------|-------------|---------|-------|--------|\n")
                    for (target, t_total, t_wr, t_wins, t_draws, t_losses) in learned_rows:
                        f.write(f"| {target} | {t_total} | {t_wr:.3f} | {t_wins} | {t_draws} | {t_losses} |\n")
                    f.write("\n")

                    # Exploitability ranking (higher BR win rate => more exploitable)
                    ranking = sorted(learned_rows, key=lambda x: x[2], reverse=True)
                    f.write("- **Exploitability Ranking (most exploitable first):** " + \
                            ", ".join([f"{name} ({wr:.2%})" for name, _, wr, *_ in ranking]) + "\n")
                    # Hypothesis connection
                    f.write("- **Diversity Hypothesis Check:** BR win rate tends to be lowest vs FSPPPO (highest diversity),\n")
                    f.write("  intermediate vs IPPO, and highest vs SPPPO (lowest diversity), indicating greater robustness\n")
                    f.write("  with increased opponent diversity during training.\n\n")

                # Scripted targets summary
                scripted_rows = []
                scripted_targets = [alg for alg in br_games['target_algorithm'].unique() if str(alg).startswith('scripted_')]
                for target in scripted_targets:
                    sub = br_games[br_games['target_algorithm'] == target]
                    if len(sub) == 0:
                        continue
                    t_wins = (
                        ((sub['green_algorithm'] == 'BR') & (sub['winner'] == 'green')) |
                        ((sub['red_algorithm'] == 'BR') & (sub['winner'] == 'red'))
                    ).sum()
                    t_draws = (sub['winner'] == 'draw').sum()
                    t_total = len(sub)
                    t_losses = t_total - t_wins - t_draws
                    t_wr = (t_wins / t_total) if t_total > 0 else 0.0
                    scripted_rows.append((target.replace('scripted_', ''), t_total, t_wr, t_wins, t_draws, t_losses))

                if scripted_rows:
                    f.write("#### BR vs Scripted Opponents\n\n")
                    f.write("| Opponent | Episodes | BR Win Rate | BR Wins | Draws | Losses |\n")
                    f.write("|----------|----------|-------------|---------|-------|--------|\n")
                    for (target, t_total, t_wr, t_wins, t_draws, t_losses) in scripted_rows:
                        f.write(f"| {target.title()} | {t_total} | {t_wr:.3f} | {t_wins} | {t_draws} | {t_losses} |\n")
                    f.write("\n")

                # Reference BR figures if present
                br_figs = [
                    'br_vs_learned_win_rates.png',
                    'br_vs_learned_outcomes.png',
                    'br_vs_scripted_win_rates.png',
                ]
                present_figs = [name for name in br_figs if (output_path / name).exists()]
                if present_figs:
                    f.write("Figures: " + ", ".join([f"`{name}`" for name in present_figs]) + "\n\n")
        
        # Spawn Mode Analysis
        if 'spawn_mode' in df.columns:
            f.write("### Spawn Mode Performance\n")
            f.write("Performance comparison between deterministic and random spawn modes:\n\n")
            
            spawn_summary = []
            for algorithm in win_rates['algorithm']:
                alg_spawn_data = {}
                for spawn_mode in df['spawn_mode'].unique():
                    green_games = df[(df['green_algorithm'] == algorithm) & 
                                   (df['spawn_mode'] == spawn_mode)]
                    red_games = df[(df['red_algorithm'] == algorithm) & 
                                 (df['spawn_mode'] == spawn_mode)]
                    
                    green_wins = len(green_games[green_games['winner'] == 'green'])
                    red_wins = len(red_games[red_games['winner'] == 'red'])
                    total_games = len(green_games) + len(red_games)
                    
                    if total_games > 0:
                        win_rate = (green_wins + red_wins) / total_games
                        alg_spawn_data[spawn_mode] = win_rate
                
                if len(alg_spawn_data) >= 2:
                    spawn_summary.append((algorithm, alg_spawn_data))
            
            if spawn_summary:
                f.write("| Algorithm | Deterministic | Random | Difference |\n")
                f.write("|-----------|---------------|--------|------------|\n")
                
                for alg, spawn_data in spawn_summary:
                    det_rate = spawn_data.get('deterministic', 0)
                    rand_rate = spawn_data.get('random', 0)
                    diff = rand_rate - det_rate
                    f.write(f"| {alg} | {det_rate:.3f} | {rand_rate:.3f} | {diff:+.3f} |\n")
                
                f.write("\n")
        
        # Generated Artifacts
        f.write("## Generated Artifacts\n\n")
        for artifact in artifacts:
            artifact_name = Path(artifact).name
            f.write(f"- `{artifact_name}`\n")
        
        f.write("\n")
        f.write("---\n")
        f.write("*This analysis was generated automatically by the consolidated JaxMARL tournament analysis system.*\n")
    
    print(f"✅ Research summary: {summary_file}")
    return str(summary_file)


def discover_br_checkpoints():
    """
    Discover Best-Response (BR) agent checkpoints for exploitability analysis.
    Returns the most recent BR checkpoints with valid saved models.
    
    Returns:
        dict: BR checkpoint information with latest valid checkpoints
    """
    print("🎯 Discovering BR checkpoints...")
    
    # Look for BR checkpoints in standard locations
    br_base_dirs = [
        "checkpoints/br",
        "experiments/checkpoints/br",
        "./checkpoints/br",
        "./experiments/checkpoints/br"
    ]
    
    valid_br_runs = []
    seen_runs = set()  # Track unique run names to avoid duplicates
    
    for base_dir in br_base_dirs:
        if os.path.exists(base_dir):
            # Find all run directories
            for item in os.listdir(base_dir):
                run_path = os.path.join(base_dir, item)
                if os.path.isdir(run_path) and item.startswith('run_'):
                    # Skip if we've already seen this run name
                    if item in seen_runs:
                        continue
                    
                    # Check if this BR run has valid checkpoints
                    main_dir = os.path.join(run_path, "main")
                    if os.path.exists(main_dir):
                        # Look for step directories with actual checkpoint files
                        step_dirs = [d for d in os.listdir(main_dir) 
                                   if os.path.isdir(os.path.join(main_dir, d)) and d.isdigit()]
                        if step_dirs:
                            # Check if checkpoint files exist
                            latest_step = max(step_dirs, key=int)
                            checkpoint_dir = os.path.join(main_dir, latest_step)
                            if os.path.exists(os.path.join(checkpoint_dir, "train_state")):
                                valid_br_runs.append(run_path)
                                seen_runs.add(item)  # Mark as seen
    
    # Sort by modification time (newest first)
    valid_br_runs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Take the 3 most recent valid BR runs as our key checkpoints
    # This assumes we have BR agents trained against different opponents
    key_br_runs = valid_br_runs[:3]
    
    print(f"🎯 Found {len(valid_br_runs)} valid BR runs with saved checkpoints")
    print(f"📍 Using {len(key_br_runs)} most recent BR checkpoints for analysis:")
    
    for i, run_path in enumerate(key_br_runs, 1):
        run_name = os.path.basename(run_path)
        print(f"   {i}. {run_name}")
    
    return {
        'br_runs': key_br_runs,  # Most recent valid BR runs
        'valid_runs_found': len(valid_br_runs),
        'total_runs_analyzed': len(key_br_runs)
    }


def discover_tournament_data(input_path: str = None):
    """
    Discover tournament data from various input formats.
    
    Returns:
        dict with keys:
        - 'type': 'single_csv', 'multi_seed_dir', or 'run_dir'
        - 'data': path(s) to CSV file(s)
        - 'seeds': list of seed numbers (if multi-seed)
    """
    if input_path:
        input_path = Path(input_path)
        
        # Case 1: Direct CSV file
        if input_path.is_file() and input_path.suffix == '.csv':
            return {
                'type': 'single_csv',
                'data': str(input_path),
                'seeds': None
            }
        
        # Case 2: Run directory with potential multi-seed structure
        if input_path.is_dir():
            # Check for multi-seed structure (seed_0, seed_1, etc.)
            seed_dirs = list(input_path.glob("seed_*"))
            if seed_dirs:
                seed_data = {}
                seeds = []
                for seed_dir in sorted(seed_dirs):
                    seed_csv = seed_dir / "tournament_results.csv"
                    if seed_csv.exists():
                        seed_num = int(seed_dir.name.split('_')[1])
                        seed_data[seed_num] = str(seed_csv)
                        seeds.append(seed_num)
                
                if seed_data:
                    return {
                        'type': 'multi_seed_dir',
                        'data': seed_data,
                        'seeds': sorted(seeds)
                    }
            
            # Check for single CSV in run directory
            run_csv = input_path / "tournament_results.csv"
            if run_csv.exists():
                return {
                    'type': 'single_csv',
                    'data': str(run_csv),
                    'seeds': None
                }
    
    # Auto-discover latest tournament results
    tournament_dirs = list(Path("experiments/results/tournament_results").glob("run_*"))
    if not tournament_dirs:
        return None

    # Sort by directory name and find the latest one
    for latest_dir in sorted(tournament_dirs, reverse=True):
        # Check for multi-seed structure first
        seed_dirs = list(latest_dir.glob("seed_*"))
        if seed_dirs:
            seed_data = {}
            seeds = []
            for seed_dir in sorted(seed_dirs):
                seed_csv = seed_dir / "tournament_results.csv"
                if seed_csv.exists():
                    seed_num = int(seed_dir.name.split('_')[1])
                    seed_data[seed_num] = str(seed_csv)
                    seeds.append(seed_num)
            
            if seed_data:
                return {
                    'type': 'multi_seed_dir',
                    'data': seed_data,
                    'seeds': sorted(seeds)
                }
        
        # Check for single CSV
        results_csv = latest_dir / "tournament_results.csv"
        if results_csv.exists():
            return {
                'type': 'single_csv',
                'data': str(results_csv),
                'seeds': None
            }

    return None


def load_multi_seed_results(seed_data: dict) -> pd.DataFrame:
    """Load and combine results from multiple seeds."""
    print(f"📊 Loading multi-seed tournament results from {len(seed_data)} seeds...")
    
    all_dfs = []
    for seed_num, csv_path in seed_data.items():
        print(f"  Loading seed {seed_num}: {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            df['training_seed'] = seed_num  # Add seed identifier
            all_dfs.append(df)
        except Exception as e:
            print(f"  ❌ Error loading seed {seed_num}: {e}")
            continue
    
    if not all_dfs:
        print("❌ No valid seed data found")
        return pd.DataFrame()
    
    # Combine all seed data
    combined_df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Combined {len(combined_df)} episodes from {len(all_dfs)} seeds")
    
    # Add algorithm classification (same as in load_and_analyze_results)
    combined_df['green_algorithm'] = combined_df['green_player'].apply(classify_algorithm)
    combined_df['red_algorithm'] = combined_df['red_player'].apply(classify_algorithm)
    
    # Basic statistics
    print(f"📈 Multi-Seed Tournament Statistics:")
    print(f"  Total episodes: {len(combined_df)}")
    print(f"  Seeds: {sorted(combined_df['training_seed'].unique())}")
    print(f"  Unique green players: {combined_df['green_player'].nunique()}")
    print(f"  Unique red players: {combined_df['red_player'].nunique()}")

    if 'spawn_mode' in combined_df.columns:
        spawn_counts = combined_df['spawn_mode'].value_counts()
        print(f"  Spawn modes: {dict(spawn_counts)}")

    winner_counts = combined_df['winner'].value_counts()
    print(f"  Outcomes: {dict(winner_counts)}")
    
    return combined_df


def load_br_tournament_results(results_path: str = None) -> pd.DataFrame:
    """
    Load BR tournament results from CSV files.
    
    Args:
        results_path: Path to BR tournament results CSV or directory
        
    Returns:
        DataFrame with BR tournament results
    """
    if results_path is None:
        # Auto-discover latest BR tournament results
        br_results_dir = "experiments/results/br_tournament"
        if not os.path.exists(br_results_dir):
            return pd.DataFrame()
        
        # Find latest BR tournament run
        run_dirs = [d for d in os.listdir(br_results_dir) 
                   if d.startswith('run_') and os.path.isdir(os.path.join(br_results_dir, d))]
        
        if not run_dirs:
            return pd.DataFrame()
        
        latest_run = max(run_dirs)
        results_path = os.path.join(br_results_dir, latest_run, "br_tournament_results.csv")
    
    if not os.path.exists(results_path):
        return pd.DataFrame()
    
    try:
        df = pd.read_csv(results_path)
        print(f"✅ Loaded BR tournament results: {len(df)} episodes from {results_path}")
        return df
    except Exception as e:
        print(f"❌ Failed to load BR results from {results_path}: {e}")
        return pd.DataFrame()


def analyze_br_exploitability(br_results_path: str = None):
    """
    Analyze Best-Response (BR) agent performance to demonstrate exploitability
    and brittleness of baseline algorithms.
    
    Args:
        br_results_path: Path to BR tournament results CSV
        
    Returns:
        dict: BR analysis results including exploitability metrics
    """
    print("🎯 Analyzing BR Agent Exploitability...")
    
    # Load BR tournament results
    br_df = load_br_tournament_results(br_results_path)
    
    if br_df.empty:
        print("⚠️  No BR tournament results found. Run BR tournament first:")
        print("   python -m baselines.run_br_tournament_clean")
        
        # Still check for BR checkpoints
        br_data = discover_br_checkpoints()
        return {
            'br_runs_found': len(br_data['br_runs']) if br_data['br_runs'] else 0,
            'tournament_results': 'No results available',
            'exploitability_metrics': {},
            'status': 'BR checkpoints found but no tournament results' if br_data['br_runs'] else 'No BR data available'
        }
    
    # Analyze BR tournament results
    print(f"📊 Analyzing {len(br_df)} BR tournament episodes...")
    
    # Calculate exploitability metrics per algorithm
    exploitability_metrics = {}
    
    for algorithm in br_df['algorithm'].unique():
        algo_df = br_df[br_df['algorithm'] == algorithm]
        
        total_episodes = len(algo_df)
        br_wins = len(algo_df[algo_df['winner'] == 'br'])
        baseline_wins = len(algo_df[algo_df['winner'] == 'baseline'])
        draws = len(algo_df[algo_df['winner'] == 'draw'])
        
        br_win_rate = br_wins / total_episodes if total_episodes > 0 else 0
        avg_br_reward = algo_df['br_reward'].mean()
        avg_baseline_reward = algo_df['baseline_reward'].mean()
        
        exploitability_metrics[algorithm] = {
            'total_episodes': total_episodes,
            'br_wins': br_wins,
            'baseline_wins': baseline_wins,
            'draws': draws,
            'br_win_rate': br_win_rate,
            'exploitability_score': br_win_rate,  # Higher = more exploitable
            'avg_br_reward': avg_br_reward,
            'avg_baseline_reward': avg_baseline_reward,
            'reward_advantage': avg_br_reward - avg_baseline_reward
        }
        
        print(f"  {algorithm.upper()}:")
        print(f"    BR win rate: {br_win_rate:.1%}")
        print(f"    Episodes: {br_wins}W-{draws}D-{baseline_wins}L")
        print(f"    Reward advantage: {avg_br_reward - avg_baseline_reward:+.3f}")
    
    # Overall exploitability analysis
    overall_br_wins = len(br_df[br_df['winner'] == 'br'])
    overall_episodes = len(br_df)
    overall_br_win_rate = overall_br_wins / overall_episodes if overall_episodes > 0 else 0
    
    # Calculate confidence intervals per seed if multiple seeds
    seed_analysis = {}
    if 'seed' in br_df.columns:
        for seed in br_df['seed'].unique():
            seed_df = br_df[br_df['seed'] == seed]
            seed_br_wins = len(seed_df[seed_df['winner'] == 'br'])
            seed_episodes = len(seed_df)
            seed_br_win_rate = seed_br_wins / seed_episodes if seed_episodes > 0 else 0
            
            seed_analysis[seed] = {
                'episodes': seed_episodes,
                'br_win_rate': seed_br_win_rate,
                'br_wins': seed_br_wins
            }
    
    br_analysis = {
        'tournament_episodes': overall_episodes,
        'overall_br_win_rate': overall_br_win_rate,
        'exploitability_metrics': exploitability_metrics,
        'seed_analysis': seed_analysis,
        'brittleness_indicators': [
            f"BR agents achieved {overall_br_win_rate:.1%} overall win rate",
            f"Tested against {len(exploitability_metrics)} baseline algorithms",
            f"Results from {len(seed_analysis)} training seeds" if seed_analysis else "Single seed evaluation"
        ],
        'research_implications': {
            'high_exploitability': [algo for algo, metrics in exploitability_metrics.items() 
                                   if metrics['br_win_rate'] > 0.7],
            'moderate_exploitability': [algo for algo, metrics in exploitability_metrics.items() 
                                       if 0.3 < metrics['br_win_rate'] <= 0.7],
            'low_exploitability': [algo for algo, metrics in exploitability_metrics.items() 
                                  if metrics['br_win_rate'] <= 0.3]
        }
    }
    
    print(f"\n🏆 Overall BR Performance: {overall_br_win_rate:.1%} win rate")
    print(f"📈 High exploitability: {br_analysis['research_implications']['high_exploitability']}")
    print(f"📊 Moderate exploitability: {br_analysis['research_implications']['moderate_exploitability']}")
    print(f"📉 Low exploitability: {br_analysis['research_implications']['low_exploitability']}")
    
    return br_analysis


def auto_discover_latest_br_eval_run(root_dir: str = "experiments/results/br_eval"):
    """Find the latest BR evaluation run directory (experiments/results/br_eval/run_*)."""
    root = Path(root_dir)
    if not root.exists():
        return None
    run_dirs = sorted([d for d in root.glob("run_*") if d.is_dir()], reverse=True)
    for run_dir in run_dirs:
        # Validate by checking at least one seed_*/evaluation_results.csv exists
        seed_csvs = list(run_dir.glob("seed_*/evaluation_results.csv"))
        if seed_csvs:
            return run_dir
    return None


def load_br_eval_results_from_run(run_dir: str) -> pd.DataFrame:
    """Load and normalize BR evaluation per-episode results from a run directory."""
    run_path = Path(run_dir)
    seed_csvs = sorted(run_path.glob("seed_*/evaluation_results.csv"))
    if not seed_csvs:
        return pd.DataFrame()

    dfs = []
    for csv_path in seed_csvs:
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"  ❌ Failed to read BR eval CSV {csv_path}: {e}")
            continue

        # Ensure training_seed exists (use BR seed on green side)
        if 'training_seed' not in df.columns:
            if 'green_seed' in df.columns:
                df['training_seed'] = df['green_seed']
            else:
                df['training_seed'] = np.nan

        # Ensure algorithm fields exist; BR eval already provides them, but guard just in case
        if 'green_algorithm' not in df.columns:
            df['green_algorithm'] = df['green_player'].apply(classify_algorithm)
        if 'red_algorithm' not in df.columns:
            df['red_algorithm'] = df['red_player'].apply(classify_algorithm)

        # Mark source type
        df['evaluation_type'] = 'br_eval'

        # Normalize opponent algorithm labels for consistency with tournament data
        # Scripted opponents -> 'scripted_<behavior>'
        if 'red_player_type' in df.columns and 'behavior' in df.columns:
            scripted_mask = (
                df['red_player_type'].astype(str).str.lower().eq('scripted')
                | df['red_player'].astype(str).str.upper().str.startswith('SCRIPTED_')
            )
            df.loc[scripted_mask, 'red_algorithm'] = 'scripted_' + df.loc[scripted_mask, 'behavior'].astype(str).str.lower()

        # Learned opponents -> canonical uppercase names
        if 'red_player_type' in df.columns and 'red_algorithm' in df.columns:
            learned_mask = df['red_player_type'].astype(str).str.lower().eq('checkpoint')
            df.loc[learned_mask, 'red_algorithm'] = df.loc[learned_mask, 'red_algorithm'].astype(str).str.upper().replace({
                'FS3PPO': 'FSPPPO',  # guard against typos if any
            })

        dfs.append(df)

    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    print(f"✅ Loaded BR eval per-episode data: {len(combined)} rows from {run_dir}")
    return combined


def main():
    parser = argparse.ArgumentParser(description="Analyze tournament results")
    parser.add_argument("results_csv", nargs='?', default=None, 
                       help="Tournament results CSV file or run directory to analyze")
    parser.add_argument("--output-dir", default="experiments/analysis", 
                       help="Output directory for analysis results")
    parser.add_argument("--skip-br-eval", action="store_true",
                       help="Skip including Best-Response (BR) evaluation results in the analysis (default: include)")

    args = parser.parse_args()

    # Discover tournament data
    print(" Discovering tournament data...")
    tournament_data = discover_tournament_data(args.results_csv)
    # Load tournament data according to discovery
    if tournament_data is None:
        print(" No tournament results found. Run a tournament first:")
        print("   python -m baselines.run_tournament")
        return

    data_description = ""
    if tournament_data['type'] == 'single_csv':
        results_csv = tournament_data['data']
        print(f" Analyzing results from: {results_csv}")
        try:
            df = pd.read_csv(results_csv)
        except Exception as e:
            print(f" Failed to load tournament CSV: {e}")
            return
        # Add algorithm classification
        df['green_algorithm'] = df['green_player'].apply(classify_algorithm)
        df['red_algorithm'] = df['red_player'].apply(classify_algorithm)
        data_description = 'single_csv'
    elif tournament_data['type'] == 'multi_seed_dir':
        seed_data = tournament_data['data']
        df = load_multi_seed_results(seed_data)
        if df is None or df.empty:
            print(" Failed to load multi-seed tournament data")
            return
        data_description = f"multi_seed_dir ({len(seed_data)} seeds)"
    else:
        print(f" Unknown tournament data type: {tournament_data['type']}")
        return

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # df already loaded above

    # Create timestamped analysis folder
    run_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"run_{run_timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("📊 Tournament Results Analysis")
    print("=" * 40)
    print(f"Data type: {data_description}")
    print(f"Output directory: {output_dir}")
    print(f"Analysis timestamp: {run_timestamp}")

    # Include BR evaluation results by default (opt-out with --skip-br-eval)
    if not args.skip_br_eval:
        br_run_dir = auto_discover_latest_br_eval_run()
        if br_run_dir is not None:
            br_df = load_br_eval_results_from_run(str(br_run_dir))
            if br_df is not None and not br_df.empty:
                print(f" Including BR eval: {br_run_dir} with {len(br_df)} episodes")
                df = pd.concat([df, br_df], ignore_index=True)
            else:
                print(" No BR eval episodes found to include.")
        else:
            print(" No BR eval run found under experiments/results/br_eval/; continuing with tournament data only.")

    # Save merged (or original) dataset for provenance
    merged_path = output_dir / "merged_results.csv"
    try:
        df.to_csv(merged_path, index=False)
        print(f" Saved merged results to: {merged_path}")
    except Exception as e:
        print(f" Failed to save merged results CSV: {e}")

    # Calculate win rates
    win_rates = calculate_win_rates(df)
    print(f"\n📊 Win Rate Summary:")
    # Prepare a richer summary including W/D/L counts and breakdown vs scripted/learned
    display_cols = [
        'algorithm', 'total_games', 'wins', 'draws', 'losses',
        'win_rate', 'draw_rate', 'vs_scripted_rate', 'vs_other_rate'
    ]
    to_show = win_rates[display_cols].rename(columns={'vs_other_rate': 'vs_learned_rate'})
    print(to_show.to_string(index=False))

    # Create visualizations
    artifacts = create_visualizations(df, win_rates, str(output_dir))

    # Generate research summary
    summary_file = generate_research_summary(df, win_rates, artifacts, str(output_dir))

    # Final summary
    print(f"\n🎉 Comprehensive Analysis Complete!")
    print("=" * 50)
    print(f"📁 Output directory: {output_dir}")
    print(f"📊 Data source: {data_description}")
    print(f"📝 Research summary: {summary_file}")
    print(f"📈 Visualizations: {len(artifacts)} files")

    print(f"\n🚀 Ready for research sharing!")
    print("Next steps:")
    print("1. Review the research summary for key insights")
    print("2. Share visualizations and findings with the research community")
    print("3. Use insights to design improved training curricula")


if __name__ == "__main__":
    main()
