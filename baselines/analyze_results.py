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
            spawn_modes = df['spawn_mode'].unique()
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


def main():
    parser = argparse.ArgumentParser(description="Analyze tournament results")
    parser.add_argument("results_csv", nargs='?', default=None, 
                       help="Tournament results CSV file or run directory to analyze")
    parser.add_argument("--output-dir", default="experiments/analysis", 
                       help="Output directory for analysis results")

    args = parser.parse_args()

    # Discover tournament data
    print("🔍 Discovering tournament data...")
    tournament_data = discover_tournament_data(args.results_csv)
    
    if not tournament_data:
        print("❌ No tournament results found in experiments/results/tournament_results/")
        print("Please run the tournament first using: python -m baselines.run_tournament")
        print("Or specify a results file/directory manually with: --results-csv path/to/results")
        return

    print(f"📊 Found tournament data: {tournament_data['type']}")
    
    if tournament_data['type'] == 'multi_seed_dir':
        print(f"  Seeds: {tournament_data['seeds']}")
        print(f"  Files: {len(tournament_data['data'])} seed CSVs")
    else:
        print(f"  File: {tournament_data['data']}")

    # Load data based on type
    if tournament_data['type'] == 'multi_seed_dir':
        df = load_multi_seed_results(tournament_data['data'])
        data_description = f"multi-seed ({len(tournament_data['seeds'])} seeds)"
    else:
        df = load_and_analyze_results(tournament_data['data'])
        data_description = "single-seed"

    if df.empty:
        print("❌ No valid results to analyze.")
        return

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

    # Calculate win rates
    win_rates = calculate_win_rates(df)
    print(f"\n📊 Win Rate Summary:")
    print(win_rates[['algorithm', 'win_rate', 'vs_scripted_rate', 'vs_other_rate']].to_string(index=False))

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
